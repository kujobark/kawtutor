import OpenAI from "openai";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---- CORS CONFIG ----
const ALLOWED_ORIGIN = "*";
function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

/**
 * Kaw Companion — Framing Routine Socratic Tutor (API)
 * Goals:
 * - Support students with the Frame routine (Key Topic → Is About → Main Ideas → Essential Details → So What?)
 * - Ask EXACTLY 1 question per turn
 * - Never lecture / never provide answers
 *
 * KEY FIXES:
 * - Accept a `stepHint` from the frontend (what the bot asked last)
 * - Interpret normal student wording in context (no need to say "One Main Idea...")
 * - Add So What handling so we don't repeat the Main Idea problem at the end of the routine
 */
const SYSTEM_PROMPT_FRAMING = `
You are Kaw Companion, a Socratic tutor for grades 4–12 using the Framing Routine.

NON-NEGOTIABLE RULES (always):
- Use QUESTIONS ONLY. Do not explain. Do not lecture. Do not summarize.
- Never give direct answers, solutions, or “the correct response.”
- Never confirm correctness (no “Correct,” “That’s right,” etc.).
- Ask EXACTLY 1 question per turn (no more).
- Keep responses short and consistent.

FRAMING ROUTINE SEQUENCE (must follow):
1) Key Topic (2–5 words: the focus/title)
2) Is About (one short phrase: what the topic is about)
3) Main Ideas (1–3 important ideas)
4) Essential Details (evidence/examples for each main idea)
5) So What? (why it matters / significance)

ANCHOR RULE:
- If Key Topic is not clear, ask ONLY for Key Topic.
- If Key Topic is clear but Is About is not clear, ask ONLY for Is About.
- Do not ask meaning/definition questions before Key Topic + Is About are captured.

REDIRECT RULE:
If the student asks for an answer or you feel pulled into explaining, redirect to the next Frame step with ONE question.

TONE:
Calm, teacher-like, encouraging, neutral toward correctness.

OUTPUT FORMAT:
Return only the single question. No headings. No bullets.
`.trim();

// ---- Hard caps (tuneable) ----
const MAX_MODEL_TOKENS = 140;
const MAX_CHARS = 420;
const MAX_QUESTIONS = 1;

// --------------------------
// Output enforcement helpers
// --------------------------
function countQuestions(text) {
  return (text.match(/\?/g) || []).length;
}

function looksLikeExplanation(text) {
  const t = (text || "").toLowerCase().trim();
  const badStarts = [
    "in summary",
    "to summarize",
    "here's",
    "this means",
    "the answer is",
    "overall,",
    "for example,",
  ];
  const hasBadStart = badStarts.some((s) => t.startsWith(s));
  const tooManySentences = (text.match(/[.!]/g) || []).length >= 2;
  const hasColonList = text.includes(":") && (text.match(/\n/g) || []).length >= 1;
  return hasBadStart || tooManySentences || hasColonList;
}

function enforceHardCap(text) {
  let out = (text || "").trim();

  // Strip accidental labels
  out = out.replace(/^(kaw companion|tutor|assistant)\s*:\s*/i, "").trim();

  // Keep only first question if more than one
  if (countQuestions(out) > MAX_QUESTIONS) {
    const firstQ = out.indexOf("?");
    out = firstQ >= 0 ? out.slice(0, firstQ + 1).trim() : out.trim();
  }

  // Char cap
  if (out.length > MAX_CHARS) out = out.slice(0, MAX_CHARS).trim();

  // Must be a question
  if (!out.includes("?")) {
    out = "What is your Key Topic (just the title, 2–5 words)?";
  }

  // Ensure ends with a question mark
  const lastQ = out.lastIndexOf("?");
  if (lastQ !== out.length - 1) out = out.slice(0, lastQ + 1).trim();

  return out;
}

function socraticFailSafe(original, routedQuestion) {
  const cleaned = (original || "").trim();

  const triggers =
    cleaned.length > MAX_CHARS ||
    looksLikeExplanation(cleaned) ||
    countQuestions(cleaned) !== 1;

  if (!triggers) return enforceHardCap(cleaned);

  const fallback = (
    routedQuestion || "What is your Key Topic (just the title, 2–5 words)?"
  ).trim();
  return enforceHardCap(fallback);
}

// -----------------------------------------------
// Frame parsing + instructional sufficiency checks
// -----------------------------------------------
function cleanText(s) {
  return String(s || "").replace(/\s+/g, " ").trim();
}

// Safely read framing fields (supports a couple common shapes)
function getFramingKeyTopic(framing) {
  return (
    framing?.keyTopic ??
    framing?.key_topic ??
    framing?.frame?.keyTopic ??
    framing?.frame?.key_topic ??
    ""
  );
}

function getFramingIsAbout(framing) {
  return (
    framing?.isAbout ??
    framing?.is_about ??
    framing?.frame?.isAbout ??
    framing?.frame?.is_about ??
    ""
  );
}

// Parse a student attempt like: "X is about Y"
function parseKeyTopicIsAbout(message) {
  const t = cleanText(message);
  const m = t.match(/^(.*?)\s+is about\s+(.+?)(?:[.?!]|$)/i);
  if (m) return { keyTopic: cleanText(m[1]), isAbout: cleanText(m[2]) };
  return { keyTopic: null, isAbout: null };
}

function isClearKeyTopicLabel(label) {
  const kt = cleanText(label);
  if (kt.length < 3 || kt.length > 80) return false;
  if (/^(topic|stuff|things|history|government|science|math|literature)$/i.test(kt)) return false;
  if (kt.split(" ").length > 10) return false;
  return true;
}

function hasMeaningfulIsAbout(isAbout) {
  const ia = cleanText(isAbout);
  if (ia.length < 6 || ia.length > 220) return false;
  if (/^(stuff|things|a topic|government|history|science)\b/i.test(ia)) return false;
  return true;
}

function canLeadToMainIdeas(isAbout) {
  const ia = cleanText(isAbout);
  if (
    /(because|so that|how|why|caused by|leads to|results in|changed|influenced|prevents|helps|shows|explains|compares|contrasts)/i.test(
      ia
    )
  ) {
    return true;
  }
  return ia.split(" ").length >= 6;
}

function instructionalSufficiency(keyTopic, isAbout) {
  const hasLabel = isClearKeyTopicLabel(keyTopic);
  const hasDir = hasMeaningfulIsAbout(isAbout);
  const leads = canLeadToMainIdeas(isAbout);
  return { sufficient: hasLabel && hasDir && leads, hasLabel, hasDir, leads };
}

function looksStuck(text) {
  const t = cleanText(text).toLowerCase();
  return (
    t === "i am not sure" ||
    t === "im not sure" ||
    t === "not sure" ||
    t === "idk" ||
    t === "i don't know" ||
    t === "dont know" ||
    t === "i'm confused" ||
    t === "im confused" ||
    t === "confused" ||
    t === "help" ||
    t.length <= 6
  );
}

function looksLikeMetaRepeat(text) {
  const t = cleanText(text).toLowerCase();
  return /^(i already|i just|i said|i told you|i shared)/i.test(t);
}

/**
 * Recognize “main idea-ish” student text without requiring “One Main Idea…”
 * This is intentionally permissive.
 */
function looksLikeMainIdeaAttempt(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  // Explicit labels
  if (t.includes("main idea") || t.includes("main point") || t.includes("reason")) return true;

  // List-ish
  if (
    t.startsWith("first") ||
    t.startsWith("second") ||
    t.startsWith("another") ||
    t.startsWith("also") ||
    t.startsWith("one") ||
    t.startsWith("my first") ||
    t.startsWith("my second")
  ) {
    return true;
  }

  // Claim-ish sentence
  const hasVerb = /\b(is|are|was|were|shows|show|reveals|suggests|means|represents|highlights)\b/i.test(t);
  const longEnough = t.split(" ").length >= 5;
  return hasVerb && longEnough;
}

/**
 * Recognize “detail-ish” student text (examples, events, quotes, actions)
 * Also permissive; we rely on stepHint to know we are in details.
 */
function looksLikeDetailAttempt(text) {
  const t = cleanText(text);
  if (!t) return false;

  // Quotation marks, page refs, or named moments
  if (/[“”"]/.test(t)) return true;
  if (/\bpage\s*\d+\b/i.test(t)) return true;

  // Often details mention a specific event/action
  const hasPastVerb = /\b(said|says|did|does|happens|happened|goes|went|kills|killed|tells|told)\b/i.test(
    t.toLowerCase()
  );
  const longEnough = t.split(" ").length >= 4;
  return hasPastVerb && longEnough;
}

/**
 * Decide the best next ONE question (routine-faithful, minimal looping)
 *
 * stepHint expected values (from frontend lastBotIntent):
 * - "keyTopic" | "isAbout" | "mainIdeas" | "details" | "soWhat" | ""
 */
function nextFrameQuestion(studentMessage, framing = {}, stepHint = "") {
  const msg = cleanText(studentMessage);

  const framedKeyTopic = getFramingKeyTopic(framing);
  const framedIsAbout = getFramingIsAbout(framing);

  const hasKeyTopic = isClearKeyTopicLabel(framedKeyTopic);
  const hasIsAbout = hasMeaningfulIsAbout(framedIsAbout);

  // 0) If student is stuck: scaffold within current progress (DO NOT reset)
  if (looksStuck(msg)) {
    if (hasKeyTopic && hasIsAbout) {
      // Keep it in Main Ideas (not “back to key topic”)
      return "What is one Main Idea that supports your “is about” statement? (One sentence.)";
    }
    if (hasKeyTopic && !hasIsAbout) {
      return `Finish this: “${cleanText(framedKeyTopic)} is about ___” (one short phrase).`;
    }
    return "What is your Key Topic (just the title, 2–5 words)?";
  }

  // 1) Context-aware routing by what we asked last (key fix for realistic texting)
  if (stepHint === "keyTopic") {
    if (isClearKeyTopicLabel(msg)) {
      return `Great — now finish this: “${msg} is about ___” (one short phrase).`;
    }
    return "What is your Key Topic (just the title, 2–5 words)?";
  }

  if (stepHint === "isAbout") {
    // Treat most replies as an Is About attempt and move forward
    return "What are 1–3 Main Ideas that explain your topic?";
  }

  if (stepHint === "mainIdeas") {
    if (looksLikeMetaRepeat(msg)) {
      return "Restate your Main Idea as one sentence (start with the character, theme, or issue).";
    }
    // Accept normal student phrasing as a main idea and move to details
    return "What Essential Details (facts, examples, or evidence) support that Main Idea?";
  }

  if (stepHint === "details") {
    if (looksLikeMetaRepeat(msg)) {
      return "Name one specific moment, action, or quote that proves it (2–10 words).";
    }
    // Move to significance with a sentence frame to reduce looping
    return "So what? Finish this: “This detail matters because ___.”";
  }

  if (stepHint === "soWhat") {
    if (looksLikeMetaRepeat(msg)) {
      return "Say why your topic matters in real life (start with: “This matters because ___.”).";
    }
    if (msg.split(" ").length < 6) {
      return "Finish this: “This matters because ___.”";
    }
    // Push to audience/impact for a stronger So What
    return "Who is most affected by this idea, and how? (one sentence)";
  }

  // 2) If they typed an explicit "X is about Y", use it
  const { keyTopic, isAbout } = parseKeyTopicIsAbout(msg);
  const check = instructionalSufficiency(keyTopic, isAbout);

  if (check.sufficient) return "What are 1–3 Main Ideas that explain your topic?";

  // 3) If we already captured Key Topic/Is About in framing, move forward
  if (hasKeyTopic && hasIsAbout) {
    // If message looks like a main idea, move to details
    if (looksLikeMainIdeaAttempt(msg)) {
      return "What Essential Details (facts, examples, or evidence) support that Main Idea?";
    }
    // If message looks like a detail, move to so what
    if (looksLikeDetailAttempt(msg)) {
      return "So what? Finish this: “This detail matters because ___.”";
    }
    // Otherwise ask for main ideas
    return "What are 1–3 Main Ideas that explain your topic?";
  }

  // 4) If their message doesn't contain a usable label, ask for Key Topic
  if (!check.hasLabel && !hasKeyTopic) return "What is your Key Topic (just the title, 2–5 words)?";

  // 5) Otherwise ask for is-about
  const usableKeyTopic = check.hasLabel ? keyTopic : framedKeyTopic;
  return `Great — now finish this: “${cleanText(usableKeyTopic)} is about ___” (one short phrase).`;
}

export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method Not Allowed" });

  try {
    // stepHint should come from frontend (your lastBotIntent)
    const { message = "", framing = {}, stepHint = "" } = req.body || {};
    const trimmed = String(message).trim();

    if (!trimmed) {
      return res.status(400).json({ error: "Missing 'message' in request body" });
    }

    const routedQuestion = nextFrameQuestion(trimmed, framing, stepHint);

    // ---- Safety check ----
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      const safeReply =
        SAFETY_RESPONSES[safety.category] ||
        "Let’s zoom back to your Frame. What is your Key Topic (just the title, 2–5 words)?";

      return res.status(200).json({
        reply: enforceHardCap(safeReply),
        flagged: true,
        flagCategory: safety.category || "unknown",
        severity: safety.severity || "low",
        safetyMode: true,
      });
    }

    // ---- OpenAI call ----
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      max_tokens: MAX_MODEL_TOKENS,
      messages: [
        { role: "system", content: SYSTEM_PROMPT_FRAMING },
        {
          role: "user",
          content:
            `Student said: ${trimmed}\n\n` +
            `Ask EXACTLY this next step as ONE question:\n${routedQuestion}`,
        },
      ],
    });

    const raw =
      completion?.choices?.[0]?.message?.content?.trim() ||
      "What is your Key Topic (just the title, 2–5 words)?";

    const reply = socraticFailSafe(raw, routedQuestion);

    return res.status(200).json({
      reply,
      flagged: false,
      flagCategory: "",
      severity: "",
      safetyMode: false,
    });
  } catch (err) {
    console.error("Tutor API error:", err);
    return res.status(500).json({
      error: "Server error",
      details: err?.message || String(err),
    });
  }
}
