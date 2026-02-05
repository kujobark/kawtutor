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
 * - Support students with the Frame routine (Key Topic → Is About → Main Ideas → Details → So What)
 * - Ask EXACTLY 1 question per turn
 * - Never lecture / never provide answers
 * - Avoid confusing “pause/safety” phrasing; always route back into the routine
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
  if (m) {
    return { keyTopic: cleanText(m[1]), isAbout: cleanText(m[2]) };
  }

  return { keyTopic: null, isAbout: null };
}

function isClearKeyTopicLabel(label) {
  const kt = cleanText(label);
  if (kt.length < 3 || kt.length > 80) return false;

  if (/^(topic|stuff|things|history|government|science|math|literature)$/i.test(kt))
    return false;

  if (kt.split(" ").length > 10) return false;

  return true;
}

function hasMeaningfulIsAbout(isAbout) {
  const ia = cleanText(isAbout);
  if (ia.length < 8 || ia.length > 220) return false;

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

  return ia.split(" ").length >= 8;
}

function instructionalSufficiency(keyTopic, isAbout) {
  const hasLabel = isClearKeyTopicLabel(keyTopic);
  const hasDir = hasMeaningfulIsAbout(isAbout);
  const leads = canLeadToMainIdeas(isAbout);
  return { sufficient: hasLabel && hasDir && leads, hasLabel, hasDir, leads };
}

function looksLikeMainIdeaAttempt(text) {
  const t = cleanText(text).toLowerCase();
  return (
    t.includes("main idea") ||
    t.startsWith("my first") ||
    t.startsWith("first") ||
    t.startsWith("my second") ||
    t.startsWith("second") ||
    t.startsWith("one reason") ||
    t.startsWith("another reason") ||
    t.startsWith("another main idea")
  );
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

/**
 * Decide the best next ONE question (routine-faithful, no looping)
 * - Uses framing state to avoid "jumping backwards" when student is stuck.
 */
function nextFrameQuestion(studentMessage, framing = {}, step = "") {
  const msg = cleanText(studentMessage);

  const framedKeyTopic = getFramingKeyTopic(framing);
  const framedIsAbout = getFramingIsAbout(framing);

  const hasKeyTopic = isClearKeyTopicLabel(framedKeyTopic);
  const hasIsAbout = hasMeaningfulIsAbout(framedIsAbout);

  // If student is stuck: scaffold within current progress (DO NOT reset)
  if (looksStuck(msg)) {
    if (hasKeyTopic && hasIsAbout) {
      // Stay on MAIN IDEAS (scaffold concrete entry point)
      return "Name one character, scene, or moment that best shows your topic (2–5 words).";
    }
    if (hasKeyTopic && !hasIsAbout) {
      // Stay on IS ABOUT
      return `Finish this: “${cleanText(framedKeyTopic)} is about ___” (one short phrase).`;
    }
    // We truly need Key Topic
    return "What is your Key Topic (just the title, 2–5 words)?";
  }

  // If they look like they are writing main ideas, push to details next
  if (looksLikeMainIdeaAttempt(msg)) {
    return "What Essential Details (facts, examples, or evidence) support that Main Idea?";
  }

  // If they typed an explicit "X is about Y", use it
  const { keyTopic, isAbout } = parseKeyTopicIsAbout(msg);
  const check = instructionalSufficiency(keyTopic, isAbout);

  if (check.sufficient) {
    return "What are 1–3 Main Ideas that explain your topic?";
  }

  // If we already captured Key Topic/Is About in framing, move forward
  if (hasKeyTopic && hasIsAbout) {
    return "What are 1–3 Main Ideas that explain your topic?";
  }

  // If their message doesn't contain a usable label, ask for Key Topic
  if (!check.hasLabel && !hasKeyTopic) {
    return "What is your Key Topic (just the title, 2–5 words)?";
  }

  // If we have (or inferred) a key topic but not is-about, ask for is-about
  const usableKeyTopic = check.hasLabel ? keyTopic : framedKeyTopic;
  return `Great — now finish this: “${cleanText(usableKeyTopic)} is about ___” (one short phrase).`;
}

export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method Not Allowed" });

  try {
    const { message = "", framing = {}, step = "" } = req.body || {};
    const trimmed = String(message).trim();

    if (!trimmed) {
      return res.status(400).json({ error: "Missing 'message' in request body" });
    }

    const routedQuestion = nextFrameQuestion(trimmed, framing, step);

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
