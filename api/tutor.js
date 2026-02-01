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

  const fallback = (routedQuestion || "What is your Key Topic (just the title, 2–5 words)?").trim();
  return enforceHardCap(fallback);
}

// -----------------------------------------------
// Frame parsing + instructional sufficiency checks
// -----------------------------------------------
function cleanText(s) {
  return String(s || "").replace(/\s+/g, " ").trim();
}

// Parse a student attempt like: "X is about Y"
function parseKeyTopicIsAbout(message) {
  const t = cleanText(message);

  // explicit "is about"
  const m = t.match(/^(.*?)\s+is about\s+(.+?)(?:[.?!]|$)/i);
  if (m) {
    return { keyTopic: cleanText(m[1]), isAbout: cleanText(m[2]) };
  }

  return { keyTopic: null, isAbout: null };
}

function isClearKeyTopicLabel(label) {
  const kt = cleanText(label);
  if (kt.length < 3 || kt.length > 80) return false;

  // too generic (tweak list anytime)
  if (/^(topic|stuff|things|history|government|science|math|literature)$/i.test(kt)) return false;

  // not a title if it's a full sentence
  if (kt.split(" ").length > 10) return false;

  return true;
}

function hasMeaningfulIsAbout(isAbout) {
  const ia = cleanText(isAbout);
  if (ia.length < 8 || ia.length > 220) return false;

  // too vague
  if (/^(stuff|things|a topic|government|history|science)\b/i.test(ia)) return false;

  return true;
}

function canLeadToMainIdeas(isAbout) {
  const ia = cleanText(isAbout);

  // relationship/process cues
  if (
    /(because|so that|how|why|caused by|leads to|results in|changed|influenced|prevents|helps|shows|explains|compares|contrasts)/i.test(
      ia
    )
  ) {
    return true;
  }

  // or simply specific enough length
  return ia.split(" ").length >= 8;
}

function instructionalSufficiency(keyTopic, isAbout) {
  const hasLabel = isClearKeyTopicLabel(keyTopic);
  const hasDir = hasMeaningfulIsAbout(isAbout);
  const leads = canLeadToMainIdeas(isAbout);
  return { sufficient: hasLabel && hasDir && leads, hasLabel, hasDir, leads };
}

/**
 * Decide the best next ONE question (routine-faithful, no looping)
 * - If student already gives sufficient "X is about Y" → ask Main Ideas next.
 * - If not, split: ask title first, then is about scaffold.
 */

function looksLikeMainIdeaAttempt(text) {
  const t = cleanText(text).toLowerCase();

  // common student signals they are writing a main idea
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

function nextFrameQuestion(studentMessage) {
  const msg = cleanText(studentMessage);
    if (looksLikeMainIdeaAttempt(msg)) {
return "What Essential Details (facts, examples, or evidence) support that Main Idea?";
  }
const { keyTopic, isAbout } = parseKeyTopicIsAbout(msg);
  const check = instructionalSufficiency(keyTopic, isAbout);

  if (check.sufficient) {
    return "What are 1–3 Main Ideas that explain your topic?";
  }

  if (!check.hasLabel) {
    return "What is your Key Topic (just the title, 2–5 words)?";
  }

  return `Great — now finish this: “${keyTopic} is about ___” (one short phrase).`;
}

export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method Not Allowed" });

  try {
    const { message = "" } = req.body || {};
    const trimmed = String(message).trim();

    if (!trimmed) {
      return res.status(400).json({ error: "Missing 'message' in request body" });
    }

    const routedQuestion = nextFrameQuestion(trimmed);

    // ---- Safety check ----
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      // IMPORTANT: No “pause” framing; route back into the routine
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

