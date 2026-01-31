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
 * Framing Routine + Socratic Guardrails (Demo)
 * Goals (tight + UI-friendly):
 * - QUESTIONS ONLY (no explaining, no answering)
 * - EXACTLY 1 question per turn
 * - Strong priority on Key Topic → Is About before anything else
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
1) Key Topic (2–5 words: the focus)
2) Is About (one short phrase: what the topic is about)
3) Main Ideas (1–3 important ideas)
4) Essential Details (evidence/examples for each main idea)
5) So What? (why it matters / significance)

ANCHOR RULE:
- If Key Topic is not clear, ask ONLY for Key Topic.
- If Key Topic is clear but Is About is not clear, ask ONLY for Is About.
- Do not ask “meaning/definition” questions before Key Topic + Is About are captured.

REDIRECT RULE:
If the student asks for an answer or you feel pulled into explaining, redirect to the next Frame step with ONE question.

TONE:
Calm, teacher-like, encouraging, neutral toward correctness.

OUTPUT FORMAT:
Return only the single question. No headings. No bullets.
`.trim();

// ---- Hard caps (tuneable) ----
const MAX_MODEL_TOKENS = 120;  // shorter responses = UI stability
const MAX_CHARS = 420;         // hard cap for transcript container stability
const MAX_QUESTIONS = 1;       // enforce exactly 1 question

function countQuestions(text) {
  return (text.match(/\?/g) || []).length;
}

// crude “lecture-y” detectors (seatbelt triggers)
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
  const tooManySentences = (text.match(/[.!]/g) || []).length >= 2; // keep VERY short
  const hasColonList = text.includes(":") && (text.match(/\n/g) || []).length >= 1;
  return hasBadStart || tooManySentences || hasColonList;
}

function enforceHardCap(text) {
  let out = (text || "").trim();

  // Strip any accidental “labels”
  out = out.replace(/^(kaw companion|tutor|assistant)\s*:\s*/i, "").trim();

  // Enforce question count by truncation (keep ONLY the first question)
  if (countQuestions(out) > MAX_QUESTIONS) {
    const firstQ = out.indexOf("?");
    out = firstQ >= 0 ? out.slice(0, firstQ + 1).trim() : out.trim();
  }

  // Enforce character cap
  if (out.length > MAX_CHARS) out = out.slice(0, MAX_CHARS).trim();

  // Must contain a question mark (questions-only)
  if (!out.includes("?")) {
    out = "What is your Key Topic (2–5 words)?";
  }

  // Must end with a single question mark
  const lastQ = out.lastIndexOf("?");
  if (lastQ !== out.length - 1) out = out.slice(0, lastQ + 1).trim();

  return out;
}

function socraticFailSafe(original, studentMessage) {
  const cleaned = (original || "").trim();

  // Seatbelt triggers: too long, too lecture-y, or not exactly 1 question
  const triggers =
    cleaned.length > MAX_CHARS ||
    looksLikeExplanation(cleaned) ||
    countQuestions(cleaned) !== 1;

  if (!triggers) return enforceHardCap(cleaned);

  // Strong reset: always return to Key Topic first
  const msg = (studentMessage || "").trim();
  const vague = msg.length < 20;

  const fallback = vague
    ? "What is your Key Topic (2–5 words)?"
    : "What is your Key Topic (2–5 words) based on what you just wrote?";

  return enforceHardCap(fallback);
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

    // ---- Safety check (Step 8 lives here via these imports) ----
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      return res.status(200).json({
        reply:
          SAFETY_RESPONSES[safety.category] ||
          "Let’s pause for a moment and try approaching this in a different way.",
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
        { role: "user", content: trimmed },
      ],
    });

    const raw =
      completion?.choices?.[0]?.message?.content?.trim() ||
      "What is your Key Topic (2–5 words)?";

    const reply = socraticFailSafe(raw, trimmed);

    return res.status(200).json({
      reply,
      flagged: false,
      flagCategory: "",
      severity: "",
      safetyMode: false,
    });
  } catch (err) {
    console.error("Tutor API error:", err);
    return res.status(500).json({ error: "Server error", details: err?.message || String(err) });
  }
}
