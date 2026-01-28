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
 * - Questions only
 * - One step at a time (1–2 questions max)
 * - Never provide answers or explanations
 * - Always anchor to the Frame structure:
 *   Key Topic → Is About → Main Ideas → Essential Details → So What?
 */
const SYSTEM_PROMPT_FRAMING = `
You are Kaw Companion, a Socratic tutor for grades 4–12 using the Framing Routine.

NON-NEGOTIABLE RULES (always):
- Use QUESTIONS ONLY. Do not explain. Do not lecture. Do not summarize content for the student.
- Never give direct answers, solutions, or “the correct response.”
- Never confirm correctness (no “Correct,” “That’s right,” etc.).
- Ask 1–2 questions per turn (max).
- Keep responses short and consistent.

FRAME ANCHOR (always guide thinking through this structure):
1) Key Topic (name the focus)
2) Is About (one short phrase describing what the topic is about)
3) Main Idea(s) (1–3 important ideas)
4) Essential Details (details/evidence supporting each main idea)
5) So What? (why it matters / significance)

REDIRECT RULE:
If the student asks for an answer or you feel pulled into explaining, redirect back into the Frame with 1–2 questions.

TONE:
Calm, teacher-like, encouraging, neutral toward correctness.

OUTPUT FORMAT:
Return only the questions (no headings, no bullets unless the student already used them).
`.trim();

// ---- Hard caps (tuneable) ----
const MAX_MODEL_TOKENS = 180;      // keeps responses short
const MAX_CHARS = 650;             // hard cap in characters for UI stability
const MAX_QUESTIONS = 2;           // enforce 1–2 questions

function countQuestions(text) {
  return (text.match(/\?/g) || []).length;
}

// crude “lecture-y” detectors (seatbelt triggers)
function looksLikeExplanation(text) {
  const t = text.toLowerCase();
  const badStarts = ["in summary", "to summarize", "here's", "this means", "the answer is", "overall,"];
  const hasBadStart = badStarts.some((s) => t.trim().startsWith(s));
  const tooManySentences = (text.match(/[.!]/g) || []).length >= 3; // tends to be paragraph-y
  const hasColonList = text.includes(":") && (text.match(/\n/g) || []).length >= 1;
  return hasBadStart || tooManySentences || hasColonList;
}

function enforceHardCap(text) {
  let out = (text || "").trim();

  // Strip any accidental “labels”
  out = out.replace(/^(kaw companion|tutor|assistant)\s*:\s*/i, "").trim();

  // Enforce question count by truncation
  if (countQuestions(out) > MAX_QUESTIONS) {
    // keep only up to the 2nd question mark
    let qCount = 0;
    let cutIndex = out.length;
    for (let i = 0; i < out.length; i++) {
      if (out[i] === "?") {
        qCount++;
        if (qCount === MAX_QUESTIONS) {
          cutIndex = i + 1;
          break;
        }
      }
    }
    out = out.slice(0, cutIndex).trim();
  }

  // Enforce character cap
  if (out.length > MAX_CHARS) out = out.slice(0, MAX_CHARS).trim();

  // Must end with a question mark (questions-only)
  if (!out.includes("?")) {
    out = "Which part of the Frame are you working on right now (Key Topic, Is About, Main Ideas, Essential Details, or So What?)?";
  }

  return out;
}

function socraticFailSafe(original, studentMessage) {
  // If reply is too long or explanation-y, replace with a Frame-based redirect
  const cleaned = (original || "").trim();
  const triggers =
    cleaned.length > MAX_CHARS ||
    looksLikeExplanation(cleaned) ||
    countQuestions(cleaned) === 0 ||
    countQuestions(cleaned) > MAX_QUESTIONS;

  if (!triggers) return enforceHardCap(cleaned);

  // Fail-safe prompts: 1–2 questions, Frame-anchored, no content giving
  // Slightly adapt based on whether student message is vague
  const msg = (studentMessage || "").trim();
  const vague = msg.length < 20;

  const fallback = vague
    ? "What’s the Key Topic you’re working on right now? What does the task say you need to produce or turn in at the end?"
    : "Which part of the Frame are you working on right now (Key Topic, Is About, Main Ideas, Essential Details, or So What?)? What is your next best guess for that box?";

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

    // ---- Safety check ----
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
      "Which part of the Frame are you working on right now?";

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
