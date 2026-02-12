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
 *
 * IMPORTANT:
 * We compute the next question deterministically with the router.
 * We return THAT question directly (no model rewriting).
 * This prevents drifting/backtracking/term invention.
 */

const SYSTEM_PROMPT_FRAMING = `
You are Kaw Companion, a Socratic tutor for grades 4–12 using the Framing Routine.

NON-NEGOTIABLE RULES (always):
- Use QUESTIONS ONLY. Do not explain. Do not lecture. Do not summarize.
- Never give direct answers, solutions, or “the correct response.”
- Never confirm correctness (no “Correct,” “That’s right,” etc.).
- Ask EXACTLY 1 question per turn (no more).
- Keep responses short and consistent.

STRICT PROGRESSION RULES (must follow):
- Stay in the current framing step until it is complete.
- Never introduce a new framing step early.
- Never return to an earlier framing step once it is complete.
- Do NOT invent or introduce new academic terms.
- Use ONLY the following framing terms:
  Key Topic, Is About, Main Ideas, Essential Details, So What.

STUCK STUDENT RULE:
If a student is unsure or stuck, scaffold WITHIN the CURRENT step
by rephrasing the question or giving a partial example,
but do not advance or backtrack.

FRAMING ROUTINE SEQUENCE (must follow):
1) Key Topic (2–5 words: the focus/title)
2) Is About (one short phrase that explains what the Key Topic focuses on)
3) Main Ideas (2–3 important ideas that explain or support the topic)
4) Essential Details (2–3 evidence/examples for EACH main idea)
5) So What? (why it matters / significance / conclusion)

EXTRACTION RULE (critical):
If a student response contains BOTH the Key Topic and the Is About
(e.g., "The Cuban Missile Crisis is about..."),
extract both elements immediately.
Do NOT ask again for Key Topic or Is About.

NO REPEAT RULE (critical):
Once a framing step is complete,
do NOT repeat that step again,
even if the student restates it differently.
Always move forward in the sequence.

TOPIC STABILITY RULE (critical):
During Main Ideas and Essential Details:
- EVERY question must explicitly include the original Key Topic.
- NEVER treat a Main Idea as if it becomes the new Key Topic.
- NEVER ask "supports ___" unless the blank is the original Key Topic
  or (ONLY during Essential Details) the current Main Idea being detailed.

Example (must follow):
If the Key Topic is "Cuban Missile Crisis" and a student says
"Castro’s rise to power," the next question MUST still reference
"the Cuban Missile Crisis," not Castro’s rise to power.

Required template during Main Ideas:
"What is another Main Idea that helps explain the [KEY TOPIC]?"

OUTPUT FORMAT:
Return only the single question. No headings. No bullets.
`.trim();

// ---- Hard caps ----
const MAX_MODEL_TOKENS = 140;
const MAX_CHARS = 420;
const MAX_QUESTIONS = 1;

// -----------------------------------------------
// Helpers
// -----------------------------------------------
function cleanText(s) {
  return String(s || "").replace(/\s+/g, " ").trim();
}

function clampQuestion(q) {
  let out = cleanText(q);

  // hard length clamp
  if (out.length > MAX_CHARS) out = out.slice(0, MAX_CHARS).trim();

  // ensure it ends with a question mark
  if (!out.endsWith("?")) out = out.replace(/[.!\s]*$/, "") + "?";

  // ensure only one question
  const lastQ = out.lastIndexOf("?");
  if (lastQ !== out.length - 1) out = out.slice(0, lastQ + 1).trim();

  return out;
}

// Pull state from framing object (supports a few key styles)
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
function getFramingMainIdeas(framing) {
  return (
    framing?.mainIdeas ??
    framing?.main_ideas ??
    framing?.frame?.mainIdeas ??
    framing?.frame?.main_ideas ??
    []
  );
}
function getFramingDetails(framing) {
  return (
    framing?.details ??
    framing?.frame?.details ??
    {}
  );
}
function getFramingDetailsIndex(framing) {
  return (
    framing?.detailsIndex ??
    framing?.details_index ??
    framing?.frame?.detailsIndex ??
    framing?.frame?.details_index ??
    0
  );
}
function getFramingSoWhat(framing) {
  return (
    framing?.soWhat ??
    framing?.so_what ??
    framing?.frame?.soWhat ??
    framing?.frame?.so_what ??
    ""
  );
}

const GENERIC_LEFT_SIDES = new Set([
  "my assignment",
  "the assignment",
  "my essay",
  "this essay",
  "my paper",
  "this paper",
  "my paragraph",
  "this paragraph",
  "my topic",
  "the topic",
  "it",
  "this",
  "that",
  "key topic",
  "topic",
]);

function looksGenericKeyTopic(left) {
  const l = cleanText(left).toLowerCase();
  if (!l) return true;
  if (GENERIC_LEFT_SIDES.has(l)) return true;
  // also block "my ____" patterns
  if (l.startsWith("my ")) return true;
  if (l.startsWith("this ")) return true;
  if (l.startsWith("the ")) return false; // "the Cuban Missile Crisis" is fine
  return false;
}

// Parse "X is about Y" from a student message
function parseKeyTopicIsAbout(text) {
  const t = cleanText(text);
  const m = t.match(/^(.*?)\s+is\s+about\s+(.+?)(?:[.?!]|$)/i);
  if (!m) return null;

  const left = cleanText(m[1]);
  const right = cleanText(m[2]);

  // basic sanity
  if (!left || !right) return null;

  // avoid junk like "My assignment is about ..."
  if (looksGenericKeyTopic(left)) return null;

  // keep Key Topic short-ish (2–8 words is realistic)
  const wordCount = left.split(" ").filter(Boolean).length;
  if (wordCount < 1 || wordCount > 10) return null;

  return { keyTopic: left, isAbout: right };
}

// -----------------------------------------------
// Router: decide the SINGLE next question
// -----------------------------------------------
function nextFrameQuestion(studentMessage, framing, stepHint = "") {
  const msg = cleanText(studentMessage);

  // read current state
  const keyTopic = cleanText(getFramingKeyTopic(framing));
  const isAbout = cleanText(getFramingIsAbout(framing));
  const mainIdeas = Array.isArray(getFramingMainIdeas(framing))
    ? getFramingMainIdeas(framing).map(cleanText).filter(Boolean)
    : [];
  const details = getFramingDetails(framing) || {};
  const detailsIndex = Number(getFramingDetailsIndex(framing) || 0);
  const soWhat = cleanText(getFramingSoWhat(framing));

  // EXTRACTION RULE safety net (server-side)
  const parsed = parseKeyTopicIsAbout(msg);

  // 1) Key Topic missing
  if (!keyTopic) {
    // If student gave full "X is about Y", move forward (don’t loop)
    if (parsed?.keyTopic) {
      return `What is your Is About statement for "${parsed.keyTopic}"?`;
    }
    return "What is your Key Topic?";
  }

  // 2) Is About missing
  if (!isAbout) {
    // If student gave full "X is about Y", move forward (don’t loop)
    if (parsed?.keyTopic && parsed?.isAbout) {
      // Key Topic exists; if parsed keyTopic conflicts, ignore the conflict and proceed
      return `What is one Main Idea that helps explain the ${keyTopic}?`;
    }
    return `Finish this sentence: "${keyTopic} is about ____."`;
  }

  // 3) Main Ideas phase (need 2, allow 3)
  if (mainIdeas.length < 2) {
    if (mainIdeas.length === 0) {
      return `What is one Main Idea that helps explain the ${keyTopic}?`;
    }
    // REQUIRED template during Main Ideas (prevents "supports X" drift)
    return `What is another Main Idea that helps explain the ${keyTopic}?`;
  }

  // Optional 3rd main idea (if not already 3)
  // Only ask for a third if the client explicitly signaled it, otherwise proceed to details.
  // (This keeps demos predictable.)
  if (mainIdeas.length === 2 && stepHint === "mainIdeas") {
    return `Do you have one more Main Idea that helps explain the ${keyTopic}?`;
  }

  // 4) Essential Details for each main idea (2–3 each)
  // We only ever reference the ORIGINAL Key Topic + the CURRENT Main Idea
  const idx = Math.max(0, Math.min(detailsIndex, mainIdeas.length - 1));
  const currentMainIdea = mainIdeas[idx] || mainIdeas[0];

  const bucket = details?.[String(idx)] || details?.[idx] || [];
  const detailCount = Array.isArray(bucket) ? bucket.filter(Boolean).length : 0;

  if (detailCount < 2) {
    return `What is one Essential Detail that supports the Main Idea "${currentMainIdea}" about the ${keyTopic}?`;
  }

  // if they’ve got 2+ details for this main idea and there are more main ideas to cover
  if (idx < mainIdeas.length - 1) {
    const nextIdea = mainIdeas[idx + 1];
    return `What is one Essential Detail that supports the Main Idea "${nextIdea}" about the ${keyTopic}?`;
  }

  // 5) So What
  if (!soWhat) {
    return `So what? Why does the ${keyTopic} matter?`;
  }

  // If everything is filled, keep it on So What refinement (still one question)
  return `What is one way you could make your So What for the ${keyTopic} more specific?`;
}

// -----------------------------------------------
// Handler
// -----------------------------------------------
export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method Not Allowed" });

  try {
    const { message = "", framing = {}, stepHint = "" } = req.body || {};
    const trimmed = cleanText(message);

    if (!trimmed) {
      return res.status(400).json({ error: "Missing 'message' in request body" });
    }

    // ---- Safety check ----
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      const safeReply =
        SAFETY_RESPONSES[safety.category] ||
        "Let’s zoom back to your Frame. What is your Key Topic? (2–5 words.)";
      return res.status(200).json({ reply: clampQuestion(safeReply) });
    }

    // ---- Deterministic next question (no model rewrite) ----
    const routedQuestion = nextFrameQuestion(trimmed, framing, stepHint);

    // Keep output compliant
    const reply = clampQuestion(routedQuestion);

    return res.status(200).json({ reply });
  } catch (err) {
    console.error("Tutor handler error:", err);
    return res.status(500).json({ error: "Internal Server Error" });
  }
}
