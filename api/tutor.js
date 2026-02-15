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

// ---- UTIL ----
function cleanText(s) {
  return (s || "").toString().trim().replace(/\s+/g, " ");
}
function isNegative(s) {
  const t = cleanText(s).toLowerCase();
  return t === "no" || t === "nope" || t === "nah" || t === "n/a" || t === "none";
}
function isAffirmative(s) {
  const t = cleanText(s).toLowerCase();
  return t === "yes" || t === "y" || t === "yeah" || t === "yep" || t === "sure" || t === "correct";
}

// Keep reply as a SINGLE question
function enforceSingleQuestion(text) {
  let out = (text || "").toString().trim();
  if (!out) return "Can you say more?";
  // Keep only up to first question mark
  const q = out.indexOf("?");
  if (q >= 0) out = out.slice(0, q + 1).trim();
  // If no question mark, turn final punctuation into a question mark
  if (!out.includes("?")) {
    if (!out.endsWith("?")) out = out.replace(/[.!\s]*$/, "") + "?";
    const lastQ = out.lastIndexOf("?");
    if (lastQ !== out.length - 1) out = out.slice(0, lastQ + 1).trim();
  }
  return out;
}

const GENERIC_KEY_TOPICS = new Set([
  "my assignment", "the assignment", "my essay", "this essay", "my paper", "this paper",
  "my paragraph", "this paragraph", "my topic", "the topic", "topic", "key topic", "it",
  "this", "that"
]);

function isBadKeyTopic(keyTopic) {
  const kt = cleanText(keyTopic).toLowerCase();
  if (!kt) return true;
  if (GENERIC_KEY_TOPICS.has(kt)) return true;
  if (kt.startsWith("my ")) return true;
  return false;
}

// Parse pattern: "X is about Y"
function parseKeyTopicIsAbout(msg) {
  const m = cleanText(msg);
  const idx = m.toLowerCase().indexOf(" is about ");
  if (idx < 0) return null;

  const keyTopic = cleanText(m.slice(0, idx));
  const isAbout = cleanText(m.slice(idx + " is about ".length));

  if (!keyTopic || !isAbout) return null;
  if (isBadKeyTopic(keyTopic)) return null;

  // Key topic should be short (2–5 words)
  const wc = keyTopic.split(/\s+/).filter(Boolean).length;
  if (wc < 2 || wc > 5) return null;

  return { keyTopic, isAbout };
}

// ---- STATE ----
function defaultState() {
  return {
    version: 1,
    frame: {
      keyTopic: "",
      isAbout: "",
      mainIdeas: [],
      // ✅ index-based buckets: details[0] supports mainIdeas[0], etc.
      details: [],
      soWhat: ""
    },
    pending: null
  };
}

function normalizeIncomingState(raw) {
  const s = raw && typeof raw === "object" ? raw : {};
  const base = defaultState();

  const frame = s.frame && typeof s.frame === "object" ? s.frame : {};

  base.frame.keyTopic = cleanText(frame.keyTopic || s.keyTopic || "");
  base.frame.isAbout = cleanText(frame.isAbout || s.isAbout || "");

  base.frame.mainIdeas = Array.isArray(frame.mainIdeas)
    ? frame.mainIdeas.map(cleanText).filter(Boolean)
    : [];

  // details normalization + migration
  if (Array.isArray(frame.details)) {
    base.frame.details = frame.details.map((bucket) =>
      Array.isArray(bucket) ? bucket.map(cleanText).filter(Boolean) : []
    );
  } else if (frame.details && typeof frame.details === "object") {
    const obj = frame.details;
    base.frame.details = base.frame.mainIdeas.map((mi) => {
      const bucket = obj[mi];
      return Array.isArray(bucket) ? bucket.map(cleanText).filter(Boolean) : [];
    });
  } else {
    base.frame.details = [];
  }

  base.frame.soWhat = cleanText(frame.soWhat || s.soWhat || "");

  base.pending = s.pending && typeof s.pending === "object" ? s.pending : null;

  return base;
}

// ---- PROGRESSION ----
function computeNextQuestion(state) {
  const s = state;

  // 🔒 Is About confirmation checkpoint (Phase 2)
  if (s.pending?.type === "confirmIsAbout") {
    return `"${s.frame.keyTopic}" is about "${s.frame.isAbout}". Is that correct, or would you like to revise it?`;
  }

  // 🔒 Main Ideas confirmation checkpoint (Phase 2)
  if (s.pending?.type === "confirmMainIdeas") {
    const lines = (s.frame.mainIdeas || []).map((mi, i) => `${i + 1}) ${mi}`).join("\n");
    return `You have identified the following Main Ideas:\n${lines}\nIs that correct, or would you like to revise one?`;
  }

  // 🧠 After Main Idea #2, offer an optional 3rd
  if (s.pending?.type === "offerThirdMainIdea") {
    return "Do you have a third Main Idea? (yes/no)";
  }

  // 🧠 If they said yes, collect the 3rd main idea
  if (s.pending?.type === "collectThirdMainIdea") {
    return `What is your third Main Idea that helps explain ${s.frame.keyTopic}?`;
  }

  if (!s.frame.keyTopic) {
    return "What is your Key Topic? (2–5 words)";
  }

  if (!s.frame.isAbout) {
    return `Finish this sentence: "${s.frame.keyTopic} is about ____."`;
  }

  if (s.frame.mainIdeas.length < 2) {
    return s.frame.mainIdeas.length === 0
      ? `What is your first Main Idea that helps explain ${s.frame.keyTopic}?`
      : `What is your second Main Idea that helps explain ${s.frame.keyTopic}?`;
  }

  // ✅ Details: collect 2 details per main idea (INDEX-BASED)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const mi = s.frame.mainIdeas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      return `Give one Essential Detail (fact/example) that supports this Main Idea: "${mi}".`;
    }
  }

  if (!s.frame.soWhat) {
    return `So what? Why does "${s.frame.keyTopic}" matter? (1–2 sentences)`;
  }

  return `Want to refine anything (Key Topic, Is About, Main Ideas, Details, or So What)?`;
}

// ---- STATE UPDATE (SSOT) ----
function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);

  // Ensure details is array buckets (defensive)
  if (!Array.isArray(s.frame.details)) s.frame.details = [];

  // 0) Handle pending confirm steps first
  if (s.pending?.type === "confirmIsAbout") {
    const normalized = msg.toLowerCase().trim();

    // Pure confirmation only
    if (isAffirmative(normalized) && normalized.length <= 5) {
      s.pending = null;
      return s;
    }

    // Otherwise treat message as revised Is About
    s.frame.isAbout = msg;
    s.pending = null;
    return s;
  }

  // 0a) Handle Main Ideas confirmation checkpoint
  if (s.pending?.type === "confirmMainIdeas") {
    const normalized = msg.toLowerCase().trim();

    // Pure confirmation only
    if (isAffirmative(normalized) && normalized.length <= 5) {
      s.pending = null;
      return s;
    }

    // For now: hold at checkpoint (revision routing later)
    return s;
  }

  // 0b) After Main Idea #2, offer optional 3rd (yes/no)
  if (s.pending?.type === "offerThirdMainIdea") {
    const normalized = msg.toLowerCase().trim();

    // Pure yes only → collect third main idea next
    if (isAffirmative(normalized) && normalized.length <= 5) {
      s.pending = { type: "collectThirdMainIdea" };
      return s;
    }

    // Otherwise treat as "no" and proceed to confirm main ideas
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  // 0c) Collect the 3rd Main Idea
  if (s.pending?.type === "collectThirdMainIdea") {
    if (!isNegative(msg)) {
      s.frame.mainIdeas.push(msg);
    }
    // After MI #3, confirm the full main ideas set
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  // Extraction rule (server-side)
  const parsed = parseKeyTopicIsAbout(msg);
  if (parsed) {
    if (!s.frame.keyTopic) s.frame.keyTopic = parsed.keyTopic;
    if (!s.frame.isAbout) s.frame.isAbout = parsed.isAbout;

    // 🔒 Trigger confirmation checkpoint
    s.pending = { type: "confirmIsAbout" };
    return s;
  }

  // ✅ FIX #1: accept plain Key Topic (2–5 words)
  if (!s.frame.keyTopic) {
    const wc = msg.split(/\s+/).filter(Boolean).length;
    if (!isBadKeyTopic(msg) && wc >= 2 && wc <= 5) {
      s.frame.keyTopic = msg;
    }
  }

  // If key topic missing, nothing else to store
  if (!s.frame.keyTopic) return s;

  // ✅ FIX #2: If collecting “Is About”, accept plain sentence/phrase and confirm it
  if (!s.frame.isAbout) {
    const lowered = msg.toLowerCase().trim();
    if (lowered !== "revise" && lowered !== "change") {
      s.frame.isAbout = msg;

      // 🔒 Trigger confirmation checkpoint
      s.pending = { type: "confirmIsAbout" };
    }
    return s;
  }

  // Main Ideas collection (need 2, then offer optional 3rd)
  if (s.frame.mainIdeas.length < 2) {
    if (!isNegative(msg)) {
      s.frame.mainIdeas.push(msg);

      // ✅ After the 2nd Main Idea, ask if they have a 3rd
      if (s.frame.mainIdeas.length === 2) {
        s.pending = { type: "offerThirdMainIdea" };
      }
    }
    return s;
  }

  // ✅ Details collection: 2 per main idea (INDEX-BASED)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      if (!isNegative(msg)) {
        s.frame.details[i] = [...arr, msg];
      }
      return s;
    }
  }

  // So What
  if (!s.frame.soWhat) {
    if (!isNegative(msg)) {
      s.frame.soWhat = msg;
    }
    return s;
  }

  return s;
}

// ---- PROMPT BUILD (kept for future; deterministic flow currently ignores model rewrite) ----
function buildSystemPrompt(state, intake) {
  const s = state;

  const intakeBlock =
    intake && typeof intake === "object"
      ? `\nINTAKE (optional metadata):\n- subject: ${cleanText(intake.subject)}\n- task: ${cleanText(intake.task)}\n- hardest: ${cleanText(intake.hardest)}\n- about: ${cleanText(intake.about)}\n`
      : "";

  return `
You are Kaw Companion, a Socratic tutor that guides a student through a framing routine.
Rules:
- Ask EXACTLY ONE question per reply.
- Be brief, direct, student-friendly.
- Do not provide the full answer; ask the next best question.
- Stay aligned to the framing progression: Key Topic -> Is About -> Main Ideas -> Essential Details -> So What.

Current state:
- keyTopic: ${cleanText(s.frame.keyTopic)}
- isAbout: ${cleanText(s.frame.isAbout)}
- mainIdeas: ${(s.frame.mainIdeas || []).map((x) => `• ${cleanText(x)}`).join("\n")}
- soWhat: ${cleanText(s.frame.soWhat)}
${intakeBlock}
`.trim();
}

// ---- HANDLER ----
export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const message = cleanText(body.message || "");
    const intake = body.intake && typeof body.intake === "object" ? body.intake : null;

    // Incoming state (SSOT roundtrip)
    const incoming = normalizeIncomingState(body.state || body.vercelState || body.framing || {});
    let state = incoming;

    // Safety
    const safety = await classifyMessage(message);
    if (safety?.blocked) {
      const reply = SAFETY_RESPONSES[safety.category] || SAFETY_RESPONSES.default;
      return res.status(200).json({ reply: enforceSingleQuestion(reply), state });
    }

    // Update state based on student message
    if (message) {
      state = updateStateFromStudent(state, message);
    }

    // Deterministic next question (SSOT)
    const nextQ = computeNextQuestion(state);
    const reply = enforceSingleQuestion(nextQ);

    return res.status(200).json({ reply, state });
  } catch (err) {
    console.error("Tutor API error:", err);
    return res.status(200).json({
      reply: "Hmm — I had trouble processing that. Can you try again?",
      state: normalizeIncomingState({})
    });
  }
}
