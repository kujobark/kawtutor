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
  return (
    t === "yes" ||
    t === "y" ||
    t === "yeah" ||
    t === "yep" ||
    t === "sure" ||
    t === "correct" ||
    t === "ok" ||
    t === "okay"
  );
}

// Keep reply as a SINGLE question (but preserve helpful guidance like "(yes/no)")
function enforceSingleQuestion(text) {
  let out = (text || "").toString().trim();
  if (!out) return "Can you say more?";

  const firstQ = out.indexOf("?");
  const lastQ = out.lastIndexOf("?");

  // If there are multiple question marks, keep only the first question.
  if (firstQ >= 0 && lastQ !== firstQ) {
    out = out.slice(0, firstQ + 1).trim();
  }

  // If no question mark exists, turn it into a question.
  if (!out.includes("?")) {
    out = out.replace(/[.!\s]*$/, "") + "?";
  }

  return out;
}

const GENERIC_KEY_TOPICS = new Set([
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
  "topic",
  "key topic",
  "it",
  "this",
  "that",
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
      // index-based buckets: details[0] supports mainIdeas[0], etc.
      details: [],
      soWhat: "",
    },
    pending: null,
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

  // Defensive: ensure buckets exist for each main idea
  for (let i = 0; i < base.frame.mainIdeas.length; i++) {
    if (!Array.isArray(base.frame.details[i])) base.frame.details[i] = [];
  }

  return base;
}

function getMainIdeaIndexNeedingDetails(state) {
  const s = state;
  const mis = s.frame.mainIdeas || [];
  for (let i = 0; i < mis.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    // we will collect at least 2; up to 3
    if (arr.length < 2) return i;
  }
  return -1;
}

// ---- PROGRESSION ----
function computeNextQuestion(state) {
  const s = state;

  // 🔒 Is About confirmation checkpoint
  if (s.pending?.type === "confirmIsAbout") {
    return `"${s.frame.keyTopic}" is about "${s.frame.isAbout}". Is that correct, or would you like to revise it?`;
  }

  // 🔒 Main Ideas confirmation checkpoint
  if (s.pending?.type === "confirmMainIdeas") {
    const lines = (s.frame.mainIdeas || [])
      .map((mi, i) => `${i + 1}) ${mi}`)
      .join("\n");
    return `You have identified the following Main Ideas:\n${lines}\nIs that correct, or would you like to revise one?`;
  }

  // 🧠 Offer optional 3rd main idea
  if (s.pending?.type === "offerThirdMainIdea") {
    return "Do you have a third Main Idea? (yes/no)";
  }

  // 🧠 Collect the 3rd main idea
  if (s.pending?.type === "collectThirdMainIdea") {
    return `What is your third Main Idea that helps explain ${s.frame.keyTopic}?`;
  }

  // 🧠 Offer optional 3rd supporting detail for a specific main idea
  if (s.pending?.type === "offerThirdDetail") {
    const i = Number(s.pending.index);
    const mi = s.frame.mainIdeas?.[i] || "";
    return `For this Main Idea: "${mi}", do you have a third Supporting Detail? (yes/no)`;
  }

  // 🧠 Collect optional 3rd supporting detail for a specific main idea
  if (s.pending?.type === "collectThirdDetail") {
    const i = Number(s.pending.index);
    const mi = s.frame.mainIdeas?.[i] || "";
    return `What is your third Supporting Detail for this Main Idea: "${mi}"?`;
  }

  // 🔒 Details confirmation checkpoint for a specific main idea
  if (s.pending?.type === "confirmDetails") {
    const i = Number(s.pending.index);
    const mi = s.frame.mainIdeas?.[i] || "";
    const arr = Array.isArray(s.frame.details?.[i]) ? s.frame.details[i] : [];
    const lines = arr.map((d, k) => `${k + 1}) ${d}`).join("\n");
    return `For this Main Idea: "${mi}", you identified the following Supporting Details:\n${lines}\nIs that correct, or would you like to revise one?`;
  }

  // 🧠 Offer optional extra So What sentence
  if (s.pending?.type === "offerMoreSoWhat") {
    return `Do you want to add one more sentence to your So What? (yes/no)`;
  }

  // 🧠 Collect optional extra So What sentence
  if (s.pending?.type === "collectMoreSoWhat") {
    return `Add one more sentence to your So What:`;
  }

  // 🔒 So What confirmation checkpoint
  if (s.pending?.type === "confirmSoWhat") {
    return `Your So What is: "${s.frame.soWhat}". Is that correct, or would you like to revise it?`;
  }

  // Base progression
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

  // Details per main idea (granular: first/second)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const mi = s.frame.mainIdeas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      return arr.length === 0
        ? `What is your first Supporting Detail for this Main Idea: "${mi}"?`
        : `What is your second Supporting Detail for this Main Idea: "${mi}"?`;
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

  if (!Array.isArray(s.frame.details)) s.frame.details = [];
  if (!Array.isArray(s.frame.mainIdeas)) s.frame.mainIdeas = [];

  // Ensure buckets exist
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    if (!Array.isArray(s.frame.details[i])) s.frame.details[i] = [];
  }

  // 0) Pending handlers first

  // confirmIsAbout
  if (s.pending?.type === "confirmIsAbout") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    // anything else is treated as revised Is About
    s.frame.isAbout = msg;
    s.pending = null;
    return s;
  }

  // confirmMainIdeas
  if (s.pending?.type === "confirmMainIdeas") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    // Hold here for now (revision routing later)
    return s;
  }

  // offerThirdMainIdea
  if (s.pending?.type === "offerThirdMainIdea") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = { type: "collectThirdMainIdea" };
      return s;
    }

    // No → confirm main ideas next
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  // collectThirdMainIdea
  if (s.pending?.type === "collectThirdMainIdea") {
    if (!isNegative(msg)) {
      s.frame.mainIdeas.push(msg);
      if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) {
        s.frame.details[s.frame.mainIdeas.length - 1] = [];
      }
    }
    // After MI #3, confirm main ideas
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  // offerThirdDetail
  if (s.pending?.type === "offerThirdDetail") {
    const normalized = msg.toLowerCase().trim();
    const idx = Number(s.pending.index);

    if (isAffirmative(normalized)) {
      s.pending = { type: "collectThirdDetail", index: idx };
      return s;
    }

    // No → confirm details for this main idea
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  // collectThirdDetail
  if (s.pending?.type === "collectThirdDetail") {
    const idx = Number(s.pending.index);
    if (!Array.isArray(s.frame.details[idx])) s.frame.details[idx] = [];
    if (!isNegative(msg)) {
      s.frame.details[idx] = [...s.frame.details[idx], msg];
    }
    // After collecting third, confirm details for this main idea
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  // confirmDetails
  if (s.pending?.type === "confirmDetails") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    // Hold here for now (revision routing later)
    return s;
  }

  // offerMoreSoWhat
  if (s.pending?.type === "offerMoreSoWhat") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = { type: "collectMoreSoWhat" };
      return s;
    }

    // No → confirm So What next
    s.pending = { type: "confirmSoWhat" };
    return s;
  }

  // collectMoreSoWhat
  if (s.pending?.type === "collectMoreSoWhat") {
    if (!isNegative(msg)) {
      // Append to existing soWhat (keep as one string)
      s.frame.soWhat = cleanText(`${s.frame.soWhat} ${msg}`);
    }
    s.pending = { type: "confirmSoWhat" };
    return s;
  }

  // confirmSoWhat
  if (s.pending?.type === "confirmSoWhat") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    // otherwise treat as revised So What
    s.frame.soWhat = msg;
    s.pending = null;
    return s;
  }

  // 1) Extraction rule: "X is about Y"
  const parsed = parseKeyTopicIsAbout(msg);
  if (parsed) {
    if (!s.frame.keyTopic) s.frame.keyTopic = parsed.keyTopic;
    if (!s.frame.isAbout) s.frame.isAbout = parsed.isAbout;

    s.pending = { type: "confirmIsAbout" };
    return s;
  }

  // 2) Key Topic capture (plain 2–5 words)
  if (!s.frame.keyTopic) {
    const wc = msg.split(/\s+/).filter(Boolean).length;
    if (!isBadKeyTopic(msg) && wc >= 2 && wc <= 5) {
      s.frame.keyTopic = msg;
    }
  }

  if (!s.frame.keyTopic) return s;

  // 3) Is About capture (plain sentence/phrase) + checkpoint
  if (!s.frame.isAbout) {
    const lowered = msg.toLowerCase().trim();
    if (lowered !== "revise" && lowered !== "change") {
      s.frame.isAbout = msg;
      s.pending = { type: "confirmIsAbout" };
    }
    return s;
  }

  // 4) Main Ideas capture (need 2 then offer optional 3rd)
  if (s.frame.mainIdeas.length < 2) {
    if (!isNegative(msg)) {
      s.frame.mainIdeas.push(msg);
      if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) {
        s.frame.details[s.frame.mainIdeas.length - 1] = [];
      }

      if (s.frame.mainIdeas.length === 2) {
        s.pending = { type: "offerThirdMainIdea" };
      }
    }
    return s;
  }

  // 5) Details capture (per main idea: 2 required, optional 3rd + confirm)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      if (!isNegative(msg)) {
        s.frame.details[i] = [...arr, msg];
      }

      // After we just collected the 2nd detail, offer optional 3rd
      const updated = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
      if (updated.length === 2) {
        s.pending = { type: "offerThirdDetail", index: i };
      }

      return s;
    }
  }

  // 6) So What capture + offer optional extra sentence
  if (!s.frame.soWhat) {
    if (!isNegative(msg)) {
      s.frame.soWhat = msg;
      s.pending = { type: "offerMoreSoWhat" };
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
- Stay aligned to the framing progression: Key Topic -> Is About -> Main Ideas -> Supporting Details -> So What.

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
      state: normalizeIncomingState({}),
    });
  }
}
