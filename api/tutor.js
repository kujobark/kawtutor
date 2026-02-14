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
 * Kaw Companion — Vercel SSOT (Single Source of Truth)
 * Vercel owns: state + progression + routing
 * Wix owns: UI only
 */

const MAX_CHARS = 420;

function cleanText(s) {
  return String(s || "").replace(/\s+/g, " ").trim();
}

function clampQuestion(q) {
  let out = cleanText(q);
  if (out.length > MAX_CHARS) out = out.slice(0, MAX_CHARS).trim();
  if (!out.endsWith("?")) out = out.replace(/[.!\s]*$/, "") + "?";
  const lastQ = out.lastIndexOf("?");
  if (lastQ !== out.length - 1) out = out.slice(0, lastQ + 1).trim();
  return out;
}

const GENERIC_KEY_TOPICS = new Set([
  "my assignment", "the assignment", "my essay", "this essay", "my paper", "this paper",
  "my paragraph", "this paragraph", "my topic", "the topic", "topic", "key topic", "it", "this", "that",
]);

function isBadKeyTopic(keyTopic) {
  const kt = cleanText(keyTopic).toLowerCase();
  if (!kt) return true;
  if (GENERIC_KEY_TOPICS.has(kt)) return true;
  if (kt.startsWith("my ")) return true;
  return false;
}

function parseKeyTopicIsAbout(text) {
  const t = cleanText(text);
  const m = t.match(/^(.*?)\s+is\s+about\s+(.+?)(?:[.?!]|$)/i);
  if (!m) return null;

  const left = cleanText(m[1]);
  const right = cleanText(m[2]);

  if (!left || !right) return null;
  if (isBadKeyTopic(left)) return null;

  const wc = left.split(" ").filter(Boolean).length;
  if (wc < 1 || wc > 10) return null;

  return { keyTopic: left, isAbout: right };
}

function isNegative(text) {
  const t = cleanText(text).toLowerCase();
  return (
    t === "no" ||
    t === "nope" ||
    t === "nah" ||
    t.includes("not really") ||
    t.includes("i don't think so") ||
    t.includes("dont think so") ||
    t.includes("i can't") ||
    t.includes("cant") ||
    t.includes("that's it") ||
    t.includes("thats it") ||
    t.includes("i'm done") ||
    t.includes("im done") ||
    t.includes("no more")
  );
}

function isAffirmative(text) {
  const t = cleanText(text).toLowerCase();
  return (
    t === "yes" ||
    t === "y" ||
    t === "yeah" ||
    t === "yep" ||
    t === "correct" ||
    t === "that's right" ||
    t === "thats right" ||
    t === "right" ||
    t === "looks good" ||
    t === "good"
  );
}

// ------------------------------
// Draft Is About from intake (fidelity-first but low friction)
// ------------------------------
function deriveIsAboutCandidate(intakeAbout, keyTopic) {
  const raw = cleanText(intakeAbout);
  if (!raw) return "";

  // If intake already has "X is about Y", extract Y
  const parsed = parseKeyTopicIsAbout(raw);
  if (parsed?.isAbout) return cleanText(parsed.isAbout);

  // Otherwise, use the intake sentence as the draft (trim punctuation)
  let cand = raw.replace(/[.?!]\s*$/, "").trim();

  // If they started with key topic, lightly remove it
  const kt = cleanText(keyTopic);
  if (kt) {
    const re = new RegExp("^" + kt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\s*[:\\-–—]?\\s*", "i");
    cand = cand.replace(re, "").trim();
  }

  return cand;
}

// ------------------------------
// Normalize incoming state
// ------------------------------
function normalizeIncomingState(body) {
  const incoming = body?.state ?? body?.framing ?? {};

  const keyTopic = cleanText(incoming?.frame?.keyTopic ?? incoming?.keyTopic ?? incoming?.key_topic ?? "");
  const isAbout = cleanText(incoming?.frame?.isAbout ?? incoming?.isAbout ?? incoming?.is_about ?? "");
  const mainIdeasRaw = incoming?.frame?.mainIdeas ?? incoming?.mainIdeas ?? incoming?.main_ideas ?? [];
  const mainIdeas = Array.isArray(mainIdeasRaw)
    ? mainIdeasRaw.map(cleanText).filter(Boolean)
    : [];
  const details = (incoming?.frame?.details ?? incoming?.details) && typeof (incoming?.frame?.details ?? incoming?.details) === "object"
    ? (incoming?.frame?.details ?? incoming?.details)
    : {};
  const detailsIndex = Number(incoming?.detailsIndex ?? incoming?.details_index ?? 0) || 0;
  const soWhat = cleanText(incoming?.frame?.soWhat ?? incoming?.soWhat ?? incoming?.so_what ?? "");

  const askedForThirdMainIdea = Boolean(incoming?.askedForThirdMainIdea ?? false);

  // pending confirm objects (for stop-check logic)
  const pending = incoming?.pending && typeof incoming.pending === "object" ? incoming.pending : null;

  return {
    frame: {
      keyTopic: isBadKeyTopic(keyTopic) ? "" : keyTopic,
      isAbout,
      mainIdeas,
      details,
      soWhat
    },
    detailsIndex: detailsIndex < 0 ? 0 : detailsIndex,
    askedForThirdMainIdea,
    pending
  };
}

// ------------------------------
// Vercel-owned state update
// ------------------------------
function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);

  // 0) Handle pending confirm steps first
  if (s.pending?.type === "confirmIsAbout" && !s.frame.isAbout) {
    if (isAffirmative(msg)) {
      s.frame.isAbout = cleanText(s.pending.candidate || "");
    } else {
      // anything else becomes the revised Is About (unless they just typed "revise")
      const lowered = msg.toLowerCase();
      if (lowered === "revise" || lowered === "change") {
        // keep pending; they didn't give a revision yet
        return s;
      }
      s.frame.isAbout = msg;
    }
    s.pending = null;
    return s;
  }

  // Extraction rule (server-side)
  const parsed = parseKeyTopicIsAbout(msg);
  if (parsed) {
    if (!s.frame.keyTopic) s.frame.keyTopic = parsed.keyTopic;
    if (!s.frame.isAbout) s.frame.isAbout = parsed.isAbout;
  }

  // If key topic missing, nothing else to store
  if (!s.frame.keyTopic) return s;

  // If isAbout missing, nothing else to store yet
  if (!s.frame.isAbout) return s;

  // Main Ideas collection (need 2, allow 3)
  if (s.frame.mainIdeas.length < 2) {
    if (!isNegative(msg)) {
      s.frame.mainIdeas = [...s.frame.mainIdeas, msg].slice(0, 3);
    }
    return s;
  }

  // Optional 3rd main idea if we asked once
  if (s.frame.mainIdeas.length === 2 && s.askedForThirdMainIdea) {
    if (!isNegative(msg)) {
      s.frame.mainIdeas = [...s.frame.mainIdeas, msg].slice(0, 3);
    }
    return s;
  }

  // Essential Details collection
  const idx = Math.max(0, Math.min(Number(s.detailsIndex || 0), s.frame.mainIdeas.length - 1));
  const key = String(idx);
  const bucket = Array.isArray(s.frame.details?.[key]) ? s.frame.details[key] : [];

  // If student says "no" while giving details: advance to next main idea
  if (isNegative(msg)) {
    const nextIdx = idx + 1;
    if (s.frame.mainIdeas[nextIdx]) s.detailsIndex = nextIdx;
    return s;
  }

  // Store detail (cap at 3)
  const nextBucket = [...bucket, msg].filter(Boolean).slice(0, 3);
  s.frame.details = { ...(s.frame.details || {}), [key]: nextBucket };

  // Auto-advance only after 3 details
  if (nextBucket.length >= 3) {
    const nextIdx = idx + 1;
    if (s.frame.mainIdeas[nextIdx]) s.detailsIndex = nextIdx;
  }

  return s;
}

// ------------------------------
// Deterministic next question (progression)
// ------------------------------
function computeNextQuestion(state) {
  const keyTopic = cleanText(state.frame.keyTopic);
  const isAbout = cleanText(state.frame.isAbout);
  const mainIdeas = Array.isArray(state.frame.mainIdeas) ? state.frame.mainIdeas.map(cleanText).filter(Boolean) : [];
  const details = state.frame.details || {};
  const soWhat = cleanText(state.frame.soWhat);

  // 1) Key Topic
  if (!keyTopic) return "What is your Key Topic? (2–5 words)";

  // 2) Is About (STOP-CHECK confirmation if pending)
  if (!isAbout) {
    if (state.pending?.type === "confirmIsAbout" && state.pending?.candidate) {
      return `To clarify, is your Is About: "${cleanText(state.pending.candidate)}"? (yes / revise)`;
    }
    return `Finish this sentence: "${keyTopic} is about ____."`;
  }

  // 3) Main Ideas
  if (mainIdeas.length < 2) {
    return mainIdeas.length === 0
      ? `What is one Main Idea that helps explain the ${keyTopic}?`
      : `What is another Main Idea that helps explain the ${keyTopic}?`;
  }

  // Determine detail counts for first two ideas (min 2 each)
  const c0 = Array.isArray(details["0"]) ? details["0"].filter(Boolean).length : 0;
  const c1 = Array.isArray(details["1"]) ? details["1"].filter(Boolean).length : 0;
  const firstTwoComplete = c0 >= 2 && c1 >= 2;

  // Ask ONCE for optional 3rd main idea ONLY after the first two ideas have 2+ details
  if (mainIdeas.length === 2 && firstTwoComplete && !state.askedForThirdMainIdea) {
    return `Do you have one more Main Idea that helps explain the ${keyTopic}?`;
  }

  // Details phase: ask 2–3 per main idea, then So What
  const idx = Math.max(0, Math.min(Number(state.detailsIndex || 0), mainIdeas.length - 1));
  const currentIdea = mainIdeas[idx];
  const bucket = Array.isArray(details[String(idx)]) ? details[String(idx)] : [];
  const count = bucket.filter(Boolean).length;

  // If all ideas have at least 2 details, go So What
  const allIdeasHave2 = mainIdeas.every((_, i) => {
    const b = Array.isArray(details[String(i)]) ? details[String(i)] : [];
    return b.filter(Boolean).length >= 2;
  });
  if (allIdeasHave2) {
    if (!soWhat) return `So what? Why does the ${keyTopic} matter?`;
    return `What is one way you could make your So What for the ${keyTopic} more specific?`;
  }

  // Otherwise, collect details for current idea
  if (count < 2) {
    return `What is one Essential Detail that supports the Main Idea "${currentIdea}" about the ${keyTopic}?`;
  }

  // If they have 2 details, offer a 3rd or "no" to move on
  if (count === 2) {
    return `What is one more Essential Detail that supports the Main Idea "${currentIdea}" about the ${keyTopic}? (or say "no")`;
  }

  // count >= 3: move to next idea’s details
  const nextIdx = Math.min(idx + 1, mainIdeas.length - 1);
  const nextIdea = mainIdeas[nextIdx];
  return `What is one Essential Detail that supports the Main Idea "${nextIdea}" about the ${keyTopic}?`;
}

// -----------------------------------------------
// Handler
// -----------------------------------------------
export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method Not Allowed" });

  try {
    const { message = "", intake = {} } = req.body || {};
    const trimmed = cleanText(message);
    if (!trimmed) return res.status(400).json({ error: "Missing 'message' in request body" });

    // ---- Safety check ----
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      const safeReply =
        SAFETY_RESPONSES[safety.category] ||
        "Let’s zoom back to your Frame. What is your Key Topic? (2–5 words.)";
      return res.status(200).json({
        reply: clampQuestion(safeReply),
        state: normalizeIncomingState(req.body || {})
      });
    }

    // ---- SSOT: normalize → update → (maybe set pending) → route ----
    let state = normalizeIncomingState(req.body || {});
    state = updateStateFromStudent(state, trimmed);

    // If Is About is still missing, create a draft from intake and ask confirm/revise
    if (!state.frame.isAbout && !state.pending) {
      const candidate = deriveIsAboutCandidate(intake?.about, state.frame.keyTopic);
      if (candidate) {
        state.pending = { type: "confirmIsAbout", candidate };
      }
    }

    // If we are about to ask for optional 3rd main idea, set the one-time flag now
    const nextQ = computeNextQuestion(state);
    if (nextQ.toLowerCase().startsWith("do you have one more main idea") && !state.askedForThirdMainIdea) {
      state.askedForThirdMainIdea = true;
    }

    return res.status(200).json({
      reply: clampQuestion(nextQ),
      state
    });
  } catch (err) {
    console.error("Tutor handler error:", err);
    return res.status(500).json({ error: "Internal Server Error" });
  }
}
