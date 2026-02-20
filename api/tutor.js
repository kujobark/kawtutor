import OpenAI from "openai";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------------------
// CONFIG
// ---------------------
const DEFAULT_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";

// Transcript cap (avoid bloating state)
const TRANSCRIPT_MAX_TURNS = 200;

// Run language detection only on “real” text
const LANG_DETECT_MIN_CHARS = 18;

// ---------------------
// CORS
// ---------------------
const ALLOWED_ORIGIN = "*";
function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

// ---------------------
// UTIL
// ---------------------
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

// ---------------------
// LANGUAGE HELPERS (LLM)
// ---------------------

// Returns { code, name, nativeName, dir } or null
async function detectLanguageViaLLM(text) {
  const input = cleanText(text);
  if (!input || input.length < LANG_DETECT_MIN_CHARS) return null;

  // Avoid detecting on tiny “yes/no/ok/correct”
  const low = input.toLowerCase();
  if (isAffirmative(low) || isNegative(low) || low === "ok" || low === "okay" || low === "correct") {
    return null;
  }

  const system = `You detect the language of user text.
Return ONLY a compact JSON object with:
{"code":"<ISO-639-1 if possible else 'und'>","name":"<English language name>","nativeName":"<native language name>","dir":"ltr|rtl","confidence":0-1}
If uncertain, use code "und" and confidence < 0.6.`;

  const user = `Text:\n${input}`;

  try {
    const resp = await client.chat.completions.create({
      model: DEFAULT_MODEL,
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    });

    const raw = resp?.choices?.[0]?.message?.content || "";
    const parsed = JSON.parse(raw);

    const code = (parsed.code || "und").toString();
    const name = (parsed.name || "Unknown").toString();
    const nativeName = (parsed.nativeName || name).toString();
    const dir = parsed.dir === "rtl" ? "rtl" : "ltr";
    const confidence = Number(parsed.confidence || 0);

    if (!code || code === "und") return null;
    if (confidence < 0.75) return null;

    return { code, name, nativeName, dir };
  } catch {
    return null;
  }
}

// Used only when we’re asking the language switch question and the student replies in their language.
async function classifyYesNoViaLLM(text) {
  const input = cleanText(text);
  if (!input) return "unknown";

  const system = `Classify the user's response as YES, NO, or UNKNOWN.
Return ONLY one token: YES or NO or UNKNOWN.`;

  try {
    const resp = await client.chat.completions.create({
      model: DEFAULT_MODEL,
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: input },
      ],
    });
    const out = (resp?.choices?.[0]?.message?.content || "").trim().toUpperCase();
    if (out === "YES" || out === "NO") return out.toLowerCase();
    return "unknown";
  } catch {
    return "unknown";
  }
}

async function translateQuestionViaLLM(question, targetLanguageName) {
  const q = enforceSingleQuestion(question);
  const system = `You are a precise translator.
Translate the following into ${targetLanguageName}.
Rules:
- Preserve meaning exactly.
- Keep it as ONE question.
- Preserve parentheses like (yes/no) and quotation marks.
- Output ONLY the translated question.`;

  try {
    const resp = await client.chat.completions.create({
      model: DEFAULT_MODEL,
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: q },
      ],
    });
    const out = resp?.choices?.[0]?.message?.content || q;
    return enforceSingleQuestion(out);
  } catch {
    return q;
  }
}

// ---------------------
// STUCK SUPPORT (SSOT)
// ---------------------

function detectStuckTone(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return "neutral";
  const frustration = ["confus", "stupid", "dumb", "hate", "annoy", "frustrat", "angry", "mad", "ugh", "this sucks", "do we have to"];
  if (frustration.some((p) => t.includes(p))) return "frustration";
  const resistance = ["do we have to", "why do we have to", "why am i doing", "what's the point", "pointless"];
  if (resistance.some((p) => t.includes(p))) return "resistance";
  return "neutral";
}

function normalizePurpose(msg) {
  const t = cleanText(msg).toLowerCase();
  if (!t) return null;
  if (t.includes("study") || t.includes("review") || t === "s") return "study";
  if (t.includes("write") || t.includes("essay") || t.includes("paragraph") || t.includes("create") || t === "w") return "write";
  if (t.includes("read") || t.includes("note") || t.includes("annot") || t === "r") return "read";
  // allow 1/2/3 mapping if you want buttons later
  if (t === "1") return "study";
  if (t === "2") return "write";
  if (t === "3") return "read";
  return null;
}

function normalizeFrameTypeSelection(input) {
  const t = (input || "").toLowerCase().trim();

  // Accept numeric choices
  if (t === "1" || t.startsWith("1 ")) return "causeEffect";
  if (t === "2" || t.startsWith("2 ")) return "themes";
  if (t === "3" || t.startsWith("3 ")) return "reading";

  // Accept common text variants (still deterministic)
  if (t.includes("cause") || t.includes("effect") || t.includes("how") || t.includes("why")) return "causeEffect";
  if (t.includes("theme") || t.includes("big idea") || t.includes("central idea")) return "themes";
  if (t.includes("read") || t.includes("text") || t.includes("source") || t.includes("note")) return "reading";

  return "";
}


function mapPurposeToFrameType(purpose) {
  // Phase 1: deterministic mapping (no teacher override yet)
  if (purpose === "study") return "causeEffect";   // Linear & Cause-and-Effect Relationships
  if (purpose === "write") return "themes";        // Framing Themes
  if (purpose === "read") return "reading";        // Reading Frames
  return "base";
}

function fillTopic(template, keyTopic) {
  return (template || "").replaceAll("[Key Topic]", keyTopic || "your topic");
}

const PROMPT_BANK = {
  // Phase 1 vertical slice: Studying/Review + Linear & Cause-and-Effect
  study: {
    causeEffect: {
      isAbout: 'In your own words, what is happening in "[Key Topic]", and why is it important?',
      mainIdea: "What is one major cause or effect that is important to remember?",
      detail: "What is an important detail that explains how or why this cause or effect occurs?",
      soWhat: 'When you look at all of these causes and effects together, what can you conclude about "[Key Topic]"?'
    },
  },
};

function getPromptForStage(state, stage) {
  const purpose = state.frameMeta?.purpose || "";
  const frameType = state.frameMeta?.frameType || "";
  const kt = state.frame?.keyTopic || "";

  // isAbout
  if (stage === "isAbout") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.isAbout;
    if (tpl) return fillTopic(tpl, kt);
  }

  if (stage === "mainIdeas") {
    const q = PROMPT_BANK?.[purpose]?.[frameType]?.mainIdea;
    if (q) return q;
  }

  if (stage.startsWith("details:")) {
    const q = PROMPT_BANK?.[purpose]?.[frameType]?.detail;
    if (q) return q;
  }

  if (stage === "soWhat") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.soWhat;
    if (tpl) return fillTopic(tpl, kt);
  }

  return null;
}

function buildStuckNudges(state, stage) {
  const purpose = state.frameMeta?.purpose || "";
  const frameType = state.frameMeta?.frameType || "";

  // Phase 1: study + cause/effect nudges
  if (purpose === "study" && frameType === "causeEffect") {
    if (stage === "mainIdeas") {
      return [
        "What caused this to happen",
        "What happened because of it",
        "Is this a cause or an effect",
      ];
    }
    if (stage.startsWith("details:")) {
      return [
        "How does this happen",
        "Why does this happen",
        "What shows the connection between the cause and the effect",
      ];
    }
    if (stage === "soWhat") {
      return [
        "What changed from beginning to end",
        "What connects all of these causes and effects",
        "What do they have in common",
        "What conclusion can you draw",
      ];
    }
  }

  // default fallback nudges (generic, no question marks)
  if (stage === "mainIdeas") return ["Think of one important part", "Think of one reason or cause", "Think of one result or effect"];
  if (stage.startsWith("details:")) return ["Look for one example", "Look for one fact that supports it", "Look for one specific detail"];
  if (stage === "soWhat") return ["What is important here", "Why should someone care", "What does this mean overall"];
  return ["Take a deep breath", "Start rough", "You can revise later"];
}

function formatNudgeText(nudges) {
  const items = (nudges || []).slice(0, 4).map((x) => `- ${cleanText(x)}`).join("\n");
  return items ? `Here are a few quick nudges (pick one):\n${items}` : "";
}


function isStuckMessage(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  // Keep it short so we don’t mis-fire on real answers
  const wc = t.split(/\s+/).filter(Boolean).length;
  if (wc > 14) return false;

  const patterns = [
    "i don't know", "i dont know", "idk", "dont know",
    "i'm confused", "im confused", "confused",
    "i don't get it", "i dont get it", "dont get it",
    "help", "can you help", "stuck", "not sure", "i forgot",
    "i can't", "i cant", "no idea"
  ];

  return patterns.some((p) => t.includes(p));
}

function getStage(state) {
  const f = state.frame;
  const m = state.frameMeta || {};

  if (!m.purpose) return "purpose";
  if (!f.keyTopic) return "keyTopic";
  if (!f.isAbout) return "isAbout";
  if ((f.mainIdeas || []).length < 2) return "mainIdeas";

  // details stage: find first main idea needing details (2 required)
  for (let i = 0; i < (f.mainIdeas || []).length; i++) {
    const arr = Array.isArray(f.details?.[i]) ? f.details[i] : [];
    if (arr.length < 2) return `details:${i}`;
  }

  if (!f.soWhat) return "soWhat";
  return "refine";
}

function buildMiniQuestion(state) {
  const stage = getStage(state);

  if (stage === "keyTopic") {
    return "If you had to title this assignment in 4 words, what would the title be?";
  }

  if (stage === "isAbout") {
    return "In one rough sentence, what is the main point you think you’re supposed to explain?";
  }

  if (stage === "mainIdeas") {
    // Purpose-aware mini question (single question)
    if (state.frameMeta?.purpose === "study" && state.frameMeta?.frameType === "causeEffect") {
      return `What is one major cause or effect related to "${state.frame.keyTopic}"? (Rough is fine.)`;
    }
    return `What is one reason, part, or cause related to "${state.frame.keyTopic}"? (Rough is fine.)`;
  }

  if (stage.startsWith("details:")) {
    const idx = Number(stage.split(":")[1]);
    const mi = state.frame.mainIdeas?.[idx] || "this Main Idea";
    return `What is one specific example from your text/notes that connects to: "${mi}"?`;
  }

  if (stage === "soWhat") {
    if (state.frameMeta?.purpose === "study" && state.frameMeta?.frameType === "causeEffect") {
      return `When you look at the causes and effects together, what can you conclude about "${state.frame.keyTopic}"?`;
    }
    return `Who is affected by "${state.frame.keyTopic}", and why should they care?`;
  }

  // refine
  return "What part feels easiest to improve right now: Key Topic, Is About, Main Ideas, Details, or So What?";
}

function normalizeStuckChoice(msg) {
  const t = cleanText(msg).toLowerCase();
  if (!t) return null;

  // direct numbers
  if (t === "1" || t.startsWith("1 ")) return "1";
  if (t === "2" || t.startsWith("2 ")) return "2";
  if (t === "3" || t.startsWith("3 ")) return "3";
  if (t === "4" || t.startsWith("4 ")) return "4";

  // keyword matches
  if (t.includes("direction") || t.includes("prompt")) return "1";
  if (t.includes("re-read") || t.includes("reread") || t.includes("notes") || t.includes("text")) return "2";
  if (t.includes("smaller") || t.includes("small") || t.includes("mini")) return "3";
  if (t.includes("skip") || t.includes("later") || t.includes("come back")) return "4";

  return null;
}

// ---------------------
// STATE
// ---------------------
function defaultState() {
  return {
    version: 2,
    frameMeta: {
      purpose: "", // study|write|read
      frameType: "", // causeEffect|themes|reading
    },
    frame: {
      keyTopic: "",
      isAbout: "",
      mainIdeas: [],
      // index-based buckets: details[0] supports mainIdeas[0], etc.
      details: [],
      soWhat: "",
    },
    pending: null,
    settings: {
      language: "en",
      languageName: "English",
      languageNativeName: "English",
      dir: "ltr",
      languageLocked: false,
    },
    transcript: [], // [{ role: "Student"|"Kaw", text }]
    exports: null,  // { frameText, transcriptText, html }
    flags: {
      exportOffered: false,
      exportChoice: null, // "frame"|"transcript"|"both"
    },
    skips: [], // optional: [{ stage, at }]
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

  // Frame meta (purpose + frame type)
  const frameMeta = s.frameMeta && typeof s.frameMeta === "object" ? s.frameMeta : {};
  base.frameMeta.purpose = cleanText(frameMeta.purpose || "") || "";
  base.frameMeta.frameType = cleanText(frameMeta.frameType || "") || (base.frameMeta.purpose ? mapPurposeToFrameType(base.frameMeta.purpose) : "");

  base.pending = s.pending && typeof s.pending === "object" ? s.pending : null;

  // Settings
  const settings = s.settings && typeof s.settings === "object" ? s.settings : {};
  base.settings.language = cleanText(settings.language || base.settings.language) || "en";
  base.settings.languageName = cleanText(settings.languageName || base.settings.languageName) || "English";
  base.settings.languageNativeName =
    cleanText(settings.languageNativeName || base.settings.languageNativeName) || base.settings.languageName;
  base.settings.dir = settings.dir === "rtl" ? "rtl" : "ltr";
  base.settings.languageLocked = !!settings.languageLocked;

  // Transcript
  if (Array.isArray(s.transcript)) {
    base.transcript = s.transcript
      .map((t) => ({
        role: cleanText(t?.role || ""),
        text: cleanText(t?.text || ""),
      }))
      .filter((t) => t.role && t.text)
      .slice(-TRANSCRIPT_MAX_TURNS);
  }

  // Exports + flags
  if (s.exports && typeof s.exports === "object") {
    base.exports = s.exports;
  }
  const flags = s.flags && typeof s.flags === "object" ? s.flags : {};
  base.flags.exportOffered = !!flags.exportOffered;
  base.flags.exportChoice = flags.exportChoice || null;

  // Skips
  if (Array.isArray(s.skips)) {
    base.skips = s.skips
      .map((x) => ({ stage: cleanText(x?.stage || ""), at: Number(x?.at || 0) }))
      .filter((x) => x.stage);
  }

  // Defensive: ensure buckets exist for each main idea
  for (let i = 0; i < base.frame.mainIdeas.length; i++) {
    if (!Array.isArray(base.frame.details[i])) base.frame.details[i] = [];
  }

  return base;
}

function isFrameComplete(s) {
  if (!s.frame.keyTopic) return false;
  if (!s.frame.isAbout) return false;
  if (!Array.isArray(s.frame.mainIdeas) || s.frame.mainIdeas.length < 2) return false;

  // require at least 2 details per main idea
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) return false;
  }

  if (!s.frame.soWhat) return false;
  return true;
}

function buildFrameText(s) {
  const lines = [];
  lines.push(`KEY TOPIC: ${s.frame.keyTopic}`);
  lines.push(`IS ABOUT: ${s.frame.isAbout}`);
  lines.push("");
  lines.push("MAIN IDEAS + SUPPORTING DETAILS:");

  s.frame.mainIdeas.forEach((mi, i) => {
    lines.push(`${i + 1}) ${mi}`);
    const details = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    details.forEach((d, k) => {
      lines.push(`   - Detail ${k + 1}: ${d}`);
    });
    lines.push("");
  });

  lines.push(`SO WHAT: ${s.frame.soWhat}`);
  return lines.join("\n").trim();
}

function buildTranscriptText(s) {
  const turns = Array.isArray(s.transcript) ? s.transcript : [];
  return turns.map((t) => `${t.role}: ${t.text}`).join("\n").trim();
}

function escapeHtml(str) {
  return (str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function buildExportHtml(s) {
  const frameText = escapeHtml(buildFrameText(s)).replaceAll("\n", "<br/>");
  const transcriptText = escapeHtml(buildTranscriptText(s)).replaceAll("\n", "<br/>");

  return `<!doctype html>
<html lang="${escapeHtml(s.settings.language || "en")}" dir="${escapeHtml(s.settings.dir || "ltr")}">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Kaw Companion — Session Export</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 24px; line-height: 1.35; }
    h1 { font-size: 20px; margin: 0 0 12px 0; }
    h2 { font-size: 16px; margin: 18px 0 8px 0; }
    .box { border: 1px solid #ddd; padding: 12px; border-radius: 10px; }
    .muted { color: #666; font-size: 12px; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>Kaw Companion — Session Export</h1>

  <h2>Structured Frame</h2>
  <div class="box">${frameText}</div>

  <h2>Full Transcript</h2>
  <div class="box">${transcriptText || "<em>(No transcript captured.)</em>"}</div>

  <div class="muted">Tip: Use your browser’s Print dialog to print or “Save as PDF.”</div>
</body>
</html>`;
}

// ---------------------
// PROGRESSION
// ---------------------
function computeNextQuestion(state) {
  const s = state;

  // 0) Language switch checkpoint
  if (s.pending?.type === "confirmLanguageSwitch") {
    const candNative = s.pending?.candidateNativeName || s.pending?.candidateName || "that language";
    const candName = s.pending?.candidateName || "that language";
    return `I notice you’re writing in ${candName}. Would you like to continue in ${candNative}? (yes/no)`;
  }

  // 🧩 STUCK (Option B): confirm -> menu -> next action
  if (s.pending?.type === "stuckConfirm") {
    return "Sounds like you’re stuck. Want a quick help move? (yes/no)";
  }

  if (s.pending?.type === "stuckMenu") {
    // IMPORTANT: only ONE question mark in this entire string (enforceSingleQuestion truncates after first '?')
    return (
      "Pick a quick help move: " +
      "1) Check directions  " +
      "2) Re-read source/notes  " +
      "3) I’ll ask a smaller question for this step  " +
      "4) Skip for now and come back. " +
      "Which one (1–4)?"
    );
  }

  if (s.pending?.type === "stuckReask") {
    const tip =
      s.pending.mode === "directions"
        ? "Quick reset: re-read the prompt and underline the verb (explain/argue/compare)."
        : "Quick reset: skim your notes/text and pick one phrase that feels important.";
    return `${tip} Then answer: ${s.pending.resumeQuestion}`;
  }

  if (s.pending?.type === "stuckNudge") {
    const tone = s.pending.tone || "neutral";
    const ack =
      tone === "frustration"
        ? "That can feel frustrating. "
        : tone === "resistance"
          ? "I hear you. "
          : "";
    const nudge = cleanText(s.pending.nudgeText || "");
    return `${ack}${nudge} Then answer: ${s.pending.resumeQuestion}`;
  }

  if (s.pending?.type === "stuckMini") {
    return s.pending.miniQuestion || buildMiniQuestion(s);
  }

  if (s.pending?.type === "stuckSkip") {
    return "Got it — we’ll come back to this. Want to try the next step now? (yes/no)";
  }

  // 🔒 Is About confirmation checkpoint
  if (s.pending?.type === "confirmIsAbout") {
    return `"${s.frame.keyTopic}" is about "${s.frame.isAbout}". Is that correct, or would you like to revise it?`;
  }

  // 🔒 Main Ideas confirmation checkpoint
  if (s.pending?.type === "confirmMainIdeas") {
    const lines = (s.frame.mainIdeas || []).map((mi, i) => `${i + 1}) ${mi}`).join("\n");
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

  // 🖨️ Export offer checkpoint (after frame complete)
  if (s.pending?.type === "offerExport") {
    return "Would you like to save or print a copy of your work? (yes/no)";
  }

  // 🖨️ Export choice checkpoint
  if (s.pending?.type === "chooseExportType") {
    return "What would you like to save/print: frame, transcript, or both? (frame/transcript/both)";
  }

  // Base progression
  if (!s.frameMeta?.purpose) {
    return "How will you use this Frame: studying/review, writing/creating, or create notes from a reading or source? (study/write/read)";
  }

  if (!s.frameMeta?.frameType) {
    // IMPORTANT: only ONE question mark in this entire string (enforceSingleQuestion truncates after first '?')
    return (
      "What kind of thinking are you doing: " +
      "1) Explain how/why something happens (Linear & Cause-and-Effect Relationships)  " +
      "2) Explain a big idea or theme (Framing Themes)  " +
      "3) Organize ideas from a text or source (Reading Frames). " +
      "Which one (1–3)?"
    );
  }


  if (!s.frame.keyTopic) {
    return "What is your Key Topic? (2–5 words)";
  }

  if (!s.frame.isAbout) {
    const pb = getPromptForStage(s, "isAbout");
    return pb || `Finish this sentence: "${s.frame.keyTopic} is about ____."`;
  }

  if (s.frame.mainIdeas.length < 2) {
    const pb = getPromptForStage(s, "mainIdeas");
    if (pb) return pb;
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
    const pb = getPromptForStage(s, "soWhat");
    return pb || `So what? Why does "${s.frame.keyTopic}" matter? (1–2 sentences)`;
  }

  return "Want to refine anything (Key Topic, Is About, Main Ideas, Details, or So What)?";
}

// ---------------------
// STATE UPDATE (SSOT)
// ---------------------
function ensureBuckets(s) {
  if (!Array.isArray(s.frame.details)) s.frame.details = [];
  if (!Array.isArray(s.frame.mainIdeas)) s.frame.mainIdeas = [];
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    if (!Array.isArray(s.frame.details[i])) s.frame.details[i] = [];
  }
}

function appendTurn(s, role, text) {
  const t = cleanText(text);
  if (!t) return;
  if (!Array.isArray(s.transcript)) s.transcript = [];
  s.transcript.push({ role, text: t });
  if (s.transcript.length > TRANSCRIPT_MAX_TURNS) {
    s.transcript = s.transcript.slice(-TRANSCRIPT_MAX_TURNS);
  }
}

function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);
  ensureBuckets(s);

  // 0) Purpose capture (before everything else, but after protected pending)
  if (!s.frameMeta) s.frameMeta = { purpose: "", frameType: "" };
  if (!s.frameMeta.purpose && !(s.pending && s.pending.type)) {
    const p = normalizePurpose(msg);
    if (p) {
      s.frameMeta.purpose = p;
      return s;
    }
  }

  
  // 0b) Frame type selection (thinking pattern) — after purpose, before progression
  if (s.frameMeta?.purpose && !s.frameMeta.frameType && !(s.pending && s.pending.type)) {
    const ft = normalizeFrameTypeSelection(msg);
    if (ft) {
      s.frameMeta.frameType = ft;
      return s;
    }
  }

// 0) Pending handlers first

  // confirmLanguageSwitch
  if (s.pending?.type === "confirmLanguageSwitch") {
    const normalized = msg.toLowerCase().trim();

    // Fast path (English yes/no)
    if (isAffirmative(normalized)) {
      s.settings.language = s.pending.candidateCode || "en";
      s.settings.languageName = s.pending.candidateName || s.settings.languageName;
      s.settings.languageNativeName = s.pending.candidateNativeName || s.settings.languageNativeName;
      s.settings.dir = s.pending.candidateDir === "rtl" ? "rtl" : "ltr";
      s.settings.languageLocked = true;
      s.pending = null;
      return s;
    }
    if (isNegative(normalized)) {
      s.settings.language = "en";
      s.settings.languageName = "English";
      s.settings.languageNativeName = "English";
      s.settings.dir = "ltr";
      s.settings.languageLocked = true;
      s.pending = null;
      return s;
    }

    // If they answered in their language, hold (handler will do LLM yes/no in the main handler)
    return s;
  }

  // 🧩 STUCK CONFIRM (Option B) — FIXED (carry forward ONLY)
  if (s.pending?.type === "stuckConfirm") {
    const low = msg.toLowerCase().trim();
    if (isAffirmative(low)) {
      s.pending = {
        type: "stuckMenu",
        stage: s.pending.stage || getStage(s),
        tone: s.pending.tone || "neutral",
        resumeQuestion: s.pending.resumeQuestion, // ✅ carry forward only
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
      };
      return s;
    }
    if (isNegative(low)) {
      s.pending = null;
      return s;
    }
    return s;
  }

  // 🧩 STUCK MENU — FIXED (never recompute resumeQuestion)
  if (s.pending?.type === "stuckMenu") {
    const choice = normalizeStuckChoice(msg);
    if (!choice) return s;

    if (choice === "1") {
      s.pending = {
        type: "stuckReask",
        mode: "directions",
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }

    if (choice === "2") {
      s.pending = {
        type: "stuckReask",
        mode: "reread",
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }

    if (choice === "3") {
      const stage = s.pending.stage || getStage(s);
      const nudges = buildStuckNudges(s, stage);
      const nudgeText = formatNudgeText(nudges);
      s.pending = {
        type: "stuckNudge",
        stage,
        tone: s.pending.tone || "neutral",
        nudgeText,
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }

    if (choice === "4") {
      if (!Array.isArray(s.skips)) s.skips = [];
      s.skips.push({ stage: s.pending.stage || getStage(s), at: Date.now() });

      s.pending = {
        type: "stuckSkip",
        stage: s.pending.stage || getStage(s),
        resumeQuestion: s.pending.resumeQuestion,
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
      };
      return s;
    }

    return s;
  }

  // stuckReask: clear pending and treat this message as the actual answer
  if (s.pending?.type === "stuckReask") {
    s.pending = null;
    // fall through to normal processing
  }

  // 🧩 stuckSkip — FIXED (compiles + no loop + carry forward only)
  if (s.pending?.type === "stuckSkip") {
    const low = msg.toLowerCase().trim();

    if (isAffirmative(low)) {
      // Move them forward with a smaller question
      s.pending = {
        type: "stuckMini",
        stage: s.pending.stage || getStage(s),
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }

    if (isNegative(low)) {
      // Go back to confirm
      s.pending = {
        type: "stuckConfirm",
        stage: s.pending.stage || getStage(s),
        resumeQuestion: s.pending.resumeQuestion,
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
      };
      return s;
    }

    return s;
  }

  // stuckMini: apply answer into the current stage if possible, then resume
  if (s.pending?.type === "stuckMini") {
    const stage = s.pending.stage || getStage(s);

    if (stage === "keyTopic") {
      const wc = msg.split(/\s+/).filter(Boolean).length;
      if (!s.frame.keyTopic && !isBadKeyTopic(msg) && wc >= 2 && wc <= 5) {
        s.frame.keyTopic = msg;
      }
      s.pending = null;
      return s;
    }

    if (stage === "isAbout") {
      if (!s.frame.isAbout) {
        s.frame.isAbout = msg;
        s.pending = { type: "confirmIsAbout" };
        return s;
      }
      s.pending = null;
      return s;
    }

    if (stage === "mainIdeas") {
      if (s.frame.mainIdeas.length < 2 && !isNegative(msg)) {
        s.frame.mainIdeas.push(msg);
        if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) {
          s.frame.details[s.frame.mainIdeas.length - 1] = [];
        }
        if (s.frame.mainIdeas.length === 2) {
          s.pending = { type: "offerThirdMainIdea" };
          return s;
        }
      }
      s.pending = null;
      return s;
    }

    if (stage.startsWith("details:")) {
      const idx = Number(stage.split(":")[1]);
      const arr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];
      if (arr.length < 2 && !isNegative(msg)) {
        s.frame.details[idx] = [...arr, msg];
        if (s.frame.details[idx].length === 2) {
          s.pending = { type: "offerThirdDetail", index: idx };
          return s;
        }
      }
      s.pending = null;
      return s;
    }

    if (stage === "soWhat") {
      if (!s.frame.soWhat && !isNegative(msg)) {
        s.frame.soWhat = msg;
        s.pending = { type: "offerMoreSoWhat" };
        return s;
      }
      s.pending = null;
      return s;
    }

    // refine: just clear and resume
    s.pending = null;
    return s;
  }

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

      // After So What is confirmed, offer export once (printing/save)
      if (isFrameComplete(s) && !s.flags.exportOffered) {
        s.flags.exportOffered = true;
        s.pending = { type: "offerExport" };
      }
      return s;
    }

    // otherwise treat as revised So What
    s.frame.soWhat = msg;
    s.pending = null;
    return s;
  }

  // offerExport
  if (s.pending?.type === "offerExport") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = { type: "chooseExportType" };
      return s;
    }

    // No → proceed to refine prompt
    s.pending = null;
    return s;
  }

  // chooseExportType
  if (s.pending?.type === "chooseExportType") {
    const normalized = msg.toLowerCase().trim();
    const choice =
      normalized.includes("both") ? "both" :
      normalized.includes("frame") ? "frame" :
      normalized.includes("transcript") ? "transcript" :
      null;

    s.flags.exportChoice = choice || "both";
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

// ---------------------
// HANDLER
// ---------------------
export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const message = cleanText(body.message || "");

    // Incoming SSOT state
    let state = normalizeIncomingState(body.state || body.vercelState || body.framing || {});

    // Safety
    if (message) {
      const safety = await classifyMessage(message);
      if (safety?.blocked) {
        const reply = SAFETY_RESPONSES[safety.category] || SAFETY_RESPONSES.default;
        const out = enforceSingleQuestion(reply);

        // transcript
        appendTurn(state, "Student", message);
        appendTurn(state, "Kaw", out);

        return res.status(200).json({ reply: out, state });
      }
    }

    // 1) If language not locked and no language-switch pending, detect and ask
    if (message && !state.settings.languageLocked && state.pending?.type !== "confirmLanguageSwitch") {
      const detected = await detectLanguageViaLLM(message);
      if (detected && detected.code && detected.code !== "en") {
        // Set pending language switch
        state.pending = {
          type: "confirmLanguageSwitch",
          candidateCode: detected.code,
          candidateName: detected.name,
          candidateNativeName: detected.nativeName,
          candidateDir: detected.dir,
        };

        const q = computeNextQuestion(state);
        const reply = enforceSingleQuestion(q);

        appendTurn(state, "Student", message);
        appendTurn(state, "Kaw", reply);

        return res.status(200).json({ reply, state });
      }
    }

    // 2) If we are in confirmLanguageSwitch and they replied (maybe in their language), classify yes/no
    if (state.pending?.type === "confirmLanguageSwitch" && message) {
      const low = message.toLowerCase().trim();
      let proceedState = state;

      if (!isAffirmative(low) && !isNegative(low)) {
        const yn = await classifyYesNoViaLLM(message);
        if (yn === "yes") {
          proceedState = updateStateFromStudent(state, "yes");
        } else if (yn === "no") {
          proceedState = updateStateFromStudent(state, "no");
        } else {
          // Ask again (single question)
          const q = computeNextQuestion(state);
          let reply = enforceSingleQuestion(q);

          // If candidate language is known, translate the question so they understand
          const candName = state.pending?.candidateName || "English";
          if ((state.pending?.candidateCode || "") !== "en") {
            reply = await translateQuestionViaLLM(reply, candName);
          }

          appendTurn(state, "Student", message);
          appendTurn(state, "Kaw", reply);

          return res.status(200).json({ reply, state });
        }
      } else {
        proceedState = updateStateFromStudent(state, message);
      }

      state = proceedState;
    } else if (message) {
      // 🧩 STUCK DETECTOR (global interrupt) — do not interrupt language switch flow
      const pendingType = state.pending?.type || null;
      const inProtectedPending =
        pendingType === "confirmLanguageSwitch" ||
        pendingType === "stuckConfirm" ||
        pendingType === "stuckMenu" ||
        pendingType === "stuckReask" ||
        pendingType === "stuckMini" ||
        pendingType === "stuckSkip";

      if (!inProtectedPending && isStuckMessage(message)) {
        const stage = getStage(state);
        const resumeQuestion = enforceSingleQuestion(computeNextQuestion(state));

        // Option B: first confirm (store resumeQuestion ONCE)
        state.pending = {
          type: "stuckConfirm",
          stage,
          tone: detectStuckTone(message),
          resumeQuestion,
          miniQuestion: buildMiniQuestion(state),
        };

        let reply = enforceSingleQuestion(computeNextQuestion(state));

        // Respect language if locked
        if (state.settings.languageLocked && state.settings.language !== "en") {
          reply = await translateQuestionViaLLM(
            reply,
            state.settings.languageName || "the target language"
          );
        }

        appendTurn(state, "Student", message);
        appendTurn(state, "Kaw", reply);

        // exports
        if (isFrameComplete(state)) {
          const frameText = buildFrameText(state);
          const transcriptText = buildTranscriptText(state);
          const html = buildExportHtml(state);
          state.exports = { frameText, transcriptText, html };
        } else {
          state.exports = null;
        }

        return res.status(200).json({ reply, state });
      }

      // Normal state update
      state = updateStateFromStudent(state, message);
    }

    // 3) Compute deterministic next question
    let nextQ = computeNextQuestion(state);
    let reply = enforceSingleQuestion(nextQ);

    // 4) If language locked and not English, translate the reply question
    if (state.settings.languageLocked && state.settings.language !== "en") {
      reply = await translateQuestionViaLLM(reply, state.settings.languageName || "the target language");
    }

    // 5) Transcript append
    if (message) appendTurn(state, "Student", message);
    appendTurn(state, "Kaw", reply);

    // 6) Build exports when frame complete (always available for UI buttons)
    if (isFrameComplete(state)) {
      const frameText = buildFrameText(state);
      const transcriptText = buildTranscriptText(state);
      const html = buildExportHtml(state);

      state.exports = { frameText, transcriptText, html };
    } else {
      state.exports = null;
    }

    return res.status(200).json({ reply, state });
  } catch (err) {
    console.error("Tutor API error:", err);
    return res.status(200).json({
      reply: "Hmm — I had trouble processing that. Can you try again?",
      state: normalizeIncomingState({}),
    });
  }
}
