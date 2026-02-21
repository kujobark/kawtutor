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

// Heuristic: looks like evidence (stats/attribution) rather than a broad cause
function looksLikeEvidence(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  // Numbers / percents / ranges
  if (/[0-9]/.test(t)) return true;
  if (t.includes("%")) return true;

  // Attribution / research-y signals
  const evidenceSignals = [
    "according to",
    "research",
    "study",
    "studies",
    "data",
    "statistics",
    "statistic",
    "survey",
    "report",
    "evidence",
    "shows that",
    "found that",
  ];
  if (evidenceSignals.some((p) => t.includes(p))) return true;

  // Common quantitative units (keep light)
  const units = ["hours", "hour", "years", "year", "minutes", "minute", "days", "day"];
  if (units.some((u) => t.includes(u))) return true;

  return false;
}


// Parse which numbered item the student wants to revise (1..max).
// Accepts: "2", "revise 2", "revise #2", "change 1", "main idea 1", "detail 2 is wrong", "second one", etc.
function parseRevisionIndex(text, max) {
  const t = cleanText(text).toLowerCase();
  if (!t || !max || max < 1) return null;

  // Ordinal words
  const ordMap = { first: 1, 1st: 1, second: 2, 2nd: 2, third: 3, 3rd: 3, fourth: 4, 4th: 4 };
  for (const [k, v] of Object.entries(ordMap)) {
    if (t.includes(k) && v <= max) return v;
  }

  // Any standalone digits
  const nums = t.match(/\b\d+\b/g) || [];
  for (const n of nums) {
    const v = Number(n);
    if (Number.isFinite(v) && v >= 1 && v <= max) return v;
  }

  return null;
}

function indicatesRevisionIntent(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;
  const signals = ["revise", "change", "fix", "replace", "edit", "not a main idea", "not a detail", "wrong", "isn't", "isnt"];
  return signals.some((p) => t.includes(p));
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

// Deterministic parser for Cause & Effect Writing:
// Accept both "lead to" and "leads to"
function parseCauseEffectLeadsTo(isAboutRaw) {
  const t0 = cleanText(isAboutRaw);
  if (!t0) return null;

  let t = t0;
  const low0 = t.toLowerCase();

  const prefix = "this topic is about how ";
  if (low0.startsWith(prefix)) {
    t = cleanText(t.slice(prefix.length));
  }

  // Support both phrases
  const low = t.toLowerCase();
  let idx = low.indexOf(" leads to ");
  let phrase = " leads to ";
  if (idx < 0) {
    idx = low.indexOf(" lead to ");
    phrase = " lead to ";
  }
  if (idx < 0) return null;

  const cause = cleanText(t.slice(0, idx));
  const effect = cleanText(t.slice(idx + phrase.length));
  if (!cause || !effect) return null;

  return { cause, effect };
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

  // Accept common text variants
  if (t.includes("cause") || t.includes("effect") || t.includes("how") || t.includes("why")) return "causeEffect";
  if (t.includes("theme") || t.includes("big idea") || t.includes("central idea")) return "themes";
  if (t.includes("read") || t.includes("text") || t.includes("source") || t.includes("note")) return "reading";

  return "";
}

function fillTopic(template, keyTopic) {
  return (template || "").replaceAll("[Key Topic]", keyTopic || "your topic");
}

function safeEffectFromState(state) {
  const eff = cleanText(state?.frameMeta?.causeEffect?.effect || "");
  return eff || "the effect";
}

const PROMPT_BANK = {
  // Studying/Review + Linear & Cause-and-Effect
  study: {
    causeEffect: {
      isAbout: 'In your own words, what is happening in "[Key Topic]", and why is it important?',
      mainIdea: "What is one major cause or effect that is important to remember?",
      detail: "What is an important detail that explains how or why this cause or effect occurs?",
      soWhat: 'When you look at all of these causes and effects together, what can you conclude about "[Key Topic]"?'
    },
  },

  // Writing/Creating + Cause-and-Effect (KU frame preserved)
  write: {
    causeEffect: {
      isAbout: "Complete this sentence: This topic is about how ______ leads to ______.",
      mainIdeaFirst: "What is the first major cause that leads to [EFFECT]?",
      mainIdeaSecond: "What is the second major cause?",
      detailExplain: "How does [MAIN IDEA] lead to [EFFECT]?",
      detailEvidence: "What evidence or example will you use to support the idea that [MAIN IDEA] leads to [EFFECT]?",
      soWhat: "Why should people care about this effect?"
    },
  },
};

function getPromptForStage(state, stage) {
  const purpose = state.frameMeta?.purpose || "";
  const frameType = state.frameMeta?.frameType || "";
  const kt = state.frame?.keyTopic || "";

  if (stage === "isAbout") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.isAbout;
    if (tpl) return fillTopic(tpl, kt);
  }

  if (stage === "mainIdeas") {
    const ce = PROMPT_BANK?.[purpose]?.[frameType];
    if (purpose === "write" && frameType === "causeEffect") return ce?.mainIdeaFirst || null;
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

function isStuckMessage(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

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

// ---------------------
// STATE
// ---------------------
function defaultState() {
  return {
    version: 2,
    frameMeta: {
      purpose: "",   // study|write|read
      frameType: "", // causeEffect|themes|reading
      causeEffect: { cause: "", effect: "" }, // for writing specificity
    },
    frame: {
      keyTopic: "",
      isAbout: "",
      mainIdeas: [],
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
    transcript: [],
    exports: null,
    flags: { exportOffered: false, exportChoice: null },
    skips: [],
  };
}

function normalizeIncomingState(raw) {
  const s = raw && typeof raw === "object" ? raw : {};
  const base = defaultState();

  const frame = s.frame && typeof s.frame === "object" ? s.frame : {};
  base.frame.keyTopic = cleanText(frame.keyTopic || s.keyTopic || "");
  base.frame.isAbout = cleanText(frame.isAbout || s.isAbout || "");
  base.frame.mainIdeas = Array.isArray(frame.mainIdeas) ? frame.mainIdeas.map(cleanText).filter(Boolean) : [];

  if (Array.isArray(frame.details)) {
    base.frame.details = frame.details.map((bucket) =>
      Array.isArray(bucket) ? bucket.map(cleanText).filter(Boolean) : []
    );
  } else {
    base.frame.details = [];
  }

  base.frame.soWhat = cleanText(frame.soWhat || s.soWhat || "");

  const frameMeta = s.frameMeta && typeof s.frameMeta === "object" ? s.frameMeta : {};
  base.frameMeta.purpose = cleanText(frameMeta.purpose || "") || "";
  base.frameMeta.frameType = cleanText(frameMeta.frameType || "") || "";

  const ce = frameMeta.causeEffect && typeof frameMeta.causeEffect === "object" ? frameMeta.causeEffect : {};
  base.frameMeta.causeEffect.cause = cleanText(ce.cause || "");
  base.frameMeta.causeEffect.effect = cleanText(ce.effect || "");

  base.pending = s.pending && typeof s.pending === "object" ? s.pending : null;

  const settings = s.settings && typeof s.settings === "object" ? s.settings : {};
  base.settings.language = cleanText(settings.language || base.settings.language) || "en";
  base.settings.languageName = cleanText(settings.languageName || base.settings.languageName) || "English";
  base.settings.languageNativeName = cleanText(settings.languageNativeName || base.settings.languageNativeName) || base.settings.languageName;
  base.settings.dir = settings.dir === "rtl" ? "rtl" : "ltr";
  base.settings.languageLocked = !!settings.languageLocked;

  if (Array.isArray(s.transcript)) {
    base.transcript = s.transcript
      .map((t) => ({ role: cleanText(t?.role || ""), text: cleanText(t?.text || "") }))
      .filter((t) => t.role && t.text)
      .slice(-TRANSCRIPT_MAX_TURNS);
  }

  if (s.exports && typeof s.exports === "object") base.exports = s.exports;

  const flags = s.flags && typeof s.flags === "object" ? s.flags : {};
  base.flags.exportOffered = !!flags.exportOffered;
  base.flags.exportChoice = flags.exportChoice || null;

  if (Array.isArray(s.skips)) {
    base.skips = s.skips
      .map((x) => ({ stage: cleanText(x?.stage || ""), at: Number(x?.at || 0) }))
      .filter((x) => x.stage);
  }

  for (let i = 0; i < base.frame.mainIdeas.length; i++) {
    if (!Array.isArray(base.frame.details[i])) base.frame.details[i] = [];
  }

  return base;
}

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
  if (s.transcript.length > TRANSCRIPT_MAX_TURNS) s.transcript = s.transcript.slice(-TRANSCRIPT_MAX_TURNS);
}

function isFrameComplete(s) {
  if (!s.frame.keyTopic) return false;
  if (!s.frame.isAbout) return false;
  if (!Array.isArray(s.frame.mainIdeas) || s.frame.mainIdeas.length < 2) return false;

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
  const eff = cleanText(s.frameMeta?.causeEffect?.effect || "");
  if (eff) lines.push(`EFFECT: ${eff}`);
  lines.push("");
  lines.push("MAIN IDEAS + SUPPORTING DETAILS:");

  s.frame.mainIdeas.forEach((mi, i) => {
    lines.push(`${i + 1}) ${mi}`);
    const details = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    details.forEach((d, k) => lines.push(`   - Detail ${k + 1}: ${d}`));
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

function getStage(state) {
  const f = state.frame;
  const m = state.frameMeta || {};

  if (!m.purpose) return "purpose";
  if (!m.frameType) return "frameType";
  if (!f.keyTopic) return "keyTopic";
  if (!f.isAbout) return "isAbout";
  if ((f.mainIdeas || []).length < 2) return "mainIdeas";

  for (let i = 0; i < (f.mainIdeas || []).length; i++) {
    const arr = Array.isArray(f.details?.[i]) ? f.details[i] : [];
    if (arr.length < 2) return `details:${i}`;
  }

  if (!f.soWhat) return "soWhat";
  return "refine";
}

// ---------------------
// QUESTION ROUTER
// ---------------------
function computeNextQuestion(state) {
  const s = state;

  // ---- Purpose-specific pending checks (CauseEffect Writing) ----
  if (s.pending?.type === "ceMainIdeaEvidenceCheck") {
    return "Is that a major cause, or is it evidence that supports a cause? (cause/evidence)";
  }
  if (s.pending?.type === "ceCollectBroaderCause") {
    return "What is the broader cause that this evidence supports?";
  }

  // Revision workflows (deterministic)
  if (s.pending?.type === "clarifyMainIdeaToRevise") {
    const max = Array.isArray(s.frame.mainIdeas) ? s.frame.mainIdeas.length : 2;
    return `Which Main Idea do you want to revise (1–${max})?`;
  }
  if (s.pending?.type === "collectMainIdeaRevision") {
    const n = Number(s.pending.index) + 1;
    return `What should Main Idea ${n} be instead?`;
  }

  if (s.pending?.type === "clarifyDetailToRevise") {
    const i = Number(s.pending.index);
    const arr = Array.isArray(s.frame.details?.[i]) ? s.frame.details[i] : [];
    const max = Math.max(arr.length, 2);
    return `Which Supporting Detail do you want to revise (1–${max})?`;
  }
  if (s.pending?.type === "collectDetailRevision") {
    const n = Number(s.pending.detailIndex) + 1;
    return `What should Supporting Detail ${n} be instead?`;
  }

  // Language switch pending
  if (s.pending?.type === "confirmLanguageSwitch") {
    const candNative = s.pending?.candidateNativeName || s.pending?.candidateName || "that language";
    const candName = s.pending?.candidateName || "that language";
    return `I notice you’re writing in ${candName}. Would you like to continue in ${candNative}? (yes/no)`;
  }

  // Export pending
  if (s.pending?.type === "offerExport") return "Would you like to save or print a copy of your work? (yes/no)";
  if (s.pending?.type === "chooseExportType") return "What would you like to save/print: frame, transcript, or both? (frame/transcript/both)";

  // Confirmation pending (kept from your earlier flow)
  if (s.pending?.type === "confirmIsAbout") {
    return `"${s.frame.keyTopic}" is about "${s.frame.isAbout}". Is that correct, or would you like to revise it?`;
  }
  if (s.pending?.type === "offerThirdMainIdea") return "Do you have a third Main Idea? (yes/no)";
  if (s.pending?.type === "collectThirdMainIdea") return `What is your third Main Idea that helps explain ${s.frame.keyTopic}?`;
  if (s.pending?.type === "confirmMainIdeas") {
    const lines = (s.frame.mainIdeas || []).map((mi, i) => `${i + 1}) ${mi}`).join("\n");
    return `You have identified the following Main Ideas:\n${lines}\nIs that correct, or would you like to revise one?`;
  }
  if (s.pending?.type === "offerThirdDetail") {
    const i = Number(s.pending.index);
    const mi = s.frame.mainIdeas?.[i] || "";
    return `For this Main Idea: "${mi}", do you have a third Supporting Detail? (yes/no)`;
  }
  if (s.pending?.type === "collectThirdDetail") {
    const i = Number(s.pending.index);
    const mi = s.frame.mainIdeas?.[i] || "";
    return `What is your third Supporting Detail for this Main Idea: "${mi}"?`;
  }
  if (s.pending?.type === "confirmDetails") {
    const i = Number(s.pending.index);
    const mi = s.frame.mainIdeas?.[i] || "";
    const arr = Array.isArray(s.frame.details?.[i]) ? s.frame.details[i] : [];
    const lines = arr.map((d, k) => `${k + 1}) ${d}`).join("\n");
    return `For this Main Idea: "${mi}", you identified the following Supporting Details:\n${lines}\nIs that correct, or would you like to revise one?`;
  }
  if (s.pending?.type === "offerMoreSoWhat") return `Do you want to add one more sentence to your So What? (yes/no)`;
  if (s.pending?.type === "collectMoreSoWhat") return `Add one more sentence to your So What:`;
  if (s.pending?.type === "confirmSoWhat") return `Your So What is: "${s.frame.soWhat}". Is that correct, or would you like to revise it?`;

  // Base progression
  if (!s.frameMeta?.purpose) {
    return "How will you use this Frame: studying/review, writing/creating, or create notes from a reading or source? (study/write/read)";
  }
  if (!s.frameMeta?.frameType) {
    return (
      "What kind of thinking are you doing: " +
      "1) Explain how/why something happens (Linear & Cause-and-Effect Relationships)  " +
      "2) Explain a big idea or theme (Framing Themes)  " +
      "3) Organize ideas from a text or source (Reading Frames). Which one (1–3)?"
    );
  }

  // Key Topic
  if (!s.frame.keyTopic) {
    if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
      return "What event, issue, or situation are you writing about? (2–5 words)";
    }
    return "What is your Key Topic? (2–5 words)";
  }

  // Is About
  if (!s.frame.isAbout) {
    const pb = getPromptForStage(s, "isAbout");
    return pb || `Finish this sentence: "${s.frame.keyTopic} is about ____."`;
  }

  // Main Ideas (2 required)
  if (s.frame.mainIdeas.length < 2) {
    if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
      const effect = safeEffectFromState(s);
      if (s.frame.mainIdeas.length === 0) {
        return PROMPT_BANK.write.causeEffect.mainIdeaFirst.replaceAll("[EFFECT]", effect);
      }
      return PROMPT_BANK.write.causeEffect.mainIdeaSecond;
    }

    const pb = getPromptForStage(s, "mainIdeas");
    return pb || (s.frame.mainIdeas.length === 0
      ? `What is your first Main Idea that helps explain ${s.frame.keyTopic}?`
      : `What is your second Main Idea that helps explain ${s.frame.keyTopic}?`);
  }

  // Details (2 per main idea)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const mi = s.frame.mainIdeas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      // Writing + CauseEffect: Explanation then Evidence
      if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
        const effect = safeEffectFromState(s);
        const tpl = arr.length === 0 ? PROMPT_BANK.write.causeEffect.detailExplain : PROMPT_BANK.write.causeEffect.detailEvidence;
        return tpl.replaceAll("[MAIN IDEA]", `"${mi}"`).replaceAll("[EFFECT]", `"${effect}"`);
      }

      const pb = getPromptForStage(s, `details:${i}`);
      if (pb) return `For this Main Idea: "${mi}", ${pb.replace(/\?\s*$/, "")}?`;
      return arr.length === 0
        ? `What is your first Supporting Detail for this Main Idea: "${mi}"?`
        : `What is your second Supporting Detail for this Main Idea: "${mi}"?`;
    }
  }

  // So What
  if (!s.frame.soWhat) {
    if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
      return PROMPT_BANK.write.causeEffect.soWhat;
    }
    const pb = getPromptForStage(s, "soWhat");
    return pb || `So what? Why does "${s.frame.keyTopic}" matter? (1–2 sentences)`;
  }

  return "Want to refine anything (Key Topic, Is About, Main Ideas, Details, or So What)?";
}

// ---------------------
// SSOT STATE UPDATE
// ---------------------
function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);
  ensureBuckets(s);

  if (!s.frameMeta) s.frameMeta = { purpose: "", frameType: "", causeEffect: { cause: "", effect: "" } };
  if (!s.frameMeta.causeEffect) s.frameMeta.causeEffect = { cause: "", effect: "" };

  // Handle CauseEffect Writing pending: evidence vs cause
  if (s.pending?.type === "ceMainIdeaEvidenceCheck") {
    const low = msg.toLowerCase().trim();
    const choice =
      low === "cause" ? "cause" :
      low === "evidence" ? "evidence" :
      low === "1" ? "cause" :
      low === "2" ? "evidence" :
      null;

    if (!choice) return s;

    if (choice === "cause") {
      const candidate = cleanText(s.pending.candidate || "");
      if (candidate) {
        s.frame.mainIdeas.push(candidate);
        if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) s.frame.details[s.frame.mainIdeas.length - 1] = [];
      }
      s.pending = null;

      if (s.frame.mainIdeas.length === 2) s.pending = { type: "offerThirdMainIdea" };
      return s;
    }

    // evidence
    s.pending = { type: "ceCollectBroaderCause" };
    return s;
  }

  if (s.pending?.type === "ceCollectBroaderCause") {
    const broader = cleanText(msg);
    if (broader && !isNegative(broader)) {
      s.frame.mainIdeas.push(broader);
      if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) s.frame.details[s.frame.mainIdeas.length - 1] = [];
    }
    s.pending = null;

    if (s.frame.mainIdeas.length === 2) s.pending = { type: "offerThirdMainIdea" };
    return s;
  }

  // ------
  // Revision workflows (deterministic)
  // ------

  if (s.pending?.type === "clarifyMainIdeaToRevise") {
    const max = Array.isArray(s.frame.mainIdeas) ? s.frame.mainIdeas.length : 2;
    const idx = parseRevisionIndex(msg, max);
    if (idx) {
      s.pending = { type: "collectMainIdeaRevision", index: idx - 1 };
      return s;
    }
    return s;
  }

  if (s.pending?.type === "collectMainIdeaRevision") {
    const i = Number(s.pending.index);
    if (!Number.isFinite(i) || i < 0 || i >= s.frame.mainIdeas.length) {
      s.pending = { type: "confirmMainIdeas" };
      return s;
    }
    const updated = cleanText(msg);
    if (updated && !isNegative(updated)) {
      s.frame.mainIdeas[i] = updated;
      if (!Array.isArray(s.frame.details[i])) s.frame.details[i] = [];
    }
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  if (s.pending?.type === "clarifyDetailToRevise") {
    const miIndex = Number(s.pending.index);
    const arr = Array.isArray(s.frame.details?.[miIndex]) ? s.frame.details[miIndex] : [];
    const max = Math.max(arr.length, 2);
    const idx = parseRevisionIndex(msg, max);
    if (idx) {
      s.pending = { type: "collectDetailRevision", index: miIndex, detailIndex: idx - 1 };
      return s;
    }
    return s;
  }

  if (s.pending?.type === "collectDetailRevision") {
    const miIndex = Number(s.pending.index);
    const dIndex = Number(s.pending.detailIndex);
    if (!Array.isArray(s.frame.details?.[miIndex])) s.frame.details[miIndex] = [];
    const arr = s.frame.details[miIndex];
    const max = Math.max(arr.length, 2);

    if (!Number.isFinite(dIndex) || dIndex < 0 || dIndex >= max) {
      s.pending = { type: "confirmDetails", index: miIndex };
      return s;
    }

    const updated = cleanText(msg);
    if (updated && !isNegative(updated)) {
      // If detail slot doesn't exist yet (should be rare), pad deterministically.
      while (arr.length <= dIndex) arr.push("");
      arr[dIndex] = updated;
      s.frame.details[miIndex] = arr.map(cleanText);
    }

    s.pending = { type: "confirmDetails", index: miIndex };
    return s;
  }


  // Purpose capture
  if (!s.frameMeta.purpose && !(s.pending && s.pending.type)) {
    const p = normalizePurpose(msg);
    if (p) {
      s.frameMeta.purpose = p;
      return s;
    }
  }

  // Frame type selection
  if (s.frameMeta?.purpose && !s.frameMeta.frameType && !(s.pending && s.pending.type)) {
    const ft = normalizeFrameTypeSelection(msg);
    if (ft) {
      s.frameMeta.frameType = ft;
      return s;
    }
  }

  // Language switch pending
  if (s.pending?.type === "confirmLanguageSwitch") {
    const normalized = msg.toLowerCase().trim();

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
    return s;
  }

  // Confirm Is About
  if (s.pending?.type === "confirmIsAbout") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    // Student revised isAbout
    s.frame.isAbout = msg;

    // Re-extract cause/effect for Writing + CauseEffect
    if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
      const parsed = parseCauseEffectLeadsTo(msg);
      s.frameMeta.causeEffect.cause = parsed?.cause || "";
      s.frameMeta.causeEffect.effect = parsed?.effect || "";
    }

    s.pending = null;
    return s;
  }

  // Offer third main idea
  if (s.pending?.type === "offerThirdMainIdea") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = { type: "collectThirdMainIdea" };
      return s;
    }
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  if (s.pending?.type === "collectThirdMainIdea") {
    if (!isNegative(msg)) {
      s.frame.mainIdeas.push(msg);
      if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) s.frame.details[s.frame.mainIdeas.length - 1] = [];
    }
    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  if (s.pending?.type === "confirmMainIdeas") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    const max = Array.isArray(s.frame.mainIdeas) ? s.frame.mainIdeas.length : 2;
    const idx = parseRevisionIndex(msg, max);

    if (idx) {
      s.pending = { type: "collectMainIdeaRevision", index: idx - 1 };
      return s;
    }

    if (indicatesRevisionIntent(msg)) {
      s.pending = { type: "clarifyMainIdeaToRevise" };
      return s;
    }

    return s;
  }

  if (s.pending?.type === "offerThirdDetail") {
    const normalized = msg.toLowerCase().trim();
    const idx = Number(s.pending.index);
    if (isAffirmative(normalized)) {
      s.pending = { type: "collectThirdDetail", index: idx };
      return s;
    }
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  if (s.pending?.type === "collectThirdDetail") {
    const idx = Number(s.pending.index);
    if (!Array.isArray(s.frame.details[idx])) s.frame.details[idx] = [];
    if (!isNegative(msg)) s.frame.details[idx] = [...s.frame.details[idx], msg];
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  if (s.pending?.type === "confirmDetails") {
    const normalized = msg.toLowerCase().trim();
    const i = Number(s.pending.index);
    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    const arr = Array.isArray(s.frame.details?.[i]) ? s.frame.details[i] : [];
    const max = Math.max(arr.length, 2);
    const idx = parseRevisionIndex(msg, max);

    if (idx) {
      s.pending = { type: "collectDetailRevision", index: i, detailIndex: idx - 1 };
      return s;
    }

    if (indicatesRevisionIntent(msg)) {
      s.pending = { type: "clarifyDetailToRevise", index: i };
      return s;
    }

    return s;
  }

  if (s.pending?.type === "offerMoreSoWhat") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = { type: "collectMoreSoWhat" };
      return s;
    }
    s.pending = { type: "confirmSoWhat" };
    return s;
  }

  if (s.pending?.type === "collectMoreSoWhat") {
    if (!isNegative(msg)) s.frame.soWhat = cleanText(`${s.frame.soWhat} ${msg}`);
    s.pending = { type: "confirmSoWhat" };
    return s;
  }

  if (s.pending?.type === "confirmSoWhat") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = null;
      if (isFrameComplete(s) && !s.flags.exportOffered) {
        s.flags.exportOffered = true;
        s.pending = { type: "offerExport" };
      }
      return s;
    }
    s.frame.soWhat = msg;
    s.pending = null;
    return s;
  }

  if (s.pending?.type === "offerExport") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = { type: "chooseExportType" };
      return s;
    }
    s.pending = null;
    return s;
  }

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

  // ---- Normal capture ----

  // 1) "X is about Y"
  const parsedKA = parseKeyTopicIsAbout(msg);
  if (parsedKA) {
    if (!s.frame.keyTopic) s.frame.keyTopic = parsedKA.keyTopic;
    if (!s.frame.isAbout) s.frame.isAbout = parsedKA.isAbout;

    if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
      const ce = parseCauseEffectLeadsTo(s.frame.isAbout);
      s.frameMeta.causeEffect.cause = ce?.cause || "";
      s.frameMeta.causeEffect.effect = ce?.effect || "";
    }

    s.pending = { type: "confirmIsAbout" };
    return s;
  }

  // 2) Key Topic capture
  if (!s.frame.keyTopic) {
    const wc = msg.split(/\s+/).filter(Boolean).length;
    if (!isBadKeyTopic(msg) && wc >= 2 && wc <= 5) {
      s.frame.keyTopic = msg;
      return s; // do not also treat this as isAbout
    }
    return s;
  }

  // 3) Is About capture
  if (!s.frame.isAbout) {
    const lowered = msg.toLowerCase().trim();
    if (lowered !== "revise" && lowered !== "change") {
      s.frame.isAbout = msg;

      // Writing + CauseEffect: extract cause/effect for specificity
      if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
        const ce = parseCauseEffectLeadsTo(msg);
        s.frameMeta.causeEffect.cause = ce?.cause || "";
        s.frameMeta.causeEffect.effect = ce?.effect || "";
      }

      s.pending = { type: "confirmIsAbout" };
    }
    return s;
  }

  // 4) Main Ideas capture (2 required)
  if (s.frame.mainIdeas.length < 2) {
    if (!isNegative(msg)) {
      // Writing + CauseEffect: evidence guardrail before committing
      if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect" && looksLikeEvidence(msg)) {
        s.pending = { type: "ceMainIdeaEvidenceCheck", candidate: msg };
        return s;
      }

      s.frame.mainIdeas.push(msg);
      if (!Array.isArray(s.frame.details[s.frame.mainIdeas.length - 1])) s.frame.details[s.frame.mainIdeas.length - 1] = [];

      if (s.frame.mainIdeas.length === 2) s.pending = { type: "offerThirdMainIdea" };
    }
    return s;
  }

  // 5) Details capture (2 per main idea)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      if (!isNegative(msg)) s.frame.details[i] = [...arr, msg];
      const updated = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
      if (updated.length === 2) s.pending = { type: "offerThirdDetail", index: i };
      return s;
    }
  }

  // 6) So What capture
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

    let state = normalizeIncomingState(body.state || body.vercelState || body.framing || {});

    // Safety
    if (message) {
      const safety = await classifyMessage(message);
      if (safety?.blocked) {
        const reply = SAFETY_RESPONSES[safety.category] || SAFETY_RESPONSES.default;
        const out = enforceSingleQuestion(reply);

        appendTurn(state, "Student", message);
        appendTurn(state, "Kaw", out);

        return res.status(200).json({ reply: out, state });
      }
    }

    // Language detect (only if not locked and not already pending)
    if (message && !state.settings.languageLocked && state.pending?.type !== "confirmLanguageSwitch") {
      const detected = await detectLanguageViaLLM(message);
      if (detected && detected.code && detected.code !== "en") {
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

    // ConfirmLanguageSwitch handling
    if (state.pending?.type === "confirmLanguageSwitch" && message) {
      const low = message.toLowerCase().trim();
      let proceedState = state;

      if (!isAffirmative(low) && !isNegative(low)) {
        const yn = await classifyYesNoViaLLM(message);
        if (yn === "yes") proceedState = updateStateFromStudent(state, "yes");
        else if (yn === "no") proceedState = updateStateFromStudent(state, "no");
        else {
          const q = computeNextQuestion(state);
          let reply = enforceSingleQuestion(q);

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
      // Optional: stuck logic can be layered here later (kept minimal in this patch)
      state = updateStateFromStudent(state, message);
    }

    let reply = enforceSingleQuestion(computeNextQuestion(state));

    if (state.settings.languageLocked && state.settings.language !== "en") {
      reply = await translateQuestionViaLLM(reply, state.settings.languageName || "the target language");
    }

    if (message) appendTurn(state, "Student", message);
    appendTurn(state, "Kaw", reply);

    if (isFrameComplete(state) && !state.pending) {
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

