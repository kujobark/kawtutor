import OpenAI from "openai";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const BUILD_TAG = "build-2026-02-21-a";

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
// STATE
// ---------------------
function defaultState() {
  return {
    version: 2,
    frameMeta: {
      purpose: "",   // study|write|read
      frameType: "", // causeEffect|themes|reading|general
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
  base.settings.languageNativeName =
    cleanText(settings.languageNativeName || base.settings.languageNativeName) || base.settings.languageName;
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
      soWhat: 'When you look at all of these causes and effects together, what can you conclude about "[Key Topic]"?',
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
      soWhat: "Why should people care about this effect?",
    },
  },
};

function fillTopic(template, keyTopic) {
  return (template || "").replaceAll("[Key Topic]", keyTopic || "your topic");
}

function getPromptForStage(state, stage) {
  const purpose = state.frameMeta?.purpose || "";
  const frameType = state.frameMeta?.frameType || "";
  const kt = state.frame?.keyTopic || "";

  if (stage === "isAbout") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.isAbout;
    if (tpl) return fillTopic(tpl, kt);
  }

  if (stage === "soWhat") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.soWhat;
    if (tpl) return fillTopic(tpl, kt);
  }

  return null;
}

// ---------------------
// QUESTION ROUTER
// ---------------------
function computeNextQuestion(state) {
  const s = state;

  // Base progression
  if (!s.frameMeta?.purpose) {
    return "How will you use this Frame: studying/review, writing/creating, or create notes from a reading or source? (study/write/read)";
  }
  if (!s.frameMeta?.frameType) {
    return (
      "What kind of thinking are you doing?\n" +
      "1) Explain how/why something happens (Linear & Cause-and-Effect Relationships)\n" +
      "2) Explain a big idea or theme (Framing Themes)\n" +
      "3) Organize ideas from a text or source (Reading Frames)\n" +
      "4) Organize my thinking (General Frame)\n\n" +
      "Reply with 1, 2, 3, or 4."
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

  // For write/causeEffect, parse effect for prompt injection
  if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
    const ce = parseCauseEffectLeadsTo(s.frame.isAbout);
    s.frameMeta.causeEffect.cause = ce?.cause || s.frameMeta.causeEffect.cause || "";
    s.frameMeta.causeEffect.effect = ce?.effect || s.frameMeta.causeEffect.effect || "";
  }

  // Main Ideas (2 required)
  if ((s.frame.mainIdeas || []).length < 2) {
    if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
      const effect = safeEffectFromState(s);
      if ((s.frame.mainIdeas || []).length === 0) return PROMPT_BANK.write.causeEffect.mainIdeaFirst.replaceAll("[EFFECT]", effect);
      return PROMPT_BANK.write.causeEffect.mainIdeaSecond;
    }

    return (s.frame.mainIdeas || []).length === 0
      ? `What is your first Main Idea that helps explain ${s.frame.keyTopic}?`
      : `What is your second Main Idea that helps explain ${s.frame.keyTopic}?`;
  }

  // Details (2 per main idea)
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const mi = s.frame.mainIdeas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
        const effect = safeEffectFromState(s);
        const tpl = arr.length === 0 ? PROMPT_BANK.write.causeEffect.detailExplain : PROMPT_BANK.write.causeEffect.detailEvidence;
        return tpl.replaceAll("[MAIN IDEA]", `"${mi}"`).replaceAll("[EFFECT]", `"${effect}"`);
      }

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
// SSOT STATE UPDATE (minimal for demo)
// ---------------------
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
  if (t === "1" || t.startsWith("1")) return "causeEffect";
  if (t === "2" || t.startsWith("2")) return "themes";
  if (t === "3" || t.startsWith("3")) return "reading";
  if (t === "4" || t.startsWith("4")) return "general";
  if (t.includes("cause") || t.includes("effect") || t.includes("how") || t.includes("why")) return "causeEffect";
  if (t.includes("theme") || t.includes("big idea") || t.includes("central idea")) return "themes";
  if (t.includes("read") || t.includes("text") || t.includes("source") || t.includes("note")) return "reading";
  if (t.includes("general") || t.includes("organize")) return "general";
  return null;
}

function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);
  ensureBuckets(s);

  // Purpose
  if (!s.frameMeta.purpose) {
    const p = normalizePurpose(msg);
    if (p) {
      s.frameMeta.purpose = p;
      return s;
    }
  }

  // Frame type
  if (s.frameMeta.purpose && !s.frameMeta.frameType) {
    const ft = normalizeFrameTypeSelection(msg);
    if (ft) {
      s.frameMeta.frameType = ft;
      return s;
    }
  }

  // Key Topic
  if (!s.frame.keyTopic) {
    const wc = msg.split(/\s+/).filter(Boolean).length;
        if (!isBadKeyTopic(msg) && wc >= 2 && wc <= 5) {
      s.frame.keyTopic = msg;
    }
    return s;
  }

  // Is About
  if (!s.frame.isAbout) {
    s.frame.isAbout = msg;
    return s;
  }

  // Main Ideas
  if (s.frame.mainIdeas.length < 2) {
    if (!isNegative(msg)) s.frame.mainIdeas.push(msg);
    ensureBuckets(s);
    return s;
  }

  // Details
  for (let i = 0; i < s.frame.mainIdeas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) {
      if (!isNegative(msg)) s.frame.details[i] = [...arr, msg];
      return s;
    }
  }

  // So What
  if (!s.frame.soWhat) {
    if (!isNegative(msg)) s.frame.soWhat = msg;
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

    if (message) state = updateStateFromStudent(state, message);

    let reply = enforceSingleQuestion(computeNextQuestion(state));

    if (message) appendTurn(state, "Student", message);
    appendTurn(state, "Kaw", reply);

    if (isFrameComplete(state)) {
      const frameText = buildFrameText(state);
      const transcriptText = buildTranscriptText(state);
      const html = buildExportHtml(state);
      state.exports = { frameText, transcriptText, html };
    } else {
      state.exports = null;
    }

    return res.status(200).json({ reply, state, build: BUILD_TAG });
  } catch (err) {
    console.error("Tutor API error:", err);
    return res.status(200).json({
      reply: "Hmm — I had trouble processing that. Can you try again?",
      state: normalizeIncomingState({}),
    });
  }
}
