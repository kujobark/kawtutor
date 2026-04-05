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
  if (!out) return "What should someone understand or learn from this idea?";

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

function getKeyTopicFeedback(input) {
  const text = cleanText(input);
  const wc = text.split(/\s+/).filter(Boolean).length;

  if (!text || isBadKeyTopic(text)) {
    return "That’s a good start, but your Key Topic should name the topic clearly, not use a generic word like “topic” or “my essay.”";
  }

  if (wc < 2) {
    return "That’s a good start, but your Key Topic should be a short phrase (2–5 words), not just one word.";
  }

  if (wc > 5) {
    return "That’s a strong idea, but your Key Topic should be a short phrase (2–5 words). Try shortening it.";
  }

  return null;
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

  const frustration = [
    "confus",
    "frustrat",
    "annoy",
    "angry",
    "mad",
    "ugh",
    "this sucks",
    "this is hard",
    "this is confusing",
    "this makes no sense",
    "i'm confused",
    "im confused",
    "i'm lost",
    "im lost",
    "i can't do this",
    "i cant do this",
    "stupid",
    "dumb",
    "hate",
  ];
  if (frustration.some((p) => t.includes(p))) return "frustration";

  const resistance = [
    "do we have to",
    "why do we have to",
    "why am i doing",
    "what's the point",
    "whats the point",
    "pointless",
    "this is pointless",
    "can you just tell me",
    "just tell me",
  ];
  if (resistance.some((p) => t.includes(p))) return "resistance";

  return "neutral";
}

function isStuckMessage(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  const exact = new Set([
    "idk",
    "i dont know",
    "i don't know",
    "dont know",
    "don't know",
    "not sure",
    "im not sure",
    "i'm not sure",
    "no idea",
    "help",
    "can you help",
    "i need help",
    "stuck",
    "skip",
    "i'm stuck",
    "im stuck",
    "confused",
    "lost",
    "blank",
    "blanking",
    "nothing",
    "i forgot",
    "i dont remember",
    "i don't remember",
  ]);

  if (exact.has(t)) return true;

  const patterns = [
    "i dont get it",
    "i don't get it",
    "i dont understand",
    "i don't understand",
    "this is hard",
    "this is confusing",
    "this makes no sense",
    "im confused",
    "i'm confused",
    "im lost",
    "i'm lost",
    "i cant do this",
    "i can't do this",
    "what do i do",
    "what am i supposed to do",
    "what does that mean",
    "can you just tell me",
    "just tell me",
    "i forgot what to do",
    "i don't remember what to do",
    "i dont remember what to do",
  ];

  if (patterns.some((p) => t.includes(p))) return true;

  const hesitantShort = new Set([
    "maybe",
    "i guess",
    "guess",
  ]);
  if (hesitantShort.has(t)) return true;

  return false;
}
function formatNudgeText(nudges) {
  const items = Array.isArray(nudges)
    ? nudges.map((n) => (n || "").toString().trim()).filter(Boolean)
    : [];

  if (!items.length) {
    return "Try one small step:";
  }

  return items.join("\n\n");
}

function normalizePurpose(msg) {
  const t = cleanText(msg).toLowerCase();
  if (!t) return null;
  if (t.includes("study") || t.includes("review") || t === "s") return "study";
  if (t.includes("write") || t.includes("essay") || t.includes("paragraph") || t.includes("create") || t === "w")
    return "write";
  if (t.includes("read") || t.includes("note") || t.includes("annot") || t === "r") return "read";
  // allow 1/2/3 mapping (for buttons)
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
  if (t === "4" || t.startsWith("4 ")) return "general";

  // Accept common text variants (still deterministic)
  if (t.includes("cause") || t.includes("effect") || t.includes("how") || t.includes("why")) return "causeEffect";
  if (t.includes("theme") || t.includes("big idea") || t.includes("central idea")) return "themes";
  if (t.includes("read") || t.includes("text") || t.includes("source") || t.includes("note")) return "reading";
  if (t.includes("general") || t.includes("organize") || t.includes("organise")) return "general";

  return null;
}

function fillTopic(template, keyTopic) {
  return (template || "").replaceAll("[Key Topic]", keyTopic || "your topic");
}

function applyPromptTokens(template, state) {
  let out = template || "";
const kt = state?.frame?.keyTopic || "";
const eff = state?.frame?.effect || "";
const isAbout = state?.frame?.isAbout || "";
const ideas = getIdeaList(state).filter(Boolean);

const activeStage = state?.pending?.stage || getStage(state) || "";
let cause = ideas.length ? ideas[ideas.length - 1] : "";


if (typeof activeStage === "string" && activeStage.startsWith("details:")) {
  const rawIndex = activeStage.split(":")[1];
  const idx = Number(rawIndex);
  if (Number.isInteger(idx) && ideas[idx]) {
    cause = ideas[idx];
  }
}

  // Key Topic token
  if (kt) out = out.replace(/\[Key Topic\]/g, kt);
  if (isAbout) out = out.replace(/\[IS_ABOUT\]/g, isAbout);
  
  // Cause/Effect tokens / phrases
  if (eff) {
    out = out.replace(/\[EFFECT\]/g, eff);
    out = out.replace(/the effect you[’']?re writing about/gi, eff);
    out = out.replace(/\bthe effect\b/gi, eff);
  }

  // Cause token
  if (cause) {
    out = out.replace(/\[CAUSE\]/g, cause);
  }

  return out;
}

// ---------------------
// PROMPT BANK
// ---------------------
const PROMPT_BANK = {

  study: {
    causeEffect: {
      isAbout: 'Your Key Topic is:\n\n"[Key Topic]"\n\nNow let\'s think about what happens with this topic.\n\nIn your own words, what is the main effect or result?',
      mainIdea: 'You are explaining why this happens:\n\n"[EFFECT]"\n\nWhat is one major cause that contributes to it?',
      detail: 'Here is the cause you are working with:\n\n"[CAUSE]"\n\nWhat is one detail or example that shows how this leads to\n\n"[EFFECT]"?',
      soWhat: 'Your frame shows that:\n\n"[CAUSE]"\n\nThis helps explain why\n\n"[EFFECT]"\n\nLooking at this pattern,\n\nwhat important takeaway should someone understand about "[Key Topic]"?',
    },
     themes: {
      isAbout: 'Your Key Topic is:\n\n"[Key Topic]"\n\nNow think about the deeper meaning.\n\nWhat message about life does this topic reveal?',
      mainIdea: 'You identified this message about life:\n\n"[IS_ABOUT]"\n\nWhat is one Main Idea that helps show this message about life?',
      detail: 'What specific example or explanation helps show this message about life in action?',
      soWhat: 'What should people understand about life or people because of this message?'
  }
  },

  write: {
    causeEffect: {
      isAbout: 'Your Key Topic is:\n\n"[Key Topic]"\n\nNow let\'s think about what happens in this topic.\n\nFinish this sentence:\n"This topic is about how ____ leads to ____."',
      mainIdea: 'You are explaining why this happens:\n\n"[EFFECT]"\n\nWhat is one major cause that contributes to it?',
      detail: 'Here is the cause you are working with:\n\n"[CAUSE]"\n\nWhat is one detail or example that shows how this leads to\n\n"[EFFECT]"?',
      soWhat: 'Your frame shows that:\n\n"[CAUSE]"\n\nThis helps explain why\n\n"[EFFECT]"\n\nLooking at this pattern,\n\nwhat does this pattern help us understand about this effect?',  
    },
   themes: {
      isAbout: 'Your Key Topic is:\n\n"[Key Topic]"\n\nNow think about the deeper meaning.\n\nWhat message about life do you want your reader to understand?',
      mainIdea: 'You want to show this message about life:\n\n"[IS_ABOUT]"\n\nWhat is one Main Idea you can use to help develop this message?',
      detail: 'What specific example or explanation helps show this message about life in action?',
      soWhat: 'What should your reader understand about life or people because of this message?'
    }
  },

  read: {
    causeEffect: {
      isAbout: 'The text is about:\n\n"[Key Topic]"\n\nNow let\'s think about what happens in this topic.\n\nWhat main effect or result does the author emphasize?',
      mainIdea: 'The text explains this effect:\n\n"[EFFECT]"\n\nWhat are the main causes the author presents that lead to this effect?',
      detail: 'Here is the cause you are working with:\n\n"[CAUSE]"\n\nWhat evidence from the text shows how this leads to\n\n"[EFFECT]"?',
      soWhat: 'Why does understanding this cause-and-effect relationship matter in the text?'
    },
     themes: {
      isAbout: 'The text focuses on:\n\n"[Key Topic]"\n\nWhat message about life does the author reveal through this topic?',
      mainIdea: 'The text shows this message about life:\n\n"[IS_ABOUT]"\n\nWhat Main Idea from the text helps reveal this message?',
      detail: 'What specific evidence or explanation helps show this message about life in action?',
      soWhat: 'What should the reader understand about life or people because of this message?'
    }
  }
};

function getPromptForStage(state, stage) {
  const purpose = state.frameMeta?.purpose || "";
  const frameType = state.frameMeta?.frameType || "";
  const kt = state.frame?.keyTopic || "";

  if (stage === "isAbout") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.isAbout;
    if (tpl) return applyPromptTokens(fillTopic(tpl, kt), state);
  }

  if (stage === "mainIdeas") {
    const q = PROMPT_BANK?.[purpose]?.[frameType]?.mainIdea;
    if (q) return applyPromptTokens(q, state);
  }

  if (stage.startsWith("details:")) {
    const q = PROMPT_BANK?.[purpose]?.[frameType]?.detail;
    if (q) return applyPromptTokens(q, state);
  }

  if (stage === "soWhat") {
    const tpl = PROMPT_BANK?.[purpose]?.[frameType]?.soWhat;
    if (tpl) return applyPromptTokens(fillTopic(tpl, kt), state);
  }

  return null;
}

// ---------------------
// STUCK NUDGES
// ---------------------
function buildStuckNudges(state, stage) {
  const purpose = state?.frameMeta?.purpose || "";
  const frameType = state?.frameMeta?.frameType || "";

  const keyTopic =
    state?.frame?.keyTopic ||
    state?.frame?.topic ||
    "";

  const effect =
    state?.frame?.effect ||
    state?.frame?.centralEffect ||
    state?.frame?.result ||
    "";

const ideas = getIdeaList(state).filter(Boolean);

const mostRecentCause =
  ideas.length > 0 ? ideas[ideas.length - 1] : "";

  const genericMainIdeas = [
    "What is one main idea you could add?",
    "What is an important part of this topic?",
    "What belongs as a main idea here?"
  ];

  const genericDetails = [
    "What is one example that fits here?",
    "What is one fact or detail you could add?",
    "What specific information supports this idea?"
  ];

  const genericSoWhat = [
    "Why does this matter?",
    "What should someone understand after reading this?",
    "What is the takeaway?"
  ];

  // MAIN IDEAS
  if (stage === "mainIdeas") {
    if (frameType === "causeEffect") {

      if (mostRecentCause && effect && keyTopic) {
        return [
`Your frame explains why ${effect} happens.

You already identified this major cause:
"${mostRecentCause}"

Ask yourself: what other cause related to ${keyTopic} might also contribute to ${effect}?`,

"What is another major cause?"
        ];
      }

      if (mostRecentCause && effect) {
        return [
`Your frame explains why ${effect} happens.

You already identified this major cause:
"${mostRecentCause}"

Ask yourself: what other cause might also contribute to ${effect}?`,

"What is another major cause?"
        ];
      }

      if (effect && keyTopic) {
        return [
`Your frame explains why ${effect} happens.

Ask yourself: what cause related to ${keyTopic} might help explain ${effect}?`,

"What is one major cause?"
        ];
      }

      if (effect) {
        return [
`Your frame explains why ${effect} happens.

Ask yourself: what cause might help explain that effect?`,

"What is one major cause?"
        ];
      }
    }

    if (frameType === "themes") {
      const theme = state?.frame?.isAbout || "";

      if (mostRecentCause && theme && keyTopic) {
        return [
`Your frame is showing this message about life:
"${theme}"

You already identified this main idea:
"${mostRecentCause}"

Ask yourself: what other idea, example, or moment related to ${keyTopic} could also help show this message?`,

"What is another main idea?"
        ];
      }

      if (mostRecentCause && theme) {
        return [
`Your frame is showing this message about life:
"${theme}"

You already identified this main idea:
"${mostRecentCause}"

Ask yourself: what other idea, example, or moment could also help show this message?`,

"What is another main idea?"
        ];
      }

      if (theme && keyTopic) {
        return [
`Your frame is showing this message about life:
"${theme}"

Ask yourself: what idea, example, or moment related to ${keyTopic} could help show this message?`,

"What is one main idea?"
        ];
      }

      if (theme) {
        return [
`Your frame is showing this message about life:
"${theme}"

Ask yourself: what idea, example, or moment could help show this message?`,

"What is one main idea?"
        ];
      }
    }

        return genericMainIdeas;
  }
  
  // DETAILS
  if (typeof stage === "string" && stage.startsWith("details:")) {
    const rawIndex = stage.split(":")[1];
    const idx = Number(rawIndex);
    const selectedMainIdea = Number.isInteger(idx) ? (ideas[idx] || "") : "";

    if (frameType === "causeEffect") {

      if (selectedMainIdea && effect) {
        return [
`You already identified this cause:
"${selectedMainIdea}"

Now help the reader understand how that cause leads to ${effect}.`,

"What specific detail, example, or explanation could support or explain that cause?"
        ];
      }

      if (selectedMainIdea) {
        return [
`You already identified this cause:
"${selectedMainIdea}"

Now help the reader understand how that cause connects to your frame.`,

"What specific detail, example, or explanation could support or explain that cause?"
        ];
      }
    }

    if (frameType === "themes") {
      const theme = state?.frame?.isAbout || "";

      if (selectedMainIdea && theme) {
        return [
`You already identified this main idea:
"${selectedMainIdea}"

Now help the reader understand how this shows the message:
"${theme}"`,

"What specific detail, example, or explanation could help show this theme in action?"
        ];
      }

      if (selectedMainIdea) {
        return [
`You already identified this main idea:
"${selectedMainIdea}"

Now help the reader understand how it connects to your frame.`,

"What specific detail, example, or explanation could help explain that support?"
        ];
      }
    }

  return genericDetails;
  }
    
  // SO WHAT
  if (stage === "soWhat") {
    if (frameType === "causeEffect") {

      if (ideas.length > 1 && effect && keyTopic) {  
        return [
`You identified causes that lead to ${effect}.

Now think about the bigger meaning of that pattern in ${keyTopic}.`,

"What larger idea or takeaway should someone understand about this topic?"
        ];
      }

      if (ideas.length > 0 && effect) {  
        return [
`You identified causes that lead to ${effect}.

Now think about the bigger meaning of that pattern.`,

"What larger idea or takeaway should someone understand?"
        ];
      }

      if (mostRecentCause && effect) {
        return [
`If "${mostRecentCause}" leads to ${effect},

think about the bigger meaning of that connection.`,

"What larger idea or takeaway should someone understand?"
        ];
      }
    }

    if (frameType === "themes") {
      const theme = state?.frame?.isAbout || "";

      if (ideas.length > 1 && theme && keyTopic) {
        return [
`You identified supports that help show this message:
"${theme}"

Now think about the bigger meaning of that pattern in ${keyTopic}.`,

"What larger idea or takeaway should someone understand about this theme?"
        ];
      }

      if (ideas.length > 0 && theme) {
        return [
`You identified supports that help show this message:
"${theme}"

Now think about the bigger meaning of that pattern.`,

"What larger idea or takeaway should someone understand?"
        ];
      }

      if (theme) {
        return [
`Your frame is showing this message about life:
"${theme}"

Now think beyond this one example or text.`,

"What larger idea or takeaway should someone understand?"
        ];
      }
    }
    
    return genericSoWhat;
  }

   return [
    "What is one small next step you could try?",
    "What is one idea you are considering?",
    "What part feels easiest to answer first?"
  ];
}

// ---------------------
// WRITE-MODE GUARDRAILS (heuristics only)
// ---------------------
function looksLikeEvidence(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  // numbers, percentages, or quoted text often signal evidence
  if (/\d/.test(t)) return true;
  if (t.includes("%")) return true;
  if (t.includes('"') || t.includes("“") || t.includes("”")) return true;

  const markers = [
    "for example",
    "for instance",
    "such as",
    "according to",
    "the text says",
    "in the text",
    "in the article",
    "in the source",
    "the author",
    "the study",
    "research",
    "survey",
    "data",
    "statistic",
    "evidence",
    "report",
    "shows that",
    "found that",
  ];
  return markers.some((p) => t.includes(p));
}

function looksLikeMechanism(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  // causal connectives often signal mechanism/explanation rather than evidence
  const markers = [
    "because",
    "leads to",
    "causes",
    "results in",
    "therefore",
    "so that",
    "this makes",
    "which makes",
    "as a result",
    "due to",
  ];
  return markers.some((p) => t.includes(p));
}

function shouldRequestEvidenceDetail(state, detailText) {
  // Only apply guardrail for: write mode + causeEffect + details stage
  if (state.frameMeta?.purpose !== "write") return false;
  if (state.frameMeta?.frameType !== "causeEffect") return false;

  const t = cleanText(detailText);
  if (!t) return false;

  // If it already looks like evidence, don't interrupt.
  if (looksLikeEvidence(t)) return false;

  // If it looks like mechanism (how/why) but not evidence, ask for evidence.
  return looksLikeMechanism(t);
}

// ---------------------
// STAGE
// ---------------------
// --------------------------------------------------
// FRAME STAGE ENGINE
// --------------------------------------------------
// Determines the current progression stage of the frame.
// This is the core deterministic state machine.
//
// Order matters. The engine always returns the FIRST
// stage that has not yet been satisfied.
//
// Stage progression:
// 1. purpose
// 2. frameType
// 3. keyTopic
// 4. isAbout
// 5. mainIdeas
// 6. details (per main idea)
// 7. soWhat
// 8. refine
//
// NOTE:
// Parent Anchor stages map onto these later via the
// Parent Anchor Bridge, but this engine remains the
// single source of truth for frame progression.

const FRAME_STAGE_SEQUENCE = [
  "purpose",
  "frameType",
  "keyTopic",
  "isAbout",
  "mainIdeas",
  "details",
  "soWhat",
  "refine",
];

function getStage(state) {
  const f = state.frame;
  const m = state.frameMeta || {};
  const ideas = getIdeaList(state);

  if (!m.purpose) return "purpose";
  if (!m.frameType) return "frameType";
  if (!f.keyTopic) return "keyTopic";
  if (!f.isAbout) return "isAbout";
  if (ideas.length < 2) return "mainIdeas";

  for (let i = 0; i < ideas.length; i++) {
    const arr = Array.isArray(f.details?.[i]) ? f.details[i] : [];
    if (arr.length < 2) return `details:${i}`;
  }

  if (!f.soWhat) return "soWhat";
  return "refine";
}

function getBaseStage(stage) {
  if (!stage) return "";
  if (stage.startsWith("details:")) return "details";
  return stage;
}

function getIdeaList(state) {
  const isCE = state?.frameMeta?.frameType === "causeEffect";

  if (isCE) {
    return Array.isArray(state?.frame?.causes)
      ? state.frame.causes
      : [];
  }

  return Array.isArray(state?.frame?.parentItems)
    ? state.frame.parentItems
    : [];
}

// ---------------------
// PARENT ANCHOR BRIDGE
// ---------------------

// PARENT ANCHOR SANDBOX GUARDRAIL
// -------------------------------
// The Parent Anchor should first become the system's best explanation
// of the engine before it becomes the system's new engine.
//
// In this sandbox phase, Parent Anchor improves observability,
// interpretation, and structural clarity — not runtime authority.
// This layer is strictly read-only in this phase.
//
// That means this layer must:
// - not change progression logic
// - not replace getStage()
// - not alter pending-state semantics
// - not become a competing controller
//
// Runtime control and state mutation remain with:
// - getStage(state)
// - computeNextQuestion(state)
// - updateStateFromStudent(state)

// This bridge does NOT change progression logic.
// It interprets the current tutor.js workflow through the
// Parent Anchor structural stage model.
//
// Parent Anchor structural stage model:
// the invariant Framing Routine spine
// Key Topic -> Is About -> Main Ideas -> Details -> So What
//
// Structural stages are the invariant Parent Anchor stages.
// Pending-state mappings are used to infer confirmation/export stages.
// Interrupt mappings map temporary correction states back to the
// structural stage they belong to.
// Overlay pending types are non-structural helpers (for example,
// stuck support or language support) and should not be treated as
// Parent Anchor stages.

const PARENT_ANCHOR_BRIDGE = {
  structuralStages: [
    "purpose",
    "frameType",
    "keyTopic",
    "isAbout",
    "isAboutConfirm",
    "parentItems",
    "parentItemsConfirm",
    "detailsLoop",
    "detailsConfirmLoop",
    "soWhat",
    "soWhatConfirm",
    "export",
  ],

  // Pending states that indicate the engine is currently inside
  // a structural confirmation/export stage.
  //
  // These mappings preserve current tutor.js behavior only.
  // They should not be mistaken for permanent instructional rules.
  confirmationStageByPending: {
    confirmIsAbout: "isAboutConfirm",

     offerAnotherMainIdea: "parentItemsConfirm",
     collectAnotherMainIdea: "parentItemsConfirm",
     confirmMainIdeas: "parentItemsConfirm",

    offerAnotherDetail: "detailsConfirmLoop",
    collectAnotherDetail: "detailsConfirmLoop",
    confirmDetails: "detailsConfirmLoop",

    // Current-behavior compatibility only:
    // tutor.js currently allows optional additional So What text,
    // but the long-term Parent Anchor contract does NOT require
    // a multi-step So What expansion loop.
    offerMoreSoWhat: "soWhatConfirm",
    collectMoreSoWhat: "soWhatConfirm",
    confirmSoWhat: "soWhatConfirm",

    offerExport: "export",
    chooseExportType: "export",
  },

  // Pending states that temporarily interrupt input capture
  // but do NOT create a new structural stage.
  interruptStageByPending: {
    needWriteCauseEffectStem: "isAbout",
    writeNeedEvidenceDetail: "detailsLoop",
  },

  // Overlay pending states are helper flows, not structural stages.
  // They should be interpreted around the current structural stage.
  overlayPendingTypes: new Set([
    "confirmLanguageSwitch",
    "stuckConfirm",
    "stuckMenu",
    "stuckReask",
    "stuckNudge",
    "stuckMini",
    "stuckSkip",
  ]),

  // Raw getStage() outputs mapped to Parent Anchor structural stages.
  //
  // Detail buckets like details:0 / details:1 collapse to the single
  // structural stage "detailsLoop".
  //
  // Post-completion states are interpreted structurally as "export"
  // so the Parent Anchor endpoint stays stable even if tutor.js
  // continues to expose completion/refine behavior around export.
  structuralStageByRawStage(rawStage) {
    if (rawStage === "purpose") return "purpose";
    if (rawStage === "frameType") return "frameType";
    if (rawStage === "keyTopic") return "keyTopic";
    if (rawStage === "isAbout") return "isAbout";
    if (rawStage === "mainIdeas") return "parentItems";
    if (typeof rawStage === "string" && rawStage.startsWith("details:")) return "detailsLoop";
    if (rawStage === "soWhat") return "soWhat";
    if (rawStage === "refine") return "export";
    if (rawStage === "export") return "export";
    return null;
  },
};

/**
 * Returns the current Parent Anchor structural stage without changing
 * any existing tutor.js progression behavior.
 *
 * This helper is an interpretation layer only.
 * It does NOT advance stages, mutate state, or replace getStage().
 *
 * It interprets the current tutor.js workflow through the Parent Anchor
 * structural stage model: the invariant Framing Routine spine
 * Key Topic -> Is About -> Main Ideas -> Details -> So What.
 *
 * How it works:
 * 1) It checks state.pending?.type first.
 *    - confirmation/export pending states override raw getStage()
 *    - interrupt pending states map back to their owning structural stage
 *    - overlay pending types do not become structural stages
 *
 * 2) If no pending override applies, it falls back to getStage(state).
 *
 * 3) Raw detail stages like "details:0" or "details:1" collapse to
 *    the single structural stage "detailsLoop".
 *
 * 4) Post-completion raw stages like "refine" are interpreted
 *    structurally as "export".
 *
 * Sandbox guardrail:
 * This helper explains the current engine structurally.
 * It must not become a new progression controller in this phase.
 */
function getParentAnchorStage(state) {
  const pendingType = state?.pending?.type || null;

  // Confirmation/export pending states take priority because they
  // represent the active structural stage the student is currently in.
  if (pendingType && PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType]) {
    return PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType];
  }

  // Interrupts belong to an underlying structural stage and do not
  // create a new Parent Anchor stage.
  if (pendingType && PARENT_ANCHOR_BRIDGE.interruptStageByPending[pendingType]) {
    return PARENT_ANCHOR_BRIDGE.interruptStageByPending[pendingType];
  }

  // Stuck overlays should use the saved stage only if it actually exists
  // in the current pending payload. Otherwise, fall back to getStage(state).
  if (pendingType && pendingType.startsWith("stuck")) {
    const savedStage = state?.pending?.stage || null;
    if (savedStage) {
      const mappedSavedStage = PARENT_ANCHOR_BRIDGE.structuralStageByRawStage(savedStage);
      if (mappedSavedStage) return mappedSavedStage;
    }
  }

  // Other overlays remain non-structural and do not override the
  // underlying Parent Anchor stage. Fall back to the raw current stage.
  const rawStage = getStage(state);
  return PARENT_ANCHOR_BRIDGE.structuralStageByRawStage(rawStage);
}

/**
 * Returns the structural Parent Anchor stage that owns the current moment.
 *
 * This is a read-only interpretation helper.
 * It does NOT advance stages, mutate state, or replace getStage().
 *
 * Difference from getParentAnchorStage(state):
 * - getParentAnchorStage(state) returns the currently interpreted structural stage
 * - getParentAnchorOwnerStage(state) returns the structural owner of the
 *   current pending flow, including interrupt and overlay cases
 */
function getParentAnchorOwnerStage(state) {
  const pendingType = state?.pending?.type || null;

  // Confirmation/export pending states explicitly own the current moment.
  if (pendingType && PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType]) {
    return PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType];
  }

  // Interrupt pending states belong to an underlying structural stage.
  if (pendingType && PARENT_ANCHOR_BRIDGE.interruptStageByPending[pendingType]) {
    return PARENT_ANCHOR_BRIDGE.interruptStageByPending[pendingType];
  }

  // Overlay helper flows do not create a new structural stage.
  // If they saved a raw resume stage, map that back to its structural owner.
  if (pendingType && PARENT_ANCHOR_BRIDGE.overlayPendingTypes.has(pendingType)) {
    const savedStage = state?.pending?.stage || null;
    if (savedStage) {
      const mappedSavedStage = PARENT_ANCHOR_BRIDGE.structuralStageByRawStage(savedStage);
      if (mappedSavedStage) return mappedSavedStage;
    }
  }

  // Otherwise, the owner is the current interpreted Parent Anchor stage.
  return getParentAnchorStage(state);
}

/**
 * Returns the normalized Parent Anchor loop/mode type for the current moment.
 *
 * This helper is read-only and classification-only.
 * It does NOT alter routing behavior.
 */
function getParentAnchorLoopType(state) {
  const pendingType = state?.pending?.type || null;
  const structuralStage = getParentAnchorStage(state);

  if (structuralStage === "export") return "export";

  if (pendingType && PARENT_ANCHOR_BRIDGE.overlayPendingTypes.has(pendingType)) {
    return "overlay";
  }

  if (pendingType && PARENT_ANCHOR_BRIDGE.interruptStageByPending[pendingType]) {
    return "interrupt";
  }

  if (pendingType && PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType]) {
    return "confirm";
  }

  return "capture";
}

/**
 * Returns a consolidated read-only Parent Anchor structural snapshot.
 *
 * This helper exists for observability, logging, debugging, and later
 * architectural extraction. It must not be used to change runtime behavior
 * in the sandbox phase.
 */
function getParentAnchorContext(state) {
  const rawStage = getStage(state);
  const structuralStage = getParentAnchorStage(state);
  const ownerStructuralStage = getParentAnchorOwnerStage(state);
  const pendingType = state?.pending?.type || null;
  const savedStage = state?.pending?.stage || null;
  const loopType = getParentAnchorLoopType(state);

  return {
    rawStage,
    structuralStage,
    ownerStructuralStage,
    pendingType,
    savedStage,
    loopType,

    isCapture: loopType === "capture",
    isConfirmation: loopType === "confirm",
    isInterrupt: loopType === "interrupt",
    isOverlay: loopType === "overlay",
    isExport: loopType === "export",
  };
}

/**
 * Read-only structural helper.
 * Returns true if the Parent Anchor owner stage matches
 * the requested structural stage.
 */
function isParentAnchorInStage(state, structuralStage) {
  return getParentAnchorOwnerStage(state) === structuralStage;
}

/**
 * Read-only structural helper.
 * Returns true if the Parent Anchor loop type matches
 * the requested loop classification.
 */
function isParentAnchorLoopType(state, loopType) {
  return getParentAnchorLoopType(state) === loopType;
}

// ---------------------
// CHILD ANCHOR ADAPTERS
// ---------------------
// In the sandbox phase, child anchors are a thin structural seam only.
// They do not own progression, pending-state routing, or loop control.
// The runtime engine remains owned by getStage(), computeNextQuestion(),
// and updateStateFromStudent().

// ---------------------
// CHILD ANCHOR CONTRACT
// ---------------------
// Child anchors are frame-specific instructional adapters that plug into
// the Parent Anchor structure.
//
// Parent Anchor owns:
// - invariant structure
// - structural stage ownership
// - loop ownership
//
// Child Anchor owns:
// - frame-specific instructional language
// - mapping structural slots to frame language
//   (for example: parentItems -> causes / themes / main ideas)
// - prompt wording and coaching tone
// - frame-specific parsing hooks, if needed
// - frame-specific sufficiency rules for a structural slot, if needed
//
// In this phase, child anchors should not replace Parent Anchor stage
// control. They translate Parent Anchor structure into the frame's
// instructional voice and behavior.

// ---------------------
// CHILD ADAPTER SHAPE (MINIMUM EXPECTATION)
// ---------------------
// Each child anchor should provide:
//
// - id:
//   unique frame identifier (e.g., "causeEffect", "themes")
//
// - getLabel(structuralStage, state):
//   returns the student-facing label for a structural stage
//   (e.g., "Cause", "Theme", "Main Idea")
//
// - getPromptTerm(structuralStage, state):
//   returns the term used inside prompts for that stage
//   (keeps Kaw's instructional language frame-specific)
//
// Optional (frame-specific, not always needed):
// - parsing behavior for student input
// - sufficiency rules if they differ from defaults
// - export labeling adjustments
//
// Child adapters do NOT:
// - control progression
// - control loop ownership
// - override Parent Anchor stage decisions

const CauseEffectFrame = {
  id: "causeEffect",
  
  getLabel(structuralStage, state) {
    switch (structuralStage) {
      case "purpose":
        return "Purpose";
      case "frameType":
        return "Frame Type";
      case "keyTopic":
        return "Key Topic";
      case "isAbout":
      case "isAboutConfirm":
        return "Is About";
      case "parentItems":
      case "parentItemsConfirm":
        return "Cause";
      case "detailsLoop":
      case "detailsConfirmLoop":
        return state?.frameMeta?.purpose === "read"
          ? "Text Evidence"
          : "Supporting Detail";
      case "soWhat":
      case "soWhatConfirm":
        return "So What";
      case "export":
        return "Export";
      default:
        return structuralStage;
    }
  },

  getPromptTerm(structuralStage, state) {
    return this.getLabel(structuralStage, state).toLowerCase();
  },
};

const ThemesFrame = {
  id: "themes",

 getLabel(structuralStage, state) {
  switch (structuralStage) {
    case "purpose":
      return "Purpose";
    case "frameType":
      return "Frame Type";
    case "keyTopic":
      return "Key Topic";
    case "isAbout":
    case "isAboutConfirm":
      return "Is About";
    case "parentItems":
    case "parentItemsConfirm":
      return "Main Idea";
    case "detailsLoop":
    case "detailsConfirmLoop":
      return "Details";
    case "soWhat":
    case "soWhatConfirm":
      return "So What";
    case "export":
      return "Export";
    default:
      return structuralStage;
  }
},

  getPromptTerm(structuralStage, state) {
    switch (structuralStage) {
      case "parentItems":
        return "Main Idea";
      case "detailsLoop":
        return "evidence and explanation";
      default:
        return this.getLabel(structuralStage, state).toLowerCase();
    }
  }
};

function evaluateThemesIsAbout(state, response) {
  const keyTopic = cleanText(state?.frame?.keyTopic || "").toLowerCase();
  const text = cleanText(response || "");
  const lower = text.toLowerCase();

  if (!text) {
    return {
      sufficient: false,
      category: "vague",
      feedback: "That is a good start, but your theme needs to be more specific.",
      revisionPrompt: "Can you revise it so it clearly shows a message about life?",
      scaffold: null,
    };
  }

  const wordCount = lower.split(/\s+/).filter(Boolean).length;

  // Topic only
  if (wordCount <= 3 && /^[a-z\s]+$/.test(lower) && !lower.includes(" is ")) {
    return {
      sufficient: false,
      category: "topic",
      feedback: "That sounds more like a topic than a theme.",
      revisionPrompt: "Can you turn it into a full idea about what it says about life?",
      scaffold: 'You might start with: "This shows that..."',
    };
  }

  // Advice / moral
  if (
    lower.startsWith("you should") ||
    lower.startsWith("people should") ||
    lower.includes("should always") ||
    lower.includes("should never")
  ) {
    return {
      sufficient: false,
      category: "advice",
      feedback: "That sounds more like advice than a message about life.",
      revisionPrompt: "Can you revise it so it states a message about life instead of telling what someone should do?",
      scaffold: 'You might start with: "This shows that..."',
    };
  }

  // Summary
  if (
    lower.startsWith("this story is about") ||
    lower.startsWith("this text is about") ||
    lower.startsWith("the story is about") ||
    lower.startsWith("the text is about") ||
    lower.includes("character") ||
    lower.includes("the author shows how")
  ) {
    return {
      sufficient: false,
      category: "summary",
      feedback: "That sounds more like a summary of what happens than a message about life.",
      revisionPrompt: "Can you revise it so it focuses on the message, not just what happens?",
      scaffold: 'You might start with: "This shows that..."',
    };
  }

  // Vague
  if (
    lower === "friendship is important" ||
    lower === "love is important" ||
    lower === "courage is important" ||
    lower.endsWith("is important") ||
    lower.endsWith("matters")
  ) {
    return {
      sufficient: false,
      category: "vague",
      feedback: "That is a good start, but your theme needs to be more specific.",
      revisionPrompt: "Can you make your idea more specific about what it shows about life?",
      scaffold: 'You might start with: "This shows that..."',
    };
  }

  // // Misaligned (temporarily disabled — too strict for first pass)
// if (keyTopic && !lower.includes(keyTopic)) {
//   return {
//     sufficient: false,
//     category: "misaligned",
//     feedback: "That idea does not clearly connect to your Key Topic.",
//     revisionPrompt: "Can you revise your statement so it shows what this topic says about life?",
//     scaffold: `Try connecting it back to "${state?.frame?.keyTopic || "your topic"}".`,
//   };
// }

  return {
    sufficient: true,
    category: null,
    feedback: null,
    revisionPrompt: null,
    scaffold: null,
  };
}

function evaluateThemesSoWhat(state, response) {
  const text = cleanText(response || "");
  const lower = text.toLowerCase();
  const isAbout = cleanText(state?.frame?.isAbout || "").toLowerCase();
  const topic = cleanText(state?.frame?.keyTopic || "").toLowerCase();

  if (!text || text.length < 10) {
    return {
      sufficient: false,
      category: "too_short",
      feedback: "That’s a start, but your response is too short.",
      revisionPrompt: "Can you explain more clearly what someone should understand from this theme?",
      scaffold: 'You might start with: "This theme matters because..."',
    };
  }

  if (isAbout && lower.includes(isAbout.slice(0, 20))) {
    return {
      sufficient: false,
      category: "restate",
      feedback: "That mostly repeats your message instead of explaining why it matters.",
      revisionPrompt: "Can you go beyond repeating the message and explain what someone should understand from it?",
      scaffold: 'You might start with: "This matters because..."',
    };
  }

  const vaguePatterns = [
    "this is important",
    "it is important"
  ];

  if (vaguePatterns.some(p => lower.startsWith(p))) {
    return {
      sufficient: false,
      category: "vague",
      feedback: "That’s a good start, but it’s too general.",
      revisionPrompt: "What lesson or understanding should someone take away from this theme?",
      scaffold: 'You might start with: "This theme matters because it shows that..."',
    };
  }

  if (topic && lower.includes(topic)) {
    return {
      sufficient: false,
      category: "summary",
      feedback: "That sounds more like returning to the topic than explaining why the theme matters.",
      revisionPrompt: "Can you explain the bigger lesson or takeaway instead?",
      scaffold: 'You might start with: "This teaches that..."',
    };
  }

  return {
    sufficient: true,
    category: null,
    feedback: null,
    revisionPrompt: null,
    scaffold: null,
  };
}
// ---------------------
// CHILD FRAME REGISTRY
// ---------------------
// Central place to register all child anchors.
// This allows new frames to plug into the system
// without modifying Parent Anchor logic.

const CHILD_FRAMES = {
  causeEffect: CauseEffectFrame,
  themes: ThemesFrame
};

function getFrameAdapter(state) {
  const frameType =
    state?.frameMeta?.frameType || state?.frameType || "causeEffect";

  return CHILD_FRAMES[frameType] || CHILD_FRAMES["causeEffect"];
}

function getFrameLabel(state, structuralStage) {
  const adapter = getFrameAdapter(state);
  return adapter.getLabel(structuralStage, state);
}

function getFramePromptTerm(state, structuralStage) {
  const adapter = getFrameAdapter(state);
  return adapter.getPromptTerm(structuralStage, state);
}

// ---------------------
// PARENT ANCHOR OBSERVATION HELPERS
// ---------------------
// These helpers are read-only and sandbox-only in purpose.
// They exist to make the engine easier to inspect structurally.
// They must not be used to alter routing or progression behavior.

function getParentAnchorDisplayLabel(state) {
  const context = getParentAnchorContext(state);
  return getFrameLabel(state, context.ownerStructuralStage);
}

function getParentAnchorObservation(state) {
  const context = getParentAnchorContext(state);
  const frameType = state?.frameMeta?.frameType || "";
  const purpose = state?.frameMeta?.purpose || "";
  const ownerLabel = getFrameLabel(state, context.ownerStructuralStage);
  const stageLabel = getFrameLabel(state, context.structuralStage);

  return {
    ...context,
    frameType,
    purpose,
    ownerLabel,
    stageLabel,

    summary: `${context.ownerStructuralStage} | ${context.loopType} | ${ownerLabel}`,
  };
}

function buildMiniQuestion(state) {
  let stage = state?.pending?.stage || getStage(state);

  if (state?.pending?.type === "collectAnotherMainIdea") {
    stage = "mainIdeas";
  }

  const baseStage = getBaseStage(stage);

  // ---------------------
  // PARENT ANCHOR LABEL (READ-ONLY)
  // ---------------------
  // This derives the structural label for the current stage
  // but does NOT control prompt selection in this phase.

  const paContext = getParentAnchorContext(state);
  const paStage = paContext.ownerStructuralStage;
  const framePromptTerm = getFramePromptTerm(state, paContext.ownerStructuralStage);
  const keyTopic = state.frame?.keyTopic || "your topic";
  const effect = state.frame?.effect || state.frame?.isAbout || "the effect";
  const isCE = state.frameMeta?.frameType === "causeEffect";

if (stage === "purpose") {
  return (
    "How will you use this Frame?\n" +
    "1) Study — think through and organize your ideas\n" +
    "2) Write — build a claim and support it\n" +
    "3) Read — pull key ideas from a text or source\n\n" +
    "Reply with 1, 2, or 3."
  );
}

  if (stage === "keyTopic") {
    return `Your frame begins with the Key Topic.\n\nIn just a few words, what is the name of the topic you are exploring?`;
  }

  if (stage === "isAbout") {
    if (isCE) {
      return `Your frame starts with this Key Topic:\n\n"${keyTopic}"\n\nNow think about what main effect this topic leads to.\n\nWhat effect is your frame trying to explain?`;
    }
    return `Your frame starts with this Key Topic:\n\n"${keyTopic}"\n\nNow think about the deeper meaning of this topic.\n\nWhat message about life or people might this topic be showing?`;
  }

 if (stage === "mainIdeas") {
  if (isCE) {
    return `You identified this effect:\n\n"${effect}"\n\nWhat are the main causes that lead to this effect?`;
  }

  return `You identified this message about life:\n\n"${state.frame?.isAbout || "your theme"}"\n\nWhat is one example, idea, or moment that helps show this message?`;
}

 const isDetailsStage = isParentAnchorInStage(state, "detailsLoop");
  if (isDetailsStage) {
  const idx = Number(stage.split(":")[1]);
  const mi = getIdeaList(state)[idx] || "this main idea";

  if (isCE) {
    return `You identified this cause:\n\n"${mi}"\n\nNow think about how that leads to this effect:\n\n"${effect}"\n\nWhat detail or example shows how this cause produces the effect?`;
  }

  return `You identified this main idea:\n\n"${mi}"\n\nNow think about how it connects to this message:\n\n"${state.frame?.isAbout || "your theme"}"\n\nWhat specific detail, example, or explanation helps show this theme in action?`;
}

  if (stage === "soWhat") {
    if (isCE) {
      return `Your frame explains why this happens:\n\n"${effect}"\n\nNow think about why this matters.\n\nWhat should people really understand about this topic?`;
    }
    return `Your frame is showing this message about life:\n\n"${state.frame?.isAbout || "your theme"}"\n\nNow think beyond this one example or text.\n\nWhat should people really understand about life or people because of this theme?`;
  }

  return `What part of your ${framePromptTerm} feels easiest to improve right now: Key Topic, Is About, Main Ideas, Details, or So What?`;
}

function normalizeStuckChoice(msg) {
  const t = cleanText(msg).toLowerCase();
  if (!t) return null;

  if (t === "1" || t.startsWith("1 ")) return "1";
  if (t === "2" || t.startsWith("2 ")) return "2";
  if (t === "3" || t.startsWith("3 ")) return "3";
  if (t === "4" || t.startsWith("4 ")) return "4";

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
      frameType: "", // causeEffect|themes|reading|general
    },
    frame: {
      keyTopic: "",
      isAbout: "",
      causes: [],
      effect: "",
      parentItems: [],
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
    flags: {
      exportOffered: false,
      exportChoice: null,
    },
    skips: [],
  };
}

function normalizeIncomingState(raw) {
  const s = raw && typeof raw === "object" ? raw : {};
  const base = defaultState();

  const frame = s.frame && typeof s.frame === "object" ? s.frame : {};

  base.frame.keyTopic = cleanText(frame.keyTopic || s.keyTopic || "");
  base.frame.isAbout = cleanText(frame.isAbout || s.isAbout || "");

  base.frame.causes = Array.isArray(frame.causes)
  ? frame.causes.map(cleanText).filter(Boolean)
  : cleanText(frame.cause || s.cause || "")
    ? [cleanText(frame.cause || s.cause || "")]
    : [];

  base.frame.effect = cleanText(frame.effect || s.effect || "");

  // Ensure effect is derived from isAbout when missing
if (!base.frame.effect && base.frame.isAbout) {
  base.frame.effect = base.frame.isAbout;
}

base.frame.parentItems = Array.isArray(frame.parentItems)
  ? frame.parentItems.map(cleanText).filter(Boolean)
  : [];

base.frame.mainIdeas = Array.isArray(frame.mainIdeas)
  ? frame.mainIdeas.map(cleanText).filter(Boolean)
  : [];

  if (Array.isArray(frame.details)) {
    base.frame.details = frame.details.map((bucket) => (Array.isArray(bucket) ? bucket.map(cleanText).filter(Boolean) : []));
  } else if (frame.details && typeof frame.details === "object") {
    // legacy object form
   const obj = frame.details;
const rawFrameType =
  s?.frameMeta && typeof s.frameMeta === "object"
    ? cleanText(s.frameMeta.frameType || "")
    : "";

const ideaSeed =
  rawFrameType === "causeEffect"
    ? base.frame.causes
    : base.frame.parentItems;

base.frame.details = ideaSeed.map((mi) => {
  const bucket = obj[mi];
  return Array.isArray(bucket) ? bucket.map(cleanText).filter(Boolean) : [];
});
  } else {
    base.frame.details = [];
  }

  base.frame.soWhat = cleanText(frame.soWhat || s.soWhat || "");

  const frameMeta = s.frameMeta && typeof s.frameMeta === "object" ? s.frameMeta : {};
  base.frameMeta.purpose = cleanText(frameMeta.purpose || "") || "";
  base.frameMeta.frameType = cleanText(frameMeta.frameType || "") || "";

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

// ensure detail buckets exist for each idea
const isCE = base.frameMeta?.frameType === "causeEffect";

const ideaSeed = isCE
  ? (base.frame.causes || [])
  : (base.frame.parentItems || []);

for (let i = 0; i < ideaSeed.length; i++) {
  if (!Array.isArray(base.frame.details[i])) {
    base.frame.details[i] = [];
  }
}
  
return base;
}
  
function ensureBuckets(s) {
  if (!Array.isArray(s.frame.details)) s.frame.details = [];

  if (s.frameMeta?.frameType === "causeEffect") {
    if (!Array.isArray(s.frame.causes)) s.frame.causes = [];

    for (let i = 0; i < s.frame.causes.length; i++) {
      if (!Array.isArray(s.frame.details[i])) {
        s.frame.details[i] = [];
      }
    }
    return;
  }

  if (!Array.isArray(s.frame.parentItems)) s.frame.parentItems = [];

  for (let i = 0; i < s.frame.parentItems.length; i++) {
    if (!Array.isArray(s.frame.details[i])) {
      s.frame.details[i] = [];
    }
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
  const ideas = getIdeaList(s);

  if (!s.frame.keyTopic) return false;
  if (!s.frame.isAbout) return false;
  if (ideas.length < 2) return false;

  for (let i = 0; i < ideas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (arr.length < 2) return false;
  }

  if (!s.frame.soWhat) return false;
  return true;
}

// ---------------------
// EXPORT
// ---------------------
function buildFrameText(s) {
  const lines = [];
  const isCE = s.frameMeta?.frameType === "causeEffect";

  lines.push(`KEY TOPIC: ${s.frame.keyTopic}`);
  lines.push(`IS ABOUT: ${s.frame.isAbout}`);
  // Cause & Effect export (supports multiple causes)
if (isCE) {
  const causes = s.frame.causes || [];

  if (causes.length) {
    causes.forEach((c, i) => {
      lines.push(`CAUSE ${i + 1}: ${c}`);
    });
  }

  if (s.frame.effect) {
    lines.push(`EFFECT: ${s.frame.effect}`);
  }
} else {
  // fallback for non-CE frames (unchanged behavior)
  if (s.frame.cause || s.frame.effect) {
    lines.push(`CAUSE: ${s.frame.cause || ""}`);
    lines.push(`EFFECT: ${s.frame.effect || ""}`);
  }
}

// Surface-labeling only (structure unchanged)
lines.push(isCE ? "CAUSES + SUPPORTING DETAILS:" : "MAIN IDEAS + SUPPORTING DETAILS:");

const ideas = getIdeaList(s);
ideas.forEach((mi, i) => {
  lines.push(`${isCE ? "Cause" : "Main Idea"} ${i + 1}: ${mi}`);
  const details = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
  const detailLabel = s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Detail";
  details.forEach((d, k) => lines.push(`  - ${detailLabel} ${k + 1}: ${d}`));
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
// CAUSE/EFFECT (WRITE) helper
// ---------------------
function parseCauseEffectFromLeadsTo(msg) {
  const raw = cleanText(msg);
  const lower = raw.toLowerCase();
  const key = "leads to";
  const idx = lower.indexOf(key);
  if (idx === -1) return null;

  const leftRaw = raw.slice(0, idx);
  const rightRaw = raw.slice(idx + key.length);

  const cause = leftRaw
    .replace(/^(this|the)\s+(key\s+)?topic\s+is\s+about\s+how\s+/i, "")
    .replace(/^\s*how\s+/i, "")
    .trim()
    .replace(/[.?!]+$/g, "");

  const effect = rightRaw.trim().replace(/[.?!]+$/g, "");

  return { cause, effect };
}

function applyIsAboutCapture(s, msg) {
  if (s.frameMeta?.frameType === "themes") {
    const result = evaluateThemesIsAbout(s, msg);

    if (!result.sufficient) {
      s.pending = {
        type: "reviseThemesIsAbout",
        category: result.category,
        feedback: result.feedback,
        revisionPrompt: result.revisionPrompt,
        scaffold: result.scaffold,
      };
      return s;
    }
  }

  // Write + causeEffect must include "leads to" and we parse/store cause/effect
  if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
    const parsed = parseCauseEffectFromLeadsTo(msg);
    if (!parsed) {
      s.pending = { type: "needWriteCauseEffectStem" };
      return s;
    }
    if (parsed.cause) s.frame.causes = [parsed.cause];
    if (parsed.effect) s.frame.effect = parsed.effect;
  }

  // Study/Read + causeEffect: accept either an effect-only answer
  // or a full "X leads to Y" relationship, but store them cleanly.
  if (
    s.frameMeta?.frameType === "causeEffect" &&
    (s.frameMeta?.purpose === "study" || s.frameMeta?.purpose === "read")
  ) {
    const parsed = parseCauseEffectFromLeadsTo(msg);

    if (parsed?.effect) {
      s.frame.effect = parsed.effect;
      s.frame.isAbout = `how ${parsed.cause} leads to ${parsed.effect}`;
      s.pending = { type: "confirmIsAbout" };
      return s;
    }

    const effectOnly = cleanText(msg).replace(/[.?!]+$/g, "");
    s.frame.effect = effectOnly;
    s.frame.isAbout = `how ${s.frame.keyTopic} leads to ${effectOnly}`;
    s.pending = { type: "confirmIsAbout" };
    return s;
  }

  s.frame.isAbout = msg;
  s.pending = { type: "confirmIsAbout" };
  return s;
}

// ---------------------
// PROGRESSION
// ---------------------
function computeNextQuestion(state) {
  const s = state;
  ensureBuckets(s); //

  const paContext = getParentAnchorContext(s);
  const paStage = paContext.ownerStructuralStage;
  
  // ---------------------
  // PARENT ANCHOR OBSERVATION HOOK (SANDBOX ONLY)
  // ---------------------
  // Leave this disabled until you are intentionally validating sandbox flows.
  // This hook exists so Parent Anchor can explain the engine in motion
  // without becoming part of the engine.
  //
  // Gated sandbox-only observation:
if (s?.settings?.debugParentAnchor) {
  const paObs = getParentAnchorObservation(s);
  const isInDetails = paContext.ownerStructuralStage === "detailsLoop";

  const stage = s.pending?.stage || getStage(s);
  const paStage = paContext.ownerStructuralStage;
  const baseStage = getBaseStage(stage);
  const engineIsDetails = baseStage === "details";

  const isAligned = isInDetails === engineIsDetails;

  console.log("[PA OBS]", paObs.summary, {
    isInDetails,
    engineIsDetails,
    isAligned,
    ...paObs
  });
}
  
  if (s.pending?.type === "confirmLanguageSwitch") {
    const candNative = s.pending?.candidateNativeName || s.pending?.candidateName || "that language";
    const candName = s.pending?.candidateName || "that language";
    return `I notice you’re writing in ${candName}. Would you like to continue in ${candNative}? (yes/no)`;
  }

  if (s.pending?.type === "reviseKeyTopic") {
  return `${s.pending.feedback}\n\nWhat is your Key Topic? (2–5 words)`;
}
  
  if (s.pending?.type === "stuckConfirm") return "Sounds like you’re stuck. Want a quick help move? (yes/no)";

if (s.pending?.type === "stuckMenu") {

  const intro = s.pending?.retryFromMini
    ? "No problem — that smaller question didn’t help enough yet. Let’s try a different help move.\n\n"
    : "";

  return (
    intro +
    "Pick a quick help move: " +
    "1) Check directions  " +
    "2) Re-read source/notes  " +
    "3) I’ll ask a smaller question for this step  " +
    "4) Skip for now and come back.  " +
    "Which one (1–4)?"
  );
}

if (s.pending?.type === "stuckReask") {
  const stage = s.pending.stage || getStage(s);
  const frameType = s?.frameMeta?.frameType || "";

  const keyTopic =
    s?.frame?.keyTopic ||
    s?.frame?.topic ||
    "";

  const effect =
    s?.frame?.effect ||
    s?.frame?.centralEffect ||
    s?.frame?.result ||
    "";

  const theme =
    s?.frame?.isAbout ||
    "";

  const ideas = getIdeaList(s).filter(Boolean);

  let selectedMainIdea = "";
  if (typeof stage === "string" && stage.startsWith("details:")) {
    const rawIndex = stage.split(":")[1];
    const idx = Number(rawIndex);
    selectedMainIdea = Number.isInteger(idx) ? (ideas[idx] || "") : "";
  } else if (ideas.length > 0) {
    selectedMainIdea = ideas[ideas.length - 1] || "";
  }

  const isCE = frameType === "causeEffect";
  const isThemes = frameType === "themes";

  if (s.pending.mode === "directions") {
    if (typeof stage === "string" && stage.startsWith("details:")) {
      if (isCE && selectedMainIdea && effect) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Details part of the Frame.\n\nYour cause is:\n"${selectedMainIdea}"\n\nWhat detail could help explain how that leads to ${effect}?`;
      }

      if (isThemes && selectedMainIdea) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Details part of the Frame.\n\nYour theme support is:\n"${selectedMainIdea}"\n\nWhat specific detail, example, or explanation could help show how this connects to your message about life?`;      
      }

      if (isCE) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Details part of the Frame.\n\nWhat detail could help explain your cause more clearly?`;
      }

      if (isThemes) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Details part of the Frame.\n\nWhat specific detail, example, or explanation could help show your message about life more clearly?`;
      }

      return `Let's check what this step is asking you to do.\n\nRight now you're working on the Details part of the Frame.\n\nWhat detail could help explain this part of your frame more clearly?`;
    }

    if (stage === "mainIdeas") {
      if (isCE && effect && keyTopic) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nYou're explaining why ${effect} happens in ${keyTopic}.\n\nWhat might be another cause that could lead to ${effect}?`;
      }

      if (isCE && effect) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nYou're explaining why ${effect} happens.\n\nWhat might be another cause that could lead to that effect?`;
      }

      if (isThemes && theme && keyTopic) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nYou're showing this message about life:\n"${theme}"\n\nWhat is one specific example, idea, or moment that helps show that message about life?`;
        
      }

      if (isThemes && theme) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nYou're showing this message about life:\n"${theme}"\n\nWhat idea, example, or moment could help show that message?`;
      }

      if (isCE) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nWhat might be another major cause to add?`;
      }

      if (isThemes) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nWhat idea, example, or moment could help show your theme?`;
      }

      return `Let's check what this step is asking you to do.\n\nRight now you're working on the Main Ideas part of the Frame.\n\nWhat might be another important idea to add?`;
    }

    if (stage === "soWhat") {
      if (isCE && effect && keyTopic) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nYou've identified causes that lead to ${effect} in ${keyTopic}.\n\nWhat larger idea should someone understand after seeing those causes?`;
      }

      if (isCE && effect) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nYou've identified causes that lead to ${effect}.\n\nWhat larger idea should someone understand after seeing those causes?`;
      }

      if (isThemes && theme && keyTopic) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nYou've shown this message about life:\n"${theme}"\n\nWhat is the most important idea people should understand about life or people because of this message?`;
      }

      if (isThemes && theme) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nYou've shown this message about life:\n"${theme}"\n\nWhat is the most important idea people should understand about life or people because of this message?`;
      }

      if (isCE) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nWhat larger idea or takeaway should someone understand?`;
      }

      if (isThemes) {
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nWhat should people understand about life or people based on your message?`;
      }

      return `Let's check what this step is asking you to do.\n\nRight now you're working on the So What part of the Frame.\n\nWhat larger idea or takeaway should someone understand?`;
    }

    return `Let's check what this step is asking you to do.\n\n${s.pending.resumeQuestion}`;
  }

  if (s.pending.mode === "reread") {
    if (typeof stage === "string" && stage.startsWith("details:")) {
      if (isCE && selectedMainIdea && effect) {
        return `Look back at your notes or source.\n\nFind a sentence that relates to this cause:\n"${selectedMainIdea}"\n\nDoes that sentence help explain how this cause leads to ${effect}?`;
      }

      if (isThemes && selectedMainIdea) {
        return `Look back at your notes or source.\n\nFind a sentence that relates to this main idea:\n"${selectedMainIdea}"\n\nCould that sentence help show how this connects to your theme?`;
      }

      if (isCE) {
        return `Look back at your notes or source.\n\nDo you see a sentence that could help explain your cause more clearly?`;
      }

      if (isThemes) {
        return `Look back at your notes or source.\n\nDo you see a sentence, example, or detail that could help show your theme more clearly?`;
      }

      return `Look back at your notes or source.\n\nDo you see a sentence or phrase there that could help you answer this step?`;
    }

    if (stage === "mainIdeas") {
      if (isCE && effect && keyTopic) {
        return `Look back at your notes or source about ${keyTopic}.\n\nDo you see a sentence that explains another reason why ${effect} happens?\n\nCould that idea become another cause in your Frame?`;
      }

      if (isCE && effect) {
        return `Look back at your notes or source.\n\nDo you see a sentence that explains another reason why ${effect} happens?\n\nCould that idea become another cause in your Frame?`;
      }

      if (isThemes && theme && keyTopic) {
        return `Look back at your notes or source about ${keyTopic}.\n\nDo you see an example, idea, or moment that helps show this message about life:\n"${theme}"\n\nCould that become another theme support in your Frame?`;
      }

      if (isThemes && theme) {
        return `Look back at your notes or source.\n\nDo you see an example, idea, or moment that helps show this message about life:\n"${theme}"\n\nCould that become another theme support in your Frame?`;
      }

      if (isCE) {
        return `Look back at your notes or source.\n\nDo you see another important reason or cause you could add to your Frame?`;
      }

      if (isThemes) {
        return `Look back at your notes or source.\n\nDo you see another example, idea, or moment you could add to help show your theme?`;
      }

      return `Look back at your notes or source.\n\nDo you see another important idea you could add to your Frame?`;
    }

    if (stage === "soWhat") {
      if (isCE && effect && keyTopic) {
        return `Look back at your notes or source about ${keyTopic}.\n\nDo you see a sentence that connects the causes you found to the bigger issue?\n\nCould that idea help you write the takeaway for your Frame?`;
      }

      if (isCE && effect) {
        return `Look back at your notes or source.\n\nDo you see a sentence that connects the causes you found to the bigger issue?\n\nCould that idea help you write the takeaway for your Frame?`;
      }

      if (isThemes && theme && keyTopic) {
        return `Look back at your notes or source about ${keyTopic}.\n\nDo you see a sentence, idea, or moment that helps show why this message matters:\n"${theme}"\n\nCould that help you explain the bigger takeaway for your Frame?`;
      }

      if (isThemes && theme) {
        return `Look back at your notes or source.\n\nDo you see a sentence, idea, or moment that helps show why this message matters:\n"${theme}"\n\nCould that help you explain the bigger takeaway for your Frame?`;
      }

      if (isCE) {
        return `Look back at your notes or source.\n\nDo you see a sentence that could help you write the takeaway for your Frame?`;
      }

      if (isThemes) {
        return `Look back at your notes or source.\n\nDo you see something there that could help you explain why your theme matters?`;
      }

      return `Look back at your notes or source.\n\nIs there a sentence or phrase there that could help you answer this step?`;
    }

    return `Look back at your notes or source.\n\nIs there a sentence or phrase there that could help you answer this step?`;
  }

  return s.pending.resumeQuestion;
}
 
  if (s.pending?.type === "stuckNudge") {
  const tone = s.pending.tone || "neutral";
  const ack = tone === "frustration" ? "That can feel frustrating. " : tone === "resistance" ? "I hear you. " : "";
  const nudge = (s.pending.nudgeText || "").toString().trim();
  return `${ack}${nudge}\n\nThen answer: ${s.pending.resumeQuestion}`;
}

  if (s.pending?.type === "stuckMini") return s.pending.miniQuestion || buildMiniQuestion(s);

  if (s.pending?.type === "stuckSkip")
    return "Got it — we’ll come back to this. Want to try the next step now? (yes/no)";

  if (s.pending?.type === "needWriteCauseEffectStem") {
    return 'That’s a strong start. Can you restate it as a clear cause-and-effect relationship? Try: "This topic is about how ___ leads to ___."';
  }

  if (s.pending?.type === "writeNeedEvidenceDetail") {
    const i = Number(s.pending.index);
    const mi = getIdeaList(s)[i] || "this Cause";
    const eff = s.frame.effect || "the effect";
    const mech = cleanText(s.pending.mechanism || "");
    const ctx = mech ? `You're explaining how it works: "${mech}". ` : "";
    return `${ctx}Can you add one concrete piece of evidence (example, fact, quote, or statistic) that shows how "${mi}" connects to ${eff}?`;
}
if (s.pending?.type === "reviseIsAbout") {
  const topic = (s.frame.keyTopic || "").trim();

  return "Let's revise the relationship in the frame.\n\n" +
    "Right now we have the topic:\n\n" +
    topic + "\n\n" +
    "Try rewriting the cause-and-effect relationship more clearly.\n\n" +
    "Use the pattern:\n" +
    "how ___ leads to ___";
}
   
// --- Return to skipped work before confirmation ---
if (
  Array.isArray(s.skips) &&
  s.skips.length > 0 &&
  ["confirmIsAbout", "confirmMainIdeas", "confirmDetails", "confirmSoWhat"].includes(s.pending?.type)
) {

  const skipped = s.skips[0];

  let label = "a part of the frame";
if (skipped.stage === "mainIdeas") {
  const nextCauseNumber =
  (s.frameMeta?.frameType === "causeEffect"
    ? (s.frame.causes || []).length
    : (s.frame.mainIdeas || []).length) + 1;
  label = `${s.frameMeta?.frameType === "causeEffect" ? "Cause" : "Main Idea"} ${nextCauseNumber}`; 
}

if (skipped.stage === "soWhat") label = "the So What statement";

if (skipped.stage?.startsWith("details")) {
  const idx = Number(skipped.stage.split(":")[1]);
  const labelBase = s.frameMeta?.frameType === "causeEffect" ? "Cause" : "Main Idea";
  const detailLabel = s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Detail";
  label = `${detailLabel} for ${labelBase} ${idx + 1}`;
}

  // Keep the skip for now; remove it after the student completes this stage.

    const intro = `Before we confirm your thinking and move on, let's return to the part we skipped earlier: ${label}.\n\n`;

  s.pending = {
    type: "stuckMini",
    stage: skipped.stage,
    miniQuestion: buildMiniQuestion({
      ...s,
      pending: { stage: skipped.stage }
    })
  };

  return intro + s.pending.miniQuestion;
}
 
if (s.pending?.type === "reviseThemesIsAbout") {
  const parts = [
    s.pending.feedback,
    s.pending.revisionPrompt,
    s.pending.scaffold
  ].filter(Boolean);

  return parts.join("\n\n");
}

if (s.pending?.type === "reviseThemesSoWhat") {
  const parts = [
    s.pending.feedback,
    s.pending.revisionPrompt,
    s.pending.scaffold
  ].filter(Boolean);

  return parts.join("\n\n");
}
  
if (s.pending?.type === "confirmIsAbout") {

  // Write + causeEffect
  if (s.frameMeta?.purpose === "write" && s.frameMeta?.frameType === "causeEffect") {
    const raw = (s.frame.isAbout || "").trim();
    const cleaned = raw.replace(/^this topic is about\s+/i, "").replace(/\.$/, "").trim();

    const keyTopic =
      s.frame.keyTopic && s.frame.keyTopic.length
        ? s.frame.keyTopic.charAt(0).toUpperCase() + s.frame.keyTopic.slice(1)
        : s.frame.keyTopic;

    return `Using your ideas, your frame now reads:

Key Topic
${keyTopic}

Is About
${cleaned}

Is that correct, or would you like to revise it?`;
  }

    // Study + causeEffect
  if (s.frameMeta?.purpose === "study" && s.frameMeta?.frameType === "causeEffect") {
    const topic = (s.frame.keyTopic || "").trim();
    const isAbout = (s.frame.isAbout || "").trim().replace(/\.$/, "");

    return `Using your ideas, your frame now reads:

Key Topic
${topic}

Is About
${isAbout}

Is that correct, or would you like to revise it?`;
  }
  
  // Read + causeEffect
  if (s.frameMeta?.purpose === "read" && s.frameMeta?.frameType === "causeEffect") {
    const topic = (s.frame.keyTopic || "").trim();
    const isAbout = (s.frame.isAbout || "").trim().replace(/\.$/, "");

    return `Using your ideas from the text, your frame now reads:

Key Topic
${topic}

Is About
${isAbout}

Is that correct, or would you like to revise it?`;
  }
  
// Themes
if (s.frameMeta?.frameType === "themes") {
  if (s.frameMeta?.purpose === "read") {
    return "Using your ideas from the text, your frame now reads:\n\n" +
      "Key Topic\n" +
      s.frame.keyTopic + "\n\n" +
      "Is About\n" +
      s.frame.isAbout + "\n\n" +
      "Is that correct, or would you like to revise it?";
  }

  return "Using your ideas, your frame now reads:\n\n" +
    "Key Topic\n" +
    s.frame.keyTopic + "\n\n" +
    "Is About\n" +
    s.frame.isAbout + "\n\n" +
    "Is that correct, or would you like to revise it?";
}
  }
  
if (s.pending?.type === "confirmMainIdeas") {
  const isCE = s.frameMeta?.frameType === "causeEffect";

  const lines = getIdeaList(s).map((mi, i) =>
    `${isCE ? "Cause" : "Main Idea"} ${i + 1}: ${mi}`
  ).join("\n");

  const label = isCE ? "Causes" : "Main Ideas";

  return `You have identified the following ${label}:\n${lines}\n\nIs that correct, or would you like to revise one?`;
}
  
if (s.pending?.type === "offerAnotherMainIdea") {
  const isCE = s.frameMeta?.frameType === "causeEffect";
  const count = getIdeaList(s).length;
  const label = isCE ? "Cause" : "Main Idea";
  return `You currently have ${count} ${label}${count > 1 ? "s" : ""}. Would you like to add another ${label}? (yes/no)`;
}

if (s.pending?.type === "collectAnotherMainIdea") {
  const isCE = s.frameMeta?.frameType === "causeEffect";
  const count = getIdeaList(s).length + 1;
  return isCE
    ? `What is another cause that leads to this effect: "${s.frame.effect}"?`
    : `What is Main Idea ${count} that helps explain ${s.frame.keyTopic}?`;
}
  
 if (s.pending?.type === "offerAnotherDetail") {
  const i = Number(s.pending.index);
  const mi = getIdeaList(s)[i] || "";

  const isCE = s.frameMeta?.frameType === "causeEffect";
  const miLabel = isCE ? "Cause" : "Main Idea";
  const dLabel = isCE && s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Detail";

  const count = (s.frame.details?.[i] || []).length;

  return `For ${miLabel} ${i + 1}: "${mi}", you currently have ${count} ${dLabel}${count > 1 ? "s" : ""}. Would you like to add another ${dLabel}? (yes/no)`;
}
  
if (s.pending?.type === "collectAnotherDetail") {
  const i = Number(s.pending.index);
  const mi = getIdeaList(s)[i] || "";  

  const isCE = s.frameMeta?.frameType === "causeEffect";
  const miLabel = isCE ? "Cause" : "Main Idea";
  const dLabel = isCE && s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Detail";

  const count = (s.frame.details?.[i] || []).length + 1;

  return `What is ${dLabel} ${count} for ${miLabel} ${i + 1}: "${mi}"?`;
}
  
  if (s.pending?.type === "confirmDetails") {
    const i = Number(s.pending.index);
    const mi = getIdeaList(s)[i] || "";
    const arr = Array.isArray(s.frame.details?.[i]) ? s.frame.details[i] : [];
    const lineLabel = s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Detail";
    const lines = arr.map((d, k) => `${lineLabel} ${k + 1}: ${d}`).join("\n");
    
    const isCE = s.frameMeta?.frameType === "causeEffect";
    const miLabel = isCE ? "Cause" : "Main Idea";
    const dLabel = isCE && s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Details";

    return `For this ${miLabel}: "${mi}", you identified the following ${dLabel}:\n${lines}\nIs that correct, or would you like to revise one?`;
  }

  if (s.pending?.type === "offerMoreSoWhat") return "Do you want to add one more sentence to your So What? (yes/no)";
  if (s.pending?.type === "collectMoreSoWhat") return "Add one more sentence to your So What:";
  if (s.pending?.type === "confirmSoWhat")
    return `Your So What is: "${s.frame.soWhat}". Is that correct, or would you like to revise it?`;
  if (s.pending?.type === "offerExport") return "Would you like to save or print a copy of your work? (yes/no)";
  if (s.pending?.type === "chooseExportType")
    return "What would you like to save/print: frame, transcript, or both? (frame/transcript/both)";

  // Base progression
  if (!s.frameMeta?.purpose) {
  return (
    "How will you use this Frame?\n" +
    "1) Study — think through and organize your ideas\n" +
    "2) Write — build a claim and support it\n" +
    "3) Read — pull key ideas from a text or source\n" +
    "Reply with 1, 2, or 3."
  );
}

  if (!s.frameMeta?.frameType) {
    return (
      "What kind of thinking are you doing.\n" +
      "1) Explain how/why something happens (Linear & Cause-and-Effect Relationships)\n" +
      "2) Explain a big idea or theme (Framing Themes)\n" +
      "3) Organize ideas from a text or source (Reading Frames)\n" +
      "4) Organize my thinking (General Frame)\n" +
      "Reply with 1, 2, 3, or 4?"
    );
  }

  if (!s.frame.keyTopic) return "What is your Key Topic? (2–5 words)";

  if (!s.frame.isAbout) {
    const pb = getPromptForStage(s, "isAbout");
    return pb || `Finish this sentence: "${s.frame.keyTopic} is about ____."`;
  }

  const ideas = getIdeaList(s);

if (paStage === "parentItems" || ideas.length < 2) {
  let pb = getPromptForStage(s, "mainIdeas");
  const c = ideas.length;

    const isCE = s.frameMeta?.frameType === "causeEffect";
    const label = isCE ? "Cause" : "Main Idea";

    if (pb) {
      if (/^What is one major cause or effect/i.test(pb) || /^What is one major cause/i.test(pb)) {
        const ord = c === 0 ? "first" : c === 1 ? "second" : "next";
        pb = pb.replace(/^What is one/i, `What is your ${ord}`);
      }
      return `${label} ${c + 1}:\n${pb}`;
    }

    const fallback =
      c === 0
        ? `What is your first ${label} that helps explain ${s.frame.keyTopic}?`
        : `What is your second ${label} that helps explain ${s.frame.keyTopic}?`;

    return `${label} ${c + 1}:\n${fallback}`;
  }

  // DETAILS LOOP (CLEANED — no duplicate fallback / brace drift)
   for (let i = 0; i < ideas.length; i++) {
    const mi = ideas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (paStage === "detailsLoop" && arr.length < 2) {
      const pb = getPromptForStage(s, `details:${i}`);
      const detailNum = arr.length + 1; // 1 or 2

      const isCE = s.frameMeta?.frameType === "causeEffect";
      const miLabel = isCE ? "Cause" : "Main Idea";
      const dLabel = isCE && s.frameMeta?.purpose === "read" ? "Text Evidence" : "Supporting Detail";

      if (pb) {
        const base = pb.replace(/\?\s*$/, "");
        return `${miLabel} ${i + 1}: ${mi}\n${dLabel} ${detailNum}:\n\n${base}?`;
      }

      const fallback =
        detailNum === 1
          ? `What is your first ${dLabel} for this ${miLabel}: "${mi}"?`
          : `What is your second ${dLabel} for this ${miLabel}: "${mi}"?`;

      return `${miLabel} ${i + 1}: ${mi}\n${dLabel} ${detailNum}: ${fallback}`;
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
function clearMatchingSkip(state, completedStage) {
  if (!Array.isArray(state.skips) || !state.skips.length) return;

  const first = state.skips[0];
  if (first?.stage === completedStage) {
    state.skips.shift();
  }
}

function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);
  ensureBuckets(s);

  if (!s.frameMeta) s.frameMeta = { purpose: "", frameType: "" };

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

  // ----------------
  // Pending handlers
  // ----------------
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

  // Write-mode cause/effect stem follow-up
  if (s.pending?.type === "needWriteCauseEffectStem") {
    s.pending = null;
    applyIsAboutCapture(s, msg);
    return s;
  }

  // Write-mode evidence guardrail follow-up
  if (s.pending?.type === "writeNeedEvidenceDetail") {
    const idx = Number(s.pending.index);
    if (!Array.isArray(s.frame.details[idx])) s.frame.details[idx] = [];

    const mechanism = cleanText(s.pending.mechanism || "");
    const evidence = msg;

    // Store a combined detail (mechanism + evidence) so the student's thinking is preserved.
    const combined = mechanism ? `${mechanism} (evidence: ${evidence})` : evidence;

    const arr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];
    if (arr.length < 2 && !isNegative(evidence)) {
      s.frame.details[idx] = [...arr, combined];
      clearMatchingSkip(s, `details:${idx}`);
      s.pending = { type: "offerAnotherDetail", index: idx };
      return s;
    }

    s.pending = null;
    return s;
  }

  // STUCK flow
  if (s.pending?.type === "stuckConfirm") {
    const low = msg.toLowerCase().trim();
    if (isAffirmative(low)) {
      s.pending = {
        type: "stuckMenu",
        stage: s.pending.stage || getStage(s),
        tone: s.pending.tone || "neutral",
        resumeQuestion: s.pending.resumeQuestion,
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

  if (s.pending?.type === "stuckMenu") {
    const choice = normalizeStuckChoice(msg);
    if (!choice) return s;

    if (choice === "1") {
      s.pending = {
        type: "stuckReask",
        mode: "directions",
        stage: s.pending.stage || getStage(s),
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }

    if (choice === "2") {
      s.pending = {
        type: "stuckReask",
        mode: "reread",
        stage: s.pending.stage || getStage(s),
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }

    if (choice === "3") {
      const stage = s.pending.stage || getStage(s);
      s.pending = {
        type: "stuckMini",
        stage,
        miniQuestion: buildMiniQuestion(s),
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

  if (s.pending?.type === "stuckReask") {
    s.pending = null;
    // fall through
  }

  if (s.pending?.type === "stuckNudge") {
    s.pending = null;
    // fall through
  }

  if (s.pending?.type === "stuckSkip") {
    const low = msg.toLowerCase().trim();
    if (isAffirmative(low)) {
      s.pending = {
        type: "stuckMini",
        stage: s.pending.stage || getStage(s),
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }
    if (isNegative(low)) {
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

  if (s.pending?.type === "stuckMini") {
    const stage = s.pending.stage || getStage(s);

    if (isStuckMessage(msg)) {
      s.pending = {
        type: "stuckMenu",
        stage,
        tone: detectStuckTone(msg),
        resumeQuestion: s.pending.resumeQuestion,
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
        retryFromMini: true,
      };
      return s;
    }

    if (stage === "purpose") {
      const p = normalizePurpose(msg);
      if (p) s.frameMeta.purpose = p;
      s.pending = null;
      return s;
    }

    if (stage === "frameType") {
      const ft = normalizeFrameTypeSelection(msg);
      if (ft) s.frameMeta.frameType = ft;
      s.pending = null;
      return s;
    }

 if (stage === "keyTopic") {
  const cleaned = cleanText(msg);
  const wc = cleaned.split(/\s+/).filter(Boolean).length;

  if (!s.frame.keyTopic && !isBadKeyTopic(cleaned) && wc >= 2 && wc <= 5) {
    s.frame.keyTopic = cleaned;
    s.pending = null;
    return s;
  }

  s.pending = {
    type: "reviseKeyTopic",
    feedback: getKeyTopicFeedback(cleaned),
  };
  return s;
}

    if (stage === "isAbout") {
      if (!s.frame.isAbout) {
        applyIsAboutCapture(s, msg);
        clearMatchingSkip(s, "isAbout");
        return s;
      }
      s.pending = null;
      return s;
    }

    if (stage === "mainIdeas") {
      s.pending = null;
      return updateStateFromStudent(s, msg);
    }

    if (typeof stage === "string" && stage.startsWith("details:")) {
      s.pending = null;
      return updateStateFromStudent(s, msg);
    }

    if (stage === "soWhat") {
      s.pending = null;
      return updateStateFromStudent(s, msg);
    }

    s.pending = null;
    return s;
  }

  if (s.pending?.type === "reviseThemesIsAbout") {
    s.pending = null;
    applyIsAboutCapture(s, msg);
    return s;
  }

  if (s.pending?.type === "reviseThemesSoWhat") {
    if (!isNegative(msg)) {
      const evaluation = evaluateThemesSoWhat(s, msg);

      if (!evaluation.sufficient) {
  s.pending = {
    type: "reviseThemesSoWhat",
    category: evaluation.category,
    feedback: evaluation.feedback,
    revisionPrompt: evaluation.revisionPrompt,
    scaffold: evaluation.scaffold,
  };
    return s;
}

      // VALID SAVE PATH 2: revision handler save (after successful revision)
      s.frame.soWhat = msg;
      s.pending = { type: "offerMoreSoWhat" };
      return s;
    }

    return s;
  }

  if (s.pending?.type === "confirmIsAbout") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    if (normalized === "revise" || normalized === "change") {
      s.pending = {
        type: s.frameMeta?.frameType === "themes" ? "reviseThemesIsAbout" : "reviseIsAbout",
      };
      return s;
    }

    applyIsAboutCapture(s, msg);
    return s;
  }

  if (s.pending?.type === "reviseIsAbout") {
    applyIsAboutCapture(s, msg);
    return s;
  }

  if (s.pending?.type === "confirmMainIdeas") {
    const normalized = msg.toLowerCase().trim();
    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }
    return s;
  }

  if (s.pending?.type === "offerAnotherMainIdea") {
    const normalized = msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      const count = getIdeaList(s).length;

      if (count >= 5) {
        s.pending = { type: "confirmMainIdeas" };
        return s;
      }

      s.pending = { type: "collectAnotherMainIdea" };
      return s;
    }

    s.pending = { type: "confirmMainIdeas" };
    return s;
  }

  if (s.pending?.type === "collectAnotherMainIdea") {
    if (!isNegative(msg)) {
      const isCE = s.frameMeta?.frameType === "causeEffect";

      if (isCE) {
        if (!Array.isArray(s.frame.causes)) s.frame.causes = [];
        if (!Array.isArray(s.frame.details)) s.frame.details = [];

        s.frame.causes.push(msg);

        if (!Array.isArray(s.frame.details[s.frame.causes.length - 1])) {
          s.frame.details[s.frame.causes.length - 1] = [];
        }
      } else {
        if (!Array.isArray(s.frame.parentItems)) s.frame.parentItems = [];
        if (!Array.isArray(s.frame.details)) s.frame.details = [];

        s.frame.parentItems.push(msg);

        if (!Array.isArray(s.frame.details[s.frame.parentItems.length - 1])) {
          s.frame.details[s.frame.parentItems.length - 1] = [];
        }
      }
    }

    const count = getIdeaList(s).length;

    if (count >= 5) {
      s.pending = { type: "confirmMainIdeas" };
      return s;
    }

    s.pending = { type: "offerAnotherMainIdea" };
    return s;
  }

if (s.pending?.type === "offerAnotherDetail") {
  const normalized = msg.toLowerCase().trim();
  const idx = Number(s.pending.index);
  const arr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];

  if (isAffirmative(normalized)) {
    if (arr.length >= 5) {
      s.pending = { type: "confirmDetails", index: idx };
      return s;
    }
    s.pending = { type: "collectAnotherDetail", index: idx };
    return s;
  }

  if (!normalized) {
    return s;
  }

  if (isNegative(normalized)) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  if (arr.length >= 5) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  if (shouldRequestEvidenceDetail(s, msg)) {
    s.pending = { type: "writeNeedEvidenceDetail", index: idx, mechanism: msg };
    return s;
  }

  s.frame.details[idx] = [...arr, msg];

  const updatedArr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];
  if (updatedArr.length >= 5) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  s.pending = { type: "offerAnotherDetail", index: idx };
  return s;
}

  if (s.pending?.type === "collectAnotherDetail") {
    const idx = Number(s.pending.index);
    if (!Array.isArray(s.frame.details[idx])) s.frame.details[idx] = [];

    if (!isNegative(msg)) {
      if (shouldRequestEvidenceDetail(s, msg)) {
        s.pending = { type: "writeNeedEvidenceDetail", index: idx, mechanism: msg };
        return s;
      }
      s.frame.details[idx] = [...s.frame.details[idx], msg];
    }

    const arr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];
    if (arr.length >= 5) {
      s.pending = { type: "confirmDetails", index: idx };
      return s;
    }

    s.pending = { type: "offerAnotherDetail", index: idx };
    return s;
  }

  if (s.pending?.type === "confirmDetails") {
    const normalized = msg.toLowerCase().trim();
    const idx = Number(s.pending.index);
    const arr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];

    if (isAffirmative(normalized)) {
      s.pending = null;
      return s;
    }

    if (isNegative(normalized)) {
      if (arr.length < 2) {
        s.pending = { type: "collectAnotherDetail", index: idx };
        return s;
      }

      s.pending = null;
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
    // VALID SAVE PATH 3: confirm fallback (user typed revision instead of yes/no)
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
      normalized.includes("both")
        ? "both"
        : normalized.includes("frame")
          ? "frame"
          : normalized.includes("transcript")
            ? "transcript"
            : null;
    s.flags.exportChoice = choice || "both";
    s.pending = null;
    return s;
  }

  // ----------------
  // Normal capture
  // ----------------

  // 1) Extraction rule: "X is about Y"
  const parsed = parseKeyTopicIsAbout(msg);
  if (parsed) {
    if (!s.frame.keyTopic) s.frame.keyTopic = parsed.keyTopic;
    if (!s.frame.isAbout) {
      // If write+c/e, enforce leads-to when capturing isAbout
      applyIsAboutCapture(s, parsed.isAbout);
      clearMatchingSkip(s, "isAbout");
    } else {
      s.pending = { type: "confirmIsAbout" };
    }
    return s;
  }

  // 2) Key Topic capture (plain 2–5 words) — return after capturing
   if (!s.frame.keyTopic) {
    const cleaned = cleanText(msg);
    const wc = cleaned.split(/\s+/).filter(Boolean).length;
  
    if (!isBadKeyTopic(cleaned) && wc >= 2 && wc <= 5) {
      s.frame.keyTopic = cleaned;
      s.pending = null;
      return s;
    }
  
    s.pending = {
      type: "reviseKeyTopic",
      feedback: getKeyTopicFeedback(cleaned),
    };
    return s;
}

  // 3) Is About capture + checkpoint
  if (!s.frame.isAbout) {
    const lowered = msg.toLowerCase().trim();
    if (lowered !== "revise" && lowered !== "change") {
      applyIsAboutCapture(s, msg);
      clearMatchingSkip(s, "isAbout");
    }
    return s;
  }

  // 4) Main Ideas capture
  const ideas = getIdeaList(s);

  if (ideas.length < 2) {
    if (!isNegative(msg)) {
      if (s.frameMeta?.frameType === "causeEffect") {
        if (!Array.isArray(s.frame.causes)) s.frame.causes = [];
        if (!Array.isArray(s.frame.details)) s.frame.details = [];

        s.frame.causes.push(msg);
        clearMatchingSkip(s, "mainIdeas");

        if (!Array.isArray(s.frame.details[s.frame.causes.length - 1])) {
          s.frame.details[s.frame.causes.length - 1] = [];
        }

        s.pending = { type: "offerAnotherMainIdea" };
      } else {
        if (!Array.isArray(s.frame.parentItems)) s.frame.parentItems = [];
        if (!Array.isArray(s.frame.details)) s.frame.details = [];

        s.frame.parentItems.push(msg);
        clearMatchingSkip(s, "mainIdeas");

        if (!Array.isArray(s.frame.details[s.frame.parentItems.length - 1])) {
          s.frame.details[s.frame.parentItems.length - 1] = [];
        }

        if (s.frame.parentItems.length === 2) {
          s.pending = { type: "offerAnotherMainIdea" };
        }
      }
    }
    return s;
  }

  // 5) Details capture
  for (let i = 0; i < ideas.length; i++) {
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];

    if (arr.length < 2) {
      if (!isNegative(msg)) {
        // Prevent "idk / not sure / help" from being saved as a detail
        if (isStuckMessage(msg)) {
          s.pending = {
            type: "stuckConfirm",
            stage: `details:${i}`,
            tone: detectStuckTone(msg),
            resumeQuestion: buildMiniQuestion(s),
            miniQuestion: buildMiniQuestion(s),
          };
          return s;
        }

        if (shouldRequestEvidenceDetail(s, msg)) {
          s.pending = { type: "writeNeedEvidenceDetail", index: i, mechanism: msg };
          return s;
        }

        s.frame.details[i] = [...arr, msg];
        clearMatchingSkip(s, `details:${i}`);
      }

      s.pending = { type: "offerAnotherDetail", index: i };
      return s;
    }
  }

  // 6) So What capture
  if (!s.frame.soWhat) {
    if (!isNegative(msg)) {
      if (s.frameMeta?.frameType === "themes") {
        const evaluation = evaluateThemesSoWhat(s, msg);

        if (!evaluation.sufficient) {
          s.pending = {
            type: "reviseThemesSoWhat",
            category: evaluation.category,
            feedback: evaluation.feedback,
            revisionPrompt: evaluation.revisionPrompt,
            scaffold: evaluation.scaffold,
          };
          return s;
        }
      }

      // VALID SAVE PATH 1: normal So What capture (after evaluation)
      s.frame.soWhat = msg;
      clearMatchingSkip(s, "soWhat");
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

  let incoming = body.state || body.vercelState || body.framing || {};
  let state = normalizeIncomingState(incoming);
    
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
      // STUCK detector (global interrupt) — do not interrupt protected pending flows
      const pendingType = state.pending?.type || null;
      const inProtectedPending =
        pendingType === "confirmLanguageSwitch" ||
        pendingType === "stuckConfirm" ||
        pendingType === "stuckMenu" ||
        pendingType === "stuckReask" ||
        pendingType === "stuckNudge" ||
        pendingType === "stuckMini" ||
        pendingType === "stuckSkip";

      if (!inProtectedPending && isStuckMessage(message)) {
        const stage = getStage(state);
        const resumeQuestion = enforceSingleQuestion(computeNextQuestion(state));

        state.pending = {
          type: "stuckConfirm",
          stage,
          tone: detectStuckTone(message),
          resumeQuestion,
          miniQuestion: buildMiniQuestion(state),
        };

        let reply = enforceSingleQuestion(computeNextQuestion(state));

        if (state.settings.languageLocked && state.settings.language !== "en") {
          reply = await translateQuestionViaLLM(reply, state.settings.languageName || "the target language");
        }

        appendTurn(state, "Student", message);
        appendTurn(state, "Kaw", reply);

        // exports only when complete and not pending
        if (isFrameComplete(state) && !state.pending) {
          const frameText = buildFrameText(state);
          const transcriptText = buildTranscriptText(state);
          const html = buildExportHtml(state);
          state.exports = { frameText, transcriptText, html };
        } else {
          state.exports = null;
        }

        return res.status(200).json({ reply, state });
      }

      state = updateStateFromStudent(state, message);
    }

    let nextQ = computeNextQuestion(state);
    let reply = enforceSingleQuestion(nextQ);

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
