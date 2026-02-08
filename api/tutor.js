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
 * PEDAGOGY (locked):
 * - Ask EXACTLY 1 question per turn
 * - No lecturing, no answers, no correctness confirmation
 * - Teacher controls transitions (no “type done”, no “do you want to move on”)
 *
 * ROUTINE:
 * 1) Key Topic
 * 2) Is About (thesis/topic sentence)
 * 3) Main Ideas (2 required, 3 optional)
 *    - After 2: sufficiency check
 *    - After 3: sufficiency check, then move on on “no”
 * 4) Essential Details (2–3 per Main Idea, nested)
 * 5) So What? (explicit)
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
1) Key Topic (2–5 words: the focus/title)
2) Is About (one short phrase: what the topic is about / claim)
3) Main Ideas (2–3 important ideas that support the claim)
4) Essential Details (2–3 evidence/examples for EACH main idea)
5) So What? (why it matters / significance / conclusion)

ANCHOR RULE:
- If Key Topic is not clear, ask ONLY for Key Topic.
- If Key Topic is clear but Is About is not clear, ask ONLY for Is About.
- Do not ask meaning/definition questions before Key Topic + Is About are captured.

REDIRECT RULE:
If the student asks for an answer or you feel pulled into explaining, redirect to the next Frame step with ONE question.

OUTPUT FORMAT:
Return only the single question. No headings. No bullets.
`.trim();

// ---- Hard caps ----
const MAX_MODEL_TOKENS = 140;
const MAX_CHARS = 420;
const MAX_QUESTIONS = 1;

// --------------------------
// Output enforcement helpers
// --------------------------
function countQuestions(text) {
  return (text.match(/\?/g) || []).length;
}

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
  const tooManySentences = (text.match(/[.!]/g) || []).length >= 2;
  const hasColonList = text.includes(":") && (text.match(/\n/g) || []).length >= 1;
  return hasBadStart || tooManySentences || hasColonList;
}

function enforceHardCap(text) {
  let out = (text || "").trim();

  out = out.replace(/^(kaw companion|tutor|assistant)\s*:\s*/i, "").trim();

  if (countQuestions(out) > MAX_QUESTIONS) {
    const firstQ = out.indexOf("?");
    out = firstQ >= 0 ? out.slice(0, firstQ + 1).trim() : out.trim();
  }

  if (out.length > MAX_CHARS) out = out.slice(0, MAX_CHARS).trim();

  if (!out.includes("?")) {
    out = "What is your Key Topic? (2–5 words.)";
  }

  const lastQ = out.lastIndexOf("?");
  if (lastQ !== out.length - 1) out = out.slice(0, lastQ + 1).trim();

  return out;
}

function socraticFailSafe(original, routedQuestion) {
  const cleaned = (original || "").trim();
  const triggers =
    cleaned.length > MAX_CHARS ||
    looksLikeExplanation(cleaned) ||
    countQuestions(cleaned) !== 1;

  if (!triggers) return enforceHardCap(cleaned);

  const fallback = (routedQuestion || "What is your Key Topic? (2–5 words.)").trim();
  return enforceHardCap(fallback);
}

// -----------------------------------------------
// Frame parsing + sufficiency helpers
// -----------------------------------------------
function cleanText(s) {
  return String(s || "").replace(/\s+/g, " ").trim();
}

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
  const v =
    framing?.mainIdeas ??
    framing?.main_ideas ??
    framing?.frame?.mainIdeas ??
    framing?.frame?.main_ideas ??
    [];
  return Array.isArray(v) ? v : [];
}

/**
 * details shape expected:
 * details: { "0": ["detail1","detail2"], "1": ["detail1"], ... }
 * (keys can also be numeric)
 */
function getFramingDetails(framing) {
  const v =
    framing?.details ??
    framing?.essentialDetails ??
    framing?.essential_details ??
    framing?.frame?.details ??
    {};
  return v && typeof v === "object" && !Array.isArray(v) ? v : {};
}

function parseKeyTopicIsAbout(message) {
  const t = cleanText(message);
  const m = t.match(/^(.*?)\s+is about\s+(.+?)(?:[.?!]|$)/i);
  if (m) return { keyTopic: cleanText(m[1]), isAbout: cleanText(m[2]) };
  return { keyTopic: null, isAbout: null };
}

function isClearKeyTopicLabel(label) {
  const kt = cleanText(label);
  if (kt.length < 3 || kt.length > 80) return false;
  if (/^(topic|stuff|things|history|government|science|math|literature)$/i.test(kt)) return false;
  if (kt.split(" ").length > 10) return false;
  return true;
}

function hasMeaningfulIsAbout(isAbout) {
  const ia = cleanText(isAbout);
  if (ia.length < 6 || ia.length > 220) return false;
  if (/^(stuff|things|a topic|government|history|science)\b/i.test(ia)) return false;
  return true;
}

function canLeadToMainIdeas(isAbout) {
  const ia = cleanText(isAbout);
  if (/(because|so that|how|why|caused by|leads to|results in|changed|influenced|prevents|helps|shows|explains|compares|contrasts)/i.test(ia)) {
    return true;
  }
  return ia.split(" ").length >= 6;
}

function instructionalSufficiency(keyTopic, isAbout) {
  const hasLabel = isClearKeyTopicLabel(keyTopic);
  const hasDir = hasMeaningfulIsAbout(isAbout);
  const leads = canLeadToMainIdeas(isAbout);
  return { sufficient: hasLabel && hasDir && leads, hasLabel, hasDir, leads };
}

function looksStuck(text) {
  const t = cleanText(text).toLowerCase();
  return (
    t === "i am not sure" ||
    t === "im not sure" ||
    t === "not sure" ||
    t === "idk" ||
    t === "i don't know" ||
    t === "dont know" ||
    t === "i'm confused" ||
    t === "im confused" ||
    t === "confused" ||
    t === "help" ||
    t.length <= 4
  );
}

function looksLikeMetaRepeat(text) {
  const t = cleanText(text).toLowerCase();
  return /^(i already|i just|i said|i told you|i shared)/i.test(t);
}

// Student saying “no / i don’t think so / that’s it” etc.
function isNegativeResponse(text) {
  const t = cleanText(text).toLowerCase();
  return (
    t === "no" ||
    t === "nope" ||
    t === "nah" ||
    t === "not really" ||
    t === "i don't think so" ||
    t === "i dont think so" ||
    t === "i think that's it" ||
    t === "i think thats it" ||
    t === "that's it" ||
    t === "thats it" ||
    t === "nothing else" ||
    t === "i'm done" ||
    t === "im done"
  );
}

// --- Discipline-aware prompts WITHOUT supplying content ---
function promptForSpecificDetail() {
  return "What specific event, action, or fact shows this idea is true?";
}

/**
 * Determine which Main Idea we should collect details for next.
 * We prefer the earliest Main Idea that has <2 details collected.
 */
function computeCurrentMainIdeaIndex(mainIdeas, detailsObj) {
  const n = mainIdeas.length;
  for (let i = 0; i < n; i++) {
    const key = String(i);
    const arr = Array.isArray(detailsObj[key]) ? detailsObj[key] : [];
    if (arr.length < 2) return i;
  }
  // If all have 2+ details, return last index (or 0)
  return Math.max(0, n - 1);
}

function getDetailsForIndex(detailsObj, idx) {
  const key = String(idx);
  const arr = Array.isArray(detailsObj[key]) ? detailsObj[key] : [];
  return arr.map(cleanText).filter(Boolean);
}

// -----------------------------------------------
// PEDAGOGY-LOCKED NEXT QUESTION ROUTER
// -----------------------------------------------
function nextFrameQuestion(studentMessage, framing = {}, stepHint = "") {
  const msg = cleanText(studentMessage);

  const keyTopic = cleanText(getFramingKeyTopic(framing));
  const isAbout = cleanText(getFramingIsAbout(framing));
  const hasKeyTopic = isClearKeyTopicLabel(keyTopic);
  const hasIsAbout = hasMeaningfulIsAbout(isAbout);

  const mainIdeas = getFramingMainIdeas(framing).map(cleanText).filter(Boolean);
  const detailsObj = getFramingDetails(framing);

  // ------------------
  // 0) Stuck handling (never reset progress)
  // ------------------
  if (looksStuck(msg)) {
    if (hasKeyTopic && hasIsAbout) {
      if (mainIdeas.length < 1) return "What is one Main Idea that helps explain your topic?";
      if (mainIdeas.length < 2) return "What is another Main Idea that supports your topic?";
      if (mainIdeas.length < 3) return "Is there another important Main Idea that helps explain your topic?";
      const first = cleanText(mainIdeas[0] || "");
      return first
        ? `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`
        : "What is one Main Idea that helps explain your topic?";
    }
    if (hasKeyTopic && !hasIsAbout) {
      return `Finish this sentence: “${keyTopic} is about ___.”`;
    }
    return "What is your Key Topic? (2–5 words.)";
  }

  // ------------------
  // 1) Honor stepHint when present
  // ------------------
  if (stepHint === "keyTopic") {
    if (isClearKeyTopicLabel(msg)) {
      return `Finish this sentence: “${msg} is about ___.”`;
    }
    return "What is your Key Topic? (2–5 words.)";
  }

  if (stepHint === "isAbout") {
    return "What is one Main Idea that helps explain your topic?";
  }

  // ------------------
  // 2) MAIN IDEAS (2 required, 3 optional) — teacher controlled
  // ------------------
  if (stepHint === "mainIdeas") {
    if (!hasKeyTopic) return "What is your Key Topic? (2–5 words.)";
    if (!hasIsAbout) return `Finish this sentence: “${keyTopic} is about ___.”`;

    if (mainIdeas.length < 1) {
      return "What is one Main Idea that helps explain your topic?";
    }

    if (mainIdeas.length < 2) {
      return "What is another Main Idea that supports your topic?";
    }

    // After 2 ideas: sufficiency check
    if (mainIdeas.length === 2) {
      if (isNegativeResponse(msg)) {
        const first = cleanText(mainIdeas[0] || "");
        return `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`;
      }
      return "Is there another important Main Idea that helps explain your topic?";
    }

    // After 3 ideas: sufficiency check, then move on if “no”
    if (mainIdeas.length >= 3) {
      if (isNegativeResponse(msg) || mainIdeas.length > 3) {
        const first = cleanText(mainIdeas[0] || "");
        return `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`;
      }
      return "Is there one more important idea needed to fully explain your topic?";
    }
  }

  // ------------------
  // 3) ESSENTIAL DETAILS — nested per Main Idea (2–3 per)
  // ------------------
  if (stepHint === "details") {
    // Ensure we have at least 2 main ideas before details
    if (mainIdeas.length < 2) {
      return "What is another Main Idea that supports your topic?";
    }

    const currentIdx = computeCurrentMainIdeaIndex(mainIdeas, detailsObj);
    const currentIdea = cleanText(mainIdeas[currentIdx] || "");
    const currentDetails = getDetailsForIndex(detailsObj, currentIdx);

    if (isNegativeResponse(msg)) {
      let nextIdx = -1;
      for (let i = currentIdx + 1; i < mainIdeas.length; i++) {
        const arr = getDetailsForIndex(detailsObj, i);
        if (arr.length < 2) { nextIdx = i; break; }
      }

      if (nextIdx >= 0) {
        const nextIdea = cleanText(mainIdeas[nextIdx] || "");
        return `Your next Main Idea was “${nextIdea}.” What are 2–3 Essential Details that support this idea?`;
      }

      return "So what? What’s important to understand about this topic?";
    }

    if (currentDetails.length < 1) {
      return `Your ${currentIdx === 0 ? "first" : currentIdx === 1 ? "next" : "last"} Main Idea was “${currentIdea}.” What are 2–3 Essential Details that support this idea?`;
    }

    if (currentDetails.length === 1) {
      return "What is another Essential Detail that helps prove this idea?";
    }

    if (currentDetails.length === 2) {
      return "Is there another important detail that strengthens this idea?";
    }

    if (currentDetails.length >= 3) {
      return "Is there anything else essential we need to understand about this idea?";
    }
  }

  // ------------------
  // 4) SO WHAT — explicit
  // ------------------
  if (stepHint === "soWhat") {
    if (msg.split(" ").length < 6) {
      return "So what? What’s important to understand about this topic?";
    }
    return "Why does this matter beyond this assignment?";
  }

  // ------------------
  // 5) Fallback: route based on sufficiency
  // ------------------
  const parsed = parseKeyTopicIsAbout(msg);
  if (parsed.keyTopic && parsed.isAbout) {
    const check = instructionalSufficiency(parsed.keyTopic, parsed.isAbout);
    if (check.sufficient) return "What is one Main Idea that helps explain your topic?";
  }

  if (hasKeyTopic && hasIsAbout) {
    if (mainIdeas.length < 1) return "What is one Main Idea that helps explain your topic?";
    if (mainIdeas.length < 2) return "What is another Main Idea that supports your topic?";
    const first = cleanText(mainIdeas[0] || "");
    if (first) return `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`;
    return "What is one Main Idea that helps explain your topic?";
  }

  if (!hasKeyTopic) return "What is your Key Topic? (2–5 words.)";
  return `Finish this sentence: “${keyTopic} is about ___.”`;
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
const trimmed = String(message).trim();

// DEBUG: log exactly what the frontend sent
console.log("=== INCOMING REQUEST BODY ===");
console.log(JSON.stringify(req.body, null, 2));

    if (!trimmed) {
      return res.status(400).json({ error: "Missing 'message' in request body" });
    }

    const routedQuestion = nextFrameQuestion(trimmed, framing, stepHint);
console.log("=== ROUTING INPUTS ===");
console.log(JSON.stringify({ trimmed, framing, stepHint }, null, 2));
console.log("=== ROUTED QUESTION ===");
console.log(routedQuestion);

    // ---- Safety check ----
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      const safeReply =
        SAFETY_RESPONSES[safety.category] ||
        "Let’s zoom back to your Frame. What is your Key Topic? (2–5 words.)";

      return res.status(200).json({
        reply: enforceHardCap(safeReply),
        flagged: true,
        flagCategory: safety.category || "unknown",
        severity: safety.severity || "low",
        safetyMode: true,
      });
    }

    // ---- OpenAI call (tone + one-question enforcement; we provide the exact question) ----
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.3,
      max_tokens: MAX_MODEL_TOKENS,
      messages: [
        { role: "system", content: SYSTEM_PROMPT_FRAMING },
        {
          role: "user",
          content:
            `Student said: ${trimmed}\n\n` +
            `Ask EXACTLY this next step as ONE question:\n${routedQuestion}`,
        },
      ],
    });
    // DEBUG: log raw model output
console.log("=== RAW MODEL RESPONSE ===");
console.log(JSON.stringify(completion.choices[0].message, null, 2));

    const raw =
      completion?.choices?.[0]?.message?.content?.trim() ||
      routedQuestion;

    const reply = socraticFailSafe(raw, routedQuestion);

    return res.status(200).json({
      reply,
      flagged: false,
      flagCategory: "",
      severity: "",
      safetyMode: false,
    });
  } catch (err) {
    console.error("Tutor API error:", err);
    return res.status(500).json({
      error: "Server error",
      details: err?.message || String(err),
    });
  }
}

