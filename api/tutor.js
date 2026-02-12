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
 * DEMO-SAFE MODE (important):
 * - We still compute the next question with the router.
 * - We DO NOT let the model rewrite the question (prevents drifting / backtracking / term invention).
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

STEP MISMATCH RULE (critical):
If a student response belongs to a DIFFERENT framing step
(e.g., gives a topic during Main Ideas, or a main idea during Essential Details),
do NOT change steps or re-anchor the frame.
Instead, restate the CURRENT step and ask again with clarification.

SYNTHESIZE-AND-ADVANCE RULE:
When a student provides a valid idea that is incomplete,
briefly rephrase it into precise academic language
by embedding the rephrased idea into the next single question within the routine.

TOPIC LOCK RULE (critical):
Once Key Topic and Is About are complete,
the Key Topic is LOCKED.
Never ask for or revisit Key Topic or Is About again,
even if a student response is incorrect or off-topic.

FRAMING ROUTINE SEQUENCE (must follow):
1) Key Topic (2–5 words: the focus/title)
2) Is About (one short phrase that explains what the Key Topic focuses on)
3) Main Ideas (2–3 important ideas that explain or support the topic)
4) Essential Details (2–3 evidence/examples for EACH main idea)
5) So What? (why it matters / significance / conclusion)

KEY TOPIC COMPLETION RULE:
A Key Topic is complete only if it is 2–5 words
and clearly names the focus of the assignment.
Once complete, move to the Is About step.
Do not ask for Key Topic again.

IS ABOUT COMPLETION RULE:
An Is About statement is complete only if it:
- Forms a clear, complete idea
- Explains what the Key Topic focuses on
- Can logically lead to 2–3 Main Ideas.
If incomplete or vague, ask a clarifying question
within the Is About step.
Do not advance early.

EXTRACTION RULE (critical – mandatory behavior):
If a student response contains a complete framing structure in one sentence
(e.g., "The Cuban Missile Crisis is about how the U.S. almost went to war with the USSR"
or "How the Cuban Missile Crisis almost forced America into nuclear war"):

1. Silently extract:
   - Key Topic
   - Is About

2. Treat BOTH as complete.

3. Do NOT ask again for Key Topic or Is About.

4. Immediately advance to the next required framing step in the sequence.

If the student response is unclear, incomplete, or missing one element,
ask ONLY for the missing element.

NO REPEAT RULE (critical):
Once a framing step is complete,
do NOT repeat that step again,
even if the student restates it differently.
Always move forward in the sequence.

TOPIC STABILITY RULE (critical):
During Main Ideas and Essential Details:

- EVERY question must explicitly include the original Key Topic.
- NEVER treat a Main Idea as if it becomes the new Key Topic.
- NEVER ask “supports ___” unless the blank is the original Key Topic,
  or (ONLY during Essential Details) the current Main Idea being detailed.
- If a student gives a Main Idea (e.g., “Castro’s rise to power”),
  treat it only as supporting the Key Topic.

Example (must follow):

If the Key Topic is “Cuban Missile Crisis” and a student says
“Castro’s rise to power,” the next question MUST still reference
“the Cuban Missile Crisis,” not Castro’s rise to power.

Required template during Main Ideas:
“What is another Main Idea that helps explain the [KEY TOPIC]?”

ABSOLUTE ANCHOR OVERRIDE (critical):
During Main Ideas:
- The Key Topic MUST be explicitly restated in EVERY Main Ideas question.
- The blank in “What is another Main Idea that helps explain ___?”
  MUST always be the original Key Topic.
- Never insert a Main Idea into that blank.

Correct pattern:
“What is another Main Idea that helps explain the Cuban Missile Crisis?”

Incorrect pattern:
“What is another Main Idea that supports Castro’s rise to power?”

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

function getFramingDetails(framing) {
  const v =
    framing?.details ??
    framing?.essentialDetails ??
    framing?.essential_details ??
    framing?.frame?.details ??
    {};
  return v && typeof v === "object" && !Array.isArray(v) ? v : {};
}

// ✅ NEW: reject “my assignment” / “assignment” / “my topic” etc as Key Topic
function isGenericKeyTopic(kt) {
  const t = cleanText(kt).toLowerCase();
  return (
    t === "my assignment" ||
    t === "assignment" ||
    t === "my topic" ||
    t === "topic" ||
    t === "my essay" ||
    t === "essay" ||
    t === "my paper" ||
    t === "paper" ||
    t === "my writing" ||
    t === "writing"
  );
}

function parseKeyTopicIsAbout(message) {
  const t = cleanText(message);
  const m = t.match(/^(.*?)\s+is about\s+(.+?)(?:[.?!]|$)/i);
  if (m) {
    const keyTopic = cleanText(m[1]);
    const isAbout = cleanText(m[2]);
    // ✅ NEW: if “my assignment is about …” don’t treat “my assignment” as Key Topic
    if (!keyTopic || isGenericKeyTopic(keyTopic)) return { keyTopic: null, isAbout: null };
    return { keyTopic, isAbout };
  }
  return { keyTopic: null, isAbout: null };
}

function isClearKeyTopicLabel(label) {
  const kt = cleanText(label);
  if (kt.length < 3 || kt.length > 80) return false;
  if (isGenericKeyTopic(kt)) return false; // ✅ NEW
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

function computeCurrentMainIdeaIndex(mainIdeas, detailsObj) {
  const n = mainIdeas.length;
  for (let i = 0; i < n; i++) {
    const key = String(i);
    const arr = Array.isArray(detailsObj[key]) ? detailsObj[key] : [];
    if (arr.length < 2) return i;
  }
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

  // 0) Stuck handling
  if (looksStuck(msg)) {
    if (hasKeyTopic && hasIsAbout) {
      if (mainIdeas.length < 1) return `What is one Main Idea that helps explain the ${keyTopic}?`;
      if (mainIdeas.length < 2) return `What is another Main Idea that helps explain the ${keyTopic}?`;
      if (mainIdeas.length < 3) return `Is there another important Main Idea that helps explain the ${keyTopic}?`;
      const first = cleanText(mainIdeas[0] || "");
      return first
        ? `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`
        : `What is one Main Idea that helps explain the ${keyTopic}?`;
    }
    if (hasKeyTopic && !hasIsAbout) return `Finish this sentence: “${keyTopic} is about ___.”`;
    return "What is your Key Topic? (2–5 words.)";
  }

  // 1) Honor stepHint
  if (stepHint === "keyTopic") {
    if (isClearKeyTopicLabel(msg)) return `Finish this sentence: “${msg} is about ___.”`;
    return "What is your Key Topic? (2–5 words.)";
  }

  if (stepHint === "isAbout") {
    if (hasKeyTopic) return `What is one Main Idea that helps explain the ${keyTopic}?`;
    return "What is one Main Idea that helps explain your topic?";
  }

  // 2) MAIN IDEAS
  if (stepHint === "mainIdeas") {
    if (!hasKeyTopic) return "What is your Key Topic? (2–5 words.)";
    if (!hasIsAbout) return `Finish this sentence: “${keyTopic} is about ___.”`;

    if (mainIdeas.length < 1) return `What is one Main Idea that helps explain the ${keyTopic}?`;
    if (mainIdeas.length < 2) return `What is another Main Idea that helps explain the ${keyTopic}?`;

    if (mainIdeas.length === 2) {
      if (isNegativeResponse(msg)) {
        const first = cleanText(mainIdeas[0] || "");
        return `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`;
      }
      return `Is there another important Main Idea that helps explain the ${keyTopic}?`;
    }

    if (mainIdeas.length >= 3) {
      if (isNegativeResponse(msg) || mainIdeas.length > 3) {
        const first = cleanText(mainIdeas[0] || "");
        return `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`;
      }
      return `Is there one more important idea needed to fully explain the ${keyTopic}?`;
    }
  }

  // 3) ESSENTIAL DETAILS
  if (stepHint === "details") {
    if (mainIdeas.length < 2) {
      if (hasKeyTopic) return `What is another Main Idea that helps explain the ${keyTopic}?`;
      return "What is another Main Idea that helps explain your topic?";
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
    if (currentDetails.length === 1) return "What is another Essential Detail that helps prove this idea?";
    if (currentDetails.length === 2) return "Is there another important detail that strengthens this idea?";
    return "Is there anything else essential we need to understand about this idea?";
  }

  // 4) SO WHAT
  if (stepHint === "soWhat") {
    if (msg.split(" ").length < 6) return "So what? What’s important to understand about this topic?";
    return "Why does this matter beyond this assignment?";
  }

  // 5) Fallback: extraction + sufficiency
  const parsed = parseKeyTopicIsAbout(msg);
  if (parsed.keyTopic && parsed.isAbout) {
    if (isClearKeyTopicLabel(parsed.keyTopic) && hasMeaningfulIsAbout(parsed.isAbout)) {
      return `What is one Main Idea that helps explain the ${parsed.keyTopic}?`;
    }
  }

  if (hasKeyTopic && hasIsAbout) {
    if (mainIdeas.length < 1) return `What is one Main Idea that helps explain the ${keyTopic}?`;
    if (mainIdeas.length < 2) return `What is another Main Idea that helps explain the ${keyTopic}?`;
    const first = cleanText(mainIdeas[0] || "");
    if (first) return `Your first Main Idea was “${first}.” What are 2–3 Essential Details that support this idea?`;
    return `What is one Main Idea that helps explain the ${keyTopic}?`;
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

    if (!trimmed) return res.status(400).json({ error: "Missing 'message' in request body" });

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

    /**
     * DEMO-SAFE MODE:
     * Return the routed question directly (prevents the model from drifting into:
     * - “supports Castro…” phrasing
     * - asking Key Topic again
     * - inventing “claim” step
     * - backtracking / skipping
     */
    return res.status(200).json({
      reply: enforceHardCap(routedQuestion),
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
