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

// ======================================================
// KAW v2 ARCHITECTURE MAP
// ======================================================
//
// Kaw is organized around four layers:
//
// 1. HIGH-IMPACT LEARNING STRATEGY
//    - Defines what students are building.
//    - Current strategy: KU Framing Routine.
//
// 2. STRUCTURAL ANCHORS
//    - Parent Anchor controls progression through the routine.
//    - Child Anchor translates that progression into strategy/component language.
//
// 3. INSTRUCTIONAL INTELLIGENCE ENGINE
//    - Formative Assessment: gathers evidence of student understanding.
//    - Diagnosis: interprets what the evidence suggests.
//    - Context Integration: combines assignment, frame, stage, and conversation context.
//    - Instructional Decision: selects the best instructional move.
//    - Student Ownership Check: ensures Kaw helps students think without thinking for them.
//
// 4. INSTRUCTIONAL MOVE LIBRARY
//    - Build
//    - Scaffold
//    - Clarify
//    - Probe
//    - Checkpoint
//    - Revise
//    - Expand
//    - Bridge
//    - Celebrate
//    - Refocus
//    - Remind
//    - Strategy Cue
//    - Reflect
//
// Phase 1 goal:
// Reorganize and label the existing instructional expertise without changing behavior.
//
// Refactor rule:
// Parent/Child Anchor progression logic is load-bearing.
// Do not move or rewrite it until the engine sections are stable.


// ======================================================
// INSTRUCTIONAL INTELLIGENCE ENGINE — READ-ONLY SHELL
// ======================================================
//
// This section begins Kaw's engine layer.
// In Phase 1, it observes and plans only.
// It does not control runtime behavior yet.

const KAW_ARCHITECTURE = {
 knowledgeLayer: {
    assignmentContext: true,
    kuFramingRoutine: true,
    frameComponentKnowledge: true,
    cognitiveStrategies: true,
    instructionalMoves: true,
},

 reasoningLayer: {
    studentThinkingModel: true,
    evidenceAnalysis: true,
    instructionalReasoning: true,
    adaptiveCoaching: true,
    instructionalPlanning: true,
},

  conversationLayer: {
    buildMode: true,
    feedbackMode: true,
    reflection: true,
    export: true,
  },
};

// ======================================================
// HIGH-IMPACT LEARNING STRATEGY — KU FRAMING ROUTINE
// ======================================================
//
// This section defines what students are building.
//
// The current strategy is the KU Framing Routine.
// KU_FRAME_COMPONENTS stores the instructional purpose,
// expected evidence, common breakdowns, cognitive strategies,
// validation rules, and conversation language for each Frame component.
//
// Future goal:
// Other high-impact learning strategies should be able to provide
// their own strategy knowledge without changing the core engine.

const KU_FRAME_COMPONENTS = {

  keyTopic: {
  purpose: "Name the topic that will be explored.",
  definition: "The title or name of the key topic.",
  studentAction: "Write the name of the topic in the Key Topic box.",
  expectedEvidence: [
    "Names the central topic",
    "Is concise",
    "Can be explored in the Frame",
    "Aligns with the assignment or source"
  ],
  commonBreakdowns: [
    "Writes a full sentence",
    "Writes a claim",
    "Gives a detail instead of the topic",
    "Uses a generic phrase like 'my assignment' or 'the topic'"
  ],
  cognitiveStrategies: [
    "identify",
    "select",
    "focus"
  ],
validation: {
    shouldNameTheTopic: true,
    disallowGenericTopics: true
},

conversationSupport: {
 term: "Key Topic",
  friendlyTerm: "main topic",
  initialPrompt:
    "Let's start with your Key Topic.\n\nWhat is the main topic you'll be exploring in this Frame?",
  revisePrompt:
    "That’s a good start, but your Key Topic should name the topic clearly.\n\nWhat is the main topic you'll be exploring in this Frame?"
},

genericNonExamples: [
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
  "that"
]
},

   isAbout: {
  purpose: "Provide a brief explanation or paraphrase of the key topic.",
  definition: "A phrase or sentence that summarizes what the whole topic is about in words students understand.",
  studentAction: "Write a paraphrase of the key topic in the Is About space.",
  expectedEvidence: [
    "Paraphrases the key topic",
    "Summarizes the whole topic",
    "Uses understandable language",
    "Prepares the reader for the main ideas"
  ],
    successIndicators: [
    "Clearly names the central topic.",
    "Matches the assignment or source.",
    "Can be explored through Main Ideas and Essential Details."
],
  commonBreakdowns: [
    "Repeats the Key Topic only",
    "Writes a claim instead of a paraphrase",
    "Gets too detailed too soon",
    "Uses vague wording"
  ],
 cognitiveStrategies: [
  "paraphrase",
  "summarize",
  "clarify"
],

validation: {
  shouldSummarizeTheWholeTopic: true,
  shouldUseStudentFriendlyLanguage: true,
  shouldNotRepeatKeyTopicOnly: true
},

conversationSupport: {
  term: "Is About",
  friendlyTerm: "description",

initialPrompt:
  'Now let\'s describe your Key Topic in your own words.\n\nWhat is "{keyTopic}" about?',

revisePrompt:
  '✏️ Let\'s revise your "Is About" statement.\n\nWhat would you like it to say instead?',

confirmationPrompt:
  '💡 Great! Based on what you shared, your Frame now says:\n\n{keyTopic} is about {isAbout}.\n\nDoes this accurately capture your thinking?\n\n1) Yes — Continue building my Frame.\n2) No — Revise this part.\n\nReply with 1 or 2.'
 }
},

mainIdeas: {
  purpose: "Identify the major ideas that help explain the Key Topic.",
  definition: "The big ideas, categories, causes, parts, or supports that organize the topic.",
  studentAction: "Write important Main Ideas that help explain the Key Topic.",

  expectedEvidence: [
    "Names an important idea",
    "Connects to the Key Topic",
    "Can be supported with Essential Details",
    "Is broader than a single detail"
  ],

  commonBreakdowns: [
    "Gives a detail instead of a Main Idea",
    "Repeats the Key Topic",
    "Writes something too broad",
    "Writes something unrelated to the topic"
  ],

  cognitiveStrategies: [
    "categorize",
    "organize",
    "prioritize",
    "explain"
  ],

  validation: {
    shouldConnectToKeyTopic: true,
    shouldBeSupportableWithDetails: true,
    shouldNotBeOnlyADetail: true
  },

  conversationSupport: {
    term: "Main Idea",
    friendlyTerm: "important idea",

    initialPrompt:
      'What is one Main Idea that helps explain "{keyTopic}"?',

    additionalPrompt:
      'What is another Main Idea that helps explain "{keyTopic}"?',

   revisePrompt:
      "You're close. Try naming a Main Idea that clearly connects to your Key Topic and can be explained with Essential Details.",

   confirmationPrompt:
      '✅ Checkpoint\n\nYou identified these Main Ideas:\n\n{mainIdeasList}\n\nDoes this accurately capture your thinking?\n\n📋 Choose an option:\n\n1) Yes — Continue building my Frame.\n2) No — Revise one Main Idea.\n\nReply with 1 or 2.'
  }
},

details: {
  purpose: "Add information that supports and explains each Main Idea.",
  definition: "Specific facts, examples, evidence, or explanations that make a Main Idea clearer.",
  studentAction: "Write Essential Details that support each Main Idea.",

  expectedEvidence: [
    "Supports a specific Main Idea",
    "Adds specific information",
    "Explains or proves the idea",
    "Is more specific than the Main Idea"
  ],

  commonBreakdowns: [
    "Repeats the Main Idea",
    "Adds a new Main Idea instead of a detail",
    "Is too vague",
    "Does not clearly support the Main Idea"
  ],

  cognitiveStrategies: [
    "support",
    "explain",
    "specify",
    "connect"
  ],

  validation: {
    shouldSupportMainIdea: true,
    shouldBeSpecific: true,
    shouldNotIntroduceNewMainIdea: true
  },

  conversationSupport: {
    term: "Essential Detail",
    friendlyTerm: "essential detail,"

    initialPrompt:
      'What is one Essential Detail that supports this Main Idea?',

    additionalPrompt:
      'What is another Essential Detail that supports this Main Idea?',

    revisePrompt:
      "You're close. Try adding a specific fact, example, or explanation that supports this Main Idea."
  }
},

soWhat: {
  purpose: "Help students state what is important to understand after seeing the whole Frame.",
  definition: "A final takeaway that explains why the information in the Frame matters.",
  studentAction: "Write the important understanding or takeaway in the So What space.",

  expectedEvidence: [
    "Connects across the Frame",
    "Explains why the ideas matter",
    "States an important takeaway",
    "Goes beyond listing details"
  ],

  commonBreakdowns: [
    "Repeats the Key Topic",
    "Repeats one Main Idea",
    "Lists details instead of explaining importance",
    "Uses vague wording like 'it is important'"
  ],

  cognitiveStrategies: [
    "synthesize",
    "generalize",
    "prioritize",
    "explain significance"
  ],

  validation: {
    shouldSynthesizeAcrossFrame: true,
    shouldExplainImportance: true,
    shouldNotSimplyRepeatEarlierParts: true
  },

  conversationSupport: {
    term: "So What",
    friendlyTerm: "important takeaway",

    initialPrompt:
      'Now let\'s think about the So What.\n\nLooking at everything in your Frame, what is the most important thing someone should understand about "{keyTopic}"?',

    revisePrompt:
      "You're close. Try explaining the bigger takeaway instead of repeating one part of the Frame."
  }
}

};

// ======================================================
// INSTRUCTIONAL INTELLIGENCE ENGINE
// ======================================================
//
// This section gathers context, interprets student needs,
// creates an instructional plan, and selects an instructional move.
//
// Phase 1 status:
// Read-only planning layer.
// Existing runtime flow still controls Kaw's actual response.

// ------------------------------------------------------
// FORMATIVE ASSESSMENT
// Gathers evidence about student understanding.
// ------------------------------------------------------

function analyzeFeedbackResponse(state) {
  const response = cleanText(state?.feedback?.currentResponse || "");
  const lower = response.toLowerCase();

  const detectedGaps = [];

  if (
  state?.frameMeta?.frameType === "themes" &&
  (
    lower.startsWith("you should") ||
    lower.startsWith("people should") ||
    lower.includes("should always") ||
    lower.includes("should never")
  )
) {
  detectedGaps.push("adviceInsteadOfInsight");
}
    
if (state?.frameMeta?.frameType === "themes") {
  const looksLikeEventSummary =
    /\b(moved|joined|went|met|lost|found|started|stopped)\b.*\band\b.*\b(moved|joined|went|met|lost|found|started|stopped)\b/i.test(lower);

  if (
    lower.startsWith("this story is about") ||
    lower.startsWith("the story is about") ||
    lower.startsWith("this text is about") ||
    lower.startsWith("the text is about") ||
    lower.includes("first ") ||
    lower.includes("then ") ||
    lower.includes("next ") ||
    lower.includes("finally ") ||
    looksLikeEventSummary
  ) {
    detectedGaps.push("summaryInsteadOfThinking");
  }
}

  if (
    lower.includes("stuff") ||
    lower.includes("things") ||
    lower.includes("something") ||
    lower.includes("important")
  ) {
    detectedGaps.push("vague");
  }

  if (response.split(/\s+/).filter(Boolean).length < 5) {
    detectedGaps.push("needsSpecificity");
  }

  const uniqueGaps = [...new Set(detectedGaps)];

  const primaryGap = uniqueGaps
    .slice()
    .sort((a, b) => {
      const pa = FEEDBACK_GAP_BANK[a]?.priority ?? 999;
      const pb = FEEDBACK_GAP_BANK[b]?.priority ?? 999;
      return pa - pb;
    })[0] || null;

  return {
    detectedGaps: uniqueGaps,
    primaryGap
  };
}

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

function isWeakFrameResponse(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return true;

  const weakExact = new Set([
    "stuff",
    "things",
    "something",
    "anything",
    "whatever",
    "maybe",
    "i guess",
    "guess",
    "idk",
    "i don't know",
    "i dont know"
  ]);

  if (weakExact.has(t)) return true;
  if (isStuckMessage(t)) return true;

  return false;
}

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

// ------------------------------------------------------
// STUDENT OWNERSHIP CHECK
// Ensures Kaw never replaces student thinking.
// ------------------------------------------------------


// ------------------------------------------------------
// CONTEXT INTEGRATION
// Combines assignment, strategy, anchors, and conversation.
// ------------------------------------------------------

function buildInstructionalContext(state, message = "") {
const currentFrameStage =
  typeof getStage === "function" ? getStage(state) : "";

const componentKnowledge =
  KU_FRAME_COMPONENTS[
    typeof getBaseStage === "function"
      ? getBaseStage(currentFrameStage)
      : currentFrameStage
  ] || null;
 return {
    message: cleanText(message),
    interactionMode: state?.interactionMode || "build",
    assignmentContext: state?.frameMeta?.assignmentContext || {},
    assignmentReasoning: state?.assignmentReasoning || {},
    useMode: state?.frameMeta?.purpose || "",
    thinkingPattern: state?.frameMeta?.frameType || "",
    frameStage: currentFrameStage,
    componentKnowledge,
    parentAnchorStage: typeof getParentAnchorContext === "function"
      ? getParentAnchorContext(state)
      : null,
    frame: state?.frame || {},
    feedback: state?.feedback || {},
    pending: state?.pending || null,
    transcript: Array.isArray(state?.transcript) ? state.transcript : [],
  };
}

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

function getComponentKnowledge(frameStage) {
  const baseStage =
    typeof getBaseStage === "function"
      ? getBaseStage(frameStage)
      : frameStage;

  return KU_FRAME_COMPONENTS[baseStage] || null;
}

function getComponentConversation(componentName) {
  return (
    KU_FRAME_COMPONENTS?.[componentName]?.conversationSupport || {}
  );
}

function getComponentPrompt(componentName, promptType = "initialPrompt", context = {}) {
  const template =
    getComponentConversation(componentName)?.[promptType] ||
    "What should you add next?";

return template
  .replaceAll("{keyTopic}", context.keyTopic || "your topic")
  .replaceAll("{isAbout}", context.isAbout || "")
  .replaceAll("{mainIdea}", context.mainIdea || "this Main Idea")
  .replaceAll("{mainIdeasList}", context.mainIdeasList || "")
  .replaceAll("{detail}", context.detail || "this detail");
}

// ------------------------------------------------------
// DIAGNOSIS
// Interprets the instructional meaning of the evidence.
// ------------------------------------------------------

function inferThinkingStrategy(context) {
  const assignmentText = cleanText(
    context?.assignmentContext?.understanding ||
    context?.assignmentContext?.studentSummary ||
    context?.assignmentContext?.raw ||
    ""
  ).toLowerCase();

  const legacyFrameType = cleanText(context?.thinkingPattern || "");

  if (
    assignmentText.includes("cause") ||
    assignmentText.includes("effect") ||
    assignmentText.includes("why") ||
    assignmentText.includes("how") ||
    legacyFrameType === "causeEffect"
  ) {
    return "explain_relationship";
  }

  if (
    assignmentText.includes("theme") ||
    assignmentText.includes("message") ||
    assignmentText.includes("central idea") ||
    assignmentText.includes("big idea") ||
    legacyFrameType === "themes"
  ) {
    return "interpret_meaning";
  }

  if (
    assignmentText.includes("compare") ||
    assignmentText.includes("contrast") ||
    assignmentText.includes("similar") ||
    assignmentText.includes("different")
  ) {
    return "compare_features";
  }

  if (
    assignmentText.includes("evidence") ||
    assignmentText.includes("source") ||
    assignmentText.includes("text") ||
    legacyFrameType === "reading"
  ) {
    return "organize_evidence";
  }

  return "organize_thinking";
}

function diagnoseInstructionalNeed(context) {
  const frameStage = context?.frameStage || "";
  const thinkingStrategy = inferThinkingStrategy(context);
  const pendingType = context?.pending?.type || "";

  if (pendingType) {
    return {
      situation: `pending_${pendingType}`,
      likelyNeed: "continue_current_flow",
      confidence: "medium",
    };
  }

  if (frameStage === "assignmentContext") {
    return {
      situation: "student_needs_assignment_context",
      likelyNeed: "elicit_assignment_understanding",
      confidence: "high",
    };
  }

  if (frameStage === "keyTopic") {
    return {
      situation: "student_identifying_key_topic",
      likelyNeed: "elicit_topic_focus",
      confidence: "high",
    };
  }

  if (frameStage === "isAbout") {
    return {
      situation: `student_building_is_about_${thinkingStrategy}`,
      likelyNeed: "clarify_relationship_or_meaning",
      confidence: "medium",
    };
  }

  if (frameStage === "mainIdeas") {
    return {
      situation: `student_generating_main_ideas_${thinkingStrategy}`,
      likelyNeed: "elicit_supporting_structure",
      confidence: "medium",
    };
  }

  if (typeof frameStage === "string" && frameStage.startsWith("details:")) {
    return {
      situation: `student_adding_details_${thinkingStrategy}`,
      likelyNeed: "strengthen_evidence_or_explanation",
      confidence: "medium",
    };
  }

  if (frameStage === "soWhat") {
    return {
      situation: `student_synthesizing_so_what_${thinkingStrategy}`,
      likelyNeed: "support_significance_and_takeaway",
      confidence: "medium",
    };
  }

  return {
    situation: "general_instructional_support",
    likelyNeed: "determine_next_instructional_move",
    confidence: "low",
  };
}

function analyzeBuildLane(state, stage, response) {
  const frameType = state?.frameMeta?.frameType || "";
  const text = cleanText(response);

  if (!text) return null;

  if (frameType === "themes" && stage === "mainIdeas") {
    if (looksLikeAdvice(text)) {
      return {
        type: "reviseBuildLane",
        stage: "mainIdeas",
        feedback: "That sounds more like advice than a Main Idea.",
        revisionPrompt: "What idea, example, or moment helps show your message about life?"
      };
    }

   const summaryPatterns = [
  /\b(moved|joined|went|met|lost|found|started|stopped)\b.*\band\b.*\b(moved|joined|went|met|lost|found|started|stopped)\b/i
];

const looksLikeEventSummary =
  summaryPatterns.some(pattern => pattern.test(text));

    if (looksLikeSequenceSummary(text) || looksLikeEventSummary) {
      return {
        type: "reviseBuildLane",
        stage: "mainIdeas",
        feedback: "That tells what happened.",
        revisionPrompt: "What idea does this experience help show?"
      };
    }
  }

  if (frameType === "themes" && stage === "details") {
  const lower = text.toLowerCase();

  const soundsBroad =
    lower === cleanText(state?.frame?.isAbout || "").toLowerCase() ||
    lower.includes("people often") ||
    lower.includes("can be difficult") ||
    lower.includes("is important") ||
    lower.includes("matters") ||
    lower.startsWith("people ") ||
    lower.startsWith("someone ") ||
    lower.startsWith("everyone ");

  if (soundsBroad) {
    return {
      type: "reviseBuildLane",
      stage: "details",
      feedback: "That is a good idea, but it sounds broad for an Essential Detail.",
      revisionPrompt: "What specific example, event, or explanation helps show this idea in action?"
    };
  }
}
  
  return null;
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

// ------------------------------------------------------
// INSTRUCTIONAL DECISION
// Chooses the best instructional move.
// ------------------------------------------------------

function createInstructionalPlan(context) {
  const diagnosis = diagnoseInstructionalNeed(context);

  return {
    conversationType: context?.interactionMode || "build",
    frameStage: context?.frameStage || "",
    legacyFrameType: context?.thinkingPattern || "",
    thinkingStrategy: inferThinkingStrategy(context),
    useMode: context?.useMode || "",

    componentKnowledge: getComponentKnowledge(context?.frameStage),

    studentThinkingModel: {
      currentUnderstanding: {},
      misconceptions: [],
      masteredConcepts: [],
      strugglingConcepts: [],
      confidence: {},
      evidence: [],
    },

    instructionalModel: {
      learningGoal: null,
      prerequisiteConcepts: [],
      currentFocus: null,
      instructionalMoves: [],
      scaffoldsUsed: [],
      examplesUsed: [],
    },

    feedbackModel: {
      strengths: [],
      growthAreas: [],
      previousCoaching: [],
      nextRecommendation: null,
    },

    diagnosis,

    adaptiveCoaching: {
      supportLevel: 0,
      reason: "Phase 1 shell only — current engine still controls response.",
    },

    move: selectInstructionalMove(context, diagnosis),
  };
}

function selectInstructionalMove(context, diagnosis) {
  const likelyNeed = diagnosis?.likelyNeed || "";

  if (likelyNeed === "elicit_assignment_understanding") {
    return {
      type: "elicitation",
      guardrail: "Do not answer the assignment. Help the student explain the task.",
      question: "What is your assignment asking you to think about, explain, or show?",
    };
  }

  if (likelyNeed === "elicit_topic_focus") {
    return {
      type: "elicitation",
      guardrail: "Do not choose the topic for the student.",
      question: "What topic or idea does your assignment seem to focus on most?",
    };
  }

  if (likelyNeed === "clarify_relationship_or_meaning") {
    return {
      type: "clarifying_question",
      guardrail: "Do not provide the relationship or meaning for the student.",
      question: "What connection or meaning are you trying to explain?",
    };
  }

  if (likelyNeed === "elicit_supporting_structure") {
    return {
      type: "probe",
      guardrail: "Do not supply main ideas. Elicit one idea from the student.",
      question: "What is one idea, cause, example, or moment that supports your thinking?",
    };
  }

  if (likelyNeed === "strengthen_evidence_or_explanation") {
    return {
      type: "probe",
      guardrail: "Do not invent evidence. Ask the student to connect evidence to thinking.",
      question: "What detail, example, or evidence helps explain this idea?",
    };
  }

  if (likelyNeed === "support_significance_and_takeaway") {
    return {
      type: "synthesis_prompt",
      guardrail: "Do not write the takeaway for the student.",
      question: "What should someone understand after seeing these ideas together?",
    };
  }

  return {
    type: "general_probe",
    guardrail: "Preserve student ownership of thinking.",
    question: "What is one small next step you can take?",
  };
}

function buildMiniQuestion(state) {
  let stage = state?.pending?.stage || getStage(state);

  if (state?.pending?.type === "collectAnotherMainIdea") {
    stage = "mainIdeas";
  }

  const paContext = getParentAnchorContext(state);
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

  if (typeof stage === "string" && stage.startsWith("details:")) {
    const idx = Number(stage.split(":")[1]);
    const mi = getIdeaList(state)[idx] || "this main idea";

    if (isCE) {
      return `You identified this cause:\n\n"${mi}"\n\nNow think about how that leads to this effect:\n\n"${effect}"\n\nWhat detail or example shows how this cause produces the effect?`;
    }

    return `You identified this main idea:\n\n"${mi}"\n\nNow think about how it connects to this message:\n\n"${state.frame?.isAbout || "your theme"}"\n\nWhat specific detail, example, or explanation helps this message about life in action?`;
  }

  if (stage === "soWhat") {
    if (isCE) {
      return `Your frame explains why this happens:\n\n"${effect}"\n\nNow think about why this matters.\n\nWhat should people really understand about this topic?`;
    }

    return `Your frame is showing this message about life:\n\n"${state.frame?.isAbout || "your theme"}"\n\nNow think beyond this one example or text.\n\nWhat should people really understand about life or people because of this theme?`;
  }

  return "What part of your Frame feels easiest to improve right now: Key Topic, Is About, Main Ideas, Details, or So What?";
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
// ======================================================
// INSTRUCTIONAL MOVE LIBRARY
// ======================================================
//
// This section contains reusable teaching moves Kaw can execute
// after the engine interprets the student's need.

// ------------------------------------------------------
// SCAFFOLD / REMIND
// Breaks the task into smaller supports and reminds students
// of relevant prior thinking without giving the answer.
// ------------------------------------------------------
function buildStuckNudges(state, stage) {
  const keyTopic = state?.frame?.keyTopic || "your topic";
  const isAbout = state?.frame?.isAbout || "";
  const ideas = getIdeaList(state).filter(Boolean);

  const frameSummary =
    isAbout && keyTopic
      ? `"${keyTopic}" → ${isAbout}`
      : keyTopic
        ? `"${keyTopic}"`
        : "your Frame";

  if (stage === "mainIdeas") {
    return [
      `🧭 You are building the Main Ideas for ${frameSummary}.`,
      "💡 Main Ideas are the big supporting ideas that explain your Key Topic and strengthen your Is About statement.",
      "🔎 Think about important categories, reasons, examples, parts, or supports that belong in this Frame.",
      "✍️ What is one Main Idea you could add?"
    ];
  }

  if (typeof stage === "string" && stage.startsWith("details:")) {
    const idx = Number(stage.split(":")[1]);
    const selectedMainIdea =
      Number.isInteger(idx) && ideas[idx]
        ? ideas[idx]
        : "this Main Idea";

    return [
      `🧭 You are adding Details for this Main Idea:\n"${selectedMainIdea}"`,
      "💡 Essential Details explain, prove, clarify, or strengthen one specific Main Idea.",
      "🔎 Think of a fact, example, quotation, observation, piece of evidence, or explanation that helps someone understand this Main Idea better.",
      "✍️ What Detail could you add?"
    ];
  }

  if (stage === "soWhat") {
    return [
      "🧭 You are writing the So What.",
      "💡 The So What explains why the completed Frame matters.",
      "🔎 Look across your Key Topic, Is About, Main Ideas, and Details. Instead of repeating them, ask what someone should understand from the whole Frame.",
      "✍️ What is the larger takeaway?"
    ];
  }

  return [
    "🧭 Let’s pause and choose the best support move for where you are in the Frame.",
    "💡 Let’s use the smallest helpful step instead of guessing.",
    "🔎 Look at your Frame and choose the part that feels easiest to restart.",
    "✍️ What part do you want to work on: Key Topic, Is About, Main Ideas, Details, or So What?"
  ];
}

function detectInstructionalState(state, msg) {
  const text = cleanText(msg);
  const lower = text.toLowerCase();
  const stage = state?.pending?.stage || getStage(state);
  const evidence = [];

  const protectedPendingTypes = new Set([
    "confirmLanguageSwitch",
    "assignmentReasoningIntro",
    "chooseWorkflow",
    "choosePurpose",
    "feedbackSelectSection",
    "feedbackCollectResponse",
    "feedbackCoach",
    "feedbackThinkingSummary",
    "feedbackRevise",
    "feedbackComplete",
    "stuckConfirm",
    "stuckMenu",
    "stuckReask",
    "stuckNudge",
    "stuckMini",
    "stuckSkip",
  ]);

  const pendingType = state?.pending?.type || null;

  if (pendingType && protectedPendingTypes.has(pendingType)) {
    return {
      state: "protected",
      level: "none",
      nextBehavior: "continueCurrentFlow",
      evidence: ["protectedPending"],
      confidence: 1,
      stage,
    };
  }

  if (!text) {
    return {
      state: "productive",
      level: "none",
      nextBehavior: "continue",
      evidence: [],
      confidence: 0,
      stage,
    };
  }

  const strongStruggle =
    isStuckMessage(text) ||
    lower.includes("i don't know") ||
    lower.includes("i dont know") ||
    lower.includes("i'm confused") ||
    lower.includes("im confused") ||
    lower.includes("i'm lost") ||
    lower.includes("im lost");

  if (strongStruggle) evidence.push("strongStruggleLanguage");

  const tone = detectStuckTone(text);
  if (tone === "frustration") evidence.push("frustrationTone");
  if (tone === "resistance") evidence.push("resistanceTone");

  const vagueWords = ["stuff", "things", "something", "anything", "whatever"];
  if (vagueWords.some((w) => lower.includes(w))) {
    evidence.push("vagueLanguage");
  }

  const uncertainWords = ["maybe", "i guess", "kind of", "sort of", "not sure"];
  if (uncertainWords.some((w) => lower.includes(w))) {
    evidence.push("uncertaintyLanguage");
  }

  const driftSignals = [
    "can we do something else",
    "can i do something else",
    "forget this",
    "skip this",
    "new assignment",
    "different assignment",
    "i want to talk about",
    "this is boring",
    "this is taking forever",
    "this takes forever",
    "this is too long",
    "can we hurry",
    "this is annoying",
    "i hate this",
    "this sucks",
    "bruh",
   ];

  if (driftSignals.some((p) => lower.includes(p))) {
    return {
      state: "drifting",
      level: "refocus",
      nextBehavior: "acknowledgeReconnectContinue",
      evidence: ["driftSignal"],
      confidence: 0.85,
      stage,
    };
  }

  const isShort = text.split(/\s+/).filter(Boolean).length <= 3;
  if (isShort) evidence.push("shortResponseWeakSignal");

  const weakSignals = evidence.filter((e) =>
    [
      "vagueLanguage",
      "uncertaintyLanguage",
      "shortResponseWeakSignal",
      "resistanceTone",
    ].includes(e)
  );

  if (evidence.includes("strongStruggleLanguage") || evidence.includes("frustrationTone")) {
    return {
      state: "struggling",
      level: "nudge",
      nextBehavior: "instructionalNudge",
      evidence,
      confidence: 0.9,
      stage,
    };
  }

  if (weakSignals.length >= 2) {
    return {
      state: "uncertain",
      level: "probe",
      nextBehavior: "probeBeforeSupport",
      evidence,
      confidence: 0.65,
      stage,
    };
  }

  return {
    state: "productive",
    level: "none",
    nextBehavior: "continue",
    evidence,
    confidence: 0.35,
    stage,
  };
}

function selectInstructionalBehavior(state, instructionalState) {
  const currentState = instructionalState?.state || "productive";
  const level = instructionalState?.level || "none";
  const nextBehavior = instructionalState?.nextBehavior || "continue";
  const evidence = instructionalState?.evidence || [];
  const stage = instructionalState?.stage || getStage(state);

  if (currentState === "protected") {
    return {
      behavior: "continueCurrentFlow",
      level: "none",
      stage,
      evidence,
      reason: "A protected pending flow is already active."
    };
  }

  if (currentState === "drifting") {
    return {
      behavior: "refocus",
      level: "refocus",
      stage,
      evidence,
      reason: "Student appears to be drifting away from the current instructional goal."
    };
  }

  if (nextBehavior === "instructionalNudge" || level === "nudge") {
    return {
      behavior: "nudge",
      level: "nudge",
      stage,
      evidence,
      reason: "Student shows evidence of unproductive struggle; provide a targeted instructional nudge."
    };
  }

  if (nextBehavior === "probeBeforeSupport" || level === "probe") {
    return {
      behavior: "probe",
      level: "probe",
      stage,
      evidence,
      reason: "Student shows weak uncertainty signals; ask a light probing question before escalating support."
    };
  }

  return {
    behavior: "continue",
    level: "none",
    stage,
    evidence,
    reason: "Student appears to be in productive struggle or normal progress."
  };
}

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

function cleanFrameText(s) {
  let text = cleanText(s);

  // Common typo cleanup for demo / obvious spelling
  text = text.replace(/\bmovimng\b/gi, "moving");

  // Capitalize first letter
  text = text.charAt(0).toUpperCase() + text.slice(1);

  // Add period for sentence-like responses
  if (text && !/[.!?]$/.test(text)) {
    text += ".";
  }

  return text;
}

function isNegative(s) {
  const t = cleanText(s).toLowerCase();
  return t === "no" || t === "nope" || t === "nah" || t === "n/a" || t === "none";
}

function isStartupCommand(text) {
  const t = cleanText(text).toLowerCase();

  return (
    t === "framing routine" ||
    t === "start" ||
    t === "begin" ||
    t === "new frame" ||
    t === "build a new frame"
  );
}

function isAffirmative(s) {
  const t = cleanText(s).toLowerCase();
  return (
    t === "1" ||
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

function isMetaResponse(s) {
  const t = cleanText(s).toLowerCase();

  return (
    isAffirmative(t) ||
    isNegative(t) ||
    t === "maybe" ||
    t === "i think so" ||
    t === "kind of" ||
    t === "sort of"
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

const GENERIC_KEY_TOPICS = new Set(
  KU_FRAME_COMPONENTS.keyTopic.genericNonExamples || []
);

function isBadKeyTopic(keyTopic) {
  const kt = cleanText(keyTopic).toLowerCase();
  if (!kt) return true;
  if (GENERIC_KEY_TOPICS.has(kt)) return true;
  if (kt.startsWith("my ")) return true;
  return false;
}

function getKeyTopicFeedback(input) {
  const text = cleanText(input);
  const support = getComponentConversation("keyTopic");

  if (!text || isBadKeyTopic(text)) {
    return (
      support.revisePrompt ||
      "That’s a good start, but your Key Topic should name the topic clearly."
    );
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

// Key Topic should clearly name the topic.
// One-word topics are allowed if they are specific.
const wc = keyTopic.split(/\s+/).filter(Boolean).length;
if (wc > 6) return null;

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
// FEEDBACK GAP BANK
// ---------------------
// Controlled internal categories Kaw can choose from.
// Student-facing questions should be generated from the
// gap + section + frame type + purpose + student response.

const FEEDBACK_GAP_BANK = {

  offTopic: {
    priority: 1,
    description:
      "The response does not clearly connect to the selected Frame section."
  },

   adviceInsteadOfInsight: {
    priority: 2,
    description:
      "The response gives advice instead of explaining an insight or message."
  },

  summaryInsteadOfThinking: {
    priority: 3,
    description:
      "The response summarizes what happened instead of explaining deeper thinking."
  },

  missingConnection: {
    priority: 4,
    description:
      "The response needs a clearer connection to the Key Topic, Is About, Main Idea/Cause, or Detail."
  },

  tooBroad: {
    priority: 5,
    description:
      "The response is generally correct but too broad."
  },

  vague: {
    priority: 6,
    description:
      "The response is unclear or uses general language."
  },

  needsSpecificity: {
    priority: 7,
    description:
      "The response needs a more specific example, detail, or explanation."
  }

};

// ---------------------
// STUCK NUDGES
// ---------------------

// ---------------------
// WRITE-MODE GUARDRAILS
// ---------------------
// Evidence detection lives in Formative Assessment.
// Evidence-request interpretation lives in Diagnosis.

// ---------------------
// BUILD MODE LANE GUARDRAILS
// ---------------------
// Light checks only.
// Purpose: keep students in the correct section/lane while building.
// This is NOT Feedback Mode.

function looksLikeSequenceSummary(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  return (
    t.startsWith("first ") ||
    t.startsWith("then ") ||
    t.startsWith("next ") ||
    t.startsWith("finally ") ||
    t.startsWith("the story is about") ||
    t.startsWith("this story is about") ||
    t.startsWith("the text is about") ||
    t.startsWith("this text is about")
  );
}

function looksLikeAdvice(text) {
  const t = cleanText(text).toLowerCase();
  if (!t) return false;

  return (
    t.startsWith("you should") ||
    t.startsWith("people should") ||
    t.includes("should always") ||
    t.includes("should never")
  );
}

// ------------------------------------------------------
// THINKING TASK LIBRARY
// ------------------------------------------------------
// Thinking tasks describe why the student is using the Frame.

const THINKING_TASKS = {

  interpret: {
    label: "Interpret",
    description: "Construct meaning or significance."
  },

  explain: {
    label: "Explain",
    description: "Make ideas, relationships, processes, or reasoning clear."
  },

  analyze: {
    label: "Analyze",
    description: "Break ideas apart to understand patterns, structure, evidence, or reasoning."
  },

  compare: {
    label: "Compare",
    description: "Examine similarities, differences, and relationships."
  },

  evaluate: {
    label: "Evaluate",
    description: "Make and justify judgments using evidence or criteria."
  },

  synthesize: {
    label: "Synthesize",
    description: "Combine ideas into new understanding, conclusions, or solutions."
  },

  reflect: {
    label: "Reflect",
    description: "Examine learning, thinking, revision, or growth."
  }

};

const THINKING_TASK_PRESENTATION = {
  interpret: {
    intro: "Let's build a Frame that helps uncover the deeper meaning and support it with evidence."
  },

  explain: {
    intro: "Let's build a Frame that helps explain your ideas clearly."
  },

  analyze: {
    intro: "Let's build a Frame that helps examine the important parts and how they work together."
  },

  compare: {
    intro: "Let's build a Frame that helps identify the most important similarities and differences."
  },

  evaluate: {
    intro: "Let's build a Frame that helps support your judgment with strong evidence."
  },

  synthesize: {
    intro: "Let's build a Frame that helps connect ideas into a new understanding."
  },

  reflect: {
    intro: "Let's build a Frame that helps capture what you learned and why it matters."
  }
};

// ------------------------------------------------------
// THINKING TASK INFERENCE
//
// Purpose:
// Infer the student's primary cognitive task from the
// complete assignment context.
//
// We intentionally evaluate the original assignment,
// AI understanding, and AI summary together so that
// instructional verbs (Analyze, Evaluate, Compare, etc.)
// are preserved while still benefiting from AI
// clarification.
//
// This inference guides coaching only.
// It NEVER changes the KU Frame structure.
// ------------------------------------------------------

const THINKING_TASK_PATTERNS = {
  interpret: {
    signals: {
      interpret: 5,
      theme: 4,
      "central message": 4,
      lesson: 3,
      symbolism: 3,
      meaning: 2
    }
  },

  explain: {
    signals: {
      explain: 5,
      describe: 4,
      process: 3,
      cause: 2,
      effect: 2,
      relationship: 2,
      how: 1,
      why: 1
    }
  },

  analyze: {
    signals: {
      analyze: 6,
      analysis: 6,
      examine: 4,
      investigate: 4,
      "break down": 4,
      patterns: 3,
      structure: 3,
      evidence: 2,
      why: 2,
      causes: 2
    }
  },

  compare: {
    signals: {
      compare: 6,
      contrast: 6,
      similar: 3,
      different: 3,
      similarities: 3,
      differences: 3
    }
  },

  evaluate: {
    signals: {
      evaluate: 6,
      critique: 5,
      judge: 5,
      assess: 5,
      defend: 5,
      argue: 5,
      arguing: 5,
      persuasive: 5,
      recommend: 4,
      effective: 3,
      quality: 3,
      should: 2
    }
  },

  synthesize: {
    signals: {
      synthesize: 6,
      combine: 4,
      connect: 4,
      integrate: 4,
      conclusion: 3,
      solution: 3,
      "new understanding": 3
    }
  },

  reflect: {
    signals: {
      reflect: 6,
      reflection: 6,
      "self-assess": 5,
      revise: 4,
      revision: 4,
      growth: 4,
      goal: 3,
      learning: 3
    }
  }
};

// ------------------------------------------------------
// THINKING TASK INFERENCE
// Infers the student's primary thinking task from the assignment context.
// ------------------------------------------------------

function inferThinkingTask(state) {
  const assignment = cleanText([
  state?.frameMeta?.assignmentContext?.raw,
  state?.frameMeta?.assignmentContext?.studentSummary,
  state?.frameMeta?.assignmentContext?.understanding
].filter(Boolean).join(" ")).toLowerCase();

  const firstWords = assignment
  .split(/\s+/)
  .slice(0, 5)
  .join(" ");
 
  let bestMode = null;
  let bestScore = 0;
  let evidence = [];

  const firstVerbBonus = {
  interpret: ["interpret"],
  explain: ["explain", "describe"],
  analyze: ["analyze", "examine", "investigate"],
  compare: ["compare", "contrast"],
  evaluate: ["evaluate", "judge", "assess", "defend", "argue"],
  synthesize: ["synthesize", "combine", "connect"],
  reflect: ["reflect"]
};

 for (const [mode, config] of Object.entries(THINKING_TASK_PATTERNS)) {
  let score = 0;
  let matches = [];

 const leadingSignals = firstVerbBonus[mode] || [];

for (const signal of leadingSignals) {
  if (firstWords.startsWith(signal)) {
    score += 10;
    matches.push(`leading:${signal}`);
  }
}

  for (const [signal, weight] of Object.entries(config.signals)) {
    if (assignment.includes(signal.toLowerCase())) {
      score += weight;
      matches.push(signal);
    }
  }

  if (score > bestScore) {
    bestMode = mode;
    bestScore = score;
    evidence = matches;
  }
}

if (!bestMode) {
  return {
    task: null,
    label: "",
    confidence: 0,
    evidence: []
  };
}
 
  return {
    task: bestMode,
    label: THINKING_TASKS[bestMode].label,
    confidence: Math.min(bestScore / 6, 1),
    evidence
  };
}

// ---------------------
// ASSIGNMENT UNDERSTANDING ENGINE
// ---------------------

function evaluateAssignmentUnderstanding(rawAssignment) {
  const assignment = cleanText(rawAssignment);
  const lower = assignment.toLowerCase();
  const words = assignment.split(/\s+/).filter(Boolean);

  const hasEnoughWords = words.length >= 6;

  const hasTaskSignal =
      lower.includes("explain") ||
      lower.includes("describ") ||
      lower.includes("compar") ||
      lower.includes("contrast") ||
      lower.includes("analy") ||
      lower.includes("argu") ||
      lower.includes("show") ||
      lower.includes("identif") ||
      lower.includes("writ") ||
      lower.includes("read");

  const hasTopicSignal = words.length >= 3;

  const needsClarification = !(hasEnoughWords && hasTaskSignal && hasTopicSignal);

  return {
    raw: assignment,
    understanding: assignment,
    confidence: needsClarification ? "low" : "high",
    needsClarification,
    inferredPurpose: "",
    childAnchor: "",
    clarificationCount: 0,
  };
}

async function evaluateAssignmentUnderstandingAI(rawAssignment) {
  const assignment = cleanText(rawAssignment);

  if (!assignment) {
    return evaluateAssignmentUnderstanding(rawAssignment);
  }

  const system = `You analyze a student's assignment description for an AI companion supporting the KU Framing Routine.

Rules:
- Do not teach content.
- Do not answer the assignment.
- Do not create the student's Frame.
- Only determine whether the assignment context is understandable enough to continue coaching.
- Return ONLY compact JSON.`;

 const user = `Student assignment:
"${assignment}"

When creating studentSummary:
- Write as if Kaw is speaking directly to the student.
- Begin with "you're..."
- Preserve important topics: people, places, concepts, books, scientific ideas, etc.
- Preserve the student's thinking task: compare, explain, argue, analyze, identify, evaluate, create, etc.
- Never refer to "the student" or "the assignment."
- Keep it to one conversational sentence.

Return ONLY valid JSON in this format:
{
  "studentSummary": "",
  "understanding": "",
  "confidence": "high",
  "needsClarification": false,
  "inferredPurpose": "",
  "childAnchor": "",
  "reasoningType": ""
}`;

  try {
    const resp = await client.chat.completions.create({
      model: DEFAULT_MODEL,
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    });

    const parsed = JSON.parse(resp?.choices?.[0]?.message?.content || "{}");

    return {
      raw: assignment,
      studentSummary: cleanText(parsed.studentSummary || assignment),
      understanding: cleanText(parsed.understanding || parsed.studentSummary || assignment),
      confidence: parsed.confidence === "high" ? "high" : "low",
      needsClarification: parsed.needsClarification === false ? false : true,
      inferredPurpose: cleanText(parsed.inferredPurpose || ""),
      childAnchor: cleanText(parsed.childAnchor || ""),
      reasoningType: cleanText(parsed.reasoningType || ""),
      clarificationCount: 0,
    };
  } catch {
    return evaluateAssignmentUnderstanding(rawAssignment);
  }
}
function hasSufficientAssignmentUnderstanding(state) {
  const context = state?.frameMeta?.assignmentContext || {};

  return (
    !!context.raw &&
    context.needsClarification === false
  );
}

async function updateAssignmentUnderstanding(state, rawAssignment) {
  const understanding =
    await evaluateAssignmentUnderstandingAI(rawAssignment);

  state.frameMeta.assignmentContext = understanding;
  state.assignmentReasoning = inferThinkingTask(state);
  state.assignmentReasoning.lastUpdated = Date.now();
  console.log("🧠 Assignment Reasoning");
  console.log("----------------------");
  console.log("Task:", state.assignmentReasoning?.task || "None");
  console.log("Label:", state.assignmentReasoning?.label || "None");
  console.log("Confidence:", state.assignmentReasoning?.confidence ?? 0);
  console.log("Evidence:", state.assignmentReasoning?.evidence || []);
    return understanding;
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
  "assignmentContext",
  "purpose",
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

  if (!m.assignmentContext?.raw) return "assignmentContext";
  if (!m.purpose) return "purpose";  
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
  return Array.isArray(state?.frame?.parentItems)
    ? state.frame.parentItems
    : [];
}

// -------------------------------------
// INTERACTION OWNERSHIP PRINCIPLE
// -------------------------------------
//
// Only one instructional mode may own
// the conversation at a time.
//
// Build Mode owns frame construction.
//
// Feedback Mode owns revision and coaching.
//
// Modes must not compete for progression,
// prompting, state mutation, or stage ownership.
//
// Transition between modes must occur
// explicitly through user choice.
//
// BUILD MODE
// Goal: Create a Frame
//
// FEEDBACK MODE
// Goal: Improve a Frame
//
// This principle preserves instructional
// clarity and mirrors teacher practice
// within the KUCRL Framing Routine.
//

// ---------------------
// EXPLICIT MODE SWITCH PRINCIPLE
// ---------------------
//
// If a student appears to request a mode
// change during an active interaction,
// Kaw must confirm before switching.
//
// Feedback Mode may not silently become
// Build Mode.
//
// Build Mode may not silently become
// Feedback Mode.
//
// Mode switching requires explicit
// student confirmation.
//

// -------------------------------------
// INSTRUCTIONAL GOAL PRIORITY PRINCIPLE
// -------------------------------------
//
// Kaw should remain focused on the
// current instructional goal even when
// student responses introduce interesting
// or unrelated topics.
//
// Student responses may provide context,
// but they should not redirect the
// instructional purpose of the interaction.
//
// Acknowledge.
// Reconnect.
// Continue.
//
// The current instructional goal always
// supersedes conversational novelty.
//
// Examples:
// - Build Mode -> Continue building
//   the Frame.
// - Feedback Mode -> Continue improving
//   the selected section.
//
// Kaw may use student responses as
// context for coaching, but should not
// abandon the current instructional task.
//

// -------------------------------------
// FEEDBACK PRIORITIZATION PRINCIPLE
// -------------------------------------
// Feedback Mode may identify multiple gaps.
//
// Kaw should coach the highest-priority gap first.
//
// After revision, gap detection should run again.
//
// Solving a primary gap may naturally resolve
// secondary gaps.
//
// Kaw should avoid overwhelming students by
// addressing multiple gaps simultaneously.
//

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
// - await updateStateFromStudent(state)

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
    if (rawStage === "assignmentContext") return "assignmentContext";
    if (rawStage === "purpose") return "purpose";
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


// ---------------------
// CHILD ANCHOR ADAPTERS
// ---------------------
// In the sandbox phase, child anchors are a thin structural seam only.
// They do not own progression, pending-state routing, or loop control.
// The runtime engine remains owned by getStage(), computeNextQuestion(),
// and await updateStateFromStudent().

// ---------------------
// PARENT ANCHOR OBSERVATION HELPERS
// ---------------------
// These helpers are read-only and sandbox-only in purpose.
// They exist to make the engine easier to inspect structurally.
// They must not be used to alter routing or progression behavior.

function getParentAnchorDisplayLabel(state) {
  const context = getParentAnchorContext(state);
  return context.ownerStructuralStage;
}

function getParentAnchorObservation(state) {
  const context = getParentAnchorContext(state);
  const frameType = state?.frameMeta?.frameType || "";
  const purpose = state?.frameMeta?.purpose || "";
  const ownerLabel = context.ownerStructuralStage;
  const stageLabel = context.structuralStage;

  return {
    ...context,
    frameType,
    purpose,
    ownerLabel,
    stageLabel,

    summary: `${context.ownerStructuralStage} | ${context.loopType} | ${ownerLabel}`,
  };
}

// ---------------------
// STATE
// ---------------------
function defaultState() {
return {
  version: 2,

  interactionMode: "build",

  frameMeta: {
    purpose: "",

    assignmentContext: {
        raw: "",
        understanding: "",
        confidence: "low",
        needsClarification: true,
        inferredPurpose: "",
        childAnchor: "",
        clarificationCount: 0,
    },
},

  feedback: {
    active: false,
    origin: null,
    targetSection: null,
    targetIndex: null,
    originalResponse: "",
    currentResponse: "",
    detectedGaps: [],
    primaryGap: null,
    coachingTurns: 0,
    maxCoachingTurns: 3,
    questionHistory: [],
    studentThinking: [],
    thinkingSummary: "",
    revisionRequested: false,
    modelOffered: false,
    pendingStep: null,
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

assignmentReasoning: {
  task: null,
  label: "",
  confidence: 0,
  evidence: [],
  lastUpdated: null,
},
  };
}

function normalizeIncomingState(raw) {
  const s = raw && typeof raw === "object" ? raw : {};
  const base = defaultState();

  base.interactionMode =
  s.interactionMode || "build";

const assignmentReasoning =
  s.assignmentReasoning && typeof s.assignmentReasoning === "object"
    ? s.assignmentReasoning
    : {};

base.assignmentReasoning = {
  task: assignmentReasoning.task || assignmentReasoning.mode || null,
  label: cleanText(assignmentReasoning.label || ""),
  confidence: Number.isFinite(Number(assignmentReasoning.confidence))
    ? Number(assignmentReasoning.confidence)
    : 0,
  evidence: Array.isArray(assignmentReasoning.evidence)
    ? assignmentReasoning.evidence.map(cleanText).filter(Boolean)
    : [],
  lastUpdated: assignmentReasoning.lastUpdated || null,
};
 
const feedback =
  s.feedback && typeof s.feedback === "object"
    ? s.feedback
    : {};

  base.feedback.active = !!feedback.active;
base.feedback.origin = feedback.origin || null;

base.feedback.targetSection = feedback.targetSection || null;
base.feedback.targetIndex =
  feedback.targetIndex === 0 || feedback.targetIndex
    ? feedback.targetIndex
    : null;

base.feedback.originalResponse = cleanText(feedback.originalResponse || "");
base.feedback.currentResponse = cleanText(feedback.currentResponse || "");

base.feedback.detectedGaps = Array.isArray(feedback.detectedGaps)
  ? feedback.detectedGaps.map(cleanText).filter(Boolean)
  : [];

base.feedback.primaryGap = feedback.primaryGap || null;

base.feedback.coachingTurns = Number.isFinite(Number(feedback.coachingTurns))
  ? Number(feedback.coachingTurns)
  : 0;

base.feedback.maxCoachingTurns = Number.isFinite(Number(feedback.maxCoachingTurns))
  ? Number(feedback.maxCoachingTurns)
  : 3;

base.feedback.questionHistory = Array.isArray(feedback.questionHistory)
  ? feedback.questionHistory.map(cleanText).filter(Boolean)
  : [];

base.feedback.studentThinking = Array.isArray(feedback.studentThinking)
  ? feedback.studentThinking.map(cleanText).filter(Boolean)
  : [];

base.feedback.thinkingSummary = cleanText(feedback.thinkingSummary || "");
base.feedback.revisionRequested = !!feedback.revisionRequested;
base.feedback.modelOffered = !!feedback.modelOffered;
base.feedback.pendingStep = feedback.pendingStep || null;

  const frame = s.frame && typeof s.frame === "object" ? s.frame : {};

  base.frame.keyTopic =
  cleanFrameText(frame.keyTopic || s.keyTopic || "")
    .replace(/[.!?]$/, "");
  base.frame.isAbout = cleanFrameText(frame.isAbout || s.isAbout || "");

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
  base.frame.details = frame.details.map((bucket) =>
    Array.isArray(bucket)
      ? bucket.map(cleanText).filter(Boolean)
      : []
  );
} else if (frame.details && typeof frame.details === "object") {
  // Legacy object format
  const obj = frame.details;

  base.frame.details = base.frame.parentItems.map((mi) => {
    const bucket = obj[mi];
    return Array.isArray(bucket)
      ? bucket.map(cleanText).filter(Boolean)
      : [];
  });
} else {
  base.frame.details = [];
}

  base.frame.soWhat = cleanText(frame.soWhat || s.soWhat || "");

  const frameMeta = s.frameMeta && typeof s.frameMeta === "object" ? s.frameMeta : {};
  base.frameMeta.purpose = cleanText(frameMeta.purpose || "") || "";

  const assignmentContext =
  frameMeta.assignmentContext && typeof frameMeta.assignmentContext === "object"
    ? frameMeta.assignmentContext
    : {};

base.frameMeta.assignmentContext = {
  raw: cleanText(assignmentContext.raw || ""),
  understanding: cleanText(assignmentContext.understanding || ""),
  confidence: cleanText(assignmentContext.confidence || "low") || "low",
  needsClarification:
    typeof assignmentContext.needsClarification === "boolean"
      ? assignmentContext.needsClarification
      : true,
  inferredPurpose: cleanText(assignmentContext.inferredPurpose || ""),
  childAnchor: cleanText(assignmentContext.childAnchor || ""),
  clarificationCount: Number.isFinite(Number(assignmentContext.clarificationCount))
    ? Number(assignmentContext.clarificationCount)
    : 0,
};

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

// ensure detail buckets exist for each parent item
for (let i = 0; i < base.frame.parentItems.length; i++) {
  if (!Array.isArray(base.frame.details[i])) {
    base.frame.details[i] = [];
  }
}
  
return base;
}
  
function ensureBuckets(s) {
  if (!Array.isArray(s.frame.details)) s.frame.details = [];

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
  const ideas = getIdeaList(s);

  lines.push(`KEY TOPIC: ${s.frame.keyTopic}`);
  lines.push(`IS ABOUT: ${s.frame.isAbout}`);
  lines.push("MAIN IDEAS + ESSENTIAL DETAILS:");

  ideas.forEach((mi, i) => {
    lines.push(`Main Idea ${i + 1}: ${mi}`);

    const details = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];

    details.forEach((d, k) => {
      lines.push(`  - Essential Detail ${k + 1}: ${d}`);
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
      s.frame.isAbout = cleanFrameText(`how ${parsed.cause} leads to ${parsed.effect}`);
      s.pending = { type: "confirmIsAbout" };
      return s;
    }

    const effectOnly = cleanText(msg).replace(/[.?!]+$/g, "");
    s.frame.effect = effectOnly;
    s.frame.isAbout = cleanFrameText(`how ${s.frame.keyTopic} leads to ${effectOnly}`);
    s.pending = { type: "confirmIsAbout" };
    return s;
  }

   s.frame.isAbout = cleanFrameText(msg);
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

if (s.pending?.type === "assignmentReasoningIntro") {
  const reasoning = s.assignmentReasoning || {};

  const presentation =
    THINKING_TASK_PRESENTATION[reasoning.task] || {
      thinkingTask: "Organize thinking",
      nextStep: "Decide how you want to use your Frame."
    };

  const intro =
  presentation.intro ||
  "Let's build a Frame that helps organize your thinking.";

  const assignment =
    s.frameMeta?.assignmentContext?.studentSummary ||
    s.frameMeta?.assignmentContext?.raw ||
    "your assignment";

 return (
  "🧠 Great! I understand your assignment.\n\n" +
  `🎯 ${intro}\n\n` +
  "🪜 How can I support your work today?\n\n" +
  "1) Build a new Frame\n" +
  "2) Get feedback on an existing Frame\n\n" +
  "Reply with 1 or 2."
);
}

  if (s.pending?.type === "choosePurpose") {
  return (
    "Great! Let's build a new Frame together.\n\n" +
    "How will you use this Frame to support your work?\n" +
    "1) Study — organize and strengthen your thinking\n" +
    "2) Write — develop a response, essay, or project\n" +
    "3) Read — organize ideas from a text or source\n" +
    "Reply with 1, 2, or 3."
  );
}
  
  if (s.pending?.type === "feedbackSelectSection") {
  return (
    "Which part of your Frame would you like feedback on?\n\n" +
    "1) Key Topic\n" +
    "2) Is About\n" +
    "3) Main Idea / Cause\n" +
    "4) Detail / Evidence\n" +
    "5) Entire Frame\n\n" +
    "Reply with 1-5."
  );
}

  if (s.pending?.type === "feedbackCollectResponse") {
  const sectionLabels = {
    keyTopic: "Key Topic",
    isAbout: "Is About",
    parentItems: "Main Idea / Cause",
    details: "Detail / Evidence",
    entireFrame: "Entire Frame",
  };

  const label =
    sectionLabels[s.feedback.targetSection] ||
    "that section";

  return (
    `Please paste your ${label} response.\n\n` +
    "Kaw will review it and help strengthen your thinking."
  );
}

  if (s.pending?.type === "feedbackCoach") {
  const gap = s.feedback?.primaryGap;

 if (gap === "tooBroad") {
  const turn = s.feedback?.coachingTurns || 0;

  if (turn === 0) {
    return "What additional detail could you add to help explain your thinking more clearly?";
  }

  if (turn === 1) {
    return "What is one specific example, event, or situation that helps illustrate your idea?";
  }

  return "How does that example, event, or situation help explain your main idea or claim?";
}

 if (gap === "vague") {
  const turn = s.feedback?.coachingTurns || 0;

  if (turn === 0) {
    return (
      "Your response is very general right now.\n\n" +
      "What specific event, action, or situation are you thinking about?"
      
    );
  }

  if (turn === 1) {
    return "Can you describe that event, action, or situation in more detail?";
  }

  return "How does that event, action, or situation connect to your main idea or claim?";
}

  if (gap === "needsSpecificity") {

  const turn = s.feedback?.coachingTurns || 0;

  if (turn === 0) {
    return "What is one detail that would help a reader understand your idea better?";
  }

  if (turn === 1) {
    return "Can you describe one example or situation that shows what you mean?";
  }

  return "How does that example support your main idea or claim?";
}

  if (gap === "missingConnection") {
  const turn = s.feedback?.coachingTurns || 0;

  if (turn === 0) {
    return "How does your response connect to your main idea or topic?";
  }

  if (turn === 1) {
    return "What part of your response connects to that idea?";
  }

  return "How does that part support or connect to your idea?";
}

  if (gap === "summaryInsteadOfThinking") {
  const turn = s.feedback?.coachingTurns || 0;

  if (turn === 0) {
    return (
      "Right now, your response tells what happened.\n\n" +
      "What do you think this shows or helps us understand?"
    );
  }

  if (turn === 1) {
    return "What can we learn from that?";
  }

  return "How would you explain that lesson in your own words?";
}

  if (gap === "adviceInsteadOfInsight") {

  const turn = s.feedback?.coachingTurns || 0;

  if (turn === 0) {
    return "What lesson or insight does this suggest, rather than advice someone should follow?";
  }

  if (turn === 1) {
    return "Good. What does this teach readers about people, life, or relationships?";
  }

    return "How could you turn that idea into a complete theme statement in your own words?";
}

  if (gap === "offTopic") {
    return "How does your response connect to the section you selected for feedback?";
  }

  return "Tell me more about your thinking.";
}

  if (s.pending?.type === "feedbackThinkingSummary") {

  return (
    "Let's summarize your thinking.\n\n" +
    "What is your strongest insight or takeaway now?"
  );

}

  if (s.pending?.type === "feedbackRevise") {
  return (
    "Based on your thinking, how would you revise that part of your Frame?"
  );
}

  if (s.pending?.type === "feedbackComplete") {
  return (
    "Nice work.\n\n" +
    "Your feedback session is complete.\n\n" +
    "Would you like to:\n" +
    "1) Revise another part of your Frame\n" +
    "2) Return to your Frame\n\n" +
    "Reply with 1 or 2."
  );
}

  if (s.pending?.type === "chooseWorkflow") {
  return (
    "How can I support your work today?\n" +
    "1) Build a new Frame\n" +
    "2) Get feedback on an existing Frame\n" +
    "Reply with 1 or 2."
  );
}
  
  if (s.pending?.type === "confirmLanguageSwitch") {
    const candNative = s.pending?.candidateNativeName || s.pending?.candidateName || "that language";
    const candName = s.pending?.candidateName || "that language";
    return `I notice you’re writing in ${candName}. Would you like to continue in ${candNative}? (yes/no)`;
  }

 if (s.pending?.type === "reviseKeyTopic") {
  return s.pending.feedback;
}

if (s.pending?.type === "reviseBuildLane") {
  return [
    s.pending.feedback,
    s.pending.revisionPrompt
  ].filter(Boolean).join("\n\n");
}
  
if (s.pending?.type === "stuckConfirm")
  return (
    "🌱 Sounds like you're stuck.\n\n" +
    "Would you like a quick thinking move?\n\n" +
    "1) Yes — Give me a thinking move.\n" +
    "2) No — Let me try again.\n\n" +
    "Reply with 1 or 2."
  );
 
if (s.pending?.type === "stuckMenu") {

  const intro = s.pending?.retryFromMini
    ? "No problem — that smaller question didn’t help enough yet. Let’s try a different help move.\n\n"
    : "";

return (
  intro +
  "📋 Pick a quick thinking move:\n\n" +
  "1) Check directions\n" +
  "2) Re-read source/notes\n" +
  "3) Ask me a smaller question for this step\n" +
  "4) Skip for now and come back\n\n" +
  "Reply with 1–4."
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
        return `Let's check what this step is asking you to do.\n\nRight now you're working on the Details part of the Frame.\n\nYour main idea is:\n"${selectedMainIdea}"\n\nWhat specific detail, example, or explanation could help show how this connects to your message about life?`;      
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
        return `Look back at your notes or source about ${keyTopic}.\n\nDo you see an example, idea, or moment that helps show this message about life:\n"${theme}"\n\nCould that become another main idea in your Frame?`;
      }

      if (isThemes && theme) {
        return `Look back at your notes or source.\n\nDo you see an example, idea, or moment that helps show this message about life:\n"${theme}"\n\nCould that become another main idea in your Frame?`;
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
  const ack =
    tone === "frustration"
      ? "🌱 That can feel frustrating. Let’s try one small step.\n\n"
      : tone === "resistance"
        ? "🌱 I hear you. Let’s use one small thinking move and keep going.\n\n"
        : "🌱 Let’s try one small step.\n\n";

  const nudge = (s.pending.nudgeText || "").toString().trim();

  return `${ack}${nudge}`;
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
  return getComponentPrompt("isAbout", "revisePrompt");
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
  const detailLabel = s.frameMeta?.purpose === "read" ? "Text Evidence" : "Essential Detail";
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
  
if (s.pending?.type === "confirmIsAbout") {
  const isAboutDisplay = (s.frame.isAbout || "")
    .trim()
    .replace(/\.$/, "")
    .replace(/^[A-Z]/, c => c.toLowerCase());

  return getComponentPrompt("isAbout", "confirmationPrompt", {
    keyTopic: s.frame.keyTopic,
    isAbout: isAboutDisplay
  });
}
 
if (s.pending?.type === "confirmMainIdeas") {
  const isCE = s.frameMeta?.frameType === "causeEffect";

  const lines = getIdeaList(s).map((mi, i) =>
    `${isCE ? "Cause" : "Main Idea"} ${i + 1}: ${mi}`
  ).join("\n");

  const label = isCE ? "Causes" : "Main Ideas";

  return getComponentPrompt("mainIdeas", "confirmationPrompt", {
  mainIdeasList: lines
});
}

if (s.pending?.type === "chooseMainIdeaToRevise") {
  const isCE = s.frameMeta?.frameType === "causeEffect";

  const lines = getIdeaList(s).map((mi, i) =>
    `${i + 1}) ${isCE ? "Cause" : "Main Idea"} ${i + 1}: ${mi}`
  ).join("\n");

  return `Which ${isCE ? "Cause" : "Main Idea"} would you like to revise?\n\n${lines}\n\nReply with the number.`;
}

if (s.pending?.type === "reviseMainIdeaAt") {
  const isCE = s.frameMeta?.frameType === "causeEffect";
  const idx = Number(s.pending.index);
  const current = getIdeaList(s)[idx] || "";

  return `Revise ${isCE ? "Cause" : "Main Idea"} ${idx + 1}:\n\n"${current}"\n\nWhat should it say instead?`;
}

if (s.pending?.type === "chooseDetailToRevise") {
  const idx = Number(s.pending.index);
  const arr = Array.isArray(s.frame.details?.[idx]) ? s.frame.details[idx] : [];
  const lineLabel = s.frameMeta?.purpose === "read" ? "Text Evidence" : "Essential Detail";

  const lines = arr.map((d, k) => `${k + 1}) ${lineLabel} ${k + 1}: ${d}`).join("\n");

  return `Which ${lineLabel} would you like to revise?\n\n${lines}\n\nReply with the number.`;
}

if (s.pending?.type === "reviseDetailAt") {
  const idx = Number(s.pending.index);
  const detailIndex = Number(s.pending.detailIndex);
  const current = s.frame.details?.[idx]?.[detailIndex] || "";
  const lineLabel = s.frameMeta?.purpose === "read" ? "Text Evidence" : "Essential Detail";

  return `Revise ${lineLabel} ${detailIndex + 1}:\n\n"${current}"\n\nWhat should it say instead?`;
}
 
if (s.pending?.type === "offerAnotherMainIdea") {
  const isCE = s.frameMeta?.frameType === "causeEffect";
  const count = getIdeaList(s).length;
  const label = isCE ? "Cause" : "Main Idea";

  return (
    `📋 You currently have ${count} ${label}${count > 1 ? "s" : ""}.\n\n` +
    `Would you like to add another ${label}?\n\n` +
    `1) Yes — Add another ${label}.\n` +
    `2) No — Continue.\n\n` +
    `Reply with 1 or 2.`
  );
}

if (s.pending?.type === "collectAnotherMainIdea") {
  const isCE = s.frameMeta?.frameType === "causeEffect";

  if (isCE) {
    return `What is another cause that leads to this effect: "${s.frame.effect}"?`;
  }

  return getComponentPrompt("mainIdeas", "additionalPrompt", {
    keyTopic: s.frame.keyTopic
  });
}
  
if (s.pending?.type === "offerAnotherDetail") {
  const i = Number(s.pending.index);
  const mi = getIdeaList(s)[i] || "";

  const isCE = s.frameMeta?.frameType === "causeEffect";
  const miLabel = isCE ? "Cause" : "Main Idea";
  const dLabel =
    isCE && s.frameMeta?.purpose === "read"
      ? "Text Evidence"
      : "Essential Detail";

  const count = (s.frame.details?.[i] || []).length;

  return (
    `📋 You currently have ${count} ${dLabel}${count > 1 ? "s" : ""} ` +
    `for ${miLabel} ${i + 1}:\n"${mi}"\n\n` +
    `Would you like to add another ${dLabel}?\n\n` +
    `1) Yes — Add another ${dLabel}.\n` +
    `2) No — Continue.\n\n` +
    `Reply with 1 or 2.`
  );
}
  
if (s.pending?.type === "collectAnotherDetail") {
  const i = Number(s.pending.index);
  const mi = getIdeaList(s)[i] || "";  

  const isCE = s.frameMeta?.frameType === "causeEffect";
  const miLabel = isCE ? "Cause" : "Main Idea";
  const dLabel = isCE && s.frameMeta?.purpose === "read" ? "Text Evidence" : "Essential Detail";

  const count = (s.frame.details?.[i] || []).length + 1;

  return `What is ${dLabel} ${count} for ${miLabel} ${i + 1}: "${mi}"?`;
}
  
 if (s.pending?.type === "confirmDetails") {
  const i = Number(s.pending.index);
  const mi = getIdeaList(s)[i] || "";
  const arr = Array.isArray(s.frame.details?.[i]) ? s.frame.details[i] : [];

  const isCE = s.frameMeta?.frameType === "causeEffect";
  const miLabel = isCE ? "Cause" : "Main Idea";
  const dLabel = isCE && s.frameMeta?.purpose === "read" ? "Text Evidence" : "Essential Detail";

  const lines = arr.map((d, k) => `${dLabel} ${k + 1}: ${d}`).join("\n");

  return (
    `✅ Checkpoint\n\n` +
    `For this ${miLabel}: "${mi}", you identified:\n\n` +
    `${lines}\n\n` +
    `Does this accurately capture your thinking?\n\n` +
    `1) Yes — Continue building my Frame.\n` +
    `2) No — Revise one ${dLabel}.\n\n` +
    `Reply with 1 or 2.`
  );
}

if (s.pending?.type === "offerMoreSoWhat") {
  return (
    "📋 You currently have a So What statement.\n\n" +
    "Would you like to add another sentence?\n\n" +
    "1) Yes — Add another sentence.\n" +
    "2) No — Continue.\n\n" +
    "Reply with 1 or 2."
  );
}

if (s.pending?.type === "collectMoreSoWhat") {
  return "Add one more sentence to your So What:";
}

if (s.pending?.type === "confirmSoWhat") {
  return `Your So What is: "${s.frame.soWhat}". Is that correct, or would you like to revise it?`;
}

if (s.pending?.type === "offerExport") {
  return (
    "📋 What would you like to do next?\n\n" +
    "1) Save or print my Frame.\n" +
    "2) Finish without saving.\n\n" +
    "Reply with 1 or 2."
  );
}

if (s.pending?.type === "chooseExportType") {
  return "What would you like to save/print: frame, transcript, or both? (frame/transcript/both)";
}

  // Base progression
  if (!s.frameMeta?.assignmentContext?.raw) {
  return (
  "Hi! 👋 Let's build a great Frame together.\n\n" +
  "First, I'd like to understand what you're working on.\n\n" +
  "What is your assignment asking you to think about, explain, or show?"
);
}

if (!hasSufficientAssignmentUnderstanding(s)) {
  return "Tell me a little more about what your teacher is asking you to think about, explain, or show?";
}
  
if (!s.frameMeta?.purpose) {
  const assignment =
    s.frameMeta?.assignmentContext?.studentSummary ||
    s.frameMeta?.assignmentContext?.raw;
    s.pending = {
     type: "assignmentReasoningIntro"
 };

  return (
  "Thanks—that gives me a better picture of what you're working on.\n\n" +
  `It sounds like ${assignment}.\n\n` +
  "How can I support your work today?\n" +
  "1) Build a new Frame\n" +
  "2) Get feedback on an existing Frame\n" +
  "Reply with 1 or 2."
);
}

if (!s.frame.keyTopic) {
  return getComponentPrompt("keyTopic", "initialPrompt");
}
 
if (!s.frame.isAbout) {
  return getComponentPrompt("isAbout", "initialPrompt", {
    keyTopic: s.frame.keyTopic
  });
}

  const ideas = getIdeaList(s);

if (paStage === "parentItems" || ideas.length < 2) {
  const c = ideas.length;

const label = "Main Idea";

const promptType = c === 0 ? "initialPrompt" : "additionalPrompt";

const fallback = getComponentPrompt("mainIdeas", promptType, {
  keyTopic: s.frame.keyTopic
});

return `${label} ${c + 1}:\n\n${fallback}`;
  }

  // DETAILS LOOP (CLEANED — no duplicate fallback / brace drift)
   for (let i = 0; i < ideas.length; i++) {
    const mi = ideas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (paStage === "detailsLoop" && arr.length < 2) {
      const detailNum = arr.length + 1; // 1 or 2

      const miLabel = "Main Idea";
      const dLabel = "Essential Detail";

const promptType = detailNum === 1 ? "initialPrompt" : "additionalPrompt";

const fallback = getComponentPrompt("details", promptType, {
  mainIdea: mi
});

if (i === 0 && detailNum === 1) {
  return (
    "🎉 Nice work! Your Main Ideas now explain your Key Topic.\n\n" +
    "➡️ Now let's support each Main Idea with Essential Details.\n\n" +
    `${miLabel} ${i + 1}\n` +
    `${mi}\n\n` +
    `✍️ ${dLabel} ${detailNum}\n\n` +
    `${fallback}`
  );
}

if (i > 0 && detailNum === 1) {
  const completedLabel =
    i === 1
      ? "first"
      : i === 2
        ? "second"
        : "previous";

  return (
    `🎉 Nice work! You've supported your ${completedLabel} Main Idea.\n\n` +
    `➡️ Now let's support ${miLabel} ${i + 1}.\n\n` +
    `${miLabel} ${i + 1}\n` +
    `${mi}\n\n` +
    `✍️ ${dLabel} ${detailNum}\n\n` +
    `${fallback}`
  );
}

return (
  `${miLabel} ${i + 1}\n` +
  `${mi}\n\n` +
  `✍️ ${dLabel} ${detailNum}\n\n` +
  `${fallback}`
);
}
}

if (!s.frame.soWhat) {
  return getComponentPrompt("soWhat", "initialPrompt", {
    keyTopic: s.frame.keyTopic
  });
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

async function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);
  ensureBuckets(s);

if (!s.frameMeta) {
  s.frameMeta = {
    purpose: "",
      assignmentContext: {
      raw: "",
      understanding: "",
      confidence: "low",
      needsClarification: true,
      inferredPurpose: "",
      childAnchor: "",
      clarificationCount: 0,
    },
  };
}

if (!s.frameMeta.assignmentContext) {
  s.frameMeta.assignmentContext = {
    raw: "",
    understanding: "",
    confidence: "low",
    needsClarification: true,
    inferredPurpose: "",
    childAnchor: "",
    clarificationCount: 0,
  };
}
  
// Assignment Understanding capture
if (!s.frameMeta.assignmentContext.raw && !(s.pending && s.pending.type)) {

  if (isStartupCommand(msg)) {
    return s;
  }

 await updateAssignmentUnderstanding(s, msg);

if (hasSufficientAssignmentUnderstanding(s)) {
  s.pending = { type: "assignmentReasoningIntro" };
}

return s;
}
 
 // Assignment Understanding clarification
if (
  s.frameMeta.assignmentContext.raw &&
  s.frameMeta.assignmentContext.needsClarification === true &&
  !(s.pending && s.pending.type)
) {
  await updateAssignmentUnderstanding(s, msg);

if (hasSufficientAssignmentUnderstanding(s)) {
  s.pending = { type: "assignmentReasoningIntro" };
}

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

  // ----------------
  // Pending handlers
  // ----------------

if (s.pending?.type === "assignmentReasoningIntro") {
  s.pending = { type: "chooseWorkflow" };
  return await updateStateFromStudent(s, msg);
}
 
 if (s.pending?.type === "chooseWorkflow") {
  const choice = msg.toLowerCase().trim();

if (choice === "1" || choice.includes("build")) {
  s.interactionMode = "build";
  s.feedback.active = false;
  s.pending = { type: "choosePurpose" };
  return s;
}

  if (choice === "2" || choice.includes("feedback")) {
    s.interactionMode = "feedback";
    s.feedback.active = true;
    s.feedback.origin = "standalone";
    s.feedback.pendingStep = "selectSection";
    s.pending = { type: "feedbackSelectSection" };
    return s;
  }

  return s;
}

  if (s.pending?.type === "choosePurpose") {
  const p = normalizePurpose(msg);

   if (p) {
    s.frameMeta.purpose = p;
    s.pending = null;
    return s;
  }

  return s;
}
  
  if (s.pending?.type === "feedbackSelectSection") {
  const choice = msg.toLowerCase().trim();

  const sectionMap = {
    "1": "keyTopic",
    "2": "isAbout",
    "3": "parentItems",
    "4": "details",
    "5": "entireFrame",
  };

  const targetSection =
    sectionMap[choice] ||
    (choice.includes("key") ? "keyTopic" : null) ||
    (choice.includes("about") ? "isAbout" : null) ||
    (choice.includes("main") || choice.includes("cause") ? "parentItems" : null) ||
    (choice.includes("detail") || choice.includes("evidence") ? "details" : null) ||
    (choice.includes("entire") || choice.includes("whole") ? "entireFrame" : null);

  if (!targetSection) return s;

  s.feedback.targetSection = targetSection;
  s.feedback.pendingStep = "collectResponse";
  s.pending = { type: "feedbackCollectResponse" };
  return s;
}

  if (s.pending?.type === "feedbackCollectResponse") {
  s.feedback.originalResponse = msg;
  s.feedback.currentResponse = msg;

  const analysis = analyzeFeedbackResponse(s);

  s.feedback.detectedGaps = analysis.detectedGaps;
  s.feedback.primaryGap = analysis.primaryGap;

  s.feedback.pendingStep = "coach";
  s.pending = { type: "feedbackCoach" };

  return s;
}

 if (s.pending?.type === "feedbackCoach") {
  s.feedback.studentThinking.push(msg);

  const previousGap = s.feedback.primaryGap;
  s.feedback.currentResponse = msg;

  const analysis = analyzeFeedbackResponse(s);
  const stillHasSameGap = analysis.detectedGaps.includes(previousGap);

  s.feedback.detectedGaps = analysis.detectedGaps;
  s.feedback.primaryGap = analysis.primaryGap;

  if (!stillHasSameGap) {
    s.feedback.pendingStep = "complete";
    s.pending = { type: "feedbackComplete" };
    return s;
  }

  s.feedback.coachingTurns += 1;

  if (s.feedback.coachingTurns < s.feedback.maxCoachingTurns) {
    s.pending = { type: "feedbackCoach" };
    return s;
  }

  s.feedback.pendingStep = "thinkingSummary";
  s.pending = { type: "feedbackThinkingSummary" };
  return s;
}

if (s.pending?.type === "feedbackThinkingSummary") {
  s.feedback.thinkingSummary = msg;
  s.feedback.pendingStep = "revise";
  s.pending = { type: "feedbackRevise" };
  return s;
}

  if (s.pending?.type === "feedbackRevise") {
  s.feedback.currentResponse = msg;
  s.feedback.revisionRequested = true;
  s.feedback.pendingStep = "complete";
  s.pending = { type: "feedbackComplete" };
  return s;
}

  if (s.pending?.type === "feedbackComplete") {
  const choice = msg.toLowerCase().trim();

  if (choice === "1" || choice.includes("another") || choice.includes("revise")) {
    s.feedback.targetSection = null;
    s.feedback.targetIndex = null;
    s.feedback.originalResponse = "";
    s.feedback.currentResponse = "";
    s.feedback.detectedGaps = [];
    s.feedback.primaryGap = null;
    s.feedback.coachingTurns = 0;
    s.feedback.questionHistory = [];
    s.feedback.studentThinking = [];
    s.feedback.thinkingSummary = "";
    s.feedback.revisionRequested = false;
    s.feedback.modelOffered = false;
    s.feedback.pendingStep = "selectSection";
    s.pending = { type: "feedbackSelectSection" };
    return s;
  }

  if (choice === "2" || choice.includes("return") || choice.includes("frame")) {
    s.interactionMode = "build";
    s.feedback.active = false;
    s.feedback.pendingStep = null;
    s.pending = null;
    return s;
  }

  return s;
}
  
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

  // Build Mode lane correction follow-up
if (s.pending?.type === "reviseBuildLane") {
  s.pending = null;
  return await updateStateFromStudent(s, msg);
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
    if (isAffirmative(low) || low === "1") {
      s.pending = {
        type: "stuckMenu",
        stage: s.pending.stage || getStage(s),
        tone: s.pending.tone || "neutral",
        resumeQuestion: s.pending.resumeQuestion,
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
      };
      return s;
    }
    if (isNegative(low) || low === "2") {
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
  const stage = s.pending.stage || getStage(s);

  if (isStuckMessage(msg)) {
    s.pending = {
      type: "stuckConfirm",
      stage,
      tone: detectStuckTone(msg),
      resumeQuestion: s.pending.resumeQuestion,
      miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
    };
    return s;
  }

  s.pending = null;
  return await updateStateFromStudent(s, msg);
}

  if (s.pending?.type === "stuckSkip") {
    const low = msg.toLowerCase().trim();
    if (isAffirmative(low) || low === "1") {
      s.pending = {
        type: "stuckMini",
        stage: s.pending.stage || getStage(s),
        miniQuestion: s.pending.miniQuestion || buildMiniQuestion(s),
        resumeQuestion: s.pending.resumeQuestion,
      };
      return s;
    }
    if (isNegative(low) || low === "2") {
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

 if (stage === "keyTopic") {
  const cleaned = cleanText(msg);
  const wc = cleaned.split(/\s+/).filter(Boolean).length;

  if (!s.frame.keyTopic && !isBadKeyTopic(cleaned) && wc <= 6) {
    s.frame.keyTopic = cleanFrameText(cleaned).replace(/[.!?]$/, "");
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
      return await updateStateFromStudent(s, msg);
    }

  if (typeof stage === "string" && stage.startsWith("details:")) {

  if (isMetaResponse(msg)) {
    s.pending = {
      type: "stuckMini",
      stage,
      miniQuestion: "Great — now write the actual detail, example, or explanation in your own words.",
      resumeQuestion: s.pending.resumeQuestion,
    };
    return s;
  }

  s.pending = null;
  return await updateStateFromStudent(s, msg);
}

    if (stage === "soWhat") {
      s.pending = null;
      return await updateStateFromStudent(s, msg);
    }

    s.pending = null;
    return s;
  }

if (s.pending?.type === "confirmIsAbout") {
  const normalized = msg.toLowerCase().trim();

  if (normalized === "1" || isAffirmative(normalized)) {
    s.pending = null;
    return s;
  }

  if (
    normalized === "2" ||
    normalized === "revise" ||
    normalized === "change" ||
    normalized === "edit"
  ) {
    s.pending = { type: "reviseIsAbout" };
    return s;
  }

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

  if (
    normalized === "2" ||
    normalized === "revise" ||
    normalized === "revise one" ||
    normalized === "change" ||
    normalized === "edit"
  ) {
    s.pending = { type: "chooseMainIdeaToRevise" };
    return s;
  }

  return s;
}

  if (s.pending?.type === "chooseMainIdeaToRevise") {
  const normalized = msg.toLowerCase().trim();
  const match = normalized.match(/\d+/);
  const idx = match ? Number(match[0]) - 1 : NaN;
  const ideas = getIdeaList(s);

  if (Number.isInteger(idx) && idx >= 0 && idx < ideas.length) {
    s.pending = { type: "reviseMainIdeaAt", index: idx };
    return s;
  }

  return s;
}

if (s.pending?.type === "reviseMainIdeaAt") {
  const idx = Number(s.pending.index);
  const isCE = s.frameMeta?.frameType === "causeEffect";

  if (isCE) {
    if (Array.isArray(s.frame.causes) && s.frame.causes[idx] !== undefined) {
      s.frame.causes[idx] = msg;
    }
  } else {
    if (Array.isArray(s.frame.parentItems) && s.frame.parentItems[idx] !== undefined) {
      s.frame.parentItems[idx] = msg;
    }
  }

  s.pending = { type: "confirmMainIdeas" };
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

            const laneCheck = analyzeBuildLane(s, "mainIdeas", msg);
      if (laneCheck) {
        s.pending = laneCheck;
        return s;
      }
      
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

 if (isAffirmative(normalized) || normalized === "1") {
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

  if (isNegative(normalized) || normalized === "2") {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  if (arr.length >= 5) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

   if (isWeakFrameResponse(msg)) {
    s.pending = {
      type: "stuckNudge",
      stage: `details:${idx}`,
      tone: detectStuckTone(msg),
      resumeQuestion: buildMiniQuestion(s),
      miniQuestion: buildMiniQuestion(s),
      nudgeText: formatNudgeText(buildStuckNudges(s, `details:${idx}`)),
    };
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
   if (isWeakFrameResponse(msg)) {
    s.pending = {
      type: "stuckNudge",
      stage: `details:${idx}`,
      tone: detectStuckTone(msg),
      resumeQuestion: buildMiniQuestion(s),
      miniQuestion: buildMiniQuestion(s),
      nudgeText: formatNudgeText(buildStuckNudges(s, `details:${idx}`)),
    };
    return s;
  }
  if (shouldRequestEvidenceDetail(s, msg)) {
    s.pending = { type: "writeNeedEvidenceDetail", index: idx, mechanism: msg };
    return s;
  }

  const laneCheck = analyzeBuildLane(s, "details", msg);
  if (laneCheck) {
    s.pending = laneCheck;
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

  if (isAffirmative(normalized) || normalized === "1") {
    s.pending = null;
    return s;
  }

  if (
    normalized === "2" ||
    normalized === "revise" ||
    normalized === "revise one" ||
    normalized === "change" ||
    normalized === "edit"
  ) {
    s.pending = { type: "chooseDetailToRevise", index: idx };
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

 if (s.pending?.type === "chooseDetailToRevise") {
  const normalized = msg.toLowerCase().trim();
  const match = normalized.match(/\d+/);
  const detailIndex = match ? Number(match[0]) - 1 : NaN;
  const idx = Number(s.pending.index);
  const arr = Array.isArray(s.frame.details[idx]) ? s.frame.details[idx] : [];

  if (Number.isInteger(detailIndex) && detailIndex >= 0 && detailIndex < arr.length) {
    s.pending = { type: "reviseDetailAt", index: idx, detailIndex };
    return s;
  }

  return s;
}

if (s.pending?.type === "reviseDetailAt") {
  const idx = Number(s.pending.index);
  const detailIndex = Number(s.pending.detailIndex);
  const normalized = msg.toLowerCase().trim();

// Do not save conversational revision directions as the new detail.
// Keep the student in the same revision step and ask again.
const revisionDirections = [
  "make that stronger",
  "make it stronger",
  "help me revise",
  "help me change",
  "i want to change it",
  "i'd like to change it",
  "actually, make that stronger",
  "actually make that stronger",
  "that doesn't sound right",
  "that doesnt sound right",
  "wait",
  "hold on",
];

if (
  revisionDirections.some(
    (signal) =>
      normalized === signal ||
      normalized.includes(signal)
  )
) {
  return s;
}
 
  // If the student declines, keep the current detail
  // and return to the confirmation checkpoint.
  if (isNegative(normalized)) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  // Do not save vague or stuck responses as revisions.
  if (isWeakFrameResponse(msg)) {
    s.pending = {
      type: "stuckNudge",
      stage: `details:${idx}`,
      tone: detectStuckTone(msg),
      resumeQuestion: buildMiniQuestion(s),
      miniQuestion: buildMiniQuestion(s),
      nudgeText: formatNudgeText(
        buildStuckNudges(s, `details:${idx}`)
      ),
    };
    return s;
  }

  // Replace only the selected detail.
  if (
    Array.isArray(s.frame.details[idx]) &&
    s.frame.details[idx][detailIndex] !== undefined
  ) {
    s.frame.details[idx][detailIndex] = msg;
  }

  // Return to the detail checkpoint.
  s.pending = { type: "confirmDetails", index: idx };
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

    if (isAffirmative(normalized) || normalized === "1") {
        s.pending = { type: "chooseExportType" };
        return s;
    }

    if (isNegative(normalized) || normalized === "2") {
        s.pending = null;
        return s;
    }

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
    if (!s.frame.keyTopic) {
  s.frame.keyTopic = cleanFrameText(parsed.keyTopic).replace(/[.!?]$/, "");
}
  
    if (!s.frame.isAbout) {
      // If write+c/e, enforce leads-to when capturing isAbout
      applyIsAboutCapture(s, parsed.isAbout);
      clearMatchingSkip(s, "isAbout");
    } else {
      s.pending = { type: "confirmIsAbout" };
    }
    return s;
  }

  // 2) Key Topic capture 
   if (!s.frame.keyTopic) {
    const cleaned = cleanText(msg);
    const wc = cleaned.split(/\s+/).filter(Boolean).length;
  
    if (!isBadKeyTopic(cleaned) && wc <= 6) {
      s.frame.keyTopic = cleanFrameText(cleaned).replace(/[.!?]$/, "");
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

        const laneCheck = analyzeBuildLane(s, "mainIdeas", msg);
      if (laneCheck) {
        s.pending = laneCheck;
        return s;
      }
      
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

        const laneCheck = analyzeBuildLane(s, "details", msg);
if (laneCheck) {
  s.pending = laneCheck;
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

// ---------------------
// INSTRUCTIONAL PLAN SNAPSHOT
// Phase 1: read-only planning layer
// ---------------------
 const instructionalContext = buildInstructionalContext(state, message);
 const instructionalPlan = createInstructionalPlan(instructionalContext);
 const instructionalState = detectInstructionalState(state, message);
 const instructionalBehavior = selectInstructionalBehavior(state, instructionalState);
 
 state.instructionalContext = instructionalContext;
 state.instructionalPlan = instructionalPlan;
 state.instructionalState = instructionalState;
 state.instructionalBehavior = instructionalBehavior;

// Optional debug only; does not affect current behavior.
if (state?.settings?.debugInstructionalPlan) {
  console.log("[KAW PLAN]", instructionalPlan);
  console.log("[KAW STATE]", instructionalState);
  console.log("[KAW BEHAVIOR]", instructionalBehavior);
}
    
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
        if (yn === "yes") proceedState = await updateStateFromStudent(state, "yes");
        else if (yn === "no") proceedState = await updateStateFromStudent(state, "no");
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
        proceedState = await updateStateFromStudent(state, message);
      }

      state = proceedState;
    } else if (message) {
      // STUCK detector (global interrupt) — do not interrupt protected pending flows
      const pendingType = state.pending?.type || null;
      const inProtectedPending =
      pendingType === "confirmLanguageSwitch" ||
      pendingType === "assignmentReasoningIntro" ||
      pendingType === "chooseWorkflow" ||
      pendingType === "choosePurpose" ||
      pendingType === "feedbackSelectSection" ||
      pendingType === "feedbackCollectResponse" ||
      pendingType === "feedbackCoach" ||
      pendingType === "feedbackThinkingSummary" ||
      pendingType === "feedbackRevise" ||
      pendingType === "feedbackComplete" ||
      pendingType === "stuckConfirm" ||
      pendingType === "stuckMenu" ||
      pendingType === "stuckReask" ||
      pendingType === "stuckNudge" ||
      pendingType === "stuckMini" ||
      pendingType === "stuckSkip";

    if (
  !inProtectedPending &&
  instructionalBehavior?.behavior === "refocus"
) {
  const currentQuestion = enforceSingleQuestion(
    computeNextQuestion(state)
  );

  let reply =
    "😊 We can come back to that, but let’s finish this part of your Frame first.\n\n" +
    currentQuestion;

  if (
    state.settings.languageLocked &&
    state.settings.language !== "en"
  ) {
    reply = await translateQuestionViaLLM(
      reply,
      state.settings.languageName || "the target language"
    );
  }

  appendTurn(state, "Student", message);
  appendTurn(state, "Kaw", reply);

  return res.status(200).json({ reply, state });
}
     
    if (!inProtectedPending && isStuckMessage(message)) {
     const stage = getStage(state);
     const resumeQuestion = enforceSingleQuestion(computeNextQuestion(state));

 state.pending = {
   type: "stuckNudge",
   stage,
   tone: detectStuckTone(message),
   resumeQuestion,
   miniQuestion: buildMiniQuestion(state),
   nudgeText: formatNudgeText(buildStuckNudges(state, stage)),
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

      state = await updateStateFromStudent(state, message);
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
