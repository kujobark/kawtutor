import OpenAI from "openai";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---------------------
// CONFIG
// ---------------------
const DEFAULT_MODEL =
  process.env.OPENAI_MODEL || "gpt-5.5";

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
    friendlyTerm: "essential detail",

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
// INSTRUCTIONAL PLAYBOOK
// ======================================================
//
// The Instructional Playbook stores predetermined,
// teacher-authored instructional contracts.
//
// The runtime may consult these contracts.
// It must never invent pedagogy.
//
// AI may contextualize a predetermined Thinking Move
// only when the selected contract explicitly allows it.

const INSTRUCTIONAL_PLAYBOOK = {
  isAbout: {
    genuineStruggle: {
      contractId: "IA-GS-001",

      frameComponent: "isAbout",
      instructionalSituation: "genuineStruggle",

      instructionalGoal: "restartThinking",

      teachingMove: "clarify",

      thinkingMove:
        "Explain what the whole Key Topic is about using your own understandable words.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes: true,

      validation: {
        type: "isAbout",
        description:
          "The student provides an Is About statement that observably paraphrases and summarizes the whole Key Topic.",
      },

      resumeBehavior: {
        type: "returnToExactInstructionalLocation",
        description:
          "Return to the Is About statement where support was requested and validate the student's next response.",
      },

      progressiveSupport: {
        principle:
          "If the first intervention does not restart productive thinking, provide progressively more targeted support without supplying the student's Is About statement.",

        scaffolds: [
          {
            level: 1,
            move: "refocus",
            purpose:
              "Reconnect the student to the accepted Key Topic.",
          },
          {
            level: 2,
            move: "remind",
            purpose:
              "Remind the student that Is About explains the whole Key Topic in their own words.",
          },
          {
            level: 3,
            move: "thinkingPrompt",
            purpose:
              "Ask what someone unfamiliar with the topic should understand about it.",
          },
        ],
      },

      studentWorkProtection: {
        preserveExistingWork: true,
        neverSaveStruggleLanguage: true,
        neverGenerateStudentWork: true,
        neverSupplyParaphrase: true,
        neverReplaceKeyTopic: true,
        neverInferMeaning: true,
      },
    },
  },

  details: {
    genuineStruggle: {
      contractId: "ED-GS-001",

      frameComponent: "details",
      instructionalSituation: "genuineStruggle",

      instructionalGoal: "restartThinking",

      teachingMove: "recall",

      thinkingMove:
        "Think of one supporting fact, example, observation, explanation, or piece of evidence that supports this Main Idea.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes: true,

      validation: {
        type: "essentialDetail",
        description:
          "The student provides a clear Essential Detail that directly supports the current Main Idea.",
      },

      resumeBehavior: {
        type: "returnToExactInstructionalLocation",
        description:
          "Return to the same Essential Detail and Main Idea where support was requested.",
      },

      progressiveSupport: {
        principle:
          "If the first intervention does not restart productive thinking, provide progressively more targeted support without supplying the student's Essential Detail.",
      },

      studentWorkProtection: {
        preserveExistingWork: true,
        neverSaveStruggleLanguage: true,
        neverGenerateStudentWork: true,
      },
    },
  },
};

function getInstructionalContract(
  frameComponent,
  instructionalSituation
) {
  const componentContracts =
    INSTRUCTIONAL_PLAYBOOK?.[frameComponent];

  if (!componentContracts) return null;

  return (
    componentContracts?.[instructionalSituation] ||
    null
  );
}

// ======================================================================
// LAYER 6 — INSTRUCTIONAL COMMUNICATION
//
// Purpose:
// Defines the approved instructional communication patterns used to
// express predetermined instructional decisions.
//
// Architectural Ownership:
// • Universal instructional commitments are inherited by every contract.
// • Contracts reference an approved communication pattern.
// • AI contextualizes the selected pattern using accumulated
//   instructional context.
//
// AI never determines instructional communication.
// AI realizes the predetermined communication pattern while preserving
// student ownership and instructional intent.
// ======================================================================

const INSTRUCTIONAL_COMMUNICATION = {

    universal: {

        preserveStudentOwnership: true,

        acknowledgeAuthenticProgressOnly: true,

        advanceOneThinkingStep: true,

        askOneIntentionalQuestion: true,

        useInstructionalRestraint: true,

        supportiveTeachingPresence: true,

        neverGenerateStudentWork: true,

        neverChangeInstructionalDecision: true

    },

    patterns: {

    questionOnly: {
        instruction:
            "Express the predetermined Thinking Move as one concise, natural question."
    },

    acknowledgeThenQuestion: {
        instruction:
            "Briefly acknowledge authentic progress supported by the student's existing work, then express the predetermined Thinking Move as one concise question."
    },

    briefReassuranceThenQuestion: {
        instruction:
            "Use one brief, supportive lead-in that does not imply progress or success, then express the predetermined Thinking Move as one concise question."
    }

}

};

function getInstructionalCommunicationPattern(patternName) {
  const patterns =
    INSTRUCTIONAL_COMMUNICATION?.patterns || {};

  return (
    patterns?.[patternName] ||
    patterns.questionOnly ||
    null
  );
}

// ------------------------------------------------------
// INSTRUCTIONAL COMMUNICATION LICENSE
//
// Converts the deterministic instructional decision into
// explicit permissions and prohibitions for AI expression.
//
// The license does not choose pedagogy.
// It limits how the predetermined Thinking Move may be
// communicated while preserving student ownership.
// ------------------------------------------------------

function buildInstructionalCommunicationLicense(
  execution
) {
  if (!execution) return null;

  const instructionalFinding =
    execution?.instructionalFinding || null;

  const relationshipStatus =
    instructionalFinding?.relationshipStatus || "";

  const communicationPattern =
    execution?.communicationPattern ||
    "questionOnly";

  return {
    contractId:
      execution.contractId,

    instructionalGoal:
      execution.instructionalGoal,

    teachingMove:
      execution.teachingMove,

    requiredThinkingMove:
      execution.thinkingMove,

    communicationPattern,

    permissions: {
      mayAskQuestion: true,

      maximumQuestions: 1,

      mayUseBriefLeadIn:
        communicationPattern ===
          "acknowledgeThenQuestion" ||
        communicationPattern ===
          "briefReassuranceThenQuestion",

      mayAcknowledgeProgress:
        communicationPattern ===
          "acknowledgeThenQuestion",

      mayUseBriefReassurance:
        communicationPattern ===
          "briefReassuranceThenQuestion",

      mayReferenceAssignmentContext: true,

      mayReferenceCurrentMainIdea: true,

      mayReferenceExistingStudentWork: true,
    },

    prohibitions: {
      mayGenerateStudentWork: false,

      mayCompleteStudentWork: false,

      maySupplyEvidence: false,

      mayChangeInstructionalGoal: false,

      mayChangeTeachingMove: false,

      mayChangeThinkingMove: false,

      mayIntroduceNewTeachingMove: false,

      mayInferStudentIntent: false,

      mayInferStudentUnderstanding: false,

      mayInferStudentEmotion: false,

      mayClaimUnsupportedProgress: false,

      mayClaimRelationshipEstablished:
        relationshipStatus === "established",

      mayClaimRelationshipNotEstablished:
        relationshipStatus ===
          "notEstablished",
    },

    relationshipStatus,

    preserveStudentOwnership: true,

    advanceOneThinkingStep: true,
  };
}

// ------------------------------------------------------
// INSTRUCTIONAL COMMUNICATION RESPONSE VALIDATION
//
// Evaluates whether an AI-generated response remained
// within the deterministic Communication License.
//
// This validator does not judge style or instructional
// quality. It checks only observable license conditions.
// ------------------------------------------------------

function validateInstructionalCommunicationResponse(
  response,
  communicationLicense
) {
  const text =
    cleanText(response);

  const lower =
    text.toLowerCase();

  const violations = [];

  const questionCount =
    (text.match(/\?/g) || []).length;

  if (!text) {
    violations.push("emptyResponse");
  }

  if (
    communicationLicense?.permissions
      ?.maximumQuestions === 1 &&
    questionCount !== 1
  ) {
    violations.push("questionCountViolation");
  }

  const unsupportedPraisePatterns = [
    "great job",
    "good job",
    "excellent",
    "nice work",
    "well done",
    "you got it",
    "you are correct",
    "that's correct",
    "that is correct",
    "strong answer",
    "great answer",
  ];

  if (
    communicationLicense?.prohibitions
      ?.mayClaimUnsupportedProgress === false &&
    unsupportedPraisePatterns.some(
      (pattern) => lower.includes(pattern)
    )
  ) {
    violations.push("unsupportedProgressClaim");
  }

  const suppliedWorkPatterns = [
    "you could write",
    "write that",
    "your answer should be",
    "the answer is",
    "use this detail",
    "an example is",
    "for example, teens",
    "for example teens",
  ];

  if (
    communicationLicense?.prohibitions
      ?.mayGenerateStudentWork === false &&
    suppliedWorkPatterns.some(
      (pattern) => lower.includes(pattern)
    )
  ) {
    violations.push("studentWorkSupplied");
  }

  const relationshipClaims = [
    "this supports",
    "that supports",
    "this proves",
    "that proves",
    "this does not support",
    "that does not support",
    "doesn't support",
    "fails to support",
  ];

  const relationshipStatus =
    communicationLicense?.relationshipStatus ||
    "";

  if (
    relationshipStatus === "undetermined" &&
    relationshipClaims.some(
      (pattern) => lower.includes(pattern)
    )
  ) {
    violations.push(
      "unauthorizedRelationshipClaim"
    );
  }

  return {
    valid:
      violations.length === 0,

    questionCount,

    violations,

    response:
      text,
  };
}

// ======================================================
// INSTRUCTIONAL CONTRACT EXECUTION
// ======================================================

function executeInstructionalContract(contract, state) {
  if (!contract) return null;

  switch (contract.contractId) {
    case "IA-GS-001":
      return executeIAGS001(contract, state);

    case "ED-GS-001":
      return executeEDGS001(contract, state);

    default:
      return null;
  }
}

function executeIAGS001(contract, state) {
  const instructionalFinding =
    state?.pending?.instructionalFinding ||
    state?.pending?.resumePending
      ?.instructionalFinding ||
    null;

  const selectedThinkingMove =
    selectIAGS001ThinkingMove(
      instructionalFinding,
      contract.thinkingMove
    );

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

   teachingMove:
    contract.teachingMove,

  thinkingMove:
    selectedThinkingMove,

  communicationPattern:
    contract.communicationPattern ||
    "questionOnly",
    
    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    context: {
      assignmentContext:
        state?.frameMeta?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      keyTopic:
        state?.frame?.keyTopic || "",

      isAbout:
        state?.frame?.isAbout || "",

      currentMainIdea:
        "",

      existingDetails:
        [],
    },
  };
}

function selectIAGS001ThinkingMove(
  instructionalFinding,
  fallbackThinkingMove
) {
  const diagnosis =
    instructionalFinding?.diagnosis || "";

  // --------------------------------------------------
  // NO COMPONENT EVIDENCE
  //
  // The student has not provided observable Is About
  // content that can be evaluated.
  //
  // Kaw reconnects the student to the accepted Key Topic
  // and asks for an explanation of the whole topic without
  // supplying the paraphrase.
  // --------------------------------------------------
    if (
      diagnosis === "emptyResponse" ||
      diagnosis === "noComponentEvidence"
  ) {
    return (
      "Using the accepted Key Topic, invite the student to explain what " +
      "the whole topic is about in their own understandable words. " +
      "Reduce cognitive load without suggesting or generating the " +
      "Is About statement."
    );
  }

  // --------------------------------------------------
  // REPEATS KEY TOPIC
  //
  // The response repeats the Key Topic but does not yet
  // explain or paraphrase what the whole topic is about.
  // --------------------------------------------------
if (diagnosis === "repeatsKeyTopic") {
  return (
    "The student has successfully identified the Key Topic but has not yet explained what it is about. " +

    "Respond like an encouraging teacher working beside a student, not like an assessment system or instructional manual. " +

    "First, briefly acknowledge the student's success identifying the Key Topic. " +
    "Next, naturally redirect their thinking toward what the whole topic is about. " +
    "Finally, ask one simple question inviting them to explain the topic in their own words. " +

    "The response should feel conversational, warm, and supportive. " +

    "Avoid phrases like 'shift from,' 'criteria,' 'statement,' 'validation,' 'correct,' or 'incorrect.' " +
    "Do not explain the rubric or the task requirements. " +
    "Do not provide, suggest, or begin the student's answer."
  );
}

  // --------------------------------------------------
  // INSUFFICIENT OBSERVABLE EVIDENCE
  //
  // The response may contain the beginning of an idea,
  // but there is not enough observable information to
  // establish a whole-topic paraphrase.
  // --------------------------------------------------
  if (
    diagnosis ===
    "insufficientObservableEvidence"
  ) {
    return (
      "Using the accepted Key Topic, ask the student to expand the response " +
      "so someone unfamiliar with the topic could understand what the whole " +
      "topic is about. Do not infer or supply the missing meaning."
    );
  }

  // --------------------------------------------------
  // RELATIONSHIP UNDETERMINED
  //
  // The response contains substantive language, but the
  // observable relationship to the whole Key Topic has
  // not been established.
  // --------------------------------------------------
  if (
    diagnosis ===
    "relationshipUndetermined"
  ) {
    return (
      "Ask the student to make clearer how the response explains the whole " +
      "accepted Key Topic. Preserve the undetermined relationship and do not " +
      "claim the response does or does not paraphrase the topic."
    );
  }

  return fallbackThinkingMove;
}

function selectEDGS001InstructionalDecision(
  instructionalFinding,
  contract
) {
  const diagnosis =
    instructionalFinding?.diagnosis || "";

  const fallbackDecision = {
    teachingMove:
      contract?.teachingMove || "recall",

    thinkingMove:
      contract?.thinkingMove ||
      "Think of one supporting fact, example, observation, explanation, or piece of evidence that supports this Main Idea.",

    communicationPattern:
      contract?.communicationPattern ||
      "briefReassuranceThenQuestion",
  };

  // --------------------------------------------------
  // NO COMPONENT EVIDENCE
  //
  // The student has not provided observable content that
  // can be evaluated as an Essential Detail.
  // --------------------------------------------------
  if (
    diagnosis === "emptyResponse" ||
    diagnosis === "noComponentEvidence"
  ) {
    return {
      teachingMove:
        "reduceCognitiveLoad",

      thinkingMove:
        "Reconnect the student to the accepted Main Idea and invite them to identify one concrete fact, example, observation, explanation, or piece of evidence that could support it. Do not suggest or generate the Essential Detail.",

      communicationPattern:
        "briefReassuranceThenQuestion",
    };
  }

  // --------------------------------------------------
  // INSUFFICIENT OBSERVABLE EVIDENCE
  //
  // The response is too brief, vague, circular, or
  // incomplete to establish a supporting relationship.
  // --------------------------------------------------
  if (
    diagnosis ===
    "insufficientObservableEvidence"
  ) {
    return {
      teachingMove:
        "increaseSpecificity",

      thinkingMove:
        "Ask the student to identify one concrete fact, example, observation, explanation, or piece of evidence related to the accepted Main Idea. Do not infer or supply the missing information.",

      communicationPattern:
        "briefReassuranceThenQuestion",
    };
  }

  // --------------------------------------------------
  // REPEATS MAIN IDEA
  //
  // The student repeated the accepted Main Idea rather
  // than adding a smaller supporting detail.
  // --------------------------------------------------
  if (diagnosis === "repeatsMainIdea") {
    return {
      teachingMove:
        "differentiate",

      thinkingMove:
        "Acknowledge that the student has returned to the Main Idea, then invite them to think smaller by identifying one specific fact, example, observation, explanation, or piece of evidence that helps explain it. Do not provide the Detail.",

      communicationPattern:
        "acknowledgeThenQuestion",
    };
  }

  // --------------------------------------------------
  // RELATIONSHIP INCOMPLETE
  //
  // The response contains substantive content, but the
  // supporting relationship is not explicit enough.
  // --------------------------------------------------
  if (
    diagnosis ===
    "relationshipIncomplete"
  ) {
    return {
      teachingMove:
        "clarifyConnection",

      thinkingMove:
        "Reference the student's observable idea without claiming it already supports the Main Idea. Invite the student to explain how the idea connects to the accepted Main Idea. Do not supply the connection.",

      communicationPattern:
        "acknowledgeThenQuestion",
    };
  }

  // --------------------------------------------------
  // RELATIONSHIP NOT ESTABLISHED
  //
  // The response contains observable content, but no
  // supporting relationship has been established.
  // --------------------------------------------------
  if (
    diagnosis ===
    "relationshipNotEstablished"
  ) {
    return {
      teachingMove:
        "refocus",

      thinkingMove:
        "Redirect the student to the accepted Main Idea and invite them to identify one fact, example, observation, explanation, or piece of evidence that directly supports it. Do not generate a replacement Detail.",

      communicationPattern:
        "briefReassuranceThenQuestion",
    };
  }

  return fallbackDecision;
}

function executeEDGS001(contract, state) {
  const ideas =
    getIdeaList(state).filter(Boolean);

  const resume =
    state?.pending?.resumePending ||
    state?.pending ||
    null;

  const currentMainIdea =
    Number.isInteger(resume?.index)
      ? ideas[resume.index] || ""
      : "";

  // Read the instructional finding established before
  // this contract was activated.
  //
  // The finding describes only the observable instructional
  // condition. It does not infer student intent, emotion,
  // understanding, effort, or meaning.
  const instructionalFinding =
    state?.pending?.instructionalFinding ||
    state?.pending?.resumePending?.instructionalFinding ||
    null;

  // Select the predetermined Thinking Move from the
  // deterministic instructional finding.
  //
  // AI does not choose this move.
   const instructionalDecision =
    selectEDGS001InstructionalDecision(
      instructionalFinding,
      contract
    );

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,
  
    teachingMove:
      instructionalDecision.teachingMove,

    thinkingMove:
      instructionalDecision.thinkingMove,

    communicationPattern:
      instructionalDecision.communicationPattern,

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    context: {
      assignmentContext:
        state?.frameMeta?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      currentMainIdea,

      existingDetails:
        Number.isInteger(resume?.index) &&
        Array.isArray(
          state?.frame?.details?.[resume.index]
        )
          ? state.frame.details[resume.index]
          : [],

      keyTopic:
        state?.frame?.keyTopic || "",

      isAbout:
        state?.frame?.isAbout || "",
    },
  };
}

function buildAIContextualizationPayload(execution) {
  if (!execution?.aiContextualizes) return null;

  return {
    contractId:
      execution.contractId,

    communicationLicense:
  buildInstructionalCommunicationLicense(
    execution
  ),

    instructionalGoal:
      execution.instructionalGoal,

    teachingMove:
      execution.teachingMove,

    thinkingMove:
      execution.thinkingMove,

    communicationPattern:
      execution.communicationPattern || "questionOnly",

    // Carry only the deterministic instructional conclusion.
    // AI may express this finding but may not reinterpret,
    // expand, or replace it.
    instructionalFinding:
      execution?.instructionalFinding || null,

    context: {

      assignmentContext:
        execution?.context?.assignmentContext || {},

      thinkingTask:
        execution?.context?.thinkingTask || {},

      frameComponent:
        execution?.context?.frameComponent || "",

      keyTopic:
        execution?.context?.keyTopic || "",

      isAbout:
        execution?.context?.isAbout || "",

      currentMainIdea:
        execution?.context?.currentMainIdea || "",

      existingDetails:
        Array.isArray(
          execution?.context?.existingDetails
        )
          ? execution.context.existingDetails
          : []
    }
  };
}

// ======================================================
// INSTRUCTIONAL CONTRACT ACTIVATION
// ======================================================

function activateInstructionalContract(contract, state) {
  if (!contract) return null;

  const execution =
    executeInstructionalContract(contract, state);

  if (!execution) return null;

  const aiPayload =
    buildAIContextualizationPayload(execution);

  return {
    contractId: contract.contractId,
    execution,
    aiPayload
  };
}

// ======================================================
// INSTRUCTIONAL RESPONSE
// ======================================================

async function getInstructionalResponse(activation) {
  const payload = activation?.aiPayload;
  console.log("AI PAYLOAD:", payload);
  
  if (!payload) return null;

    const communicationLicense =
    payload?.communicationLicense || null;

  const communicationPattern =
  getInstructionalCommunicationPattern(
    payload.communicationPattern
  );

  const communicationInstruction =
  communicationPattern?.instruction ||
  "Express the predetermined Thinking Move as one concise, natural question.";

  const assignmentContext =
    payload?.context?.assignmentContext || {};

  const assignment =
    assignmentContext.studentSummary ||
    assignmentContext.understanding ||
    assignmentContext.raw ||
    "";

  const thinkingTask =
    payload?.context?.thinkingTask?.label ||
    payload?.context?.thinkingTask?.task ||
    "";

  const currentMainIdea =
    payload?.context?.currentMainIdea || "";

  const existingDetails =
    Array.isArray(payload?.context?.existingDetails)
      ? payload.context.existingDetails
      : [];
  
  // Deterministic instructional conclusions established
  // before AI contextualization.
  //
  // AI may express these conclusions but may not revise,
  // reinterpret, strengthen, weaken, or replace them.
  const instructionalFinding =
    payload?.instructionalFinding || null;
  
   const system = `You are the language contextualization layer for Kaw, a structured instructional companion.

The instructional decision and instructional findings have already been established by a deterministic Instructional Reasoning Engine.

Your only job is to express the predetermined Thinking Move using natural, assignment-specific language.

You must follow these rules:
- The Communication License is authoritative and binding.
- Perform only actions explicitly permitted by the Communication License.
- Never perform an action prohibited by the Communication License, even if it might produce a helpful response.
- Do not rewrite or complete student work.
- Do not change the Instructional Goal, Teaching Move, or Thinking Move.
- Do not reinterpret, expand, weaken, strengthen, or replace the established Instructional Finding.
- Do not infer student intent, understanding, confusion, emotion, effort, motivation, or meaning.
- Do not claim that a response supports or fails to support the Main Idea unless the established Instructional Finding explicitly says so.
- When relationship status is undetermined, preserve that uncertainty and ask for observable information that would make the relationship clearer.
- Preserve student ownership.
- Follow the Approved Communication Instruction exactly.
- Ask exactly one concise question.
- You may include one brief student-facing lead-in before the question only when the Approved Communication Instruction requires it.
- Any acknowledgement must be directly supported by the established Instructional Finding.
- Before responding, silently verify that the response remains within every permission and prohibition in the Communication License.
- Return only the complete student-facing response.`;
  
  const user = `Contract ID:
  ${payload.contractId}

Communication License:
${JSON.stringify(
  communicationLicense || {},
  null,
  2
)}

Instructional Goal:
${payload.instructionalGoal}

Teaching Move:
${payload.teachingMove}

Predetermined Thinking Move:
${payload.thinkingMove}

Approved Communication Pattern:
${payload.communicationPattern || "questionOnly"}

Approved Communication Instruction:
${communicationInstruction}

Established Instructional Finding:
${JSON.stringify(
  instructionalFinding || {},
  null,
  2
)}

Assignment Context:
${assignment || "(not available)"}

Thinking Task:
${thinkingTask || "(not available)"}

Key Topic:
${payload?.context?.keyTopic || "(not available)"}

Is About:
${payload?.context?.isAbout || "(not available)"}

Current Main Idea:
${currentMainIdea || "(not available)"}

Existing Essential Details:
${
  existingDetails.length
    ? existingDetails.join(" | ")
    : "(none yet)"
}

Express the predetermined Thinking Move as one natural, assignment-specific student-facing response.

Use only instructional conclusions explicitly contained in the Established Instructional Finding.

Do not introduce praise, alignment claims, diagnoses, assumptions, or interpretations that were not deterministically established.

Ask exactly one question.`;
  
  try {
    const resp = await client.chat.completions.create({
      model: DEFAULT_MODEL,
      reasoning_effort: "none",
      temperature: 0,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    });

const response =
  resp?.choices?.[0]?.message?.content || "";

if (!response) {
  return null;
}

const communicationValidation =
  validateInstructionalCommunicationResponse(
    response,
    communicationLicense
  );

console.log(
  "COMMUNICATION VALIDATION:",
  communicationValidation
);

if (!communicationValidation.valid) {
  console.warn(
    "AI communication rejected by license:",
    communicationValidation.violations
  );

  return null;
}

return cleanText(response);
  } catch (error) {
    console.error(
      "Instructional contextualization error:",
      error
    );

    return null;
  }
}

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
  const t = cleanText(text)
    .toLowerCase()
    .replace(/[’‘]/g, "'")
    .replace(/[.!?]+$/g, "")
    .trim();

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
// ESSENTIAL DETAIL RELATIONSHIP ANALYSIS
//
// Instructional Contract:
//
// An Essential Detail must explicitly demonstrate how or
// why the student's detail supports the accepted Main Idea.
//
// This analyzer evaluates only observable structure.
// It does not use assignment-specific vocabulary,
// infer unstated meaning, or decide whether a claim is true.
//
// Deterministic outcomes:
//
// - established:
//   The response contains an observable relationship to
//   the Main Idea.
//
// - incomplete:
//   The response contains substantive detail content, but
//   the supporting relationship is not explicit enough to
//   establish without reader inference.
//
// A substantive response is not classified as unrelated
// merely because lexical overlap is absent. That would
// require semantic inference beyond deterministic rules.
// ------------------------------------------------------

const ESSENTIAL_DETAIL_STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "been",
  "being",
  "by",
  "can",
  "could",
  "did",
  "do",
  "does",
  "for",
  "from",
  "had",
  "has",
  "have",
  "he",
  "her",
  "hers",
  "him",
  "his",
  "how",
  "i",
  "in",
  "into",
  "is",
  "it",
  "its",
  "may",
  "might",
  "of",
  "on",
  "or",
  "our",
  "she",
  "should",
  "so",
  "some",
  "than",
  "that",
  "the",
  "their",
  "them",
  "then",
  "there",
  "these",
  "they",
  "this",
  "those",
  "to",
  "us",
  "was",
  "we",
  "were",
  "what",
  "when",
  "where",
  "which",
  "who",
  "why",
  "will",
  "with",
  "would",
  "you",
  "your",
]);

function normalizeInstructionalToken(token) {
  let normalized =
    cleanText(token)
      .toLowerCase()
      .replace(/[^a-z0-9'-]/g, "");

  if (!normalized) return "";

  if (
    normalized.length > 5 &&
    normalized.endsWith("ing")
  ) {
    normalized =
      normalized.slice(0, -3);
  } else if (
    normalized.length > 4 &&
    normalized.endsWith("ed")
  ) {
    normalized =
      normalized.slice(0, -2);
  } else if (
    normalized.length > 4 &&
    normalized.endsWith("es")
  ) {
    normalized =
      normalized.slice(0, -2);
  } else if (
    normalized.length > 3 &&
    normalized.endsWith("s")
  ) {
    normalized =
      normalized.slice(0, -1);
  }

  return normalized;
}

function getInstructionalContentTokens(text) {
  return cleanText(text)
    .toLowerCase()
    .split(/\s+/)
    .map(normalizeInstructionalToken)
    .filter(
      (token) =>
        token &&
        token.length >= 3 &&
        !ESSENTIAL_DETAIL_STOP_WORDS.has(
          token
        )
    );
}

function hasObservableRelationshipLanguage(
  response
) {
  const lower =
    cleanText(response).toLowerCase();

  if (!lower) return false;

  // --------------------------------------------------
  // EXPLICIT RELATIONSHIP CONNECTORS
  //
  // These words and constructions explicitly connect
  // one idea to another through cause, consequence,
  // explanation, interpretation, or support.
  // --------------------------------------------------

  const connectorPatterns = [
    /\bbecause\b/,
    /\bsince\b/,
    /\btherefore\b/,
    /\bthus\b/,
    /\bas a result\b/,
    /\bdue to\b/,
    /\bso that\b/,

    /\bleads?\s+to\b/,
    /\bcaus(?:e|es|ed|ing)\b/,
    /\bresults?\s+in\b/,
    /\bresulted\s+in\b/,
    /\bcontributes?\s+to\b/,

    /\bmakes?\b/,
    /\bmeans?\b/,
    /\baffect(?:s|ed|ing)?\b/,
    /\bimpact(?:s|ed|ing)?\b/,
    /\bincreas(?:e|es|ed|ing)\b/,
    /\bdecreas(?:e|es|ed|ing)\b/,

    /\bshow(?:s|ed|ing)?\b/,
    /\bdemonstrat(?:e|es|ed|ing)\b/,
    /\billustrat(?:e|es|ed|ing)\b/,
    /\breveal(?:s|ed|ing)?\b/,
    /\bindicat(?:e|es|ed|ing)\b/,
    /\bsuggest(?:s|ed|ing)?\b/,

    /\bexplain(?:s|ed|ing)?\b/,
    /\bsupport(?:s|ed|ing)?\b/,
    /\bprov(?:e|es|ed|ing)\b/,
    /\bconfirm(?:s|ed|ing)?\b/,
  ];

  return connectorPatterns.some(
    (pattern) => pattern.test(lower)
  );
}


function analyzeEssentialDetailRelationship(
  response,
  currentMainIdea
) {
  const responseTokens =
    getInstructionalContentTokens(
      response
    );

  const mainIdeaTokens =
    getInstructionalContentTokens(
      currentMainIdea
    );

  const responseTokenSet =
    new Set(responseTokens);

  const sharedTokens =
    [...new Set(mainIdeaTokens)].filter(
      (token) =>
        responseTokenSet.has(token)
    );

  const hasRelationshipLanguage =
    hasObservableRelationshipLanguage(
      response
    );

  const hasObservableConnection =
    hasRelationshipLanguage &&
    sharedTokens.length > 0;

  if (hasObservableConnection) {
    return {
      relationshipStatus:
        "established",

      relationshipEvidence: {
        sharedTokens,

        hasRelationshipLanguage,

        readerInferenceRequired:
          false,
      },
    };
  }

  return {
    relationshipStatus:
      "incomplete",

    relationshipEvidence: {
      sharedTokens,

      hasRelationshipLanguage,

      readerInferenceRequired:
        true,
    },
  };
}

// ------------------------------------------------------
// IS ABOUT RELATIONSHIP ANALYSIS
//
// Instructional Contract:
//
// The Is About statement must paraphrase the Key Topic
// by expressing what the whole topic is about in language
// the student can understand.
//
// This analyzer evaluates only observable evidence.
// It does not determine whether the student's statement
// is factually complete or conceptually accurate beyond
// what deterministic structure can establish.
// ------------------------------------------------------

function analyzeIsAboutRelationship(
  response,
  keyTopic
) {
  const responseTokens =
    getInstructionalContentTokens(response);

  const keyTopicTokens =
    getInstructionalContentTokens(keyTopic);

  const responseTokenSet =
    new Set(responseTokens);

  const sharedTokens =
    [...new Set(keyTopicTokens)].filter(
      (token) =>
        responseTokenSet.has(token)
    );

  const normalizedResponse =
    cleanText(response)
      .toLowerCase()
      .replace(/[.!?]+$/g, "");

  const normalizedKeyTopic =
    cleanText(keyTopic)
      .toLowerCase()
      .replace(/[.!?]+$/g, "");

  const repeatsKeyTopic =
    !!normalizedKeyTopic &&
    normalizedResponse ===
      normalizedKeyTopic;

  if (repeatsKeyTopic) {
    return {
      relationshipStatus:
        "notEstablished",

      relationshipEvidence: {
        sharedTokens,

        repeatsKeyTopic: true,

        readerInferenceRequired:
          false,
      },
    };
  }

  const hasAdditionalMeaning =
    responseTokens.length >
      keyTopicTokens.length;

  const hasLexicalConnection =
    sharedTokens.length > 0;

  const requiresSemanticInference =
    hasAdditionalMeaning &&
    !hasLexicalConnection;

const addsObservableMeaning =
    hasAdditionalMeaning &&
    hasLexicalConnection;

   if (addsObservableMeaning) {
  return {
    relationshipStatus: "established",

    relationshipEvidence: {
      sharedTokens,

      repeatsKeyTopic: false,

      hasAdditionalMeaning,

      hasLexicalConnection,

      requiresSemanticInference,

      readerInferenceRequired: false,
    },
  };
}

    return {
    relationshipStatus:
      "undetermined",

    relationshipEvidence: {
      sharedTokens,

      repeatsKeyTopic:
        false,

      hasAdditionalMeaning,

      hasLexicalConnection,

      requiresSemanticInference,

      readerInferenceRequired:
        true,
    },
  };
}

function validateIsAboutResponse(
  response,
  keyTopic = ""
) {
  const text =
    cleanText(response);

  const words =
    text
      .split(/\s+/)
      .filter(Boolean);

  if (!text) {
    return {
      valid: false,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "emptyResponse",
    };
  }

  if (
    isStuckMessage(text) ||
    isWeakFrameResponse(text) ||
    isMetaResponse(text)
  ) {
    return {
      valid: false,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "noComponentEvidence",
    };
  }

  const relationshipAnalysis =
    analyzeIsAboutRelationship(
      text,
      keyTopic
    );

  if (
    relationshipAnalysis
      .relationshipEvidence
      ?.repeatsKeyTopic
  ) {
    return {
      valid: false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      diagnosis:
        "repeatsKeyTopic",

      relationshipEvidence:
        relationshipAnalysis
          .relationshipEvidence,
    };
  }

  if (words.length < 4) {
    return {
      valid: false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "insufficientObservableEvidence",

      relationshipEvidence:
        relationshipAnalysis
          .relationshipEvidence,
    };
  }

  if (
    relationshipAnalysis
      .relationshipStatus ===
    "established"
  ) {
    return {
      valid: true,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      diagnosis:
        null,

      relationshipEvidence:
        relationshipAnalysis
          .relationshipEvidence,
    };
  }

  return {
    valid: false,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "partiallySatisfied",

    relationshipStatus:
      "undetermined",

    diagnosis:
      "relationshipUndetermined",

    relationshipEvidence:
      relationshipAnalysis
        .relationshipEvidence,
  };
}

  // ------------------------------------------------------
// IS ABOUT SEMANTIC EVIDENCE
//
// Purpose:
//
// Provides narrowly governed semantic evidence only when
// deterministic analysis confirms that the response is
// substantive but lacks lexical overlap with the Key Topic.
//
// AI does not validate the Is About statement.
// AI does not determine progression.
// AI returns semantic evidence only.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

async function getIsAboutSemanticEvidence(
  response,
  keyTopic
) {
  const studentResponse =
    cleanText(response);

  const acceptedKeyTopic =
    cleanText(keyTopic);

  if (
    !studentResponse ||
    !acceptedKeyTopic
  ) {
    return {
      semanticEquivalent: false,
      confidence: 0,
      source: "notRequested",
    };
  }

  const system = `You provide semantic evidence for a deterministic instructional validator supporting the KU Framing Routine.

The accepted Key Topic and the student's proposed Is About statement will be provided.

Determine only whether the student's statement expresses what the whole Key Topic is about using different words.

Rules:
- Do not rewrite the student's response.
- Do not improve the student's response.
- Do not teach the content.
- Do not judge writing quality.
- Do not require the exact Key Topic words to appear.
- Do not treat a related fact, opinion, example, question, or isolated detail as a whole-topic paraphrase.
- Return semantic evidence only.
- Return only the required JSON object.`;

  const user = `Accepted Key Topic:
"${acceptedKeyTopic}"

Student's proposed Is About statement:
"${studentResponse}"

Does the student's statement express what the whole Key Topic is about using different words?`;

  try {
    const resp =
      await client.chat.completions.create({
        model: DEFAULT_MODEL,

        reasoning_effort:
          "none",

        temperature:
          0,

        response_format: {
          type: "json_schema",

          json_schema: {
            name:
              "is_about_semantic_evidence",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                semanticEquivalent: {
                  type:
                    "boolean",
                },

                confidence: {
                  type:
                    "number",

                  minimum:
                    0,

                  maximum:
                    1,
                },
              },

              required: [
                "semanticEquivalent",
                "confidence",
              ],
            },
          },
        },

        messages: [
          {
            role:
              "system",

            content:
              system,
          },

          {
            role:
              "user",

            content:
              user,
          },
        ],
      });

    const parsed =
      JSON.parse(
        resp?.choices?.[0]?.message
          ?.content || "{}"
      );

    const confidence =
      Number(parsed.confidence || 0);

    return {
      semanticEquivalent:
        parsed.semanticEquivalent === true,

      confidence:
        Number.isFinite(confidence)
          ? Math.max(
              0,
              Math.min(confidence, 1)
            )
          : 0,

      source:
        "aiSemanticEvidence",
    };
  } catch (error) {
    console.error(
      "Is About semantic evidence error:",
      error
    );

    return {
      semanticEquivalent:
        false,

      confidence:
        0,

      source:
        "semanticEvidenceUnavailable",
    };
  }
}


// ------------------------------------------------------
// GOVERNED IS ABOUT VALIDATION
//
// Runs deterministic validation first.
//
// Semantic evidence is requested only when deterministic
// evidence explicitly identifies a semantic inference gap.
//
// JavaScript owns the final instructional decision.
// ------------------------------------------------------

async function validateIsAboutResponseGoverned(
  response,
  keyTopic = ""
) {
  const deterministicValidation =
    validateIsAboutResponse(
      response,
      keyTopic
    );

  console.log("IS ABOUT VALIDATION:", deterministicValidation);

  const requiresSemanticInference =
    deterministicValidation
      ?.relationshipEvidence
      ?.requiresSemanticInference === true;

  // Deterministic results remain authoritative unless
  // the analyzer explicitly identifies a semantic gap.
  if (!requiresSemanticInference) {
    return {
      ...deterministicValidation,

      validationSource:
        "deterministic",
    };
  }

  const semanticEvidence =
    await getIsAboutSemanticEvidence(
      response,
      keyTopic
    );

  const semanticRelationshipEstablished =
    semanticEvidence
      .semanticEquivalent === true &&
    semanticEvidence
      .confidence >= 0.9;

  // JavaScript makes the final decision from the
  // deterministic gate and bounded semantic evidence.
  if (semanticRelationshipEstablished) {
    return {
      valid:
        true,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      diagnosis:
        null,

      relationshipEvidence: {
        ...deterministicValidation
          .relationshipEvidence,

        semanticEquivalent:
          true,

        semanticConfidence:
          semanticEvidence.confidence,

        semanticEvidenceSource:
          semanticEvidence.source,

        readerInferenceRequired:
          false,
      },

      validationSource:
        "deterministicWithSemanticEvidence",
    };
  }

  return {
    ...deterministicValidation,

    relationshipEvidence: {
      ...deterministicValidation
        .relationshipEvidence,

      semanticEquivalent:
        semanticEvidence
          .semanticEquivalent,

      semanticConfidence:
        semanticEvidence
          .confidence,

      semanticEvidenceSource:
        semanticEvidence
          .source,
    },

    validationSource:
      "deterministicWithSemanticEvidence",
  };
}

// ------------------------------------------------------
// MAIN IDEA VALIDATION
//
// Instructional Contract:
//
// A Main Idea must express one major organizing idea
// connected to the accepted Key Topic and Is About.
//
// It must be broad enough to organize multiple Essential
// Details and must not function only as one supporting
// fact, example, observation, or explanation.
//
// Deterministic validation handles only observable
// conditions that can be established without semantic
// inference.
//
// Governed semantic evidence is requested for substantive
// responses whose instructional relationship must be
// evaluated in context.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

function validateMainIdeaResponse(
  response,
  keyTopic = "",
  isAbout = ""
) {
  const text =
    cleanText(response);

  // --------------------------------------------------
  // NO COMPONENT EVIDENCE
  // --------------------------------------------------

  if (!text) {
    return {
      valid: false,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "emptyResponse",
    };
  }

  if (
    isStuckMessage(text) ||
    isWeakFrameResponse(text) ||
    isMetaResponse(text)
  ) {
    return {
      valid: false,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "noComponentEvidence",
    };
  }

  const normalizedResponse =
    normalizeInstructionalComparisonText(
      text
    );

  const normalizedKeyTopic =
    normalizeInstructionalComparisonText(
      keyTopic
    );

  const normalizedIsAbout =
    normalizeInstructionalComparisonText(
      isAbout
    );

  // --------------------------------------------------
  // REPEATS KEY TOPIC
  // --------------------------------------------------

  if (
    normalizedKeyTopic &&
    normalizedResponse ===
      normalizedKeyTopic
  ) {
    return {
      valid: false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      diagnosis:
        "repeatsKeyTopic",
    };
  }

  // --------------------------------------------------
  // REPEATS IS ABOUT
  // --------------------------------------------------

  if (
    normalizedIsAbout &&
    normalizedResponse ===
      normalizedIsAbout
  ) {
    return {
      valid: false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      diagnosis:
        "repeatsIsAbout",
    };
  }

  // --------------------------------------------------
  // INSUFFICIENT OBSERVABLE EVIDENCE
  //
  // One-word responses do not provide enough observable
  // evidence to determine whether the response functions
  // as a Main Idea.
  //
  // Multiword Main Ideas remain eligible for governed
  // semantic evidence. This preserves valid concise
  // responses such as "They think before."
  // --------------------------------------------------

  const words =
    text
      .split(/\s+/)
      .filter(Boolean);

  if (words.length < 2) {
    return {
      valid: false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "insufficientObservableEvidence",
    };
  }

  // --------------------------------------------------
  // SEMANTIC INFERENCE GAP
  //
  // The response contains substantive Main Idea content,
  // but whether it functions as a major organizing idea
  // cannot be established through exact comparison alone.
  //
  // No vocabulary list or phrase pattern is used.
  // --------------------------------------------------

  return {
    valid: false,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "partiallySatisfied",

    relationshipStatus:
      "undetermined",

    diagnosis:
      "relationshipUndetermined",

    relationshipEvidence: {
      requiresSemanticInference:
        true,

      readerInferenceRequired:
        true,
    },
  };
}


// ------------------------------------------------------
// MAIN IDEA SEMANTIC EVIDENCE
//
// Purpose:
//
// Provides narrowly governed semantic evidence only after
// deterministic validation confirms that the student has
// supplied substantive Main Idea content.
//
// AI does not save the Main Idea.
// AI does not determine progression.
// AI does not rewrite or improve student work.
// AI returns bounded instructional evidence only.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

async function getMainIdeaSemanticEvidence(
  response,
  keyTopic,
  isAbout
) {
  const studentResponse =
    cleanText(response);

  const acceptedKeyTopic =
    cleanText(keyTopic);

  const acceptedIsAbout =
    cleanText(isAbout);

  if (
    !studentResponse ||
    !acceptedKeyTopic ||
    !acceptedIsAbout
  ) {
    return {
      connectedToKeyTopic:
        false,

      supportsIsAbout:
        false,

      functionsAsOrganizingIdea:
        false,

      supportableWithMultipleDetails:
        false,

      functionsOnlyAsDetail:
        false,

      confidence:
        0,

      source:
        "notRequested",
    };
  }

  const system = `You provide semantic evidence for a deterministic instructional validator supporting the KU Framing Routine.

The accepted Key Topic, accepted Is About statement, and the student's proposed Main Idea will be provided.

Determine only whether the student's response functions as one valid Main Idea within that Frame.

A valid Main Idea:
- expresses one major idea connected to the accepted Key Topic;
- supports or helps organize the accepted Is About statement;
- can function as a heading or organizing idea;
- is broad enough to support multiple meaningful Essential Details;
- is not merely one isolated fact, example, observation, or supporting detail.

Rules:
- Do not rewrite the student's response.
- Do not improve the student's response.
- Do not generate a replacement Main Idea.
- Do not teach the content.
- Do not judge grammar, spelling, style, or factual accuracy.
- Do not require exact words from the Key Topic or Is About statement.
- Do not reject a response merely because it is concise.
- Evaluate the instructional function of the response in this specific Frame.
- Return semantic evidence only.
- Return only the required JSON object.`;

  const user = `Accepted Key Topic:
"${acceptedKeyTopic}"

Accepted Is About statement:
"${acceptedIsAbout}"

Student's proposed Main Idea:
"${studentResponse}"

Does this response function as one major organizing Main Idea in this Frame?`;

  try {
    const resp =
      await client.chat.completions.create({
        model:
          DEFAULT_MODEL,

        reasoning_effort:
          "none",

        temperature:
          0,

        response_format: {
          type:
            "json_schema",

          json_schema: {
            name:
              "main_idea_semantic_evidence",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                connectedToKeyTopic: {
                  type:
                    "boolean",
                },

                supportsIsAbout: {
                  type:
                    "boolean",
                },

                functionsAsOrganizingIdea: {
                  type:
                    "boolean",
                },

                supportableWithMultipleDetails: {
                  type:
                    "boolean",
                },

                functionsOnlyAsDetail: {
                  type:
                    "boolean",
                },

                confidence: {
                  type:
                    "number",

                  minimum:
                    0,

                  maximum:
                    1,
                },
              },

              required: [
                "connectedToKeyTopic",
                "supportsIsAbout",
                "functionsAsOrganizingIdea",
                "supportableWithMultipleDetails",
                "functionsOnlyAsDetail",
                "confidence",
              ],
            },
          },
        },

        messages: [
          {
            role:
              "system",

            content:
              system,
          },

          {
            role:
              "user",

            content:
              user,
          },
        ],
      });

    const parsed =
      JSON.parse(
        resp?.choices?.[0]?.message
          ?.content || "{}"
      );

    const confidence =
      Number(parsed.confidence || 0);

    return {
      connectedToKeyTopic:
        parsed.connectedToKeyTopic ===
        true,

      supportsIsAbout:
        parsed.supportsIsAbout ===
        true,

      functionsAsOrganizingIdea:
        parsed.functionsAsOrganizingIdea ===
        true,

      supportableWithMultipleDetails:
        parsed.supportableWithMultipleDetails ===
        true,

      functionsOnlyAsDetail:
        parsed.functionsOnlyAsDetail ===
        true,

      confidence:
        Number.isFinite(confidence)
          ? Math.max(
              0,
              Math.min(confidence, 1)
            )
          : 0,

      source:
        "aiSemanticEvidence",
    };
  } catch (error) {
    console.error(
      "Main Idea semantic evidence error:",
      error
    );

    return {
      connectedToKeyTopic:
        false,

      supportsIsAbout:
        false,

      functionsAsOrganizingIdea:
        false,

      supportableWithMultipleDetails:
        false,

      functionsOnlyAsDetail:
        false,

      confidence:
        0,

      source:
        "semanticEvidenceUnavailable",
    };
  }
}


// ------------------------------------------------------
// GOVERNED MAIN IDEA VALIDATION
//
// Runs deterministic validation first.
//
// Semantic evidence is requested only when deterministic
// validation identifies a semantic inference gap.
//
// JavaScript applies the instructional contract and makes
// the final validation and progression decision.
// ------------------------------------------------------

async function validateMainIdeaResponseGoverned(
  response,
  keyTopic = "",
  isAbout = ""
) {
  const deterministicValidation =
    validateMainIdeaResponse(
      response,
      keyTopic,
      isAbout
    );

  console.log(
    "MAIN IDEA VALIDATION:",
    deterministicValidation
  );

  const requiresSemanticInference =
    deterministicValidation
      ?.relationshipEvidence
      ?.requiresSemanticInference ===
    true;

  // Deterministic outcomes remain authoritative.
  if (!requiresSemanticInference) {
    return {
      ...deterministicValidation,

      validationSource:
        "deterministic",
    };
  }

  const semanticEvidence =
    await getMainIdeaSemanticEvidence(
      response,
      keyTopic,
      isAbout
    );

  // --------------------------------------------------
  // JAVASCRIPT FINAL DECISION
  //
  // AI provides bounded evidence.
  // JavaScript applies the complete instructional contract.
  // --------------------------------------------------

  const mainIdeaRelationshipEstablished =
    semanticEvidence
      .connectedToKeyTopic === true &&

    semanticEvidence
      .supportsIsAbout === true &&

    semanticEvidence
      .functionsAsOrganizingIdea === true &&

    semanticEvidence
      .supportableWithMultipleDetails === true &&

    semanticEvidence
      .functionsOnlyAsDetail === false &&

    semanticEvidence
      .confidence >= 0.9;

  if (mainIdeaRelationshipEstablished) {
    return {
      valid:
        true,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      diagnosis:
        null,

      relationshipEvidence: {
        ...deterministicValidation
          .relationshipEvidence,

        connectedToKeyTopic:
          semanticEvidence
            .connectedToKeyTopic,

        supportsIsAbout:
          semanticEvidence
            .supportsIsAbout,

        functionsAsOrganizingIdea:
          semanticEvidence
            .functionsAsOrganizingIdea,

        supportableWithMultipleDetails:
          semanticEvidence
            .supportableWithMultipleDetails,

        functionsOnlyAsDetail:
          semanticEvidence
            .functionsOnlyAsDetail,

        semanticConfidence:
          semanticEvidence.confidence,

        semanticEvidenceSource:
          semanticEvidence.source,

        readerInferenceRequired:
          false,
      },

      validationSource:
        "deterministicWithSemanticEvidence",
    };
  }

  return {
    valid:
      false,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "notSatisfied",

    relationshipStatus:
      "notEstablished",

    diagnosis:
      semanticEvidence
        .functionsOnlyAsDetail === true
          ? "detailInsteadOfMainIdea"
          : "relationshipNotEstablished",

    relationshipEvidence: {
      ...deterministicValidation
        .relationshipEvidence,

      connectedToKeyTopic:
        semanticEvidence
          .connectedToKeyTopic,

      supportsIsAbout:
        semanticEvidence
          .supportsIsAbout,

      functionsAsOrganizingIdea:
        semanticEvidence
          .functionsAsOrganizingIdea,

      supportableWithMultipleDetails:
        semanticEvidence
          .supportableWithMultipleDetails,

      functionsOnlyAsDetail:
        semanticEvidence
          .functionsOnlyAsDetail,

      semanticConfidence:
        semanticEvidence.confidence,

      semanticEvidenceSource:
        semanticEvidence.source,
    },

    validationSource:
      "deterministicWithSemanticEvidence",
  };
}

function validateEssentialDetailResponse(
  response,
  currentMainIdea = ""
) {
  const text = cleanText(response);
  const normalized = text.toLowerCase();

  const mainIdea =
    cleanText(currentMainIdea).toLowerCase();

  // --------------------------------------------------
  // ESSENTIAL DETAIL INSTRUCTIONAL RELATIONSHIP
  //
  // An Essential Detail must establish a supporting
  // relationship to the current Main Idea.
  //
  // Kaw may establish only what observable evidence
  // directly supports. When evidence is insufficient,
  // the relationship remains undetermined.
  // --------------------------------------------------

  if (!text) {
    return {
      valid: false,

      componentEvidenceLevel: "none",

      componentCriteriaStatus: "notSatisfied",

      relationshipStatus: "undetermined",

      diagnosis: "emptyResponse",
    };
  }

  if (
    isStuckMessage(text) ||
    isWeakFrameResponse(text) ||
    isMetaResponse(text)
  ) {
    return {
      valid: false,

      componentEvidenceLevel: "none",

      componentCriteriaStatus: "notSatisfied",

      relationshipStatus: "undetermined",

      diagnosis: "noComponentEvidence",
    };
  }

  const circularResponses = new Set([
    "because it does",
    "because they do",
    "because it is",
    "because that happens",
    "it just does",
    "they just do",
    "that is why",
    "because of that",
    "it is true",
    "that is true",
  ]);

  if (circularResponses.has(normalized)) {
    return {
      valid: false,

      componentEvidenceLevel: "none",

      componentCriteriaStatus: "notSatisfied",

      relationshipStatus: "undetermined",

      diagnosis: "insufficientObservableEvidence",
    };
  }

  const words =
    text.split(/\s+/).filter(Boolean);

  if (words.length < 4) {
    return {
      valid: false,

      componentEvidenceLevel: "limited",

      componentCriteriaStatus: "notSatisfied",

      relationshipStatus: "undetermined",

      diagnosis: "insufficientObservableEvidence",
    };
  }

  if (
    mainIdea &&
    normalized === mainIdea
  ) {
    return {
      valid: false,

      componentEvidenceLevel: "limited",

      componentCriteriaStatus: "notSatisfied",

      relationshipStatus: "notEstablished",

      diagnosis: "repeatsMainIdea",
    };
  }

  const relationshipAnalysis =
    analyzeEssentialDetailRelationship(
      text,
      currentMainIdea
    );

  if (
    relationshipAnalysis.relationshipStatus ===
    "established"
  ) {
    return {
      valid: true,

      componentEvidenceLevel: "substantive",

      componentCriteriaStatus: "satisfied",

      relationshipStatus: "established",

      diagnosis: null,

      relationshipEvidence:
        relationshipAnalysis
          .relationshipEvidence
    };
  }

  if (
    relationshipAnalysis.relationshipStatus ===
    "incomplete"
  ) {
    return {
      valid: false,

      componentEvidenceLevel: "substantive",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus: "incomplete",

      diagnosis:
        "relationshipIncomplete",

      relationshipEvidence:
        relationshipAnalysis
          .relationshipEvidence
    };
  }

  return {
    valid: false,

    componentEvidenceLevel: "substantive",

    componentCriteriaStatus:
      "notSatisfied",

    relationshipStatus:
      "notEstablished",

    diagnosis:
      "relationshipNotEstablished",

    relationshipEvidence:
      relationshipAnalysis
        .relationshipEvidence
  };
}

// ------------------------------------------------------
// ESSENTIAL DETAIL SEMANTIC EVIDENCE
//
// Purpose:
//
// Provides narrowly governed semantic evidence only when
// deterministic validation confirms that the student has
// supplied substantive Essential Detail content but the
// supporting relationship cannot be established through
// observable structure alone.
//
// AI does not validate or save the Essential Detail.
// AI does not rewrite or improve student work.
// AI returns bounded instructional evidence only.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

async function getEssentialDetailSemanticEvidence(
  response,
  currentMainIdea,
  instructionalContext = {}
) {
  const studentResponse =
    cleanText(response);

  const acceptedMainIdea =
    cleanText(currentMainIdea);

  const keyTopic =
  cleanText(
    instructionalContext
      ?.keyTopic
  );

const isAbout =
  cleanText(
    instructionalContext
      ?.isAbout
  );

  if (
    !studentResponse ||
    !acceptedMainIdea
  ) {
    return {
      supportsMainIdea:
        false,

      functionsAsEssentialDetail:
        false,

      specificEnough:
        false,

      introducesSeparateMainIdea:
        false,

      confidence:
        0,

      source:
        "notRequested",
    };
  }

  const system = `You provide semantic evidence for a deterministic instructional validator supporting the KU Framing Routine.

The accepted Main Idea and the student's proposed Essential Detail will be provided.

Determine only whether the student's response functions as one valid Essential Detail supporting that Main Idea.

A valid Essential Detail:
- directly supports, explains, illustrates, demonstrates, or provides evidence for the accepted Main Idea;
- adds information that is more specific than the Main Idea;
- can function as a fact, example, observation, explanation, event, or piece of evidence;
- does not merely repeat the Main Idea;
- does not function primarily as a separate major organizing Main Idea.

Rules:
- Do not rewrite the student's response.
- Do not improve the student's response.
- Do not generate a replacement Essential Detail.
- Do not teach the content.
- Do not judge grammar, spelling, style, or factual accuracy.
- Do not require exact words from the Main Idea.
- Do not require explicit connector words such as "because," "shows," or "supports."
- Evaluate the instructional relationship within the complete Frame context provided.
- Essential Details on a Frame may be concise words or phrases rather than complete sentences.
- Do not reduce confidence merely because a valid Essential Detail is brief.
- Confidence represents how clearly the response functions beneath the accepted Main Idea within the supplied Frame—not certainty about outside factual knowledge.
- When all four instructional judgments are clear within the supplied Frame, confidence should normally be 0.90 or higher.
- Return semantic evidence only.
- Return only the required JSON object.
`;
  
const user = `Frame context:

Key Topic:
"${keyTopic || "(not provided)"}"

Is About:
"${isAbout || "(not provided)"}"

Accepted Main Idea:
"${acceptedMainIdea}"

Student's proposed Essential Detail:
"${studentResponse}"

Determine whether the student's response functions as one essential detail beneath the accepted Main Idea within this specific Frame context.`;
  
  try {
    const resp =
      await client.chat.completions.create({
        model:
          DEFAULT_MODEL,

        reasoning_effort:
          "none",

        temperature:
          0,

        response_format: {
          type:
            "json_schema",

          json_schema: {
            name:
              "essential_detail_semantic_evidence",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                supportsMainIdea: {
                  type:
                    "boolean",
                },

                functionsAsEssentialDetail: {
                  type:
                    "boolean",
                },

                specificEnough: {
                  type:
                    "boolean",
                },

                introducesSeparateMainIdea: {
                  type:
                    "boolean",
                },

                confidence: {
                  type:
                    "number",

                  minimum:
                    0,

                  maximum:
                    1,
                },
              },

              required: [
                "supportsMainIdea",
                "functionsAsEssentialDetail",
                "specificEnough",
                "introducesSeparateMainIdea",
                "confidence",
              ],
            },
          },
        },

        messages: [
          {
            role:
              "system",

            content:
              system,
          },

          {
            role:
              "user",

            content:
              user,
          },
        ],
      });

    const parsed =
      JSON.parse(
        resp?.choices?.[0]?.message
          ?.content || "{}"
      );

    const confidence =
      Number(
        parsed.confidence || 0
      );

    return {
      supportsMainIdea:
        parsed.supportsMainIdea ===
        true,

      functionsAsEssentialDetail:
        parsed.functionsAsEssentialDetail ===
        true,

      specificEnough:
        parsed.specificEnough ===
        true,

      introducesSeparateMainIdea:
        parsed.introducesSeparateMainIdea ===
        true,

      confidence:
        Number.isFinite(confidence)
          ? Math.max(
              0,
              Math.min(
                confidence,
                1
              )
            )
          : 0,

      source:
        "aiSemanticEvidence",
    };
  } catch (error) {
    console.error(
      "Essential Detail semantic evidence error:",
      error
    );

    return {
      supportsMainIdea:
        false,

      functionsAsEssentialDetail:
        false,

      specificEnough:
        false,

      introducesSeparateMainIdea:
        false,

      confidence:
        0,

      source:
        "semanticEvidenceUnavailable",
    };
  }
}


// ------------------------------------------------------
// GOVERNED ESSENTIAL DETAIL VALIDATION
//
// Runs deterministic validation first.
//
// Semantic evidence is requested only when deterministic
// validation identifies a substantive response whose
// relationship to the accepted Main Idea requires semantic
// inference.
//
// JavaScript applies the instructional contract and makes
// the final validation and progression decision.
// ------------------------------------------------------

async function validateEssentialDetailResponseGoverned(
  response,
  currentMainIdea = "",
  instructionalContext = {}
) {
  const deterministicValidation =
    validateEssentialDetailResponse(
      response,
      currentMainIdea
    );

  console.log(
    "ESSENTIAL DETAIL VALIDATION:",
    deterministicValidation
  );

  const semanticInferenceDiagnoses = [
  "insufficientObservableEvidence",
  "relationshipIncomplete",
  "relationshipNotEstablished",
];

const limitedResponseCanBeReviewed =
  deterministicValidation
    ?.componentEvidenceLevel ===
    "limited" &&

  semanticInferenceDiagnoses.includes(
    deterministicValidation
      ?.diagnosis
  );

const substantiveResponseCanBeReviewed =
  deterministicValidation
    ?.componentEvidenceLevel ===
    "substantive" &&

  (
    deterministicValidation
      ?.relationshipEvidence
      ?.readerInferenceRequired ===
      true ||

    semanticInferenceDiagnoses.includes(
      deterministicValidation
        ?.diagnosis
    )
  );

const requiresSemanticInference =
  limitedResponseCanBeReviewed ||
  substantiveResponseCanBeReviewed;

  // Deterministic outcomes remain authoritative when
  // semantic inference is not explicitly required.
  if (!requiresSemanticInference) {
    return {
      ...deterministicValidation,

      validationSource:
        "deterministic",
    };
  }

  const semanticEvidence =
  await getEssentialDetailSemanticEvidence(
    response,
    currentMainIdea,
    instructionalContext
  );

  // --------------------------------------------------
  // JAVASCRIPT FINAL DECISION
  //
  // AI provides bounded evidence.
  // JavaScript applies the complete instructional contract.
  // --------------------------------------------------

  const essentialDetailRelationshipEstablished =
    semanticEvidence
      .supportsMainIdea === true &&

    semanticEvidence
      .functionsAsEssentialDetail === true &&

    semanticEvidence
      .specificEnough === true &&

    semanticEvidence
      .introducesSeparateMainIdea === false &&

    semanticEvidence
      .confidence >= 0.9;

  if (
    essentialDetailRelationshipEstablished
  ) {
    return {
      valid:
        true,

      componentEvidenceLevel:
        deterministicValidation
          .componentEvidenceLevel,

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      diagnosis:
        null,

      relationshipEvidence: {
        ...deterministicValidation
          .relationshipEvidence,

        supportsMainIdea:
          semanticEvidence
            .supportsMainIdea,

        functionsAsEssentialDetail:
          semanticEvidence
            .functionsAsEssentialDetail,

        specificEnough:
          semanticEvidence
            .specificEnough,

        introducesSeparateMainIdea:
          semanticEvidence
            .introducesSeparateMainIdea,

        semanticConfidence:
          semanticEvidence.confidence,

        semanticEvidenceSource:
          semanticEvidence.source,

        readerInferenceRequired:
          false,
      },

      validationSource:
        "deterministicWithSemanticEvidence",
    };
  }

  return {
    valid:
      false,

     componentEvidenceLevel:
      deterministicValidation
        .componentEvidenceLevel,

    componentCriteriaStatus:
      "notSatisfied",

    relationshipStatus:
      "notEstablished",

    diagnosis:
      semanticEvidence
        .introducesSeparateMainIdea ===
        true
          ? "mainIdeaInsteadOfDetail"
          : "relationshipNotEstablished",

    relationshipEvidence: {
      ...deterministicValidation
        .relationshipEvidence,

      supportsMainIdea:
        semanticEvidence
          .supportsMainIdea,

      functionsAsEssentialDetail:
        semanticEvidence
          .functionsAsEssentialDetail,

      specificEnough:
        semanticEvidence
          .specificEnough,

      introducesSeparateMainIdea:
        semanticEvidence
          .introducesSeparateMainIdea,

      semanticConfidence:
        semanticEvidence.confidence,

      semanticEvidenceSource:
        semanticEvidence.source,
    },

    validationSource:
      "deterministicWithSemanticEvidence",
  };
}

// ------------------------------------------------------
// SO WHAT VALIDATION
//
// Instructional Contract:
//
// The So What communicates an important understanding
// about the accepted Key Topic that is supported by the
// completed Frame.
//
// The completed Frame includes:
//
// - Assignment Context
// - Thinking Task
// - Key Topic
// - Is About
// - Main Ideas
// - Essential Details
//
// A successful So What must:
//
// - remain anchored to the Key Topic;
// - be traceable to the completed Frame;
// - be supported by the completed Frame;
// - communicate a meaningful understanding or takeaway;
// - go beyond merely repeating an earlier Frame component.
//
// The So What may take different rhetorical forms,
// including a conclusion, definition, principle, theme,
// implication, recommendation, warning, value statement,
// generalization, or call to action.
//
// Rhetorical form does not determine validity.
//
// Deterministic validation handles only observable
// conditions that can be established without semantic
// inference.
//
// Governed semantic evidence evaluates the student's
// synthesis within the completed Frame.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

function validateSoWhatResponse(
  response,
  instructionalContext = {}
) {
  const text =
    cleanText(response);

  const keyTopic =
    cleanText(
      instructionalContext?.keyTopic
    );

  const isAbout =
    cleanText(
      instructionalContext?.isAbout
    );

  const mainIdeas =
    Array.isArray(
      instructionalContext?.mainIdeas
    )
      ? instructionalContext.mainIdeas
          .map(cleanText)
          .filter(Boolean)
      : [];

  const details =
    Array.isArray(
      instructionalContext?.details
    )
      ? instructionalContext.details
          .flatMap((bucket) =>
            Array.isArray(bucket)
              ? bucket
              : []
          )
          .map(cleanText)
          .filter(Boolean)
      : [];

  // --------------------------------------------------
  // NO COMPONENT EVIDENCE
  // --------------------------------------------------

  if (!text) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      synthesisState:
        "none",

      diagnosis:
        "emptyResponse",
    };
  }

  if (
    isStuckMessage(text) ||
    isWeakFrameResponse(text) ||
    isMetaResponse(text)
  ) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      synthesisState:
        "none",

      diagnosis:
        "noComponentEvidence",
    };
  }

  const normalizedResponse =
    normalizeInstructionalComparisonText(
      text
    );

  const normalizedKeyTopic =
    normalizeInstructionalComparisonText(
      keyTopic
    );

  const normalizedIsAbout =
    normalizeInstructionalComparisonText(
      isAbout
    );

  const normalizedMainIdeas =
    mainIdeas.map(
      normalizeInstructionalComparisonText
    );

  const normalizedDetails =
    details.map(
      normalizeInstructionalComparisonText
    );

  // --------------------------------------------------
  // REPEATS KEY TOPIC
  //
  // Naming the topic again does not communicate a
  // culminating understanding from the completed Frame.
  // --------------------------------------------------

  if (
    normalizedKeyTopic &&
    normalizedResponse ===
      normalizedKeyTopic
  ) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      synthesisState:
        "none",

      diagnosis:
        "repeatsKeyTopic",
    };
  }

  // --------------------------------------------------
  // REPEATS IS ABOUT
  //
  // Repeating the whole-topic paraphrase does not yet
  // synthesize the completed Frame.
  // --------------------------------------------------

  if (
    normalizedIsAbout &&
    normalizedResponse ===
      normalizedIsAbout
  ) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      synthesisState:
        "none",

      diagnosis:
        "repeatsIsAbout",
    };
  }

  // --------------------------------------------------
  // REPEATS ONE MAIN IDEA
  //
  // A So What should emerge from the completed Frame,
  // not merely repeat one organizing idea.
  // --------------------------------------------------

  if (
    normalizedMainIdeas.includes(
      normalizedResponse
    )
  ) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      synthesisState:
        "none",

      diagnosis:
        "repeatsMainIdea",
    };
  }

  // --------------------------------------------------
  // REPEATS ONE ESSENTIAL DETAIL
  //
  // Repeating one supporting detail does not communicate
  // a culminating understanding.
  // --------------------------------------------------

  if (
    normalizedDetails.includes(
      normalizedResponse
    )
  ) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      synthesisState:
        "none",

      diagnosis:
        "repeatsEssentialDetail",
    };
  }

  const words =
    text
      .split(/\s+/)
      .filter(Boolean);

  // --------------------------------------------------
  // INSUFFICIENT OBSERVABLE EVIDENCE
  //
  // Very short responses do not provide enough observable
  // evidence to establish synthesis.
  //
  // They remain eligible for instructional coaching but
  // are not sent for semantic validation.
  // --------------------------------------------------

  if (words.length < 4) {
    return {
      valid:
        false,

      componentEvidenceLevel:
        "limited",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus:
        "undetermined",

      synthesisState:
        "emerging",

      diagnosis:
        "insufficientObservableEvidence",
    };
  }

  // --------------------------------------------------
  // SEMANTIC INFERENCE GAP
  //
  // The response contains substantive content.
  //
  // Whether it communicates a supported culminating
  // understanding must be evaluated within the complete
  // Frame context.
  // --------------------------------------------------

  return {
    valid:
      false,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "partiallySatisfied",

    relationshipStatus:
      "undetermined",

    synthesisState:
      "undetermined",

    diagnosis:
      "synthesisUndetermined",

    relationshipEvidence: {
      requiresSemanticInference:
        true,

      readerInferenceRequired:
        true,
    },
  };
}


// ------------------------------------------------------
// SO WHAT SEMANTIC EVIDENCE
//
// Purpose:
//
// Provides narrowly governed semantic evidence only after
// deterministic validation confirms that the student has
// supplied substantive So What content.
//
// AI does not validate or save the So What.
// AI does not rewrite or improve student work.
// AI does not determine progression.
// AI does not require one predetermined conclusion.
//
// AI evaluates only whether the student's response
// functions as a supported culminating understanding
// within the supplied completed Frame.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

async function getSoWhatSemanticEvidence(
  response,
  instructionalContext = {}
) {
  const studentResponse =
    cleanText(response);

  const assignmentContext =
    instructionalContext
      ?.assignmentContext || {};

  const assignment =
    cleanText(
      assignmentContext
        ?.understanding ||
      assignmentContext
        ?.studentSummary ||
      assignmentContext
        ?.raw ||
      ""
    );

  const thinkingTask =
    cleanText(
      instructionalContext
        ?.thinkingTask?.label ||
      instructionalContext
        ?.thinkingTask?.task ||
      ""
    );

  const keyTopic =
    cleanText(
      instructionalContext?.keyTopic
    );

  const isAbout =
    cleanText(
      instructionalContext?.isAbout
    );

  const mainIdeas =
    Array.isArray(
      instructionalContext?.mainIdeas
    )
      ? instructionalContext.mainIdeas
          .map(cleanText)
          .filter(Boolean)
      : [];

  const detailBuckets =
    Array.isArray(
      instructionalContext?.details
    )
      ? instructionalContext.details
      : [];

  const completedFrame =
    mainIdeas.map(
      (mainIdea, index) => {
        const details =
          Array.isArray(
            detailBuckets[index]
          )
            ? detailBuckets[index]
                .map(cleanText)
                .filter(Boolean)
            : [];

        return {
          mainIdea,
          details,
        };
      }
    );

  if (
    !studentResponse ||
    !keyTopic
  ) {
    return {
      anchoredToKeyTopic:
        false,

      traceableToCompletedFrame:
        false,

      supportedByCompletedFrame:
        false,

      communicatesMeaningfulUnderstanding:
        false,

      specificEnoughToUnderstand:
        false,

      merelyRepeatsEarlierFrameContent:
        false,

      confidence:
        0,

      source:
        "notRequested",
    };
  }

  const system = `You provide semantic evidence for a deterministic instructional validator supporting the KU Framing Routine.

The student's assignment context, Thinking Task, completed Frame, and proposed So What will be provided.

Determine only whether the student's response functions as a supported culminating understanding within that specific completed Frame.

The So What communicates an important understanding about the Key Topic that is supported by the completed Frame.

A successful So What:
- remains anchored to the accepted Key Topic;
- can be traced to ideas or evidence developed in the completed Frame;
- is supported by the completed Frame;
- communicates a meaningful understanding or takeaway;
- is specific enough for a reader to understand the student's takeaway;
- goes beyond merely repeating the Key Topic, Is About, one Main Idea, or one Essential Detail.

The So What may naturally function as:
- a conclusion;
- a definition;
- a principle;
- a theme;
- an implication;
- a consequence;
- a recommendation;
- a warning;
- a generalization;
- a value statement;
- a call to action;
- another supported culminating understanding.

Rules:
- Do not require one predetermined or uniquely correct So What.
- Do not reject a response merely because it expresses an opinion, recommendation, value statement, implication, or warning.
- An opinion or recommendation must still be traceable to and supported by the completed Frame.
- Do not require the response to summarize every Main Idea or Detail.
- Do not require exact vocabulary from the completed Frame.
- Do not require a particular rhetorical form.
- Do not judge grammar, spelling, style, or outside factual accuracy.
- Do not rewrite the student's response.
- Do not improve the student's response.
- Do not generate a replacement So What.
- Do not teach the content.
- Do not add evidence that is absent from the completed Frame.
- Evaluate only the instructional function of the student's response within the supplied Frame.
- Confidence represents how clearly the instructional relationship can be established from the supplied Frame.
- Return semantic evidence only.
- Return only the required JSON object.`;

  const user = `Assignment Context:
"${assignment || "(not provided)"}"

Thinking Task:
"${thinkingTask || "(not provided)"}"

Accepted Key Topic:
"${keyTopic}"

Accepted Is About:
"${isAbout || "(not provided)"}"

Completed Frame:
${JSON.stringify(
  completedFrame,
  null,
  2
)}

Student's proposed So What:
"${studentResponse}"

Determine whether this response functions as a supported culminating understanding within this completed Frame.`;

  try {
    const resp =
      await client.chat.completions.create({
        model:
          DEFAULT_MODEL,

        reasoning_effort:
          "none",

        temperature:
          0,

        response_format: {
          type:
            "json_schema",

          json_schema: {
            name:
              "so_what_semantic_evidence",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                anchoredToKeyTopic: {
                  type:
                    "boolean",
                },

                traceableToCompletedFrame: {
                  type:
                    "boolean",
                },

                supportedByCompletedFrame: {
                  type:
                    "boolean",
                },

                communicatesMeaningfulUnderstanding: {
                  type:
                    "boolean",
                },

                specificEnoughToUnderstand: {
                  type:
                    "boolean",
                },

                merelyRepeatsEarlierFrameContent: {
                  type:
                    "boolean",
                },

                confidence: {
                  type:
                    "number",

                  minimum:
                    0,

                  maximum:
                    1,
                },
              },

              required: [
                "anchoredToKeyTopic",
                "traceableToCompletedFrame",
                "supportedByCompletedFrame",
                "communicatesMeaningfulUnderstanding",
                "specificEnoughToUnderstand",
                "merelyRepeatsEarlierFrameContent",
                "confidence",
              ],
            },
          },
        },

        messages: [
          {
            role:
              "system",

            content:
              system,
          },

          {
            role:
              "user",

            content:
              user,
          },
        ],
      });

    const parsed =
      JSON.parse(
        resp?.choices?.[0]?.message
          ?.content || "{}"
      );

    const confidence =
      Number(
        parsed.confidence || 0
      );

    return {
      anchoredToKeyTopic:
        parsed.anchoredToKeyTopic ===
        true,

      traceableToCompletedFrame:
        parsed
          .traceableToCompletedFrame ===
        true,

      supportedByCompletedFrame:
        parsed
          .supportedByCompletedFrame ===
        true,

      communicatesMeaningfulUnderstanding:
        parsed
          .communicatesMeaningfulUnderstanding ===
        true,

      specificEnoughToUnderstand:
        parsed
          .specificEnoughToUnderstand ===
        true,

      merelyRepeatsEarlierFrameContent:
        parsed
          .merelyRepeatsEarlierFrameContent ===
        true,

      confidence:
        Number.isFinite(confidence)
          ? Math.max(
              0,
              Math.min(
                confidence,
                1
              )
            )
          : 0,

      source:
        "aiSemanticEvidence",
    };
  } catch (error) {
    console.error(
      "So What semantic evidence error:",
      error
    );

    return {
      anchoredToKeyTopic:
        false,

      traceableToCompletedFrame:
        false,

      supportedByCompletedFrame:
        false,

      communicatesMeaningfulUnderstanding:
        false,

      specificEnoughToUnderstand:
        false,

      merelyRepeatsEarlierFrameContent:
        false,

      confidence:
        0,

      source:
        "semanticEvidenceUnavailable",
    };
  }
}


// ------------------------------------------------------
// GOVERNED SO WHAT VALIDATION
//
// Runs deterministic validation first.
//
// Semantic evidence is requested only when deterministic
// validation identifies substantive So What content whose
// relationship to the completed Frame requires semantic
// inference.
//
// JavaScript applies the complete instructional contract
// and determines the student's synthesis state.
//
// Supported synthesis:
// The response satisfies all four instructional
// constraints and is specific enough to understand.
//
// Emerging synthesis:
// The response remains anchored, traceable, and supported,
// but the culminating understanding needs greater meaning,
// specificity, or distinction from earlier Frame content.
//
// Unsupported synthesis:
// The response cannot be sufficiently anchored, traced,
// or supported by the completed Frame.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

async function validateSoWhatResponseGoverned(
  response,
  instructionalContext = {}
) {
  const deterministicValidation =
    validateSoWhatResponse(
      response,
      instructionalContext
    );

  console.log(
    "SO WHAT VALIDATION:",
    deterministicValidation
  );

  const requiresSemanticInference =
    deterministicValidation
      ?.relationshipEvidence
      ?.requiresSemanticInference ===
    true;

  // Deterministic outcomes remain authoritative when
  // semantic inference is not explicitly required.
  if (!requiresSemanticInference) {
    return {
      ...deterministicValidation,

      validationSource:
        "deterministic",
    };
  }

  const semanticEvidence =
    await getSoWhatSemanticEvidence(
      response,
      instructionalContext
    );

  // --------------------------------------------------
  // JAVASCRIPT FINAL DECISION
  //
  // AI provides bounded instructional evidence.
  //
  // JavaScript determines:
  // - whether the So What is accepted;
  // - the student's synthesis state;
  // - the instructional diagnosis;
  // - whether progression may continue.
  // --------------------------------------------------

  const supportedSynthesis =
    semanticEvidence
      .anchoredToKeyTopic === true &&

    semanticEvidence
      .traceableToCompletedFrame === true &&

    semanticEvidence
      .supportedByCompletedFrame === true &&

    semanticEvidence
      .communicatesMeaningfulUnderstanding === true &&

    semanticEvidence
      .specificEnoughToUnderstand === true &&

    semanticEvidence
      .merelyRepeatsEarlierFrameContent === false &&

    semanticEvidence
      .confidence >= 0.9;

  if (supportedSynthesis) {
    return {
      valid:
        true,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      synthesisState:
        "supported",

      diagnosis:
        null,

      relationshipEvidence: {
        ...deterministicValidation
          .relationshipEvidence,

        anchoredToKeyTopic:
          semanticEvidence
            .anchoredToKeyTopic,

        traceableToCompletedFrame:
          semanticEvidence
            .traceableToCompletedFrame,

        supportedByCompletedFrame:
          semanticEvidence
            .supportedByCompletedFrame,

        communicatesMeaningfulUnderstanding:
          semanticEvidence
            .communicatesMeaningfulUnderstanding,

        specificEnoughToUnderstand:
          semanticEvidence
            .specificEnoughToUnderstand,

        merelyRepeatsEarlierFrameContent:
          semanticEvidence
            .merelyRepeatsEarlierFrameContent,

        semanticConfidence:
          semanticEvidence.confidence,

        semanticEvidenceSource:
          semanticEvidence.source,

        readerInferenceRequired:
          false,
      },

      validationSource:
        "deterministicWithSemanticEvidence",
    };
  }

  // --------------------------------------------------
  // EMERGING SYNTHESIS
  //
  // The response has a legitimate foundation in the
  // completed Frame but needs one additional thinking move.
  //
  // Kaw should ask the student to clarify, deepen, or make
  // the takeaway more specific rather than declaring the
  // response incorrect.
  // --------------------------------------------------

  const emergingSynthesis =
    semanticEvidence
      .anchoredToKeyTopic === true &&

    semanticEvidence
      .traceableToCompletedFrame === true &&

    semanticEvidence
      .supportedByCompletedFrame === true &&

    semanticEvidence
      .confidence >= 0.85;

  if (emergingSynthesis) {
    const repeatsEarlierContent =
      semanticEvidence
        .merelyRepeatsEarlierFrameContent ===
      true;

    const needsMeaning =
      semanticEvidence
        .communicatesMeaningfulUnderstanding ===
      false;

    const needsSpecificity =
      semanticEvidence
        .specificEnoughToUnderstand ===
      false;

    let diagnosis =
      "needsMoreSpecificSynthesis";

    if (repeatsEarlierContent) {
      diagnosis =
        "repeatsEarlierFrameContent";
    } else if (
      needsMeaning &&
      !needsSpecificity
    ) {
      diagnosis =
        "needsMoreMeaningfulSynthesis";
    }

    return {
      valid:
        false,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus:
        "incomplete",

      synthesisState:
        "emerging",

      diagnosis,

      relationshipEvidence: {
        ...deterministicValidation
          .relationshipEvidence,

        anchoredToKeyTopic:
          semanticEvidence
            .anchoredToKeyTopic,

        traceableToCompletedFrame:
          semanticEvidence
            .traceableToCompletedFrame,

        supportedByCompletedFrame:
          semanticEvidence
            .supportedByCompletedFrame,

        communicatesMeaningfulUnderstanding:
          semanticEvidence
            .communicatesMeaningfulUnderstanding,

        specificEnoughToUnderstand:
          semanticEvidence
            .specificEnoughToUnderstand,

        merelyRepeatsEarlierFrameContent:
          semanticEvidence
            .merelyRepeatsEarlierFrameContent,

        semanticConfidence:
          semanticEvidence.confidence,

        semanticEvidenceSource:
          semanticEvidence.source,

        readerInferenceRequired:
          false,
      },

      validationSource:
        "deterministicWithSemanticEvidence",
    };
  }

  // --------------------------------------------------
  // UNSUPPORTED SYNTHESIS
  //
  // The response contains substantive content, but the
  // necessary relationship to the completed Frame has not
  // been established.
  // --------------------------------------------------

  let diagnosis =
    "synthesisNotEstablished";

  if (
    semanticEvidence
      .anchoredToKeyTopic === false
  ) {
    diagnosis =
      "notAnchoredToKeyTopic";
  } else if (
    semanticEvidence
      .traceableToCompletedFrame ===
    false
  ) {
    diagnosis =
      "notTraceableToCompletedFrame";
  } else if (
    semanticEvidence
      .supportedByCompletedFrame ===
    false
  ) {
    diagnosis =
      "notSupportedByCompletedFrame";
  }

  return {
    valid:
      false,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "notSatisfied",

    relationshipStatus:
      "notEstablished",

    synthesisState:
      "unsupported",

    diagnosis,

    relationshipEvidence: {
      ...deterministicValidation
        .relationshipEvidence,

      anchoredToKeyTopic:
        semanticEvidence
          .anchoredToKeyTopic,

      traceableToCompletedFrame:
        semanticEvidence
          .traceableToCompletedFrame,

      supportedByCompletedFrame:
        semanticEvidence
          .supportedByCompletedFrame,

      communicatesMeaningfulUnderstanding:
        semanticEvidence
          .communicatesMeaningfulUnderstanding,

      specificEnoughToUnderstand:
        semanticEvidence
          .specificEnoughToUnderstand,

      merelyRepeatsEarlierFrameContent:
        semanticEvidence
          .merelyRepeatsEarlierFrameContent,

      semanticConfidence:
        semanticEvidence.confidence,

      semanticEvidenceSource:
        semanticEvidence.source,
    },

    validationSource:
      "deterministicWithSemanticEvidence",
  };
}

// ------------------------------------------------------
// SO WHAT RUNTIME CONTEXT
//
// Builds the complete instructional context required by
// governed So What validation.
//
// This helper is read-only.
// It does not validate, save, or modify student work.
// ------------------------------------------------------

function buildSoWhatValidationContext(state) {
  return {
    assignmentContext:
      state?.frameMeta?.assignmentContext || {},

    thinkingTask:
      state?.assignmentReasoning || {},

    keyTopic:
      state?.frame?.keyTopic || "",

    isAbout:
      state?.frame?.isAbout || "",

    mainIdeas:
      getIdeaList(state).filter(Boolean),

    details:
      Array.isArray(state?.frame?.details)
        ? state.frame.details.map(
            (bucket) =>
              Array.isArray(bucket)
                ? bucket.filter(Boolean)
                : []
          )
        : [],
  };
}

// ======================================================
// DETERMINISTIC SELF-TEST SUITES
// ======================================================
//
// Purpose:
//
// Provides deterministic benchmark suites that verify
// Kaw's instructional reasoning and runtime behavior.
//
// These tests do not affect production behavior.
// They run only when explicitly invoked by developers.
//
// ======================================================

// ------------------------------------------------------
// Essential Detail Test Suite
// ------------------------------------------------------
//
// Purpose:
// Quickly verify deterministic Essential Detail validation
// without building a full Frame or calling AI.
//
// These tests do not change production behavior.
// They run only when explicitly called.
// ======================================================

async function runEssentialDetailSelfTests() {
  const currentMainIdea =
    "Social media can increase anxiety and stress.";

  const tests = [
    {
      name: "ED - Stuck response",
      response: "idk",
      expected: {
        valid: false,
        componentEvidenceLevel: "none",
        componentCriteriaStatus: "notSatisfied",
        relationshipStatus: "undetermined",
        diagnosis: "noComponentEvidence",
      },
    },

    {
      name: "ED - Circular vague response",
      response: "because it does",
      expected: {
        valid: false,
        componentEvidenceLevel: "none",
        componentCriteriaStatus: "notSatisfied",
        relationshipStatus: "undetermined",
        diagnosis: "insufficientObservableEvidence",
      },
    },

    {
      name: "ED - Too little observable evidence",
      response: "They compare",
      expected: {
        valid: false,
        componentEvidenceLevel: "limited",
        componentCriteriaStatus: "notSatisfied",
        relationshipStatus: "undetermined",
        diagnosis: "insufficientObservableEvidence",
      },
    },

    {
      name: "ED - Repeats Main Idea",
      response:
        "Social media can increase anxiety and stress.",
      expected: {
        valid: false,
        componentEvidenceLevel: "limited",
        componentCriteriaStatus: "notSatisfied",
        relationshipStatus: "notEstablished",
        diagnosis: "repeatsMainIdea",
      },
    },

  {
  name: "ED - Substantive but relationship incomplete",
  response:
    "Teens compare themselves to people online.",
  expected: {
    valid: false,
    componentEvidenceLevel: "substantive",
    componentCriteriaStatus:
      "partiallySatisfied",
    relationshipStatus: "incomplete",
    diagnosis:
      "relationshipIncomplete",
  },
},

{
  name: "ED - Explicit supporting relationship",
  response:
    "Teens compare themselves to influencers, which can make them feel inadequate and increase anxiety.",
  expected: {
    valid: true,
    componentEvidenceLevel: "substantive",
    componentCriteriaStatus:
      "satisfied",
    relationshipStatus:
      "established",
    diagnosis: null,
  },
},

{
  name:
    "ED - Explicit theme relationship using which shows",

  mainIdea:
    "True friendship requires sacrifice.",

  response:
    "Some friends risk their own safety to protect each other, which shows that true friendship sometimes requires sacrifice.",

  expected: {
    valid: true,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "satisfied",

    relationshipStatus:
      "established",

    diagnosis:
      null,
  },
},

{
  name:
    "ED - Theme relationship using demonstrates",

  mainIdea:
    "True friendship requires sacrifice.",

  response:
    "The friends put themselves in danger to protect each other, demonstrating that friendship can require sacrifice.",
  expected: {
    valid: true,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "satisfied",

    relationshipStatus:
      "established",

    diagnosis:
      null,
  },
},
{
  name:
    "ED - Relationship language without Main Idea connection",

  response:
    "This shows that school rules can affect students.",

  expected: {
    valid: false,

    componentEvidenceLevel:
      "substantive",

    componentCriteriaStatus:
      "partiallySatisfied",

    relationshipStatus:
      "incomplete",

    diagnosis:
      "relationshipIncomplete",
  },
},
];
  
  let results = tests.map((test) => {
    const actual =
      validateEssentialDetailResponse(
        test.response,
        test.mainIdea || currentMainIdea
);

    const passed =
      actual.valid === test.expected.valid &&
      actual.componentEvidenceLevel ===
        test.expected.componentEvidenceLevel &&
      actual.componentCriteriaStatus ===
        test.expected.componentCriteriaStatus &&
      actual.relationshipStatus ===
        test.expected.relationshipStatus &&
      actual.diagnosis === test.expected.diagnosis;

    return {
      name: test.name,
      passed,
      response: test.response,
      expected: test.expected,
      actual,
    };
  });

    // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that the actual first Essential Detail
  // pathway blocks an invalid response before saving it.
  // --------------------------------------------------

  const runtimeState = defaultState();

  runtimeState.frameMeta.assignmentContext = {
    raw:
      "Explain how social media can affect teen mental health.",
    understanding:
      "Explain how social media can affect teen mental health.",
    studentSummary:
      "you're explaining how social media can affect teen mental health.",
    confidence: "high",
    needsClarification: false,
    inferredPurpose: "",
    childAnchor: "",
    clarificationCount: 0,
  };

  runtimeState.frameMeta.purpose = "study";

  runtimeState.frame.keyTopic =
    "Social Media and Teen Mental Health";

  runtimeState.frame.isAbout =
    "How social media can affect teen mental health.";

  runtimeState.frame.parentItems = [
    "Social media can increase anxiety and stress.",
    "Social media can affect self-esteem.",
  ];

  runtimeState.frame.details = [
    [],
    [],
  ];

  const runtimeActual =
    await updateStateFromStudent(
      runtimeState,
      "because it does"
    );

  const runtimePassed =
    Array.isArray(
      runtimeActual?.frame?.details?.[0]
    ) &&
    runtimeActual.frame.details[0].length === 0 &&
    runtimeActual?.pending?.type ===
      "stuckNudge" &&
    runtimeActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "insufficientObservableEvidence";

  results.push({
    name:
      "ED Runtime - First detail blocks circular response",

    passed:
      runtimePassed,

    response:
      "because it does",

    expected: {
      savedDetailCount: 0,
      pendingType: "stuckNudge",
      diagnosis:
        "insufficientObservableEvidence",
    },

    actual: {
      savedDetailCount:
        Array.isArray(
          runtimeActual?.frame?.details?.[0]
        )
          ? runtimeActual.frame.details[0].length
          : null,

      pendingType:
        runtimeActual?.pending?.type || null,

      diagnosis:
        runtimeActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,
    },
  });

    // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that the actual first Essential Detail
  // pathway blocks a no-evidence response before saving it.
  // --------------------------------------------------

  const stuckRuntimeState = defaultState();

  stuckRuntimeState.frameMeta.assignmentContext = {
    raw:
      "Explain how social media can affect teen mental health.",
    understanding:
      "Explain how social media can affect teen mental health.",
    studentSummary:
      "you're explaining how social media can affect teen mental health.",
    confidence: "high",
    needsClarification: false,
    inferredPurpose: "",
    childAnchor: "",
    clarificationCount: 0,
  };

  stuckRuntimeState.frameMeta.purpose = "study";

  stuckRuntimeState.frame.keyTopic =
    "Social Media and Teen Mental Health";

  stuckRuntimeState.frame.isAbout =
    "How social media can affect teen mental health.";

  stuckRuntimeState.frame.parentItems = [
    "Social media can increase anxiety and stress.",
    "Social media can affect self-esteem.",
  ];

  stuckRuntimeState.frame.details = [
    [],
    [],
  ];

  const stuckRuntimeActual =
    await updateStateFromStudent(
      stuckRuntimeState,
      "idk"
    );

  const stuckRuntimePassed =
    Array.isArray(
      stuckRuntimeActual?.frame?.details?.[0]
    ) &&
    stuckRuntimeActual.frame.details[0].length === 0 &&
    stuckRuntimeActual?.pending?.type ===
      "stuckNudge" &&
    stuckRuntimeActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "noComponentEvidence";

  results.push({
    name:
      "ED Runtime - First detail blocks no-evidence response",

    passed:
      stuckRuntimePassed,

    response:
      "idk",

    expected: {
      savedDetailCount: 0,
      pendingType: "stuckNudge",
      diagnosis:
        "noComponentEvidence",
    },

    actual: {
      savedDetailCount:
        Array.isArray(
          stuckRuntimeActual?.frame?.details?.[0]
        )
          ? stuckRuntimeActual.frame.details[0].length
          : null,

      pendingType:
        stuckRuntimeActual?.pending?.type || null,

      diagnosis:
        stuckRuntimeActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,
    },
  });

    // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that a valid first Essential Detail is saved
  // and progression continues to the required second Detail.
  // --------------------------------------------------

  const validRuntimeState = defaultState();

  validRuntimeState.frameMeta.assignmentContext = {
    raw:
      "Explain how social media can affect teen mental health.",
    understanding:
      "Explain how social media can affect teen mental health.",
    studentSummary:
      "you're explaining how social media can affect teen mental health.",
    confidence: "high",
    needsClarification: false,
    inferredPurpose: "",
    childAnchor: "",
    clarificationCount: 0,
  };

  validRuntimeState.frameMeta.purpose = "study";

  validRuntimeState.frame.keyTopic =
    "Social Media and Teen Mental Health";

  validRuntimeState.frame.isAbout =
    "How social media can affect teen mental health.";

  validRuntimeState.frame.parentItems = [
    "Social media can increase anxiety and stress.",
    "Social media can affect self-esteem.",
  ];

  validRuntimeState.frame.details = [
    [],
    [],
  ];

  const validRuntimeResponse =
    "Teens compare themselves to influencers, which can make them feel inadequate and increase anxiety.";
  
  const validRuntimeActual =
    await updateStateFromStudent(
      validRuntimeState,
      validRuntimeResponse
    );

  const validRuntimePassed =
    Array.isArray(
      validRuntimeActual?.frame?.details?.[0]
    ) &&
    validRuntimeActual.frame.details[0].length === 1 &&
    validRuntimeActual.frame.details[0][0] ===
      validRuntimeResponse &&
    validRuntimeActual?.pending?.type ===
      "collectAnotherDetail" &&
    validRuntimeActual?.pending?.index === 0;

  results.push({
    name:
      "ED Runtime - First valid detail is saved and advances",

    passed:
      validRuntimePassed,

    response:
      validRuntimeResponse,

    expected: {
      savedDetailCount: 1,
      savedDetail:
        validRuntimeResponse,
      pendingType:
        "collectAnotherDetail",
      pendingIndex: 0,
    },

    actual: {
      savedDetailCount:
        Array.isArray(
          validRuntimeActual?.frame?.details?.[0]
        )
          ? validRuntimeActual.frame.details[0].length
          : null,

      savedDetail:
        validRuntimeActual?.frame
          ?.details?.[0]?.[0] || null,

      pendingType:
        validRuntimeActual?.pending?.type || null,

      pendingIndex:
        Number.isInteger(
          validRuntimeActual?.pending?.index
        )
          ? validRuntimeActual.pending.index
          : null,
    },
  });

    // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that the second required Essential Detail
  // also blocks a circular response without overwriting
  // or losing the first valid Detail.
  // --------------------------------------------------

  const secondDetailInvalidState =
    defaultState();

  secondDetailInvalidState.frameMeta
    .assignmentContext = {
      raw:
        "Explain how social media can affect teen mental health.",
      understanding:
        "Explain how social media can affect teen mental health.",
      studentSummary:
        "you're explaining how social media can affect teen mental health.",
      confidence: "high",
      needsClarification: false,
      inferredPurpose: "",
      childAnchor: "",
      clarificationCount: 0,
    };

  secondDetailInvalidState.frameMeta.purpose =
    "study";

  secondDetailInvalidState.frame.keyTopic =
    "Social Media and Teen Mental Health";

  secondDetailInvalidState.frame.isAbout =
    "How social media can affect teen mental health.";

  secondDetailInvalidState.frame.parentItems = [
    "Social media can increase anxiety and stress.",
    "Social media can affect self-esteem.",
  ];

  const existingFirstDetail =
    "Teens compare themselves to people online.";

  secondDetailInvalidState.frame.details = [
    [existingFirstDetail],
    [],
  ];

  secondDetailInvalidState.pending = {
    type: "collectAnotherDetail",
    index: 0,
  };

  const secondDetailInvalidActual =
    await updateStateFromStudent(
      secondDetailInvalidState,
      "because it does"
    );

  const secondDetailInvalidPassed =
    Array.isArray(
      secondDetailInvalidActual?.frame
        ?.details?.[0]
    ) &&
    secondDetailInvalidActual.frame
      .details[0].length === 1 &&
    secondDetailInvalidActual.frame
      .details[0][0] ===
      existingFirstDetail &&
    secondDetailInvalidActual?.pending?.type ===
      "stuckNudge" &&
    secondDetailInvalidActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "insufficientObservableEvidence" &&
    secondDetailInvalidActual?.pending
      ?.resumePending?.type ===
      "collectAnotherDetail" &&
    secondDetailInvalidActual?.pending
      ?.resumePending?.index === 0;

  results.push({
    name:
      "ED Runtime - Second required detail blocks circular response",

    passed:
      secondDetailInvalidPassed,

    response:
      "because it does",

    expected: {
      savedDetailCount: 1,
      preservedFirstDetail:
        existingFirstDetail,
      pendingType: "stuckNudge",
      diagnosis:
        "insufficientObservableEvidence",
      resumePendingType:
        "collectAnotherDetail",
      resumePendingIndex: 0,
    },

    actual: {
      savedDetailCount:
        Array.isArray(
          secondDetailInvalidActual?.frame
            ?.details?.[0]
        )
          ? secondDetailInvalidActual.frame
              .details[0].length
          : null,

      preservedFirstDetail:
        secondDetailInvalidActual?.frame
          ?.details?.[0]?.[0] || null,

      pendingType:
        secondDetailInvalidActual?.pending
          ?.type || null,

      diagnosis:
        secondDetailInvalidActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      resumePendingType:
        secondDetailInvalidActual?.pending
          ?.resumePending?.type || null,

      resumePendingIndex:
        Number.isInteger(
          secondDetailInvalidActual?.pending
            ?.resumePending?.index
        )
          ? secondDetailInvalidActual.pending
              .resumePending.index
          : null,
    },
  });
  
  const passedCount =
    results.filter((result) => result.passed).length;

  const failedCount =
    results.length - passedCount;

  console.log("");
  console.log("======================================");
  console.log("KAW ESSENTIAL DETAIL SELF-TEST RESULTS");
  console.log("======================================");
  console.log(
    `Passed: ${passedCount}/${results.length}`
  );
  console.log(`Failed: ${failedCount}`);
  console.log("");

  results.forEach((result) => {
    if (result.passed) {
      console.log(`✅ ${result.name}`);
      return;
    }

    console.log(`❌ ${result.name}`);
    console.log("Response:", result.response);
    console.log("Expected:", result.expected);
    console.log("Actual:", result.actual);
    console.log("");
  });

  return {
    passed:
      failedCount === 0,
    passedCount,
    failedCount,
    total: results.length,
    results,
  };
}

function formatEssentialDetailSelfTestResults(
  testResults
) {
  const lines = [
    "🧪 KAW DETERMINISTIC SELF-TESTS",
    "",
    "Essential Detail Validation",
    "",
  ];

  testResults.results.forEach((result) => {
    lines.push(
      `${result.passed ? "✅" : "❌"} ${result.name}`
    );

    if (!result.passed) {
      lines.push(
        `Expected: ${JSON.stringify(
          result.expected
        )}`
      );

      lines.push(
        `Actual: ${JSON.stringify(
          result.actual
        )}`
      );
    }

    lines.push("");
  });

  lines.push("────────────────────────");
  lines.push(
    `Passed: ${testResults.passedCount}/${testResults.total}`
  );
  lines.push(
    `Failed: ${testResults.failedCount}`
  );

  if (testResults.passed) {
    lines.push("");
    lines.push(
      "🚀 All current deterministic tests passed."
    );
  }

  return lines.join("\n");
}

// ------------------------------------------------------
// Is About Test Suite
// ------------------------------------------------------
//
// Purpose:
// Verify deterministic Is About validation without
// changing live tutoring behavior.
//
// These tests evaluate whether the student's response
// observably paraphrases the Key Topic.
// ------------------------------------------------------

async function runIsAboutSelfTests() {
  const keyTopic =
    "Photosynthesis";

  const tests = [
    {
      name:
        "IA - Empty response",

      response:
        "",

      expected: {
        valid: false,

        componentEvidenceLevel:
          "none",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "emptyResponse",
      },
    },

    {
      name:
        "IA - Stuck response",

      response:
        "idk",

      expected: {
        valid: false,

        componentEvidenceLevel:
          "none",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "noComponentEvidence",
      },
    },

    {
      name:
        "IA - Repeats Key Topic",

      response:
        "Photosynthesis",

      expected: {
        valid: false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        diagnosis:
          "repeatsKeyTopic",
      },
    },

    {
      name:
        "IA - Too little observable evidence",

      response:
        "Plants make food",

      expected: {
        valid: false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "partiallySatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "insufficientObservableEvidence",
      },
    },

    {
      name:
        "IA - Observable paraphrase",

      response:
        "Photosynthesis is the process plants use to make food using sunlight.",

      expected: {
        valid: true,

        componentEvidenceLevel:
          "substantive",

        componentCriteriaStatus:
          "satisfied",

        relationshipStatus:
          "established",

        diagnosis:
          null,
      },
    },
  ];

  const results =
    tests.map((test) => {
      const actual =
        validateIsAboutResponse(
          test.response,
          keyTopic
        );

      const passed =
        actual.valid ===
          test.expected.valid &&

        actual.componentEvidenceLevel ===
          test.expected
            .componentEvidenceLevel &&

        actual.componentCriteriaStatus ===
          test.expected
            .componentCriteriaStatus &&

        actual.relationshipStatus ===
          test.expected
            .relationshipStatus &&

        actual.diagnosis ===
          test.expected.diagnosis;

      return {
        name:
          test.name,

        passed,

        response:
          test.response,

        expected:
          test.expected,

        actual,
      };
    });

  // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that repeating the Key Topic is blocked
  // before the response is saved as the Is About.
  // --------------------------------------------------

  const repeatedTopicState =
    defaultState();

  repeatedTopicState.frameMeta.assignmentContext = {
    raw:
      "Explain how photosynthesis helps plants make food.",

    understanding:
      "Explain how photosynthesis helps plants make food.",

    studentSummary:
      "you're explaining how photosynthesis helps plants make food.",

    confidence:
      "high",

    needsClarification:
      false,

    inferredPurpose:
      "",

    childAnchor:
      "",

    clarificationCount:
      0,
  };

  repeatedTopicState.frameMeta.purpose =
    "study";

  repeatedTopicState.frame.keyTopic =
    "Photosynthesis";

  const repeatedTopicActual =
    await updateStateFromStudent(
      repeatedTopicState,
      "Photosynthesis"
    );

  const repeatedTopicPassed =
    repeatedTopicActual?.frame?.isAbout ===
      "" &&

    repeatedTopicActual?.pending?.type ===
      "stuckNudge" &&

    repeatedTopicActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "repeatsKeyTopic" &&

    repeatedTopicActual?.pending
      ?.resumePending
      ?.type ===
      "reviseIsAbout";

  results.push({
    name:
      "IA Runtime - Repeated Key Topic is blocked",

    passed:
      repeatedTopicPassed,

    response:
      "Photosynthesis",

    expected: {
      savedIsAbout:
        "",

      pendingType:
        "stuckNudge",

      diagnosis:
        "repeatsKeyTopic",

      resumePendingType:
        "reviseIsAbout",
    },

    actual: {
      savedIsAbout:
        repeatedTopicActual?.frame
          ?.isAbout || "",

      pendingType:
        repeatedTopicActual?.pending
          ?.type || null,

      diagnosis:
        repeatedTopicActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      resumePendingType:
        repeatedTopicActual?.pending
          ?.resumePending
          ?.type || null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that a valid Is About paraphrase is saved
  // and advances to the confirmation checkpoint.
  // --------------------------------------------------

  const validIsAboutState =
    defaultState();

  validIsAboutState.frameMeta.assignmentContext = {
    raw:
      "Explain how photosynthesis helps plants make food.",

    understanding:
      "Explain how photosynthesis helps plants make food.",

    studentSummary:
      "you're explaining how photosynthesis helps plants make food.",

    confidence:
      "high",

    needsClarification:
      false,

    inferredPurpose:
      "",

    childAnchor:
      "",

    clarificationCount:
      0,
  };

  validIsAboutState.frameMeta.purpose =
    "study";

  validIsAboutState.frame.keyTopic =
    "Photosynthesis";

  const validIsAboutResponse =
    "Photosynthesis is the process plants use to make food using sunlight.";

  const validIsAboutActual =
    await updateStateFromStudent(
      validIsAboutState,
      validIsAboutResponse
    );

  const validIsAboutPassed =
    validIsAboutActual?.frame?.isAbout ===
      validIsAboutResponse &&

    validIsAboutActual?.pending?.type ===
      "confirmIsAbout";

  results.push({
    name:
      "IA Runtime - Valid paraphrase is saved and advances",

    passed:
      validIsAboutPassed,

    response:
      validIsAboutResponse,

    expected: {
      savedIsAbout:
        validIsAboutResponse,

      pendingType:
        "confirmIsAbout",
    },

    actual: {
      savedIsAbout:
        validIsAboutActual?.frame
          ?.isAbout || null,

      pendingType:
        validIsAboutActual?.pending
          ?.type || null,
    },
  });
  
  const passedCount =
    results.filter(
      (result) =>
        result.passed
    ).length;

  const failedCount =
    results.length -
    passedCount;

  return {
    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    results,
  };
}

function formatIsAboutSelfTestResults(
  testResults
) {
  const lines = [
    "🧪 KAW DETERMINISTIC SELF-TESTS",
    "",
    "Is About Validation",
    "",
  ];

  testResults.results.forEach(
    (result) => {
      lines.push(
        `${result.passed ? "✅" : "❌"} ${result.name}`
      );

      if (!result.passed) {
        lines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        lines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      lines.push("");
    }
  );

  lines.push(
    "────────────────────────"
  );

  lines.push(
    `Passed: ${testResults.passedCount}/${testResults.total}`
  );

  lines.push(
    `Failed: ${testResults.failedCount}`
  );

  if (testResults.passed) {
    lines.push("");
    lines.push(
      "🚀 All current Is About tests passed."
    );
  }

  return lines.join("\n");
}

// ------------------------------------------------------
// Main Idea Test Suite
//
// Purpose:
//
// Verifies governed Main Idea validation and the live
// runtime routes used for required capture, optional
// capture, and revision.
//
// These tests confirm that invalid student responses are
// blocked before state mutation and that valid responses
// save and progress through the correct pending states.
// ------------------------------------------------------

async function runMainIdeaSelfTests() {
  const keyTopic =
    "Social Media and Teen Mental Health";

  const isAbout =
    "How social media can affect teen mental health.";

  const validMainIdea =
    "Social media can increase anxiety and stress.";

  const secondMainIdea =
    "Social media can affect self-esteem.";

  const optionalMainIdea =
    "Social media can increase feelings of isolation.";

  const revisedMainIdea =
    "Social media can disrupt healthy sleep patterns.";

  const detailOnlyResponse =
    "A survey found that many teens check social media before bed.";

  const results = [];

  // --------------------------------------------------
  // TEST STATE FACTORY
  //
  // Creates a stable Build Mode state positioned at the
  // Main Ideas stage.
  // --------------------------------------------------

  function createMainIdeaTestState() {
    const state =
      defaultState();

    state.interactionMode =
      "build";

    state.frameMeta.assignmentContext = {
      raw:
        "Explain how social media can affect teen mental health.",

      understanding:
        "Explain how social media can affect teen mental health.",

      studentSummary:
        "you're explaining how social media can affect teen mental health.",

      confidence:
        "high",

      needsClarification:
        false,

      inferredPurpose:
        "",

      childAnchor:
        "",

      clarificationCount:
        0,
    };

    state.frameMeta.purpose =
      "study";

    state.frame.keyTopic =
      keyTopic;

    state.frame.isAbout =
      isAbout;

    state.frame.parentItems =
      [];

    state.frame.details =
      [];

    state.pending =
      null;

    return state;
  }

  // --------------------------------------------------
  // DETERMINISTIC VALIDATOR TESTS
  // --------------------------------------------------

  const deterministicTests = [
    {
      name:
        "MI - Empty response",

      response:
        "",

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "none",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "emptyResponse",
      },
    },

    {
      name:
        "MI - Stuck response",

      response:
        "idk",

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "none",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "noComponentEvidence",
      },
    },

    {
      name:
        "MI - Repeats Key Topic",

      response:
        keyTopic,

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        diagnosis:
          "repeatsKeyTopic",
      },
    },

    {
      name:
        "MI - Repeats Is About",

      response:
        isAbout,

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        diagnosis:
          "repeatsIsAbout",
      },
    },

    {
      name:
        "MI - One-word response",

      response:
        "Anxiety",

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "partiallySatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "insufficientObservableEvidence",
      },
    },

    {
      name:
        "MI - Substantive response requires semantic evidence",

      response:
        validMainIdea,

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "substantive",

        componentCriteriaStatus:
          "partiallySatisfied",

        relationshipStatus:
          "undetermined",

        diagnosis:
          "relationshipUndetermined",

        requiresSemanticInference:
          true,
      },
    },
  ];

  deterministicTests.forEach(
    (test) => {
      const actual =
        validateMainIdeaResponse(
          test.response,
          keyTopic,
          isAbout
        );

      const passed =
        actual.valid ===
          test.expected.valid &&

        actual.componentEvidenceLevel ===
          test.expected
            .componentEvidenceLevel &&

        actual.componentCriteriaStatus ===
          test.expected
            .componentCriteriaStatus &&

        actual.relationshipStatus ===
          test.expected
            .relationshipStatus &&

        actual.diagnosis ===
          test.expected.diagnosis &&

        (
          test.expected
            .requiresSemanticInference ===
            undefined ||

          actual?.relationshipEvidence
            ?.requiresSemanticInference ===
            test.expected
              .requiresSemanticInference
        );

      results.push({
        name:
          test.name,

        passed,

        response:
          test.response,

        expected:
          test.expected,

        actual,
      });
    }
  );

  // --------------------------------------------------
  // GOVERNED VALIDATION TEST
  //
  // Confirms bounded semantic evidence may establish a
  // valid Main Idea, while JavaScript retains the final
  // decision.
  // --------------------------------------------------

  const governedValidActual =
    await validateMainIdeaResponseGoverned(
      validMainIdea,
      keyTopic,
      isAbout
    );

  const governedValidPassed =
    governedValidActual.valid === true &&

    governedValidActual
      .componentEvidenceLevel ===
      "substantive" &&

    governedValidActual
      .componentCriteriaStatus ===
      "satisfied" &&

    governedValidActual
      .relationshipStatus ===
      "established" &&

    governedValidActual
      .diagnosis ===
      null &&

    governedValidActual
      .validationSource ===
      "deterministicWithSemanticEvidence";

  results.push({
    name:
      "MI Governed - Valid organizing idea is accepted",

    passed:
      governedValidPassed,

    response:
      validMainIdea,

    expected: {
      valid:
        true,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      diagnosis:
        null,

      validationSource:
        "deterministicWithSemanticEvidence",
    },

    actual:
      governedValidActual,
  });

  // --------------------------------------------------
  // GOVERNED DETAIL-ONLY TEST
  //
  // Confirms an isolated supporting fact does not pass as
  // a major organizing Main Idea.
  // --------------------------------------------------

  const governedDetailActual =
    await validateMainIdeaResponseGoverned(
      detailOnlyResponse,
      keyTopic,
      isAbout
    );

  const governedDetailPassed =
    governedDetailActual.valid === false &&

    governedDetailActual
      .relationshipStatus ===
      "notEstablished" &&

    (
      governedDetailActual
        .diagnosis ===
        "detailInsteadOfMainIdea" ||

      governedDetailActual
        .diagnosis ===
        "relationshipNotEstablished"
    );

  results.push({
    name:
      "MI Governed - Isolated detail is blocked",

    passed:
      governedDetailPassed,

    response:
      detailOnlyResponse,

    expected: {
      valid:
        false,

      relationshipStatus:
        "notEstablished",

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished",
      ],
    },

    actual:
      governedDetailActual,
  });

  // --------------------------------------------------
  // LIVE RUNTIME: REQUIRED CAPTURE
  //
  // Confirms the first required Main Idea routes through
  // governed validation, saves, and advances.
  // --------------------------------------------------

  const requiredState =
    createMainIdeaTestState();

  const requiredActual =
    await updateStateFromStudent(
      requiredState,
      validMainIdea
    );

  const requiredPassed =
    Array.isArray(
      requiredActual?.frame?.parentItems
    ) &&

    requiredActual
      .frame
      .parentItems
      .length ===
      1 &&

    requiredActual
      .frame
      .parentItems[0] ===
      validMainIdea &&

    Array.isArray(
      requiredActual?.frame?.details?.[0]
    ) &&

    requiredActual
      ?.pending
      ?.type ===
      "offerAnotherMainIdea";

  results.push({
    name:
      "MI Runtime - Required Main Idea saves and advances",

    passed:
      requiredPassed,

    response:
      validMainIdea,

    expected: {
      mainIdeaCount:
        1,

      firstMainIdea:
        validMainIdea,

      firstDetailBucketExists:
        true,

      pendingType:
        "offerAnotherMainIdea",
    },

    actual: {
      mainIdeaCount:
        requiredActual?.frame
          ?.parentItems?.length || 0,

      firstMainIdea:
        requiredActual?.frame
          ?.parentItems?.[0] || null,

      firstDetailBucketExists:
        Array.isArray(
          requiredActual?.frame
            ?.details?.[0]
        ),

      pendingType:
        requiredActual?.pending
          ?.type || null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME: OPTIONAL CAPTURE
  //
  // Confirms collectAnotherMainIdea uses optional capture
  // mode, appends the new Main Idea, and advances.
  // --------------------------------------------------

  const optionalState =
    createMainIdeaTestState();

  optionalState.frame.parentItems = [
    validMainIdea,
    secondMainIdea,
  ];

  optionalState.frame.details = [
    [],
    [],
  ];

  optionalState.pending = {
    type:
      "collectAnotherMainIdea",
  };

  const optionalActual =
    await updateStateFromStudent(
      optionalState,
      optionalMainIdea
    );

  const optionalPassed =
    optionalActual?.frame
      ?.parentItems?.length ===
      3 &&

    optionalActual?.frame
      ?.parentItems?.[2] ===
      optionalMainIdea &&

    Array.isArray(
      optionalActual?.frame
        ?.details?.[2]
    ) &&

    optionalActual?.pending
      ?.type ===
      "offerAnotherMainIdea";

  results.push({
    name:
      "MI Runtime - Optional Main Idea saves and advances",

    passed:
      optionalPassed,

    response:
      optionalMainIdea,

    expected: {
      mainIdeaCount:
        3,

      optionalMainIdea:
        optionalMainIdea,

      optionalDetailBucketExists:
        true,

      pendingType:
        "offerAnotherMainIdea",
    },

    actual: {
      mainIdeaCount:
        optionalActual?.frame
          ?.parentItems?.length || 0,

      optionalMainIdea:
        optionalActual?.frame
          ?.parentItems?.[2] || null,

      optionalDetailBucketExists:
        Array.isArray(
          optionalActual?.frame
            ?.details?.[2]
        ),

      pendingType:
        optionalActual?.pending
          ?.type || null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME: REVISION
  //
  // Confirms revision capture replaces the selected Main
  // Idea instead of appending a new one.
  // --------------------------------------------------

  const revisionState =
    createMainIdeaTestState();

  revisionState.frame.parentItems = [
    validMainIdea,
    secondMainIdea,
  ];

  revisionState.frame.details = [
    [],
    [],
  ];

  revisionState.pending = {
    type:
      "reviseMainIdeaAt",

    index:
      0,
  };

  const revisionActual =
    await updateStateFromStudent(
      revisionState,
      revisedMainIdea
    );

  const revisionPassed =
    revisionActual?.frame
      ?.parentItems?.length ===
      2 &&

    revisionActual?.frame
      ?.parentItems?.[0] ===
      revisedMainIdea &&

    revisionActual?.frame
      ?.parentItems?.[1] ===
      secondMainIdea &&

    revisionActual?.pending
      ?.type ===
      "confirmMainIdeas";

  results.push({
    name:
      "MI Runtime - Revision replaces selected Main Idea",

    passed:
      revisionPassed,

    response:
      revisedMainIdea,

    expected: {
      mainIdeaCount:
        2,

      revisedMainIdea:
        revisedMainIdea,

      preservedSecondMainIdea:
        secondMainIdea,

      pendingType:
        "confirmMainIdeas",
    },

    actual: {
      mainIdeaCount:
        revisionActual?.frame
          ?.parentItems?.length || 0,

      revisedMainIdea:
        revisionActual?.frame
          ?.parentItems?.[0] || null,

      preservedSecondMainIdea:
        revisionActual?.frame
          ?.parentItems?.[1] || null,

      pendingType:
        revisionActual?.pending
          ?.type || null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME: INVALID REVISION
  //
  // Confirms invalid revision content does not overwrite
  // the existing Main Idea and preserves the exact resume
  // location.
  // --------------------------------------------------

  const invalidRevisionState =
    createMainIdeaTestState();

  invalidRevisionState.frame.parentItems = [
    validMainIdea,
    secondMainIdea,
  ];

  invalidRevisionState.frame.details = [
    [],
    [],
  ];

  invalidRevisionState.pending = {
    type:
      "reviseMainIdeaAt",

    index:
      0,
  };

  const invalidRevisionActual =
    await updateStateFromStudent(
      invalidRevisionState,
      keyTopic
    );

  const invalidRevisionPassed =
    invalidRevisionActual?.frame
      ?.parentItems?.length ===
      2 &&

    invalidRevisionActual?.frame
      ?.parentItems?.[0] ===
      validMainIdea &&

    invalidRevisionActual?.pending
      ?.type ===
      "stuckNudge" &&

    invalidRevisionActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "repeatsKeyTopic" &&

    invalidRevisionActual?.pending
      ?.resumePending
      ?.type ===
      "reviseMainIdeaAt" &&

    invalidRevisionActual?.pending
      ?.resumePending
      ?.index ===
      0;

  results.push({
    name:
      "MI Runtime - Invalid revision preserves original work",

    passed:
      invalidRevisionPassed,

    response:
      keyTopic,

    expected: {
      mainIdeaCount:
        2,

      preservedMainIdea:
        validMainIdea,

      pendingType:
        "stuckNudge",

      diagnosis:
        "repeatsKeyTopic",

      resumePendingType:
        "reviseMainIdeaAt",

      resumePendingIndex:
        0,
    },

    actual: {
      mainIdeaCount:
        invalidRevisionActual?.frame
          ?.parentItems?.length || 0,

      preservedMainIdea:
        invalidRevisionActual?.frame
          ?.parentItems?.[0] || null,

      pendingType:
        invalidRevisionActual?.pending
          ?.type || null,

      diagnosis:
        invalidRevisionActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      resumePendingType:
        invalidRevisionActual?.pending
          ?.resumePending
          ?.type || null,

      resumePendingIndex:
        Number.isInteger(
          invalidRevisionActual?.pending
            ?.resumePending?.index
        )
          ? invalidRevisionActual
              .pending
              .resumePending
              .index
          : null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME: INVALID OPTIONAL CAPTURE
  //
  // Confirms invalid optional content is not appended and
  // Kaw returns to the optional Main Idea capture location.
  // --------------------------------------------------

  const invalidOptionalState =
    createMainIdeaTestState();

  invalidOptionalState.frame.parentItems = [
    validMainIdea,
    secondMainIdea,
  ];

  invalidOptionalState.frame.details = [
    [],
    [],
  ];

  invalidOptionalState.pending = {
    type:
      "collectAnotherMainIdea",
  };

  const invalidOptionalActual =
    await updateStateFromStudent(
      invalidOptionalState,
      isAbout
    );

  const invalidOptionalPassed =
    invalidOptionalActual?.frame
      ?.parentItems?.length ===
      2 &&

    invalidOptionalActual?.frame
      ?.parentItems?.[0] ===
      validMainIdea &&

    invalidOptionalActual?.frame
      ?.parentItems?.[1] ===
      secondMainIdea &&

    invalidOptionalActual?.pending
      ?.type ===
      "stuckNudge" &&

    invalidOptionalActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "repeatsIsAbout" &&

    invalidOptionalActual?.pending
      ?.resumePending
      ?.type ===
      "collectAnotherMainIdea";

  results.push({
    name:
      "MI Runtime - Invalid optional Main Idea is not saved",

    passed:
      invalidOptionalPassed,

    response:
      isAbout,

    expected: {
      mainIdeaCount:
        2,

      pendingType:
        "stuckNudge",

      diagnosis:
        "repeatsIsAbout",

      resumePendingType:
        "collectAnotherMainIdea",
    },

    actual: {
      mainIdeaCount:
        invalidOptionalActual?.frame
          ?.parentItems?.length || 0,

      pendingType:
        invalidOptionalActual?.pending
          ?.type || null,

      diagnosis:
        invalidOptionalActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      resumePendingType:
        invalidOptionalActual?.pending
          ?.resumePending
          ?.type || null,
    },
  });

  const passedCount =
    results.filter(
      (result) =>
        result.passed
    ).length;

  const failedCount =
    results.length -
    passedCount;

  return {
    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    results,
  };
}

function formatMainIdeaSelfTestResults(
  testResults
) {
  const lines = [
    "🧪 KAW GOVERNED SELF-TESTS",
    "",
    "Main Idea Validation",
    "",
  ];

  testResults.results.forEach(
    (result) => {
      lines.push(
        `${result.passed ? "✅" : "❌"} ${result.name}`
      );

      if (!result.passed) {
        lines.push(
          `Response: ${JSON.stringify(
            result.response
          )}`
        );

        lines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        lines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      lines.push("");
    }
  );

  lines.push(
    "────────────────────────"
  );

  lines.push(
    `Passed: ${testResults.passedCount}/${testResults.total}`
  );

  lines.push(
    `Failed: ${testResults.failedCount}`
  );

  if (testResults.passed) {
    lines.push("");
    lines.push(
      "🚀 All current Main Idea tests passed."
    );
  }

  return lines.join("\n");
}

// ------------------------------------------------------
// So What Test Suite
//
// Purpose:
//
// Verifies deterministic and governed So What validation.
//
// These tests confirm that:
//
// - empty and struggle responses are blocked;
// - exact repetition of earlier Frame content is blocked;
// - substantive responses are routed to governed semantic
//   evidence;
// - supported synthesis is accepted;
// - emerging synthesis is preserved as legitimate thinking
//   that needs one additional instructional move;
// - unsupported conclusions do not pass merely because
//   they sound meaningful.
//
// Runtime save-path tests will be added after governed
// validation is connected to So What capture and revision.
// ------------------------------------------------------

async function runSoWhatSelfTests() {
  const instructionalContext = {
    assignmentContext: {
      raw:
        "Explain how social media can affect teen mental health.",

      understanding:
        "Explain how social media can affect teen mental health.",

      studentSummary:
        "you're explaining how social media can affect teen mental health.",
    },

    thinkingTask: {
      task:
        "explain",

      label:
        "Explain",
    },

    keyTopic:
      "Social Media and Teen Mental Health",

    isAbout:
      "How social media can affect teen mental health.",

    mainIdeas: [
      "Social media can increase anxiety and stress.",
      "Social media can affect self-esteem.",
    ],

    details: [
      [
        "Teens may compare themselves to carefully edited images online.",
        "Constant notifications can make it difficult for teens to relax.",
      ],

      [
        "Teens may judge their lives against the lives people display online.",
        "Negative comments can make teens question their appearance or abilities.",
      ],
    ],
  };

  const supportedSoWhat =
    "Social media can harm teen mental health when online comparison and constant pressure increase anxiety and weaken self-esteem.";

  const emergingSoWhat =
    "Social media has important effects on teenagers.";

  const unsupportedSoWhat =
    "Schools should completely ban phones because students cannot learn while using them.";

  const results = [];

  // --------------------------------------------------
  // DETERMINISTIC VALIDATOR TESTS
  // --------------------------------------------------

  const deterministicTests = [
    {
      name:
        "SW - Empty response",

      response:
        "",

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "none",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "undetermined",

        synthesisState:
          "none",

        diagnosis:
          "emptyResponse",
      },
    },

    {
      name:
        "SW - Stuck response",

      response:
        "idk",

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "none",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "undetermined",

        synthesisState:
          "none",

        diagnosis:
          "noComponentEvidence",
      },
    },

    {
      name:
        "SW - Repeats Key Topic",

      response:
        instructionalContext.keyTopic,

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        synthesisState:
          "none",

        diagnosis:
          "repeatsKeyTopic",
      },
    },

    {
      name:
        "SW - Repeats Is About",

      response:
        instructionalContext.isAbout,

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        synthesisState:
          "none",

        diagnosis:
          "repeatsIsAbout",
      },
    },

    {
      name:
        "SW - Repeats Main Idea",

      response:
        instructionalContext.mainIdeas[0],

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        synthesisState:
          "none",

        diagnosis:
          "repeatsMainIdea",
      },
    },

    {
      name:
        "SW - Repeats Essential Detail",

      response:
        instructionalContext.details[0][0],

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "notSatisfied",

        relationshipStatus:
          "notEstablished",

        synthesisState:
          "none",

        diagnosis:
          "repeatsEssentialDetail",
      },
    },

    {
      name:
        "SW - Too little observable evidence",

      response:
        "It really matters",

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "limited",

        componentCriteriaStatus:
          "partiallySatisfied",

        relationshipStatus:
          "undetermined",

        synthesisState:
          "emerging",

        diagnosis:
          "insufficientObservableEvidence",
      },
    },

    {
      name:
        "SW - Substantive response requires semantic evidence",

      response:
        supportedSoWhat,

      expected: {
        valid:
          false,

        componentEvidenceLevel:
          "substantive",

        componentCriteriaStatus:
          "partiallySatisfied",

        relationshipStatus:
          "undetermined",

        synthesisState:
          "undetermined",

        diagnosis:
          "synthesisUndetermined",

        requiresSemanticInference:
          true,
      },
    },
  ];

  deterministicTests.forEach(
    (test) => {
      const actual =
        validateSoWhatResponse(
          test.response,
          instructionalContext
        );

      const passed =
        actual.valid ===
          test.expected.valid &&

        actual.componentEvidenceLevel ===
          test.expected
            .componentEvidenceLevel &&

        actual.componentCriteriaStatus ===
          test.expected
            .componentCriteriaStatus &&

        actual.relationshipStatus ===
          test.expected
            .relationshipStatus &&

        actual.synthesisState ===
          test.expected
            .synthesisState &&

        actual.diagnosis ===
          test.expected.diagnosis &&

        (
          test.expected
            .requiresSemanticInference ===
            undefined ||

          actual?.relationshipEvidence
            ?.requiresSemanticInference ===
            test.expected
              .requiresSemanticInference
        );

      results.push({
        name:
          test.name,

        passed,

        response:
          test.response,

        expected:
          test.expected,

        actual,
      });
    }
  );

  // --------------------------------------------------
  // GOVERNED SUPPORTED SYNTHESIS
  //
  // Confirms a meaningful understanding that is anchored,
  // traceable, and supported by the completed Frame passes.
  // --------------------------------------------------

  const governedSupportedActual =
    await validateSoWhatResponseGoverned(
      supportedSoWhat,
      instructionalContext
    );

  const governedSupportedPassed =
    governedSupportedActual.valid ===
      true &&

    governedSupportedActual
      .componentEvidenceLevel ===
      "substantive" &&

    governedSupportedActual
      .componentCriteriaStatus ===
      "satisfied" &&

    governedSupportedActual
      .relationshipStatus ===
      "established" &&

    governedSupportedActual
      .synthesisState ===
      "supported" &&

    governedSupportedActual
      .diagnosis ===
      null &&

    governedSupportedActual
      .validationSource ===
      "deterministicWithSemanticEvidence";

  results.push({
    name:
      "SW Governed - Supported synthesis is accepted",

    passed:
      governedSupportedPassed,

    response:
      supportedSoWhat,

    expected: {
      valid:
        true,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      synthesisState:
        "supported",

      diagnosis:
        null,

      validationSource:
        "deterministicWithSemanticEvidence",
    },

    actual:
      governedSupportedActual,
  });

  // --------------------------------------------------
  // GOVERNED EMERGING SYNTHESIS
  //
  // Confirms Kaw may recognize a legitimate foundation
  // while still asking the student to become more specific
  // or meaningful.
  //
  // This test intentionally allows the governed model to
  // select the most accurate emerging-synthesis diagnosis.
  // --------------------------------------------------

  const governedEmergingActual =
    await validateSoWhatResponseGoverned(
      emergingSoWhat,
      instructionalContext
    );

  const allowedEmergingDiagnoses =
    new Set([
      "needsMoreSpecificSynthesis",
      "needsMoreMeaningfulSynthesis",
      "repeatsEarlierFrameContent",
    ]);

  const governedEmergingPassed =
    governedEmergingActual.valid ===
      false &&

    governedEmergingActual
      .componentEvidenceLevel ===
      "substantive" &&

    governedEmergingActual
      .componentCriteriaStatus ===
      "partiallySatisfied" &&

    governedEmergingActual
      .relationshipStatus ===
      "incomplete" &&

    governedEmergingActual
      .synthesisState ===
      "emerging" &&

    allowedEmergingDiagnoses.has(
      governedEmergingActual.diagnosis
    ) &&

    governedEmergingActual
      .validationSource ===
      "deterministicWithSemanticEvidence";

  results.push({
    name:
      "SW Governed - Broad takeaway remains emerging synthesis",

    passed:
      governedEmergingPassed,

    response:
      emergingSoWhat,

    expected: {
      valid:
        false,

      componentEvidenceLevel:
        "substantive",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus:
        "incomplete",

      synthesisState:
        "emerging",

      allowedDiagnoses: [
        "needsMoreSpecificSynthesis",
        "needsMoreMeaningfulSynthesis",
        "repeatsEarlierFrameContent",
      ],

      validationSource:
        "deterministicWithSemanticEvidence",
    },

    actual:
      governedEmergingActual,
  });

  // --------------------------------------------------
  // GOVERNED UNSUPPORTED SYNTHESIS
  //
  // Confirms a meaningful-sounding conclusion does not
  // pass when its central claim is not supported by the
  // completed Frame.
  // --------------------------------------------------

  const governedUnsupportedActual =
    await validateSoWhatResponseGoverned(
      unsupportedSoWhat,
      instructionalContext
    );

  const allowedUnsupportedDiagnoses =
    new Set([
      "notAnchoredToKeyTopic",
      "notTraceableToCompletedFrame",
      "notSupportedByCompletedFrame",
      "synthesisNotEstablished",
    ]);

  const governedUnsupportedPassed =
    governedUnsupportedActual.valid ===
      false &&

    governedUnsupportedActual
      .componentCriteriaStatus ===
      "notSatisfied" &&

    governedUnsupportedActual
      .relationshipStatus ===
      "notEstablished" &&

    governedUnsupportedActual
      .synthesisState ===
      "unsupported" &&

    allowedUnsupportedDiagnoses.has(
      governedUnsupportedActual.diagnosis
    ) &&

    governedUnsupportedActual
      .validationSource ===
      "deterministicWithSemanticEvidence";

  results.push({
    name:
      "SW Governed - Unsupported conclusion is blocked",

    passed:
      governedUnsupportedPassed,

    response:
      unsupportedSoWhat,

    expected: {
      valid:
        false,

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "notEstablished",

      synthesisState:
        "unsupported",

      allowedDiagnoses: [
        "notAnchoredToKeyTopic",
        "notTraceableToCompletedFrame",
        "notSupportedByCompletedFrame",
        "synthesisNotEstablished",
      ],

      validationSource:
        "deterministicWithSemanticEvidence",
    },

    actual:
      governedUnsupportedActual,
  });

  // --------------------------------------------------
  // GOVERNED VALUE STATEMENT
  //
  // Confirms the validator does not reject a So What only
  // because it uses normative or value-oriented language.
  // The statement must still be supported by the Frame.
  // --------------------------------------------------

  const valueStatement =
    "People should use social media carefully because online comparison and constant pressure can damage teen mental health.";

  const governedValueActual =
    await validateSoWhatResponseGoverned(
      valueStatement,
      instructionalContext
    );

  const governedValuePassed =
    governedValueActual.valid ===
      true &&

    governedValueActual
      .synthesisState ===
      "supported" &&

    governedValueActual
      .relationshipStatus ===
      "established";

  results.push({
    name:
      "SW Governed - Supported value statement is accepted",

    passed:
      governedValuePassed,

    response:
      valueStatement,

    expected: {
      valid:
        true,

      synthesisState:
        "supported",

      relationshipStatus:
        "established",
    },

    actual:
      governedValueActual,
  });

  // --------------------------------------------------
  // GOVERNED SUPPORTED INFERENCE
  //
  // Confirms the So What does not need to repeat every
  // Main Idea or Detail when the larger takeaway can be
  // reasonably traced to the completed Frame.
  // --------------------------------------------------

  const supportedInference =
    "The way teens experience social media matters as much as how often they use it because comparison and social pressure can shape how they feel about themselves.";

  const governedInferenceActual =
    await validateSoWhatResponseGoverned(
      supportedInference,
      instructionalContext
    );

  const governedInferencePassed =
    governedInferenceActual.valid ===
      true &&

    governedInferenceActual
      .synthesisState ===
      "supported" &&

    governedInferenceActual
      .relationshipStatus ===
      "established";

  results.push({
    name:
      "SW Governed - Supported inference is accepted",

    passed:
      governedInferencePassed,

    response:
      supportedInference,

    expected: {
      valid:
        true,

      synthesisState:
        "supported",

      relationshipStatus:
        "established",
    },

    actual:
      governedInferenceActual,
  });

  const passedCount =
    results.filter(
      (result) =>
        result.passed
    ).length;

  const failedCount =
    results.length -
    passedCount;

  return {
    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    results,
  };
}


function formatSoWhatSelfTestResults(
  testResults
) {
  const lines = [
    "🧠 KAW GOVERNED SELF-TESTS",
    "",
    "So What Validation",
    "",
  ];

  testResults.results.forEach(
    (result) => {
      lines.push(
        `${result.passed ? "✅" : "❌"} ${result.name}`
      );

      if (!result.passed) {
        lines.push(
          `Response: ${JSON.stringify(
            result.response
          )}`
        );

        lines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        lines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      lines.push("");
    }
  );

  lines.push(
    "────────────────────────"
  );

  lines.push(
    `Passed: ${testResults.passedCount}/${testResults.total}`
  );

  lines.push(
    `Failed: ${testResults.failedCount}`
  );

  if (testResults.passed) {
    lines.push("");
    lines.push(
      "🚀 All current So What tests passed."
    );
  }

  return lines.join("\n");
}

// ------------------------------------------------------
// AI COMMUNICATION LICENSING TEST SUITE
//
// Runs live AI contextualization through the same
// deterministic contracts, licenses, and response validator
// used by Kaw's instructional runtime.
//
// These tests evaluate whether AI remains within its
// communication license. They do not require one exact
// sentence because natural wording may vary.
// ------------------------------------------------------

async function runAICommunicationSelfTests() {
  const tests = [
    {
      name:
        "AI Communication - No component evidence",

      diagnosis:
        "noComponentEvidence",

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",
    },

    {
      name:
        "AI Communication - Insufficient observable evidence",

      diagnosis:
        "insufficientObservableEvidence",

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",
    },
  ];

  const results = [];

  for (const test of tests) {
    const state = defaultState();

    state.frameMeta.assignmentContext = {
      raw:
        "Explain how social media can affect teen mental health.",

      understanding:
        "Explain how social media can affect teen mental health.",

      studentSummary:
        "you're explaining how social media can affect teen mental health.",

      confidence: "high",
      needsClarification: false,
      inferredPurpose: "",
      childAnchor: "",
      clarificationCount: 0,
    };

    state.assignmentReasoning = {
      task: "explain",
      label: "Explain",
      confidence: 1,
      evidence: ["leading:explain"],
      lastUpdated: null,
    };

    state.frameMeta.purpose = "study";

    state.frame.keyTopic =
      "Social Media and Teen Mental Health";

    state.frame.isAbout =
      "How social media can affect teen mental health.";

    state.frame.parentItems = [
      "Social media can increase anxiety and stress.",
      "Social media can affect self-esteem.",
    ];

    state.frame.details = [
      [],
      [],
    ];

    state.pending = {
      type: "collectAnotherDetail",
      index: 0,

      instructionalFinding: {
        frameComponent: "details",

        componentEvidenceLevel:
          test.componentEvidenceLevel,

        componentCriteriaStatus:
          test.componentCriteriaStatus,

        relationshipStatus:
          test.relationshipStatus,

        diagnosis:
          test.diagnosis,

        currentMainIdea:
          state.frame.parentItems[0],

        currentDetailIndex: 0,
      },
    };

    const contract =
      getInstructionalContract(
        "details",
        "genuineStruggle"
      );

    const activation =
      activateInstructionalContract(
        contract,
        state
      );

    const response =
      await getInstructionalResponse(
        activation
      );

    const communicationLicense =
      activation?.aiPayload
        ?.communicationLicense || null;

    const validation =
      validateInstructionalCommunicationResponse(
        response || "",
        communicationLicense
      );

    const passed =
      !!response &&
      validation.valid &&
      validation.questionCount === 1;

    results.push({
      name:
        test.name,

      passed,

      diagnosis:
        test.diagnosis,

      response:
        response || null,

      expected: {
        nonEmptyResponse: true,
        validLicenseResponse: true,
        questionCount: 1,
        relationshipStatus:
          test.relationshipStatus,
      },

      actual: {
        nonEmptyResponse:
          !!response,

        validLicenseResponse:
          validation.valid,

        questionCount:
          validation.questionCount,

        violations:
          validation.violations,

        relationshipStatus:
          communicationLicense
            ?.relationshipStatus || null,
      },
    });
  }

  const passedCount =
    results.filter(
      (result) => result.passed
    ).length;

  const failedCount =
    results.length - passedCount;

  return {
    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    results,
  };
}

function formatAICommunicationSelfTestResults(
  testResults
) {
  const lines = [
    "🗣️ KAW AI COMMUNICATION LICENSING",
    "",
  ];

  testResults.results.forEach(
    (result) => {
      lines.push(
        `${result.passed ? "✅" : "❌"} ${result.name}`
      );

      lines.push(
        `Kaw: ${
          result.response ||
          "(AI response rejected or unavailable)"
        }`
      );

      if (!result.passed) {
        lines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        lines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      lines.push("");
    }
  );

  lines.push(
    "────────────────────────"
  );

  lines.push(
    `Passed: ${testResults.passedCount}/${testResults.total}`
  );

  lines.push(
    `Failed: ${testResults.failedCount}`
  );

  if (testResults.passed) {
    lines.push("");
    lines.push(
      "🚀 All live AI communication tests passed."
    );
  }

  return lines.join("\n");
}

// ------------------------------------------------------
// DETERMINISTIC SELF-TEST SUITE REGISTRY
//
// Each instructional subsystem owns its own test suite.
// The registry allows /run tests to execute every suite
// without combining all tests into one giant function.
// ------------------------------------------------------

const DETERMINISTIC_SELF_TEST_SUITES = [
  {
    id: "essentialDetail",
    label: "Essential Detail Validation",
    run: runEssentialDetailSelfTests,
    format: formatEssentialDetailSelfTestResults,
  },
  {
    id: "isAbout",
    label: "Is About Validation",
    run: runIsAboutSelfTests,
    format: formatIsAboutSelfTestResults,
  },
  {
    id: "mainIdeas",
    label: "Main Idea Validation",
    run: runMainIdeaSelfTests,
    format: formatMainIdeaSelfTestResults,
  },
  {
    id: "soWhat",
    label: "So What Validation",
    run: runSoWhatSelfTests,
    format: formatSoWhatSelfTestResults,
  },
];

// ------------------------------------------------------
// STUDENT SIMULATION TEST SUITE
//
// Runs a scripted student interaction through the actual
// Kaw state-update and prompt-generation functions.
//
// This verifies progression across multiple turns without
// changing production behavior.
// ------------------------------------------------------

async function runStudentSimulationSelfTests() {
  const results = [];

  let state = defaultState();

  // --------------------------------------------------
  // STEP 1: Assignment capture
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "Explain how social media can affect teen mental health."
  );

  const assignmentPassed =
    state?.frameMeta?.assignmentContext?.raw ===
      "Explain how social media can affect teen mental health." &&
    state?.frameMeta?.assignmentContext
      ?.needsClarification === false &&
    state?.pending?.type ===
      "assignmentReasoningIntro";

  results.push({
    name:
      "Student Simulation - Assignment is understood",

    passed:
      assignmentPassed,

    expected: {
      needsClarification: false,
      pendingType:
        "assignmentReasoningIntro",
    },

    actual: {
      needsClarification:
        state?.frameMeta?.assignmentContext
          ?.needsClarification,

      pendingType:
        state?.pending?.type || null,

      thinkingTask:
        state?.assignmentReasoning?.task || null,
    },
  });

  // --------------------------------------------------
  // STEP 2: Choose Build Mode
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "1"
  );

  const workflowPassed =
    state?.interactionMode === "build" &&
    state?.pending?.type ===
      "choosePurpose";

  results.push({
    name:
      "Student Simulation - Build Mode selected",

    passed:
      workflowPassed,

    expected: {
      interactionMode: "build",
      pendingType: "choosePurpose",
    },

    actual: {
      interactionMode:
        state?.interactionMode || null,

      pendingType:
        state?.pending?.type || null,
    },
  });

  // --------------------------------------------------
  // STEP 3: Choose Study purpose
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "1"
  );

  const purposePassed =
    state?.frameMeta?.purpose === "study" &&
    state?.pending === null;

  results.push({
    name:
      "Student Simulation - Study purpose selected",

    passed:
      purposePassed,

    expected: {
      purpose: "study",
      pendingType: null,
    },

    actual: {
      purpose:
        state?.frameMeta?.purpose || null,

      pendingType:
        state?.pending?.type || null,
    },
  });

  // --------------------------------------------------
  // STEP 4: Key Topic capture
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "Social Media and Teen Mental Health"
  );

  const keyTopicPassed =
    state?.frame?.keyTopic ===
      "Social Media and Teen Mental Health";

  results.push({
    name:
      "Student Simulation - Key Topic saved",

    passed:
      keyTopicPassed,

    expected: {
      keyTopic:
        "Social Media and Teen Mental Health",
    },

    actual: {
      keyTopic:
        state?.frame?.keyTopic || null,
    },
  });

  // --------------------------------------------------
  // STEP 5: Is About capture
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "How social media can affect teen mental health"
  );

  const isAboutPassed =
    !!state?.frame?.isAbout &&
    state?.pending?.type ===
      "confirmIsAbout";

  results.push({
    name:
      "Student Simulation - Is About saved",

    passed:
      isAboutPassed,

    expected: {
      pendingType: "confirmIsAbout",
    },

    actual: {
      isAbout:
        state?.frame?.isAbout || null,

      pendingType:
        state?.pending?.type || null,
    },
  });

  // --------------------------------------------------
  // STEP 6: Confirm Is About
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "1"
  );

  const confirmIsAboutPassed =
    state?.pending === null;

  results.push({
    name:
      "Student Simulation - Is About confirmed",

    passed:
      confirmIsAboutPassed,

    expected: {
      pendingType: null,
    },

    actual: {
      pendingType:
        state?.pending?.type || null,
    },
  });

  // --------------------------------------------------
  // STEP 7: Main Idea 1
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "Social media can increase anxiety and stress."
  );

  const mainIdeaOnePassed =
    state?.frame?.parentItems?.[0] ===
      "Social media can increase anxiety and stress.";

  results.push({
    name:
      "Student Simulation - First Main Idea saved",

    passed:
      mainIdeaOnePassed,

    expected: {
      mainIdeaCount: 1,
    },

    actual: {
      mainIdeaCount:
        state?.frame?.parentItems?.length || 0,

      firstMainIdea:
        state?.frame?.parentItems?.[0] || null,
    },
  });

  // --------------------------------------------------
  // STEP 8: Main Idea 2
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "Social media can affect self-esteem."
  );

  const mainIdeaTwoPassed =
    state?.frame?.parentItems?.length === 2;

  results.push({
    name:
      "Student Simulation - Second Main Idea saved",

    passed:
      mainIdeaTwoPassed,

    expected: {
      mainIdeaCount: 2,
    },

    actual: {
      mainIdeaCount:
        state?.frame?.parentItems?.length || 0,
    },
  });

  // --------------------------------------------------
// STEP 9: Decline an optional third Main Idea
// --------------------------------------------------

state = await updateStateFromStudent(
  state,
  "2"
);

const declineAdditionalMainIdeaPassed =
  state?.pending?.type ===
    "confirmMainIdeas";

results.push({
  name:
    "Student Simulation - Optional third Main Idea declined",

  passed:
    declineAdditionalMainIdeaPassed,

  expected: {
    pendingType:
      "confirmMainIdeas",
  },

  actual: {
    pendingType:
      state?.pending?.type || null,
  },
});

// --------------------------------------------------
// STEP 10: Confirm Main Ideas
// --------------------------------------------------

state = await updateStateFromStudent(
  state,
  "1"
);

const confirmMainIdeasPassed =
  state?.pending === null;

results.push({
  name:
    "Student Simulation - Main Ideas confirmed",

  passed:
    confirmMainIdeasPassed,

  expected: {
    pendingType: null,
  },

  actual: {
    pendingType:
      state?.pending?.type || null,
  },
});

  // --------------------------------------------------
  // STEP 11: Incomplete Essential Detail is blocked
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "Teens compare themselves to influencers."
  );

  const incompleteDetailPassed =
    state?.frame?.details?.[0]?.length === 0 &&
    state?.pending?.type ===
      "stuckNudge" &&
    state?.pending?.instructionalFinding
      ?.diagnosis ===
      "relationshipIncomplete";

  results.push({
    name:
      "Student Simulation - Incomplete Detail is blocked",

    passed:
      incompleteDetailPassed,

    expected: {
      savedDetailCount: 0,
      pendingType: "stuckNudge",
      diagnosis: "relationshipIncomplete",
    },

    actual: {
      savedDetailCount:
        state?.frame?.details?.[0]?.length || 0,

      pendingType:
        state?.pending?.type || null,

      diagnosis:
        state?.pending?.instructionalFinding
          ?.diagnosis || null,
    },
  });

  // --------------------------------------------------
  // STEP 12: Revised Essential Detail is accepted
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "Teens compare themselves to influencers, which can make them feel inadequate and increase anxiety."
  );

  const revisedDetailPassed =
    state?.frame?.details?.[0]?.length === 1 &&
    state?.pending?.type ===
      "collectAnotherDetail";

  results.push({
    name:
      "Student Simulation - Revised Detail is accepted",

    passed:
      revisedDetailPassed,

    expected: {
      savedDetailCount: 1,
      pendingType:
        "collectAnotherDetail",
    },

    actual: {
      savedDetailCount:
        state?.frame?.details?.[0]?.length || 0,

      pendingType:
        state?.pending?.type || null,
    },
  });

  const passedCount =
    results.filter(
      (result) => result.passed
    ).length;

  const failedCount =
    results.length - passedCount;

  return {
    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    results,
  };
}

function formatStudentSimulationSelfTestResults(
  testResults
) {
  const lines = [
    "🎒 KAW STUDENT SIMULATION",
    "",
  ];

  testResults.results.forEach(
    (result) => {
      lines.push(
        `${result.passed ? "✅" : "❌"} ${result.name}`
      );

      if (!result.passed) {
        lines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        lines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      lines.push("");
    }
  );

  lines.push("────────────────────────");

  lines.push(
    `Passed: ${testResults.passedCount}/${testResults.total}`
  );

  lines.push(
    `Failed: ${testResults.failedCount}`
  );

  if (testResults.passed) {
    lines.push("");
    lines.push(
      "🚀 Student simulation passed."
    );
  }

  return lines.join("\n");
}

async function runAllDeterministicSelfTests() {
  const suiteResults = [];

  for (const suite of DETERMINISTIC_SELF_TEST_SUITES) {
    const result = await suite.run();

    suiteResults.push({
      id: suite.id,
      label: suite.label,
      format: suite.format,
      result,
    });
  }

  const passedCount =
    suiteResults.reduce(
      (total, suite) =>
        total + suite.result.passedCount,
      0
    );

  const failedCount =
    suiteResults.reduce(
      (total, suite) =>
        total + suite.result.failedCount,
      0
    );

  const total =
    suiteResults.reduce(
      (count, suite) =>
        count + suite.result.total,
      0
    );

  return {
    passed: failedCount === 0,
    passedCount,
    failedCount,
    total,
    suites: suiteResults,
  };
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

  if (
    assignmentText.includes("cause") ||
    assignmentText.includes("effect") ||
    assignmentText.includes("why") ||
    assignmentText.includes("how")
  ) {
    return "explain_relationship";
  }

  if (
    assignmentText.includes("theme") ||
    assignmentText.includes("message") ||
    assignmentText.includes("central idea") ||
    assignmentText.includes("big idea")
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
    assignmentText.includes("text")
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

  const contract = state.pending?.instructionalContract || null;

  if (contract?.thinkingMove === "Recall observable evidence") {
    return [
      "👀 Let's start with what you can actually observe.",
      "Think about the text, image, data, or source—not your opinion yet.",
      "What is one specific piece of evidence you notice?"
    ];
  }

  if (typeof stage === "string" && stage.startsWith("details:")) {
    const idx = Number(stage.split(":")[1]);
    const selectedMainIdea =
      Number.isInteger(idx) && ideas[idx]
        ? ideas[idx]
        : "this Main Idea";

     return [
    `🧭 You're working on this Main Idea:\n"${selectedMainIdea}"`,
    "💡 A strong Essential Detail helps your reader better understand this Main Idea.",
    "🔎 Think of one fact, example, observation, piece of evidence, or explanation that supports it.",
    "✍️ What Essential Detail could you add?"
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

  // Is About confirmation and revision
  "confirmIsAbout",
  "reviseIsAbout",

  // Main Idea optional capture, confirmation, and revision
  "offerAnotherMainIdea",
  "collectAnotherMainIdea",
  "confirmMainIdeas",
  "chooseMainIdeaToRevise",
  "reviseMainIdeaAt",

  // Essential Detail optional capture, confirmation, and revision
  "offerAnotherDetail",
  "collectAnotherDetail",
  "confirmDetails",
  "chooseDetailToRevise",
  "reviseDetailAt",

  // So What expansion and confirmation
  "offerMoreSoWhat",
  "collectMoreSoWhat",
  "confirmSoWhat",

  // Export choice
  "offerExport",
  "chooseExportType",

  // Feedback Mode
  "feedbackSelectSection",
  "feedbackCollectResponse",
  "feedbackCoach",
  "feedbackThinkingSummary",
  "feedbackRevise",
  "feedbackComplete",

  // Stuck-support engine
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

// ------------------------------------------------------
// AI INTENT FALLBACK
// Used only when deterministic rules do not recognize
// possible student struggle.
//
// This classifier never changes state or writes student work.
// It returns a controlled intent label for the existing
// deterministic engine to interpret.
// ------------------------------------------------------

async function classifyStudentIntentViaAI(state, message) {
  const text = cleanText(message);

  if (!text) {
    return {
      intent: "productive",
      confidence: 0,
    };
  }

  const stage =
    state?.pending?.stage ||
    getStage(state) ||
    "";

  const keyTopic =
    state?.frame?.keyTopic ||
    "";

  const mainIdeas =
    getIdeaList(state)
      .filter(Boolean)
      .slice(0, 5);

  const system = `You classify a student's conversational intent during a structured learning routine.

Return ONLY valid compact JSON.

Allowed intent values:
- "productive"
- "stuck"
- "frustrated"
- "uncertain"
- "off_task"
- "revision_direction"

Rules:
- Do not answer the student.
- Do not evaluate factual correctness.
- Do not rewrite student work.
- Do not infer struggle merely because an answer is short.
- Use "stuck" when the student cannot begin, has no idea, or does not know what to write.
- Use "frustrated" when the student expresses annoyance, discouragement, or emotional resistance.
- Use "uncertain" when the student is hesitant but may still be attempting an answer.
- Use "revision_direction" when the student is instructing the companion to revise rather than supplying replacement text.
- Use "off_task" only when the student clearly changes away from the learning task.
- Otherwise use "productive".

Return:
{"intent":"productive","confidence":0.0}`;

  const user = `Current Frame stage: ${stage}
Key Topic: ${keyTopic || "(not entered yet)"}
Main Ideas: ${mainIdeas.length ? mainIdeas.join(" | ") : "(none yet)"}

Student message:
"${text}"`;

 try {
   const resp = await client.chat.completions.create({
  model: DEFAULT_MODEL,
  reasoning_effort: "none",
  response_format: {
    type: "json_schema",
    json_schema: {
      name: "student_intent_classification",
      strict: true,
      schema: {
        type: "object",
        additionalProperties: false,
        properties: {
          intent: {
            type: "string",
            enum: [
              "productive",
              "stuck",
              "frustrated",
              "uncertain",
              "off_task",
              "revision_direction",
            ],
          },
          confidence: {
            type: "number",
            minimum: 0,
            maximum: 1,
          },
        },
        required: ["intent", "confidence"],
      },
    },
  },
  messages: [
    { role: "system", content: system },
    { role: "user", content: user },
  ],
});

    const parsed = JSON.parse(
      resp?.choices?.[0]?.message?.content || "{}"
    );

    const allowedIntents = new Set([
      "productive",
      "stuck",
      "frustrated",
      "uncertain",
      "off_task",
      "revision_direction",
    ]);

    const intent = allowedIntents.has(parsed.intent)
      ? parsed.intent
      : "productive";

    const confidence = Number(parsed.confidence || 0);

    return {
      intent,
      confidence:
        Number.isFinite(confidence)
          ? Math.max(0, Math.min(confidence, 1))
          : 0,
    };
  } catch {
    return {
      intent: "productive",
      confidence: 0,
    };
  }
}

async function detectsUnrecognizedStruggle(state, message) {
  // Deterministic rules always get first priority.
  if (isStuckMessage(message) || isWeakFrameResponse(message)) {
    return {
      detected: true,
      intent: "stuck",
      confidence: 1,
      source: "deterministic",
    };
  }

  const aiIntent =
    await classifyStudentIntentViaAI(state, message);

  const detected =
    aiIntent.confidence >= 0.75 &&
    (
      aiIntent.intent === "stuck" ||
      aiIntent.intent === "frustrated"
    );

  return {
    detected,
    intent: aiIntent.intent,
    confidence: aiIntent.confidence,
    source: detected ? "aiIntentFallback" : "none",
  };
}

// ------------------------------------------------------
// STUDENT-WORK MUTATION PROTECTION
// Determines whether a response is actual student-authored
// Frame content before existing work is replaced or new
// optional content is added.
//
// JavaScript remains the final authority.
// AI only classifies ambiguous intent.
// ------------------------------------------------------
async function classifyStudentWorkMutationIntent(state, message) {
  const text = cleanText(message);
  const normalized = text.toLowerCase();

  // Deterministic rules always receive first priority.
  if (!text || isWeakFrameResponse(text)) {
    return {
      accept: false,
      intent: "stuck",
      confidence: 1,
      source: "deterministic",
    };
  }

  // Choice language and conversational responses are not
  // replacement Frame content.
    if (
    isAffirmative(normalized) ||
    isNegative(normalized) ||
    normalized === "2" ||
    isMetaResponse(normalized)
  ) {
    return {
      accept: false,
      intent: "uncertain",
      confidence: 1,
      source: "deterministic",
    };
  }

  const revisionDirections = [
    "revise",
    "change",
    "edit",
    "make it stronger",
    "make that stronger",
    "help me revise",
    "help me change it",
    "change it for me",
    "fix it",
    "fix that",
    "make it better",
    "that doesn't sound right",
    "that does not sound right",
    "wait",
    "hold on",
  ];

  if (
    revisionDirections.some(
      (direction) =>
        normalized === direction ||
        normalized.startsWith(`${direction} `)
    )
  ) {
    return {
      accept: false,
      intent: "revision_direction",
      confidence: 1,
      source: "deterministic",
    };
  }

  // AI is consulted only when deterministic rules do not
  // confidently identify the student's intent.
  const aiIntent =
    await classifyStudentIntentViaAI(state, text);

  if (
    aiIntent.confidence >= 0.75 &&
    aiIntent.intent !== "productive"
  ) {
    return {
      accept: false,
      intent: aiIntent.intent,
      confidence: aiIntent.confidence,
      source: "aiIntentFallback",
    };
  }

  return {
    accept: true,
    intent: "productive",
    confidence:
      aiIntent.intent === "productive"
        ? aiIntent.confidence
        : 0,
    source:
      aiIntent.confidence >= 0.75
        ? "aiIntentFallback"
        : "deterministicDefault",
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

function normalizeInstructionalComparisonText(
  text
) {
  return cleanText(text)
    .toLowerCase()
    .replace(/[.!?]+$/g, "");
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

function cloneResumePending(pending) {
  if (!pending || typeof pending !== "object") {
    return null;
  }

  return structuredClone(pending);
}

function restoreResumePending(stuckPending) {
  return cloneResumePending(
    stuckPending?.resumePending || null
  );
}

function beginStuckSupportFromPending(
  state,
  message,
  intentResult = {}
) {
  const resumePending =
    cloneResumePending(state.pending);

  const pendingType =
    resumePending?.type || "";

  let stage = getStage(state);

  if (
    pendingType === "collectAnotherMainIdea" ||
    pendingType === "reviseMainIdeaAt"
  ) {
    stage = "mainIdeas";
  }

  if (
    pendingType === "collectAnotherDetail" ||
    pendingType === "reviseDetailAt"
  ) {
    const index = Number(resumePending?.index);

    if (Number.isInteger(index)) {
      stage = `details:${index}`;
    }
  }

  if (
    pendingType === "reviseIsAbout"
  ) {
    stage = "isAbout";
  }

  if (
    pendingType === "collectMoreSoWhat" ||
    pendingType === "confirmSoWhat"
  ) {
    stage = "soWhat";
  }

  const frameComponent =
    typeof stage === "string" && stage.startsWith("details:")
      ? "details"
      : getBaseStage(stage);

  const instructionalSituation =
    intentResult.intent === "stuck" ||
    intentResult.intent === "frustrated"
      ? "genuineStruggle"
      : null;

  const instructionalContract =
    instructionalSituation
      ? getInstructionalContract(
          frameComponent,
          instructionalSituation
        )
      : null;

  // Build a temporary activation state that includes the
  // deterministic instructional finding before the new
  // pending support state is committed.
  //
  // This ensures contract execution can read the finding
  // during activation without changing the student's saved
  // Frame or instructional location.
  const activationState = {
    ...state,

    pending: {
      ...(state?.pending || {}),

      instructionalFinding:
        intentResult?.instructionalFinding || null,
    },
  };

  const instructionalActivation =
    instructionalContract
      ? activateInstructionalContract(
          instructionalContract,
          activationState
        )
      : null;
    console.log(
  "ACTIVATION:",
  instructionalActivation
);
  
  state.pending = {
  type: "stuckNudge",
  stage,
  // Preserve the deterministic instructional finding that
  // caused this support sequence.
  //
  // This allows downstream contract selection and
  // communication to respond to what was instructionally
  // established rather than treating every invalid response
  // as generic struggle.
  instructionalFinding:
    intentResult?.instructionalFinding || null,

  instructionalContract:
    instructionalContract
      ? {
          contractId:
            instructionalContract.contractId,
          frameComponent:
            instructionalContract.frameComponent,
          instructionalSituation:
            instructionalContract.instructionalSituation,
          instructionalGoal:
            instructionalContract.instructionalGoal,
          teachingMove:
            instructionalContract.teachingMove,
          thinkingMove:
            instructionalContract.thinkingMove,
          aiContextualizes:
            instructionalContract.aiContextualizes,
        }
      : null,

  instructionalActivation:
    instructionalActivation
      ? {
          contractId:
            instructionalActivation.contractId,
          execution:
            instructionalActivation.execution,
          aiPayload:
            instructionalActivation.aiPayload,
        }
      : null,
     
    tone:
      intentResult.intent === "frustrated"
        ? "frustration"
        : detectStuckTone(message),
    resumePending,
    resumeQuestion: computeNextQuestion({
      ...state,
      pending: resumePending,
    }),
    miniQuestion: buildMiniQuestion({
      ...state,
      pending: {
        ...resumePending,
        stage,
      },
    }),
    nudgeText:
      intentResult.intent === "revision_direction"
        ? formatNudgeText([
            "💡 It sounds like you want help strengthening this part.",
            "🧭 I can help you think it through, but the wording needs to stay yours.",
            ...buildStuckNudges(state, stage),
          ])
        : formatNudgeText(
            buildStuckNudges(state, stage)
          ),
    detectedBy:
      intentResult.source || "deterministic",
    aiIntent:
      intentResult.source === "aiIntentFallback"
        ? intentResult.intent
        : undefined,
    aiConfidence:
      intentResult.source === "aiIntentFallback"
        ? intentResult.confidence
        : undefined,
  };

  return state;
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

// ---------------------
// FEEDBACK GAP BANK
// ---------------------
// Controlled internal categories Kaw can choose from.
// Student-facing questions should be generated from the
// gap + section + assignment context + purpose + student response.

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

  const hasTaskSignal =
    lower.includes("explain") ||
    lower.includes("describ") ||
    lower.includes("compar") ||
    lower.includes("contrast") ||
    lower.includes("analy") ||
    lower.includes("evaluat") ||
    lower.includes("argu") ||
    lower.includes("show") ||
    lower.includes("identif") ||
    lower.includes("interpret") ||
    lower.includes("reflect") ||
    lower.includes("summar") ||
    lower.includes("creat") ||
    lower.includes("writ") ||
    lower.includes("read") ||
    lower.startsWith("why ") ||
    lower.startsWith("how ") ||
    lower.startsWith("what causes ") ||
    lower.startsWith("what caused ") ||
    lower.startsWith("what effect ") ||
    lower.startsWith("what are the effects ");

  const hasTopicSignal = words.length >= 2;

  const needsClarification =
    !(hasTaskSignal && hasTopicSignal);

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
  const deterministicCheck =
  evaluateAssignmentUnderstanding(assignment);

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
      confidence:
        deterministicCheck.needsClarification === false ||
        parsed.confidence === "high"
          ? "high"
          : "low",

needsClarification:
  deterministicCheck.needsClarification === false
    ? false
    : parsed.needsClarification !== false,
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
// 1. assignmentContext
// 2. purpose
// 3. keyTopic
// 4. isAbout
// 5. mainIdeas
// 6. details (per Main Idea)
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
  const purpose = state?.frameMeta?.purpose || "";
  const ownerLabel = context.ownerStructuralStage;
  const stageLabel = context.structuralStage;

  return {
    ...context,
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

async function applyIsAboutCapture(s, msg) {
  const validation =
    await validateIsAboutResponseGoverned(
      msg,
      s.frame?.keyTopic || ""
    );

 if (!validation.valid) {
  const instructionalFinding = {
    frameComponent:
      "isAbout",

    componentEvidenceLevel:
      validation.componentEvidenceLevel,

    componentCriteriaStatus:
      validation.componentCriteriaStatus,

    relationshipStatus:
      validation.relationshipStatus,

    diagnosis:
      validation.diagnosis,

    keyTopic:
      s.frame?.keyTopic || "",

    attemptedIsAbout:
      cleanText(msg),

    relationshipEvidence:
      validation.relationshipEvidence || null,
  };

  const instructionalContract =
    getInstructionalContract(
      "isAbout",
      "genuineStruggle"
    );

  const activationState = {
    ...s,

    pending: {
      type: "reviseIsAbout",

      instructionalFinding,
    },
  };

  const instructionalActivation =
    instructionalContract
      ? activateInstructionalContract(
          instructionalContract,
          activationState
        )
      : null;

  s.pending = {
    type: "stuckNudge",
    stage: "isAbout",

    instructionalFinding,

    instructionalContract:
      instructionalContract
        ? {
            contractId:
              instructionalContract.contractId,

            frameComponent:
              instructionalContract.frameComponent,

            instructionalSituation:
              instructionalContract
                .instructionalSituation,

            instructionalGoal:
              instructionalContract
                .instructionalGoal,

            teachingMove:
              instructionalContract.teachingMove,

            thinkingMove:
              instructionalContract.thinkingMove,

            communicationPattern:
              instructionalContract
                .communicationPattern,

            aiContextualizes:
              instructionalContract
                .aiContextualizes,
          }
        : null,

    instructionalActivation:
      instructionalActivation
        ? {
            contractId:
              instructionalActivation.contractId,

            execution:
              instructionalActivation.execution,

            aiPayload:
              instructionalActivation.aiPayload,
          }
        : null,

    resumePending: {
      type: "reviseIsAbout",
    },

    resumeQuestion:
      getComponentPrompt(
        "isAbout",
        "revisePrompt"
      ),

    miniQuestion:
      getComponentPrompt(
        "isAbout",
        "initialPrompt",
        {
          keyTopic:
            s.frame?.keyTopic || "",
        }
      ),

    // Temporary deterministic fallback only.
    // The governed AI response is used when contract
    // activation and communication validation succeed.
    nudgeText:
      "Your Is About statement should explain the whole Key Topic in your own words.\n\nWhat is this topic mainly about?",

    tone:
      "neutral",
  };

  return s;
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
// MAIN IDEA CAPTURE
// ---------------------
//
// Governed Main Idea capture mirrors the completed
// Is About capture architecture.
//
// Deterministic validation runs first.
// Governed semantic evidence is requested only when
// deterministic validation identifies a semantic gap.
//
// JavaScript owns validation, state mutation, saving,
// revision routing, and progression.
//
// AI provides bounded semantic evidence only.
// ------------------------------------------------------

async function applyMainIdeaCapture(
  s,
  msg,
  options = {}
) {
  const text =
    cleanText(msg);

  const captureMode =
    options.captureMode || "required";

  const revisionIndex =
    Number.isInteger(options.index)
      ? options.index
      : null;

  const isRevision =
    captureMode === "revision" &&
    Number.isInteger(revisionIndex);

  const isOptional =
    captureMode === "optional";

  const isCE =
    s.frameMeta?.frameType ===
    "causeEffect";

  const validation =
    await validateMainIdeaResponseGoverned(
      text,
      s.frame?.keyTopic || "",
      s.frame?.isAbout || ""
    );

  if (!validation.valid) {
    const instructionalFinding = {
      frameComponent:
        "mainIdeas",

      componentEvidenceLevel:
        validation.componentEvidenceLevel,

      componentCriteriaStatus:
        validation.componentCriteriaStatus,

      relationshipStatus:
        validation.relationshipStatus,

      diagnosis:
        validation.diagnosis,

      keyTopic:
        s.frame?.keyTopic || "",

      isAbout:
        s.frame?.isAbout || "",

      attemptedMainIdea:
        text,

      captureMode,

      revisionIndex,

      relationshipEvidence:
        validation.relationshipEvidence || null,
    };

    let resumePending;

    if (isRevision) {
      resumePending = {
        type:
          "reviseMainIdeaAt",

        index:
          revisionIndex,
      };
    } else if (isOptional) {
      resumePending = {
        type:
          "collectAnotherMainIdea",
      };
    } else {
      resumePending = {
        type:
          "collectAnotherMainIdea",
      };
    }

    s.pending = {
      ...resumePending,

      instructionalFinding,
    };

    return beginStuckSupportFromPending(
      s,
      text,
      {
        intent:
          "stuck",

        confidence:
          1,

        source:
          `mainIdeaValidation:${validation.diagnosis}`,

        instructionalFinding,
      }
    );
  }

  // Preserve the existing Build Mode lane guardrail.
  //
  // Governed component validation determines whether the
  // response functions as a Main Idea.
  //
  // The existing lane check may still enforce specialized
  // frame-type behavior without replacing governed
  // validation.
  const laneCheck =
    analyzeBuildLane(
      s,
      "mainIdeas",
      text
    );

  if (laneCheck) {
    s.pending =
      laneCheck;

    return s;
  }

  if (isRevision) {
    if (isCE) {
      if (
        Array.isArray(s.frame.causes) &&
        s.frame.causes[revisionIndex] !==
          undefined
      ) {
        s.frame.causes[revisionIndex] =
          text;
      }
    } else {
      if (
        Array.isArray(
          s.frame.parentItems
        ) &&
        s.frame.parentItems[
          revisionIndex
        ] !== undefined
      ) {
        s.frame.parentItems[
          revisionIndex
        ] = text;
      }
    }

    s.pending = {
      type:
        "confirmMainIdeas",
    };

    return s;
  }

  if (isCE) {
    if (
      !Array.isArray(s.frame.causes)
    ) {
      s.frame.causes = [];
    }

    if (
      !Array.isArray(s.frame.details)
    ) {
      s.frame.details = [];
    }

    s.frame.causes.push(text);

    const newIndex =
      s.frame.causes.length - 1;

    if (
      !Array.isArray(
        s.frame.details[newIndex]
      )
    ) {
      s.frame.details[newIndex] = [];
    }
  } else {
    if (
      !Array.isArray(
        s.frame.parentItems
      )
    ) {
      s.frame.parentItems = [];
    }

    if (
      !Array.isArray(s.frame.details)
    ) {
      s.frame.details = [];
    }

    s.frame.parentItems.push(text);

    const newIndex =
      s.frame.parentItems.length - 1;

    if (
      !Array.isArray(
        s.frame.details[newIndex]
      )
    ) {
      s.frame.details[newIndex] = [];
    }
  }

  clearMatchingSkip(
    s,
    "mainIdeas"
  );

  const count =
    getIdeaList(s).length;

  if (count >= 5) {
    s.pending = {
      type:
        "confirmMainIdeas",
    };

    return s;
  }

  s.pending = {
    type:
      "offerAnotherMainIdea",
  };

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

  const count =
    Array.isArray(s.frame.details?.[i])
      ? s.frame.details[i].length
      : 0;

  const completionMessage =
  count === 2
    ? `✅ You now have the two required ${dLabel}s for ${miLabel} ${i + 1}:\n`
    : `✅ You currently have ${count} ${dLabel}s for ${miLabel} ${i + 1}:\n`;

return (
  completionMessage +
  `"${mi}"\n\n` +
  `Would you like to strengthen this ${miLabel} by adding another ${dLabel}?\n\n` +
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
  const dLabel =
    isCE && s.frameMeta?.purpose === "read"
      ? "Text Evidence"
      : "Essential Detail";

  const currentCount =
    Array.isArray(s.frame.details?.[i])
      ? s.frame.details[i].length
      : 0;

  const nextCount = currentCount + 1;

  // The second Detail is required to fully support the Main Idea.
  if (currentCount === 1) {
    return (
      `🎉 Great! You have your first ${dLabel}.\n\n` +
      `✍️ Let's add one more required ${dLabel} to fully support this ${miLabel}.\n\n` +
      `${miLabel} ${i + 1}: "${mi}"\n\n` +
      `What is ${dLabel} ${nextCount}?`
    );
  }

  const ideas = getIdeaList(s);
  const hasNextMainIdea = i < ideas.length - 1;

  const nextDestination = hasNextMainIdea
    ? `${miLabel} ${i + 2}`
    : "your So What statement";

  return (
    `What is ${dLabel} ${nextCount} for ${miLabel} ${i + 1}?\n` +
    `"${mi}"\n\n` +
    `💡 This additional ${dLabel} is optional and can help strengthen your Frame.\n\n` +
    `Add another ${dLabel}, or reply with 2 to review these Details and continue to ${nextDestination}.`
  );
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
      resumePending: cloneResumePending(
        s.pending.resumePending
      ),
      miniQuestion:
        s.pending.miniQuestion ||
        buildMiniQuestion(s),
    };
    return s;
  }

  if (isNegative(low) || low === "2") {
    const resumePending =
      restoreResumePending(s.pending);

    s.pending = resumePending;
    return s;
  }

  return s;
}

if (s.pending?.type === "stuckMenu") {
  const choice = normalizeStuckChoice(msg);
  if (!choice) return s;

  const resumePending =
    cloneResumePending(s.pending.resumePending);

  if (choice === "1") {
    s.pending = {
      type: "stuckReask",
      mode: "directions",
      stage: s.pending.stage || getStage(s),
      resumeQuestion: s.pending.resumeQuestion,
      resumePending,
    };
    return s;
  }

  if (choice === "2") {
    s.pending = {
      type: "stuckReask",
      mode: "reread",
      stage: s.pending.stage || getStage(s),
      resumeQuestion: s.pending.resumeQuestion,
      resumePending,
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
      resumePending,
    };
    return s;
  }

  if (choice === "4") {
    if (!Array.isArray(s.skips)) s.skips = [];

    s.skips.push({
      stage: s.pending.stage || getStage(s),
      at: Date.now(),
    });

    s.pending = {
      type: "stuckSkip",
      stage: s.pending.stage || getStage(s),
      resumeQuestion: s.pending.resumeQuestion,
      resumePending,
      miniQuestion:
        s.pending.miniQuestion ||
        buildMiniQuestion(s),
    };
    return s;
  }

  return s;
}

 if (s.pending?.type === "stuckReask") {
  const resumePending =
    restoreResumePending(s.pending);

  if (resumePending) {
    s.pending = resumePending;
    return await updateStateFromStudent(s, msg);
  }

  s.pending = null;
  // Legacy fallback: continue through the normal runtime path.
}

if (s.pending?.type === "stuckNudge") {
  const stage = s.pending.stage || getStage(s);

  if (isStuckMessage(msg)) {
    s.pending = {
      type: "stuckConfirm",
      stage,
      tone: detectStuckTone(msg),
      resumeQuestion: s.pending.resumeQuestion,
      resumePending: cloneResumePending(
        s.pending.resumePending
      ),
      miniQuestion:
        s.pending.miniQuestion ||
        buildMiniQuestion(s),
    };
    return s;
  }

  const resumePending =
    restoreResumePending(s.pending);

  if (resumePending) {
    s.pending = resumePending;
    return await updateStateFromStudent(s, msg);
  }

  s.pending = null;
  return await updateStateFromStudent(s, msg);
}

  if (s.pending?.type === "stuckSkip") {
  const low = msg.toLowerCase().trim();

  const resumePending =
    cloneResumePending(s.pending.resumePending);

  if (isAffirmative(low) || low === "1") {
    s.pending = {
      type: "stuckMini",
      stage: s.pending.stage || getStage(s),
      miniQuestion:
        s.pending.miniQuestion ||
        buildMiniQuestion(s),
      resumeQuestion: s.pending.resumeQuestion,
      resumePending,
    };
    return s;
  }

  if (isNegative(low) || low === "2") {
    s.pending = {
      type: "stuckConfirm",
      stage: s.pending.stage || getStage(s),
      resumeQuestion: s.pending.resumeQuestion,
      resumePending,
      miniQuestion:
        s.pending.miniQuestion ||
        buildMiniQuestion(s),
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
      resumePending: cloneResumePending(
        s.pending.resumePending
      ),
      miniQuestion:
        s.pending.miniQuestion ||
        buildMiniQuestion(s),
      retryFromMini: true,
    };
    return s;
  }

  const resumePending =
    restoreResumePending(s.pending);

  if (resumePending) {
    s.pending = resumePending;
    return await updateStateFromStudent(s, msg);
  }

  // Legacy Stuck flows without an exact saved pending state
  // continue through the existing stage-based handling below.
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
        await applyIsAboutCapture(s, msg);
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
    const mutationIntent =
      await classifyStudentWorkMutationIntent(s, msg);

   if (!mutationIntent.accept) {
  // Genuine struggle or frustration enters the existing
  // Stuck Support sequence without losing the exact
  // Is About revision location.
  if (
    mutationIntent.intent === "stuck" ||
    mutationIntent.intent === "frustrated"
  ) {
    return beginStuckSupportFromPending(
      s,
      msg,
      mutationIntent
    );
  }

  // Revision directions, uncertainty, and other non-content
  // responses remain protected until their coaching behavior
  // is handled explicitly.
  return s;
}

    await applyIsAboutCapture(s, msg);
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
    await applyMainIdeaCapture(
      s,
      msg,
      {
        captureMode: "revision",
        index: Number(s.pending.index),
      }
    );

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
    await applyMainIdeaCapture(
      s,
      msg,
      {
        captureMode: "optional",
      }
    );

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

  const struggleCheck =
  await detectsUnrecognizedStruggle(s, msg);

if (struggleCheck.detected) {
  const stage = `details:${idx}`;

  s.pending = {
    type: "stuckNudge",
    stage,
    tone:
      struggleCheck.intent === "frustrated"
        ? "frustration"
        : detectStuckTone(msg),
    resumeQuestion: buildMiniQuestion(s),
    miniQuestion: buildMiniQuestion(s),
    nudgeText: formatNudgeText(
      buildStuckNudges(s, stage)
    ),
    detectedBy: struggleCheck.source,
    aiIntent:
      struggleCheck.source === "aiIntentFallback"
        ? struggleCheck.intent
        : undefined,
    aiConfidence:
      struggleCheck.source === "aiIntentFallback"
        ? struggleCheck.confidence
        : undefined,
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
  const normalized = msg.toLowerCase().trim();

  if (!Array.isArray(s.frame.details[idx])) {
    s.frame.details[idx] = [];
  }

  const currentCount = s.frame.details[idx].length;

  // Declining is available only after the two required
  // Essential Details have been completed.
  if (
    currentCount >= 2 &&
    (isNegative(normalized) || normalized === "2")
  ) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  const mutationIntent =
    await classifyStudentWorkMutationIntent(s, msg);

if (!mutationIntent.accept) {
  // Genuine struggle, frustration, or a request to strengthen
  // the current work enters intent-specific coaching without
  // losing this exact optional Detail capture location.
  if (
    mutationIntent.intent === "stuck" ||
    mutationIntent.intent === "frustrated" ||
    mutationIntent.intent === "revision_direction"
  ) {
    return beginStuckSupportFromPending(
      s,
      msg,
      mutationIntent
    );
  }

  // Uncertainty and other non-content responses remain
  // protected until their coaching behavior is explicit.
  return s;
}

  if (shouldRequestEvidenceDetail(s, msg)) {
    s.pending = {
      type: "writeNeedEvidenceDetail",
      index: idx,
      mechanism: msg,
    };
    return s;
  }

  const laneCheck =
    analyzeBuildLane(s, "details", msg);

  if (laneCheck) {
    s.pending = laneCheck;
    return s;
  }

  const currentMainIdea =
  getIdeaList(s)[idx] || "";

const detailValidation =
  validateEssentialDetailResponse(
    msg,
    currentMainIdea
  );

if (!detailValidation.valid) {
  // Preserve exactly what the deterministic validator
  // established about this response.
  //
  // Do not infer intent, understanding, confusion, or effort.
  // The finding describes only the observable instructional
  // condition of the response.
  const instructionalFinding = {
    frameComponent: "details",

    componentEvidenceLevel:
      detailValidation.componentEvidenceLevel,

    componentCriteriaStatus:
      detailValidation.componentCriteriaStatus,

    relationshipStatus:
      detailValidation.relationshipStatus,

    diagnosis:
      detailValidation.diagnosis,

    currentMainIdea,

    currentDetailIndex:
      s.frame.details[idx].length,
  };

  return beginStuckSupportFromPending(
    s,
    msg,
    {
      // This remains the current contract-routing behavior
      // temporarily. The next change will select support from
      // the instructional finding rather than this generic label.
      intent: "stuck",

      confidence: 1,

      source:
        `detailValidation:${detailValidation.diagnosis}`,

      instructionalFinding,
    }
  );
}

  s.frame.details[idx] = [
    ...s.frame.details[idx],
    msg,
  ];

const arr = Array.isArray(s.frame.details[idx])
  ? s.frame.details[idx]
  : [];

// The first two Essential Details are required.
// Do not offer optional expansion until both exist.
if (arr.length < 2) {
  s.pending = {
    type: "collectAnotherDetail",
    index: idx,
  };
  return s;
}

if (arr.length >= 5) {
  s.pending = {
    type: "confirmDetails",
    index: idx,
  };
  return s;
}

// Two required Details now exist.
// Offer optional strengthening.
s.pending = {
  type: "offerAnotherDetail",
  index: idx,
};

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

  // Preserve the current Essential Detail when the student
  // explicitly declines the revision and return to confirmation.
  if (isNegative(normalized)) {
    s.pending = { type: "confirmDetails", index: idx };
    return s;
  }

  const mutationIntent =
    await classifyStudentWorkMutationIntent(s, msg);

  if (!mutationIntent.accept) {
  // Genuine struggle or frustration enters the existing
  // Stuck Support sequence without losing the selected
  // Essential Detail revision location.
  if (
    mutationIntent.intent === "stuck" ||
    mutationIntent.intent === "frustrated"
  ) {
    return beginStuckSupportFromPending(
      s,
      msg,
      mutationIntent
    );
  }

  // Revision directions, uncertainty, and other non-content
  // responses remain protected until their coaching behavior
  // is handled explicitly.
  return s;
}

  // Replace only the selected Essential Detail.
  if (
    Array.isArray(s.frame.details[idx]) &&
    s.frame.details[idx][detailIndex] !== undefined
  ) {
    s.frame.details[idx][detailIndex] = msg;
  }

  // Return to the Detail confirmation checkpoint.
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
    const normalized = msg.toLowerCase().trim();

    // A genuine decline keeps the existing So What unchanged
    // and returns the student to the confirmation checkpoint.
    if (isNegative(normalized) || normalized === "2") {
    s.pending = { type: "confirmSoWhat" };
    return s;
}

    const mutationIntent =
      await classifyStudentWorkMutationIntent(s, msg);

  if (!mutationIntent.accept) {
  // Genuine struggle or frustration enters the existing
  // Stuck Support sequence without losing the exact
  // additional So What sentence location.
  if (
    mutationIntent.intent === "stuck" ||
    mutationIntent.intent === "frustrated"
  ) {
    return beginStuckSupportFromPending(
      s,
      msg,
      mutationIntent
    );
  }

  // Revision directions, uncertainty, and other non-content
  // responses remain protected until their coaching behavior
  // is handled explicitly.
  return s;
}

    s.frame.soWhat =
      cleanText(`${s.frame.soWhat} ${msg}`);

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

  const mutationIntent =
    await classifyStudentWorkMutationIntent(s, msg);

  // The student wants to revise but has not yet supplied
  // replacement wording. Preserve the existing So What and
  // remain at the current confirmation step.
  if (
    isNegative(normalized) ||
    normalized === "2" ||
    mutationIntent.intent === "revision_direction"
  ) {
    return s;
  }

  // Accept only validated student-authored replacement content.
  if (mutationIntent.accept) {
    s.frame.soWhat = msg;
    s.pending = null;
    return s;
  }

  // Genuine struggle or frustration enters the existing
// Stuck Support sequence without losing the So What
// confirmation location.
if (
  mutationIntent.intent === "stuck" ||
  mutationIntent.intent === "frustrated"
) {
  return beginStuckSupportFromPending(
    s,
    msg,
    mutationIntent
  );
}

// Uncertainty, off-task responses, and other non-content
// replies preserve the existing So What and remain at
// the current confirmation step.
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
      await applyIsAboutCapture(s, parsed.isAbout);
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
      await applyIsAboutCapture(s, msg);
      clearMatchingSkip(s, "isAbout");
    }
    return s;
  }

   // 4) Main Ideas capture
  const ideas = getIdeaList(s);

  if (ideas.length < 2) {
    if (!isNegative(msg)) {
      await applyMainIdeaCapture(
        s,
        msg
      );

      clearMatchingSkip(
        s,
        "mainIdeas"
      );
    }

    return s;
  }
  
    // 5) Details capture
  for (let i = 0; i < ideas.length; i++) {
    const arr =
      Array.isArray(s.frame.details[i])
        ? s.frame.details[i]
        : [];

    if (arr.length < 2) {
      if (!isNegative(msg)) {
        const currentMainIdea =
          getIdeaList(s)[i] || "";

        const detailValidation =
          validateEssentialDetailResponse(
            msg,
            currentMainIdea
          );

        if (!detailValidation.valid) {
          const instructionalFinding = {
            frameComponent: "details",

            componentEvidenceLevel:
              detailValidation.componentEvidenceLevel,

            componentCriteriaStatus:
              detailValidation.componentCriteriaStatus,

            relationshipStatus:
              detailValidation.relationshipStatus,

            diagnosis:
              detailValidation.diagnosis,

            currentMainIdea,

            currentDetailIndex:
              arr.length,
          };

          s.pending = {
            type: "collectAnotherDetail",
            index: i,
          };

          return beginStuckSupportFromPending(
            s,
            msg,
            {
              intent: "stuck",
              confidence: 1,

              source:
                `detailValidation:${detailValidation.diagnosis}`,

              instructionalFinding,
            }
          );
        }

        // Only responses that pass deterministic validation
        // may reach the AI fallback struggle detector.
        const struggleCheck =
          await detectsUnrecognizedStruggle(
            s,
            msg
          );

        if (struggleCheck.detected) {
          const stage = `details:${i}`;

          const instructionalContract =
            getInstructionalContract(
              "details",
              "genuineStruggle"
            );

          const activationState = {
            ...s,
            pending: {
              type: "collectAnotherDetail",
              index: i,
            },
          };

          const instructionalActivation =
            instructionalContract
              ? activateInstructionalContract(
                  instructionalContract,
                  activationState
                )
              : null;

          s.pending = {
            type: "stuckNudge",
            stage,

            instructionalContract:
              instructionalContract
                ? {
                    contractId:
                      instructionalContract.contractId,
                    frameComponent:
                      instructionalContract.frameComponent,
                    instructionalSituation:
                      instructionalContract.instructionalSituation,
                    instructionalGoal:
                      instructionalContract.instructionalGoal,
                    teachingMove:
                      instructionalContract.teachingMove,
                    thinkingMove:
                      instructionalContract.thinkingMove,
                    aiContextualizes:
                      instructionalContract.aiContextualizes,
                  }
                : null,

            instructionalActivation:
              instructionalActivation
                ? {
                    contractId:
                      instructionalActivation.contractId,
                    execution:
                      instructionalActivation.execution,
                    aiPayload:
                      instructionalActivation.aiPayload,
                  }
                : null,

            tone:
              struggleCheck.intent === "frustrated"
                ? "frustration"
                : detectStuckTone(msg),

            resumePending: {
              type: "collectAnotherDetail",
              index: i,
            },

            resumeQuestion:
              buildMiniQuestion(
                activationState
              ),

            miniQuestion:
              buildMiniQuestion(
                activationState
              ),

            nudgeText:
              formatNudgeText(
                buildStuckNudges(
                  s,
                  stage
                )
              ),

            detectedBy:
              struggleCheck.source,

            aiIntent:
              struggleCheck.source ===
              "aiIntentFallback"
                ? struggleCheck.intent
                : undefined,

            aiConfidence:
              struggleCheck.source ===
              "aiIntentFallback"
                ? struggleCheck.confidence
                : undefined,
          };

          return s;
        }

        if (
          shouldRequestEvidenceDetail(
            s,
            msg
          )
        ) {
          s.pending = {
            type: "writeNeedEvidenceDetail",
            index: i,
            mechanism: msg,
          };

          return s;
        }

        const laneCheck =
          analyzeBuildLane(
            s,
            "details",
            msg
          );

        if (laneCheck) {
          s.pending = laneCheck;
          return s;
        }

        s.frame.details[i] = [
          ...arr,
          msg,
        ];

        clearMatchingSkip(
          s,
          `details:${i}`
        );
      }

      const updatedArr =
        Array.isArray(s.frame.details[i])
          ? s.frame.details[i]
          : [];

      // The first two Essential Details are required.
      // After Detail 1, move directly to required Detail 2.
      if (updatedArr.length < 2) {
        s.pending = {
          type: "collectAnotherDetail",
          index: i,
        };

        return s;
      }

      // Two required Details now exist.
      // Offer optional strengthening.
      s.pending = {
        type: "offerAnotherDetail",
        index: i,
      };

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

  // Preserve the last safely normalized incoming state so an
  // unexpected error never erases the student's work or location.
  let safeState = defaultState();

  try {
    const body = req.body && typeof req.body === "object" ? req.body : {};
    const message = cleanText(body.message || "");

  // ------------------------------------------------------
// TEMPORARY KAW SELF-TEST TRIGGER
//
// Send:
// {
//   "runSelfTests": true
// }
//
// This bypasses the student interaction flow and returns
// deterministic Essential Detail test results.
// Remove or disable before public production use.
// ------------------------------------------------------
if (body.runSelfTests === true) {
  const testResults =
    await runEssentialDetailSelfTests();

  return res.status(200).json({
    selfTest: true,
    suite:
      "essentialDetailDeterministicValidation",
    ...testResults,
  });
}

  // ------------------------------------------------------
// HIDDEN KAW DEVELOPER COMMAND
//
// Type "/run tests" in the Wix Kaw chat to run the
// deterministic Essential Detail validation suite.
//
// This command bypasses the normal student interaction
// flow and does not modify the student's Frame or state.
// ------------------------------------------------------
if (message.toLowerCase() === "/run tests") {
  const testResults =
    await runAllDeterministicSelfTests();

  const formattedSuites =
    testResults.suites.map((suite) =>
      suite.format(suite.result)
    );

  const reply = [
    ...formattedSuites,
    "",
    "════════════════════════",
    "ALL DETERMINISTIC SUITES",
    "════════════════════════",
    "",
    `Passed: ${testResults.passedCount}/${testResults.total}`,
    `Failed: ${testResults.failedCount}`,
    "",
    testResults.passed
      ? "🚀 All deterministic suites passed."
      : "⚠️ One or more deterministic suites failed.",
  ].join("\n");

  return res.status(200).json({
    reply,
    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),
    selfTest: testResults,
  });
}

// ------------------------------------------------------
// HIDDEN KAW AI COMMUNICATION TEST COMMAND
//
// Type "/run ai tests" in the Wix Kaw chat to run
// live AI contextualization tests against deterministic
// Communication Licenses.
//
// This command calls AI and therefore runs separately
// from the fast deterministic regression suite.
// ------------------------------------------------------

if (
  message.toLowerCase() ===
  "/run ai tests"
) {
  const testResults =
    await runAICommunicationSelfTests();

  const reply =
    formatAICommunicationSelfTestResults(
      testResults
    );

  return res.status(200).json({
    reply,

    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),

    selfTest: {
      suite:
        "aiCommunicationLicensing",

      passed:
        testResults.passed,

      passedCount:
        testResults.passedCount,

      failedCount:
        testResults.failedCount,

      total:
        testResults.total,

      results:
        testResults.results,
    },
  });
}

  // ------------------------------------------------------
// HIDDEN KAW STUDENT SIMULATION COMMAND
//
// Type "/run student tests" in the Wix Kaw chat.
// This runs scripted student interactions through the
// actual Kaw runtime without modifying the active Frame.
// ------------------------------------------------------

if (
  message.toLowerCase() ===
  "/run student tests"
) {
  const testResults =
    await runStudentSimulationSelfTests();

  const reply =
    formatStudentSimulationSelfTestResults(
      testResults
    );

  return res.status(200).json({
    reply,

    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),

    selfTest: {
      suite:
        "studentSimulation",

      ...testResults,
    },
  });
}

// ------------------------------------------------------
// HIDDEN IA-020 GOVERNED SEMANTIC TEST COMMAND
//
// Type "/ivl ia020" in the Wix Kaw chat.
//
// Runs one controlled Is About benchmark through the
// governed semantic validator without modifying the
// student's active Frame.
// ------------------------------------------------------

if (
  message.toLowerCase() ===
  "/ivl ia020"
) {
  const result =
    await runIA020GovernedTest();

  const deterministic =
    result?.deterministic || {};

  const governed =
    result?.governed || {};

  const semanticEvidence =
    governed?.relationshipEvidence || {};

  const reply = [
    "🧪 IA-020 GOVERNED SEMANTIC TEST",
    "",
    `${result.passed ? "✅ PASS" : "❌ FAIL"}`,
    "",
    `Key Topic: ${result.keyTopic || "(not found)"}`,
    "",
    `Student response: ${result.studentResponse || "(not found)"}`,
    "",
    `Expected: ${JSON.stringify(
      result.expected || {}
    )}`,
    "",
    `Deterministic: ${JSON.stringify(
      deterministic
    )}`,
    "",
    `Governed: ${JSON.stringify(
      governed
    )}`,
    "",
    `Validation source: ${
      governed.validationSource ||
      "(not returned)"
    }`,
    "",
    `Semantic equivalent: ${
      semanticEvidence.semanticEquivalent ??
      "(not returned)"
    }`,
    "",
    `Semantic confidence: ${
      semanticEvidence.semanticConfidence ??
      "(not returned)"
    }`,
  ].join("\n");

  return res.status(200).json({
    reply,

    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),

    instructionalValidationTest: {
      suite:
        "isAboutGoverned",

      ...result,
    },
  });
}

// ------------------------------------------------------
// HIDDEN BATCHED MAIN IDEA IVL COMMAND
//
// Available commands:
//
// /ivl mainideas
// /ivl mainideas 1
// /ivl mainideas 2
// etc.
//
// Each numbered command runs five Main Idea benchmarks.
// ------------------------------------------------------

const mainIdeaIVLCommand =
  message
    .toLowerCase()
    .match(
      /^\/ivl\s+mainideas(?:\s+(\d+))?$/
    );

if (mainIdeaIVLCommand) {
  const requestedBatch =
    Number(
      mainIdeaIVLCommand[1]
    );

  const batchSize =
    5;

  const totalBatches =
    Math.ceil(
      IVL.benchmarks.mainIdeas.length /
      batchSize
    );

  if (
    !Number.isInteger(
      requestedBatch
    ) ||
    requestedBatch < 1
  ) {
    const reply = [
      "🧪 MAIN IDEA IVL",
      "",
      "Choose a batch number.",
      "",
      `Available batches: 1–${totalBatches}`,
      "",
      "Example:",
      "/ivl mainideas 1",
    ].join("\n");

    return res.status(200).json({
      reply,

      state:
        body.state ||
        body.vercelState ||
        body.framing ||
        defaultState(),
    });
  }

  if (
    requestedBatch >
    totalBatches
  ) {
    const reply = [
      "🧪 MAIN IDEA IVL",
      "",
      `Batch ${requestedBatch} does not exist.`,
      "",
      `Available batches: 1–${totalBatches}`,
    ].join("\n");

    return res.status(200).json({
      reply,

      state:
        body.state ||
        body.vercelState ||
        body.framing ||
        defaultState(),
    });
  }

  IVL.results.mainIdeas =
    null;

  const mainIdeaResults =
    await runIVLMainIdeaBenchmarks(
      requestedBatch,
      batchSize
    );

  const replyLines = [
    "🧪 KAW MAIN IDEA IVL",
    "",
    `Batch: ${mainIdeaResults.batchNumber}/${mainIdeaResults.totalBatches}`,
    "",
  ];

  mainIdeaResults.results.forEach(
    (result) => {
      replyLines.push(
        `${result.passed ? "✅" : "❌"} ${result.id}: ${result.title}`
      );

      replyLines.push(
        `Student response: ${
          result.studentResponse ||
          "(empty response)"
        }`
      );

      if (!result.passed) {
        replyLines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        replyLines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      replyLines.push("");
    }
  );

  replyLines.push(
    "────────────────────────"
  );

  replyLines.push(
    `Passed: ${mainIdeaResults.passedCount}/${mainIdeaResults.total}`
  );

  replyLines.push(
    `Failed: ${mainIdeaResults.failedCount}`
  );

  const reply =
    replyLines.join("\n");

  return res.status(200).json({
    reply,

    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),

    instructionalValidationLab: {
      suite:
        "mainIdeas",

      ...mainIdeaResults,
    },
  });
}

// ------------------------------------------------------
// HIDDEN BATCHED ESSENTIAL DETAIL IVL COMMAND
//
// Available commands:
//
// /ivl essentialdetails
// /ivl essentialdetails 1
// /ivl essentialdetails 2
// etc.
//
// Each numbered command runs five Essential Detail
// benchmarks.
// ------------------------------------------------------

const essentialDetailIVLCommand =
  message
    .toLowerCase()
    .match(
      /^\/ivl\s+essentialdetails(?:\s+(\d+))?$/
    );

if (essentialDetailIVLCommand) {
  const requestedBatch =
    Number(
      essentialDetailIVLCommand[1]
    );

  const batchSize =
    5;

  const totalBatches =
    Math.ceil(
      IVL.benchmarks.essentialDetails.length /
      batchSize
    );

  if (
    !Number.isInteger(
      requestedBatch
    ) ||
    requestedBatch < 1
  ) {
    const reply = [
      "🧪 ESSENTIAL DETAIL IVL",
      "",
      "Choose a batch number.",
      "",
      `Available batches: 1–${totalBatches}`,
      "",
      "Example:",
      "/ivl essentialdetails 1",
    ].join("\n");

    return res.status(200).json({
      reply,

      state:
        body.state ||
        body.vercelState ||
        body.framing ||
        defaultState(),
    });
  }

  if (
    requestedBatch >
    totalBatches
  ) {
    const reply = [
      "🧪 ESSENTIAL DETAIL IVL",
      "",
      `Batch ${requestedBatch} does not exist.`,
      "",
      `Available batches: 1–${totalBatches}`,
    ].join("\n");

    return res.status(200).json({
      reply,

      state:
        body.state ||
        body.vercelState ||
        body.framing ||
        defaultState(),
    });
  }

  IVL.results.essentialDetails =
    null;

  const essentialDetailResults =
    await runIVLEssentialDetailBenchmarks(
      requestedBatch,
      batchSize
    );

  const replyLines = [
    "🧪 KAW ESSENTIAL DETAIL IVL",
    "",
    `Batch: ${essentialDetailResults.batchNumber}/${essentialDetailResults.totalBatches}`,
    "",
  ];

  essentialDetailResults.results.forEach(
    (result) => {
      replyLines.push(
        `${result.passed ? "✅" : "❌"} ${result.id}: ${result.title}`
      );

      replyLines.push(
        `Student response: ${
          result.studentResponse ||
          "(empty response)"
        }`
      );

      if (!result.passed) {
        replyLines.push(
          `Expected: ${JSON.stringify(
            result.expected
          )}`
        );

        replyLines.push(
          `Actual: ${JSON.stringify(
            result.actual
          )}`
        );
      }

      replyLines.push("");
    }
  );

  replyLines.push(
    "────────────────────────"
  );

  replyLines.push(
    `Passed: ${essentialDetailResults.passedCount}/${essentialDetailResults.total}`
  );

  replyLines.push(
    `Failed: ${essentialDetailResults.failedCount}`
  );

  const reply =
    replyLines.join("\n");

  return res.status(200).json({
    reply,

    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),

    instructionalValidationLab: {
      suite:
        "essentialDetails",

      ...essentialDetailResults,
    },
  });
}
    
// ------------------------------------------------------
// HIDDEN KAW INSTRUCTIONAL VALIDATION LAB COMMAND
//
// Type "/ivl" in the Wix Kaw chat to run the current
// instructional benchmark library.
//
// This command does not modify the student's active Frame.
// ------------------------------------------------------

if (
  message.toLowerCase() ===
  "/ivl"
) {
  const ivlResults =
    await runInstructionalValidationLab();

  const reply =
    formatInstructionalValidationLabResults(
      ivlResults
    );

  return res.status(200).json({
    reply,

    state:
      body.state ||
      body.vercelState ||
      body.framing ||
      defaultState(),

    instructionalValidationLab:
      ivlResults
  });
}
    
  let incoming = body.state || body.vercelState || body.framing || {};
  let state = normalizeIncomingState(incoming);

  // Keep an unchanged recovery copy from before this request
  // begins mutating instructional state.
  safeState = structuredClone(state);

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
    
      // Is About confirmation and revision
      pendingType === "confirmIsAbout" ||
      pendingType === "reviseIsAbout" ||
    
      // Main Idea optional capture, confirmation, and revision
      pendingType === "offerAnotherMainIdea" ||
      pendingType === "collectAnotherMainIdea" ||
      pendingType === "confirmMainIdeas" ||
      pendingType === "chooseMainIdeaToRevise" ||
      pendingType === "reviseMainIdeaAt" ||
    
      // Essential Detail optional capture, confirmation, and revision
      pendingType === "offerAnotherDetail" ||
      pendingType === "collectAnotherDetail" ||
      pendingType === "confirmDetails" ||
      pendingType === "chooseDetailToRevise" ||
      pendingType === "reviseDetailAt" ||
    
      // So What expansion and confirmation
      pendingType === "offerMoreSoWhat" ||
      pendingType === "collectMoreSoWhat" ||
      pendingType === "confirmSoWhat" ||
    
      // Export choice
      pendingType === "offerExport" ||
      pendingType === "chooseExportType" ||
    
      // Feedback Mode
      pendingType === "feedbackSelectSection" ||
      pendingType === "feedbackCollectResponse" ||
      pendingType === "feedbackCoach" ||
      pendingType === "feedbackThinkingSummary" ||
      pendingType === "feedbackRevise" ||
      pendingType === "feedbackComplete" ||

  // Stuck-support engine
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
     
    if (
      !inProtectedPending &&
      isStuckMessage(message) &&
      getBaseStage(getStage(state)) !== "details"
    ) {
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

      const instructionalActivation =
        state?.pending?.instructionalActivation || null;

      const instructionalResponse =
        instructionalActivation
          ? await getInstructionalResponse(
              instructionalActivation
            )
          : null;

      console.log(
        "AI RESPONSE:",
        instructionalResponse
    );
      const nextQ =
        instructionalResponse ||
        computeNextQuestion(state);
      
      let reply =
        enforceSingleQuestion(nextQ);

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
      state: safeState,
    });
  }
}

// ======================================================
// INSTRUCTIONAL VALIDATION LAB (IVL)
// ======================================================
//
// Purpose:
// Provides an isolated environment for validating
// instructional behavior without affecting student runtime.
//
// Components:
// • Is About
// • Main Ideas
// • Essential Details
// • So What
//
// This section is never called during normal tutoring.
// ======================================================

const IVL = {
  prompts: {},

  benchmarks: {
    isAbout: [],
    mainIdeas: [],
    essentialDetails: [],
    soWhat: []
  },

results: {
  isAbout: null,
  mainIdeas: null,
  essentialDetails: null,
  soWhat: null,
  overall: null
}};

// ======================================================
// IS ABOUT BENCHMARKS
// ======================================================

IVL.benchmarks.isAbout.push(
  {
    id: "IA-001",
    title: "Empty Response",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse: "",

    expected: {
      valid: false,
      diagnosis: "emptyResponse"
    }
  },

  {
    id: "IA-002",
    title: "Explicit Stuck Response",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse: "idk",

    expected: {
      valid: false,
      diagnosis: "noComponentEvidence"
    }
  },

  {
    id: "IA-003",
    title: "Meta Response Instead of Is About",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse: "yes",

    expected: {
      valid: false,
      diagnosis: "noComponentEvidence"
    }
  },

  {
    id: "IA-004",
    title: "Repeats Key Topic Exactly",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse: "Renewable Energy",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "IA-005",
    title: "Repeats Key Topic With Different Capitalization",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse: "renewable energy.",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "IA-006",
    title: "Too Little Observable Evidence",

    context: {
      keyTopic: "Photosynthesis"
    },

    studentResponse: "Plants make food",

    expected: {
      valid: false,
      diagnosis: "insufficientObservableEvidence"
    }
  },

  {
    id: "IA-007",
    title: "Short Fragment Related to Topic",

    context: {
      keyTopic: "Climate Change"
    },

    studentResponse: "Changing global temperatures",

    expected: {
      valid: false,
      diagnosis: "insufficientObservableEvidence"
    }
  },

  {
    id: "IA-008",
    title: "Substantive Response Without Observable Topic Connection",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse:
      "People use many different resources throughout their daily lives.",

    expected: {
      valid: false,
      diagnosis: "relationshipUndetermined"
    }
  },

  {
    id: "IA-009",
    title: "Unrelated Substantive Response",

    context: {
      keyTopic: "Photosynthesis"
    },

    studentResponse:
      "Ancient civilizations developed complex systems of government and trade.",

    expected: {
      valid: false,
      diagnosis: "relationshipUndetermined"
    }
  },

  {
    id: "IA-010",
    title: "Clear Whole-Topic Paraphrase",

    context: {
      keyTopic: "Renewable Energy"
    },

    studentResponse:
      "Renewable energy is power produced from resources that can naturally be replaced.",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "IA-011",
    title: "Student-Friendly Science Paraphrase",

    context: {
      keyTopic: "Photosynthesis"
    },

    studentResponse:
      "Photosynthesis is the process plants use to make food from sunlight.",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "IA-012",
    title: "Clear Social Studies Paraphrase",

    context: {
      keyTopic: "The Industrial Revolution"
    },

    studentResponse:
      "The Industrial Revolution was a period when new machines changed how goods were made.",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "IA-013",
    title: "Clear Literary Topic Paraphrase",

    context: {
      keyTopic: "Friendship in The Outsiders"
    },

    studentResponse:
      "Friendship in The Outsiders is about how loyalty helps characters survive difficult experiences.",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "IA-014",
    title: "Clear Cause and Effect Paraphrase",

    context: {
      keyTopic: "Social Media and Teen Mental Health"
    },

    studentResponse:
      "Social media and teen mental health is about how online experiences can affect teenagers emotionally.",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "IA-015",
    title: "Clear Government Topic Paraphrase",

    context: {
      keyTopic: "Checks and Balances"
    },

    studentResponse:
      "Checks and balances is a system that prevents one branch of government from gaining too much power.",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

{
    id: "IA-016",
    title: "Single Word Response",

    context: {
        keyTopic: "Renewable Energy"
    },

    studentResponse: "Energy",

    expected: {
        valid: false,
        diagnosis: "insufficientObservableEvidence"
    }
},

{
    id: "IA-017",
    title: "Very Short Fragment",

    context: {
        keyTopic: "Photosynthesis"
    },

    studentResponse: "Plants",

    expected: {
        valid: false,
        diagnosis: "insufficientObservableEvidence"
    }
},

{
    id: "IA-018",
    title: "Question Instead of Explanation",

    context: {
        keyTopic: "The Water Cycle"
    },

    studentResponse: "Isn't this about rain?",

    expected: {
        valid: false,
        diagnosis: "relationshipUndetermined"
    }
},

{
    id: "IA-019",
    title: "Opinion Instead of Topic",

    context: {
        keyTopic: "Renewable Energy"
    },

    studentResponse: "I think renewable energy is awesome.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-020",
    title: "Student Uses Everyday Language",

    context: {
        keyTopic: "Photosynthesis"
    },

    studentResponse: "It's about how plants make their own food using sunlight.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-021",
    title: "Long Academic Explanation",

    context: {
        keyTopic: "Checks and Balances"
    },

    studentResponse:
        "Checks and balances is the constitutional system that allows each branch of government to limit the power of the others.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-022",
    title: "Excellent Student Paraphrase",

    context: {
        keyTopic: "Artificial Intelligence"

    },

    studentResponse:
        "Artificial intelligence is technology that allows computers to perform tasks that usually require human thinking.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-023",
    title: "Strong AP CSP Context",

    context: {
        keyTopic: "Algorithms"
    },

    studentResponse:
        "Algorithms are step-by-step procedures used to solve problems or complete tasks.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-024",
    title: "Strong ELA Context",

    context: {
        keyTopic: "Theme"

    },

    studentResponse:
        "A theme is the central message or lesson an author wants readers to understand.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-025",
    title: "Strong Math Context",

    context: {
        keyTopic: "Linear Functions"

    },

    studentResponse:
        "Linear functions describe relationships that change at a constant rate.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-026",
    title: "Strong Science Context",

    context: {
        keyTopic: "Natural Selection"

    },

    studentResponse:
        "Natural selection explains how organisms with helpful traits are more likely to survive and reproduce.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-027",
    title: "Strong History Context",

    context: {
        keyTopic: "The American Revolution"

    },

    studentResponse:
        "The American Revolution was the conflict that led the American colonies to gain independence from Britain.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-028",
    title: "Reader Must Infer Relationship",

    context: {
        keyTopic: "Internet Safety"

    },

    studentResponse:
        "People should be careful online.",

    expected: {
        valid: false,
        diagnosis: "relationshipUndetermined"
    }
},

{
    id: "IA-029",
    title: "Very Broad Generalization",

    context: {
        keyTopic: "Climate Change"

    },

    studentResponse:
        "The world is changing.",

    expected: {
        valid: false,
        diagnosis: "relationshipUndetermined"
    }
},

{
    id: "IA-030",
    title: "Gold Standard Response",

    context: {
        keyTopic: "Machine Learning"

    },

    studentResponse:
        "Machine learning is a branch of artificial intelligence in which computers improve their performance by learning from data rather than being explicitly programmed for every situation.",

    expected: {
        valid: true,
        diagnosis: null
    }
},

  {
    id: "IA-031",
    title: "Strategic Learners - Prepare Monitor Reflect",

    context: {
        keyTopic: "Strategic learners"
    },

    studentResponse:
        "prepare for, monitor, and reflect on learning",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-032",
    title: "Strategic Learners - Thinking Strategies",

    context: {
        keyTopic: "Strategic learners"
    },

    studentResponse:
        "how students use thinking strategies before, during, and after learning",

    expected: {
        valid: true,
        diagnosis: null
    }
},

{
    id: "IA-033",
    title: "Strategic Learners - Think Before During After",

    context: {
        keyTopic: "Strategic learners"
    },

    studentResponse:
        "how students think before, during, and after learning",

    expected: {
        valid: true,
        diagnosis: null
    }
}
);

// ======================================================
// MAIN IDEA BENCHMARKS
// ======================================================
//
// Instructional Contract v1.0:
//
// A valid Main Idea:
//
// 1. Organizes one major part of the Key Topic.
// 2. Supports the accepted Is About statement.
// 3. Can function as a heading for multiple Essential Details.
// 4. Fits the organizational pattern of the Frame.
// 5. Can reasonably be expanded with two or more
//    Essential Details.
//
// Invalid responses include:
//
// - repeated Key Topics;
// - repeated Is About statements;
// - isolated Essential Details;
// - unrelated responses;
// - responses too vague to organize information;
// - responses that cannot reasonably support multiple Details.
//
// These benchmarks are derived from instructional examples
// in the KU Framing Routine manual.
//
// ======================================================

IVL.benchmarks.mainIdeas.push(

  // ====================================================
  // COLUMBUS — REASONS / CATEGORIES
  // ====================================================

  {
    id: "MI-001",
    title:
      "Columbus - Financial Reasons",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Financial Reasons",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-002",
    title:
      "Columbus - Religious Reasons",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Religious Reasons",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-003",
    title:
      "Columbus - Egotistical Reasons",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Egotistical Reasons",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-004",
    title:
      "Columbus - Repeats Key Topic",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "What motivated Columbus?",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "MI-005",
    title:
      "Columbus - Repeats Is About",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Why did Columbus cross the Atlantic Ocean?",

    expected: {
      valid: false,
      diagnosis: "repeatsIsAbout"
    }
  },

  {
    id: "MI-006",
    title:
      "Columbus - Financial Detail Instead of Main Idea",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Get rich by selling spices at home",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-007",
    title:
      "Columbus - Religious Detail Instead of Main Idea",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Spread Christianity to other parts of the world",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-008",
    title:
      "Columbus - Egotistical Detail Instead of Main Idea",

    context: {
      keyTopic:
        "What motivated Columbus?",

      isAbout:
        "Why did Columbus cross the Atlantic Ocean?"
    },

    studentResponse:
      "Gain respect",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  // ====================================================
  // FRENCH REVOLUTION — ANTICIPATION CATEGORIES
  // ====================================================

  {
    id: "MI-009",
    title:
      "French Revolution - Know Already",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "Know already",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-010",
    title:
      "French Revolution - Expect to Learn",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "Expect to learn",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-011",
    title:
      "French Revolution - Want to Know",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "Want to know",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-012",
    title:
      "French Revolution - Repeats Key Topic",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "French Revolution",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "MI-013",
    title:
      "French Revolution - Repeats Is About",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "A war that resulted from a bad social situation",

    expected: {
      valid: false,
      diagnosis: "repeatsIsAbout"
    }
  },

  {
    id: "MI-014",
    title:
      "French Revolution - Detail Instead of Main Idea",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "Many poor people were imprisoned in the Bastille for no reason",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-015",
    title:
      "French Revolution - Another Detail Instead of Main Idea",

    context: {
      keyTopic:
        "French Revolution",

      isAbout:
        "A war that resulted from a bad social situation"
    },

    studentResponse:
      "The French Revolution used the guillotine frequently",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  // ====================================================
  // CUBAN MISSILE CRISIS — CHRONOLOGICAL EVENTS
  // ====================================================

  {
    id: "MI-016",
    title:
      "Cuban Missile Crisis - Castro Comes to Power",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Castro comes to power in Cuba",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-017",
    title:
      "Cuban Missile Crisis - Bay of Pigs Invasion",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Bay of Pigs Invasion",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-018",
    title:
      "Cuban Missile Crisis - Nuclear Face-Off",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Nuclear face off with the USSR",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-019",
    title:
      "Cuban Missile Crisis - Repeats Key Topic",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Cuban Missile Crisis",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "MI-020",
    title:
      "Cuban Missile Crisis - Castro Detail Instead of Main Idea",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Castro nationalized United States-owned businesses",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-021",
    title:
      "Cuban Missile Crisis - Bay of Pigs Detail Instead of Main Idea",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Twenty thousand Cuban troops defeated fourteen hundred invaders",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-022",
    title:
      "Cuban Missile Crisis - Face-Off Detail Instead of Main Idea",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "The USSR agreed to remove its missiles",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  // ====================================================
  // TITANIC — CONTRIBUTING FACTORS
  // ====================================================

  {
    id: "MI-023",
    title:
      "Titanic - Lack of Planning",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "Lack of planning",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-024",
    title:
      "Titanic - Class System",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "Class system",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-025",
    title:
      "Titanic - Competition",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "Competition",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-026",
    title:
      "Titanic - Repeats Key Topic",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "Sinking of the Titanic",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "MI-027",
    title:
      "Titanic - Planning Detail Instead of Main Idea",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "There were not enough lifeboats",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-028",
    title:
      "Titanic - Class Detail Instead of Main Idea",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "Wealthy passengers stayed on the upper deck",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  {
    id: "MI-029",
    title:
      "Titanic - Competition Detail Instead of Main Idea",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "The ship traveled at a fast speed to break the crossing record",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "detailInsteadOfMainIdea",
        "relationshipNotEstablished"
      ]
    }
  },

  // ====================================================
  // FEMINIST MOVEMENT — PERSPECTIVES
  // ====================================================

  {
    id: "MI-030",
    title:
      "Feminist Movement - Views of Opponents",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Different perspectives on efforts to expand women's rights"
    },

    studentResponse:
      "Views of opponents",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-031",
    title:
      "Feminist Movement - Views of Supporters",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Different perspectives on efforts to expand women's rights"
    },

    studentResponse:
      "Views of supporters",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-032",
    title:
      "Feminist Movement - Our Perspective",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Different perspectives on efforts to expand women's rights"
    },

    studentResponse:
      "Our perspective",

    expected: {
      valid: true,
      diagnosis: null
    }
  },

  {
    id: "MI-033",
    title:
      "Feminist Movement - Repeats Key Topic",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Different perspectives on efforts to expand women's rights"
    },

    studentResponse:
      "Feminist Movement",

    expected: {
      valid: false,
      diagnosis: "repeatsKeyTopic"
    }
  },

  {
    id: "MI-034",
    title:
      "Feminist Movement - Repeats Is About",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Different perspectives on efforts to expand women's rights"
    },

    studentResponse:
      "Different perspectives on efforts to expand women's rights",

    expected: {
      valid: false,
      diagnosis: "repeatsIsAbout"
    }
  },

  // ====================================================
  // GENERAL INSTRUCTIONAL BOUNDARIES
  // ====================================================

  {
    id: "MI-035",
    title:
      "General - Empty Response",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "",

    expected: {
      valid: false,
      diagnosis: "emptyResponse"
    }
  },

  {
    id: "MI-036",
    title:
      "General - Stuck Response",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "idk",

    expected: {
      valid: false,
      diagnosis: "noComponentEvidence"
    }
  },

  {
    id: "MI-037",
    title:
      "General - Meta Response",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "yes",

    expected: {
      valid: false,
      diagnosis: "noComponentEvidence"
    }
  },

  {
    id: "MI-038",
    title:
      "General - Unrelated Organizing Category",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society and safety"
    },

    studentResponse:
      "Types of renewable energy",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "relationshipNotEstablished",
        "detailInsteadOfMainIdea"
      ]
    }
  },

  {
    id: "MI-039",
    title:
      "General - Vague Non-Organizing Phrase",

    context: {
      keyTopic:
        "Cuban Missile Crisis",

      isAbout:
        "A political crisis that nearly led to nuclear war with the USSR"
    },

    studentResponse:
      "Important things",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "relationshipNotEstablished",
        "detailInsteadOfMainIdea"
      ]
    }
  },

  {
    id: "MI-040",
    title:
      "General - Related but Non-Organizing Comment",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Different perspectives on efforts to expand women's rights"
    },

    studentResponse:
      "The movement was important",

    expected: {
      valid: false,

      allowedDiagnoses: [
        "relationshipNotEstablished",
        "detailInsteadOfMainIdea"
      ]
    }
  }
);

// ======================================================
// ======================================================
// ESSENTIAL DETAIL BENCHMARKS
//
// Benchmark structure:
//
// Section A:
// Known architecture and regression cases.
//
// Section B:
// Canonical examples taken from completed Frames in
// the KU Framing Routine manual.
//
// Section C:
// Manual-grounded contrast cases testing whether a
// response belongs under the selected Main Idea.
//
// These benchmarks treat the instructional manual as
// executable instructional documentation.
// ======================================================

IVL.benchmarks.essentialDetails.push(
  // ====================================================
  // SECTION A
  // ARCHITECTURE AND REGRESSION CASES
  // ====================================================

  {
    id:
      "ED-001",

    title:
      "Regression — Repeats Main Idea",

    source:
      "Kaw regression",

    context: {
      keyTopic:
        "Social Media",

      isAbout:
        "How social media affects mental health",

      mainIdea:
        "Social media can increase anxiety and stress.",
    },

    studentResponse:
      "Social media can increase anxiety and stress.",

    expected: {
      valid:
        false,

      diagnosis:
        "repeatsMainIdea",
    },
  },

  {
    id:
      "ED-002",

    title:
      "Regression — Inferable Supporting Relationship",

    source:
      "Kaw regression",

    context: {
      keyTopic:
        "Renewable Energy",

      isAbout:
        "How renewable energy helps the environment",

      mainIdea:
        "Renewable energy reduces pollution.",
    },

    studentResponse:
      "Solar panels generate electricity without burning fossil fuels.",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  // ====================================================
  // SECTION B
  // KU MANUAL — PROGRESSIVE ERA FRAME
  // ====================================================

  {
    id:
      "ED-003",

    title:
      "Manual Figure 1 — Social Problem",

    source:
      "KU Framing Routine Manual — Figure 1",

    context: {
      keyTopic:
        "Progressive Era",

      isAbout:
        "A period of social change in the U.S.",

      mainIdea:
        "Social Problems",
    },

    studentResponse:
      "Unsafe food",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-004",

    title:
      "Manual Figure 1 — Tool for Social Change",

    source:
      "KU Framing Routine Manual — Figure 1",

    context: {
      keyTopic:
        "Progressive Era",

      isAbout:
        "A period of social change in the U.S.",

      mainIdea:
        "Tools for Social Change",
    },

    studentResponse:
      "Muckrakers wrote about problems",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-005",

    title:
      "Manual Figure 1 — Social Change",

    source:
      "KU Framing Routine Manual — Figure 1",

    context: {
      keyTopic:
        "Progressive Era",

      isAbout:
        "A period of social change in the U.S.",

      mainIdea:
        "Social Changes",
    },

    studentResponse:
      "Meat Inspection Act",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  // ====================================================
  // KU MANUAL — STRATEGIC LEARNERS FRAME
  // ====================================================

  {
    id:
      "ED-006",

    title:
      "Manual Figure 2 — Think Before",

    source:
      "KU Framing Routine Manual — Figure 2",

    context: {
      keyTopic:
        "Strategic Learners",

      isAbout:
        "Students who use good study plans",

      mainIdea:
        "They think BEFORE",
    },

    studentResponse:
      "By organizing books and materials",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-007",

    title:
      "Manual Figure 2 — Think During",

    source:
      "KU Framing Routine Manual — Figure 2",

    context: {
      keyTopic:
        "Strategic Learners",

      isAbout:
        "Students who use good study plans",

      mainIdea:
        "They think DURING",
    },

    studentResponse:
      "By asking and answering questions",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-008",

    title:
      "Manual Figure 2 — Think After",

    source:
      "KU Framing Routine Manual — Figure 2",

    context: {
      keyTopic:
        "Strategic Learners",

      isAbout:
        "Students who use good study plans",

      mainIdea:
        "They think AFTER",
    },

    studentResponse:
      "By evaluating results",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  // ====================================================
  // KU MANUAL — COLUMBUS FRAME
  // ====================================================

  {
    id:
      "ED-009",

    title:
      "Manual Figure 3 — Financial Reason",

    source:
      "KU Framing Routine Manual — Figure 3",

    context: {
      keyTopic:
        "What Motivated Columbus?",

      isAbout:
        "Why Columbus crossed the Atlantic Ocean",

      mainIdea:
        "Financial Reasons",
    },

    studentResponse:
      "Get rich by selling spices at home",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-010",

    title:
      "Manual Figure 3 — Religious Reason",

    source:
      "KU Framing Routine Manual — Figure 3",

    context: {
      keyTopic:
        "What Motivated Columbus?",

      isAbout:
        "Why Columbus crossed the Atlantic Ocean",

      mainIdea:
        "Religious Reasons",
    },

    studentResponse:
      "Spread Christianity to other parts of the world",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-011",

    title:
      "Manual Figure 3 — Egotistical Reason",

    source:
      "KU Framing Routine Manual — Figure 3",

    context: {
      keyTopic:
        "What Motivated Columbus?",

      isAbout:
        "Why Columbus crossed the Atlantic Ocean",

      mainIdea:
        "Egotistical Reasons",
    },

    studentResponse:
      "Gain respect",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-012",

    title:
      "Manual Figure 3 — Desire for Recognition",

    source:
      "KU Framing Routine Manual — Figure 3",

    context: {
      keyTopic:
        "What Motivated Columbus?",

      isAbout:
        "Why Columbus crossed the Atlantic Ocean",

      mainIdea:
        "Egotistical Reasons",
    },

    studentResponse:
      "Be the first to prove the world was round",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  // ====================================================
  // KU MANUAL — TITANIC FRAME
  // ====================================================

  {
    id:
      "ED-013",

    title:
      "Manual Figure 6 — Lack of Planning",

    source:
      "KU Framing Routine Manual — Figure 6",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society",

      mainIdea:
        "Lack of Planning",
    },

    studentResponse:
      "Not enough lifeboats",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-014",

    title:
      "Manual Figure 6 — Class System",

    source:
      "KU Framing Routine Manual — Figure 6",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society",

      mainIdea:
        "Class System",
    },

    studentResponse:
      "Rich passengers stayed on the upper deck in luxury",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-015",

    title:
      "Manual Figure 6 — Competition",

    source:
      "KU Framing Routine Manual — Figure 6",

    context: {
      keyTopic:
        "Sinking of the Titanic",

      isAbout:
        "An event that taught lessons about society",

      mainIdea:
        "Competition",
    },

    studentResponse:
      "The ship traveled at its fastest speed to break a crossing record.",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  // ====================================================
  // KU MANUAL — FEMINIST MOVEMENT FRAME
  // ====================================================

  {
    id:
      "ED-016",

    title:
      "Manual Figure 7 — View of Opponents",

    source:
      "KU Framing Routine Manual — Figure 7",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Women having the same rights as men and being treated equally",

      mainIdea:
        "Views of Opponents",
    },

    studentResponse:
      "A woman's place is in the home.",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-017",

    title:
      "Manual Figure 7 — View of Supporters",

    source:
      "KU Framing Routine Manual — Figure 7",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Women having the same rights as men and being treated equally",

      mainIdea:
        "Views of Supporters",
    },

    studentResponse:
      "Men should equally share home responsibilities.",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  {
    id:
      "ED-018",

    title:
      "Manual Figure 7 — Equal Treatment",

    source:
      "KU Framing Routine Manual — Figure 7",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Women having the same rights as men and being treated equally",

      mainIdea:
        "Views of Supporters",
    },

    studentResponse:
      "Laws are needed to ensure equal treatment.",

    expected: {
      valid:
        true,

      diagnosis:
        null,
    },
  },

  // ====================================================
  // SECTION C
  // MANUAL-GROUNDED CONTRAST CASES
  // ====================================================

  {
    id:
      "ED-019",

    title:
      "Manual Contrast — Detail Belongs to Opposing Main Idea",

    source:
      "KU Framing Routine Manual — Figure 7 contrast",

    context: {
      keyTopic:
        "Feminist Movement",

      isAbout:
        "Women having the same rights as men and being treated equally",

      mainIdea:
        "Views of Supporters",
    },

    studentResponse:
      "A woman's place is in the home.",

    expected: {
      valid:
        false,

      allowedDiagnoses: [
        "relationshipNotEstablished",
        "relationshipIncomplete",
        "mainIdeaInsteadOfDetail",
      ],
    },
  },

  {
    id:
      "ED-020",

    title:
      "Manual Contrast — Esoteric Trivia",

    source:
      "KU Framing Routine Manual — Essential Detail classification",

    context: {
      keyTopic:
        "What Motivated Columbus?",

      isAbout:
        "Why Columbus crossed the Atlantic Ocean",

      mainIdea:
        "Financial Reasons",
    },

    studentResponse:
      "Columbus sailed on the Niña, the Pinta, and the Santa María.",

    expected: {
      valid:
        false,

      allowedDiagnoses: [
        "relationshipNotEstablished",
        "relationshipIncomplete",
      ],
    },
  }
);


// ------------------------------------------------------
// IS ABOUT IVL BENCHMARK RUNNER
//
// Runs the full Is About benchmark library through the
// deterministic validator only.
//
// This preserves the existing IVL baseline and does not
// call AI.
// ------------------------------------------------------

async function runIVLIsAboutBenchmarks() {
  const results = [];

  for (
    const benchmark of
    IVL.benchmarks.isAbout
  ) {
    
    const actual =
      await validateIsAboutResponseGoverned(
        benchmark.studentResponse,
        benchmark.context.keyTopic
  );
    
    const passed =
      actual.valid ===
        benchmark.expected.valid &&
      actual.diagnosis ===
        benchmark.expected.diagnosis;

    results.push({
      id:
        benchmark.id,

      title:
        benchmark.title,

      component:
        "isAbout",

      studentResponse:
        benchmark.studentResponse,

      expected:
        benchmark.expected,

      actual,

      passed,
    });
  }

  const passedCount =
    results.filter(
      (result) => result.passed
    ).length;

  const failedCount =
    results.length - passedCount;

  const summary = {
    component:
      "isAbout",

    passed:
      failedCount === 0,

    total:
      results.length,

    passedCount,

    failedCount,

    results,
  };

  IVL.results.isAbout =
    summary;

  return summary;
}

// ------------------------------------------------------
// IA-020 GOVERNED SEMANTIC TEST
//
// Runs only IA-020 through both validators:
//
// 1. deterministic validation
// 2. governed semantic validation
//
// This test does not alter the full IVL or student runtime.
// ------------------------------------------------------

async function runIA020GovernedTest() {
  const benchmark =
    IVL.benchmarks.isAbout.find(
      (item) =>
        item.id === "IA-020"
    );

  if (!benchmark) {
    return {
      passed:
        false,

      id:
        "IA-020",

      error:
        "IA-020 benchmark was not found.",
    };
  }

  const deterministic =
    validateIsAboutResponse(
      benchmark.studentResponse,
      benchmark.context.keyTopic
    );

  const governed =
    await validateIsAboutResponseGoverned(
      benchmark.studentResponse,
      benchmark.context.keyTopic
    );

  const passed =
    governed.valid ===
      benchmark.expected.valid &&

    governed.diagnosis ===
      benchmark.expected.diagnosis;

  const result = {
    id:
      benchmark.id,

    title:
      benchmark.title,

    keyTopic:
      benchmark.context.keyTopic,

    studentResponse:
      benchmark.studentResponse,

    expected:
      benchmark.expected,

    deterministic,

    governed,

    passed,
  };

  console.log("");
  console.log(
    "===================================="
  );
  console.log(
    "IA-020 GOVERNED SEMANTIC TEST"
  );
  console.log(
    "===================================="
  );
  console.log(
    passed
      ? "✅ PASS"
      : "❌ FAIL"
  );
  console.log(
    "Key Topic:",
    result.keyTopic
  );
  console.log(
    "Student Response:",
    result.studentResponse
  );
  console.log(
    "Expected:",
    result.expected
  );
  console.log(
    "Deterministic:",
    result.deterministic
  );
  console.log(
    "Governed:",
    result.governed
  );

  return result;
}

async function runIVLEssentialDetailBenchmarks(
  batchNumber = null,
  batchSize = 5
) {
  console.log("");
  console.log("====================================");
  console.log("IVL - Essential Detail Benchmarks");
  console.log("====================================");

  const allBenchmarks =
    IVL.benchmarks.essentialDetails;

  const totalBenchmarks =
    allBenchmarks.length;

  const totalBatches =
    Math.ceil(
      totalBenchmarks /
      batchSize
    );

  let benchmarksToRun =
    allBenchmarks;

  let normalizedBatchNumber =
    null;

  if (
    Number.isInteger(batchNumber) &&
    batchNumber >= 1
  ) {
    normalizedBatchNumber =
      Math.min(
        batchNumber,
        totalBatches
      );

    const startIndex =
      (normalizedBatchNumber - 1) *
      batchSize;

    const endIndex =
      startIndex +
      batchSize;

    benchmarksToRun =
      allBenchmarks.slice(
        startIndex,
        endIndex
      );
  }

  console.log(
    normalizedBatchNumber
      ? `Batch: ${normalizedBatchNumber}/${totalBatches}`
      : "Batch: all"
  );

  console.log(
    `Benchmarks in this run: ${benchmarksToRun.length}`
  );

  const results = [];

  for (
    const benchmark of
    benchmarksToRun
  ) {
    const actual =
      await validateEssentialDetailResponseGoverned(
        benchmark.studentResponse,
        benchmark.context.mainIdea,
        benchmark.context
  );

    const expectedDiagnosis =
      benchmark.expected.diagnosis;

    const allowedDiagnoses =
      Array.isArray(
        benchmark.expected.allowedDiagnoses
      )
        ? benchmark.expected.allowedDiagnoses
        : [];

    const diagnosisPassed =
      allowedDiagnoses.length > 0
        ? allowedDiagnoses.includes(
            actual.diagnosis
          )
        : actual.diagnosis ===
            expectedDiagnosis;

    const passed =
      actual.valid ===
        benchmark.expected.valid &&
      diagnosisPassed;

    const result = {
      id:
        benchmark.id,

      title:
        benchmark.title,

      component:
        "essentialDetails",

      passed,

      studentResponse:
        benchmark.studentResponse,

      expected:
        benchmark.expected,

      actual
    };

    console.log("");

    console.log(
      `${passed ? "✅ PASS" : "❌ FAIL"} — ${benchmark.id}: ${benchmark.title}`
    );

    if (!passed) {
      console.log(
        "Student Response:",
        benchmark.studentResponse
      );

      console.log(
        "Expected:",
        benchmark.expected
      );

      console.log(
        "Actual:",
        actual
      );
    }

    results.push(result);
  }

  const passedCount =
    results.filter(
      (result) =>
        result.passed
    ).length;

  const failedCount =
    results.length -
    passedCount;

  const summary = {
    component:
      "essentialDetails",

    batchNumber:
      normalizedBatchNumber,

    totalBatches,

    batchSize,

    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    totalBenchmarks,

    results
  };

  IVL.results.essentialDetails =
    summary;

  console.log("");
  console.log("------------------------------------");

  console.log(
    `Passed: ${passedCount}/${results.length}`
  );

  console.log(
    `Failed: ${failedCount}`
  );

  console.log("------------------------------------");

  return summary;
}

async function runIVLMainIdeaBenchmarks(
  batchNumber = null,
  batchSize = 5
) {
  console.log("");
  console.log("====================================");
  console.log("IVL - Main Idea Benchmarks");
  console.log("====================================");

  const allBenchmarks =
    IVL.benchmarks.mainIdeas;

  const totalBenchmarks =
    allBenchmarks.length;

  const totalBatches =
    Math.ceil(
      totalBenchmarks /
      batchSize
    );

  let benchmarksToRun =
    allBenchmarks;

  let normalizedBatchNumber =
    null;

  if (
    Number.isInteger(batchNumber) &&
    batchNumber >= 1
  ) {
    normalizedBatchNumber =
      Math.min(
        batchNumber,
        totalBatches
      );

    const startIndex =
      (normalizedBatchNumber - 1) *
      batchSize;

    const endIndex =
      startIndex +
      batchSize;

    benchmarksToRun =
      allBenchmarks.slice(
        startIndex,
        endIndex
      );
  }

  console.log(
    normalizedBatchNumber
      ? `Batch: ${normalizedBatchNumber}/${totalBatches}`
      : "Batch: all"
  );

  console.log(
    `Benchmarks in this run: ${benchmarksToRun.length}`
  );

  const results = [];

  for (
    const benchmark of
    benchmarksToRun
  ) {
    const actual =
      await validateMainIdeaResponseGoverned(
        benchmark.studentResponse,
        benchmark.context.keyTopic,
        benchmark.context.isAbout
      );

    const expectedDiagnosis =
      benchmark.expected.diagnosis;

    const allowedDiagnoses =
      Array.isArray(
        benchmark.expected.allowedDiagnoses
      )
        ? benchmark.expected.allowedDiagnoses
        : [];

    const diagnosisPassed =
      allowedDiagnoses.length > 0
        ? allowedDiagnoses.includes(
            actual.diagnosis
          )
        : actual.diagnosis ===
            expectedDiagnosis;

    const passed =
      actual.valid ===
        benchmark.expected.valid &&
      diagnosisPassed;

    const result = {
      id:
        benchmark.id,

      title:
        benchmark.title,

      component:
        "mainIdeas",

      passed,

      studentResponse:
        benchmark.studentResponse,

      expected:
        benchmark.expected,

      actual
    };

    console.log("");

    console.log(
      `${passed ? "✅ PASS" : "❌ FAIL"} — ${benchmark.id}: ${benchmark.title}`
    );

    if (!passed) {
      console.log(
        "Student Response:",
        benchmark.studentResponse
      );

      console.log(
        "Expected:",
        benchmark.expected
      );

      console.log(
        "Actual:",
        actual
      );
    }

    results.push(result);
  }

  const passedCount =
    results.filter(
      (result) =>
        result.passed
    ).length;

  const failedCount =
    results.length -
    passedCount;

  const summary = {
    component:
      "mainIdeas",

    batchNumber:
      normalizedBatchNumber,

    batchSize,

    totalBatches,

    totalBenchmarks,

    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total:
      results.length,

    results
  };

  IVL.results.mainIdeas =
    summary;

  console.log("");
  console.log("------------------------------------");

  console.log(
    `Passed: ${passedCount}/${results.length}`
  );

  console.log(
    `Failed: ${failedCount}`
  );

  console.log("------------------------------------");

  return summary;
}

async function runInstructionalValidationLab() {
  IVL.results = {
    isAbout: null,
    mainIdeas: null,
    essentialDetails: null,
    soWhat: null,
    overall: null
  };

  const isAbout =
    await runIVLIsAboutBenchmarks();

  const mainIdeas =
    await runIVLMainIdeaBenchmarks();

  const essentialDetails =
    runIVLEssentialDetailBenchmarks();

  const componentResults = [
    isAbout,
    mainIdeas,
    essentialDetails
  ].filter(Boolean);

  const passedCount =
    componentResults.reduce(
      (total, component) =>
        total + component.passedCount,
      0
    );

  const failedCount =
    componentResults.reduce(
      (total, component) =>
        total + component.failedCount,
      0
    );

  const total =
    componentResults.reduce(
      (sum, component) =>
        sum + component.total,
      0
    );

  IVL.results.overall = {
    passed:
      failedCount === 0,

    passedCount,

    failedCount,

    total,

    components: {
      isAbout: {
        passed:
          isAbout?.passed || false,

        passedCount:
          isAbout?.passedCount || 0,

        failedCount:
          isAbout?.failedCount || 0,

        total:
          isAbout?.total || 0
      },

      mainIdeas: {
        passed:
          mainIdeas?.passed || false,

        passedCount:
          mainIdeas?.passedCount || 0,

        failedCount:
          mainIdeas?.failedCount || 0,

        total:
          mainIdeas?.total || 0
      },

      essentialDetails: {
        passed:
          essentialDetails?.passed || false,

        passedCount:
          essentialDetails?.passedCount || 0,

        failedCount:
          essentialDetails?.failedCount || 0,

        total:
          essentialDetails?.total || 0
      }
    }
  };

  console.log("");
  console.log("====================================");
  console.log("IVL - Overall Results");
  console.log("====================================");

  console.log(
    `Is About: ${isAbout.passedCount}/${isAbout.total}`
  );

  console.log(
    `Main Ideas: ${mainIdeas.passedCount}/${mainIdeas.total}`
  );

  console.log(
    `Essential Details: ${essentialDetails.passedCount}/${essentialDetails.total}`
  );

  console.log(
    `Overall: ${passedCount}/${total}`
  );

  console.log("====================================");

  return IVL.results;
}

function formatInstructionalValidationLabResults(
  ivlResults
) {
  const isAboutSuite =
    ivlResults?.isAbout;

  const mainIdeaSuite =
    ivlResults?.mainIdeas;

  const detailSuite =
    ivlResults?.essentialDetails;

  const overall =
    ivlResults?.overall;

  const lines = [
    "🧪 KAW INSTRUCTIONAL VALIDATION LAB",
    ""
  ];

  function addSuiteResults(
    label,
    suite,
    emptyMessage
  ) {
    lines.push(label);
    lines.push("");

    if (!suite) {
      lines.push(emptyMessage);
      lines.push("");
      return;
    }

    suite.results.forEach(
      (result) => {
        lines.push(
          `${result.passed ? "✅" : "❌"} ${result.id}: ${result.title}`
        );

        lines.push(
          `Student response: ${
            result.studentResponse ||
            "(empty response)"
          }`
        );

        if (!result.passed) {
          lines.push(
            `Expected: ${JSON.stringify(
              result.expected
            )}`
          );

          lines.push(
            `Actual: ${JSON.stringify(
              result.actual
            )}`
          );
        }

        lines.push("");
      }
    );

    lines.push(
      "────────────────────────"
    );

    lines.push(
      `${label}: ${suite.passedCount}/${suite.total} passed`
    );

    lines.push("");
    lines.push("");
  }

  addSuiteResults(
    "Is About",
    isAboutSuite,
    "No Is About results were returned."
  );

  addSuiteResults(
    "Main Ideas",
    mainIdeaSuite,
    "No Main Idea results were returned."
  );

  addSuiteResults(
    "Essential Details",
    detailSuite,
    "No Essential Detail results were returned."
  );

  if (overall) {
    lines.push(
      "================================"
    );

    lines.push(
      `Overall: ${overall.passedCount}/${overall.total} passed`
    );

    lines.push(
      `Failed: ${overall.failedCount}`
    );

    lines.push("");

    lines.push(
      overall.passed
        ? "🚀 All current IVL benchmarks passed."
        : "⚠️ One or more IVL benchmarks did not match the expected instructional outcome."
    );
  }

  return lines.join("\n");
}
