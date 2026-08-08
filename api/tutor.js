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
// KAW OPERATING SYSTEM — DRAFT 1
// ======================================================
//
// Kaw is governed by one coherent instructional operating
// system.
//
// The runtime is organized through the following layers:
//
// 1. CONSTITUTIONAL GOVERNANCE
//    - Defines the permanent principles that govern every
//      architectural layer and runtime implementation.
//    - AI never owns instructional decisions.
//    - Student thinking and student work remain protected.
//    - Instruction advances one intentional thinking step.
//
// 2. INSTRUCTIONAL FRAMEWORK KNOWLEDGE
//    - Defines what students are building.
//    - Stores the instructional purpose, criteria,
//      breakdowns, cognitive strategies, and progression
//      expectations of the active learning framework.
//    - Current framework: KU Framing Routine.
//
// 3. EVIDENCE STATE
//    - Organizes current student evidence, accumulated
//      student evidence, assignment context, completed
//      Frame content, and current instructional location.
//    - Evidence is observed before instruction is selected.
//
// 4. INSTRUCTIONAL ASSESSMENT
//    - Criteria Assessment determines whether current
//      evidence fulfills component expectations.
//    - Relational Assessment determines how current
//      evidence relates to accepted and accumulated
//      evidence.
//    - Interaction Assessment identifies observable
//      conditions such as productive work, uncertainty,
//      struggle, frustration, or off-task behavior.
//    - Assessment produces instructional findings.
//    - Assessment does not select or communicate the
//      instructional response.
//
// 5. INSTRUCTIONAL STRATEGY
//    - Applies predetermined instructional contracts to
//      the established instructional findings.
//    - Selects the instructional objective, Teaching Move,
//      Thinking Move, support level, progression behavior,
//      and student-work protections.
//    - Strategy is deterministic and teacher-authored.
//    - AI does not select pedagogy.
//
// 6. INSTRUCTIONAL COMMUNICATION
//    - Converts the predetermined Instructional Strategy
//      into an approved communication specification.
//    - Defines communication patterns, permissions,
//      prohibitions, and contextual information.
//    - AI may contextualize the predetermined Thinking Move
//      only within the approved Communication License.
//
// 7. RUNTIME PROGRESSION
//    - Preserves the student's exact instructional
//      location.
//    - Controls capture, validation, confirmation,
//      revision, optional expansion, interruption,
//      resumption, and export.
//    - Parent Anchor progression is load-bearing and
//      remains authoritative until intentionally migrated.
//
// 8. DEVELOPMENT VERIFICATION
//    - Deterministic self-tests, governed validation tests,
//      AI communication tests, IVL benchmarks, and student
//      simulations protect behavior during refactoring.
//
// ======================================================
// CONSOLIDATION RULE
// ======================================================
//
// Every runtime responsibility must belong to one layer
// of this operating system.
//
// Legacy Kaw 1.0 code must be:
//
// - deleted when obsolete;
// - merged when it duplicates an authoritative subsystem;
// - revised when useful logic remains;
// - reused only when it supports the current architecture;
// - temporarily retained only when removal would endanger
//   working behavior.
//
// Temporary legacy code must have an explicit migration or
// removal destination.
//
// ======================================================
// REFACTOR SAFETY RULE
// ======================================================
//
// Parent Anchor, Child Anchor, pending-state progression,
// accepted student work, and validated runtime behavior are
// load-bearing.
//
// Do not rewrite load-bearing behavior merely to improve
// organization.
//
// First establish the new layer around existing behavior.
// Then migrate responsibility.
// Then verify behavior.
// Only then remove the superseded pathway.

// ======================================================
// HIGH-IMPACT LEARNING STRATEGY — KU FRAMING ROUTINE
// ======================================================
//
// This section defines the instructional knowledge of the
// active learning framework.
//
// It is permanent instructional knowledge—not runtime
// evidence, assessment, strategy, or communication.
//
// Current framework:
// KU Framing Routine
//
// Responsibilities:
//
// • Define each Frame component.
// • Define instructional purpose.
// • Define success criteria.
// • Define common misconceptions.
// • Define cognitive strategies.
// • Define progression expectations.
// • Support deterministic instructional reasoning.
//
// This layer describes what Kaw knows about the framework.
// It does not determine what Kaw does for an individual
// student.
//
// ======================================================

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
    "Let's strengthen your Key Topic so it clearly names the topic you'll be exploring in this Frame.\n\nWhat is the main topic?"
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
  '💬 Is About\n\nLet\'s strengthen your Is About so it clearly explains what your whole Key Topic is about in your own words.\n\nWhat would you like it to say instead?',

confirmationPrompt:
  '✅ Checkpoint\n\n💬 Is About\n\nNice work! Your Is About explains your Key Topic in your own words:\n\n🧩 Key Topic: {keyTopic}\n💬 Is About: {isAbout}\n\nDoes this accurately capture your thinking?\n\n1) Yes — Continue building my Frame.\n2) No — Revise my Is About.\n\nReply with 1 or 2.' 
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
      'So far your Frame says:\n\n🧩 Key Topic: {keyTopic}\n💬 Is About: {isAbout}\n\nWhat is one Main Idea that helps explain your Key Topic?',

    additionalPrompt:
      'So far your Frame says:\n\n🧩 Key Topic: {keyTopic}\n💬 Is About: {isAbout}\n\nWhat is another Main Idea that helps explain your Key Topic?',

    revisePrompt:
      "💡 Let's strengthen your Main Idea so it clearly connects to your Key Topic and can be explained with Essential Details.",

    confirmationPrompt:
      '✅ Checkpoint\n\n💡 Main Ideas\n\nYou\'ve built these Main Ideas to help explain your Key Topic:\n\n{mainIdeasList}\n\nDoes this accurately capture your thinking?\n\n1) Yes — Continue building my Frame.\n2) No — Revise one Main Idea.\n\nReply with 1 or 2.'
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
      'What is one Essential Detail that helps explain this Main Idea?',

    additionalPrompt:
      'What is another Essential Detail that helps explain this Main Idea?',

    revisePrompt:
      "✍️ Essential Detail\n\nLet's strengthen your Essential Detail so it adds specific information that helps explain this Main Idea."
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
      "🎯 Let's strengthen your So What so it explains what's important to understand from your whole Frame, rather than repeating just one part."
  }
}

};

// ======================================================
// AI OBSERVATION LAYER
// ======================================================
//
// The AI Observation Layer examines the student's current
// interaction and produces one governed Observation Report.
//
// Observation reports directly observable evidence only.
//
// Observation may identify:
// • uncertainty language;
// • clarification requests;
// • answer-seeking;
// • frustration language;
// • refusal;
// • off-task shifts;
// • assignment references;
// • Framing Routine references;
// • acknowledgement of prior coaching;
// • repeated attempts.
//
// Observation may not determine:
// • instructional situation;
// • genuine struggle;
// • mastery;
// • misconception;
// • readiness;
// • progression;
// • teaching strategy;
// • instructional intent.
//
// The Observation Report is evidence.
// It is not an instructional decision.
//
// Current migration status:
//
// Shadow mode.
//
// The Observation Report is included in Evidence State but
// does not yet control Instructional Assessment, Strategy,
// progression, pending state, or communication.
//
// ======================================================

const OBSERVATION_CATEGORIES = new Set([
  "uncertaintyExpression",
  "clarificationRequest",
  "answerSeeking",
  "frustrationExpression",
  "refusal",
  "offTaskShift",
  "assignmentReference",
  "framingRoutineReference",
  "acknowledgesPriorCoaching",
  "repeatedAttempt",
]);

function buildEmptyObservationReport(
  studentInteraction = "",
  source = "notObserved"
) {
  return {
    version: "1.0",

    source,

    studentInteraction:
      cleanText(studentInteraction),

    observations: [],

    ambiguityPresent: false,
  };
}

function getRecentStudentResponses(
  state,
  limit = 3
) {
  const transcript =
    Array.isArray(state?.transcript)
      ? state.transcript
      : [];

  return transcript
    .filter(
      (turn) =>
        turn?.role === "Student" &&
        cleanText(turn?.text)
    )
    .slice(-limit)
    .map(
      (turn) =>
        cleanText(turn.text)
    );
}

function sanitizeObservationReport(
  rawReport,
  studentInteraction
) {
  const text =
    cleanText(studentInteraction);

  const normalizedText =
    text.toLowerCase();

  const rawObservations =
    Array.isArray(rawReport?.observations)
      ? rawReport.observations
      : [];

  const observations =
    rawObservations
      .filter((observation) => {
        const category =
          cleanText(
            observation?.category
          );

        const evidenceText =
          cleanText(
            observation?.evidenceText
          );

        if (
          !OBSERVATION_CATEGORIES.has(
            category
          )
        ) {
          return false;
        }

        if (!evidenceText) {
          return false;
        }

        // Every observation must point to language that
        // actually appears in the student's interaction.
        return normalizedText.includes(
          evidenceText.toLowerCase()
        );
      })
      .map((observation) => {
        const confidence =
          Number(
            observation?.confidence || 0
          );

        return {
          category:
            cleanText(
              observation.category
            ),

          evidenceText:
            cleanText(
              observation.evidenceText
            ),

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
        };
      });

  return {
    version: "1.0",

    source:
      "aiObservation",

    studentInteraction:
      text,

    observations,

    ambiguityPresent:
      rawReport?.ambiguityPresent ===
      true,
  };
}

async function buildObservationReport(
  state,
  studentInteraction = ""
) {
  const text =
    cleanText(studentInteraction);

  if (!text) {
    return buildEmptyObservationReport(
      text,
      "emptyInteraction"
    );
  }

  const recentStudentResponses =
    getRecentStudentResponses(
      state,
      3
    );

  const system = `You are the governed AI Observation Layer for Kaw Companion.

Your only responsibility is to report directly observable evidence from the student's current interaction.

You may identify only these observation categories:
- uncertaintyExpression
- clarificationRequest
- answerSeeking
- frustrationExpression
- refusal
- offTaskShift
- assignmentReference
- framingRoutineReference
- acknowledgesPriorCoaching
- repeatedAttempt

Rules:
- Report observations only.
- Do not determine instructional meaning.
- Do not classify genuine struggle.
- Do not determine mastery, readiness, success, failure, misconception, progression, support level, teaching strategy, or instructional intent.
- Do not infer hidden emotion, motivation, effort, ability, knowledge, or understanding.
- Every observation must include an exact excerpt copied from the student's current interaction.
- Do not paraphrase the evidence excerpt.
- Include only observations directly supported by the student's words.
- An empty observations array is valid.
- Return only the required JSON object.`;

  const user = `Current student interaction:
"${text}"

Recent student responses:
${JSON.stringify(
  recentStudentResponses,
  null,
  2
)}

Report only directly observable evidence from the current student interaction.`;

  try {
    const response =
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
              "kaw_observation_report",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                observations: {
                  type:
                    "array",

                  items: {
                    type:
                      "object",

                    additionalProperties:
                      false,

                    properties: {
                      category: {
                        type:
                          "string",

                        enum: [
                          "uncertaintyExpression",
                          "clarificationRequest",
                          "answerSeeking",
                          "frustrationExpression",
                          "refusal",
                          "offTaskShift",
                          "assignmentReference",
                          "framingRoutineReference",
                          "acknowledgesPriorCoaching",
                          "repeatedAttempt",
                        ],
                      },

                      evidenceText: {
                        type:
                          "string",
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
                      "category",
                      "evidenceText",
                      "confidence",
                    ],
                  },
                },

                ambiguityPresent: {
                  type:
                    "boolean",
                },
              },

              required: [
                "observations",
                "ambiguityPresent",
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
        response?.choices?.[0]
          ?.message?.content || "{}"
      );

    return sanitizeObservationReport(
      parsed,
      text
    );
  } catch (error) {
    console.error(
      "Observation Report error:",
      error
    );

    // Observation failure must never prevent Kaw from
    // continuing through the existing runtime.
    return buildEmptyObservationReport(
      text,
      "observationUnavailable"
    );
  }
}

// ======================================================
// EVIDENCE STATE
// ======================================================
//
// Evidence State provides one read-only representation of
// the observable information available to Kaw at the
// current instructional moment.
//
// It organizes:
//
// • the student's current response;
// • accepted and accumulated student work;
// • assignment and Thinking Task context;
// • the student's current instructional location;
// • active pending and feedback context.
//
// Evidence State does not:
//
// • validate student work;
// • diagnose an instructional condition;
// • select an instructional strategy;
// • change progression;
// • mutate runtime state;
// • generate student work.
//
// It observes and organizes evidence only.
//
// ======================================================

function buildEvidenceState(
  state,
  currentResponse = "",
  observationReport = null
) {
  const safeState =
    state && typeof state === "object"
      ? state
      : {};

  const frame =
    safeState?.frame &&
    typeof safeState.frame === "object"
      ? safeState.frame
      : {};

  const frameMeta =
    safeState?.frameMeta &&
    typeof safeState.frameMeta === "object"
      ? safeState.frameMeta
      : {};

  const assignmentContext =
    frameMeta?.assignmentContext &&
    typeof frameMeta.assignmentContext === "object"
      ? frameMeta.assignmentContext
      : {};

  const thinkingTask =
    safeState?.assignmentReasoning &&
    typeof safeState.assignmentReasoning === "object"
      ? safeState.assignmentReasoning
      : {};

  const mainIdeas =
    getIdeaList(safeState)
      .map((idea) => cleanText(idea))
      .filter(Boolean);

  const details =
    Array.isArray(frame?.details)
      ? frame.details.map((detailGroup) =>
          Array.isArray(detailGroup)
            ? detailGroup
                .map((detail) =>
                  cleanText(detail)
                )
                .filter(Boolean)
            : []
        )
      : [];

  const pending =
    safeState?.pending &&
    typeof safeState.pending === "object"
      ? structuredClone(safeState.pending)
      : null;

  return {
  currentEvidence: {
    response:
      cleanText(currentResponse),
  },

  observationReport:
    observationReport &&
    typeof observationReport === "object"
      ? structuredClone(
          observationReport
        )
      : buildEmptyObservationReport(
          currentResponse,
          "notProvided"
        ),

  accumulatedEvidence: {
      assignmentContext:
        structuredClone(assignmentContext),

      thinkingTask:
        structuredClone(thinkingTask),

      frame: {
          keyTopic:
            cleanText(frame?.keyTopic || ""),
  
          isAbout:
            cleanText(frame?.isAbout || ""),
  
          mainIdeas,
  
          details,
  
          soWhat:
            cleanText(frame?.soWhat || ""),
      },
    },

    instructionalLocation: {
      interactionMode:
        cleanText(
          safeState?.interactionMode ||
          "build"
        ),

      rawStage:
        getStage(safeState),

      parentAnchor:
        getParentAnchorContext(safeState),

      pendingType:
        cleanText(
          safeState?.pending?.type || ""
        ),

      pending,
    },
  };
}

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
    noComponentEvidence: {
      contractId:
        "IA-NCE-001",

      frameComponent:
        "isAbout",

      instructionalSituation:
        "noComponentEvidence",

      instructionalGoal:
        "elicitComponentEvidence",

      teachingMove:
        "reduceCognitiveLoad",

      thinkingMove:
        "Reconnect the student to the accepted Key Topic and invite them to explain what the whole topic is about in their own understandable words without suggesting or supplying the Is About statement.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "isAbout",

        description:
          "The student's next response must provide observable Is About content that can be evaluated as a whole-topic paraphrase.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the active Is About capture or revision location and validate the student's next response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyParaphrase:
          true,

        neverReplaceKeyTopic:
          true,

        neverInferMeaning:
          true,
      },
    },

    componentNeedsRevision: {
      contractId:
        "IA-CNR-001",

      frameComponent:
        "isAbout",

      instructionalSituation:
        "componentNeedsRevision",

      instructionalGoal:
        "strengthenWholeTopicParaphrase",

      teachingMove:
        "increaseSpecificity",

      thinkingMove:
        "Invite the student to expand the response so that someone unfamiliar with the accepted Key Topic could understand what the whole topic is about without supplying the missing meaning.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "isAbout",

        description:
          "The revised response must provide enough observable meaning to function as an understandable whole-topic paraphrase.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the active Is About capture or revision location and validate the student's revised response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyParaphrase:
          true,

        neverReplaceKeyTopic:
          true,

        neverInferMeaning:
          true,
      },
    },

    relationshipNeedsRepair: {
      contractId:
        "IA-RNR-001",

      frameComponent:
        "isAbout",

      instructionalSituation:
        "relationshipNeedsRepair",

      instructionalGoal:
        "establishWholeTopicRelationship",

      teachingMove:
        "differentiate",

      thinkingMove:
        "Reconnect the student to the accepted Key Topic and invite them to explain what the whole topic is about rather than repeating only the Key Topic or another disconnected idea.",

      communicationPattern:
        "acknowledgeThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "isAbout",

        description:
          "The revised response must establish an observable whole-topic paraphrase relationship to the accepted Key Topic.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the active Is About capture or revision location and validate whether the required relationship is established.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyParaphrase:
          true,

        neverReplaceKeyTopic:
          true,

        neverInferRelationship:
          true,
      },
    },

    genuineStruggle: {
      contractId:
        "IA-GS-001",

      frameComponent:
        "isAbout",

      instructionalSituation:
        "genuineStruggle",

      instructionalGoal:
        "restartThinking",

      teachingMove:
        "clarify",

      thinkingMove:
        "Explain what the whole Key Topic is about using your own understandable words.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "isAbout",

        description:
          "The student provides an Is About statement that observably paraphrases and summarizes the whole Key Topic.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the Is About statement where support was requested and validate the student's next response.",
      },

      progressiveSupport: {
        principle:
          "If the first intervention does not restart productive thinking, provide progressively more targeted support without supplying the student's Is About statement.",

        scaffolds: [
          {
            level:
              1,

            move:
              "refocus",

            purpose:
              "Reconnect the student to the accepted Key Topic.",
          },

          {
            level:
              2,

            move:
              "remind",

            purpose:
              "Remind the student that Is About explains the whole Key Topic in their own words.",
          },

          {
            level:
              3,

            move:
              "thinkingPrompt",

            purpose:
              "Ask what someone unfamiliar with the topic should understand about it.",
          },
        ],
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyParaphrase:
          true,

        neverReplaceKeyTopic:
          true,

        neverInferMeaning:
          true,
      },
    },

    readyToProgress: {
      contractId:
        "IA-RTP-001",

      frameComponent:
        "isAbout",

      instructionalSituation:
        "readyToProgress",

      instructionalGoal:
        "preserveAcceptedThinkingAndAdvance",

      teachingMove:
        "confirm",

      thinkingMove:
        null,

      communicationPattern:
        null,

      aiContextualizes:
        false,

      validation: {
        type:
          "isAbout",

        description:
          "The accepted Is About statement satisfies component criteria and establishes the required relationship to the Key Topic.",
      },

      progressionBehavior: {
        type:
          "continueExistingRuntimeProgression",

        description:
          "Preserve the accepted Is About statement and continue through the existing confirmation and Parent Anchor progression pathway.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveAcceptedIsAbout:
          true,

        neverRewriteStudentWork:
          true,

        neverGenerateStudentWork:
          true,

        neverAdvanceWithoutEstablishedRelationship:
          true,
      },
    },
  },
  
    mainIdeas: {
    noComponentEvidence: {
      contractId:
        "MI-NCE-001",

      frameComponent:
        "mainIdeas",

      instructionalSituation:
        "noComponentEvidence",

      instructionalGoal:
        "elicitMainIdeaEvidence",

      teachingMove:
        "reduceCognitiveLoad",

      thinkingMove:
        "Reconnect the student to the accepted Key Topic and Is About statement, then invite them to identify one larger idea, category, cause, effect, part, stage, pattern, or major event that helps organize the topic. Do not suggest or generate the Main Idea.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "mainIdea",

        description:
          "The student's next response must provide observable Main Idea content that can be evaluated as one major organizing idea within the accepted Frame.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Main Idea location and validate the student's next response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyMainIdea:
          true,

        neverReplaceKeyTopic:
          true,

        neverReplaceIsAbout:
          true,

        neverInferMeaning:
          true,
      },
    },

    relationshipNeedsRepair: {
      contractId:
        "MI-RNR-001",

      frameComponent:
        "mainIdeas",

      instructionalSituation:
        "relationshipNeedsRepair",

      instructionalGoal:
        "establishMainIdeaRelationship",

      teachingMove:
        "differentiate",

      thinkingMove:
        "Reconnect the student to the accepted Key Topic and Is About statement, then invite them to identify one larger organizing idea within that topic rather than repeating the whole topic or supplying only one specific Essential Detail. Do not provide the Main Idea.",

      communicationPattern:
        "acknowledgeThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "mainIdea",

        description:
          "The revised response must establish an observable relationship to the accepted Key Topic and Is About statement and function as one organizing Main Idea rather than only an Essential Detail.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Main Idea location and validate whether the required organizing relationship is established.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyMainIdea:
          true,

        neverReplaceKeyTopic:
          true,

        neverReplaceIsAbout:
          true,

        neverInferRelationship:
          true,
      },
    },

    componentNeedsRevision: {
      contractId:
        "MI-CNR-001",

      frameComponent:
        "mainIdeas",

      instructionalSituation:
        "componentNeedsRevision",

      instructionalGoal:
        "strengthenOrganizingMainIdea",

      teachingMove:
        "increaseSpecificity",

      thinkingMove:
        "Invite the student to expand or clarify the response enough to show the larger idea they want this section of the Frame to organize and the kinds of Essential Details that could fit beneath it. Do not infer or supply the missing meaning.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "mainIdea",

        description:
          "The revised response must provide enough observable meaning to function as one understandable organizing Main Idea that can support multiple Essential Details.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Main Idea location and validate the student's revised response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyMainIdea:
          true,

        neverReplaceStudentThinking:
          true,

        neverInferMeaning:
          true,
      },
    },

    genuineStruggle: {
      contractId:
        "MI-GS-001",

      frameComponent:
        "mainIdeas",

      instructionalSituation:
        "genuineStruggle",

      instructionalGoal:
        "restartThinking",

      teachingMove:
        "clarify",

      thinkingMove:
        "Explain the larger idea that this Main Idea helps your reader understand about the topic.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "mainIdea",

        description:
          "The student provides an organizing idea that explains the topic and can be supported by multiple Essential Details.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Main Idea location where support was requested and validate the student's next response.",
      },

      progressiveSupport: {
        principle:
          "If the first intervention does not restart productive thinking, provide progressively more targeted support without supplying the student's Main Idea.",

        scaffolds: [
          {
            level:
              1,

            move:
              "refocus",

            purpose:
              "Reconnect the student to the accepted Key Topic and Is About statement.",
          },

          {
            level:
              2,

            move:
              "differentiate",

            purpose:
              "Remind the student that a Main Idea is one larger organizing idea rather than the whole topic or one specific Essential Detail.",
          },

          {
            level:
              3,

            move:
              "organizingPrompt",

            purpose:
              "Invite the student to identify one major category, cause, effect, part, stage, pattern, or event that could organize several Essential Details.",
          },
        ],
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyMainIdea:
          true,

        neverChooseOrganizer:
          true,

        neverReplaceStudentThinking:
          true,
      },
    },

    readyToProgress: {
      contractId:
        "MI-RTP-001",

      frameComponent:
        "mainIdeas",

      instructionalSituation:
        "readyToProgress",

      instructionalGoal:
        "preserveAcceptedThinkingAndAdvance",

      teachingMove:
        "confirm",

      thinkingMove:
        null,

      communicationPattern:
        null,

      aiContextualizes:
        false,

      validation: {
        type:
          "mainIdea",

        description:
          "The accepted response satisfies Main Idea criteria, establishes the required relationship to the accepted Frame, and can organize multiple Essential Details.",
      },

      progressionBehavior: {
        type:
          "continueExistingRuntimeProgression",

        description:
          "Preserve the accepted Main Idea and continue through the existing required, optional, or revision Main Idea progression pathway.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveAcceptedMainIdea:
          true,

        preserveCaptureMode:
          true,

        neverRewriteStudentWork:
          true,

        neverGenerateStudentWork:
          true,

        neverAdvanceWithoutEstablishedRelationship:
          true,
      },
    },
  },

    details: {
    noComponentEvidence: {
      contractId:
        "ED-NCE-001",

      frameComponent:
        "details",

      instructionalSituation:
        "noComponentEvidence",

      instructionalGoal:
        "elicitEssentialDetailEvidence",

      teachingMove:
        "reduceCognitiveLoad",

      thinkingMove:
        "Reconnect the student to the accepted Main Idea and invite them to identify one concrete fact, example, observation, explanation, or piece of evidence that could support it. Do not suggest or generate the Essential Detail.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "essentialDetail",

        description:
          "The student's next response must provide observable Essential Detail content that can be evaluated as support for the accepted Main Idea.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Essential Detail location and validate the student's next response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyEssentialDetail:
          true,

        neverReplaceMainIdea:
          true,

        neverInferMeaning:
          true,
      },
    },

    relationshipNeedsRepair: {
      contractId:
        "ED-RNR-001",

      frameComponent:
        "details",

      instructionalSituation:
        "relationshipNeedsRepair",

      instructionalGoal:
        "establishSupportingRelationship",

      teachingMove:
        "clarifyConnection",

      thinkingMove:
        "Reference the student's observable idea without claiming that it already supports the accepted Main Idea, then invite the student to explain how the idea connects to and supports that Main Idea. Do not supply the connection or generate a replacement Essential Detail.",

      communicationPattern:
        "acknowledgeThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "essentialDetail",

        description:
          "The revised response must establish an observable supporting relationship between the Essential Detail and the accepted Main Idea.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Essential Detail location and validate whether the required supporting relationship is established.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyEssentialDetail:
          true,

        neverSupplyConnection:
          true,

        neverReplaceMainIdea:
          true,

        neverInferRelationship:
          true,
      },
    },

    componentNeedsRevision: {
      contractId:
        "ED-CNR-001",

      frameComponent:
        "details",

      instructionalSituation:
        "componentNeedsRevision",

      instructionalGoal:
        "strengthenEssentialDetailEvidence",

      teachingMove:
        "increaseSpecificity",

      thinkingMove:
        "Invite the student to make the response more specific by identifying one concrete fact, example, observation, explanation, or piece of evidence related to the accepted Main Idea. Do not infer or supply the missing information.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "essentialDetail",

        description:
          "The revised response must provide enough observable and specific information to function as an Essential Detail beneath the accepted Main Idea.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Essential Detail location and validate the student's revised response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyEssentialDetail:
          true,

        neverReplaceStudentThinking:
          true,

        neverInferMeaning:
          true,
      },
    },

    genuineStruggle: {
      contractId:
        "ED-GS-001",

      frameComponent:
        "details",

      instructionalSituation:
        "genuineStruggle",

      instructionalGoal:
        "restartThinking",

      teachingMove:
        "recall",

      thinkingMove:
        "Think of one supporting fact, example, observation, explanation, or piece of evidence that supports this Main Idea.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "essentialDetail",

        description:
          "The student provides a clear Essential Detail that directly supports the current Main Idea.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact required, optional, or revision Essential Detail location and Main Idea where support was requested.",
      },

      progressiveSupport: {
        principle:
          "If the first intervention does not restart productive thinking, provide progressively more targeted support without supplying the student's Essential Detail.",

        scaffolds: [
          {
            level:
              1,

            move:
              "refocus",

            purpose:
              "Reconnect the student to the accepted Main Idea.",
          },

          {
            level:
              2,

            move:
              "differentiate",

            purpose:
              "Remind the student that an Essential Detail is one specific fact, example, observation, explanation, or piece of evidence that supports the Main Idea.",
          },

          {
            level:
              3,

            move:
              "supportingEvidencePrompt",

            purpose:
              "Invite the student to recall one concrete piece of information that helps explain, illustrate, demonstrate, or support the accepted Main Idea.",
          },
        ],
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyEssentialDetail:
          true,

        neverChooseEvidence:
          true,

        neverReplaceStudentThinking:
          true,
      },
    },

    readyToProgress: {
      contractId:
        "ED-RTP-001",

      frameComponent:
        "details",

      instructionalSituation:
        "readyToProgress",

      instructionalGoal:
        "preserveAcceptedThinkingAndAdvance",

      teachingMove:
        "confirm",

      thinkingMove:
        null,

      communicationPattern:
        null,

      aiContextualizes:
        false,

      validation: {
        type:
          "essentialDetail",

        description:
          "The accepted response satisfies Essential Detail criteria and establishes the required supporting relationship to the accepted Main Idea.",
      },

      progressionBehavior: {
        type:
          "continueExistingRuntimeProgression",

        description:
          "Preserve the accepted Essential Detail and continue through the existing required, optional, or revision Essential Detail progression pathway.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveAcceptedEssentialDetail:
          true,

        preserveCaptureMode:
          true,

        neverRewriteStudentWork:
          true,

        neverGenerateStudentWork:
          true,

        neverAdvanceWithoutEstablishedRelationship:
          true,
      },
    },
  },
  
    soWhat: {
    noComponentEvidence: {
      contractId:
        "SW-NCE-001",

      frameComponent:
        "soWhat",

      instructionalSituation:
        "noComponentEvidence",

      instructionalGoal:
        "elicitSoWhatEvidence",

      teachingMove:
        "reduceCognitiveLoad",

      thinkingMove:
        "Reconnect the student to the completed Frame and invite them to identify one larger understanding, conclusion, connection, implication, or takeaway that becomes clear when the Main Ideas and Essential Details are considered together. Do not suggest or generate the So What.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "soWhat",

        description:
          "The student's next response must provide observable So What content that can be evaluated as a culminating understanding supported by the completed Frame.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact initial, additional-content, or revision So What location and validate the student's next response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyTakeaway:
          true,

        neverChooseConclusion:
          true,

        neverInferMeaning:
          true,
      },
    },

    relationshipNeedsRepair: {
      contractId:
        "SW-RNR-001",

      frameComponent:
        "soWhat",

      instructionalSituation:
        "relationshipNeedsRepair",

      instructionalGoal:
        "establishCompletedFrameRelationship",

      teachingMove:
        "reconnectToFrame",

      thinkingMove:
        "Reconnect the student to the accepted Key Topic, Main Ideas, and Essential Details, then invite them to explain the larger understanding that those completed parts support together. Do not supply the connection or generate a replacement So What.",

      communicationPattern:
        "acknowledgeThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "soWhat",

        description:
          "The revised response must establish an observable relationship to the completed Frame and communicate a supported culminating understanding.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact initial, additional-content, or revision So What location and validate whether the required completed-Frame relationship is established.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyTakeaway:
          true,

        neverSupplyConnection:
          true,

        neverChooseConclusion:
          true,

        neverInferRelationship:
          true,
      },
    },

    componentNeedsRevision: {
      contractId:
        "SW-CNR-001",

      frameComponent:
        "soWhat",

      instructionalSituation:
        "componentNeedsRevision",

      instructionalGoal:
        "strengthenCulminatingUnderstanding",

      teachingMove:
        "increaseSpecificity",

      thinkingMove:
        "Invite the student to clarify the actual larger understanding, significance, implication, connection, or takeaway they reached after considering the completed Frame. Do not infer or supply the missing meaning.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "soWhat",

        description:
          "The revised response must communicate a meaningful and sufficiently specific culminating understanding rather than only a vague statement or repetition of earlier Frame content.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact initial, additional-content, or revision So What location and validate the student's revised response.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveUnvalidatedRevision:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyTakeaway:
          true,

        neverReplaceStudentSynthesis:
          true,

        neverInferMeaning:
          true,
      },
    },

    genuineStruggle: {
      contractId:
        "SW-GS-001",

      frameComponent:
        "soWhat",

      instructionalSituation:
        "genuineStruggle",

      instructionalGoal:
        "strengthenSynthesis",

      teachingMove:
        "synthesize",

      thinkingMove:
        "Explain the larger understanding, conclusion, connection, implication, or takeaway that becomes clear after considering the completed Frame.",

      communicationPattern:
        "briefReassuranceThenQuestion",

      aiContextualizes:
        true,

      validation: {
        type:
          "soWhat",

        description:
          "The student provides a meaningful culminating understanding that is anchored to and supported by the completed Frame.",
      },

      resumeBehavior: {
        type:
          "returnToExactInstructionalLocation",

        description:
          "Return to the exact initial, additional-content, or revision So What location where support was requested and validate the student's next response.",
      },

      progressiveSupport: {
        principle:
          "If the first intervention does not restart productive synthesis, provide progressively more targeted support without supplying the student's So What.",

        scaffolds: [
          {
            level:
              1,

            move:
              "refocus",

            purpose:
              "Reconnect the student to the completed Frame and its Key Topic.",
          },

          {
            level:
              2,

            move:
              "synthesize",

            purpose:
              "Invite the student to identify what becomes important or clear when the Main Ideas and Essential Details are considered together.",
          },

          {
            level:
              3,

            move:
              "significancePrompt",

            purpose:
              "Invite the student to explain a larger conclusion, connection, implication, application, or life truth supported by the completed Frame.",
          },
        ],
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveCaptureMode:
          true,

        neverSaveStruggleLanguage:
          true,

        neverGenerateStudentWork:
          true,

        neverSupplyTakeaway:
          true,

        neverChooseConclusion:
          true,

        neverReplaceStudentSynthesis:
          true,
      },
    },

    readyToProgress: {
      contractId:
        "SW-RTP-001",

      frameComponent:
        "soWhat",

      instructionalSituation:
        "readyToProgress",

      instructionalGoal:
        "preserveAcceptedThinkingAndAdvance",

      teachingMove:
        "confirm",

      thinkingMove:
        null,

      communicationPattern:
        null,

      aiContextualizes:
        false,

      validation: {
        type:
          "soWhat",

        description:
          "The accepted response communicates a meaningful culminating understanding that is anchored to, traceable to, and supported by the completed Frame.",
      },

      progressionBehavior: {
        type:
          "continueExistingRuntimeProgression",

        description:
          "Preserve the accepted So What and continue through the existing optional expansion, confirmation, and export progression pathway.",
      },

      studentWorkProtection: {
        preserveExistingWork:
          true,

        preserveAcceptedSoWhat:
          true,

        preserveCaptureMode:
          true,

        neverRewriteStudentWork:
          true,

        neverGenerateStudentWork:
          true,

        neverAdvanceWithoutSupportedSynthesis:
          true,
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

// ------------------------------------------------------
// INSTRUCTIONAL CONTRACT SELECTION
// ------------------------------------------------------
//
// Selects the predetermined Instructional Contract that
// corresponds to the established governed Instructional
// Situation.
//
// This selector answers exactly one question:
//
// Which predetermined contract corresponds to the
// established Frame component and Instructional Situation?
//
// Current authoritative scope:
//
// • Is About
// • Main Ideas
// • Essential Details
// • So What
// The selected contract controls authoritative
// instructional activation and communication for the
// currently migrated Frame components.
//
// This selector does not:
//
// • validate student work;
// • determine the Instructional Situation;
// • determine support level;
// • save or reject student work;
// • directly change runtime progression.
//
// ------------------------------------------------------

function buildInstructionalContractSelection(
  instructionalSituationArtifact
) {
  const safeSituation =
    instructionalSituationArtifact &&
    typeof instructionalSituationArtifact ===
      "object"
      ? instructionalSituationArtifact
      : null;

  const frameComponent =
    cleanText(
      safeSituation?.frameComponent || ""
    );

  const instructionalSituation =
    cleanText(
      safeSituation
        ?.instructionalSituation || ""
    );

  const isWithinCurrentAuthoritativeScope =
    frameComponent === "isAbout" ||
    frameComponent === "mainIdeas" ||
    frameComponent === "details" ||
    frameComponent === "soWhat";
  
  const selectedContract =
    isWithinCurrentAuthoritativeScope &&
    instructionalSituation
      ? getInstructionalContract(
          frameComponent,
          instructionalSituation
        )
      : null;

  return {
    artifactType:
      "instructionalContractSelection",

    version:
      "1.0",

    source:
      "deterministicInstructionalContractSelector",

    frameComponent:
      frameComponent || null,

    instructionalSituation:
      instructionalSituation || null,

    selectionStatus:
      !isWithinCurrentAuthoritativeScope
        ? "outsideCurrentAuthoritativeScope"
        : selectedContract
          ? "contractSelected"
          : "contractUnavailable",

    selectedContractId:
      selectedContract?.contractId || null,

    selectedContract:
      selectedContract
        ? structuredClone(
            selectedContract
          )
        : null,

    governance: {
        currentAuthoritativeScope:
        "isAbout, mainIdeas, details, soWhat",

      contractExecuted:
        selectedContract !== null,

      controlsProgression:
        false,

      controlsPendingState:
        false,

      controlsCommunication:
        selectedContract !== null,

      authoritative:
        true,

      shadowMode:
        false,
    },
  };
}

// ======================================================
// LAYER 4 — INSTRUCTIONAL ASSESSMENT
// ======================================================
//
// Instructional Assessment organizes and interprets
// observable instructional evidence.
//
// It produces:
//
// • Criteria Assessment;
// • Relational Assessment;
// • Interaction Assessment;
// • governed instructional findings.
//
// Instructional Assessment does not:
//
// • save or reject student work;
// • select an Instructional Contract;
// • determine a Teaching Move or Thinking Move;
// • control runtime progression;
// • generate student-facing communication.
//
// Current component validation occurs inside the active
// runtime branch. After validation, the assessment is
// refreshed with the current Component Instructional
// Finding and governed Instructional Situation.
//
// ======================================================

function buildInstructionalAssessment(
  evidenceState
) {
  const safeEvidenceState =
    evidenceState &&
    typeof evidenceState === "object"
      ? evidenceState
      : {};

  const instructionalLocation =
    safeEvidenceState
      ?.instructionalLocation &&
    typeof safeEvidenceState
      .instructionalLocation === "object"
      ? safeEvidenceState
          .instructionalLocation
      : {};

  const observationReport =
    safeEvidenceState
      ?.observationReport &&
    typeof safeEvidenceState
      .observationReport === "object"
      ? safeEvidenceState
          .observationReport
      : buildEmptyObservationReport(
          safeEvidenceState
            ?.currentEvidence
            ?.response || "",
          "notProvided"
        );

  const observations =
    Array.isArray(
      observationReport?.observations
    )
      ? observationReport.observations
          .filter(
            (observation) =>
              observation &&
              typeof observation ===
                "object"
          )
          .map(
            (observation) =>
              structuredClone(
                observation
              )
          )
      : [];

  const observedCategories =
    [
      ...new Set(
        observations
          .map(
            (observation) =>
              cleanText(
                observation?.category
              )
          )
          .filter(Boolean)
      ),
    ];

  const hasObservedCategory =
    (category) =>
      observedCategories.includes(
        category
      );

  const existingInstructionalFinding =
    instructionalLocation
      ?.pending
      ?.instructionalFinding &&
    typeof instructionalLocation
      .pending
      .instructionalFinding === "object"
      ? structuredClone(
          instructionalLocation
            .pending
            .instructionalFinding
        )
      : null;

  const criteriaAssessment =
    existingInstructionalFinding
      ? {
          frameComponent:
            existingInstructionalFinding
              .frameComponent || null,

          componentEvidenceLevel:
            existingInstructionalFinding
              .componentEvidenceLevel || null,

          componentCriteriaStatus:
            existingInstructionalFinding
              .componentCriteriaStatus || null,

          diagnosis:
            existingInstructionalFinding
              .diagnosis || null,
        }
      : null;

  const relationalAssessment =
    existingInstructionalFinding
      ? {
          frameComponent:
            existingInstructionalFinding
              .frameComponent || null,

          relationshipStatus:
            existingInstructionalFinding
              .relationshipStatus || null,

          relationshipEvidence:
            existingInstructionalFinding
              .relationshipEvidence
              ? structuredClone(
                  existingInstructionalFinding
                    .relationshipEvidence
                )
              : null,
        }
      : null;

  // --------------------------------------------------
  // OBSERVATION-BASED INTERACTION ASSESSMENT
  //
  // This assessment records only what the governed
  // Observation Report directly established.
  //
  // It does not determine:
  // • genuine struggle;
  // • instructional situation;
  // • support level;
  // • teaching strategy;
  // • progression;
  // • instructional intent.
  //
  // Those conclusions belong to later deterministic
  // reasoning layers.
  // --------------------------------------------------

  const interactionAssessment = {
    observationSource:
      cleanText(
        observationReport?.source ||
        "notObserved"
      ),

    ambiguityPresent:
      observationReport
        ?.ambiguityPresent === true,

    observationCount:
      observations.length,

    observedCategories,

    observations,

    observableConditions: {
      uncertaintyExpressed:
        hasObservedCategory(
          "uncertaintyExpression"
        ),

      clarificationRequested:
        hasObservedCategory(
          "clarificationRequest"
        ),

      answerSeekingObserved:
        hasObservedCategory(
          "answerSeeking"
        ),

      frustrationExpressed:
        hasObservedCategory(
          "frustrationExpression"
        ),

      refusalObserved:
        hasObservedCategory(
          "refusal"
        ),

      offTaskShiftObserved:
        hasObservedCategory(
          "offTaskShift"
        ),

      assignmentReferenced:
        hasObservedCategory(
          "assignmentReference"
        ),

      framingRoutineReferenced:
        hasObservedCategory(
          "framingRoutineReference"
        ),

      priorCoachingAcknowledged:
        hasObservedCategory(
          "acknowledgesPriorCoaching"
        ),

      repeatedAttemptObserved:
        hasObservedCategory(
          "repeatedAttempt"
        ),
    },
  };

  return {
    criteriaAssessment,

    relationalAssessment,

    interactionAssessment,

    findings:
      existingInstructionalFinding
        ? [
            existingInstructionalFinding,
          ]
        : [],
  };
}

// ------------------------------------------------------
// INTERACTION INSTRUCTIONAL FINDING
// ------------------------------------------------------
//
// Deterministically interprets the governed Interaction
// Assessment within the student's current instructional
// location.
//
// Observation reports what the student said.
//
// Interaction Assessment organizes those observations.
//
// This function determines the limited instructional
// meaning that can be established from that evidence.
//
// It does not:
//
// • classify genuine struggle;
// • select an instructional contract;
// • determine support level;
// • choose a Teaching Move or Thinking Move;
// • change progression;
// • change pending state;
// • generate communication.
//
// Current status:
//
// Shadow mode only.
//
// ------------------------------------------------------

function buildInteractionInstructionalFinding(
  evidenceState,
  instructionalAssessment
) {
  const safeEvidenceState =
    evidenceState &&
    typeof evidenceState === "object"
      ? evidenceState
      : {};

  const safeAssessment =
    instructionalAssessment &&
    typeof instructionalAssessment === "object"
      ? instructionalAssessment
      : {};

  const interactionAssessment =
    safeAssessment
      ?.interactionAssessment &&
    typeof safeAssessment
      .interactionAssessment === "object"
      ? safeAssessment
          .interactionAssessment
      : {};

  const observableConditions =
    interactionAssessment
      ?.observableConditions &&
    typeof interactionAssessment
      .observableConditions === "object"
      ? interactionAssessment
          .observableConditions
      : {};

  const currentResponse =
    cleanText(
      safeEvidenceState
        ?.currentEvidence
        ?.response || ""
    );

  const rawStage =
    cleanText(
      safeEvidenceState
        ?.instructionalLocation
        ?.rawStage || ""
    );

  const frameComponent =
    getBaseStage(rawStage);

  const observations =
    Array.isArray(
      interactionAssessment?.observations
    )
      ? interactionAssessment.observations
      : [];

  const normalizedResponse =
    currentResponse
      .toLowerCase()
      .replace(/[’‘]/g, "'")
      .replace(/[.!?]+$/g, "")
      .trim();

  // An observation occupies the complete response only
  // when its exact evidence excerpt matches the student's
  // entire normalized interaction.
  //
  // This permits Kaw to distinguish:
  //
  // "idk"
  //
  // from:
  //
  // "I don't know, but I think the topic is photosynthesis."
  //
  // The first contains no component contribution.
  // The second may still contain student thinking that
  // requires component validation.
  const observationsCoverEntireResponse =
    Boolean(normalizedResponse) &&
    observations.some(
      (observation) => {
        const normalizedEvidence =
          cleanText(
            observation?.evidenceText
          )
            .toLowerCase()
            .replace(/[’‘]/g, "'")
            .replace(/[.!?]+$/g, "")
            .trim();

        return (
          normalizedEvidence &&
          normalizedEvidence ===
            normalizedResponse
        );
      }
    );

  const activeFrameComponents =
    new Set([
      "keyTopic",
      "isAbout",
      "mainIdeas",
      "details",
      "soWhat",
    ]);

  const componentCaptureActive =
    activeFrameComponents.has(
      frameComponent
    );

  let interactionSituation =
    "noSpecialInteractionCondition";

  if (
    observableConditions
      .frustrationExpressed === true
  ) {
    interactionSituation =
      "frustrationExpressed";
  } else if (
    observableConditions
      .refusalObserved === true
  ) {
    interactionSituation =
      "refusalObserved";
  } else if (
    observableConditions
      .offTaskShiftObserved === true
  ) {
    interactionSituation =
      "offTaskShiftObserved";
  } else if (
    observableConditions
      .answerSeekingObserved === true
  ) {
    interactionSituation =
      "answerSeekingObserved";
  } else if (
    observableConditions
      .clarificationRequested === true
  ) {
    interactionSituation =
      "clarificationRequested";
  } else if (
    observableConditions
      .uncertaintyExpressed === true
  ) {
    interactionSituation =
      "uncertaintyExpressed";
  } else if (
    observableConditions
      .repeatedAttemptObserved === true
  ) {
    interactionSituation =
      "repeatedAttemptObserved";
  }

  const interactionOnlyCategories =
    new Set([
      "uncertaintyExpression",
      "clarificationRequest",
      "answerSeeking",
      "frustrationExpression",
      "refusal",
      "offTaskShift",
    ]);

  const responseFunctionsOnlyAsInteraction =
    componentCaptureActive &&
    observationsCoverEntireResponse &&
    observations.some(
      (observation) =>
        interactionOnlyCategories.has(
          cleanText(
            observation?.category
          )
        )
    );

  const componentEvidenceFinding =
    responseFunctionsOnlyAsInteraction
      ? "noComponentEvidenceObserved"
      : "componentEvidenceNotDetermined";

  return {
    findingType:
      "interactionInstructionalFinding",

    source:
      "deterministicInterpretation",

    frameComponent:
      componentCaptureActive
        ? frameComponent
        : null,

    instructionalLocation: {
      rawStage:
        rawStage || null,

      pendingType:
        cleanText(
          safeEvidenceState
            ?.instructionalLocation
            ?.pendingType || ""
        ) || null,
    },

    interactionSituation,

    componentEvidenceFinding,

    responseFunctionsOnlyAsInteraction,

    observationSource:
      cleanText(
        interactionAssessment
          ?.observationSource ||
        "notObserved"
      ),

    observedCategories:
      Array.isArray(
        interactionAssessment
          ?.observedCategories
      )
        ? [
            ...interactionAssessment
              .observedCategories,
          ]
        : [],

    evidence: {
      currentResponse,

      observations:
        structuredClone(
          observations
        ),

      ambiguityPresent:
        interactionAssessment
          ?.ambiguityPresent === true,
    },

    governance: {
      genuineStruggleEstablished:
        false,

      instructionalContractSelected:
        false,

      supportLevelDetermined:
        false,

      progressionAuthority:
        false,

      shadowMode:
        true,
    },
  };
}

// ======================================================
// INSTRUCTIONAL SITUATION ENGINE
// ======================================================
//
// The Instructional Situation Engine is the deterministic
// bridge between instructional findings and instructional
// strategy.
//
// It answers exactly one question:
//
// What instructional situation exists right now?
//
// Inputs:
//
// • Evidence State
// • Interaction Instructional Finding
// • Component Instructional Finding
// • Relationship Finding
// • prior governed instructional history
//
// Output:
//
// • one mutually exclusive Instructional Situation
//
// The engine does not:
//
// • observe student language;
// • validate component evidence;
// • select an Instructional Contract;
// • determine a Teaching Move;
// • determine a Thinking Move;
// • determine support level;
// • generate communication;
// • change progression or pending state.
//
// Current migration status:
//
// Shadow mode.
//
// The engine produces and stores one governed situation,
// but it does not yet control the authoritative runtime.
//
// ======================================================

const INSTRUCTIONAL_SITUATIONS =
  Object.freeze({
    ASSIGNMENT_UNDERSTANDING_REQUIRED:
      "assignmentUnderstandingRequired",

    NO_COMPONENT_EVIDENCE:
      "noComponentEvidence",

    COMPONENT_EVIDENCE_REQUIRES_VALIDATION:
      "componentEvidenceRequiresValidation",

    COMPONENT_NEEDS_REVISION:
      "componentNeedsRevision",

    RELATIONSHIP_NEEDS_REPAIR:
      "relationshipNeedsRepair",

    CLARIFICATION_NEEDED:
      "clarificationNeeded",

    INTERACTION_REDIRECTION_NEEDED:
      "interactionRedirectionNeeded",

    GENUINE_STRUGGLE:
      "genuineStruggle",

    READY_TO_PROGRESS:
      "readyToProgress",
  });

function hasEstablishedAssignmentUnderstandingFromEvidence(
  evidenceState
) {
  const assignmentContext =
    evidenceState
      ?.accumulatedEvidence
      ?.assignmentContext;

  if (
    !assignmentContext ||
    typeof assignmentContext !== "object"
  ) {
    return false;
  }

  return (
    assignmentContext.valid === true &&

    assignmentContext
      .assignmentContextStatus ===
      "established" &&

    assignmentContext
      .assignmentDemandStatus ===
      "established" &&

    assignmentContext
      .summaryReadinessStatus ===
      "ready"
  );
}

function getInstructionalSituationEvidenceHistory(
  evidenceState
) {
  const instructionalLocation =
    evidenceState
      ?.instructionalLocation &&
    typeof evidenceState
      .instructionalLocation ===
        "object"
      ? evidenceState
          .instructionalLocation
      : {};

  const pending =
    instructionalLocation?.pending &&
    typeof instructionalLocation
      .pending === "object"
      ? instructionalLocation.pending
      : null;

  const rawStage =
    cleanText(
      instructionalLocation?.rawStage || ""
    );

  const currentFrameComponent =
    cleanText(
      getBaseStage(rawStage) || ""
    );

  const priorFinding =
    pending?.instructionalFinding &&
    typeof pending
      .instructionalFinding ===
        "object"
      ? pending.instructionalFinding
      : null;

  const priorFrameComponent =
    cleanText(
      priorFinding?.frameComponent || ""
    );

  const sameInstructionalComponent =
    Boolean(
      currentFrameComponent &&
      priorFrameComponent &&
      currentFrameComponent ===
        priorFrameComponent
    );

  const governedContractPresent =
    Boolean(
      pending?.instructionalContract &&
      typeof pending
        .instructionalContract ===
          "object"
    );

  const governedActivationPresent =
    Boolean(
      pending?.instructionalActivation &&
      typeof pending
        .instructionalActivation ===
          "object"
    );

  // Prior governed support is established by Kaw 2.5's
  // actual instructional artifacts—not by a legacy
  // pending-state identity.
  const priorSupportActive =
    Boolean(
      priorFinding &&
      governedContractPresent &&
      governedActivationPresent &&
      sameInstructionalComponent
    );

  const priorDiagnosis =
    cleanText(
      priorFinding?.diagnosis || ""
    );

  const priorNoEvidence =
    priorDiagnosis ===
      "emptyResponse" ||
    priorDiagnosis ===
      "noComponentEvidence";

  return {
    priorSupportActive,

    priorDiagnosis:
      priorDiagnosis || null,

    priorNoEvidence,

    priorFrameComponent:
      priorFrameComponent || null,

    currentFrameComponent:
      currentFrameComponent || null,

    sameInstructionalComponent,

    governedContractPresent,

    governedActivationPresent,

    priorFinding:
      priorFinding
        ? structuredClone(
            priorFinding
          )
        : null,
  };
}

function buildInstructionalSituation({
  evidenceState = null,
  instructionalAssessment = null,
  componentFinding = null,
  relationshipFinding = null,
} = {}) {
  const safeEvidenceState =
    evidenceState &&
    typeof evidenceState === "object"
      ? evidenceState
      : {};

  const safeAssessment =
    instructionalAssessment &&
    typeof instructionalAssessment ===
      "object"
      ? instructionalAssessment
      : {};

  const interactionFinding =
    safeAssessment
      ?.interactionInstructionalFinding &&
    typeof safeAssessment
      .interactionInstructionalFinding ===
        "object"
      ? safeAssessment
          .interactionInstructionalFinding
      : null;

  const safeComponentFinding =
    componentFinding &&
    typeof componentFinding === "object"
      ? componentFinding
      : null;

  const safeRelationshipFinding =
    relationshipFinding &&
    typeof relationshipFinding === "object"
      ? relationshipFinding
      : safeComponentFinding;

  const currentResponse =
    cleanText(
      safeEvidenceState
        ?.currentEvidence
        ?.response || ""
    );

  const rawStage =
    cleanText(
      safeEvidenceState
        ?.instructionalLocation
        ?.rawStage || ""
    );

   const frameComponent =
    cleanText(
      safeComponentFinding
        ?.frameComponent ||
      interactionFinding
        ?.frameComponent ||
      getBaseStage(rawStage) ||
      ""
    );

  const assignmentUnderstandingEstablished =
    hasEstablishedAssignmentUnderstandingFromEvidence(
      safeEvidenceState
    );

  const interactionSituation =
    cleanText(
      interactionFinding
        ?.interactionSituation ||
      "noSpecialInteractionCondition"
    );

    const noCurrentComponentEvidence =
    interactionFinding
      ?.componentEvidenceFinding ===
      "noComponentEvidenceObserved" ||

    safeComponentFinding
      ?.componentEvidenceLevel ===
      "none" ||

    safeComponentFinding
      ?.diagnosis ===
      "emptyResponse" ||

    safeComponentFinding
      ?.diagnosis ===
      "noComponentEvidence";

  const responseFunctionsOnlyAsInteraction =
    interactionFinding
      ?.responseFunctionsOnlyAsInteraction ===
      true;

  const evidenceHistory =
    getInstructionalSituationEvidenceHistory(
      safeEvidenceState
    );

  const componentEvidenceLevel =
    cleanText(
      safeComponentFinding
        ?.componentEvidenceLevel || ""
    );

  const componentCriteriaStatus =
    cleanText(
      safeComponentFinding
        ?.componentCriteriaStatus || ""
    );

  const relationshipStatus =
    cleanText(
      safeRelationshipFinding
        ?.relationshipStatus ||
      safeComponentFinding
        ?.relationshipStatus ||
      ""
    );

  const diagnosis =
    cleanText(
      safeComponentFinding
        ?.diagnosis || ""
    );

  const validationCompleted =
    Boolean(
      safeComponentFinding &&
      (
        componentEvidenceLevel ||
        componentCriteriaStatus ||
        relationshipStatus ||
        diagnosis
      )
    );

  const componentCriteriaSatisfied =
    componentCriteriaStatus ===
    "satisfied";

  const relationshipEstablished =
    relationshipStatus ===
    "established";

  const relationshipRequiresRepair =
    relationshipStatus ===
      "notEstablished" ||
    relationshipStatus ===
      "incomplete";

  const componentRequiresRevision =
    validationCompleted &&
    !componentCriteriaSatisfied &&
    !relationshipRequiresRepair;

  const persistentNoEvidenceAfterSupport =
    noCurrentComponentEvidence &&
    evidenceHistory
      .priorSupportActive === true &&
    evidenceHistory
      .priorNoEvidence === true;

  let instructionalSituation =
    INSTRUCTIONAL_SITUATIONS
      .COMPONENT_EVIDENCE_REQUIRES_VALIDATION;

  let situationReason =
    "The current response contains possible component evidence that requires governed component validation.";

  let evidenceBasis = [
    "currentResponseAvailable",
  ];

  // --------------------------------------------------
  // PRIORITY 1 — ASSIGNMENT UNDERSTANDING
  //
  // Frame instruction may not proceed until sufficient
  // Assignment Understanding has been established.
  // --------------------------------------------------

  if (!assignmentUnderstandingEstablished) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .ASSIGNMENT_UNDERSTANDING_REQUIRED;

    situationReason =
      "Sufficient shared Assignment Understanding has not yet been established.";

    evidenceBasis = [
      "assignmentUnderstandingNotEstablished",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 2 — GENUINE STRUGGLE
  //
  // Genuine struggle requires persistence.
  //
  // One uncertainty response, clarification request, or
  // answer-seeking interaction is not sufficient.
  //
  // The student must remain unable to provide component
  // evidence after governed support has already occurred.
  // --------------------------------------------------

  else if (
    persistentNoEvidenceAfterSupport
  ) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .GENUINE_STRUGGLE;

    situationReason =
      "The student again provided no component evidence after a prior governed support intervention at the same instructional location.";

    evidenceBasis = [
      "noCurrentComponentEvidence",
      "priorSupportActive",
      "priorNoComponentEvidence",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 3 — CLARIFICATION REQUEST
  //
  // Clarification is a legitimate instructional need.
  // It is not automatically struggle.
  // --------------------------------------------------

  else if (
    interactionSituation ===
      "clarificationRequested" &&
    responseFunctionsOnlyAsInteraction
  ) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .CLARIFICATION_NEEDED;

    situationReason =
      "The student's current interaction functions as a clarification request rather than component evidence.";

    evidenceBasis = [
      "clarificationRequested",
      "responseFunctionsOnlyAsInteraction",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 4 — INTERACTION REDIRECTION
  //
  // Answer-seeking, refusal, or off-task interaction
  // requires Kaw to preserve the instructional objective
  // and return the student to the current thinking step.
  // --------------------------------------------------

  else if (
    responseFunctionsOnlyAsInteraction &&
    (
      interactionSituation ===
        "answerSeekingObserved" ||

      interactionSituation ===
        "refusalObserved" ||

      interactionSituation ===
        "offTaskShiftObserved"
    )
  ) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .INTERACTION_REDIRECTION_NEEDED;

    situationReason =
      "The current response functions as an interaction move rather than component evidence and requires return to the active thinking step.";

    evidenceBasis = [
      interactionSituation,
      "responseFunctionsOnlyAsInteraction",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 5 — NO COMPONENT EVIDENCE
  //
  // A first no-evidence response does not establish
  // genuine struggle.
  // --------------------------------------------------

  else if (noCurrentComponentEvidence) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .NO_COMPONENT_EVIDENCE;

    situationReason =
      "The current interaction contains no observable contribution to the active Frame component.";

    evidenceBasis = [
      "noCurrentComponentEvidence",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 6 — READY TO PROGRESS
  //
  // Progression requires completed component validation
  // and an established instructional relationship.
  // --------------------------------------------------

  else if (
    validationCompleted &&
    componentCriteriaSatisfied &&
    relationshipEstablished
  ) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .READY_TO_PROGRESS;

    situationReason =
      "The current component evidence satisfies its criteria and its required instructional relationship is established.";

    evidenceBasis = [
      "componentCriteriaSatisfied",
      "relationshipEstablished",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 7 — RELATIONSHIP REPAIR
  //
  // Component content may exist while its required
  // relationship to the active Frame remains incomplete
  // or unestablished.
  // --------------------------------------------------

  else if (
    validationCompleted &&
    relationshipRequiresRepair
  ) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .RELATIONSHIP_NEEDS_REPAIR;

    situationReason =
      "The student supplied component evidence, but the required relationship to accepted Frame evidence is incomplete or not established.";

    evidenceBasis = [
      `relationshipStatus:${relationshipStatus}`,
      diagnosis
        ? `diagnosis:${diagnosis}`
        : "relationshipRepairRequired",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 8 — COMPONENT REVISION
  //
  // Evidence exists, but component criteria have not yet
  // been fully satisfied.
  // --------------------------------------------------

  else if (componentRequiresRevision) {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .COMPONENT_NEEDS_REVISION;

    situationReason =
      "The current evidence has been validated but does not yet satisfy the active component criteria.";

    evidenceBasis = [
      `componentCriteriaStatus:${componentCriteriaStatus}`,
      diagnosis
        ? `diagnosis:${diagnosis}`
        : "componentRevisionRequired",
    ];
  }

  // --------------------------------------------------
  // PRIORITY 9 — VALIDATION REQUIRED
  //
  // The response contains possible student thinking, but
  // no current Component Finding has yet been supplied to
  // the engine.
  // --------------------------------------------------

  else {
    instructionalSituation =
      INSTRUCTIONAL_SITUATIONS
        .COMPONENT_EVIDENCE_REQUIRES_VALIDATION;

    situationReason =
      "The current response may contain component evidence, but governed component validation has not yet produced a current finding.";

    evidenceBasis = [
      currentResponse
        ? "currentResponseAvailable"
        : "currentResponseEmpty",

      validationCompleted
        ? "validationIncomplete"
        : "validationNotYetProvided",
    ];
  }

  return {
    artifactType:
      "instructionalSituation",

    version:
      "1.0",

    source:
      "deterministicInstructionalSituationEngine",

    instructionalSituation,

    situationReason,

    evidenceBasis,

    frameComponent:
      frameComponent || null,

    instructionalLocation: {
      rawStage:
        rawStage || null,

      pendingType:
        cleanText(
          safeEvidenceState
            ?.instructionalLocation
            ?.pendingType || ""
        ) || null,
    },

    assignmentUnderstandingEstablished,

    inputs: {
      interactionFinding:
        interactionFinding
          ? structuredClone(
              interactionFinding
            )
          : null,

      componentFinding:
        safeComponentFinding
          ? structuredClone(
              safeComponentFinding
            )
          : null,

      relationshipFinding:
        safeRelationshipFinding
          ? structuredClone(
              safeRelationshipFinding
            )
          : null,

      evidenceHistory:
        structuredClone(
          evidenceHistory
        ),
    },

    governance: {
      mutuallyExclusiveSituation:
        true,

      genuineStruggleRequiresPersistence:
        true,

      selectsInstructionalContract:
        false,

      controlsProgression:
        false,

      controlsPendingState:
        false,

      controlsCommunication:
        false,

      shadowMode:
        true,
    },
  };
}

// ------------------------------------------------------
// COMPONENT INSTRUCTIONAL FINDING
// ------------------------------------------------------
//
// Converts completed deterministic validation into one
// explicit instructional finding.
//
// Validation determines whether observable student
// evidence fulfills component expectations.
//
// Instructional Assessment organizes that conclusion for
// downstream strategy selection.
//
// This function does not:
//
// • validate the response;
// • select an instructional contract;
// • choose a Teaching Move or Thinking Move;
// • change runtime progression;
// • generate communication.
//
// ------------------------------------------------------

function buildComponentInstructionalFinding({
  frameComponent = "",
  validation = null,
  evidence = {},
} = {}) {
  const safeValidation =
    validation &&
    typeof validation === "object"
      ? validation
      : {};

  const safeEvidence =
    evidence &&
    typeof evidence === "object"
      ? evidence
      : {};

  return {
    frameComponent:
      cleanText(frameComponent),

    componentEvidenceLevel:
      safeValidation
        .componentEvidenceLevel || null,

    componentCriteriaStatus:
      safeValidation
        .componentCriteriaStatus || null,

    relationshipStatus:
      safeValidation
        .relationshipStatus || null,

    diagnosis:
      safeValidation
        .diagnosis || null,

    relationshipEvidence:
      safeValidation
        .relationshipEvidence || null,

    evidence:
      structuredClone(safeEvidence),
  };
}

// ------------------------------------------------------
// COMPONENT FINDING → INSTRUCTIONAL SITUATION REFRESH
// ------------------------------------------------------
//
// Re-runs the governed Instructional Situation Engine after
// current component validation produces a completed
// Component Instructional Finding.
//
// The beginning-of-cycle Instructional Situation cannot
// include the current Component Finding because component
// validation occurs later inside the active runtime branch.
//
// This refresh:
//
// • rebuilds Evidence State for the validated response;
// • preserves the governed Interaction Instructional
//   Finding;
// • attaches the current Component Instructional Finding;
// • rebuilds the governed Instructional Situation;
// • selects the corresponding Instructional Contract for
//   migrated Frame components;
// • stores the refreshed governed artifacts.
//
// This function does not:
//
// • determine support level;
// • directly change pending location;
// • save or reject student work;
// • directly control runtime progression;
// • generate student-facing communication.
//
// ------------------------------------------------------
function refreshInstructionalSituationWithComponentFinding({
  state,
  currentResponse = "",
  componentFinding = null,
} = {}) {
  if (
    !state ||
    typeof state !== "object" ||
    !componentFinding ||
    typeof componentFinding !== "object"
  ) {
    return null;
  }

  const observationReport =
    state?.observationReport &&
    typeof state.observationReport === "object"
      ? state.observationReport
      : buildEmptyObservationReport(
          currentResponse,
          "notProvided"
        );

  const evidenceState =
    buildEvidenceState(
      state,
      currentResponse,
      observationReport
    );

  const instructionalAssessment =
    state?.instructionalAssessment &&
    typeof state.instructionalAssessment === "object"
      ? structuredClone(
          state.instructionalAssessment
        )
      : buildInstructionalAssessment(
          evidenceState
        );

  instructionalAssessment
    .componentInstructionalFinding =
      structuredClone(
        componentFinding
      );

  const instructionalSituation =
    buildInstructionalSituation({
      evidenceState,

      instructionalAssessment,

      componentFinding,

      relationshipFinding:
        componentFinding,
    });

  instructionalAssessment
    .instructionalSituation =
      structuredClone(
        instructionalSituation
      );

  state.instructionalAssessment =
    structuredClone(
      instructionalAssessment
    );

  state.componentInstructionalFinding =
    structuredClone(
      componentFinding
    );

    state.instructionalSituation =
    structuredClone(
      instructionalSituation
    );

    const instructionalContractSelection =
    buildInstructionalContractSelection(
      instructionalSituation
    );

  instructionalAssessment
    .instructionalContractSelection =
      structuredClone(
        instructionalContractSelection
      );

  state.instructionalAssessment =
    structuredClone(
      instructionalAssessment
    );

  state.instructionalContractSelection =
    structuredClone(
      instructionalContractSelection
    );

  return instructionalSituation;
}

// ------------------------------------------------------
// GOVERNED PROGRESSION AUTHORIZATION
// ------------------------------------------------------
//
// Determines whether the governed instructional pipeline
// has authorized progression for the active Frame
// component.
//
// Authorization requires:
//
// • Instructional Situation = readyToProgress;
// • the situation belongs to the expected Frame component;
// • one predetermined Instructional Contract was selected;
// • the selected contract is the expected ready-to-progress
//   contract;
// • the contract explicitly authorizes continuation through
//   the current runtime progression pathway.
//
// This helper does not:
//
// • save or modify student work;
// • change pending state;
// • select a different Instructional Contract;
// • determine the Instructional Situation;
// • execute progression.
//
// It creates the governed authorization that the Runtime
// Progression Layer will later be required to obey.
//
// ------------------------------------------------------

function buildProgressionAuthorization(
  state,
  {
    frameComponent = "",
    expectedContractId = "",
  } = {}
) {
  const instructionalSituation =
    state?.instructionalSituation &&
    typeof state.instructionalSituation ===
      "object"
      ? state.instructionalSituation
      : null;

  const contractSelection =
    state?.instructionalContractSelection &&
    typeof state.instructionalContractSelection ===
      "object"
      ? state.instructionalContractSelection
      : null;

  const selectedContract =
    contractSelection?.selectedContract &&
    typeof contractSelection.selectedContract ===
      "object"
      ? contractSelection.selectedContract
      : null;

  const situationReady =
    instructionalSituation
      ?.instructionalSituation ===
    INSTRUCTIONAL_SITUATIONS
      .READY_TO_PROGRESS;

  const componentMatches =
    cleanText(
      instructionalSituation?.frameComponent ||
      ""
    ) === cleanText(frameComponent);

  const contractSelected =
    contractSelection?.selectionStatus ===
      "contractSelected" &&
    selectedContract !== null;

  const contractMatches =
    cleanText(
      selectedContract?.contractId || ""
    ) === cleanText(expectedContractId);

  const continuationAuthorized =
    selectedContract
      ?.progressionBehavior
      ?.type ===
    "continueExistingRuntimeProgression";

  const authorized =
    situationReady &&
    componentMatches &&
    contractSelected &&
    contractMatches &&
    continuationAuthorized;

  return {
    artifactType:
      "progressionAuthorization",

    version:
      "1.0",

    source:
      "deterministicProgressionAuthorization",

    authorized,

    frameComponent:
      cleanText(frameComponent) || null,

    expectedContractId:
      cleanText(expectedContractId) || null,

    instructionalSituation:
      instructionalSituation
        ?.instructionalSituation || null,

    selectedContractId:
      selectedContract?.contractId || null,

    progressionBehavior:
      selectedContract?.progressionBehavior
        ? structuredClone(
            selectedContract.progressionBehavior
          )
        : null,

    evidence: {
      situationReady,

      componentMatches,

      contractSelected,

      contractMatches,

      continuationAuthorized,
    },

    governance: {
      controlsProgression:
        false,

      controlsPendingState:
        false,

      authorizationEstablished:
        authorized,

      migrationStage:
        "authorizationOnly",
    },
  };
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
        "Express the predetermined Thinking Move as one concise, natural, student-friendly question. Sound like a supportive teacher speaking directly to a student, not a rubric, system message, or instructional manual. Preserve Framing Routine vocabulary when it is relevant."
    },

    acknowledgeThenQuestion: {
      instruction:
        "Briefly acknowledge only the authentic progress established by the Instructional Finding. When helpful, reference the student's own accepted or observable language so the response shows that Kaw is listening to their thinking. Then express the predetermined Thinking Move as one concise, natural, student-friendly question. Sound warm and encouraging without exaggerating success, evaluating beyond the established finding, or generating student work."
    },

    briefReassuranceThenQuestion: {
      instruction:
        "Use one brief, natural, supportive lead-in that helps the student stay engaged without implying success, failure, effort, emotion, or understanding that has not been established. Then express the predetermined Thinking Move as one concise, student-friendly question. Sound like a calm, encouraging teacher rather than a system message, rubric, or scripted tutor."
    },
  },
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
      
      mayReferenceStudentWorkVerbatim: true,
    },

    prohibitions: {
      mayGenerateStudentWork: false,
    
      mayCompleteStudentWork: false,
    
      mayParaphraseStudentWork: false,
    
      mayStrengthenStudentWork: false,
    
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

 const relationshipClaimPatterns = [
  /(?:^|[.!]\s+)this supports\b/i,
  /(?:^|[.!]\s+)that supports\b/i,
  /(?:^|[.!]\s+)this proves\b/i,
  /(?:^|[.!]\s+)that proves\b/i,
  /(?:^|[.!]\s+)this does not support\b/i,
  /(?:^|[.!]\s+)that does not support\b/i,
  /(?:^|[.!]\s+)this doesn't support\b/i,
  /(?:^|[.!]\s+)that doesn't support\b/i,
  /(?:^|[.!]\s+)this fails to support\b/i,
  /(?:^|[.!]\s+)that fails to support\b/i,
];

const relationshipStatus =
  communicationLicense?.relationshipStatus ||
  "";

if (
  relationshipStatus === "undetermined" &&
  relationshipClaimPatterns.some(
    (pattern) => pattern.test(text)
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

function executeInstructionalContract(
  contract,
  state
) {
  if (!contract) return null;

  switch (contract.contractId) {
    case "IA-NCE-001":
    case "IA-CNR-001":
    case "IA-RNR-001":
    case "IA-GS-001":
      return executeIsAboutInstructionalContract(
        contract,
        state
      );

    case "MI-NCE-001":
    case "MI-CNR-001":
    case "MI-RNR-001":
    case "MI-GS-001":
      return executeMainIdeaInstructionalContract(
        contract,
        state
  );

    case "ED-NCE-001":
    case "ED-CNR-001":
    case "ED-RNR-001":
    case "ED-GS-001":
      return executeEssentialDetailInstructionalContract(
        contract,
        state
  );

    case "SW-NCE-001":
    case "SW-CNR-001":
    case "SW-RNR-001":
    case "SW-GS-001":
      return executeSoWhatInstructionalContract(
        contract,
        state
      );

    default:
      return null;
  }
}

function selectIsAboutGenuineStruggleThinkingMove(
  supportLevel
) {
  if (supportLevel >= 3) {
    return (
      "Reconnect the student to the accepted Key Topic. " +
      "Ask what one person who knows nothing about the topic " +
      "should understand about it. Do not suggest, begin, or " +
      "supply the student's Is About statement."
    );
  }

  if (supportLevel === 2) {
    return (
      "Remind the student that Is About explains the whole " +
      "Key Topic in their own understandable words. Ask them " +
      "to describe the big picture rather than give one small " +
      "detail. Do not suggest or supply the student's answer."
    );
  }

  return (
    "Reconnect the student to the accepted Key Topic and ask " +
    "them to explain what the whole topic is about in their " +
    "own understandable words. Do not suggest or supply the " +
    "student's Is About statement."
  );
}

function executeIsAboutInstructionalContract(
  contract,
  state
) {
  const instructionalFinding =
    state?.pending
      ?.instructionalFinding ||

    state
      ?.componentInstructionalFinding ||

    null;

  const requestedSupportLevel =
    Number(
      state?.pending?.supportLevel || 1
    );

  const supportLevel =
    Number.isFinite(
      requestedSupportLevel
    )
      ? Math.max(
          1,
          Math.min(
            requestedSupportLevel,
            3
          )
        )
      : 1;

  const thinkingMove =
    contract.contractId ===
      "IA-GS-001"
      ? selectIsAboutGenuineStruggleThinkingMove(
          supportLevel
        )
      : contract.thinkingMove;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    supportLevel:
      contract.contractId ===
        "IA-GS-001"
        ? supportLevel
        : null,

    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      keyTopic:
        state?.frame
          ?.keyTopic || "",

      isAbout:
        state?.frame
          ?.isAbout || "",

      currentMainIdea:
        "",

      existingDetails:
        [],
    },
  };
}

// ------------------------------------------------------
// ESSENTIAL DETAIL INSTRUCTIONAL CONTRACT EXECUTION
// ------------------------------------------------------
//
// Executes the already-selected authoritative Essential
// Detail contract.
//
// The Instructional Situation Engine and Instructional
// Contract Selector have already determined the contract.
//
// This executor does not select a different Teaching Move,
// Thinking Move, or communication pattern by diagnosis.
//
// ------------------------------------------------------

function executeEssentialDetailInstructionalContract(
  contract,
  state
) {
  const ideas =
    getIdeaList(state)
      .filter(Boolean);

  const pending =
    state?.pending &&
    typeof state.pending === "object"
      ? state.pending
      : null;

  const currentMainIdea =
    Number.isInteger(pending?.index)
      ? ideas[pending.index] || ""
      : "";

  const instructionalFinding =
    state?.pending
      ?.instructionalFinding ||

    state
      ?.componentInstructionalFinding ||

    null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      keyTopic:
        state?.frame
          ?.keyTopic || "",

      isAbout:
        state?.frame
          ?.isAbout || "",

      mainIdeas:
        ideas,

      currentMainIdea,

      existingDetails:
        Number.isInteger(pending?.index) &&
        Array.isArray(
          state?.frame
            ?.details?.[pending.index]
        )
          ? state.frame.details[
              pending.index
            ].filter(Boolean)
          : [],
    },
  };
}

// ------------------------------------------------------
// MAIN IDEA INSTRUCTIONAL CONTRACT EXECUTION
// ------------------------------------------------------
//
// Executes the already-selected authoritative Main Idea
// contract.
//
// The Instructional Situation Engine and Instructional
// Contract Selector have already determined the contract.
//
// This executor does not select a different Teaching Move,
// Thinking Move, or communication pattern by diagnosis.
//
// ------------------------------------------------------

function executeMainIdeaInstructionalContract(
  contract,
  state
) {
  const instructionalFinding =
    state?.pending
      ?.instructionalFinding ||

    state
      ?.componentInstructionalFinding ||

    null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      keyTopic:
        state?.frame
          ?.keyTopic || "",

      isAbout:
        state?.frame
          ?.isAbout || "",

      mainIdeas:
        getIdeaList(state)
          .filter(Boolean),

      currentMainIdea:
        "",

      existingDetails:
        [],
    },
  };
}

// ------------------------------------------------------
// SO WHAT INSTRUCTIONAL CONTRACT EXECUTION
// ------------------------------------------------------
//
// Executes the already-selected authoritative So What
// instructional contract.
//
// The Instructional Situation Engine and Instructional
// Contract Selector have already determined the contract.
//
// This executor does not select a different Teaching Move,
// Thinking Move, or communication pattern by diagnosis.
//
// ------------------------------------------------------

function executeSoWhatInstructionalContract(
  contract,
  state
) {
  const instructionalFinding =
    state?.pending
      ?.instructionalFinding ||

    state
      ?.componentInstructionalFinding ||

    null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      keyTopic:
        state?.frame
          ?.keyTopic || "",

      isAbout:
        state?.frame
          ?.isAbout || "",

      mainIdeas:
        getIdeaList(state)
          .filter(Boolean),

      details:
        Array.isArray(
          state?.frame
            ?.details
        )
          ? state.frame.details.map(
              (bucket) =>
                Array.isArray(bucket)
                  ? bucket.filter(Boolean)
                  : []
            )
          : [],

      currentSoWhat:
        state?.frame
          ?.soWhat || "",

      currentMainIdea:
        "",

      existingDetails:
        [],
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

      mainIdeas:
        Array.isArray(
          execution?.context?.mainIdeas
  )
    ? execution.context.mainIdeas
    : [],

    details:
      Array.isArray(
        execution?.context?.details
      )
        ? execution.context.details
        : [],

    currentSoWhat:
      execution?.context?.currentSoWhat || "",
    
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

  const mainIdeas =
    Array.isArray(
      payload?.context?.mainIdeas
    )
      ? payload.context.mainIdeas
      : [];

  const details =
    Array.isArray(
      payload?.context?.details
    )
      ? payload.context.details
      : [];

  const currentSoWhat =
    payload?.context?.currentSoWhat || "";
  
  // Deterministic instructional conclusions established
  // before AI contextualization.
  //
  // AI may express these conclusions but may not revise,
  // reinterpret, strengthen, weaken, or replace them.
  const instructionalFinding =
    payload?.instructionalFinding || null;
  
  const system = `You are the language contextualization layer for Kaw Companion, a structured instructional companion that supports students using the KU Framing Routine.

The instructional decision and instructional findings have already been established by a deterministic Instructional Reasoning Engine.

You do not decide what to teach, whether student work is successful, what support the student needs, or what should happen next.

Your only job is to express the predetermined Thinking Move in natural, student-facing teacher language while preserving the exact instructional decision.

TEACHER VOICE

Kaw should sound like a warm, attentive, encouraging teacher speaking directly with a student.

Kaw's language should be:
- natural and conversational;
- concise and easy for a student to understand;
- supportive without sounding artificial, overly enthusiastic, or scripted;
- responsive to the student's actual thinking and the work already present in the Frame;
- appropriate for middle and high school students;
- instructionally purposeful rather than generic;
- consistent with KU Framing Routine vocabulary when that vocabulary is relevant.

Kaw should not sound like:
- a rubric;
- a diagnostic report;
- a software or system message;
- an instructional manual;
- a generic chatbot;
- a scripted tutor repeating the same phrases.

STUDENT THINKING

The student's language and ideas belong to the student.

When the Communication License permits reference to existing student work:
- use the student's actual accepted or observable language when it helps make the response feel connected to their thinking;
- preserve the student's wording rather than rewriting it into a stronger, clearer, or more sophisticated version;
- reference only the amount of student work needed for the current Thinking Move;
- never turn a reference to student work into a model answer, completion, or replacement.

INSTRUCTIONAL RESTRAINT

Advance only the one Thinking Move that has already been selected.

Do not add a second teaching move, extra task, additional question, example answer, hint, explanation, or new instructional demand unless the predetermined Thinking Move explicitly requires it.

The goal is not to say everything that might be helpful. The goal is to say the right thing for this instructional moment.

COMMUNICATION GOVERNANCE

You must follow these rules:
- The Communication License is authoritative and binding.
- Perform only actions explicitly permitted by the Communication License.
- Never perform an action prohibited by the Communication License, even if it might seem helpful.
- Do not rewrite, improve, complete, or generate student work.
- Do not change the Instructional Goal, Teaching Move, or Thinking Move.
- Do not reinterpret, expand, weaken, strengthen, or replace the established Instructional Finding.
- Do not infer student intent, understanding, confusion, emotion, effort, motivation, or meaning.
- Do not make claims about success, progress, correctness, relationships, or quality unless the established Instructional Finding and Communication License permit that claim.
- When relationship status is undetermined, preserve that uncertainty rather than resolving it yourself.
- Preserve student ownership at all times.
- Follow the Approved Communication Instruction exactly.
- Ask exactly one concise question.
- Include a brief student-facing lead-in only when the Approved Communication Instruction permits or requires it.
- Any acknowledgement or reassurance must stay within the evidence established by the Instructional Finding.
- Use Framing Routine terms such as Key Topic, Is About, Main Idea, Essential Detail, and So What accurately and naturally when relevant.
- Do not expose internal terms such as Instructional Finding, Instructional Situation, Teaching Move, Thinking Move, Communication License, contract, diagnosis, evidence state, or support level to the student.
- Before responding, silently verify that the response stays within every permission and prohibition in the Communication License.
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

Completed Main Ideas:
${
  mainIdeas.length
    ? mainIdeas.join(" | ")
    : "(none available)"
}

Completed Essential Details:
${
  details.length
    ? JSON.stringify(
        details,
        null,
        2
      )
    : "(none available)"
}

Current Accepted So What:
${currentSoWhat || "(none yet)"}

Express the predetermined Thinking Move as one natural, assignment-specific student-facing response.

Follow the Approved Communication Instruction and Communication License exactly.

Use the student's existing accepted or observable language when the license permits it and when doing so helps connect the response to the student's thinking.

Use only instructional conclusions explicitly established by the Instructional Finding.

Do not introduce unsupported praise, success claims, relationship claims, diagnoses, assumptions, interpretations, examples, hints, or student work.

Do not expose internal governance or architecture language to the student.

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

  activation.communicationDebug = {
    rawResponse:
      cleanText(response),

    validation:
      structuredClone(
        communicationValidation
      ),
  };

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

// ============================================================
// INSTRUCTIONAL REASONING LAYER (SSOT)
// ============================================================
//
// Purpose
// -------
// Performs deterministic instructional reasoning for the
// Framing Routine.
//
// Responsibilities
// ----------------
// • Analyze student evidence.
// • Classify instructional situations.
// • Select instructional contracts.
// • Govern instructional progression.
// • Protect accepted student work.
// • Prepare instructional context for the
//   Communication Layer.
//
// This layer owns instructional decisions.
// The Communication Layer owns expression.
//

// ------------------------------------------------------
// FORMATIVE ASSESSMENT
// Gathers evidence about student understanding.
// ------------------------------------------------------

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
    
  Student's proposed Is About response:
  "${studentResponse}"
    
  Interpret the student's response only as the completion of this Frame sentence:
    
  "${acceptedKeyTopic} is about ${studentResponse}"
    
  Does the completed Frame sentence express what the whole Key Topic is about?
    
  A response may validly use pronouns such as he, she, they, or it when the pronoun clearly refers to the accepted Key Topic.`;

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
  
  console.log(
    "IS ABOUT SEMANTIC EVIDENCE:",
    parsed
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
  // --------------------------------------------------
  // STEP 1 — DETERMINISTIC VALIDATION
  //
  // Observable instructional evidence is evaluated
  // before semantic evidence may be requested.
  // --------------------------------------------------

  const deterministicResult =
    validateIsAboutResponse(
      response,
      keyTopic
    );

  console.log(
    "IS ABOUT VALIDATION:",
    deterministicResult
  );

  // --------------------------------------------------
  // STEP 2 — SEMANTIC EVIDENCE GATE
  //
  // Semantic evidence is permitted only when the
  // deterministic analyzer explicitly identifies a
  // semantic inference gap.
  // --------------------------------------------------

  const requiresSemanticEvidence =
    deterministicResult
      ?.relationshipEvidence
      ?.requiresSemanticInference ===
    true;

  // --------------------------------------------------
  // STEP 3 — DETERMINISTIC FINAL RESULT
  //
  // When no semantic inference is required, the
  // deterministic result remains authoritative.
  // --------------------------------------------------

  if (!requiresSemanticEvidence) {
    return {
      ...deterministicResult,

      validationSource:
        "deterministic",
    };
  }

  // --------------------------------------------------
  // STEP 4 — BOUNDED SEMANTIC EVIDENCE
  //
  // AI supplies semantic evidence only.
  // It does not validate, save, or advance student work.
  // --------------------------------------------------

  const semanticEvidence =
    await getIsAboutSemanticEvidence(
      response,
      keyTopic
    );

  // --------------------------------------------------
  // STEP 5 — JAVASCRIPT GOVERNANCE DECISION
  //
  // JavaScript applies the instructional criteria to the
  // bounded semantic evidence and retains final authority.
  // --------------------------------------------------

  const relationshipEstablished =
    semanticEvidence
      .semanticEquivalent === true &&

    semanticEvidence
      .confidence >= 0.9;

  // --------------------------------------------------
  // STEP 6 — GOVERNED ACCEPTANCE
  // --------------------------------------------------

  if (relationshipEstablished) {
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
        ...deterministicResult
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

  // --------------------------------------------------
  // STEP 7 — GOVERNED NON-ACCEPTANCE
  //
  // Preserve the deterministic instructional conclusion
  // while attaching the bounded semantic evidence.
  // --------------------------------------------------

  return {
    ...deterministicResult,

    relationshipEvidence: {
      ...deterministicResult
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
// RESTATES IS ABOUT WITHOUT NEW ORGANIZATION
//
// A Main Idea must add one organizing idea beyond the
// accepted whole-topic Is About statement.
//
// When every meaningful content token in the proposed
// Main Idea already appears in the accepted Is About,
// the response has not yet established a distinct
// organizing contribution.
//
// This check uses observable lexical containment only.
// It does not infer semantic equivalence.
// --------------------------------------------------

const mainIdeaContentTokens =
  getInstructionalContentTokens(
    text
  );

const isAboutContentTokens =
  getInstructionalContentTokens(
    isAbout
  );

const isAboutTokenSet =
  new Set(
    isAboutContentTokens
  );

const mainIdeaAddsNoNewContent =
  mainIdeaContentTokens.length > 0 &&
  isAboutContentTokens.length > 0 &&
  mainIdeaContentTokens.every(
    (token) =>
      isAboutTokenSet.has(token)
  );

if (mainIdeaAddsNoNewContent) {
  return {
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

    relationshipEvidence: {
      addsNewOrganizingContent:
        false,

      responseContentTokens:
        mainIdeaContentTokens,

      isAboutContentTokens,

      readerInferenceRequired:
        false,
    },
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

Determine only whether the student's response functions as one valid Main Idea within that specific Frame.

A valid Main Idea may function in either of these ways:

1. CONTENT ORGANIZER
- expresses one major idea connected to the accepted Key Topic;
- supports or helps organize the accepted Is About statement;
- functions as a broad category, cause, effect, part, stage, pattern, or major idea;
- is broad enough to organize multiple meaningful Essential Details.

2. INSTRUCTIONAL ORGANIZER
- functions as a legitimate heading used to organize the student's thinking or learning about the accepted Key Topic;
- may organize prior knowledge, expected learning, questions, observations, evidence, or other instructional categories;
- examples include "Know Already," "Expect to Learn," and "Want to Know";
- does not need to make a direct content claim about the Key Topic when its instructional organizing function is clear.

3. CHRONOLOGICAL OR SEQUENTIAL ORGANIZER
- names a major event, stage, development, or turning point within a sequence;
- may be written as an event title, short phrase, or concise action;
- can organize multiple Essential Details explaining what happened, why it happened, who was involved, or what resulted;
- remains a valid Main Idea even when it describes one historical event, because the event itself functions as a major section of the larger topic.

4. REQUIRED INPUT, CONDITION, OR COMPONENT ORGANIZER
- identifies a meaningful group of inputs, requirements, conditions, materials, or components needed for a process, system, event, or outcome;
- may name multiple related requirements within one organizing statement;
- can organize several Essential Details explaining each input, condition, component, its role, or why it is necessary;
- remains a valid Main Idea when the grouped requirements form one major section of the larger topic.

Examples:

Valid required-input Main Idea:
- Photosynthesis requires water and carbon dioxide.

Possible Essential Details beneath it:
- Plants absorb water through their roots.
- Carbon dioxide enters leaves through stomata.
- Water supplies hydrogen used in producing glucose.
- Carbon dioxide supplies carbon used in producing glucose.

Valid component Main Idea:
- A computer system depends on hardware and software.

Possible Essential Details beneath it:
- Hardware includes physical devices such as the processor and memory.
- Software provides instructions that tell the hardware what to do.

Do not classify a grouped requirement, input set, or component relationship as an Essential Detail merely because it names specific items. Determine whether the complete response can organize several parallel explanations beneath it.

Examples of valid chronological or sequential Main Ideas:
- Castro Comes to Power in Cuba
- Bay of Pigs Invasion
- Nuclear Face-Off with the USSR
- Outbreak of War
- The First Stage of the Process
- Final Resolution

Do not classify a major event or stage as an Essential Detail merely because it is specific in time.

Distinguish between:
- a major event that organizes a section of the topic; and
- one smaller fact, statistic, action, or outcome that explains that event.

For example:

Valid chronological Main Idea:
- Castro Comes to Power in Cuba

Essential Detail beneath that Main Idea:
- Castro nationalized United States-owned businesses

Valid chronological Main Idea:
- Bay of Pigs Invasion

Essential Detail beneath that Main Idea:
- Twenty thousand Cuban troops defeated fourteen hundred invaders

Carefully distinguish a Main Idea from an Essential Detail.

A Main Idea is the broader organizing category, heading, cause,
effect, stage, perspective, pattern, or major event that structures
a section of the Frame.

An Essential Detail is one specific condition, fact, example,
reason, action, outcome, observation, statistic, or piece of
evidence that belongs beneath that broader organizer.

Do not classify a response as a Main Idea merely because several
additional facts could be written about it. Most Essential Details
can also be explained with more information.

Use the nesting and hierarchical-level test:

- Ask whether the response naturally belongs beneath a broader
  organizing label already implied by the Frame.
- If it does, classify it as an Essential Detail.

- Ask whether the response functions at the same hierarchical level
  as the few major ideas needed to organize and explain the accepted
  Key Topic and Is About statement.

- A response is not a Main Idea merely because it names something
  related to the topic, could serve as a possible category, or could
  have several facts written beneath it.

- Ask whether several parallel facts, examples, conditions, actions,
  or outcomes could sit beneath the response as one coherent section
  AND whether that section represents a significant organizing part
  of the whole Frame.

- If the response is narrower than the level at which the significant
  organizing ideas of the Frame belong, treat it as an Essential
  Detail rather than promoting it to Main Idea status.

- Only when the response functions as one of the significant
  organizing sections of this specific Frame may it function as a
  Main Idea.

For contributing-factor Frames, distinguish the broad factor from
one manifestation of that factor.

Example:

Main Idea:
- Lack of planning

Essential Details beneath it:
- There were not enough lifeboats
- The crew was not adequately prepared
- Evacuation procedures were unclear

Therefore, "There were not enough lifeboats" is not the organizing
category. It is one specific condition demonstrating the broader
Main Idea "Lack of planning."

When the response is one specific item beneath a broader category:
- functionsAsOrganizingIdea must be false;
- supportableWithMultipleDetails must be false;
- functionsOnlyAsDetail must be true.

DECISION PRIORITY:

When a response could be interpreted either as a specific detail or as a major chronological organizer, determine its function within the complete supplied Frame.

A major event, stage, development, or turning point that could organize multiple smaller facts must be treated as a Main Idea, even though it is specific in time.

This chronological-organizer rule takes priority over the general rule that a specific event or action may function as an Essential Detail.

Use this distinction:

- "Castro comes to power in Cuba" is a major chronological organizer and therefore functions as a Main Idea.
- "Castro nationalized United States-owned businesses" is one smaller fact explaining that organizer and therefore functions only as an Essential Detail.
- "Bay of Pigs Invasion" is a major chronological organizer and therefore functions as a Main Idea.

When a response clearly meets the applicable Main Idea criteria:
- functionsAsOrganizingIdea must be true;
- supportableWithMultipleDetails must be true;
- functionsOnlyAsDetail must be false,

Confidence represents how clearly the response functions as a Main Idea within the supplied Frame—not certainty about outside factual knowledge.

Do not lower confidence merely because a valid Main Idea is concise, written as an event title, expressed as a short chronological phrase, or written as a one-word category.

When all five instructional judgments are clear within the supplied Frame, confidence should normally be 0.90 or higher.

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
  // --------------------------------------------------
  // STEP 1 — DETERMINISTIC VALIDATION
  //
  // Observable instructional evidence is evaluated
  // before semantic evidence may be requested.
  // --------------------------------------------------

  const deterministicResult =
    validateMainIdeaResponse(
      response,
      keyTopic,
      isAbout
    );

  console.log(
    "MAIN IDEA VALIDATION:",
    deterministicResult
  );

  // --------------------------------------------------
  // STEP 2 — SEMANTIC EVIDENCE GATE
  //
  // Semantic evidence is permitted only for limited or
  // substantive responses whose instructional function
  // cannot be established deterministically.
  // --------------------------------------------------

  const semanticEvidenceDiagnoses = [
  "insufficientObservableEvidence",
  "relationshipIncomplete",
  "relationshipNotEstablished",
  "relationshipUndetermined",
];
  
  const limitedResponseCanBeReviewed =
    deterministicResult
      ?.componentEvidenceLevel ===
      "limited" &&

    semanticEvidenceDiagnoses.includes(
      deterministicResult
        ?.diagnosis
    );

  const substantiveResponseCanBeReviewed =
    deterministicResult
      ?.componentEvidenceLevel ===
      "substantive" &&

    (
      deterministicResult
        ?.relationshipEvidence
        ?.requiresSemanticInference ===
        true ||

      semanticEvidenceDiagnoses.includes(
        deterministicResult
          ?.diagnosis
      )
    );

  const requiresSemanticEvidence =
    limitedResponseCanBeReviewed ||
    substantiveResponseCanBeReviewed;

  // --------------------------------------------------
  // STEP 3 — DETERMINISTIC FINAL RESULT
  //
  // When semantic evidence is not permitted or required,
  // the deterministic result remains authoritative.
  // --------------------------------------------------

  if (!requiresSemanticEvidence) {
    return {
      ...deterministicResult,

      validationSource:
        "deterministic",
    };
  }

  // --------------------------------------------------
  // STEP 4 — BOUNDED SEMANTIC EVIDENCE
  //
  // AI supplies semantic evidence only.
  // It does not validate, save, or advance student work.
  // --------------------------------------------------

  const semanticEvidence =
    await getMainIdeaSemanticEvidence(
      response,
      keyTopic,
      isAbout
    );

  // --------------------------------------------------
  // STEP 5 — JAVASCRIPT GOVERNANCE DECISION
  //
  // JavaScript applies the complete Main Idea contract
  // to the bounded semantic evidence.
  // --------------------------------------------------

  const relationshipEstablished =
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
  // --------------------------------------------------
  // STEP 6 — GOVERNED ACCEPTANCE
  // --------------------------------------------------

  if (relationshipEstablished) {
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
        ...deterministicResult
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

  // --------------------------------------------------
  // STEP 7 — GOVERNED NON-ACCEPTANCE
  //
  // JavaScript determines whether the response functions
  // only as a Detail or fails to establish the required
  // Main Idea relationship.
  // --------------------------------------------------

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
      ...deterministicResult
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

  const hasObservableContent =
  /[a-z0-9]/i.test(text);

if (!hasObservableContent) {
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
- adds concrete information that is more specific than the Main Idea;
- helps the reader understand how, why, when, where, what happened, what resulted, what example demonstrates the idea, or what evidence supports it;
- can function as a fact, example, observation, explanation, event, condition, action, result, or piece of evidence;
- does not merely repeat, shorten, or make a more general statement about the Main Idea;
- does not function primarily as a separate major organizing Main Idea.

Specificity test:
- specificEnough must be false when the response merely states a broad condition, requirement, or related idea without adding concrete supporting information.
- A response is not specific enough merely because it is related to the Main Idea.
- Ask whether the response gives the reader new supporting information beyond what the Main Idea already communicates.
- If the reader would still need to ask "How?", "Why?", "What specifically?", or "What evidence shows that?", specificEnough should be false.

Example:

Accepted Main Idea:
"Plants use sunlight to produce glucose."

Proposed Essential Detail:
"Plants need sunlight."

This is related to the Main Idea, but it does not add concrete supporting information about how sunlight is used, why it is needed, what happens, or what evidence supports the Main Idea.

For this example:
- supportsMainIdea may be true;
- functionsAsEssentialDetail may be emerging or context-dependent;
- specificEnough must be false;
- introducesSeparateMainIdea must be false.

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
  // --------------------------------------------------
  // STEP 1 — DETERMINISTIC VALIDATION
  //
  // Observable instructional evidence is evaluated
  // before semantic evidence may be requested.
  // --------------------------------------------------

  const deterministicResult =
    validateEssentialDetailResponse(
      response,
      currentMainIdea
    );

  console.log(
    "ESSENTIAL DETAIL VALIDATION:",
    deterministicResult
  );

  // --------------------------------------------------
  // STEP 2 — SEMANTIC EVIDENCE GATE
  //
  // Semantic evidence is permitted only for limited or
  // substantive responses whose supporting relationship
  // cannot be established deterministically.
  // --------------------------------------------------

  const semanticEvidenceDiagnoses = [
    "insufficientObservableEvidence",
    "relationshipIncomplete",
    "relationshipNotEstablished",
  ];

  const limitedResponseCanBeReviewed =
    deterministicResult
      ?.componentEvidenceLevel ===
      "limited" &&

    semanticEvidenceDiagnoses.includes(
      deterministicResult
        ?.diagnosis
    );

  const substantiveResponseCanBeReviewed =
    deterministicResult
      ?.componentEvidenceLevel ===
      "substantive" &&

    (
      deterministicResult
        ?.relationshipEvidence
        ?.readerInferenceRequired ===
        true ||

      semanticEvidenceDiagnoses.includes(
        deterministicResult
          ?.diagnosis
      )
    );

  const requiresSemanticEvidence =
    limitedResponseCanBeReviewed ||
    substantiveResponseCanBeReviewed;

  // --------------------------------------------------
  // STEP 3 — DETERMINISTIC FINAL RESULT
  //
  // When semantic evidence is not permitted or required,
  // the deterministic result remains authoritative.
  // --------------------------------------------------

  if (!requiresSemanticEvidence) {
    return {
      ...deterministicResult,

      validationSource:
        "deterministic",
    };
  }

  // --------------------------------------------------
  // STEP 4 — BOUNDED SEMANTIC EVIDENCE
  //
  // AI supplies semantic evidence only.
  // It does not validate, save, or advance student work.
  // --------------------------------------------------

  const semanticEvidence =
    await getEssentialDetailSemanticEvidence(
      response,
      currentMainIdea,
      instructionalContext
    );

  // --------------------------------------------------
  // STEP 5 — JAVASCRIPT GOVERNANCE DECISION
  //
  // JavaScript applies the complete Essential Detail
  // contract to the bounded semantic evidence.
  // --------------------------------------------------

  const relationshipEstablished =
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

  // --------------------------------------------------
  // STEP 6 — GOVERNED ACCEPTANCE
  // --------------------------------------------------

  if (relationshipEstablished) {
    return {
      valid:
        true,

      componentEvidenceLevel:
        deterministicResult
          .componentEvidenceLevel,

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      diagnosis:
        null,

      relationshipEvidence: {
        ...deterministicResult
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

  // --------------------------------------------------
  // STEP 7 — GOVERNED NON-ACCEPTANCE
  //
  // JavaScript determines whether the response functions
  // as a separate Main Idea or fails to establish the
  // required Essential Detail relationship.
  // --------------------------------------------------

  return {
    valid:
      false,

    componentEvidenceLevel:
      deterministicResult
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
      ...deterministicResult
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

  const system = `You provide bounded semantic evidence for a deterministic instructional validator supporting the KU Framing Routine.

The student's assignment context, Thinking Task, completed Frame, and proposed So What will be provided.

Determine only whether the student's response functions as a supported culminating understanding within that specific completed Frame.

The So What should communicate an important understanding about the Key Topic that is supported by the completed Frame.

A successful So What may take different legitimate forms, including:
- a conclusion or generalization;
- an application or implication;
- a connection to another topic or real-world situation;
- a metaphor or analogy;
- a value statement;
- a basic life truth.

Do not require one fixed sentence structure or one predetermined conclusion.

Evaluate each evidence field independently.

anchoredToKeyTopic:
- true when the response clearly concerns the accepted Key Topic;
- false when the response shifts to an unrelated subject.

traceableToCompletedFrame:
- true when a reasonable reader can connect the response to ideas, relationships, examples, or evidence developed in the completed Frame;
- the response may use new wording and may make a supported inference;
- false when the completed Frame provides no meaningful basis for the response.

supportedByCompletedFrame:
- true when the completed Frame reasonably supports the takeaway, connection, implication, metaphor, value, or generalization;
- false when the response introduces an unsupported conclusion or recommendation.

communicatesMeaningfulUnderstanding:
- true only when the response communicates what the student understands after considering the whole Frame;
- the response must express an actual takeaway, relationship, implication, significance, lesson, or conclusion;
- false when the response merely says the topic is important, has effects, matters, is interesting, is good, is bad, or affects people without explaining what is important, meaningful, or consequential;
- false for broad statements such as "Social media has important effects on teenagers" because the reader still does not know what the important understanding is.

specificEnoughToUnderstand:
- true when the reader can identify the student's actual takeaway, even when the response uses a metaphor, analogy, application, or broad life truth;
- false when the wording is so general that it could apply to many topics without meaningful change;
- false when the reader would still need to ask "What effect?", "Why does it matter?", "What does this show?", or "What is the actual takeaway?";
- specificity does not require copying Main Ideas or Essential Details.

merelyRepeatsEarlierFrameContent:
- true when the response simply restates the Key Topic, Is About statement, one Main Idea, or one Essential Detail without creating a larger understanding;
- false when the response synthesizes, generalizes, applies, connects, interprets, or explains significance.

Important distinctions:
- A response may be traceable and supported but still lack a meaningful or specific culminating understanding.
- Being grammatically complete does not make a response a successful So What.
- Mentioning the Key Topic does not by itself establish synthesis.
- Do not mark a vague response successful merely because it is generally true.
- Legitimate manual-style metaphors, applications, real-world connections, unit connections, and basic life truths may be fully successful when their meaning is understandable and supported by the completed Frame.
- Do not rewrite, improve, or complete the student's response.
- Do not decide whether the response is accepted.
- Return semantic evidence only.

Return ONLY valid JSON using exactly this structure:
{
  "anchoredToKeyTopic": true,
  "traceableToCompletedFrame": true,
  "supportedByCompletedFrame": true,
  "communicatesMeaningfulUnderstanding": true,
  "specificEnoughToUnderstand": true,
  "merelyRepeatsEarlierFrameContent": false,
  "confidence": 0.00
}

Use a confidence from 0 to 1 representing confidence in the complete evidence assessment.`;

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
  // --------------------------------------------------
  // STEP 1 — DETERMINISTIC VALIDATION
  //
  // Observable instructional evidence is evaluated
  // before semantic evidence may be requested.
  // --------------------------------------------------

  const deterministicResult =
    validateSoWhatResponse(
      response,
      instructionalContext
    );

  console.log(
    "SO WHAT VALIDATION:",
    deterministicResult
  );

  // --------------------------------------------------
  // STEP 2 — SEMANTIC EVIDENCE GATE
  //
  // Semantic evidence is permitted only when the
  // deterministic validator identifies substantive
  // synthesis that requires interpretation within the
  // completed Frame.
  // --------------------------------------------------

  const requiresSemanticEvidence =
    deterministicResult
      ?.relationshipEvidence
      ?.requiresSemanticInference ===
    true;

  // --------------------------------------------------
  // STEP 3 — DETERMINISTIC FINAL RESULT
  //
  // When semantic evidence is not permitted or required,
  // the deterministic result remains authoritative.
  // --------------------------------------------------

  if (!requiresSemanticEvidence) {
    return {
      ...deterministicResult,

      validationSource:
        "deterministic",
    };
  }

  // --------------------------------------------------
  // STEP 4 — BOUNDED SEMANTIC EVIDENCE
  //
  // AI supplies semantic evidence only.
  // It does not validate, save, revise, or advance
  // student work.
  // --------------------------------------------------

  const semanticEvidence =
    await getSoWhatSemanticEvidence(
      response,
      instructionalContext
    );

  // --------------------------------------------------
  // STEP 5 — JAVASCRIPT GOVERNANCE DECISION
  //
  // JavaScript applies the complete So What contract
  // and determines the student's synthesis state.
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
      .confidence >= 0.75;

  // --------------------------------------------------
  // STEP 6 — GOVERNED ACCEPTANCE
  //
  // Supported synthesis satisfies the complete So What
  // instructional contract and may progress.
  // --------------------------------------------------

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
        ...deterministicResult
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
  // STEP 7 — GOVERNED NON-ACCEPTANCE
  //
  // JavaScript distinguishes emerging synthesis from
  // unsupported synthesis and selects the corresponding
  // deterministic diagnosis.
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
        ...deterministicResult
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
      ...deterministicResult
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
// Evidence State Test Suite
// ------------------------------------------------------
//
// Purpose:
//
// Verifies that Evidence State creates one read-only
// representation of the current instructional evidence.
//
// These tests confirm that:
//
// - current evidence remains separate from accumulated
//   evidence;
// - accepted Frame content is preserved;
// - instructional location is observable;
// - pending context is copied;
// - the original runtime state is not mutated.
//
// ------------------------------------------------------

async function runEvidenceStateSelfTests() {
  const originalState = {
    interactionMode:
      "build",

    frameMeta: {
      assignmentContext: {
        raw:
          "Explain how muckrakers influenced Progressive Era reforms.",

        understanding:
          "Explain how muckrakers influenced Progressive Era reforms.",

        studentSummary:
          "you're explaining how muckrakers influenced Progressive Era reforms.",
      },
    },

    assignmentReasoning: {
      task:
        "explain",

      label:
        "Explain",

      confidence:
        1,
    },

    frame: {
      keyTopic:
        "Muckrakers",

      isAbout:
        "Investigative journalists who exposed social and political problems.",

      parentItems: [
        "Problems exposed by muckrakers",
        "Reforms influenced by public awareness",
      ],

      details: [
        [
          "Journalists investigated unsafe working and living conditions.",
        ],

        [
          "Public pressure contributed to consumer-protection laws.",
        ],
      ],

      soWhat:
        "Muckrakers showed that journalism could build public support for reform.",
    },

  pending: {
    type:
      "collectAnotherDetail",

  index:
    1,
  },
};

const originalSnapshot =
  structuredClone(originalState);

  const currentResponse =
    "Public awareness created pressure for reform.";

  const evidenceState =
    buildEvidenceState(
      originalState,
      currentResponse
    );

  const results = [];

  results.push({
    name:
      "Evidence State - Current response is separated",

    passed:
      evidenceState
        ?.currentEvidence
        ?.response ===
      currentResponse,

    expected: {
      response:
        currentResponse,
    },

    actual: {
      response:
        evidenceState
          ?.currentEvidence
          ?.response || null,
    },
  });

  results.push({
    name:
      "Evidence State - Accumulated Frame is preserved",

    passed:
      evidenceState
        ?.accumulatedEvidence
        ?.frame
        ?.keyTopic ===
        "Muckrakers" &&

      evidenceState
        ?.accumulatedEvidence
        ?.frame
        ?.mainIdeas
        ?.length ===
        2 &&

      evidenceState
        ?.accumulatedEvidence
        ?.frame
        ?.details
        ?.[0]
        ?.length ===
        1,

    expected: {
      keyTopic:
        "Muckrakers",

      mainIdeaCount:
        2,

      firstDetailCount:
        1,
    },

    actual: {
      keyTopic:
        evidenceState
          ?.accumulatedEvidence
          ?.frame
          ?.keyTopic || null,

      mainIdeaCount:
        evidenceState
          ?.accumulatedEvidence
          ?.frame
          ?.mainIdeas
          ?.length || 0,

      firstDetailCount:
        evidenceState
          ?.accumulatedEvidence
          ?.frame
          ?.details
          ?.[0]
          ?.length || 0,
    },
  });

  results.push({
    name:
      "Evidence State - Instructional location is observable",

    passed:
      evidenceState
        ?.instructionalLocation
        ?.interactionMode ===
        "build" &&

      evidenceState
        ?.instructionalLocation
        ?.pendingType ===
        "collectAnotherDetail" &&

      evidenceState
        ?.instructionalLocation
        ?.parentAnchor !==
        null,

    expected: {
      interactionMode:
        "build",

      pendingType:
        "collectAnotherDetail",

      parentAnchorAvailable:
        true,
    },

    actual: {
      interactionMode:
        evidenceState
          ?.instructionalLocation
          ?.interactionMode || null,

      pendingType:
        evidenceState
          ?.instructionalLocation
          ?.pendingType || null,

      parentAnchorAvailable:
        evidenceState
          ?.instructionalLocation
          ?.parentAnchor !==
        null,
    },
  });

  results.push({
    name:
      "Evidence State - Original state is not mutated",

    passed:
      JSON.stringify(originalState) ===
      JSON.stringify(originalSnapshot),

    expected: {
      stateUnchanged:
        true,
    },

    actual: {
      stateUnchanged:
        JSON.stringify(originalState) ===
        JSON.stringify(originalSnapshot),
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

function formatEvidenceStateSelfTestResults(
  testResults
) {
  const lines = [
    "🔎 KAW EVIDENCE STATE SELF-TESTS",
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
      "🚀 Evidence State is operating correctly."
    );
  }

  return lines.join("\n");
}

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
  // GOVERNED ESSENTIAL DETAIL SPECIFICITY REGRESSION
  //
  // A response may be related to the accepted Main Idea
  // without adding enough concrete supporting information
  // to function as an Essential Detail.
  // --------------------------------------------------

  const specificityRegressionActual =
    await validateEssentialDetailResponseGoverned(
      "Plants need sunlight.",
      "Plants use sunlight to produce glucose.",
      {
        keyTopic:
          "Photosynthesis",

        isAbout:
          "How plants make food using sunlight.",
      }
    );

  const specificityRegressionPassed =
    specificityRegressionActual
      ?.valid === false &&

    specificityRegressionActual
      ?.relationshipEvidence
      ?.specificEnough === false;

  results.push({
    name:
      "ED Governed - Related but nonspecific detail is blocked",

    passed:
      specificityRegressionPassed,

    response:
      "Plants need sunlight.",

    expected: {
      valid:
        false,

      specificEnough:
        false,
    },

    actual: {
      valid:
        specificityRegressionActual
          ?.valid === true,

      diagnosis:
        specificityRegressionActual
          ?.diagnosis || null,

      specificEnough:
        specificityRegressionActual
          ?.relationshipEvidence
          ?.specificEnough ?? null,

      validationSource:
        specificityRegressionActual
          ?.validationSource || null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME TEST
  //
  // Confirms that the actual first Essential Detail
  // pathway blocks an invalid response before saving it.
  // --------------------------------------------------

  const runtimeState = defaultState();

    runtimeState.interactionMode =
    "build";

  runtimeState.frameMeta.assignmentContext = {
    valid:
      true,

    raw:
      "Explain how social media can affect teen mental health.",

    understanding:
      "Explain how social media can affect teen mental health.",

    studentSummary:
      "you're explaining how social media can affect teen mental health.",

    reasoningType:
      "explain",

    confidence:
      "high",

    confirmed:
      true,

    assignmentEvidenceLevel:
      "substantive",

    assignmentCriteriaStatus:
      "satisfied",

    assignmentContextStatus:
      "established",

    assignmentDemandStatus:
      "established",

    summaryReadinessStatus:
      "ready",

    diagnosis:
      null,

    assignmentEvidence:
      null,

    validationSource:
      "deterministic",

    needsClarification:
      false,

    childAnchor:
      "",

    clarificationCount:
      0,
  };

  runtimeState.assignmentReasoning = {
    task:
      "explain",

    label:
      "Explain",

    confidence:
      1,

    evidence: [
      "assignmentTestState",
    ],

    lastUpdated:
      null,
  };

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
        "collectAnotherDetail" &&
      runtimeActual?.pending?.index ===
        0 &&
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
    savedDetailCount:
      0,

    pendingType:
      "collectAnotherDetail",

    pendingIndex:
      0,

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

    pendingIndex:
      Number.isInteger(
        runtimeActual?.pending?.index
      )
        ? runtimeActual.pending.index
        : null,

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

    stuckRuntimeState.interactionMode =
    "build";

  stuckRuntimeState.frameMeta.assignmentContext = {
    valid:
      true,

    raw:
      "Explain how social media can affect teen mental health.",

    understanding:
      "Explain how social media can affect teen mental health.",

    studentSummary:
      "you're explaining how social media can affect teen mental health.",

    reasoningType:
      "explain",

    confidence:
      "high",

    confirmed:
      true,

    assignmentEvidenceLevel:
      "substantive",

    assignmentCriteriaStatus:
      "satisfied",

    assignmentContextStatus:
      "established",

    assignmentDemandStatus:
      "established",

    summaryReadinessStatus:
      "ready",

    diagnosis:
      null,

    assignmentEvidence:
      null,

    validationSource:
      "deterministic",

    needsClarification:
      false,

    childAnchor:
      "",

    clarificationCount:
      0,
  };

  stuckRuntimeState.assignmentReasoning = {
    task:
      "explain",

    label:
      "Explain",

    confidence:
      1,

    evidence: [
      "assignmentTestState",
    ],

    lastUpdated:
      null,
  };

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
    "collectAnotherDetail" &&
  stuckRuntimeActual?.pending?.index ===
    0 &&
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
    savedDetailCount:
      0,

    pendingType:
      "collectAnotherDetail",

    pendingIndex:
      0,

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

    pendingIndex:
      Number.isInteger(
        stuckRuntimeActual?.pending?.index
      )
        ? stuckRuntimeActual.pending.index
        : null,

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

    validRuntimeState.interactionMode =
    "build";

  validRuntimeState.frameMeta.assignmentContext = {
    valid:
      true,

    raw:
      "Explain how social media can affect teen mental health.",

    understanding:
      "Explain how social media can affect teen mental health.",

    studentSummary:
      "you're explaining how social media can affect teen mental health.",

    reasoningType:
      "explain",

    confidence:
      "high",

    confirmed:
      true,

    assignmentEvidenceLevel:
      "substantive",

    assignmentCriteriaStatus:
      "satisfied",

    assignmentContextStatus:
      "established",

    assignmentDemandStatus:
      "established",

    summaryReadinessStatus:
      "ready",

    diagnosis:
      null,

    assignmentEvidence:
      null,

    validationSource:
      "deterministic",

    needsClarification:
      false,

    childAnchor:
      "",

    clarificationCount:
      0,
  };

  validRuntimeState.assignmentReasoning = {
    task:
      "explain",

    label:
      "Explain",

    confidence:
      1,

    evidence: [
      "assignmentTestState",
    ],

    lastUpdated:
      null,
  };

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

    validRuntimeActual
      .frame
      .details[0]
      .length ===
      1 &&

    validRuntimeActual
      .frame
      .details[0][0] ===
      validRuntimeResponse &&

    validRuntimeActual
      ?.pending
      ?.type ===
      "collectAnotherDetail" &&

    validRuntimeActual
      ?.pending
      ?.index ===
      0 &&

    validRuntimeActual
      ?.progressionAuthorization
      ?.authorized ===
      true &&

    validRuntimeActual
      ?.progressionAuthorization
      ?.selectedContractId ===
      "ED-RTP-001";

  results.push({
    name:
      "ED Runtime - First valid detail is saved and advances",

    passed:
      validRuntimePassed,

    response:
      validRuntimeResponse,

    expected: {
      savedDetailCount:
        1,

    savedDetail:
      validRuntimeResponse,

    pendingType:
      "collectAnotherDetail",

    pendingIndex:
      0,

    progressionAuthorized:
      true,

    selectedContractId:
      "ED-RTP-001",
    },

      actual: {
      savedDetailCount:
        Array.isArray(
          validRuntimeActual?.frame
            ?.details?.[0]
        )
          ? validRuntimeActual
              .frame
              .details[0]
              .length
          : null,

      savedDetail:
        validRuntimeActual?.frame
          ?.details?.[0]?.[0] || null,

      pendingType:
        validRuntimeActual?.pending
          ?.type || null,

      pendingIndex:
        Number.isInteger(
          validRuntimeActual?.pending
            ?.index
        )
          ? validRuntimeActual
              .pending
              .index
          : null,

      progressionAuthorized:
        validRuntimeActual
          ?.progressionAuthorization
          ?.authorized === true,

      selectedContractId:
        validRuntimeActual
          ?.progressionAuthorization
          ?.selectedContractId || null,
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
      valid:
        true,

      raw:
        "Explain how social media can affect teen mental health.",

      understanding:
        "Explain how social media can affect teen mental health.",

      studentSummary:
        "you're explaining how social media can affect teen mental health.",

      reasoningType:
        "explain",

      confidence:
        "high",

      confirmed:
        true,

      assignmentEvidenceLevel:
        "substantive",

      assignmentCriteriaStatus:
        "satisfied",

      assignmentContextStatus:
        "established",

      assignmentDemandStatus:
        "established",

      summaryReadinessStatus:
        "ready",

      diagnosis:
        null,

      assignmentEvidence:
        null,

      validationSource:
        "deterministic",

      needsClarification:
        false,

      childAnchor:
        "",

      clarificationCount:
        0,
    };

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
    "collectAnotherDetail" &&
  secondDetailInvalidActual?.pending?.index ===
    0 &&
  secondDetailInvalidActual?.pending
    ?.instructionalFinding
    ?.diagnosis ===
    "insufficientObservableEvidence";

results.push({
  name:
    "ED Runtime - Second required detail blocks circular response",

  passed:
    secondDetailInvalidPassed,

  response:
    "because it does",

  expected: {
    savedDetailCount:
      1,

    preservedFirstDetail:
      existingFirstDetail,

    pendingType:
      "collectAnotherDetail",

    pendingIndex:
      0,

    diagnosis:
      "insufficientObservableEvidence",
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

    pendingIndex:
      Number.isInteger(
        secondDetailInvalidActual?.pending
          ?.index
      )
        ? secondDetailInvalidActual.pending.index
        : null,

    diagnosis:
      secondDetailInvalidActual?.pending
        ?.instructionalFinding
        ?.diagnosis || null,
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

  const validIsAboutResponse =
    "Photosynthesis is the process plants use to make food using sunlight.";

  // --------------------------------------------------
  // IS ABOUT RUNTIME TEST STATE FACTORY
  //
  // Creates a fully established Assignment Understanding
  // state positioned at Is About capture.
  //
  // Fully established Assignment Understanding is
  // required so the Instructional Situation Engine may
  // evaluate the current component evidence rather than
  // correctly stopping at assignmentUnderstandingRequired.
  // --------------------------------------------------

  function createIsAboutRuntimeTestState() {
    const state =
      defaultState();

    state.interactionMode =
      "build";

    state.frameMeta.assignmentContext = {
      valid:
        true,

      raw:
        "Explain how photosynthesis helps plants make food.",

      understanding:
        "Explain how photosynthesis helps plants make food.",

      studentSummary:
        "You're explaining how photosynthesis helps plants make food.",

      reasoningType:
        "explain",

      confidence:
        "high",

      confirmed:
        true,

      assignmentEvidenceLevel:
        "substantive",

      assignmentCriteriaStatus:
        "satisfied",

      assignmentContextStatus:
        "established",

      assignmentDemandStatus:
        "established",

      summaryReadinessStatus:
        "ready",

      diagnosis:
        null,

      assignmentEvidence:
        null,

      validationSource:
        "deterministic",

      needsClarification:
        false,

      childAnchor:
        "",

      clarificationCount:
        0,
    };

    state.assignmentReasoning = {
      task:
        "explain",

      label:
        "Explain",

      confidence:
        1,

      evidence: [
        "assignmentTestState",
      ],

      lastUpdated:
        null,
    };

    state.frame.keyTopic =
      keyTopic;

    state.frame.isAbout =
      "";

    state.frame.parentItems =
      [];

    state.frame.details =
      [];

    state.pending =
      null;

    return state;
  }

  const tests = [
    {
      name:
        "IA - Empty response",

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
        "IA - Stuck response",

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
        "IA - Repeats Key Topic",

      response:
        "Photosynthesis",

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
        "IA - Too little observable evidence",

      response:
        "Plants make food",

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
        "IA - Observable paraphrase",

      response:
        validIsAboutResponse,

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

// LIVE RUNTIME + SHADOW SITUATION TEST
//
// Repeating the accepted Key Topic fails the required
// Is About relationship.
//
// Authoritative runtime:
// • blocks the response;
// • preserves the empty Is About;
// • preserves the reviseIsAbout location;
// • attaches the governed instructional artifacts.
//
// Governed result:
// • relationshipNeedsRepair.
// --------------------------------------------------

  const repeatedTopicState =
    createIsAboutRuntimeTestState();

  const repeatedTopicActual =
    await updateStateFromStudent(
      repeatedTopicState,
      "Photosynthesis"
    );

  const repeatedTopicPassed =
    repeatedTopicActual
      ?.frame
      ?.isAbout === "" &&

    repeatedTopicActual
    ?.pending
    ?.type ===
    "reviseIsAbout" &&

    repeatedTopicActual
      ?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "repeatsKeyTopic" &&

    repeatedTopicActual
      ?.componentInstructionalFinding
      ?.frameComponent ===
      "isAbout" &&

    repeatedTopicActual
      ?.componentInstructionalFinding
      ?.diagnosis ===
      "repeatsKeyTopic" &&

    repeatedTopicActual
      ?.instructionalSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .RELATIONSHIP_NEEDS_REPAIR &&

   repeatedTopicActual
    ?.instructionalSituation
    ?.governance
    ?.shadowMode === true &&

  repeatedTopicActual
    ?.progressionAuthorization
    ?.authorized === false &&

  repeatedTopicActual
    ?.progressionAuthorization
    ?.selectedContractId ===
    "IA-RNR-001";

  results.push({
    name:
      "IA Runtime - Repeated Key Topic produces relationship repair shadow situation",

    passed:
      repeatedTopicPassed,

    response:
      "Photosynthesis",

    expected: {
      savedIsAbout:
        "",

      pendingType:
        "reviseIsAbout",

      diagnosis:
        "repeatsKeyTopic",

      governedSituation:
        INSTRUCTIONAL_SITUATIONS
          .RELATIONSHIP_NEEDS_REPAIR,

      shadowMode:
        true,

      progressionAuthorized:
        false,

      selectedContractId:
        "IA-RNR-001",
    },

    actual: {
      savedIsAbout:
        repeatedTopicActual
          ?.frame
          ?.isAbout || "",

      pendingType:
        repeatedTopicActual
          ?.pending
          ?.type || null,

      diagnosis:
        repeatedTopicActual
          ?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      componentFinding:
        repeatedTopicActual
          ?.componentInstructionalFinding ||
        null,

      governedSituation:
        repeatedTopicActual
          ?.instructionalSituation
          ?.instructionalSituation ||
        null,

    shadowMode:
        repeatedTopicActual
          ?.instructionalSituation
          ?.governance
          ?.shadowMode === true,

      progressionAuthorized:
        repeatedTopicActual
          ?.progressionAuthorization
          ?.authorized === true,

      selectedContractId:
        repeatedTopicActual
          ?.progressionAuthorization
          ?.selectedContractId || null,
    },
  });

// LIVE RUNTIME + SHADOW SITUATION TEST
//
// Limited Is About evidence requires component
// revision but does not establish relationship failure.
//
// Authoritative runtime:
// • blocks the response;
// • preserves the empty Is About;
// • preserves the reviseIsAbout location;
// • attaches the governed instructional artifacts.
//
// Governed result:
// • componentNeedsRevision.
// --------------------------------------------------
  const limitedEvidenceState =
    createIsAboutRuntimeTestState();

    const limitedEvidenceActual =
    await updateStateFromStudent(
      limitedEvidenceState,
      "Food for plants"
    );

  const limitedEvidencePassed =
    limitedEvidenceActual
      ?.frame
      ?.isAbout === "" &&

    limitedEvidenceActual
      ?.pending
      ?.type ===
      "reviseIsAbout" &&

    limitedEvidenceActual
      ?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "insufficientObservableEvidence" &&

    limitedEvidenceActual
      ?.componentInstructionalFinding
      ?.componentCriteriaStatus ===
      "partiallySatisfied" &&

    limitedEvidenceActual
      ?.componentInstructionalFinding
      ?.relationshipStatus ===
      "undetermined" &&

    limitedEvidenceActual
      ?.instructionalSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .COMPONENT_NEEDS_REVISION &&

    limitedEvidenceActual
      ?.instructionalSituation
      ?.governance
      ?.shadowMode === true;

  results.push({
    name:
      "IA Runtime - Limited evidence produces component revision shadow situation",

    passed:
      limitedEvidencePassed,

    response:
      "Food for plants",

    expected: {
      savedIsAbout:
        "",

      pendingType:
        "reviseIsAbout",

      diagnosis:
        "insufficientObservableEvidence",

      componentCriteriaStatus:
        "partiallySatisfied",

      relationshipStatus:
        "undetermined",

      governedSituation:
        INSTRUCTIONAL_SITUATIONS
          .COMPONENT_NEEDS_REVISION,

      shadowMode:
        true,
    },

    actual: {
      savedIsAbout:
        limitedEvidenceActual
          ?.frame
          ?.isAbout || "",

      pendingType:
        limitedEvidenceActual
          ?.pending
          ?.type || null,

      diagnosis:
        limitedEvidenceActual
          ?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      componentCriteriaStatus:
        limitedEvidenceActual
          ?.componentInstructionalFinding
          ?.componentCriteriaStatus ||
        null,

      relationshipStatus:
        limitedEvidenceActual
          ?.componentInstructionalFinding
          ?.relationshipStatus ||
        null,

      governedSituation:
        limitedEvidenceActual
          ?.instructionalSituation
          ?.instructionalSituation ||
        null,

      shadowMode:
        limitedEvidenceActual
          ?.instructionalSituation
          ?.governance
          ?.shadowMode === true,
    },
  });

 // LIVE RUNTIME + SHADOW SITUATION TEST
//
// A first no-evidence response does not establish
// genuine struggle.
//
// Authoritative runtime:
// • blocks the response;
// • preserves the empty Is About;
// • preserves the reviseIsAbout location;
// • attaches the governed instructional artifacts.
//
// Governed result:
// • noComponentEvidence.
// --------------------------------------------------
  const noEvidenceState =
    createIsAboutRuntimeTestState();

  const noEvidenceActual =
    await updateStateFromStudent(
      noEvidenceState,
      "idk"
    );

  const noEvidencePassed =
    noEvidenceActual
      ?.frame
      ?.isAbout === "" &&

    noEvidenceActual
      ?.pending
      ?.type ===
      "reviseIsAbout" &&

    noEvidenceActual
      ?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "noComponentEvidence" &&

    noEvidenceActual
      ?.componentInstructionalFinding
      ?.componentEvidenceLevel ===
      "none" &&

    noEvidenceActual
      ?.instructionalSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .NO_COMPONENT_EVIDENCE &&

    noEvidenceActual
      ?.instructionalSituation
      ?.governance
      ?.shadowMode === true;

  results.push({
    name:
      "IA Runtime - First no-evidence response does not establish genuine struggle",

    passed:
      noEvidencePassed,

    response:
      "idk",

    expected: {
      savedIsAbout:
        "",

      pendingType:
        "reviseIsAbout",

      diagnosis:
        "noComponentEvidence",

      componentEvidenceLevel:
        "none",

      governedSituation:
        INSTRUCTIONAL_SITUATIONS
          .NO_COMPONENT_EVIDENCE,

      shadowMode:
        true,
    },

    actual: {
      savedIsAbout:
        noEvidenceActual
          ?.frame
          ?.isAbout || "",

      pendingType:
        noEvidenceActual
          ?.pending
          ?.type || null,

      diagnosis:
        noEvidenceActual
          ?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      componentEvidenceLevel:
        noEvidenceActual
          ?.componentInstructionalFinding
          ?.componentEvidenceLevel ||
        null,

      governedSituation:
        noEvidenceActual
          ?.instructionalSituation
          ?.instructionalSituation ||
        null,

      shadowMode:
        noEvidenceActual
          ?.instructionalSituation
          ?.governance
          ?.shadowMode === true,
    },
  });

// LIVE RUNTIME + SHADOW SITUATION TEST
//
// A valid Is About is saved by the authoritative runtime,
// and the governed engine establishes readyToProgress.
// --------------------------------------------------
  const validIsAboutState =
    createIsAboutRuntimeTestState();

  const validIsAboutActual =
    await updateStateFromStudent(
      validIsAboutState,
      validIsAboutResponse
    );

  const validIsAboutPassed =
    validIsAboutActual
      ?.frame
      ?.isAbout ===
      validIsAboutResponse &&

    validIsAboutActual
      ?.pending
      ?.type ===
      "confirmIsAbout" &&

    validIsAboutActual
      ?.componentInstructionalFinding
      ?.componentCriteriaStatus ===
      "satisfied" &&

    validIsAboutActual
      ?.componentInstructionalFinding
      ?.relationshipStatus ===
      "established" &&

    validIsAboutActual
      ?.instructionalSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .READY_TO_PROGRESS &&

    validIsAboutActual
      ?.instructionalSituation
      ?.governance
      ?.controlsProgression ===
      false &&

    validIsAboutActual
      ?.instructionalSituation
      ?.governance
      ?.shadowMode === true &&

    validIsAboutActual
      ?.progressionAuthorization
      ?.authorized === true &&

    validIsAboutActual
      ?.progressionAuthorization
      ?.selectedContractId ===
      "IA-RTP-001";

  results.push({
    name:
      "IA Runtime - Valid paraphrase produces ready-to-progress shadow situation",

    passed:
      validIsAboutPassed,

    response:
      validIsAboutResponse,

    expected: {
      savedIsAbout:
        validIsAboutResponse,

      pendingType:
        "confirmIsAbout",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      governedSituation:
        INSTRUCTIONAL_SITUATIONS
          .READY_TO_PROGRESS,

      controlsProgression:
        false,

      shadowMode:
        true,

      progressionAuthorized:
        true,

      selectedContractId:
        "IA-RTP-001",
    },

    actual: {
      savedIsAbout:
        validIsAboutActual
          ?.frame
          ?.isAbout || null,

      pendingType:
        validIsAboutActual
          ?.pending
          ?.type || null,

      componentCriteriaStatus:
        validIsAboutActual
          ?.componentInstructionalFinding
          ?.componentCriteriaStatus ||
        null,

      relationshipStatus:
        validIsAboutActual
          ?.componentInstructionalFinding
          ?.relationshipStatus ||
        null,

      governedSituation:
        validIsAboutActual
          ?.instructionalSituation
          ?.instructionalSituation ||
        null,

      controlsProgression:
        validIsAboutActual
          ?.instructionalSituation
          ?.governance
          ?.controlsProgression === true,

      shadowMode:
        validIsAboutActual
          ?.instructionalSituation
          ?.governance
          ?.shadowMode === true,

      progressionAuthorized:
        validIsAboutActual
          ?.progressionAuthorization
          ?.authorized === true,

      selectedContractId:
        validIsAboutActual
          ?.progressionAuthorization
          ?.selectedContractId || null,
    },
  });

  // --------------------------------------------------
  // GOVERNED PERSISTENCE TEST
  //
  // Genuine struggle requires:
  //
  // • current no-component evidence;
  // • prior governed support;
  // • prior no-component evidence;
  // • the same active instructional location.
  //
  // This test exercises the shadow refresh directly so
  // contract activation and communication do not influence
  // the Instructional Situation result.
  // --------------------------------------------------

  const persistentStruggleState =
    createIsAboutRuntimeTestState();

  persistentStruggleState.pending = {
  type:
    "reviseIsAbout",

  instructionalFinding: {
    frameComponent:
      "isAbout",

    componentEvidenceLevel:
      "none",

    componentCriteriaStatus:
      "notSatisfied",

    relationshipStatus:
      "undetermined",

    diagnosis:
      "noComponentEvidence",
  },

  instructionalContract: {
    contractId:
      "IA-NCE-001",

    frameComponent:
      "isAbout",

    instructionalSituation:
      "noComponentEvidence",

    instructionalGoal:
      "elicitComponentEvidence",

    teachingMove:
      "reduceCognitiveLoad",

    thinkingMove:
      "Reconnect the student to the accepted Key Topic and invite them to explain what the whole topic is about in their own understandable words without suggesting or supplying the Is About statement.",

    communicationPattern:
      "briefReassuranceThenQuestion",

    aiContextualizes:
      true,
  },

  instructionalActivation: {
    contractId:
      "IA-NCE-001",

    execution: {
      contractId:
        "IA-NCE-001",
    },

    aiPayload: {
      contractId:
        "IA-NCE-001",
    },
  },
};
  
  const persistentObservationReport = {
    version:
      "1.0",

    source:
      "selfTestObservation",

    studentInteraction:
      "idk",

    observations: [
      {
        category:
          "uncertaintyExpression",

        evidenceText:
          "idk",

        confidence:
          1,
      },
    ],

    ambiguityPresent:
      false,
  };

  persistentStruggleState.observationReport =
    structuredClone(
      persistentObservationReport
    );

  const persistentEvidenceState =
    buildEvidenceState(
      persistentStruggleState,
      "idk",
      persistentObservationReport
    );

  const persistentAssessment =
    buildInstructionalAssessment(
      persistentEvidenceState
    );

  persistentAssessment
    .interactionInstructionalFinding =
      buildInteractionInstructionalFinding(
        persistentEvidenceState,
        persistentAssessment
      );

  persistentStruggleState
    .instructionalAssessment =
      structuredClone(
        persistentAssessment
      );

  const persistentValidation =
    validateIsAboutResponse(
      "idk",
      keyTopic
    );

  const persistentComponentFinding =
    buildComponentInstructionalFinding({
      frameComponent:
        "isAbout",

      validation:
        persistentValidation,

      evidence: {
        keyTopic,

        attemptedIsAbout:
          "idk",

        normalizedIsAbout:
          "idk",
      },
    });

  const persistentSituation =
    refreshInstructionalSituationWithComponentFinding({
      state:
        persistentStruggleState,

      currentResponse:
        "idk",

      componentFinding:
        persistentComponentFinding,
    });

  const persistentStrugglePassed =
    persistentSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .GENUINE_STRUGGLE &&

    persistentSituation
      ?.inputs
      ?.evidenceHistory
      ?.priorSupportActive === true &&

    persistentSituation
      ?.inputs
      ?.evidenceHistory
      ?.priorNoEvidence === true &&

    persistentSituation
      ?.governance
      ?.genuineStruggleRequiresPersistence ===
      true &&

    persistentSituation
      ?.governance
      ?.selectsInstructionalContract ===
      false &&

    persistentSituation
      ?.governance
      ?.shadowMode === true;

  results.push({
    name:
      "IA Governed - Persistent no-evidence after support establishes genuine struggle",

    passed:
      persistentStrugglePassed,

    response:
      "idk",

    expected: {
      governedSituation:
        INSTRUCTIONAL_SITUATIONS
          .GENUINE_STRUGGLE,

      priorSupportActive:
        true,

      priorNoEvidence:
        true,

      genuineStruggleRequiresPersistence:
        true,

      selectsInstructionalContract:
        false,

      shadowMode:
        true,
    },

    actual: {
      governedSituation:
        persistentSituation
          ?.instructionalSituation ||
        null,

      priorSupportActive:
        persistentSituation
          ?.inputs
          ?.evidenceHistory
          ?.priorSupportActive === true,

      priorNoEvidence:
        persistentSituation
          ?.inputs
          ?.evidenceHistory
          ?.priorNoEvidence === true,

      genuineStruggleRequiresPersistence:
        persistentSituation
          ?.governance
          ?.genuineStruggleRequiresPersistence ===
        true,

      selectsInstructionalContract:
        persistentSituation
          ?.governance
          ?.selectsInstructionalContract ===
        true,

      shadowMode:
        persistentSituation
          ?.governance
          ?.shadowMode === true,
    },
  });

  // --------------------------------------------------
  // AUTHORITATIVE INSTRUCTIONAL CONTRACT SELECTION TEST
  //
  // Confirms that each established governed Is About
  // Instructional Situation selects its matching
  // predetermined authoritative contract.
  // --------------------------------------------------

  const contractSelectionPassed =
    repeatedTopicActual
      ?.instructionalContractSelection
      ?.selectedContractId ===
      "IA-RNR-001" &&

    limitedEvidenceActual
      ?.instructionalContractSelection
      ?.selectedContractId ===
      "IA-CNR-001" &&

    noEvidenceActual
      ?.instructionalContractSelection
      ?.selectedContractId ===
      "IA-NCE-001" &&

    validIsAboutActual
      ?.instructionalContractSelection
      ?.selectedContractId ===
      "IA-RTP-001" &&

    persistentStruggleState
      ?.instructionalContractSelection
      ?.selectedContractId ===
      "IA-GS-001" &&

    repeatedTopicActual
      ?.instructionalContractSelection
      ?.selectionStatus ===
      "contractSelected" &&

    limitedEvidenceActual
      ?.instructionalContractSelection
      ?.selectionStatus ===
      "contractSelected" &&

    noEvidenceActual
      ?.instructionalContractSelection
      ?.selectionStatus ===
      "contractSelected" &&

    validIsAboutActual
      ?.instructionalContractSelection
      ?.selectionStatus ===
      "contractSelected" &&

    persistentStruggleState
      ?.instructionalContractSelection
      ?.selectionStatus ===
      "contractSelected" &&

    repeatedTopicActual
      ?.instructionalContractSelection
      ?.governance
      ?.contractExecuted ===
      true &&

    repeatedTopicActual
      ?.instructionalContractSelection
      ?.governance
      ?.controlsCommunication ===
      true &&

    repeatedTopicActual
      ?.instructionalContractSelection
      ?.governance
      ?.authoritative ===
      true &&

    repeatedTopicActual
      ?.instructionalContractSelection
      ?.governance
      ?.shadowMode ===
      false;

  results.push({
    name:
      "IA Governed - Instructional situations select matching authoritative contracts",

    passed:
      contractSelectionPassed,

    expected: {
      relationshipNeedsRepairContract:
        "IA-RNR-001",

      componentNeedsRevisionContract:
        "IA-CNR-001",

      noComponentEvidenceContract:
        "IA-NCE-001",

      readyToProgressContract:
        "IA-RTP-001",

      genuineStruggleContract:
        "IA-GS-001",

      selectionStatus:
        "contractSelected",

      contractExecuted:
        true,

      controlsCommunication:
        true,

      authoritative:
        true,

      shadowMode:
        false,
    },

    actual: {
      relationshipNeedsRepairContract:
        repeatedTopicActual
          ?.instructionalContractSelection
          ?.selectedContractId ||
        null,

      componentNeedsRevisionContract:
        limitedEvidenceActual
          ?.instructionalContractSelection
          ?.selectedContractId ||
        null,

      noComponentEvidenceContract:
        noEvidenceActual
          ?.instructionalContractSelection
          ?.selectedContractId ||
        null,

      readyToProgressContract:
        validIsAboutActual
          ?.instructionalContractSelection
          ?.selectedContractId ||
        null,

      genuineStruggleContract:
        persistentStruggleState
          ?.instructionalContractSelection
          ?.selectedContractId ||
        null,

      selectionStatuses: {
        relationshipNeedsRepair:
          repeatedTopicActual
            ?.instructionalContractSelection
            ?.selectionStatus ||
          null,

        componentNeedsRevision:
          limitedEvidenceActual
            ?.instructionalContractSelection
            ?.selectionStatus ||
          null,

        noComponentEvidence:
          noEvidenceActual
            ?.instructionalContractSelection
            ?.selectionStatus ||
          null,

        readyToProgress:
          validIsAboutActual
            ?.instructionalContractSelection
            ?.selectionStatus ||
          null,

        genuineStruggle:
          persistentStruggleState
            ?.instructionalContractSelection
            ?.selectionStatus ||
          null,
      },

      contractExecuted:
        repeatedTopicActual
          ?.instructionalContractSelection
          ?.governance
          ?.contractExecuted ===
        true,

      controlsCommunication:
        repeatedTopicActual
          ?.instructionalContractSelection
          ?.governance
          ?.controlsCommunication ===
        true,

      authoritative:
        repeatedTopicActual
          ?.instructionalContractSelection
          ?.governance
          ?.authoritative ===
        true,

      shadowMode:
        repeatedTopicActual
          ?.instructionalContractSelection
          ?.governance
          ?.shadowMode ===
        true,
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

  const requiredInputsMainIdea =
    "Photosynthesis requires water and carbon dioxide.";

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
      valid:
        true,

      raw:
        "Explain how social media can affect teen mental health.",

      understanding:
        "Explain how social media can affect teen mental health.",

      studentSummary:
        "you're explaining how social media can affect teen mental health.",

      reasoningType:
        "explain",

      confidence:
        "high",

      confirmed:
        true,

      assignmentEvidenceLevel:
        "substantive",

      assignmentCriteriaStatus:
        "satisfied",

      assignmentContextStatus:
        "established",

      assignmentDemandStatus:
        "established",

      summaryReadinessStatus:
        "ready",

      diagnosis:
        null,

      assignmentEvidence:
        null,

      validationSource:
        "deterministic",

      needsClarification:
        false,

      childAnchor:
        "",

      clarificationCount:
        0,
    };

    state.assignmentReasoning = {
      task:
        "explain",

      label:
        "Explain",

      confidence:
        1,

      evidence: [
        "assignmentTestState",
      ],

      lastUpdated:
        null,
    };

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
        "MI - Restates Is About without new organizing content",

      response:
        "Plants make food.",

      keyTopic:
        "Photosynthesis",

      isAbout:
        "How plants make food using sunlight.",

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
        test.keyTopic ??
          keyTopic,
        test.isAbout ??
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

  const requiredInputsActual =
  await validateMainIdeaResponseGoverned(
    requiredInputsMainIdea,
    "Photosynthesis",
    "How plants make food."
  );

const requiredInputsPassed =
  requiredInputsActual.valid === true &&

  requiredInputsActual
    .componentCriteriaStatus ===
    "satisfied" &&

  requiredInputsActual
    .relationshipStatus ===
    "established" &&

  requiredInputsActual
    .diagnosis ===
    null &&

  requiredInputsActual
    .validationSource ===
    "deterministicWithSemanticEvidence";

results.push({
  name:
    "MI Governed - Required inputs function as an organizer",

  passed:
    requiredInputsPassed,

  response:
    requiredInputsMainIdea,

  expected: {
    valid:
      true,

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
    requiredInputsActual,
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
      "offerAnotherMainIdea" &&

    requiredActual
      ?.progressionAuthorization
      ?.authorized ===
      true &&

    requiredActual
      ?.progressionAuthorization
      ?.selectedContractId ===
      "MI-RTP-001";
  
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

    progressionAuthorized:
      true,

    selectedContractId:
      "MI-RTP-001",
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

      progressionAuthorized:
        requiredActual
          ?.progressionAuthorization
          ?.authorized === true,

      selectedContractId:
        requiredActual
          ?.progressionAuthorization
          ?.selectedContractId || null,
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
      "reviseMainIdeaAt" &&

    invalidRevisionActual?.pending
      ?.index ===
      0 &&

    invalidRevisionActual?.pending
      ?.instructionalFinding
      ?.diagnosis ===
      "repeatsKeyTopic" &&

    invalidRevisionActual
      ?.progressionAuthorization
      ?.authorized ===
      false &&

    invalidRevisionActual
      ?.progressionAuthorization
      ?.selectedContractId ===
      "MI-RNR-001";

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
        "reviseMainIdeaAt",

      pendingIndex:
        0,

      diagnosis:
        "repeatsKeyTopic",

      progressionAuthorized:
        false,

      selectedContractId:
        "MI-RNR-001",
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

      pendingIndex:
        Number.isInteger(
          invalidRevisionActual?.pending
            ?.index
        )
          ? invalidRevisionActual
              .pending
              .index
          : null,

      diagnosis:
        invalidRevisionActual?.pending
          ?.instructionalFinding
          ?.diagnosis || null,

      progressionAuthorized:
        invalidRevisionActual
          ?.progressionAuthorization
          ?.authorized === true,

      selectedContractId:
        invalidRevisionActual
          ?.progressionAuthorization
          ?.selectedContractId || null,
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
    "collectAnotherMainIdea" &&

  invalidOptionalActual?.pending
    ?.instructionalFinding
    ?.diagnosis ===
    "repeatsIsAbout";

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
    "collectAnotherMainIdea",

  diagnosis:
    "repeatsIsAbout",
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

async function runSoWhatSelfTests(
  batch = "all"
) {
  const runValidationBatch =
    batch === "all" ||
    batch === "validation";

  const runRuntimeBatch =
    batch === "all" ||
    batch === "runtime";

  const runManualBatch =
    batch === "all" ||
    batch === "manual";
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

  if (runValidationBatch) {
  
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
    
    }
    if (runValidationBatch) {
      
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

  }

  // ==================================================
  // KU FRAMING ROUTINE MANUAL SO WHAT PRESSURE TESTS
  //
  // These governed benchmarks use completed Frames and
  // approved So What examples from the KU manual.
  //
  // Their purpose is to confirm that Kaw:
  //
  // - accepts the manual's completed So What statements;
  // - accepts different manual-approved rhetorical forms;
  // - evaluates support from the completed Frame;
  // - does not require one fixed So What sentence pattern.
  // ==================================================

  const runManualSupportedSoWhatTest =
    async ({
      name,
      response,
      context,
    }) => {
      const actual =
        await validateSoWhatResponseGoverned(
          response,
          context
        );

      const passed =
        actual.valid ===
          true &&

        actual.componentEvidenceLevel ===
          "substantive" &&

        actual.componentCriteriaStatus ===
          "satisfied" &&

        actual.relationshipStatus ===
          "established" &&

        actual.synthesisState ===
          "supported" &&

        actual.diagnosis ===
          null &&

        actual.validationSource ===
          "deterministicWithSemanticEvidence";

      results.push({
        name,

        passed,

        response,

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

        actual,
      });
    };

  // --------------------------------------------------
  // MANUAL FRAME 1: STRATEGIC LEARNERS
  // --------------------------------------------------

  const strategicLearnersContext = {
    assignmentContext:
      "Understand what strategic learners do before, during, and after learning.",

    thinkingTask:
      "explain",

    keyTopic:
      "Strategic Learners",

    isAbout:
      "students who use good study plans",

    mainIdeas: [
      "They think BEFORE",
      "They think DURING",
      "They think AFTER",
    ],

    details: [
      [
        "By organizing books and materials",
        "By setting goals and making plans",
        "By scheduling time wisely",
      ],

      [
        "By asking and answering questions",
        "By linking new information to background knowledge",
        "By looking for patterns",
      ],

      [
        "By thinking how new information can be used",
        "By evaluating results",
        "By anticipating future needs",
      ],
    ],
  };

  if (runValidationBatch) {
    await runManualSupportedSoWhatTest({
      name:
        "SW Manual - Strategic Learners completed So What is accepted",

      response:
        "Strategic learners actively and purposefully use smart strategies before, during, and after learning.",

      context:
        strategicLearnersContext,
    });
  }

  if (runRuntimeBatch) {

  // ==================================================
  // SO WHAT LIVE RUNTIME TESTS
  // ==================================================
  // These tests exercise the actual So What capture,
  // expansion, and revision pathways through
  // updateStateFromStudent().
  //
  // They confirm that governed validation occurs before
  // student work is saved or replaced.
  // ==================================================

    function createSoWhatRuntimeState() {
    const state =
      defaultState();

    state.interactionMode =
      "build";

    state.frameMeta.assignmentContext = {
      valid:
        true,

      raw:
        instructionalContext
          .assignmentContext
          .raw,

      understanding:
        instructionalContext
          .assignmentContext
          .understanding,

      studentSummary:
        instructionalContext
          .assignmentContext
          .studentSummary,

      reasoningType:
        instructionalContext
          .thinkingTask
          .task,

      confidence:
        "high",

      confirmed:
        true,

      assignmentEvidenceLevel:
        "substantive",

      assignmentCriteriaStatus:
        "satisfied",

      assignmentContextStatus:
        "established",

      assignmentDemandStatus:
        "established",

      summaryReadinessStatus:
        "ready",

      diagnosis:
        null,

      assignmentEvidence:
        null,

      validationSource:
        "deterministic",

      needsClarification:
        false,

      childAnchor:
        "",

      clarificationCount:
        0,
    };

    state.assignmentReasoning = {
      task:
        instructionalContext
          .thinkingTask
          .task,

      label:
        instructionalContext
          .thinkingTask
          .label,

      confidence:
        1,

      evidence: [
        "assignmentTestState",
      ],

      lastUpdated:
        null,
    };

    state.frame.keyTopic =
      instructionalContext.keyTopic;

    state.frame.isAbout =
      instructionalContext.isAbout;

    state.frame.parentItems = [
      ...instructionalContext.mainIdeas,
    ];

    state.frame.details =
      instructionalContext.details.map(
        (bucket) => [...bucket]
      );

    state.frame.soWhat =
      "";

    state.pending =
      null;

    return state;
  }

  // --------------------------------------------------
  // LIVE RUNTIME: INVALID INITIAL CAPTURE
  //
  // Confirms an insufficient initial So What is blocked
  // before it can be saved.
  // --------------------------------------------------

  const invalidInitialState =
    createSoWhatRuntimeState();

  const invalidInitialResponse =
    "It really matters";

  const invalidInitialActual =
    await updateStateFromStudent(
      invalidInitialState,
      invalidInitialResponse
    );

  const invalidInitialPassed =
  invalidInitialActual?.frame?.soWhat ===
    "" &&

  invalidInitialActual?.pending?.type ===
    "collectMoreSoWhat" &&

  invalidInitialActual?.pending
    ?.instructionalFinding
    ?.frameComponent ===
    "soWhat" &&

  invalidInitialActual?.pending
    ?.instructionalFinding
    ?.diagnosis ===
    "insufficientObservableEvidence";

results.push({
  name:
    "SW Runtime - Invalid initial So What is blocked",

  passed:
    invalidInitialPassed,

  response:
    invalidInitialResponse,

  expected: {
    savedSoWhat:
      "",

    pendingType:
      "collectMoreSoWhat",

    frameComponent:
      "soWhat",

    diagnosis:
      "insufficientObservableEvidence",
  },

  actual: {
    savedSoWhat:
      invalidInitialActual?.frame
        ?.soWhat || "",

    pendingType:
      invalidInitialActual?.pending
        ?.type || null,

    frameComponent:
      invalidInitialActual?.pending
        ?.instructionalFinding
        ?.frameComponent || null,

    diagnosis:
      invalidInitialActual?.pending
        ?.instructionalFinding
        ?.diagnosis || null,
  },
});
    
  // --------------------------------------------------
  // LIVE RUNTIME: VALID INITIAL CAPTURE
  //
  // Confirms a supported initial So What is saved and
  // advances to the optional expansion offer.
  // --------------------------------------------------

  const validInitialState =
    createSoWhatRuntimeState();

  const validInitialActual =
    await updateStateFromStudent(
      validInitialState,
      supportedSoWhat
    );

  const validInitialPassed =
    validInitialActual?.frame?.soWhat ===
      supportedSoWhat &&

    validInitialActual?.pending?.type ===
      "offerMoreSoWhat" &&

    validInitialActual
      ?.progressionAuthorization
      ?.authorized ===
      true &&

    validInitialActual
      ?.progressionAuthorization
      ?.selectedContractId ===
      "SW-RTP-001";

  results.push({
    name:
      "SW Runtime - Valid initial So What saves and advances",

    passed:
      validInitialPassed,

    response:
      supportedSoWhat,

    expected: {
      savedSoWhat:
        supportedSoWhat,

      pendingType:
        "offerMoreSoWhat",

      progressionAuthorized:
        true,

      selectedContractId:
        "SW-RTP-001",
    },

    actual: {
      savedSoWhat:
        validInitialActual?.frame
          ?.soWhat || null,

      pendingType:
        validInitialActual?.pending
          ?.type || null,

      progressionAuthorized:
        validInitialActual
          ?.progressionAuthorization
          ?.authorized === true,

      selectedContractId:
        validInitialActual
          ?.progressionAuthorization
          ?.selectedContractId || null,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME: INVALID REVISION
  //
  // Confirms an invalid replacement does not overwrite
  // the student's accepted So What.
  // --------------------------------------------------

  const invalidRevisionState =
    createSoWhatRuntimeState();

  invalidRevisionState.frame.soWhat =
    supportedSoWhat;

  invalidRevisionState.pending = {
    type:
      "confirmSoWhat",

    awaitingRevision:
      true,
  };

  const invalidRevisionResponse =
    "It really matters";

  const invalidRevisionActual =
    await updateStateFromStudent(
      invalidRevisionState,
      invalidRevisionResponse
    );

  const invalidRevisionPassed =
  invalidRevisionActual?.frame?.soWhat ===
    supportedSoWhat &&

  invalidRevisionActual?.pending?.type ===
    "confirmSoWhat" &&

  invalidRevisionActual?.pending
    ?.awaitingRevision ===
    true &&

  invalidRevisionActual?.pending
    ?.instructionalFinding
    ?.frameComponent ===
    "soWhat" &&

  invalidRevisionActual?.pending
    ?.instructionalFinding
    ?.diagnosis ===
    "insufficientObservableEvidence";

results.push({
  name:
    "SW Runtime - Invalid revision preserves original So What",

  passed:
    invalidRevisionPassed,

  response:
    invalidRevisionResponse,

  expected: {
    preservedSoWhat:
      supportedSoWhat,

    pendingType:
      "confirmSoWhat",

    awaitingRevision:
      true,

    frameComponent:
      "soWhat",

    diagnosis:
      "insufficientObservableEvidence",
  },

  actual: {
    preservedSoWhat:
      invalidRevisionActual?.frame
        ?.soWhat || null,

    pendingType:
      invalidRevisionActual?.pending
        ?.type || null,

    awaitingRevision:
      invalidRevisionActual?.pending
        ?.awaitingRevision === true,

    frameComponent:
      invalidRevisionActual?.pending
        ?.instructionalFinding
        ?.frameComponent || null,

    diagnosis:
      invalidRevisionActual?.pending
        ?.instructionalFinding
        ?.diagnosis || null,
  },
});
    
  // --------------------------------------------------
  // LIVE RUNTIME: VALID REVISION
  //
  // Confirms a supported replacement overwrites the old
  // So What and returns to confirmation.
  // --------------------------------------------------

  const validRevisionState =
    createSoWhatRuntimeState();

  validRevisionState.frame.soWhat =
    supportedSoWhat;

  validRevisionState.pending = {
    type:
      "confirmSoWhat",

    awaitingRevision:
      true,
  };

  const validRevisionResponse =
    "Online comparison and constant pressure can shape how teenagers feel about themselves, so careful social media use can help protect their mental health.";

  const validRevisionActual =
    await updateStateFromStudent(
      validRevisionState,
      validRevisionResponse
    );

  const validRevisionPassed =
    validRevisionActual?.frame?.soWhat ===
      validRevisionResponse &&

    validRevisionActual?.pending?.type ===
      "confirmSoWhat" &&

    validRevisionActual?.pending
      ?.awaitingRevision !==
      true;

  results.push({
    name:
      "SW Runtime - Valid revision replaces original So What",

    passed:
      validRevisionPassed,

    response:
      validRevisionResponse,

    expected: {
      savedSoWhat:
        validRevisionResponse,

      pendingType:
        "confirmSoWhat",

      awaitingRevision:
        false,
    },

    actual: {
      savedSoWhat:
        validRevisionActual?.frame
          ?.soWhat || null,

      pendingType:
        validRevisionActual?.pending
          ?.type || null,

      awaitingRevision:
        validRevisionActual?.pending
          ?.awaitingRevision === true,
    },
  });

  // --------------------------------------------------
  // LIVE RUNTIME: ADDITIONAL CONTENT
  //
  // Confirms additional So What content is validated
  // together with the accepted So What before being
  // appended.
  // --------------------------------------------------

  const additionalContentState =
    createSoWhatRuntimeState();

  additionalContentState.frame.soWhat =
    supportedSoWhat;

  additionalContentState.pending = {
    type:
      "collectMoreSoWhat",
  };

  const additionalSentence =
    "This also shows that the way teenagers experience online pressure matters as much as how often they use social media.";

  const expectedExpandedSoWhat =
    cleanText(
      `${supportedSoWhat} ${additionalSentence}`
    );

  const additionalContentActual =
    await updateStateFromStudent(
      additionalContentState,
      additionalSentence
    );

  const additionalContentPassed =
    additionalContentActual?.frame
      ?.soWhat ===
      expectedExpandedSoWhat &&

    additionalContentActual?.pending
      ?.type ===
      "confirmSoWhat";

  results.push({
    name:
      "SW Runtime - Governed additional content is appended",

    passed:
      additionalContentPassed,

    response:
      additionalSentence,

    expected: {
      savedSoWhat:
        expectedExpandedSoWhat,

      pendingType:
        "confirmSoWhat",
    },

    actual: {
      savedSoWhat:
        additionalContentActual?.frame
          ?.soWhat || null,

      pendingType:
        additionalContentActual?.pending
          ?.type || null,
    },
  });
  
  }

  if (runManualBatch) {
  
  // --------------------------------------------------
  // MANUAL FRAME 2: PROGRESSIVE ERA
  // --------------------------------------------------

  const progressiveEraContext = {
    assignmentContext:
      "Understand the social problems, tools for change, and social changes of the Progressive Era.",

    thinkingTask:
      "explain",

    keyTopic:
      "Progressive Era",

    isAbout:
      "a period of social change in the U.S.",

    mainIdeas: [
      "Social Problems",
      "Tools for Social Change",
      "Social Changes",
    ],

    details: [
      [
        "Unsafe food",
        "Monopolies",
        "Unsafe and unfair working conditions",
        "Limited voting rights",
      ],

      [
        "Muckrakers wrote about problems",
        "Bully pulpits forced new laws",
        "Activists organized protests",
        "Demonstrators created public pressure",
      ],

      [
        "Meat Inspection Act",
        "Anti-trust Act",
        "Commerce and Labor Departments",
        "Voting rights expanded",
      ],
    ],
  };

  await runManualSupportedSoWhatTest({
    name:
      "SW Manual - Progressive Era completed So What is accepted",

    response:
      "To really create social change, many people have to be organized, outspoken, and persistent!",

    context:
      progressiveEraContext,
  });

  // --------------------------------------------------
  // MANUAL-APPROVED REAL-WORLD CONNECTION
  // --------------------------------------------------

  await runManualSupportedSoWhatTest({
    name:
      "SW Manual - Progressive Era real-world connection is accepted",

    response:
      "Considering the disparity between the lifestyles of the wealthy versus the poor, this country could use another Progressive Era now.",

    context:
      progressiveEraContext,
  });

  // --------------------------------------------------
  // MANUAL-APPROVED METAPHOR
  // --------------------------------------------------

  await runManualSupportedSoWhatTest({
    name:
      "SW Manual - Progressive Era metaphor is accepted",

    response:
      "The Progressive Era is like Habitat for Humanity because both involve leaders, organization, awareness, lots of involvement, persistence, and hard work. In addition, both are about improving things.",

    context:
      progressiveEraContext,
  });

  // --------------------------------------------------
  // MANUAL-APPROVED UNIT CONNECTION
  // --------------------------------------------------

  await runManualSupportedSoWhatTest({
    name:
      "SW Manual - Progressive Era unit connection is accepted",

    response:
      "The Progressive Era resulted from abuses of wealth and power, plus a weak federal government.",

    context:
      progressiveEraContext,
  });

  // --------------------------------------------------
  // MANUAL-APPROVED APPLICATION OR IMPLICATION
  // --------------------------------------------------

  await runManualSupportedSoWhatTest({
    name:
      "SW Manual - Progressive Era implication is accepted",

    response:
      "To change things, many people have to be organized, outspoken, and persistent!",

    context:
      progressiveEraContext,
  });

  // --------------------------------------------------
  // MANUAL-APPROVED BASIC LIFE TRUTH
  // --------------------------------------------------

  await runManualSupportedSoWhatTest({
    name:
      "SW Manual - Progressive Era basic life truth is accepted",

    response:
      "The behavior of governments often swings like a pendulum, from too little control and regulation to too much control and regulation. The Progressive Era marked the beginning of a swing away from too little control.",

    context:
      progressiveEraContext,
  });
  }

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

    const rejectedCommunication =
      activation?.communicationDebug || null;
    
    const responseForInspection =
      response ||
      rejectedCommunication?.rawResponse ||
      "";

    const communicationLicense =
      activation?.aiPayload
        ?.communicationLicense || null;

     const validation =
      rejectedCommunication?.validation ||
      validateInstructionalCommunicationResponse(
        responseForInspection,
        communicationLicense
  );

    const normalizedResponse =
  cleanText(response).toLowerCase();

const internalArchitectureTerms = [
  "instructional finding",
  "instructional situation",
  "teaching move",
  "thinking move",
  "communication license",
  "evidence state",
  "support level",
  "contract id",
  "diagnosis",
];

const exposesInternalArchitecture =
  internalArchitectureTerms.some(
    (term) =>
      normalizedResponse.includes(term)
  );

const unsupportedSuccessLanguage = [
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

const makesUnsupportedSuccessClaim =
  unsupportedSuccessLanguage.some(
    (phrase) =>
      normalizedResponse.includes(phrase)
  );

const passed =
  !!response &&
  validation.valid &&
  validation.questionCount === 1 &&
  !exposesInternalArchitecture &&
  !makesUnsupportedSuccessClaim;

  results.push({
  name:
    test.name,

  passed,

  response:
    responseForInspection,
    
  expected: {
    nonEmptyResponse:
      true,

    validLicenseResponse:
      true,

    questionCount:
      1,

    exposesInternalArchitecture:
      false,

    makesUnsupportedSuccessClaim:
      false,

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

    exposesInternalArchitecture,

    makesUnsupportedSuccessClaim,

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
    id: "evidenceState",
    label: "Evidence State",
    run: runEvidenceStateSelfTests,
    format: formatEvidenceStateSelfTestResults,
  },
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
// COMPONENT SELF-TEST RUNNER
//
// Runs one registered deterministic or governed suite by
// component ID.
//
// This exists so cleanup work can be verified without
// executing the full monolithic /run tests command.
//
// It does not modify student state.
// ------------------------------------------------------

async function runDeterministicSelfTestSuiteById(
  suiteId
) {
  const suite =
    DETERMINISTIC_SELF_TEST_SUITES.find(
      (candidate) =>
        candidate.id === suiteId
    );

  if (!suite) {
    return null;
  }

  const result =
    await suite.run();

  return {
    id:
      suite.id,

    label:
      suite.label,

    result,

    formatted:
      suite.format(result),
  };
}

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

  hasSufficientAssignmentUnderstanding(
    state
  ) &&

  state?.frameMeta?.assignmentContext
    ?.confirmed === false &&

  state?.pending?.type ===
    "confirmAssignmentUnderstanding";

  results.push({
    name:
      "Student Simulation - Assignment is understood",

    passed:
      assignmentPassed,

    expected: {
  sufficientUnderstanding:
    true,

  confirmed:
    false,

  pendingType:
    "confirmAssignmentUnderstanding",
},

actual: {
  sufficientUnderstanding:
    hasSufficientAssignmentUnderstanding(
      state
    ),

  confirmed:
    state?.frameMeta
      ?.assignmentContext
      ?.confirmed === true,

  pendingType:
    state?.pending?.type || null,

  thinkingTask:
    state?.assignmentReasoning?.task || null,
},
  });

  // --------------------------------------------------
// STEP 2: Confirm shared assignment understanding
// --------------------------------------------------

state = await updateStateFromStudent(
  state,
  "1"
);

const assignmentConfirmationPassed =
  state?.frameMeta
    ?.assignmentContext
    ?.confirmed === true &&

  state?.pending?.type ===
    "assignmentReasoningIntro";

results.push({
  name:
    "Student Simulation - Assignment understanding confirmed",

  passed:
    assignmentConfirmationPassed,

  expected: {
    confirmed:
      true,

    pendingType:
      "assignmentReasoningIntro",
  },

  actual: {
    confirmed:
      state?.frameMeta
        ?.assignmentContext
        ?.confirmed === true,

    pendingType:
      state?.pending?.type || null,
  },
});

// --------------------------------------------------
// WORKFLOW GATEWAY: Strengthen branch remains bounded
// --------------------------------------------------

  const strengthenWorkflowState =
    structuredClone(
      state
    );

  const strengthenSelectedState =
    await updateStateFromStudent(
      strengthenWorkflowState,
      "2"
    );

  const strengthenWorkflowPassed =
    strengthenSelectedState
      ?.interactionMode ===
      "strengthen" &&

    strengthenSelectedState
      ?.pending
      ?.type ===
      "strengthenComponentSelection" &&

    strengthenSelectedState
      ?.frame
      ?.keyTopic === "";

  results.push({
    name:
      "Student Simulation - Strengthen workflow selected without changing student work",

    passed:
      strengthenWorkflowPassed,

    expected: {
      interactionMode:
        "strengthen",

      pendingType:
        "strengthenComponentSelection",

      keyTopic:
        "",
    },

    actual: {
      interactionMode:
        strengthenSelectedState
          ?.interactionMode || null,

      pendingType:
        strengthenSelectedState
          ?.pending
          ?.type || null,

      keyTopic:
        strengthenSelectedState
          ?.frame
          ?.keyTopic || "",
    },
  });

  // --------------------------------------------------
  // STEP 3: Choose Build Mode
  // --------------------------------------------------

  state = await updateStateFromStudent(
    state,
    "1"
  );

    const workflowPassed =
    state?.interactionMode ===
      "build" &&

    state?.pending ===
      null &&

    getStage(state) ===
      "keyTopic" &&

    state?.frame?.keyTopic ===
      "";
  
  results.push({
    name:
      "Student Simulation - Build Mode selected",

    passed:
      workflowPassed,

        expected: {
      interactionMode:
        "build",

      pendingType:
        null,

      stage:
        "keyTopic",

      keyTopic:
        "",
    },

    actual: {
      interactionMode:
        state?.interactionMode || null,

      pendingType:
        state?.pending?.type || null,

      stage:
        getStage(state),

      keyTopic:
        state?.frame
          ?.keyTopic || "",
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
    "collectAnotherDetail" &&
  state?.pending?.index ===
    0 &&
  state?.pending?.instructionalFinding
    ?.diagnosis ===
    "relationshipIncomplete";

results.push({
  name:
    "Student Simulation - Incomplete Detail is blocked",

  passed:
    incompleteDetailPassed,

  expected: {
    savedDetailCount:
      0,

    pendingType:
      "collectAnotherDetail",

    pendingIndex:
      0,

    diagnosis:
      "relationshipIncomplete",
  },

  actual: {
    savedDetailCount:
      state?.frame?.details?.[0]?.length || 0,

    pendingType:
      state?.pending?.type || null,

    pendingIndex:
      Number.isInteger(
        state?.pending?.index
      )
        ? state.pending.index
        : null,

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

function getParentAnchorStage(state) {
  const pendingType = state?.pending?.type || null;

  // Confirmation/export pending states take priority because they
  // represent the active structural stage the student is currently in.
  if (pendingType && PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType]) {
    return PARENT_ANCHOR_BRIDGE.confirmationStageByPending[pendingType];
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
    isOverlay: loopType === "overlay",
    isExport: loopType === "export",
  };
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
// STUDENT-WORK MUTATION PROTECTION
//
// Prevents clearly conversational responses from being
// saved as replacement or optional Frame content.
//
// This guard does not determine whether proposed student
// work satisfies component criteria.
//
// Any response not deterministically identified as
// conversational proceeds to the governed component
// validator, which remains the instructional authority.
// ------------------------------------------------------

async function classifyStudentWorkMutationIntent(
  state,
  message
) {
  const text =
    cleanText(message);

  const normalized =
    text.toLowerCase();

  if (
    !text ||
    isWeakFrameResponse(text)
  ) {
    return {
      accept:
        false,

      intent:
        "stuck",

      confidence:
        1,

      source:
        "deterministic",
    };
  }

  // Choice language and other conversational responses
  // are not replacement Frame content.
  if (
    isAffirmative(normalized) ||
    isNegative(normalized) ||
    normalized === "2" ||
    isMetaResponse(normalized)
  ) {
    return {
      accept:
        false,

      intent:
        "uncertain",

      confidence:
        1,

      source:
        "deterministic",
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
        normalized.startsWith(
          `${direction} `
        )
    )
  ) {
    return {
      accept:
        false,

      intent:
        "revision_direction",

      confidence:
        1,

      source:
        "deterministic",
    };
  }

  // Proposed student-authored content proceeds to the
  // governed component validator.
  return {
    accept:
      true,

    intent:
      "productive",

    confidence:
      1,

    source:
      "deterministicDefault",
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
  const kt =
    cleanText(keyTopic).toLowerCase();

  if (!kt) {
    return true;
  }

  // A struggle, uncertainty, or meta response is not
  // student evidence for the Key Topic component.
  if (
    isStuckMessage(kt) ||
    isWeakFrameResponse(kt) ||
    isMetaResponse(kt)
  ) {
    return true;
  }

  // Reject explicitly generic nonexamples.
  // Natural topic phrases such as "My grandfather,"
  // "My first job," and "My greatest accomplishment"
  // remain valid Key Topics.
  if (GENERIC_KEY_TOPICS.has(kt)) {
    return true;
  }

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

// ------------------------------------------------------
// GOVERNED INSTRUCTIONAL SUPPORT ATTACHMENT
// ------------------------------------------------------
function attachGovernedSupportToPending(
  state,
  message,
  intentResult = {}
) {
  console.log(
    "[KAW][GOVERNED SUPPORT] Student Response:",
    message
  );

  const currentPending =
    state?.pending &&
    typeof state.pending === "object"
      ? structuredClone(
          state.pending
        )
      : null;

  if (!currentPending) {
    throw new Error(
      "Governed support requires an active instructional location."
    );
  }

  const instructionalFinding =
    intentResult
      ?.instructionalFinding ||
    null;

  const instructionalSituation =
    state?.instructionalSituation &&
    typeof state
      .instructionalSituation === "object"
      ? state.instructionalSituation
      : null;

  const instructionalContract =
    state?.instructionalContractSelection
      ?.selectedContract ||
    null;

  console.log(
    "[KAW][GOVERNED SUPPORT] Instructional Situation:",
    instructionalSituation
      ?.instructionalSituation || null
  );

  console.log(
    "[KAW][GOVERNED SUPPORT] Selected Contract:",
    instructionalContract
      ?.contractId || null
  );

  if (
    !instructionalSituation ||
    !instructionalContract
  ) {
    throw new Error(
      "Governed support requires an established Instructional Situation and selected Instructional Contract."
    );
  }

  // Contract execution must see the current deterministic
  // finding while preserving the exact instructional
  // location already owned by runtime progression.
  const activationState = {
    ...state,

    pending: {
      ...currentPending,

      instructionalFinding,
    },
  };

  const instructionalActivation =
    activateInstructionalContract(
      instructionalContract,
      activationState
    );

  console.log(
    "ACTIVATION:",
    instructionalActivation
  );

  if (!instructionalActivation) {
    throw new Error(
      "Governed support requires a valid Instructional Contract activation."
    );
  }

  // Preserve the real instructional location directly.
  //
  // No recovery overlay, resume wrapper, or alternate
  // pending-state identity is created.
  state.pending = {
    ...currentPending,

    instructionalFinding,

    instructionalContract: {
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
    },

    instructionalActivation: {
      contractId:
        instructionalActivation.contractId,

      execution:
        instructionalActivation.execution,

      aiPayload:
        instructionalActivation.aiPayload,
    },
  };

  return state;
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

// ======================================================
// ASSIGNMENT UNDERSTANDING VALIDATOR
// ======================================================
//
// Constitutional Role:
//
// The Assignment Understanding Validator is the
// instructional gateway for Kaw.
//
// It determines whether sufficient assignment evidence
// exists to begin instructional reasoning safely.
//
// Every downstream inference, validator, and instructional
// decision depends upon this gateway.
//
// The validator evaluates three gates:
//
// 1. Assignment Context
//    Is there enough evidence to understand what the work
//    is about?
//
// 2. Assignment Demand
//    Is there enough evidence to understand what the
//    student is expected to think about, explain, analyze,
//    compare, evaluate, create, or otherwise accomplish?
//
// 3. Shared Summary Readiness
//    Can Kaw summarize the assignment without guessing or
//    introducing unsupported meaning?
//
// AI supplies bounded semantic evidence only.
//
// JavaScript applies the instructional criteria and
// retains final authority over readiness, clarification,
// confirmation, and progression.
//
// ======================================================


// ------------------------------------------------------
// DETERMINISTIC ASSIGNMENT UNDERSTANDING VALIDATION
//
// Evaluates only observable evidence that does not require
// semantic interpretation.
//
// This validator does not infer the assignment's meaning.
// It determines whether the response contains enough
// substantive language to permit bounded semantic review.
// ------------------------------------------------------

function validateAssignmentUnderstanding(
  rawAssignment
) {
  const assignment =
    cleanText(rawAssignment);

  const words =
    assignment
      .split(/\s+/)
      .filter(Boolean);

  if (!assignment) {
    return {
      valid:
        false,

      assignmentEvidenceLevel:
        "none",

      assignmentCriteriaStatus:
        "notSatisfied",

      assignmentContextStatus:
        "undetermined",

      assignmentDemandStatus:
        "undetermined",

      summaryReadinessStatus:
        "notReady",

      diagnosis:
        "emptyAssignmentEvidence",
    };
  }

  if (
    isStartupCommand(assignment) ||
    isStuckMessage(assignment) ||
    isMetaResponse(assignment)
  ) {
    return {
      valid:
        false,

      assignmentEvidenceLevel:
        "none",

      assignmentCriteriaStatus:
        "notSatisfied",

      assignmentContextStatus:
        "undetermined",

      assignmentDemandStatus:
        "undetermined",

      summaryReadinessStatus:
        "notReady",

      diagnosis:
        "noAssignmentEvidence",
    };
  }

  if (words.length < 2) {
    return {
      valid:
        false,

      assignmentEvidenceLevel:
        "limited",

      assignmentCriteriaStatus:
        "notSatisfied",

      assignmentContextStatus:
        "undetermined",

      assignmentDemandStatus:
        "undetermined",

      summaryReadinessStatus:
        "notReady",

      diagnosis:
        "insufficientAssignmentEvidence",
    };
  }

  // --------------------------------------------------
  // SEMANTIC INFERENCE GAP
  //
  // The response contains substantive assignment language.
  //
  // Whether that language establishes Assignment Context,
  // Assignment Demand, and Shared Summary Readiness
  // requires bounded semantic evaluation.
  //
  // No task-word list controls the final decision.
  // --------------------------------------------------

  return {
    valid:
      false,

    assignmentEvidenceLevel:
      "substantive",

    assignmentCriteriaStatus:
      "partiallySatisfied",

    assignmentContextStatus:
      "undetermined",

    assignmentDemandStatus:
      "undetermined",

    summaryReadinessStatus:
      "undetermined",

    diagnosis:
      "assignmentUnderstandingUndetermined",

    assignmentEvidence: {
      requiresSemanticInference:
        true,

      readerInferenceRequired:
        true,
    },
  };
}


// ------------------------------------------------------
// ASSIGNMENT UNDERSTANDING SEMANTIC EVIDENCE
//
// Provides bounded semantic evidence for the three AUV
// gates.
//
// AI does not determine whether instruction begins.
// AI does not select the Thinking Task.
// AI does not select the next instructional move.
// AI does not answer or complete the assignment.
//
// AI returns evidence only.
//
// JavaScript remains the final instructional authority.
// ------------------------------------------------------

async function getAssignmentUnderstandingSemanticEvidence(
  rawAssignment
) {
  const assignment =
    cleanText(rawAssignment);

  if (!assignment) {
    return {
      assignmentContextEstablished:
        false,

      assignmentDemandEstablished:
        false,

      sharedSummaryReady:
        false,

      studentSummary:
        "",

      understanding:
        "",

      reasoningType:
        "",

      confidence:
        0,

      source:
        "notRequested",
    };
  }

  const system = `You provide bounded semantic evidence for a deterministic instructional validator supporting the KU Framing Routine.

The student's description of an assignment will be provided.

Evaluate only whether sufficient evidence exists to establish three instructional gates:

1. ASSIGNMENT CONTEXT
Can a reasonable reader understand what topic, text, concept, issue, event, process, product, or body of work the assignment concerns?

2. ASSIGNMENT DEMAND
Can a reasonable reader understand what the student is expected to think about, explain, analyze, compare, evaluate, interpret, summarize, create, organize, demonstrate, or otherwise accomplish?

3. SHARED SUMMARY READINESS
Can the assignment be summarized back to the student accurately without guessing, inventing an instructional demand, or adding unsupported meaning?

Important distinctions:

- An instructional activity is not automatically an instructional demand.
- Reading about a topic does not by itself establish what the student must do with that topic.
- Studying, learning about, researching, watching, working on, or having homework about a topic may establish context while leaving the assignment demand unknown.
- A required product such as an essay, presentation, model, report, response, or project does not by itself establish the thinking demand.
- Assignment demand is established only when the expected intellectual work or intended accomplishment is reasonably understandable.
- Do not require one particular academic verb when the demand is otherwise clear.
- Do not infer an unstated demand from the subject area.
- Do not answer the assignment.
- Do not teach the content.
- Do not create any part of the student's work.
- Do not select progression.
- Return semantic evidence only.

When writing studentSummary:
- Speak directly to the student.
- Begin with "you're..."
- Preserve the actual topic and demand.
- Do not add missing requirements.
- Keep it to one natural sentence.
- Leave it empty when a faithful summary would require guessing.

When writing understanding:
- Preserve the assignment meaning in neutral language.
- Do not add requirements or conclusions.
- Leave it empty when a faithful understanding cannot yet be established.

reasoningType may contain a concise label such as explain, compare, analyze, evaluate, interpret, summarize, create, organize, or unknown.
It is evidence only and does not control Thinking Task inference.

Return only the required JSON object.`;

  const user = `Student's accumulated assignment description:

"${assignment}"

Determine whether the Assignment Context, Assignment Demand, and Shared Summary Readiness gates are established.`;

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
              "assignment_understanding_semantic_evidence",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                assignmentContextEstablished: {
                  type:
                    "boolean",
                },

                assignmentDemandEstablished: {
                  type:
                    "boolean",
                },

                sharedSummaryReady: {
                  type:
                    "boolean",
                },

                studentSummary: {
                  type:
                    "string",
                },

                understanding: {
                  type:
                    "string",
                },

                reasoningType: {
                  type:
                    "string",
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
                "assignmentContextEstablished",
                "assignmentDemandEstablished",
                "sharedSummaryReady",
                "studentSummary",
                "understanding",
                "reasoningType",
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
      assignmentContextEstablished:
        parsed
          .assignmentContextEstablished ===
        true,

      assignmentDemandEstablished:
        parsed
          .assignmentDemandEstablished ===
        true,

      sharedSummaryReady:
        parsed.sharedSummaryReady ===
        true,

      studentSummary:
        cleanText(
          parsed.studentSummary || ""
        ),

      understanding:
        cleanText(
          parsed.understanding || ""
        ),

      reasoningType:
        cleanText(
          parsed.reasoningType || ""
        ),

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
      "Assignment Understanding semantic evidence error:",
      error
    );

    return {
      assignmentContextEstablished:
        false,

      assignmentDemandEstablished:
        false,

      sharedSummaryReady:
        false,

      studentSummary:
        "",

      understanding:
        "",

      reasoningType:
        "",

      confidence:
        0,

      source:
        "semanticEvidenceUnavailable",
    };
  }
}


// ------------------------------------------------------
// GOVERNED ASSIGNMENT UNDERSTANDING VALIDATION
//
// Runs deterministic validation first.
//
// Semantic evidence is requested only when deterministic
// validation identifies substantive assignment evidence.
//
// JavaScript applies all three AUV gates and determines
// whether the assignment may advance to student
// confirmation.
//
// Student confirmation remains a separate runtime gate.
// ------------------------------------------------------

async function validateAssignmentUnderstandingGoverned(
  rawAssignment
) {
  // --------------------------------------------------
  // STEP 1 — DETERMINISTIC VALIDATION
  // --------------------------------------------------

  const deterministicResult =
    validateAssignmentUnderstanding(
      rawAssignment
    );

  console.log(
    "ASSIGNMENT UNDERSTANDING VALIDATION:",
    deterministicResult
  );

  // --------------------------------------------------
  // STEP 2 — SEMANTIC EVIDENCE GATE
  // --------------------------------------------------

  const requiresSemanticEvidence =
    deterministicResult
      ?.assignmentEvidence
      ?.requiresSemanticInference ===
    true;

  // --------------------------------------------------
  // STEP 3 — DETERMINISTIC FINAL RESULT
  // --------------------------------------------------

  if (!requiresSemanticEvidence) {
    return {
      ...deterministicResult,

      raw:
        cleanText(rawAssignment),

      studentSummary:
        "",

      understanding:
        "",

      reasoningType:
        "",

      confidence:
        "low",

      validationSource:
        "deterministic",
    };
  }

  // --------------------------------------------------
  // STEP 4 — BOUNDED SEMANTIC EVIDENCE
  // --------------------------------------------------

  const semanticEvidence =
    await getAssignmentUnderstandingSemanticEvidence(
      rawAssignment
    );

  // --------------------------------------------------
  // STEP 5 — JAVASCRIPT GOVERNANCE DECISION
  //
  // All three AUV gates must be established.
  //
  // AI confidence alone can never authorize progression.
  // --------------------------------------------------

  const assignmentUnderstandingEstablished =
    semanticEvidence
      .assignmentContextEstablished ===
      true &&

    semanticEvidence
      .assignmentDemandEstablished ===
      true &&

    semanticEvidence
      .sharedSummaryReady ===
      true &&

    !!semanticEvidence
      .studentSummary &&

    !!semanticEvidence
      .understanding &&

    semanticEvidence
      .confidence >= 0.85;

  // --------------------------------------------------
  // STEP 6 — GOVERNED ACCEPTANCE
  //
  // A valid result authorizes only the confirmation
  // checkpoint. Instruction has not begun yet.
  // --------------------------------------------------

  if (assignmentUnderstandingEstablished) {
    return {
      valid:
        true,

      raw:
        cleanText(rawAssignment),

      studentSummary:
        semanticEvidence
          .studentSummary,

      understanding:
        semanticEvidence
          .understanding,

      reasoningType:
        semanticEvidence
          .reasoningType,

      confidence:
        "high",

      needsClarification:
        false,
      
      assignmentEvidenceLevel:
        "substantive",

      assignmentCriteriaStatus:
        "satisfied",

      assignmentContextStatus:
        "established",

      assignmentDemandStatus:
        "established",

      summaryReadinessStatus:
        "ready",

      diagnosis:
        null,

      assignmentEvidence: {
        assignmentContextEstablished:
          true,

        assignmentDemandEstablished:
          true,

        sharedSummaryReady:
          true,

        semanticConfidence:
          semanticEvidence
            .confidence,

        semanticEvidenceSource:
          semanticEvidence
            .source,

        readerInferenceRequired:
          false,
      },

      validationSource:
        "deterministicWithSemanticEvidence",
    };
  }

  // --------------------------------------------------
  // STEP 7 — GOVERNED CLARIFICATION
  //
  // JavaScript identifies the first unestablished gate.
  //
  // This diagnosis will later support increasingly
  // targeted clarification contracts.
  // --------------------------------------------------

  let diagnosis =
    "assignmentUnderstandingNotEstablished";

  if (
    semanticEvidence
      .assignmentContextEstablished ===
    false
  ) {
    diagnosis =
      "assignmentContextNotEstablished";
  } else if (
    semanticEvidence
      .assignmentDemandEstablished ===
    false
  ) {
    diagnosis =
      "assignmentDemandNotEstablished";
  } else if (
    semanticEvidence
      .sharedSummaryReady ===
    false
  ) {
    diagnosis =
      "sharedSummaryNotReady";
  }

  return {
    valid:
      false,

    raw:
      cleanText(rawAssignment),

    studentSummary:
      "",

    understanding:
      "",

    reasoningType:
      semanticEvidence
        .reasoningType,

    confidence:
      "low",

    confirmed:
      false,

    assignmentEvidenceLevel:
      "substantive",

    assignmentCriteriaStatus:
      "partiallySatisfied",

    assignmentContextStatus:
      semanticEvidence
        .assignmentContextEstablished
          ? "established"
          : "notEstablished",

    assignmentDemandStatus:
      semanticEvidence
        .assignmentDemandEstablished
          ? "established"
          : "notEstablished",

    summaryReadinessStatus:
      semanticEvidence
        .sharedSummaryReady
          ? "ready"
          : "notReady",

    diagnosis,

    assignmentEvidence: {
      assignmentContextEstablished:
        semanticEvidence
          .assignmentContextEstablished,

      assignmentDemandEstablished:
        semanticEvidence
          .assignmentDemandEstablished,

      sharedSummaryReady:
        semanticEvidence
          .sharedSummaryReady,

      semanticConfidence:
        semanticEvidence
          .confidence,

      semanticEvidenceSource:
        semanticEvidence
          .source,

      readerInferenceRequired:
        true,
    },

    validationSource:
      "deterministicWithSemanticEvidence",
  };
}


// ------------------------------------------------------
// ASSIGNMENT UNDERSTANDING SUFFICIENCY
//
// Reads only the governed AUV result.
//
// This helper does not independently interpret assignment
// language. It reads only the governed AUV result.
// ------------------------------------------------------

function hasSufficientAssignmentUnderstanding(
  state
) {
  const context =
    state?.frameMeta
      ?.assignmentContext || {};

  return (
    context.valid === true &&
    context.assignmentContextStatus ===
      "established" &&
    context.assignmentDemandStatus ===
      "established" &&
    context.summaryReadinessStatus ===
      "ready"
  );
}

// ------------------------------------------------------
// ASSIGNMENT UNDERSTANDING UPDATE
//
// Accumulates clarification evidence before governed
// validation.
//
// A clarification response supplements the original
// assignment description rather than replacing it.
//
// Thinking Task inference occurs only after the governed
// AUV result has been stored.
// ------------------------------------------------------

async function updateAssignmentUnderstanding(
  state,
  rawAssignment
) {
  const currentContext =
    state?.frameMeta
      ?.assignmentContext || {};

  const newEvidence =
    cleanText(rawAssignment);

  const existingEvidence =
    cleanText(
      currentContext.raw || ""
    );

   const shouldAccumulateEvidence =
    !!existingEvidence &&
    !hasSufficientAssignmentUnderstanding(
      state
  );

  const accumulatedAssignment =
    shouldAccumulateEvidence
      ? cleanText(
          `${existingEvidence} ${newEvidence}`
        )
      : newEvidence;

  const understanding =
    await validateAssignmentUnderstandingGoverned(
      accumulatedAssignment
    );

  const previousClarificationCount =
    Number(
      currentContext
        .clarificationCount || 0
    );

  understanding.clarificationCount =
    shouldAccumulateEvidence
      ? previousClarificationCount + 1
      : previousClarificationCount;

  state.frameMeta.assignmentContext =
    understanding;

  // Thinking Task inference remains downstream from AUV.
  //
  // The inferred task may be displayed at the shared
  // confirmation checkpoint, but no instruction begins
  // until the student confirms the assignment summary.
  state.assignmentReasoning =
    inferThinkingTask(state);

  state.assignmentReasoning.lastUpdated =
    Date.now();

  console.log(
    "🧠 ASSIGNMENT UNDERSTANDING"
  );

  console.log(
    "--------------------------"
  );

  console.log(
    "Valid:",
    understanding.valid
  );

  console.log(
    "Context:",
    understanding
      .assignmentContextStatus
  );

  console.log(
    "Demand:",
    understanding
      .assignmentDemandStatus
  );

  console.log(
    "Summary:",
    understanding
      .summaryReadinessStatus
  );

  console.log(
    "Diagnosis:",
    understanding.diagnosis ||
      "None"
  );

  console.log(
    "Validation Source:",
    understanding.validationSource
  );

  console.log("");
  console.log(
    "🧠 Assignment Reasoning"
  );

  console.log(
    "----------------------"
  );

  console.log(
    "Task:",
    state.assignmentReasoning
      ?.task || "None"
  );

  console.log(
    "Label:",
    state.assignmentReasoning
      ?.label || "None"
  );

  console.log(
    "Confidence:",
    state.assignmentReasoning
      ?.confidence ?? 0
  );

  console.log(
    "Evidence:",
    state.assignmentReasoning
      ?.evidence || []
  );

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
// 2. keyTopic
// 3. isAbout
// 4. mainIdeas
// 5. details (per Main Idea)
// 6. soWhat
// 7. refine
//
// NOTE:
// Parent Anchor stages map onto these later via the
// Parent Anchor Bridge, but this engine remains the
// single source of truth for frame progression.

function getStage(state) {
  const f = state.frame;
  const m = state.frameMeta || {};
  const ideas = getIdeaList(state);

  if (!m.assignmentContext?.raw) return "assignmentContext";
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

  // Overlay pending states are helper flows, not structural stages.
  // They should be interpreted around the current structural stage.
  overlayPendingTypes: new Set([
    "confirmLanguageSwitch",
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

function getParentAnchorObservation(state) {
  const context =
    getParentAnchorContext(state);

  const ownerLabel =
    context.ownerStructuralStage;

  const stageLabel =
    context.structuralStage;

  return {
    ...context,

    ownerLabel,

    stageLabel,

    summary:
      `${context.ownerStructuralStage} | ${context.loopType} | ${ownerLabel}`,
  };
}

// ---------------------
// STATE
// ---------------------
function defaultState() {
return {
  version: 2,

  interactionMode: "build",

strengthenContext: {
  targetComponent: "",

  keyTopic: "",

  isAbout: "",

  currentMainIdea: "",

  supportingMainIdea: "",

  currentEssentialDetail: "",

  completionTarget:
    "strengthenComponentComplete",
},
  
  frameMeta: {
    assignmentContext: {
        raw: "",
        understanding: "",
        confidence: "low",
        childAnchor: "",
        clarificationCount: 0,
    },
},

    frame: {
      keyTopic: "",
      isAbout: "",
      parentItems: [],
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
  const s =
    raw &&
    typeof raw === "object"
      ? raw
      : {};

  const base =
    defaultState();

  base.interactionMode =
    s.interactionMode ||
    "build";

    const strengthenContext =
    s.strengthenContext &&
    typeof s.strengthenContext ===
      "object"
      ? s.strengthenContext
      : {};

  base.strengthenContext = {
    targetComponent:
      cleanText(
        strengthenContext
          .targetComponent ||
        ""
      ),

    keyTopic:
      cleanText(
        strengthenContext
          .keyTopic ||
        ""
      ),

    isAbout:
      cleanText(
        strengthenContext
          .isAbout ||
        ""
      ),

    currentMainIdea:
      cleanText(
        strengthenContext
          .currentMainIdea ||
        ""
      ),

    supportingMainIdea:
      cleanText(
        strengthenContext
          .supportingMainIdea ||
        ""
      ),

    currentEssentialDetail:
      cleanText(
        strengthenContext
          .currentEssentialDetail ||
        ""
      ),

    mainIdeas:
  Array.isArray(
    strengthenContext
      .mainIdeas
  )
    ? strengthenContext
        .mainIdeas
        .map(cleanText)
        .filter(Boolean)
    : [],

    currentSoWhat:
  cleanText(
    strengthenContext
      .currentSoWhat ||
    ""
  ),
    
    completionTarget:
      cleanText(
        strengthenContext
          .completionTarget ||
        "strengthenComponentComplete"
      ),
  };

    const assignmentReasoning =
    s.assignmentReasoning &&
    typeof s.assignmentReasoning ===
      "object"
      ? s.assignmentReasoning
      : {};

  base.assignmentReasoning = {
    task:
      assignmentReasoning.task ||
      null,

    label:
      cleanText(
        assignmentReasoning.label ||
        ""
      ),

    confidence:
      Number.isFinite(
        Number(
          assignmentReasoning.confidence
        )
      )
        ? Number(
            assignmentReasoning.confidence
          )
        : 0,

    evidence:
      Array.isArray(
        assignmentReasoning.evidence
      )
        ? assignmentReasoning.evidence
            .map(cleanText)
            .filter(Boolean)
        : [],

    lastUpdated:
      assignmentReasoning.lastUpdated ||
      null,
  };

  const frame =
    s.frame &&
    typeof s.frame === "object"
      ? s.frame
      : {};

  base.frame.keyTopic =
    cleanFrameText(
      frame.keyTopic ||
      ""
    )
      .replace(/[.!?]$/, "");

  base.frame.isAbout =
    cleanFrameText(
      frame.isAbout ||
      ""
    );

  base.frame.parentItems =
    Array.isArray(
      frame.parentItems
    )
      ? frame.parentItems
          .map(cleanText)
          .filter(Boolean)
      : [];

  base.frame.details =
    Array.isArray(
      frame.details
    )
      ? frame.details.map(
          (bucket) =>
            Array.isArray(bucket)
              ? bucket
                  .map(cleanText)
                  .filter(Boolean)
              : []
        )
      : [];

  base.frame.soWhat =
    cleanText(
      frame.soWhat ||
      ""
    );
    
  const frameMeta = s.frameMeta && typeof s.frameMeta === "object" ? s.frameMeta : {};

  const assignmentContext =
  frameMeta.assignmentContext && typeof frameMeta.assignmentContext === "object"
    ? frameMeta.assignmentContext
    : {};

base.frameMeta.assignmentContext = {
  valid:
    assignmentContext.valid === true,

  raw:
    cleanText(
      assignmentContext.raw || ""
    ),

  understanding:
    cleanText(
      assignmentContext.understanding || ""
    ),

  studentSummary:
    cleanText(
      assignmentContext.studentSummary || ""
    ),

  reasoningType:
    cleanText(
      assignmentContext.reasoningType || ""
    ),

  confidence:
    cleanText(
      assignmentContext.confidence || "low"
    ) || "low",

  confirmed:
    assignmentContext.confirmed === true,

  assignmentEvidenceLevel:
    cleanText(
      assignmentContext
        .assignmentEvidenceLevel || "none"
    ) || "none",

  assignmentCriteriaStatus:
    cleanText(
      assignmentContext
        .assignmentCriteriaStatus ||
      "notSatisfied"
    ) || "notSatisfied",

  assignmentContextStatus:
    cleanText(
      assignmentContext
        .assignmentContextStatus ||
      "undetermined"
    ) || "undetermined",

  assignmentDemandStatus:
    cleanText(
      assignmentContext
        .assignmentDemandStatus ||
      "undetermined"
    ) || "undetermined",

  summaryReadinessStatus:
    cleanText(
      assignmentContext
        .summaryReadinessStatus ||
      "notReady"
    ) || "notReady",

  diagnosis:
    cleanText(
      assignmentContext.diagnosis ||
      "emptyAssignmentEvidence"
    ),

  assignmentEvidence:
    assignmentContext
        .assignmentEvidence &&
      typeof assignmentContext
        .assignmentEvidence === "object"
      ? structuredClone(
          assignmentContext.assignmentEvidence
        )
      : null,

  validationSource:
    cleanText(
      assignmentContext.validationSource ||
      "deterministic"
    ) || "deterministic",

  childAnchor:
    cleanText(
      assignmentContext.childAnchor || ""
    ),

  clarificationCount:
    Number.isFinite(
      Number(
        assignmentContext.clarificationCount
      )
    )
      ? Number(
          assignmentContext.clarificationCount
        )
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


async function applyIsAboutCapture(
  s,
  msg,
  options = {}
) {
  const captureMode =
    options.captureMode || "build";

  const isStrengthen =
    captureMode === "strengthen";

  const cleanedIsAbout =
    cleanFrameText(msg);

  const keyTopic =
    cleanText(s.frame?.keyTopic || "");

  const keyTopicPrefix =
    `${keyTopic} is about `;

  const normalizedIsAbout =
    cleanedIsAbout
      .toLowerCase()
      .startsWith(
        keyTopicPrefix.toLowerCase()
      )
        ? cleanedIsAbout.slice(
            keyTopicPrefix.length
          )
        : cleanedIsAbout;

    const validation =
    await validateIsAboutResponseGoverned(
      normalizedIsAbout,
      keyTopic
    );

  const instructionalFinding =
    buildComponentInstructionalFinding({
      frameComponent:
        "isAbout",

      validation,

      evidence: {
        keyTopic:
          s.frame?.keyTopic || "",

        attemptedIsAbout:
          cleanText(msg),

        normalizedIsAbout:
          cleanText(
            normalizedIsAbout
          ),
      },
    });

  refreshInstructionalSituationWithComponentFinding({
    state:
      s,

    currentResponse:
      normalizedIsAbout,

    componentFinding:
      instructionalFinding,
  });

  const progressionAuthorization =
    buildProgressionAuthorization(
      s,
      {
        frameComponent:
          "isAbout",

        expectedContractId:
          "IA-RTP-001",
      }
    );

  s.progressionAuthorization =
    structuredClone(
      progressionAuthorization
    );

    if (
    !validation.valid ||
    progressionAuthorization
      ?.authorized !== true
  ) {
      
        const instructionalContract =
      s?.instructionalContractSelection
        ?.selectedContract ||
      null;

  const activationState = {
    ...s,

      pending: {
  type:
    isStrengthen
      ? "strengthenReviseIsAbout"
      : "reviseIsAbout",

  captureMode,

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
  type:
    isStrengthen
      ? "strengthenReviseIsAbout"
      : "reviseIsAbout",

  captureMode,

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
};
      
return s;
}

s.frame.isAbout =
  cleanFrameText(normalizedIsAbout);

s.pending =
  isStrengthen
    ? {
        type:
          "strengthenComponentComplete",

        component:
          "isAbout",

        componentLabel:
          "Is About statement",

        completedWork:
          s.frame.isAbout || "",

        revisePendingType:
          "strengthenReviseIsAbout",

         successMessage:
          "💬 Nice work! Your Is About clearly explains what your whole Key Topic is about.",
        
        displayIcon:
          "💬",
      
        displayLabel:
          "Is About",
      }
    : {
        type:
          "confirmIsAbout",
      };

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

  const isStrengthen =
    captureMode === "strengthen" &&
    Number.isInteger(revisionIndex);

  const validation =
    await validateMainIdeaResponseGoverned(
      text,
      s.frame?.keyTopic || "",
      s.frame?.isAbout || ""
    );

  const instructionalFinding =
    buildComponentInstructionalFinding({
      frameComponent:
        "mainIdeas",

      validation,

      evidence: {
        keyTopic:
          s.frame?.keyTopic || "",

        isAbout:
          s.frame?.isAbout || "",

        attemptedMainIdea:
          text,

        captureMode,

        revisionIndex,
      },
    });

  refreshInstructionalSituationWithComponentFinding({
    state:
      s,

    currentResponse:
      text,

    componentFinding:
      instructionalFinding,
  });

  const progressionAuthorization =
  buildProgressionAuthorization(
    s,
    {
      frameComponent:
        "mainIdeas",

      expectedContractId:
        "MI-RTP-001",
    }
  );

s.progressionAuthorization =
  structuredClone(
    progressionAuthorization
  );
  
if (
  !validation.valid ||
  progressionAuthorization
    ?.authorized !== true
) {
  
let pendingLocation;

if (isStrengthen) {
    pendingLocation = {
        type:
            "strengthenCurrentMainIdea",

        index:
            revisionIndex,
    };
} else if (isRevision) {
    pendingLocation = {
        type:
            "reviseMainIdeaAt",

        index:
            revisionIndex,
    };
} else if (isOptional) {
    pendingLocation = {
        type:
            "collectAnotherMainIdea",
    };
} else {
    pendingLocation = {
        type:
            "collectAnotherMainIdea",
    };
}
  
const instructionalContract =
  s?.instructionalContractSelection
    ?.selectedContract ||
  null;

const activationState = {
  ...s,

  pending: {
    ...pendingLocation,

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
  ...pendingLocation,

  instructionalFinding,

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
};

return s;
}

  // Preserve the existing Build Mode lane guardrail.
  //
  // Governed component validation determines whether the
  // response functions as a Main Idea.
  //
  // The existing lane check may still enforce specialized
  // frame-type behavior without replacing governed
  // validation.

// --------------------------------------------------
// CANONICAL MAIN IDEA STATE MUTATION
//
// frame.parentItems is the authoritative runtime
// collection for every Main Idea.
// --------------------------------------------------

  if (
    !Array.isArray(
      s.frame.parentItems
    )
  ) {
    s.frame.parentItems = [];
  }

  if (
    !Array.isArray(
      s.frame.details
    )
  ) {
    s.frame.details = [];
  }

    if (isStrengthen) {
    if (
      s.frame.parentItems[
        revisionIndex
      ] !== undefined
    ) {
      s.frame.parentItems[
        revisionIndex
      ] = text;
    }

    s.strengthenContext
      .currentMainIdea =
      text;

  s.pending = {
  type:
    "strengthenComponentComplete",

  component:
    "mainIdeas",

  componentLabel:
    "Main Idea",

  completedWork:
    s.strengthenContext
      ?.currentMainIdea ||
    s.frame
      ?.parentItems
      ?.[revisionIndex] ||
    "",

  revisePendingType:
    "strengthenCurrentMainIdea",

  successMessage:
    "💡 You got it! Your Main Idea captures a key idea that your Essential Details can help explain.",

  displayIcon:
    "💡",

  displayLabel:
    "Main Idea",

  index:
    revisionIndex,

   instructionalFinding:
    structuredClone(
      instructionalFinding
    ),
};

    return s;
  }

  if (isRevision) {
    if (
      s.frame.parentItems[
        revisionIndex
      ] !== undefined
    ) {
      s.frame.parentItems[
        revisionIndex
      ] = text;
    }

    s.pending = {
      type:
        "confirmMainIdeas",
    };

    return s;
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

  const paContext =
    getParentAnchorContext(s);

  // Parent Anchor stage is part of normal runtime
  // progression and must remain available throughout
  // computeNextQuestion().
  //
  // The observation hook below may inspect this value,
  // but it does not own or create it.
  const paStage =
    paContext.ownerStructuralStage;
  
  // ---------------------
  // PARENT ANCHOR OBSERVATION HOOK (SANDBOX ONLY)
  // ---------------------
  // Leave this disabled until you are intentionally
  // validating sandbox flows.
  //
  // This hook exists so Parent Anchor can explain the
  // engine in motion without becoming part of the engine.
  //
  // Gated sandbox-only observation:
  if (
    s?.settings?.debugParentAnchor
  ) {
    const paObs =
      getParentAnchorObservation(s);

    const isInDetails =
      paStage === "detailsLoop";

    const stage =
      s.pending?.stage ||
      getStage(s);

    const baseStage =
      getBaseStage(stage);

    const engineIsDetails =
      baseStage === "details";

    const isAligned =
      isInDetails ===
      engineIsDetails;

    console.log(
      "[PA OBS]",
      paObs.summary,
      {
        isInDetails,
        engineIsDetails,
        isAligned,
        ...paObs,
      }
    );
  }

  if (
  s.pending?.type ===
  "confirmAssignmentUnderstanding"
) {
  const assignment =
    s.frameMeta
      ?.assignmentContext
      ?.studentSummary ||
    s.frameMeta
      ?.assignmentContext
      ?.understanding ||
    s.frameMeta
      ?.assignmentContext
      ?.raw ||
    "your assignment";

  const thinkingTask =
    s.assignmentReasoning
      ?.label ||
    s.assignmentReasoning
      ?.task ||
    "Organize thinking";

    const displayedAssignment =
    assignment.charAt(0).toUpperCase() +
    assignment.slice(1);

    return (
    "🧠 Here's what I understand about your assignment:\n\n" +
    `${displayedAssignment}\n\n` +
    `🎯 Thinking Task: ${thinkingTask}\n\n` +
    "Does this accurately capture what your assignment is asking you to do?\n\n" +
    "1) Yes — That is accurate.\n" +
    "2) Not quite — I need to clarify something.\n\n" +
    "Reply with 1 or 2."
  );
}

   if (
    s.pending?.type ===
    "assignmentReasoningIntro"
  ) {
    return (
      "✨ Great—we have a shared understanding of your assignment!\n\n" +
      "🎯 How can I support your thinking today?\n\n" +
      "🛠️  1. Build a new Frame\n" +
      "    Start a new Framing Routine one step at a time.\n\n" +
      "🔧  2. Strengthen an existing Frame\n" +
      "    Improve one part of a Frame you've already started.\n\n" +
      "Reply with 1 or 2."
    );
  }

if (
  s.pending?.type ===
  "strengthenComponentSelection"
) {
  return (
    "🔧 Which part of your Frame would you like to strengthen?\n\n" +
    "1) 💬 Is About\n" +
    "   Strengthen how you explain what your Key Topic is about.\n\n" +
    "2) 💡 Main Idea\n" +
    "   Strengthen one Main Idea that helps explain your Key Topic.\n\n" +
    "3) ✍️ Essential Detail\n" +
    "   Strengthen one Essential Detail that helps explain a Main Idea.\n\n" +
    "4) 🎯 So What\n" +
    "   Strengthen what is important to understand from your whole Frame.\n\n" +
    "Reply with 1, 2, 3, or 4."
  );
}

if (
  s.pending?.type ===
  "strengthenCurrentTopicContext"
) {
  return (
    "🧭 Before we strengthen this part, remind me what your Frame is about.\n\n" +
    "Please share both parts:\n\n" +
    "🧩 Key Topic:\n" +
    "💬 Is About:"
  );
}

  if (
  s.pending?.type ===
  "strengthenReviseIsAbout"
) {
  return (
    "💬 Is About\n\n" +
    "Let's strengthen your Is About so it clearly explains what your whole Key Topic is about in your own words.\n\n" +
    "What would you like it to say instead?"
  );
}

if (
  s.pending?.type ===
  "strengthenSoWhatMainIdeas"
) {
  return (
    "💡 Main Ideas\n\n" +
    "Share the Main Ideas from your Frame so we can look across your thinking before strengthening your So What."
  );
}

  if (
    s.pending?.type ===
    "strengthenCurrentSoWhat"
) {
  return (
    "🎯 So What\n\n" +
    "What does your current So What say?"
  );
}
  
  if (
  s.pending?.type ===
  "strengthenCurrentMainIdea"
) {
  return (
    "💡 Main Idea\n\n" +
    "What does the Main Idea you want to strengthen say?"
  );
}

  if (
    s.pending?.type ===
    "strengthenSupportingMainIdea"
  ) {
    return (
      "💡 Main Idea\n\n" +
      "What Main Idea does this Essential Detail help explain?"
);
}

 if (
    s.pending?.type ===
    "strengthenCurrentEssentialDetail"
) {
  return (
    "✍️ Essential Detail\n\nWhat does the Essential Detail you want to strengthen say?"
  );
}

if (
  s.pending?.type ===
  "strengthenSessionComplete"
) {
  return (
    "🎉 Great work today!\n\n" +
    "You took time to look closely at your " +
    s.pending.componentLabel +
    " and strengthen your thinking.\n\n" +
    "That kind of reflection helps you get better at using Frames on your own."
  );
}

  if (
  s.pending?.type ===
  "strengthenComponentComplete"
) {
  const completedWork =
    cleanText(
      s.pending?.completedWork || ""
    );

  return (
    s.pending.successMessage +
    "\n\n" +
    s.pending.displayIcon +
    " " +
    s.pending.displayLabel +
    "\n\n" +
    completedWork +
    "\n\n" +
    "What would you like to do next?\n\n" +
    "1) Keep it as written and end this session.\n" +
    "2) Strengthen it further.\n" +
    "3) Strengthen another part of my Frame.\n\n" +
    "Reply with 1, 2, or 3."
  );
}
  
  if (
  s.pending?.type ===
  "strengthenReadyForGovernedConnection"
) {
  const componentLabels = {
    keyTopic:
      "Key Topic",

    isAbout:
      "Is About",

    mainIdeas:
      "Main Idea",

    details:
      "Essential Detail",
  };

  const label =
    componentLabels[
      s.strengthenContext
        ?.targetComponent
    ] ||
    "Frame component";

  return (
    `➡️ Ready to strengthen your ${label}\n\n` +
    "We've got the context we need. Let's work on it together."
  );
}
  
if (
  s.pending?.type ===
  "confirmLanguageSwitch"
) {
  const candNative =
    s.pending?.candidateNativeName ||
    s.pending?.candidateName ||
    "that language";

  const candName =
    s.pending?.candidateName ||
    "that language";

  return (
    `🌐 I notice you're writing in ${candName}.\n\n` +
    `Would you like to continue in ${candNative}?\n\n` +
    `Reply yes or no.`
  );
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
  
if (s.pending?.type === "confirmIsAbout") {
  const isAboutDisplay =
    cleanText(
      s.frame.isAbout
  );

if (s.pending?.type === "reviseIsAbout") {
  return getComponentPrompt(
    "isAbout",
    "revisePrompt"
  );
}

  return getComponentPrompt("isAbout", "confirmationPrompt", {
    keyTopic: s.frame.keyTopic,
    isAbout: isAboutDisplay
  });
}
 
  if (s.pending?.type === "confirmMainIdeas") {
  const lines = getIdeaList(s)
    .map(
      (mainIdea, index) =>
        `Main Idea ${index + 1}: ${mainIdea}`
    )
    .join("\n");

  return getComponentPrompt(
    "mainIdeas",
    "confirmationPrompt",
    {
      mainIdeasList: lines,
    }
  );
}

if (
  s.pending?.type ===
  "chooseMainIdeaToRevise"
) {
  const lines = getIdeaList(s)
    .map(
      (mainIdea, index) =>
        `${index + 1}) Main Idea ${index + 1}: ${mainIdea}`
    )
    .join("\n");

  return (
    `💡 You've built these Main Ideas:\n\n` +
    `${lines}\n\n` +
    `Which one would you like to strengthen?\n\n` +
    `Reply with the number.`
  );
}

if (
  s.pending?.type ===
  "reviseMainIdeaAt"
) {
  const index =
    Number(s.pending.index);

  const currentMainIdea =
    getIdeaList(s)[index] || "";

  return (
    `💡 Main Idea ${index + 1}\n\n` +
    `"${currentMainIdea}"\n\n` +
    `Let's strengthen this Main Idea so it more clearly helps explain your Key Topic.\n\n` +
    `What would you change?`
);
}

if (
  s.pending?.type ===
  "chooseDetailToRevise"
) {
  const index =
    Number(s.pending.index);

  const details =
    Array.isArray(
      s.frame.details?.[index]
    )
      ? s.frame.details[index]
      : [];

  const lines = details
    .map(
      (detail, detailIndex) =>
        `${detailIndex + 1}) Essential Detail ${detailIndex + 1}: ${detail}`
    )
    .join("\n");

  return (
  `✍️ Essential Details\n\n` +
  `You've built:\n\n` +
  `${lines}\n\n` +
  `Which Essential Detail would you like to strengthen?\n\n` +
  `Reply with the number.`
);
}
  
if (
  s.pending?.type ===
  "reviseDetailAt"
) {
  const index =
    Number(s.pending.index);

  const detailIndex =
    Number(s.pending.detailIndex);

  const currentDetail =
    s.frame.details?.[index]
      ?.[detailIndex] || "";

    return (
    `✍️ Essential Detail ${detailIndex + 1}\n\n` +
    `"${currentDetail}"\n\n` +
    `Let's strengthen this Essential Detail so it adds specific information that helps explain your Main Idea.\n\n` +
    `What would you change?`
  );
}

if (
  s.pending?.type ===
  "offerAnotherMainIdea"
) {
  const count =
    getIdeaList(s).length;

  const ideaNoun =
    count === 1
      ? "Main Idea"
      : "Main Ideas";

  const helpVerb =
    count === 1
      ? "helps"
      : "help";

  return (
    `🙌 Nice work! You've built ${count} ${ideaNoun} that ${helpVerb} explain "${s.frame.keyTopic}".\n\n` +
    `Would you like to add another Main Idea?\n\n` +
    `1) Yes — Add another Main Idea.\n` +
    `2) No — Continue.\n\n` +
    `Reply with 1 or 2.`
  );
}

if (
  s.pending?.type ===
  "collectAnotherMainIdea"
) {
  return getComponentPrompt(
    "mainIdeas",
    "additionalPrompt",
    {
      keyTopic:
        s.frame.keyTopic,

      isAbout:
        s.frame.isAbout,
    }
  );
}

if (
  s.pending?.type ===
  "offerAnotherDetail"
) {
  const index =
    Number(s.pending.index);

  const detailCount =
    Array.isArray(
      s.frame.details?.[index]
    )
      ? s.frame.details[index].length
      : 0;

  const completionMessage =
    detailCount === 2
      ? `🙌 Nice work! You've added the two Essential Details needed to build out Main Idea ${index + 1}:\n`
      : `🙌 You're building this idea out! You now have ${detailCount} Essential Details for Main Idea ${index + 1}:\n`;

  const details =
    Array.isArray(
      s.frame.details?.[index]
    )
      ? s.frame.details[index]
      : [];

  const lines =
    details
      .map(
        (detail, detailIndex) =>
          `• Essential Detail ${detailIndex + 1}: ${detail}`
      )
      .join("\n");

  return (
    `${completionMessage}\n` +
    `${lines}\n\n` +
    `Would you like to add another Essential Detail to strengthen this Main Idea further?\n\n` +
    `1) Yes — Add another Essential Detail.\n` +
    `2) No — Continue.\n\n` +
    `Reply with 1 or 2.`
  );
}

if (
  s.pending?.type ===
  "collectAnotherDetail"
) {
  const index =
    Number(s.pending.index);

  const currentMainIdea =
    getIdeaList(s)[index] || "";

  const currentDetailCount =
    Array.isArray(
      s.frame.details?.[index]
    )
      ? s.frame.details[index].length
      : 0;

  const nextDetailNumber =
    currentDetailCount + 1;

  // The second Essential Detail is required to fully
  // support the current Main Idea.
  if (currentDetailCount === 1) {
  return (
    `🎉 Great start! You've added your first Essential Detail for "${currentMainIdea}".\n\n` +
    `✍️ Let's add one more Essential Detail to help explain this Main Idea more fully.\n\n` +
    `Main Idea ${index + 1}: "${currentMainIdea}"\n\n` +
    `What is Essential Detail ${nextDetailNumber}?`
  );
}

  const mainIdeas =
    getIdeaList(s);

  const hasNextMainIdea =
    index <
    mainIdeas.length - 1;

  const nextDestination =
    hasNextMainIdea
      ? `Main Idea ${index + 2}`
      : "your So What statement";

  return (
    `✍️ You can add another Essential Detail to strengthen this Main Idea further.\n\n` +
    `Main Idea ${index + 1}: "${currentMainIdea}"\n\n` +
    `What is Essential Detail ${nextDetailNumber}?\n\n` +
    `Or reply with 2 to review your Essential Details and continue to ${nextDestination}.`
  );
}

if (
  s.pending?.type ===
  "confirmDetails"
) {
  const index =
    Number(s.pending.index);

  const currentMainIdea =
    getIdeaList(s)[index] || "";

  const details =
    Array.isArray(
      s.frame.details?.[index]
    )
      ? s.frame.details[index]
      : [];

  const lines =
    details
      .map(
        (detail, detailIndex) =>
          `✍️ Essential Detail ${detailIndex + 1}: ${detail}`
      )
      .join("\n");

  return (
    `✅ Checkpoint\n\n` +
    `💡 Main Idea ${index + 1}: ${currentMainIdea}\n\n` +
    `✍️ Essential Details\n\n` +
    `${lines}\n\n` +
    `Does this accurately capture your thinking?\n\n` +
    `1) Yes — Continue building my Frame.\n` +
    `2) No — Revise one Essential Detail.\n\n` +
    `Reply with 1 or 2.`
  );
}
  
if (s.pending?.type === "offerMoreSoWhat") {
  return (
    `🎯 So What\n\n` +
    `"${s.frame.soWhat}"\n\n` +
    `Would you like to add another sentence to strengthen your So What?\n\n` +
    `1) Yes — Add another sentence.\n` +
    `2) No — Continue.\n\n` +
    `Reply with 1 or 2.`
  );
}

if (s.pending?.type === "collectMoreSoWhat") {
  return (
    `🎯 So What\n\n` +
    `"${s.frame.soWhat}"\n\n` +
    `What would you like to add to strengthen it further?`
  );
}

if (s.pending?.type === "confirmSoWhat") {
  if (s.pending?.awaitingRevision) {
    return (
      `🎯 So What\n\n` +
      `Let's strengthen your So What.\n\n` +
      `What would you like it to say instead?`
    );
  }

  return (
    `✅ Checkpoint\n\n` +
    `🎯 So What\n\n` +
    `"${s.frame.soWhat}"\n\n` +
    `Does this accurately capture what is important to understand from your Frame?\n\n` +
    `1) Yes — Keep my So What.\n` +
    `2) No — Revise my So What.\n\n` +
    `Reply with 1 or 2.`
  );
}

if (
  s.pending?.type ===
  "offerExport"
) {
  const mainIdeas =
    getIdeaList(s)
      .filter(Boolean);

  const frameLines = [
    "🎉 Your Frame is complete!",
    "",
    "Here’s the thinking you built:",
    "",
    `🧩 Key Topic`,
    `${s.frame.keyTopic}`,
    "",
    `💬 Is About`,
    `${s.frame.isAbout}`,
  ];

  mainIdeas.forEach(
    (mainIdea, index) => {
      frameLines.push(
        "",
        `💡 Main Idea ${index + 1}`,
        `${mainIdea}`
      );

      const details =
        Array.isArray(
          s.frame.details?.[index]
        )
          ? s.frame.details[index]
              .filter(Boolean)
          : [];

      details.forEach(
        (detail, detailIndex) => {
          frameLines.push(
            "",
            `✍️ Essential Detail ${detailIndex + 1}`,
            `${detail}`
          );
        }
      );
    }
  );

  frameLines.push(
    "",
    "🎯 So What",
    `${s.frame.soWhat}`,
    "",
    "You built this Frame one step at a time using your own thinking.",
    "",
    "What would you like to do next?",
    "",
    "1) Save or print my Frame.",
    "2) Finish without saving.",
    "",
    "Reply with 1 or 2."
  );

  return frameLines.join("\n");
}

if (
  s.pending?.type ===
  "chooseExportType"
) {
  return (
    "📄 What would you like to save or print?\n\n" +
    "1) My Frame\n" +
    "2) My conversation with Kaw\n" +
    "3) Both\n\n" +
    "Reply with frame, transcript, or both."
  );
}

  // Base progression
  if (!s.frameMeta?.assignmentContext?.raw) {
    return (
      "🌟 Welcome to Kaw Companion!\n\n" +
      "I'm excited to work with you today!\n\n" +
      "📚 Before we begin, tell me about the assignment you're working on.\n\n" +
      "❓ What are you being asked to do?"
  );
}

  if (!hasSufficientAssignmentUnderstanding(s)) {
    return (
      "🔎 I want to make sure I understand your assignment before we begin.\n\n" +
      "What does your assignment ask you to think about or accomplish?"
  );
}

if (!s.frame.keyTopic) {
  const assignment =
    s.frameMeta?.assignmentContext?.studentSummary ||
    s.frameMeta?.assignmentContext?.understanding ||
    s.frameMeta?.assignmentContext?.raw ||
    "your assignment";

  const displayedAssignment =
    assignment.charAt(0).toUpperCase() +
    assignment.slice(1);

  return (
    "🧩 Key Topic\n\n" +
    `${displayedAssignment}\n\n` +
    "What is the main topic you'll be exploring in this Frame?"
  );
}
 
if (!s.frame.isAbout) {
  const assignment =
    s.frameMeta?.assignmentContext?.studentSummary ||
    s.frameMeta?.assignmentContext?.understanding ||
    s.frameMeta?.assignmentContext?.raw ||
    "your assignment";

  const displayedAssignment =
    assignment.charAt(0).toUpperCase() +
    assignment.slice(1);

  return (
    `➡️ Next: Is About\n\n` +
    `${displayedAssignment}\n\n` +
    `🧩 Key Topic: "${s.frame.keyTopic}"\n\n` +
    `💬 Is About\n\n` +
    `Now describe your Key Topic in your own words.\n\n` +
    `What is "${s.frame.keyTopic}" about?`
  );
}
  
const ideas = getIdeaList(s);

if (paStage === "parentItems" || ideas.length < 2) {
  const c = ideas.length;

  const isAboutDisplay =
    cleanText(
      s.frame.isAbout
    );

  if (c === 0) {
    return (
      `➡️ Next: Main Ideas\n\n` +
      `So far your Frame says:\n\n` +
      `🧩 Key Topic: ${s.frame.keyTopic}\n` +
      `💬 Is About: ${isAboutDisplay}\n\n` +
      `💡 Main Idea 1\n\n` +
      `What is one Main Idea that helps explain your Key Topic?`
    );
  }

  return (
    `💡 Main Idea ${c + 1}\n\n` +
    `So far your Frame says:\n\n` +
    `🧩 Key Topic: ${s.frame.keyTopic}\n` +
    `💬 Is About: ${isAboutDisplay}\n\n` +
    `What is another Main Idea that helps explain your Key Topic?`
  );
}

  // DETAILS LOOP (CLEANED — no duplicate fallback / brace drift)
   for (let i = 0; i < ideas.length; i++) {
    const mi = ideas[i];
    const arr = Array.isArray(s.frame.details[i]) ? s.frame.details[i] : [];
    if (paStage === "detailsLoop" && arr.length < 2) {
      const detailNum = arr.length + 1; // 1 or 2

      const miLabel = "Main Idea";
      const dLabel = "Essential Detail";

const promptType =
  detailNum === 1
    ? "initialPrompt"
    : "additionalPrompt";

const fallback =
  getComponentPrompt(
    "details",
    promptType,
    {
      mainIdea: mi
    }
  );

if (i === 0 && detailNum === 1) {
  return (
    "💡 Great work! You've identified the Main Ideas that help explain your Key Topic.\n\n" +
    "➡️ Next: Essential Details\n\n" +
    `💡 ${miLabel} ${i + 1}\n` +
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
    `🙌 Nicely done! You've added the Essential Details that help explain your ${completedLabel} Main Idea.\n\n` +
    `💡 ${miLabel} ${i + 1}\n` +
    `${mi}\n\n` +
    `✍️ ${dLabel} ${detailNum}\n\n` +
    `${fallback}`
  );
}

return (
  `💡 ${miLabel} ${i + 1}\n` +
  `${mi}\n\n` +
  `✍️ ${dLabel} ${detailNum}\n\n` +
  `${fallback}`
);
}
}

if (!s.frame.soWhat) {
  const mainIdeas =
    getIdeaList(s)
      .filter(Boolean);

  return (
    `🙌 Great work! You've built your Main Ideas and Essential Details.\n\n` +
    `➡️ Next: So What\n\n` +
    `Look across your Frame:\n\n` +
    `🧩 Key Topic: ${s.frame.keyTopic}\n\n` +
    mainIdeas
      .map(
        (idea, index) =>
          `💡 Main Idea ${index + 1}: ${idea}`
      )
      .join("\n") +
    `\n\n🎯 So What\n\n` +
    `What is the most important thing someone should understand from your whole Frame?`
  );
}
  
// ---------------------
// STATE UPDATE (SSOT)
// ---------------------
async function updateStateFromStudent(state, message) {
  const msg = cleanText(message);
  const s = structuredClone(state);

  ensureBuckets(s);

  if (!s.frameMeta) {
    s.frameMeta = {
      assignmentContext: {
        raw: "",
        understanding: "",
        confidence: "low",
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
      childAnchor: "",
      clarificationCount: 0,
    };
  }

  // --------------------------------------------------
  // EVIDENCE STATE
  // --------------------------------------------------
  //
  // Every runtime cycle begins by organizing the current
  // response and accumulated instructional evidence.
  //
  // Evidence State is read-only. It does not validate,
  // select strategy, change progression, or mutate state.
  //
  // It is intentionally constructed before the runtime
  // begins interpreting or responding to the message.
  // --------------------------------------------------

  const observationReport =
  await buildObservationReport(
    s,
    msg
  );

// Store the current governed observation artifact so it
// remains inspectable during migration and testing.
s.observationReport =
  structuredClone(
    observationReport
  );

const evidenceState =
  buildEvidenceState(
    s,
    msg,
    observationReport
  );
  
// --------------------------------------------------
// INSTRUCTIONAL ASSESSMENT
// --------------------------------------------------
//
// Organizes current observable evidence into governed
// instructional findings.
//
// At the beginning of the request cycle, the assessment
// contains interaction evidence and the initial
// Instructional Situation.
//
// Component validation occurs later in the active runtime
// branch. After validation, the assessment and
// Instructional Situation are refreshed with the current
// Component Instructional Finding.
//
// Assessment does not itself save student work, control
// progression, or generate communication.
//
// --------------------------------------------------

const instructionalAssessment =
  buildInstructionalAssessment(
    evidenceState
  );

const interactionInstructionalFinding =
  buildInteractionInstructionalFinding(
    evidenceState,
    instructionalAssessment
  );

// Attach the deterministic Interaction Instructional
// Finding to the current assessment artifact.
//
// This finding interprets only the governed Observation
// Report within the student's current instructional
// location.
//
// It may establish whether the response functions only as
// an interaction, but it does not independently classify
// genuine struggle, select a contract, control progression,
// change pending state, or generate communication.

instructionalAssessment
  .interactionInstructionalFinding =
    structuredClone(
      interactionInstructionalFinding
    );

const instructionalSituation =
  buildInstructionalSituation({
    evidenceState,

    instructionalAssessment,

    // Current component validation occurs later in the
    // authoritative runtime pathway. Until that current
    // Component Finding is supplied, substantive responses
    // remain componentEvidenceRequiresValidation.
    componentFinding:
      null,

    relationshipFinding:
      null,
  });

// Attach the beginning-of-cycle Instructional Situation
// to the current assessment artifact.
//
// At this point, current component validation has not yet
// occurred, so substantive responses may remain in
// componentEvidenceRequiresValidation.
//
// After the active runtime branch completes component
// validation, the Instructional Situation is refreshed
// with the current Component Instructional Finding.
//
// The situation does not directly save student work or
// control progression. For migrated Frame components, its
// refreshed result supports deterministic contract
// selection and governed communication.

instructionalAssessment
  .instructionalSituation =
    structuredClone(
      instructionalSituation
    );

// Store the complete governed Instructional Assessment
// artifact for use by later runtime reasoning and for
// development verification.
s.instructionalAssessment =
  structuredClone(
    instructionalAssessment
  );

// Store the beginning-of-cycle Instructional Situation
// separately so it remains available until current
// component validation refreshes the governed situation.
s.instructionalSituation =
  structuredClone(
    instructionalSituation
  );
  
// --------------------------------------------------
// ASSIGNMENT UNDERSTANDING RUNTIME GATE
//
// All assignment evidence is routed through the governed
// Assignment Understanding Validator.
//
// The runtime does not independently interpret individual
// Assignment Context fields.
//
// When sufficient understanding is established, Kaw moves
// only to the shared confirmation checkpoint.
//
// When understanding remains insufficient, pending remains
// clear so the student may provide additional assignment
// evidence on the next turn.
// --------------------------------------------------

// Initial Assignment Understanding capture
if (
  !s.frameMeta.assignmentContext.raw &&
  !(s.pending && s.pending.type)
) {
  if (isStartupCommand(msg)) {
    return s;
  }

  await updateAssignmentUnderstanding(
    s,
    msg
  );

  if (
    hasSufficientAssignmentUnderstanding(
      s
    )
  ) {
    s.pending = {
      type:
        "confirmAssignmentUnderstanding",
    };
  }

  return s;
}


// Additional Assignment Understanding evidence
if (
  s.frameMeta.assignmentContext.raw &&
  !hasSufficientAssignmentUnderstanding(
    s
  ) &&
  !(s.pending && s.pending.type)
) {
  await updateAssignmentUnderstanding(
    s,
    msg
  );

  if (
    hasSufficientAssignmentUnderstanding(
      s
    )
  ) {
    s.pending = {
      type:
        "confirmAssignmentUnderstanding",
    };
  }

  return s;
}
  
  // ----------------
  // Pending handlers
  // ----------------

  if (
  s.pending?.type ===
  "confirmAssignmentUnderstanding"
) {
  const choice = msg
    .toLowerCase()
    .trim();

  if (
    choice === "1" ||
    choice === "yes" ||
    choice.includes("correct") ||
    choice.includes("accurate") ||
    choice.includes("right")
  ) {
    s.frameMeta.assignmentContext.confirmed =
      true;

    s.pending = {
      type:
        "assignmentReasoningIntro",
    };

    return s;
  }

  if (
    choice === "2" ||
    choice.includes("not") ||
    choice.includes("clarify") ||
    choice.includes("wrong")
  ) {
   s.frameMeta.assignmentContext.confirmed =
  false;

// The student rejected Kaw's shared summary.
//
// The existing assignment evidence remains preserved,
// but Shared Summary Readiness is no longer established.
// The next student response will be accumulated as
// additional assignment evidence and re-evaluated through
// the governed AUV.
s.frameMeta.assignmentContext.valid =
  false;

s.frameMeta.assignmentContext
  .summaryReadinessStatus =
  "notReady";

s.frameMeta.assignmentContext.diagnosis =
  "studentRejectedSharedSummary";

s.pending = null;

return s;
  }

  return s;
}
  
  if (
  s.pending?.type ===
  "assignmentReasoningIntro"
) {
  const choice =
    msg
      .toLowerCase()
      .trim();

  if (
    choice === "1" ||
    choice === "build" ||
    choice.includes(
      "build a new"
    ) ||
    choice.includes(
      "new frame"
    )
  ) {
    s.interactionMode =
      "build";

    s.pending =
      null;

    return s;
  }

  if (
    choice === "2" ||
    choice === "strengthen" ||
    choice.includes(
      "strengthen an existing"
    ) ||
    choice.includes(
      "existing frame"
    )
  ) {
    s.interactionMode =
      "strengthen";

    s.pending = {
      type:
        "strengthenComponentSelection",
    };

    return s;
  }

  return s;
}

// --------------------------------------------------
// IS ABOUT CONFIRMATION
//
// The accepted Is About has already passed governed
// validation and progression authorization.
//
// This checkpoint allows the student to either:
// • confirm the accepted Is About and continue; or
// • return to the same component to revise their own work.
//
// Confirmation does not revalidate, rewrite, or alter
// accepted student thinking.
// --------------------------------------------------

if (
  s.pending?.type ===
  "confirmIsAbout"
) {
  const choice =
    msg
      .toLowerCase()
      .trim();

  if (
    choice === "1" ||
    choice === "yes" ||
    choice.includes("continue") ||
    choice.includes("accurate") ||
    choice.includes("correct") ||
    choice.includes("right")
  ) {
    s.pending =
      null;

    return s;
  }

  if (
    choice === "2" ||
    choice === "no" ||
    choice.includes("revise") ||
    choice.includes("change") ||
    choice.includes("not quite")
  ) {
    s.pending = {
      type:
        "reviseIsAbout",
    };

    return s;
  }

  return s;
}

    if (
    s.pending?.type ===
    "strengthenComponentSelection"
  ) {
    const choice =
      msg
        .toLowerCase()
        .trim();

    let selectedComponent =
      null;

    if (
  choice === "1" ||
  choice === "is about" ||
  choice === "isabout" ||
  choice.includes(
    "is about"
  )
) {
  selectedComponent =
    "isAbout";
}

if (
  choice === "2" ||
  choice === "main idea" ||
  choice === "mainidea" ||
  choice.includes(
    "main idea"
  )
) {
  selectedComponent =
    "mainIdeas";
}

if (
  choice === "3" ||
  choice ===
    "essential detail" ||
  choice ===
    "essentialdetail" ||
  choice.includes(
    "essential detail"
  ) ||
  choice === "detail"
) {
  selectedComponent =
    "details";
}

if (
  choice === "4" ||
  choice === "so what" ||
  choice === "sowhat" ||
  choice.includes(
    "so what"
  )
) {
  selectedComponent =
    "soWhat";
}
    
    if (!selectedComponent) {
      return s;
    }

    s.strengthenContext = {
      targetComponent:
        selectedComponent,

      keyTopic:
        "",

      isAbout:
        "",

      currentMainIdea:
        "",

      supportingMainIdea:
        "",

      currentEssentialDetail:
        "",

      mainIdeas: [],

      currentSoWhat:
        "",

      completionTarget:
        "strengthenComponentComplete",
    };

    if (
      selectedComponent === "isAbout" ||
      selectedComponent === "mainIdeas" ||
      selectedComponent === "details" ||
      selectedComponent === "soWhat"
    ) {
        s.pending = {
          type:
              "strengthenCurrentTopicContext",

        targetComponent:
            selectedComponent,
    };
      return s;
    }

    return s;
    }
  
    if (
    s.pending?.type ===
    "strengthenCurrentTopicContext"
  ) {
    const topicContextText =
      String(msg || "").trim();

    const keyTopicMatch =
      topicContextText.match(
        /(?:^|\s)(?:🔑\s*)?key\s*topic\s*:\s*(.*?)(?=\s+(?:🧩\s*)?is\s*about\s*:|$)/i
      );

    const isAboutMatch =
      topicContextText.match(
        /(?:^|\s)(?:🧩\s*)?is\s*about\s*:\s*(.+)$/i
      );

    const currentKeyTopic =
      cleanText(
        keyTopicMatch?.[1] || ""
      )
        .replace(/[.!?]$/, "");

    const currentIsAbout =
      cleanText(
        isAboutMatch?.[1] || ""
      );

    if (
      !currentKeyTopic ||
      !currentIsAbout
    ) {
      return s;
    }

    s.strengthenContext
      .keyTopic =
      currentKeyTopic;

    s.strengthenContext
      .isAbout =
      currentIsAbout;

     const targetComponent =
  s.pending?.targetComponent ||
  s.strengthenContext
    ?.targetComponent ||
  "";
      
        if (
      targetComponent ===
      "isAbout"
    ) {
      // Hydrate the canonical Frame state so the existing
      // governed Is About architecture receives the same
      // evidence shape used by Build Mode.
      s.frame.keyTopic =
        currentKeyTopic;

      s.frame.isAbout =
        "";

      await applyIsAboutCapture(
        s,
        currentIsAbout,
        {
          captureMode:
            "strengthen",
        }
      );

      return s;
    }

    if (
      targetComponent ===
      "mainIdeas"
    ) {
      s.pending = {
        type:
          "strengthenCurrentMainIdea",
      };

      return s;
    }

    if (
      targetComponent ===
      "details"
    ) {
      s.pending = {
        type:
          "strengthenSupportingMainIdea",
      };

      return s;
    }

    if (
      targetComponent ===
      "soWhat"
    ) {
      s.pending = {
        type:
          "strengthenSoWhatMainIdeas",
      };
  
    return s;
  }
      
    return s;
  }

    if (
  s.pending?.type ===
  "strengthenCurrentSoWhat"
) {
  const currentSoWhat =
    cleanText(msg);

  if (!currentSoWhat) {
    return s;
  }

  s.strengthenContext
    .currentSoWhat =
    currentSoWhat;

  // ----------------------------------------------
  // Hydrate canonical Frame context so governed
  // So What validation receives the same evidence
  // shape used by Build Mode.
  // ----------------------------------------------

  s.frame.keyTopic =
    cleanText(
      s.strengthenContext
        ?.keyTopic || ""
    );

  s.frame.isAbout =
    cleanText(
      s.strengthenContext
        ?.isAbout || ""
    );

  s.frame.parentItems =
    Array.isArray(
      s.strengthenContext
        ?.mainIdeas
    )
      ? [
          ...s.strengthenContext
            .mainIdeas,
        ]
      : [];

  // Strengthen So What intentionally does not require
  // the student to re-enter Essential Details.
  //
  // Preserve the canonical shape expected by the
  // shared So What validator with empty detail buckets.
  s.frame.details =
    s.frame.parentItems.map(
      () => []
    );

  // The student's existing So What is the proposed
  // revision target. Do not save it to frame.soWhat
  // before governed validation.
  s.frame.soWhat =
    "";

  const soWhatValidation =
    await validateSoWhatResponseGoverned(
      currentSoWhat,
      buildSoWhatValidationContext(s)
    );

  const instructionalFinding = {
    ...buildComponentInstructionalFinding({
      frameComponent:
        "soWhat",

      validation:
        soWhatValidation,

      evidence: {
        keyTopic:
          s.frame?.keyTopic || "",

        isAbout:
          s.frame?.isAbout || "",

        mainIdeas:
          getIdeaList(s)
            .filter(Boolean),

        details:
          Array.isArray(
            s.frame?.details
          )
            ? s.frame.details.map(
                (bucket) =>
                  Array.isArray(bucket)
                    ? bucket.filter(Boolean)
                    : []
              )
            : [],

        attemptedSoWhat:
          currentSoWhat,

        captureMode:
          "strengthen",
      },
    }),

    synthesisState:
      soWhatValidation
        .synthesisState || null,

    validationSource:
      soWhatValidation
        .validationSource || null,

    captureMode:
      "strengthen",
  };

  refreshInstructionalSituationWithComponentFinding({
    state:
      s,

    currentResponse:
      currentSoWhat,

    componentFinding:
      instructionalFinding,
  });

  const progressionAuthorization =
    buildProgressionAuthorization(
      s,
      {
        frameComponent:
          "soWhat",

        expectedContractId:
          "SW-RTP-001",
      }
    );

  s.progressionAuthorization =
    structuredClone(
      progressionAuthorization
    );

  if (
    !soWhatValidation.valid ||
    progressionAuthorization
      ?.authorized !== true
  ) {
    return attachGovernedSupportToPending(
      s,
      msg,
      {
        intent:
          "stuck",

        confidence:
          1,

        source:
          `soWhatValidation:${soWhatValidation.diagnosis}`,

        instructionalFinding,
      }
    );
  }

  s.frame.soWhat =
    currentSoWhat;

   s.pending = {
  type:
    "strengthenComponentComplete",

  component:
    "soWhat",

  componentLabel:
    "So What",

  completedWork:
    currentSoWhat,

  revisePendingType:
    "strengthenCurrentSoWhat",

  successMessage:
    "🎯 Great work! Your So What captures what is important to understand after looking across your Frame.",

  displayIcon:
    "🎯",

  displayLabel:
    "So What",
};
  return s;
}

    if (
  s.pending?.type ===
  "strengthenSoWhatMainIdeas"
) {
  const mainIdeasText =
    String(msg || "").trim();

  const mainIdeas =
    mainIdeasText
      .split(/\n+/)
      .map((idea) =>
        cleanText(
          idea.replace(
            /^(?:[-•*]|\d+[.)])\s*/,
            ""
          )
        )
      )
      .filter(Boolean);

  if (mainIdeas.length === 0) {
    return s;
  }

  s.strengthenContext
    .mainIdeas =
    mainIdeas;

  s.pending = {
    type:
      "strengthenCurrentSoWhat",
  };

  return s;
}

    if (
    s.pending?.type ===
    "strengthenCurrentMainIdea"
  ) {
    const currentMainIdea =
      cleanText(msg);

    if (!currentMainIdea) {
      return s;
    }

    s.strengthenContext
      .currentMainIdea =
      currentMainIdea;

    // Hydrate the existing canonical Frame state so the
    // governed Main Idea architecture receives the same
    // evidence shape used by Build Mode.
    s.frame.keyTopic =
      s.strengthenContext
        ?.keyTopic ||
      "";

    s.frame.isAbout =
      s.strengthenContext
        ?.isAbout ||
      "";

    s.frame.parentItems = [
      currentMainIdea,
    ];

    s.frame.details = [
      [],
    ];

    // The student is strengthening existing work, so the
    // current Main Idea enters the governed architecture
    // as revision evidence rather than new Build evidence.

    await applyMainIdeaCapture(
      s,
      currentMainIdea,
      {
        captureMode:
          "strengthen",
    
        index:
          0,
  }
);

    return s;
  }

  if (
    s.pending?.type ===
    "strengthenSupportingMainIdea"
  ) {
    const supportingMainIdea =
      cleanText(msg);

    if (!supportingMainIdea) {
      return s;
    }

    s.strengthenContext
      .supportingMainIdea =
      supportingMainIdea;

    s.pending = {
      type:
        "strengthenCurrentEssentialDetail",
    };

    return s;
  }

   if (
  s.pending?.type ===
  "strengthenCurrentEssentialDetail"
) {
  const currentEssentialDetail =
    cleanText(msg);

  if (!currentEssentialDetail) {
    return s;
  }

  s.strengthenContext
    .currentEssentialDetail =
    currentEssentialDetail;

  const supportingMainIdea =
    cleanText(
      s.strengthenContext
        ?.supportingMainIdea || ""
    );

  // Hydrate the canonical Frame so the governed
  // Essential Detail engine receives the same context
  // shape used elsewhere in the runtime.
  s.frame.keyTopic =
    s.strengthenContext
      ?.keyTopic || "";

  s.frame.isAbout =
    s.strengthenContext
      ?.isAbout || "";

  s.frame.parentItems = [
    supportingMainIdea,
  ];

  s.frame.details = [
    [
      currentEssentialDetail,
    ],
  ];

  const {
    detailValidation,
    instructionalFinding,
    progressionAuthorization,
  } =
    await applyEssentialDetailCapture(
      s,
      currentEssentialDetail,
      {
        index:
          0,

        detailIndex:
          0,

        captureMode:
          "strengthen",
      }
    );

  if (
    !detailValidation.valid ||
    progressionAuthorization
      ?.authorized !== true
  ) {
    return attachGovernedSupportToPending(
      s,
      currentEssentialDetail,
      {
        intent:
          "stuck",

        confidence:
          1,

        source:
          `detailValidation:${detailValidation.diagnosis}`,

        instructionalFinding,
      }
    );
  }

 s.pending = {
  type:
    "strengthenComponentComplete",

  component:
    "details",

  componentLabel:
    "Essential Detail",

  completedWork:
    s.strengthenContext
      ?.currentEssentialDetail ||
    s.frame
      ?.details
      ?.[0]
      ?.[0] ||
    "",

  revisePendingType:
    "strengthenCurrentEssentialDetail",

  successMessage:
    "✍️ Nice thinking! Your Essential Detail adds important information that helps explain your Main Idea.",

displayIcon:
    "✍️",

  displayLabel:
    "Essential Detail",
};

  return s;
}

  if (
  s.pending?.type ===
  "strengthenComponentComplete"
) {
  const choice =
    cleanText(msg);

  if (choice === "1") {
    s.pending = {
      type:
        "strengthenSessionComplete",

      component:
        s.pending.component,

      componentLabel:
        s.pending.componentLabel,

      completedWork:
        s.pending.completedWork,
    };

    return s;
  }

  if (choice === "2") {
    const revisePendingType =
      s.pending.revisePendingType;

    s.pending = {
      type:
        revisePendingType,
    };

    return s;
  }

  if (choice === "3") {
    s.pending = {
      type:
        "strengthenComponentSelection",
    };

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

if (
  s.pending?.type ===
  "strengthenReviseIsAbout"
) {
  await applyIsAboutCapture(
    s,
    msg,
    {
      captureMode:
        "strengthen",
    }
  );

  return s;
}
  
if (s.pending?.type === "reviseIsAbout") {

  // All proposed responses proceed to governed Is About
  // validation.
  //
  // Conversational, meta, uncertainty, and struggle language
  // are identified as no component evidence by the governed
  // validator. They must not enter a separate recovery router.

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
  const normalized =
    msg.toLowerCase().trim();

  const idx =
    Number(s.pending.index);

  const arr =
    Array.isArray(
      s.frame.details[idx]
    )
      ? s.frame.details[idx]
      : [];

  const currentMainIdea =
    getIdeaList(s)[idx] || "";

  const detailIndex =
    arr.length;

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
  
  const detailValidation =
    await validateEssentialDetailResponseGoverned(
      msg,
      currentMainIdea,
      {
        keyTopic:
          s.frame.keyTopic || "",

        isAbout:
          s.frame.isAbout || "",
      }
    );

  const instructionalFinding = {
    ...buildComponentInstructionalFinding({
      frameComponent:
        "details",

      validation:
        detailValidation,

      evidence: {
        keyTopic:
          s.frame.keyTopic || "",

        isAbout:
          s.frame.isAbout || "",

        currentMainIdea,

        currentMainIdeaIndex:
          idx,

        currentDetailIndex:
          detailIndex,

        captureMode:
          "optionalDirectEntry",

        attemptedDetail:
          cleanText(msg),
      },
    }),

    validationSource:
      detailValidation.validationSource || null,

    currentMainIdea,

    currentMainIdeaIndex:
      idx,

    currentDetailIndex:
      detailIndex,

    captureMode:
      "optionalDirectEntry",
  };

  refreshInstructionalSituationWithComponentFinding({
    state:
      s,

    currentResponse:
      msg,

    componentFinding:
      instructionalFinding,
  });

  const progressionAuthorization =
    buildProgressionAuthorization(
      s,
      {
        frameComponent:
          "details",

        expectedContractId:
          "ED-RTP-001",
      }
    );

  s.progressionAuthorization =
    structuredClone(
      progressionAuthorization
    );

  if (
    !detailValidation.valid ||
    progressionAuthorization
      ?.authorized !== true
) {
  
  const instructionalContract =
    s?.instructionalContractSelection
      ?.selectedContract ||
    null;

  s.pending = {
    type:
      "collectAnotherDetail",

    index:
      idx,

    instructionalFinding,

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
  };

  return attachGovernedSupportToPending(
    s,
    msg,
    {
      intent:
        "stuck",

      confidence:
        1,

      source:
        `detailValidation:${detailValidation.diagnosis}`,

      instructionalFinding,
    }
  );
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
  s.pending = {
    type: "confirmDetails",
    index: idx,
  };

  return s;
}

// All proposed responses proceed to governed Essential
// Detail validation.
//
// Conversational, meta, uncertainty, and struggle language
// are identified as no component evidence by the governed
// validator. They must not enter a separate recovery router.

const currentMainIdea =
  getIdeaList(s)[idx] || "";

const detailValidation =
  await validateEssentialDetailResponseGoverned(
    msg,
    currentMainIdea,
    {
      keyTopic:
        s.frame.keyTopic || "",

      isAbout:
        s.frame.isAbout || "",
    }
  );

const currentDetailIndex =
  s.frame.details[idx].length;

const captureMode =
  currentDetailIndex < 2
    ? "required"
    : "optional";

const instructionalFinding = {
  ...buildComponentInstructionalFinding({
    frameComponent:
      "details",

    validation:
      detailValidation,

    evidence: {
      keyTopic:
        s.frame.keyTopic || "",

      isAbout:
        s.frame.isAbout || "",

      currentMainIdea,

      currentMainIdeaIndex:
        idx,

      currentDetailIndex,

      captureMode,

      attemptedDetail:
        cleanText(msg),
    },
  }),

  validationSource:
    detailValidation.validationSource || null,

  currentMainIdea,

  currentMainIdeaIndex:
    idx,

  currentDetailIndex,

  captureMode,
};

refreshInstructionalSituationWithComponentFinding({
  state:
    s,

  currentResponse:
    msg,

  componentFinding:
    instructionalFinding,
});

const progressionAuthorization =
  buildProgressionAuthorization(
    s,
    {
      frameComponent:
        "details",

      expectedContractId:
        "ED-RTP-001",
    }
  );

s.progressionAuthorization =
  structuredClone(
    progressionAuthorization
  );

  if (
    !detailValidation.valid ||
    progressionAuthorization
      ?.authorized !== true
) {
    
  // Preserve exactly what the deterministic validator
  // established about this response.
  //
  // Do not infer intent, understanding, confusion, or effort.
  // The finding describes only the observable instructional
  // condition of the response.

  return attachGovernedSupportToPending(
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

  async function applyEssentialDetailCapture(
  s,
  msg,
  {
    index = 0,
    detailIndex = 0,
    captureMode = "revision",
  } = {}
) {
  const currentMainIdea =
    getIdeaList(s)[index] || "";

  const detailValidation =
    await validateEssentialDetailResponseGoverned(
      msg,
      currentMainIdea,
      {
        keyTopic:
          s.frame.keyTopic || "",

        isAbout:
          s.frame.isAbout || "",
      }
    );

  const instructionalFinding = {
    ...buildComponentInstructionalFinding({
      frameComponent:
        "details",

      validation:
        detailValidation,

      evidence: {
        keyTopic:
          s.frame.keyTopic || "",

        isAbout:
          s.frame.isAbout || "",

        currentMainIdea,

        currentMainIdeaIndex:
          index,

        currentDetailIndex:
          detailIndex,

        captureMode,

        previousDetail:
          s.frame.details
            ?.[index]
            ?.[detailIndex] || "",

        attemptedDetail:
          cleanText(msg),
      },
    }),

    validationSource:
      detailValidation
        .validationSource || null,

    currentMainIdea,

    currentMainIdeaIndex:
      index,

    currentDetailIndex:
      detailIndex,

    captureMode,
  };

  refreshInstructionalSituationWithComponentFinding({
    state:
      s,

    currentResponse:
      msg,

    componentFinding:
      instructionalFinding,
  });

  const progressionAuthorization =
    buildProgressionAuthorization(
      s,
      {
        frameComponent:
          "details",

        expectedContractId:
          "ED-RTP-001",
      }
    );

  s.progressionAuthorization =
    structuredClone(
      progressionAuthorization
    );

  return {
    detailValidation,
    instructionalFinding,
    progressionAuthorization,
  };
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

// The proposed revision proceeds to governed Essential
// Detail validation before any accepted student work may
// be replaced.
//
// Conversational or no-evidence responses fail governed
// validation and preserve the existing Essential Detail.

  const {
  detailValidation,
  instructionalFinding,
  progressionAuthorization,
} =
  await applyEssentialDetailCapture(
    s,
    msg,
    {
      index:
        idx,

      detailIndex,

      captureMode:
        "revision",
    }
  );

  if (
    !detailValidation.valid ||
    progressionAuthorization
      ?.authorized !== true
) {
  return attachGovernedSupportToPending(
    s,
    msg,
    {
      intent:
        "stuck",

      confidence:
        1,

      source:
        `detailValidation:${detailValidation.diagnosis}`,

      instructionalFinding,
    }
  );
}

// Replace only the selected Essential Detail after the
// proposed revision has passed governed validation.
if (
  Array.isArray(s.frame.details[idx]) &&
  s.frame.details[idx][detailIndex] !== undefined
) {
  s.frame.details[idx][detailIndex] =
    msg;
}

  // Return to the Detail confirmation checkpoint.
  s.pending = { type: "confirmDetails", index: idx };
  return s;
}
  
  // --------------------------------------------------
  // SO WHAT OPTIONAL EXPANSION
  // --------------------------------------------------

  if (s.pending?.type === "offerMoreSoWhat") {
    const normalized =
      msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = {
        type: "collectMoreSoWhat",
      };

      return s;
    }

    s.pending = {
      type: "confirmSoWhat",
    };

    return s;
  }

  // --------------------------------------------------
  // SO WHAT ADDITIONAL CONTENT
  //
  // The student's existing So What and proposed additional
  // sentence are validated together before state mutation.
  // --------------------------------------------------

  if (s.pending?.type === "collectMoreSoWhat") {
    const normalized =
      msg.toLowerCase().trim();

    // A genuine decline preserves the existing So What.
    if (
      isNegative(normalized) ||
      normalized === "2"
    ) {
      s.pending = {
        type: "confirmSoWhat",
      };

      return s;
    }

// Additional proposed So What content proceeds to governed
// validation.
//
// No-evidence and conversational responses must be handled
// by the governed So What finding and contract pathway.

    const proposedSoWhat =
      cleanText(
        `${s.frame.soWhat} ${msg}`
      );

   const soWhatValidation =
  await validateSoWhatResponseGoverned(
    proposedSoWhat,
    buildSoWhatValidationContext(s)
  );

const instructionalFinding = {
  ...buildComponentInstructionalFinding({
    frameComponent:
      "soWhat",

    validation:
      soWhatValidation,

    evidence: {
      keyTopic:
        s.frame?.keyTopic || "",

      isAbout:
        s.frame?.isAbout || "",

      mainIdeas:
        getIdeaList(s)
          .filter(Boolean),

      details:
        Array.isArray(
          s.frame?.details
        )
          ? s.frame.details.map(
              (bucket) =>
                Array.isArray(bucket)
                  ? bucket.filter(Boolean)
                  : []
            )
          : [],

      previousSoWhat:
        s.frame?.soWhat || "",

      additionalContent:
        cleanText(msg),

      proposedSoWhat,

      captureMode:
        "additionalContent",
    },
  }),

  synthesisState:
    soWhatValidation
      .synthesisState || null,

  validationSource:
    soWhatValidation
      .validationSource || null,

  captureMode:
    "additionalContent",
};

refreshInstructionalSituationWithComponentFinding({
  state:
    s,

  currentResponse:
    proposedSoWhat,

  componentFinding:
    instructionalFinding,
});

const progressionAuthorization =
  buildProgressionAuthorization(
    s,
    {
      frameComponent:
        "soWhat",

      expectedContractId:
        "SW-RTP-001",
    }
  );

s.progressionAuthorization =
  structuredClone(
    progressionAuthorization
  );

if (
  !soWhatValidation.valid ||
  progressionAuthorization
    ?.authorized !== true
) {
      return attachGovernedSupportToPending(
        s,
        msg,
        {
          intent:
            "stuck",

          confidence:
            1,

          source:
            `soWhatValidation:${soWhatValidation.diagnosis}`,

          instructionalFinding,
        }
      );
    }

    // Save only after governed validation.
    s.frame.soWhat =
      proposedSoWhat;

    s.pending = {
      type: "confirmSoWhat",
    };

    return s;
  }

  // --------------------------------------------------
  // SO WHAT CONFIRMATION AND REVISION
  // --------------------------------------------------

  if (s.pending?.type === "confirmSoWhat") {
    const normalized =
      msg.toLowerCase().trim();

    if (isAffirmative(normalized)) {
      s.pending = null;

      if (
        isFrameComplete(s) &&
        !s.flags.exportOffered
      ) {
        s.flags.exportOffered = true;

        s.pending = {
          type: "offerExport",
        };
      }

      return s;
    }

  const mutationIntent =
  await classifyStudentWorkMutationIntent(
    s,
    msg
  );

// A decline or revision-direction command is not replacement
// So What content. Preserve the accepted student work and
// remain at the revision decision.
if (
  isNegative(normalized) ||
  normalized === "2" ||
  mutationIntent.intent ===
    "revision_direction"
) {
  s.pending = {
    type:
      "confirmSoWhat",

    awaitingRevision:
      true,
  };

  return s;
}

// All other proposed replacement wording proceeds to the
// governed So What validator.
//
// No-evidence responses fail validation and cannot replace
// the accepted So What.

    const previousSoWhat =
  s.frame?.soWhat || "";

const soWhatValidation =
  await validateSoWhatResponseGoverned(
    msg,
    buildSoWhatValidationContext(s)
  );

const instructionalFinding = {
  ...buildComponentInstructionalFinding({
    frameComponent:
      "soWhat",

    validation:
      soWhatValidation,

    evidence: {
      keyTopic:
        s.frame?.keyTopic || "",

      isAbout:
        s.frame?.isAbout || "",

      mainIdeas:
        getIdeaList(s)
          .filter(Boolean),

      details:
        Array.isArray(
          s.frame?.details
        )
          ? s.frame.details.map(
              (bucket) =>
                Array.isArray(bucket)
                  ? bucket.filter(Boolean)
                  : []
            )
          : [],

      previousSoWhat,

      attemptedSoWhat:
        cleanText(msg),

      captureMode:
        "revision",
    },
  }),

  synthesisState:
    soWhatValidation
      .synthesisState || null,

  validationSource:
    soWhatValidation
      .validationSource || null,

  captureMode:
    "revision",
};

refreshInstructionalSituationWithComponentFinding({
  state:
    s,

  currentResponse:
    msg,

  componentFinding:
    instructionalFinding,
});

const progressionAuthorization =
  buildProgressionAuthorization(
    s,
    {
      frameComponent:
        "soWhat",

      expectedContractId:
        "SW-RTP-001",
    }
  );

s.progressionAuthorization =
  structuredClone(
    progressionAuthorization
  );

 if (
  !soWhatValidation.valid ||
  progressionAuthorization
    ?.authorized !== true
) {   
  return attachGovernedSupportToPending(
    s,
    msg,
    {
      intent:
        "stuck",

      confidence:
        1,

      source:
        `soWhatValidation:${soWhatValidation.diagnosis}`,

      instructionalFinding,
    }
  );
}

// Replace only after governed validation.
    s.frame.soWhat =
      msg;
    
    s.pending = {
      type: "confirmSoWhat",
    };
    
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
      // Route extracted Is About content through governed capture.
      await applyIsAboutCapture(s, parsed.isAbout);
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
  
  if (
  isStuckMessage(cleaned) ||
  isWeakFrameResponse(cleaned) ||
  isMetaResponse(cleaned)
) {
  s.pending = {
    type: "reviseKeyTopic",
    feedback:
      "🧩 Key Topic\n\nThat’s okay—let’s keep it simple. Your Key Topic is the main subject of your assignment.\n\nWhat main topic are you exploring?",
  };

  return s;
}

s.pending = {
  type: "reviseKeyTopic",
  feedback:
    getKeyTopicFeedback(cleaned),
};

return s;
}

  // 3) Is About capture + checkpoint
  if (!s.frame.isAbout) {
    const lowered = msg.toLowerCase().trim();
    if (lowered !== "revise" && lowered !== "change") {
      await applyIsAboutCapture(s, msg);
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
        await validateEssentialDetailResponseGoverned(
          msg,
          currentMainIdea,
          {
            keyTopic:
              s.frame.keyTopic || "",

            isAbout:
              s.frame.isAbout || "",
          }
        );

      const instructionalFinding = {
        ...buildComponentInstructionalFinding({
          frameComponent:
            "details",

          validation:
            detailValidation,

          evidence: {
            keyTopic:
              s.frame.keyTopic || "",

            isAbout:
              s.frame.isAbout || "",

            currentMainIdea,

            currentMainIdeaIndex:
              i,

            currentDetailIndex:
              arr.length,

            captureMode:
              "required",

            attemptedDetail:
              cleanText(msg),
          },
        }),

        validationSource:
          detailValidation.validationSource ||
          null,

        currentMainIdea,

        currentMainIdeaIndex:
          i,

        currentDetailIndex:
          arr.length,

        captureMode:
          "required",
      };

      refreshInstructionalSituationWithComponentFinding({
        state:
          s,

        currentResponse:
          msg,

        componentFinding:
          instructionalFinding,
      });

      const progressionAuthorization =
        buildProgressionAuthorization(
          s,
          {
            frameComponent:
              "details",

            expectedContractId:
              "ED-RTP-001",
          }
        );

      s.progressionAuthorization =
        structuredClone(
          progressionAuthorization
        );

      if (
        !detailValidation.valid ||
        progressionAuthorization
          ?.authorized !== true
      ) {
        const instructionalContract =
          s?.instructionalContractSelection
            ?.selectedContract ||
          null;

        s.pending = {
          type:
            "collectAnotherDetail",

          index:
            i,

          instructionalFinding,

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
        };

        return attachGovernedSupportToPending(
          s,
          msg,
          {
            intent:
              "stuck",

            confidence:
              1,

            source:
              `detailValidation:${detailValidation.diagnosis}`,

            instructionalFinding,
          }
        );
      }

        s.frame.details[i] = [
          ...arr,
          msg,
        ];

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
}

   // --------------------------------------------------
  // 6) SO WHAT INITIAL CAPTURE
  //
  // The student's first proposed So What must pass
  // governed validation before it is saved.
  // --------------------------------------------------

  if (!s.frame.soWhat) {
    if (isNegative(msg)) {
      return s;
    }

// The student's proposed So What proceeds directly to
// governed validation.
//
// Conversational, meta, revision-direction, and struggle
// language produce no component evidence and must not be
// interpreted by a separate recovery router.

   const soWhatValidation =
  await validateSoWhatResponseGoverned(
    msg,
    buildSoWhatValidationContext(s)
  );

const instructionalFinding = {
  ...buildComponentInstructionalFinding({
    frameComponent:
      "soWhat",

    validation:
      soWhatValidation,

    evidence: {
      keyTopic:
        s.frame?.keyTopic || "",

      isAbout:
        s.frame?.isAbout || "",

      mainIdeas:
        getIdeaList(s)
          .filter(Boolean),

      details:
        Array.isArray(
          s.frame?.details
        )
          ? s.frame.details.map(
              (bucket) =>
                Array.isArray(bucket)
                  ? bucket.filter(Boolean)
                  : []
            )
          : [],

      attemptedSoWhat:
        cleanText(msg),

      captureMode:
        "initial",
    },
  }),

  synthesisState:
    soWhatValidation
      .synthesisState || null,

  validationSource:
    soWhatValidation
      .validationSource || null,

  captureMode:
    "initial",
};

refreshInstructionalSituationWithComponentFinding({
  state:
    s,

  currentResponse:
    msg,

  componentFinding:
    instructionalFinding,
});

const progressionAuthorization =
  buildProgressionAuthorization(
    s,
    {
      frameComponent:
        "soWhat",

      expectedContractId:
        "SW-RTP-001",
    }
  );

s.progressionAuthorization =
  structuredClone(
    progressionAuthorization
  );

if (
  !soWhatValidation.valid ||
  progressionAuthorization
    ?.authorized !== true
) {
    s.pending = {
    type:
      "collectMoreSoWhat",
  };
  return attachGovernedSupportToPending(
    s,
    msg,
    {
      intent:
        "stuck",

      confidence:
        1,

      source:
        `soWhatValidation:${soWhatValidation.diagnosis}`,

      instructionalFinding,
    }
  );
}

// Save only after governed validation.
s.frame.soWhat =
  msg;

s.pending = {
  type:
    "offerMoreSoWhat",
};

return s;
}

return s;
}
// ---------------------
// HANDLER
// ---------------------
export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res
      .status(405)
      .json({ error: "Method not allowed" });
  }

  // Preserve the last safely normalized incoming state so an
  // unexpected error never erases the student's work or location.
  let safeState = defaultState();

  try {
    const body =
      req.body &&
      typeof req.body === "object"
        ? req.body
        : {};

    const message =
      cleanText(body.message || "");

    // ------------------------------------------------------
// HIDDEN KAW DEVELOPER COMMAND
//
// Type "/run tests" in the Wix Kaw chat to run every
// registered deterministic and governed validation suite.
//
// This bypasses the normal student interaction flow and
// does not modify the student's active Frame.
// ------------------------------------------------------

if (
  message.toLowerCase() ===
  "/run tests"
) {
  const testResults =
    await runAllDeterministicSelfTests();

  const formattedSuites =
    testResults.suites.map(
      (suite) =>
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
      defaultState(),
    
    selfTest:
      testResults,
  });
}

// ------------------------------------------------------
// HIDDEN KAW COMPONENT TEST COMMANDS
//
// Runs one existing registered suite without executing the
// full /run tests command.
//
// Supported commands:
//
// /run ia
// /run mi
// /run ed
// /run sw
// /run core
//
// These commands do not modify the student's active Frame.
// ------------------------------------------------------
const componentTestCommandMap = {
  "/run ia":
    "isAbout",

  "/run mi":
    "mainIdeas",

  "/run ed":
    "essentialDetail",

  "/run sw":
    "soWhat",

  "/run core":
    "evidenceState",

  "/run sw1":
    "soWhatValidation",

  "/run sw2":
    "soWhatRuntime",

  "/run sw3":
    "soWhatManual",
};

const requestedComponentSuiteId =
  componentTestCommandMap[
    message.toLowerCase()
  ];

const soWhatBatchMap = {
  soWhatValidation:
    "validation",

  soWhatRuntime:
    "runtime",

  soWhatManual:
    "manual",
};

const requestedSoWhatBatch =
  soWhatBatchMap[
    requestedComponentSuiteId
  ];

if (requestedSoWhatBatch) {
  const testResults =
    await runSoWhatSelfTests(
      requestedSoWhatBatch
    );

  const reply =
    formatSoWhatSelfTestResults(
      testResults
    );

  return res.status(200).json({
    reply,

   state:
    body.state ||
    defaultState(),

    selfTest: {
      suite:
        requestedComponentSuiteId,

      batch:
        requestedSoWhatBatch,

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

if (requestedComponentSuiteId) {
  const suiteExecution =
    await runDeterministicSelfTestSuiteById(
      requestedComponentSuiteId
    );

  if (!suiteExecution) {
    return res.status(404).json({
      reply:
        "The requested Kaw self-test suite could not be found.",

     state:
      body.state ||
      defaultState(),

      selfTest: {
        suite:
          requestedComponentSuiteId,

        found:
          false,
      },
    });
  }

  const testResults =
    suiteExecution.result;

  return res.status(200).json({
    reply:
      suiteExecution.formatted,

  state:
    body.state ||
    defaultState(),

    selfTest: {
      suite:
        suiteExecution.id,

      label:
        suiteExecution.label,

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
      defaultState(),

    instructionalValidationLab:
      ivlResults
  });
}

let incoming =
  body.state ||
  {};

let state =
  normalizeIncomingState(incoming);

  // Keep an unchanged recovery copy from before this request
  // begins mutating instructional state.
  safeState = structuredClone(state);

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

        // ==================================================
    // ACTIVE PENDING CONTRACT EXCLUSIVITY
    // ==================================================
    //
    // When an active pending state exists, that pending
    // contract owns the student's next response.
    //
    // No language detection, global stuck detection,
    // refocus behavior, stage routing, or other runtime
    // interrupt may process the response first.
    //
    // confirmLanguageSwitch remains handled by its
    // dedicated language-confirmation pathway below.
    // ==================================================

    const activePendingType =
      state?.pending?.type || null;

    const pendingContractOwnsTurn =
      !!activePendingType &&
      activePendingType !==
        "confirmLanguageSwitch";

    if (
      message &&
      pendingContractOwnsTurn
    ) {
      state =
        await updateStateFromStudent(
          state,
          message
        );
    } else {

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
  state = await updateStateFromStudent(
    state,
      message
      );
    }
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

// A selected Kaw 2.5 Instructional Contract is the sole
// instructional authority for this support response.
//
// Do not silently switch to the legacy pending-state
// communication engine when governed contextualization or
// Communication Validation fails.
if (
  instructionalActivation &&
  !instructionalResponse
) {
  throw new Error(
    "Governed instructional communication failed."
  );
}

const nextQ =
  instructionalActivation
    ? instructionalResponse
    : computeNextQuestion(state);
      
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
    reply:
      "⚠️ Something went wrong while I was processing that. Please try your response again.",
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
    await runIVLEssentialDetailBenchmarks();

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
