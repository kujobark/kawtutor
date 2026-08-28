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
// KAW OPERATING SYSTEM
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
//      Thinking Move, Progressive Support stage, progression
//      behavior, and student-work protections.
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
//      revision, continuation, interruption,
//      resumption, and export.
//    - Deterministic Frame progression remains authoritative,
//      while Parent Anchor provides read-only structural interpretation.
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
// Do not rewrite load-bearing behavior merely to improve
// organization.
//
// First establish the new layer around existing behavior.
// Then migrate responsibility.
// Then verify behavior.
// Only then remove the superseded pathway.
//
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
      "Briefly explains what the whole Key Topic is about.",
      "Paraphrases or summarizes the Key Topic rather than simply repeating it.",
      "Uses words that are understandable to the student."
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
// Current authoritative role:
//
// The Observation Report contributes governed evidence to
// Evidence State and downstream Instructional Assessment.
//
// It does not itself determine instructional strategy,
// progression, pending state, or communication.
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

    componentContribution: {
      observed:
        false,

      evidenceText:
        "",
    },

    ambiguityPresent:
      false,
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

    // --------------------------------------------------
  // OBSERVABLE COMPONENT CONTRIBUTION
  //
  // AI may observe whether the student's current words
  // contain any candidate contribution to the active
  // Frame component.
  //
  // This is not component validation.
  //
  // The evidence excerpt must appear verbatim in the
  // student's actual interaction.
  // --------------------------------------------------

  const rawComponentContribution =
    rawReport
      ?.componentContribution &&
    typeof rawReport
      .componentContribution ===
      "object"
      ? rawReport
          .componentContribution
      : {};

  const componentContributionEvidenceText =
    cleanText(
      rawComponentContribution
        ?.evidenceText || ""
    );

  const componentContributionObserved =
    rawComponentContribution
      ?.observed === true &&

    Boolean(
      componentContributionEvidenceText
    ) &&

    normalizedText.includes(
      componentContributionEvidenceText
        .toLowerCase()
    );

  const componentContribution = {
    observed:
      componentContributionObserved,

    evidenceText:
      componentContributionObserved
        ? componentContributionEvidenceText
        : "",
  };
  
  return {
    version: "1.0",

    source:
      "aiObservation",

    studentInteraction:
      text,

    observations,

    componentContribution,

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

    const rawStage =
    cleanText(
      getStage(state) || ""
    );

  const activeFrameComponent =
    cleanText(
      getBaseStage(rawStage) || ""
    );

  const activeComponentKnowledge =
    KU_FRAME_COMPONENTS
      ?.[activeFrameComponent] ||
    null;

  const system = `You are the governed AI Observation Layer for Kaw Companion.

Your only responsibility is to report directly observable evidence from the student's current interaction.
You must also observe whether the student's current words contain any candidate contribution to the active Frame component.

A component contribution means the student actually expresses content that adds meaning toward the requested Frame component, even if that content is incomplete, vague, imperfect, or ultimately invalid.

The contribution must contain student-owned subject-matter thinking that goes beyond merely referring to the topic, the Frame component, the task, or the student's ability to answer.

Important distinctions:
- Do not judge whether the contribution is correct, sufficient, strong, valid, or ready to progress.
- Do not decide the instructional situation.
- Referring to the Key Topic is not itself a component contribution.
- Saying that something "is about" the Key Topic is not itself a contribution unless the student also expresses what the topic involves, means, does, includes, causes, affects, or otherwise communicates subject-matter meaning about it.
- Referring to what the student knows, understands, remembers, cannot explain, is unsure about, or is trying to say is interaction language unless the response separately contains subject-matter content for the active component.
- Do not treat words copied or repeated from the accepted Key Topic as new component meaning by themselves.
- A response may contain both interaction language and a genuine component contribution. When it does, report both.
- If a component contribution is present, copy the smallest useful exact excerpt containing the actual subject-matter contribution, not the surrounding interaction language.
- If the response only discusses the task, topic, component, uncertainty, inability, or process of answering, return observed=false and evidenceText="".

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

  const user = `Active Frame component:
${activeFrameComponent || "(none)"}

Component purpose:
${
  cleanText(
    activeComponentKnowledge
      ?.purpose || ""
  ) || "(not available)"
}

Component definition:
${
  cleanText(
    activeComponentKnowledge
      ?.definition || ""
  ) || "(not available)"
}

Current student interaction:
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

                        componentContribution: {
                  type:
                    "object",

                  additionalProperties:
                    false,

                  properties: {
                    observed: {
                      type:
                        "boolean",
                    },

                    evidenceText: {
                      type:
                        "string",
                    },
                  },

                  required: [
                    "observed",
                    "evidenceText",
                  ],
                },

                ambiguityPresent: {
                  type:
                    "boolean",
                },
              },

                          required: [
                "observations",
                "componentContribution",
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
          "Reconnect the student to the accepted Key Topic and invite them to explain, in their own words, what the topic is about without suggesting or supplying the Is About statement.",
      
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
          "Progressive Support increases access to the same instructional objective through Prompt, Model, and Guided Construction while preserving student ownership.",

        scaffolds: [
          {
            level: 1,
            move: "prompt",
            supportType: "prompt",
            purpose:
              "Reconnect the student to accepted context and invite them to perform the intended component thinking with a light instructional nudge.",
            cue:
              "Briefly signal that Kaw is helping the student use what they already have.",
           thinkingMove:
              "Reconnect the student to the accepted Key Topic and invite them to explain, in their own understandable words, what the whole Key Topic is about. Keep the support light and do not suggest, model, or supply any Is About content.",
          },

    {
      level: 2,
      move: "model",
      supportType: "model",
      purpose:
        "Make the required kind of thinking visible through one brief content-distant example, then return immediately to the student's own Frame.",
      cue:
        "Briefly tell the student Kaw will show the kind of thinking using a different topic.",
      modelRules: {
        contentDistant: true,
        brief: true,
        structurallyAnalogous: true,
        mayNotFunctionAsCandidateAnswer: true,
        returnImmediatelyToStudentTask: true,
      },
      thinkingMove:
            "Briefly model what an Is About does using one simple Key Topic that is clearly unrelated to the student's assignment. Show one concise Is About statement that explains what the whole Key Topic is about in understandable words without turning it into a claim, Main Idea, or detailed explanation. Keep the model brief, structurally analogous, and content-distant so it cannot function as a hint or candidate response for the student's actual Frame. Then return immediately to the student's accepted Key Topic and invite the student to explain what their whole Key Topic is about in their own words.",  
      },

        {
      level: 3,
      move: "guidedConstruction",
      supportType: "guidedConstruction",
      purpose:
        "Break the component thinking into smaller sequential operations and build only from the student's own responses.",
      cue:
        "Briefly tell the student Kaw will work through the thinking one step at a time.",

      guidedConstructionRuleSource:
        "GUIDED_CONSTRUCTION_RULES",

      guidedSteps: {
        1: {
          ruleComponent:
            "isAbout",

          ruleStep:
            1,

          operation:
            "identify",
        },

        2: {
          ruleComponent:
            "isAbout",

          ruleStep:
            2,

          operation:
            "explainOrExtend",
        },

        3: {
          ruleComponent:
            "isAbout",

          ruleStep:
            3,

          operation:
            "assemble",
        },
      },

      // Stage 3 compatibility fallback.
      // The step-aware selector replaces this with the
      // current GUIDED_CONSTRUCTION_RULES thinking move.
      thinkingMove:
        "Begin Guided Construction of the Is About by reducing the whole-topic paraphrase into one smaller thinking step. Reconnect to the accepted Key Topic, tell the student Kaw will work through it one step at a time, and ask the student to identify one thing that happens, is involved, is true, or is important to know about the Key Topic. Do not ask for the finished Is About yet, and do not suggest or supply any content.",
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
          "Reconnect the student to the accepted Key Topic and Is About statement, then invite them to identify one larger idea or important part that could help explain the topic. Do not suggest or generate the Main Idea.",
        
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
          "Progressive Support increases access to the same instructional objective through Prompt, Model, and Guided Construction while preserving student ownership.",

        scaffolds: [
          {
            level: 1,
            move: "prompt",
            supportType: "prompt",
            purpose:
              "Reconnect the student to accepted context and invite them to perform the intended component thinking with a light instructional nudge.",
            cue:
              "Briefly signal that Kaw is helping the student use what they already have.",
            thinkingMove:
                 "Reconnect the student to the accepted Key Topic and Is About statement, then invite them to identify one important part of the topic that could become a Main Idea. Use only the authorized Stage 1 thinking lenses supplied in the Communication License to make the choices concrete. Keep the support light and do not suggest, model, choose, or supply the Main Idea.",
              },

    {
      level: 2,
      move: "model",
      supportType: "model",
      purpose:
        "Make the required kind of thinking visible through one brief content-distant example, then return immediately to the student's own Frame.",
      cue:
        "Briefly tell the student Kaw will show the kind of thinking using a different topic.",
      modelRules: {
        contentDistant: true,
        brief: true,
        structurallyAnalogous: true,
        mayNotFunctionAsCandidateAnswer: true,
        returnImmediatelyToStudentTask: true,
      },
      thinkingMove:
        "Briefly model what a Main Idea does using one simple topic that is clearly unrelated to the student's assignment. Show one example Main Idea that functions as a larger organizing idea and could be supported by several Essential Details. Keep the example brief, structurally analogous, and content-distant so it cannot function as a hint or candidate response for the student's actual topic. Then return immediately to the student's accepted Key Topic and Is About statement and invite the student to identify one Main Idea of their own.",
    },

       {
      level: 3,
      move: "guidedConstruction",
      supportType: "guidedConstruction",
      purpose:
        "Break the component thinking into smaller sequential operations and build only from the student's own responses.",
      cue:
        "Briefly tell the student Kaw will work through the thinking one step at a time.",

      guidedConstructionRuleSource:
        "GUIDED_CONSTRUCTION_RULES",

      guidedSteps: {
        1: {
          ruleComponent:
            "mainIdeas",

          ruleStep:
            1,

          operation:
            "identifyOrganizer",
        },

        2: {
          ruleComponent:
            "mainIdeas",

          ruleStep:
            2,

          operation:
            "establishOrganizingPower",
        },

        3: {
          ruleComponent:
            "mainIdeas",

          ruleStep:
            3,

          operation:
            "formulateOrganizer",
        },
      },

       // Stage 3 compatibility fallback.
      // The step-aware selector replaces this with the
      // current GUIDED_CONSTRUCTION_RULES thinking move.
      thinkingMove:
        "Begin Guided Construction of the Main Idea by reducing the organizing task into one smaller thinking step. Reconnect to the accepted Key Topic and Is About statement, tell the student Kaw will work through it one step at a time, and ask the student to identify one important aspect, cause, effect, part, stage, pattern, event, category, or other organizing idea they notice in the topic. Do not ask for the finished Main Idea yet, and do not suggest, choose, or supply the student's organizer.",
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
          "Reconnect the student to the accepted Main Idea and invite them to identify one specific thing that could help explain or support it. Do not suggest or generate the Essential Detail.",
        
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
          "Progressive Support increases access to the same instructional objective through Prompt, Model, and Guided Construction while preserving student ownership.",

        scaffolds: [
          {
            level: 1,
            move: "prompt",
            supportType: "prompt",
            purpose:
              "Reconnect the student to accepted context and invite them to perform the intended component thinking with a light instructional nudge.",
            cue:
              "Briefly signal that Kaw is helping the student use what they already have.",
            thinkingMove:
              "Reconnect the student to the accepted Main Idea and invite them to identify one specific thing that could help explain or support it. Use only the authorized Stage 1 thinking lenses supplied in the Communication License to make the choices concrete. Keep the support light and do not suggest, model, choose, or supply the Essential Detail.",
            },

    {
      level: 2,
      move: "model",
      supportType: "model",
      purpose:
        "Make the required kind of thinking visible through one brief content-distant example, then return immediately to the student's own Frame.",
      cue:
        "Briefly tell the student Kaw will show the kind of thinking using a different topic.",
      modelRules: {
        contentDistant: true,
        brief: true,
        structurallyAnalogous: true,
        mayNotFunctionAsCandidateAnswer: true,
        returnImmediatelyToStudentTask: true,
      },
      thinkingMove:
        "Briefly model what an Essential Detail does using one simple Main Idea from content that is clearly unrelated to the student's assignment. Show one specific supporting fact, example, observation, explanation, event, condition, action, result, or piece of evidence beneath that example Main Idea. Keep the example brief, structurally analogous, and content-distant so it cannot function as a hint or candidate response for the student's actual Main Idea. Then return immediately to the student's accepted Main Idea and invite the student to identify one Essential Detail of their own.",
    },

           {
      level: 3,
      move: "guidedConstruction",
      supportType: "guidedConstruction",
      purpose:
        "Break the component thinking into smaller sequential operations and build only from the student's own responses.",
      cue:
        "Briefly tell the student Kaw will work through the thinking one step at a time.",

      guidedConstructionRuleSource:
        "GUIDED_CONSTRUCTION_RULES",

      guidedSteps: {
        1: {
          ruleComponent:
            "details",

          ruleStep:
            1,

          operation:
            "identifySpecificInformation",
        },

        2: {
          ruleComponent:
            "details",

          ruleStep:
            2,

          operation:
            "establishSupportAndEssentiality",
        },

        3: {
          ruleComponent:
            "details",

          ruleStep:
            3,

          operation:
            "formulateEssentialDetail",
        },
      },

      // Stage 3 compatibility fallback.
      // The step-aware selector replaces this with the
      // current GUIDED_CONSTRUCTION_RULES thinking move.
      thinkingMove:
        "Begin Guided Construction of the Essential Detail by separating identification of supporting information from explanation of the supporting relationship. Reconnect to the accepted Main Idea, tell the student Kaw will work through it one step at a time, and first ask the student to identify one specific thing they know, noticed, read, observed, or can point to that relates to the Main Idea. Do not ask them to explain the connection yet, and do not suggest or supply the evidence.",
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
          "Reconnect the student to the completed Frame and invite them to identify the larger understanding or takeaway that becomes clear when they consider their Main Ideas and Essential Details together. Do not suggest or generate the So What.",
        
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
          "Return to the exact initial, continuation, or revision So What location and validate the student's next response."
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
          "Return to the exact initial, continuation, or revision So What location and validate the student's next response."
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
          "Return to the exact initial, continuation, or revision So What location and validate the student's next response."
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
          "Return to the exact initial, continuation, or revision So What location and validate the student's next response."
      },

           progressiveSupport: {
        principle:
          "Progressive Support increases access to the same instructional objective through Prompt, Model, and Guided Construction while preserving student ownership.",

        scaffolds: [
          {
            level: 1,
            move: "prompt",
            supportType: "prompt",
            purpose:
              "Reconnect the student to accepted context and invite them to perform the intended component thinking with a light instructional nudge.",
            cue:
              "Briefly signal that Kaw is helping the student use what they already have.",
            thinkingMove:
              "Reconnect the student to the completed Frame and invite them to identify the most important larger understanding or takeaway that becomes clear when they consider their Main Ideas and Essential Details together. Keep the support light and do not suggest, model, choose, or supply the So What.",
          },

    {
      level: 2,
      move: "model",
      supportType: "model",
      purpose:
        "Make the required kind of thinking visible through one brief content-distant example, then return immediately to the student's own Frame.",
      cue:
        "Briefly tell the student Kaw will show the kind of thinking using a different topic.",
      modelRules: {
        contentDistant: true,
        brief: true,
        structurallyAnalogous: true,
        mayNotFunctionAsCandidateAnswer: true,
        returnImmediatelyToStudentTask: true,
      },
      thinkingMove:
          "Briefly model what a So What does using one simple completed Frame that is clearly unrelated to the student's assignment. Name two brief Main Ideas from that example Frame, then show one concise So What that combines those ideas into a larger understanding rather than simply repeating them. Keep the example brief, structurally analogous, and content-distant so it cannot function as a hint or candidate response for the student's actual Frame. Then return immediately to the student's completed Frame and invite the student to identify the larger understanding that emerges from their own ideas together.",
    },

    {
      level: 3,
      move: "guidedConstruction",
      supportType: "guidedConstruction",
      purpose:
        "Break the component thinking into smaller sequential operations and build only from the student's own responses.",
      cue:
        "Briefly tell the student Kaw will work through the thinking one step at a time.",

      guidedConstructionRuleSource:
        "GUIDED_CONSTRUCTION_RULES",

      guidedSteps: {
        1: {
          ruleComponent:
            "soWhat",

          ruleStep:
            1,

          operation:
            "connectFrame",
        },

        2: {
          ruleComponent:
            "soWhat",

          ruleStep:
            2,

          operation:
            "determineImportance",
        },

        3: {
          ruleComponent:
            "soWhat",

          ruleStep:
            3,

          operation:
            "synthesize",
        },
      },

      // Stage 3 compatibility fallback.
      // The step-aware selector replaces this with the
      // current GUIDED_CONSTRUCTION_RULES thinking move.
      thinkingMove:
        "Begin Guided Construction of the So What by helping the student look across the completed Frame and identify one meaningful relationship, pattern, connection, contrast, sequence, cause-and-effect relationship, or larger idea that emerges from the accepted Main Ideas and Essential Details in relation to the Key Topic and Is About. Do not require a particular relationship structure, and do not interpret or supply the connection for the student.",
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
          "Preserve the accepted So What and continue through the existing confirmation and export progression pathway.",
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

// ======================================================
// GUIDED CONSTRUCTION RULES
// ======================================================
//
// Guided Construction is Progressive Support Stage 3.
//
// This structure defines the component-specific smaller
// thinking operations used while Guided Construction is
// active.
//
// It is declarative instructional knowledge only.
//
// It does not:
//
// • determine whether Guided Construction begins;
// • validate a completed Frame component;
// • advance Guided Construction steps;
// • mutate pending state;
// • save student work;
// • select an Instructional Contract;
// • determine an Instructional Situation;
// • generate student-facing communication.
//
// Runtime Guided Construction consumes these rules to
// govern step-aware instructional continuation.
//
// Active Pathway Authority:
//
// Genuine Struggle governs entry into Guided Construction.
// Once Guided Construction is active at the same exact
// instructional location, Guided Construction rules govern
// continuation until the component is completed or the
// pathway reaches its defined endpoint.
//
// The existing governed component validators remain the
// final authority for full component acceptance.
//
// ======================================================

const GUIDED_CONSTRUCTION_RULES = Object.freeze({

  isAbout: {
    component:
      "isAbout",

    steps: {
      1: {
        step:
          1,

        operation:
          "identify",

        purpose:
          "Identify one meaningful student-owned idea about the accepted Key Topic that can become part of a whole-topic Is About paraphrase.",

        sufficientMicroStepEvidence:
          [
            "The student expresses one understandable idea about what happens, is involved, is true, or is important to know about the accepted Key Topic.",
            "The idea contributes observable meaning beyond merely repeating the Key Topic.",
            "The idea is coherent enough to build from without Kaw supplying the next idea.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response contains no usable component thinking.",
            "The response merely repeats the Key Topic.",
            "The response is too vague or unclear to identify one meaningful idea.",
            "The response is disconnected from the accepted Key Topic.",
            "The response consists only of struggle, meta-commentary, or unrelated language.",
          ],

        thinkingMove:
          "Begin Guided Construction of the Is About by reducing the whole-topic paraphrase into one smaller thinking step. Reconnect to the accepted Key Topic, tell the student Kaw will work through it one step at a time, and ask the student to identify one thing that happens, is involved, is true, or is important to know about the Key Topic. Do not ask for the finished Is About yet, and do not suggest or supply any content.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          maySupplyContent:
            false,

          mayCompleteStudentThinking:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },

      2: {
        step:
          2,

        operation:
          "explainOrExtend",

        purpose:
          "Help the student explain, extend, clarify, connect, or qualify the student-owned idea identified in Step 1 so that it contributes understandable meaning about the whole Key Topic.",

        sufficientMicroStepEvidence:
          [
            "The student adds understandable meaning to the Step-1 idea.",
            "The response explains, extends, clarifies, connects, qualifies, or describes what the idea shows about the accepted Key Topic.",
            "The response provides enough student-owned meaning to continue without Kaw supplying the next idea.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response merely repeats the Step-1 idea.",
            "The response adds no observable meaning.",
            "The relationship to the accepted Key Topic remains unclear or disconnected.",
            "The response contains no usable component thinking.",
          ],

        thinkingMove:
          "Continue Guided Construction of the Is About by building only from the student-owned idea identified in the previous step. Reconnect to that idea and the accepted Key Topic, then ask the student to explain, extend, clarify, connect, or describe what that idea tells someone about the Key Topic. Do not ask Kaw to supply the explanation, and do not interpret or complete the student's meaning for them.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          maySupplyExplanation:
            false,

          mayInferMeaning:
            false,

          mayCompleteStudentThinking:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },

      3: {
        step:
          3,

        operation:
          "assemble",

        purpose:
          "Help the student assemble their own accepted guided evidence into an understandable whole-topic Is About paraphrase.",

        sufficientMicroStepEvidence:
          [
            "The student attempts to combine or formulate their own guided evidence into a whole-topic explanation.",
            "The response remains grounded in the student-owned ideas established during Guided Construction.",
            "The response contains usable formulation or synthesis even if the normal Is About validator has not yet accepted it.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response contains no usable attempt to formulate the Is About.",
            "The response abandons or contradicts the student-owned guided evidence without establishing new usable thinking.",
            "The response consists only of struggle, meta-commentary, repetition, or unrelated language.",
          ],

        thinkingMove:
          "Continue Guided Construction of the Is About by reconnecting to the student-owned ideas established in the earlier guided steps and inviting the student to put those ideas together into a whole-topic Is About in their own words. Kaw may remind the student of ideas they already supplied, but must not combine, complete, rewrite, interpret, or supply the Is About for them.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayCombineStudentEvidence:
            false,

          mayCompleteStudentThinking:
            false,

          mayRewriteStudentThinking:
            false,

          maySupplyParaphrase:
            false,
        },
      },
    },
  },

  mainIdeas: {
    component:
      "mainIdeas",

    steps: {
      1: {
        step:
          1,

        operation:
          "identifyOrganizer",

        purpose:
          "Identify one meaningful organizing idea that helps structure the accepted Key Topic and Is About and could reasonably hold multiple Essential Details.",

        sufficientMicroStepEvidence:
          [
            "The student identifies a meaningful aspect, cause, effect, part, stage, pattern, event, category, or other organizing idea.",
            "The proposed organizer is connected to the accepted Key Topic and Is About.",
            "The idea is coherent enough to explore as a potential organizer.",
            "For later Main Ideas, the proposed organizer is meaningfully distinct from already accepted Main Ideas.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response merely repeats the Key Topic or Is About.",
            "The response contains no meaningful organizing idea.",
            "The response is only one isolated detail with no observable organizing potential.",
            "The response duplicates an already accepted Main Idea without adding a distinct organizing contribution.",
            "The response is unrelated, vague, or unusable.",
          ],

        thinkingMove:
          "Begin Guided Construction of the Main Idea by reducing the organizing task into one smaller thinking step. Reconnect to the accepted Key Topic and Is About statement, tell the student Kaw will work through it one step at a time, and ask the student to identify one important aspect, cause, effect, part, stage, pattern, event, category, or other organizing idea they notice in the topic. Do not ask for the finished Main Idea yet, and do not suggest, choose, or supply the student's organizer.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayChooseOrganizer:
            false,

          maySupplyMainIdea:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },

      2: {
        step:
          2,

        operation:
          "establishOrganizingPower",

        purpose:
          "Help the student establish that the proposed organizer can meaningfully hold multiple kinds of supporting information without prematurely constructing Essential Details.",

        sufficientMicroStepEvidence:
          [
            "The student identifies more than one kind, category, example type, fact type, event, reason, condition, or other information that could reasonably fit beneath the organizer.",
            "The response demonstrates that the organizer has enough breadth or structure to function as a Main Idea.",
            "The student does not need to formulate finished Essential Details.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response gives only one isolated detail without demonstrating organizing power.",
            "The response simply repeats the proposed Main Idea.",
            "The response shows that the proposed idea functions only as a single detail.",
            "The response provides no usable evidence that multiple supporting details could fit beneath it.",
          ],

        thinkingMove:
          "Continue Guided Construction of the Main Idea by building from the student-owned organizer identified in the previous step. Ask the student what kinds of facts, examples, events, reasons, conditions, observations, or other information could fit underneath that organizer. The student does not need to write finished Essential Details yet. Do not choose, invent, or supply the supporting information for the student.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayGenerateSupportingDetails:
            false,

          mayChooseOrganizer:
            false,

          mayCompleteStudentThinking:
            false,
        },
      },

      3: {
        step:
          3,

        operation:
          "formulateOrganizer",

        purpose:
          "Help the student state the organizing idea in whatever form best represents their thinking without imposing a sentence-length or grammatical-form requirement.",

        sufficientMicroStepEvidence:
          [
            "The student attempts to state or formulate the organizer they established.",
            "The formulation remains grounded in the student-owned organizer and organizing relationship.",
            "The response may be a word, phrase, category, event title, heading, or sentence.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response does not state a usable organizer.",
            "The response abandons the established organizing idea without producing another coherent organizer.",
            "The response consists only of repetition, struggle language, or unrelated content.",
          ],

        thinkingMove:
          "Continue Guided Construction of the Main Idea by reconnecting to the student-owned organizer and the kinds of information the student said could fit beneath it. Invite the student to state that Main Idea in the form that best represents their thinking. It may be a word, phrase, category, heading, event, or sentence. Do not require a longer form merely because the response is concise, and do not rewrite or supply the Main Idea.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayRequireSentenceForm:
            false,

          maySupplyMainIdea:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },
    },
  },

  details: {
    component:
      "details",

    steps: {
      1: {
        step:
          1,

        operation:
          "identifySpecificInformation",

        purpose:
          "Identify one concrete piece of student-owned information relevant enough to the accepted Main Idea and larger Frame that its supporting relationship can be explored.",

        sufficientMicroStepEvidence:
          [
            "The student identifies one specific fact, example, observation, explanation, event, condition, action, result, or piece of evidence.",
            "The information is understandable and sufficiently connected to the accepted Main Idea to explore its support relationship.",
            "The student does not yet need to fully explain the supporting relationship.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response contains no concrete information.",
            "The response merely repeats the Main Idea.",
            "The response is too vague to identify one specific piece of information.",
            "The response is unrelated to the accepted Main Idea and larger Frame.",
            "The response contains only struggle, meta-commentary, or unrelated language.",
          ],

        thinkingMove:
          "Begin Guided Construction of the Essential Detail by separating identification of supporting information from explanation of the supporting relationship. Reconnect to the accepted Main Idea, tell the student Kaw will work through it one step at a time, and first ask the student to identify one specific thing they know, noticed, read, observed, or can point to that relates to the Main Idea. Do not ask them to explain the connection yet, and do not suggest or supply the evidence.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayChooseEvidence:
            false,

          maySupplyEssentialDetail:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },

      2: {
        step:
          2,

        operation:
          "establishSupportAndEssentiality",

        purpose:
          "Help the student establish both how the identified information supports the accepted Main Idea and why it contributes important understanding rather than being merely related, interesting, incidental, or trivial.",

        sufficientMicroStepEvidence:
          [
            "The student's response makes the supporting relationship to the accepted Main Idea understandable.",
            "The information contributes important understanding of the Main Idea rather than functioning only as a related or interesting detail.",
            "The relationship remains coherent with the accepted Is About and Key Topic.",
            "Only the missing thinking must be established; redundant justification is not required.",
          ],

        insufficientMicroStepEvidence:
          [
            "The student merely says the information relates to the Main Idea.",
            "The response repeats the Step-1 information without explaining its support relationship.",
            "The response gives a vague connection that still requires Kaw to infer why it supports the Main Idea.",
            "The information appears merely interesting or incidental rather than important to understanding the Main Idea.",
            "The response drifts from the accepted Main Idea or larger Frame.",
          ],

        thinkingMove:
          "Continue Guided Construction of the Essential Detail by building from the specific student-owned information identified in the previous step. Reconnect to that information and the accepted Main Idea, then help the student make clear how or why the information supports, illustrates, develops, exemplifies, or provides evidence for the Main Idea and why it contributes important understanding rather than merely being related, interesting, or incidental. The supporting relationship must remain coherent with the accepted Is About and Key Topic. Ask only for the missing thinking needed to establish these relationships; do not require redundant justification when the detail already satisfies the normal governed Essential Detail criteria, and do not interpret, complete, choose, or supply the relationship for the student.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          maySupplyRelationship:
            false,

          mayInferRelationship:
            false,

          mayChooseEvidence:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },

      3: {
        step:
          3,

        operation:
          "formulateEssentialDetail",

        purpose:
          "Help the student state the student-owned information and established support relationship as one clear Essential Detail beneath the accepted Main Idea.",

        sufficientMicroStepEvidence:
          [
            "The student attempts to state the Essential Detail using their own identified information and support relationship.",
            "The response remains grounded in the student-owned evidence established in earlier guided steps.",
            "The response contains usable formulation even if the normal Essential Detail validator has not yet accepted it.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response contains no usable attempt to formulate the Essential Detail.",
            "The response abandons the student-owned evidence or support relationship without establishing new usable thinking.",
            "The response consists only of struggle language, repetition, or unrelated content.",
          ],

        thinkingMove:
          "Continue Guided Construction of the Essential Detail by reconnecting to the student-owned information and supporting relationship established in the earlier guided steps. Invite the student to state the Essential Detail in their own words so that the specific information and its support for the accepted Main Idea are understandable. Kaw may remind the student of ideas they already supplied, but must not combine, complete, rewrite, interpret, or supply the Essential Detail for them.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayCombineStudentEvidence:
            false,

          maySupplyEssentialDetail:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },
    },
  },

  soWhat: {
    component:
      "soWhat",

    steps: {
      1: {
        step:
          1,

        operation:
          "connectFrame",

        purpose:
          "Help the student look across the completed Frame and identify one meaningful relationship, pattern, connection, contrast, sequence, cause-and-effect relationship, or larger idea that emerges from the accepted Frame content.",

        sufficientMicroStepEvidence:
          [
            "The student identifies a meaningful relationship, pattern, connection, contrast, sequence, cause-and-effect relationship, or larger idea grounded in the completed Frame.",
            "The connection may emerge from Main Ideas, Essential Details, or relationships among accepted Frame content.",
            "The response provides enough student-owned thinking to explore what is important to understand.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response merely lists or repeats accepted Frame components.",
            "The response identifies no meaningful relationship, pattern, or larger idea.",
            "The response is disconnected from the completed Frame.",
            "The response contains only a vague statement such as saying the topic is important without identifying what emerges from the Frame.",
          ],

        thinkingMove:
          "Begin Guided Construction of the So What by helping the student look across the completed Frame and identify one meaningful relationship, pattern, connection, contrast, sequence, cause-and-effect relationship, or larger idea that emerges from the accepted Main Ideas and Essential Details in relation to the Key Topic and Is About. Do not require a particular relationship structure, and do not interpret or supply the connection for the student.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayChooseRelationship:
            false,

          mayInterpretFrameForStudent:
            false,

          maySupplyTakeaway:
            false,
        },
      },

      2: {
        step:
          2,

        operation:
          "determineImportance",

        purpose:
          "Help the student determine what is important to understand about the relationship or larger idea they identified in the completed Frame.",

        sufficientMicroStepEvidence:
          [
            "The student identifies a meaningful significance, conclusion, implication, application, real-world connection, topic connection, broader principle, metaphorical understanding, or other larger meaning grounded in the Frame.",
            "The importance is understandable without Kaw supplying the conclusion.",
            "The response moves beyond merely repeating accepted Frame content.",
          ],

        insufficientMicroStepEvidence:
          [
            "The student gives only a generic statement that the topic is important.",
            "The response simply repeats the Step-1 relationship without interpreting its importance.",
            "The proposed importance is unsupported by the completed Frame.",
            "The response provides no usable larger meaning.",
          ],

        thinkingMove:
          "Continue Guided Construction of the So What by building from the student-owned relationship or pattern identified in the previous step. Ask the student to determine what is important to understand about that relationship or what larger significance, conclusion, implication, application, real-world connection, connection to another topic, broader principle, metaphorical understanding, or other meaningful synthesis emerges from it. Use the student's own Frame as the evidence base, and do not choose or supply the significance for the student.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayChooseSignificance:
            false,

          maySupplyConclusion:
            false,

          mayInterpretFrameForStudent:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },

      3: {
        step:
          3,

        operation:
          "synthesize",

        purpose:
          "Help the student express the larger understanding established through the previous guided steps in a form appropriate to the completed Frame.",

        sufficientMicroStepEvidence:
          [
            "The student attempts to express a larger understanding grounded in the completed Frame.",
            "The response uses the student-owned relationship and significance established in earlier guided steps.",
            "The synthesis may take a legitimate form such as a conclusion, broader understanding, implication, application, real-world connection, connection to another topic, metaphor or simile, basic life truth, or other meaningful synthesis.",
            "The response contains usable final-step thinking even if the normal So What validator has not yet accepted it.",
          ],

        insufficientMicroStepEvidence:
          [
            "The response merely repeats the Key Topic, Is About, one Main Idea, or one Essential Detail.",
            "The response only lists Frame content without larger understanding.",
            "The response makes an unsupported leap not grounded in the completed Frame.",
            "The response contains no usable attempt at synthesis.",
          ],

        thinkingMove:
          "Continue Guided Construction of the So What by helping the student express the larger understanding established through the previous guided steps. Reconnect to the student-owned relationship and significance, then invite the student to state the So What in their own words and in a form appropriate to the completed Frame. The response may take the form of a conclusion, broader understanding, implication, application, real-world connection, connection to another topic, metaphor or simile, basic life truth, or another meaningful synthesis grounded in the Frame. Kaw may remind the student of ideas the student has already supplied, but must not interpret, combine, rewrite, complete, or supply the So What for them.",

        studentWorkProtection: {
          preserveStudentOwnership:
            true,

          mayReferencePriorStudentEvidence:
            true,

          mayCombineStudentEvidence:
            false,

          maySupplyTakeaway:
            false,

          mayInterpretFrameForStudent:
            false,

          mayRewriteStudentThinking:
            false,
        },
      },
    },
  },

});

// ======================================================
// GUIDED CONSTRUCTION ACTIVE CONTEXT
// ======================================================
//
// Provides one read-only representation of the currently
// active Guided Construction pathway.
//
// Guided Construction is active only when:
//
// • Progressive Support Stage is exactly 3;
// • guidedConstructionStep is 1, 2, or 3;
// • the current Frame component is one governed by the
//   Guided Construction rules.
//
// This helper observes active pathway state only.
//
// It does not:
//
// • determine whether Guided Construction should begin;
// • determine whether the instructional location is the
//   same as a prior Guided Construction location;
// • validate student evidence;
// • advance or reset a Guided Construction step;
// • mutate pending state;
// • select an Instructional Contract;
// • change runtime progression.
//
// Exact instructional-location identity is governed
// separately by the Guided Construction location helper.
//
// ======================================================

function getActiveGuidedConstructionContext(
  state
) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const pending =
    safeState?.pending &&
    typeof safeState.pending === "object"
      ? safeState.pending
      : null;

  if (!pending) {
    return {
      active:
        false,

      frameComponent:
        null,

      guidedConstructionStep:
        null,

      rawStage:
        null,

      pendingType:
        null,

      pendingIndex:
        null,

      captureMode:
        null,
    };
  }

  const progressiveSupportStage =
    Number(
      pending?.progressiveSupportStage
    );

  const guidedConstructionStep =
    Number(
      pending?.guidedConstructionStep
    );

  const rawStage =
    cleanText(
      getStage(safeState) || ""
    );

  const frameComponent =
    cleanText(
      getBaseStage(rawStage) || ""
    );

  const componentRules =
    GUIDED_CONSTRUCTION_RULES
      ?.[frameComponent] || null;

  const validGuidedStep =
    Number.isInteger(
      guidedConstructionStep
    ) &&
    guidedConstructionStep >= 1 &&
    guidedConstructionStep <= 3;

  const active =
    progressiveSupportStage === 3 &&
    validGuidedStep &&
    componentRules !== null;

  return {
    active,

    frameComponent:
      active
        ? frameComponent
        : null,

    guidedConstructionStep:
      active
        ? guidedConstructionStep
        : null,

    rawStage:
      active
        ? rawStage
        : null,

    pendingType:
      active
        ? cleanText(
            pending?.type || ""
          ) || null
        : null,

    pendingIndex:
      active &&
      Number.isInteger(
        pending?.index
      )
        ? pending.index
        : null,

    captureMode:
      active
        ? cleanText(
            pending?.captureMode || ""
          ) || null
        : null,
  };
}

// ======================================================
// GUIDED CONSTRUCTION INSTRUCTIONAL LOCATION
// ======================================================
//
// Builds one deterministic, read-only identity for the
// student's exact current instructional target.
//
// Guided Construction may continue only when the active
// pathway remains at this same exact location.
//
// Location identity is intentionally narrower than the
// broader Frame component.
//
// Examples:
//
// • Main Idea 1 is different from Main Idea 2.
// • Main Idea revision index 0 is different from index 1.
// • Essential Detail 1 beneath Main Idea 0 is different
//   from Essential Detail 2 beneath Main Idea 0.
// • Essential Details beneath different Main Ideas are
//   different locations.
// • Build, revision, optional, and Strengthen capture
//   remain distinguishable when the runtime exposes that
//   distinction.
//
// This helper does not:
//
// • determine whether Guided Construction should begin;
// • determine Guided Construction progression;
// • validate student evidence;
// • mutate pending state;
// • preserve or clear Guided Construction metadata;
// • select an Instructional Contract;
// • change runtime progression.
//
// ======================================================

function buildGuidedConstructionInstructionalLocation(
  state
) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const pending =
    safeState?.pending &&
    typeof safeState.pending === "object"
      ? safeState.pending
      : null;

  const instructionalFinding =
    pending?.instructionalFinding &&
    typeof pending
      .instructionalFinding === "object"
      ? pending.instructionalFinding
      : safeState?.componentInstructionalFinding &&
        typeof safeState
          .componentInstructionalFinding ===
          "object"
        ? safeState.componentInstructionalFinding
        : null;

  const findingEvidence =
    instructionalFinding?.evidence &&
    typeof instructionalFinding
      .evidence === "object"
      ? instructionalFinding.evidence
      : {};

  const rawStage =
    cleanText(
      getStage(safeState) || ""
    );

  const frameComponent =
    cleanText(
      instructionalFinding
        ?.frameComponent ||
      getBaseStage(rawStage) ||
      ""
    );

  const pendingType =
    cleanText(
      pending?.type || ""
    );

  const interactionMode =
    cleanText(
      safeState?.interactionMode ||
      "build"
    );

  const captureMode =
    cleanText(
      pending?.captureMode ||
      instructionalFinding
        ?.captureMode ||
      findingEvidence
        ?.captureMode ||
      ""
    );

  // --------------------------------------------------
  // MAIN IDEA TARGET INDEX
  //
  // Revisions and Strengthen locations expose an index.
  //
  // Required / optional Main Idea collection may not
  // store one directly in pending, so the next available
  // Main Idea slot identifies the active target.
  // --------------------------------------------------

  let mainIdeaIndex =
    null;

  if (
    frameComponent ===
    "mainIdeas"
  ) {
    if (
      Number.isInteger(
        pending?.index
      )
    ) {
      mainIdeaIndex =
        pending.index;
    } else if (
      Number.isInteger(
        findingEvidence
          ?.revisionIndex
      )
    ) {
      mainIdeaIndex =
        findingEvidence.revisionIndex;
    } else {
      mainIdeaIndex =
        getIdeaList(
          safeState
        ).length;
    }
  }

  // --------------------------------------------------
  // ESSENTIAL DETAIL LOCATION
  //
  // An Essential Detail requires two coordinates:
  //
  // • Main Idea index;
  // • Detail index beneath that Main Idea.
  //
  // The pending object does not always retain both, but
  // the governed instructional finding already records
  // them during Essential Detail validation.
  // --------------------------------------------------

  let detailMainIdeaIndex =
    null;

  let detailIndex =
    null;

  if (
    frameComponent ===
    "details"
  ) {
    if (
      Number.isInteger(
        pending?.index
      )
    ) {
      detailMainIdeaIndex =
        pending.index;
    } else if (
      Number.isInteger(
        instructionalFinding
          ?.currentMainIdeaIndex
      )
    ) {
      detailMainIdeaIndex =
        instructionalFinding
          .currentMainIdeaIndex;
    } else if (
      Number.isInteger(
        findingEvidence
          ?.currentMainIdeaIndex
      )
    ) {
      detailMainIdeaIndex =
        findingEvidence
          .currentMainIdeaIndex;
    }

    if (
      Number.isInteger(
        pending?.detailIndex
      )
    ) {
      detailIndex =
        pending.detailIndex;
    } else if (
      Number.isInteger(
        instructionalFinding
          ?.currentDetailIndex
      )
    ) {
      detailIndex =
        instructionalFinding
          .currentDetailIndex;
    } else if (
      Number.isInteger(
        findingEvidence
          ?.currentDetailIndex
      )
    ) {
      detailIndex =
        findingEvidence
          .currentDetailIndex;
    } else if (
      Number.isInteger(
        detailMainIdeaIndex
      )
    ) {
      const detailBucket =
        Array.isArray(
          safeState?.frame
            ?.details
            ?.[detailMainIdeaIndex]
        )
          ? safeState.frame.details[
              detailMainIdeaIndex
            ]
          : [];

      detailIndex =
        detailBucket.length;
    }
  }

  const locationEstablished =
    Boolean(
      frameComponent &&
      GUIDED_CONSTRUCTION_RULES
        ?.[frameComponent]
    );

  return {
    locationEstablished,

    interactionMode:
      locationEstablished
        ? interactionMode
        : null,

    frameComponent:
      locationEstablished
        ? frameComponent
        : null,

    rawStage:
      locationEstablished
        ? rawStage
        : null,

    pendingType:
      locationEstablished
        ? pendingType || null
        : null,

    captureMode:
      locationEstablished
        ? captureMode || null
        : null,

    mainIdeaIndex:
      locationEstablished &&
      frameComponent ===
        "mainIdeas"
        ? mainIdeaIndex
        : null,

    detailMainIdeaIndex:
      locationEstablished &&
      frameComponent ===
        "details"
        ? detailMainIdeaIndex
        : null,

    detailIndex:
      locationEstablished &&
      frameComponent ===
        "details"
        ? detailIndex
        : null,
  };
}

// ------------------------------------------------------
// GUIDED CONSTRUCTION LOCATION COMPARISON
// ------------------------------------------------------
//
// Determines only whether two previously established
// Guided Construction locations represent the same exact
// instructional target.
//
// It does not decide whether Guided Construction remains
// active or whether any state should be preserved.
//
// ------------------------------------------------------

function isSameGuidedConstructionInstructionalLocation(
  firstLocation,
  secondLocation
) {
  const first =
    firstLocation &&
    typeof firstLocation === "object"
      ? firstLocation
      : null;

  const second =
    secondLocation &&
    typeof secondLocation === "object"
      ? secondLocation
      : null;

  if (
    !first ||
    !second ||
    first.locationEstablished !==
      true ||
    second.locationEstablished !==
      true
  ) {
    return false;
  }

  return (
    first.interactionMode ===
      second.interactionMode &&

    first.frameComponent ===
      second.frameComponent &&

    first.rawStage ===
      second.rawStage &&

    first.pendingType ===
      second.pendingType &&

    first.captureMode ===
      second.captureMode &&

    first.mainIdeaIndex ===
      second.mainIdeaIndex &&

    first.detailMainIdeaIndex ===
      second.detailMainIdeaIndex &&

    first.detailIndex ===
      second.detailIndex
  );
}

// ======================================================
// GUIDED CONSTRUCTION EVIDENCE ASSESSMENT
// ======================================================
//
// Evaluates the student's response only for the currently
// active Guided Construction micro-step.
//
// The normal governed component validator always receives
// first authority.
//
// If the full component is already valid, Guided
// Construction ends immediately and normal component
// progression remains authoritative.
//
// This assessor may deterministically establish clear
// no-evidence conditions.
//
// Meaningful micro-step sufficiency may require bounded
// semantic evidence because Guided Construction evaluates
// component-specific thinking operations rather than
// surface form, length, or vocabulary.
//
// Semantic evidence does not determine progression.
// It supplies bounded evidence to this deterministic
// assessor.
//
// This helper does not:
//
// • validate a completed Frame component independently;
// • select an Instructional Contract;
// • advance a Guided Construction step;
// • mutate pending state;
// • save student work;
// • generate student-facing communication.
//
// ======================================================

const GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES =
  Object.freeze({
    COMPONENT_COMPLETE:
      "componentComplete",

    INSUFFICIENT_MICRO_STEP_EVIDENCE:
      "insufficientMicroStepEvidence",

    SUFFICIENT_MICRO_STEP_EVIDENCE:
      "sufficientMicroStepEvidence",

    USABLE_FINAL_STEP_EVIDENCE:
      "usableFinalStepEvidence",

    NO_USABLE_FINAL_STEP_EVIDENCE:
      "noUsableFinalStepEvidence",
  });

function assessGuidedConstructionEvidence({
  state = null,
  response = "",
  frameComponent = "",
  guidedConstructionStep = null,
  componentValidation = null,
  microStepSemanticEvidence = null,
} = {}) {
  const text =
    cleanText(response);

  const component =
    cleanText(
      frameComponent
    );

  const step =
    Number(
      guidedConstructionStep
    );

  const validation =
    componentValidation &&
    typeof componentValidation ===
      "object"
      ? componentValidation
      : {};

  const semanticEvidence =
    microStepSemanticEvidence &&
    typeof microStepSemanticEvidence ===
      "object"
      ? microStepSemanticEvidence
      : null;

  const componentRules =
    GUIDED_CONSTRUCTION_RULES
      ?.[component] || null;

  const stepRules =
    componentRules
      ?.steps
      ?.[step] || null;

  const validStep =
    Number.isInteger(step) &&
    step >= 1 &&
    step <= 3;

  // --------------------------------------------------
  // FULL COMPONENT VALIDATION OVERRIDE
  //
  // Existing governed component validation always gets
  // first say.
  //
  // Guided Construction may never become an additional
  // hoop after the student has already produced a valid
  // component.
  // --------------------------------------------------

  if (
    validation?.valid === true
  ) {
    return {
      assessmentStatus:
        "complete",

      outcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .COMPONENT_COMPLETE,

      frameComponent:
        component || null,

      guidedConstructionStep:
        validStep
          ? step
          : null,

      studentEvidence:
        text || null,

      rule:
        stepRules
          ? structuredClone(
              stepRules
            )
          : null,

      evidenceBasis: [
        "governedComponentValidationPassed",
      ],
    };
  }

  // --------------------------------------------------
  // ASSESSMENT CONTEXT REQUIRED
  //
  // Micro-step assessment is valid only for one governed
  // Guided Construction component and Step 1–3.
  // --------------------------------------------------

  if (
    !componentRules ||
    !stepRules ||
    !validStep
  ) {
    return {
      assessmentStatus:
        "unavailable",

      outcome:
        null,

      frameComponent:
        component || null,

      guidedConstructionStep:
        validStep
          ? step
          : null,

      studentEvidence:
        text || null,

      rule:
        null,

      evidenceBasis: [
        "guidedConstructionAssessmentContextUnavailable",
      ],
    };
  }

  // --------------------------------------------------
  // CLEAR NO-EVIDENCE CONDITIONS
  //
  // These conditions can be established without semantic
  // interpretation.
  //
  // They do not depend on response length because short
  // student responses may still satisfy a Guided
  // Construction micro-step.
  // --------------------------------------------------

  const noObservableStudentEvidence =
    !text ||

    validation
      ?.componentEvidenceLevel ===
      "none" ||

    validation?.diagnosis ===
      "emptyResponse" ||

    validation?.diagnosis ===
      "noComponentEvidence" ||

    isStuckMessage(text) ||

    isWeakFrameResponse(text) ||

    isMetaResponse(text);

  if (
    noObservableStudentEvidence
  ) {
    return {
      assessmentStatus:
        "established",

      outcome:
        step === 3
          ? GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
              .NO_USABLE_FINAL_STEP_EVIDENCE
          : GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
              .INSUFFICIENT_MICRO_STEP_EVIDENCE,

      frameComponent:
        component,

      guidedConstructionStep:
        step,

      studentEvidence:
        text || null,

      rule:
        structuredClone(
          stepRules
        ),

      evidenceBasis: [
        "noObservableGuidedConstructionEvidence",
      ],
    };
  }

  // --------------------------------------------------
  // BOUNDED SEMANTIC EVIDENCE
  //
  // Component-specific micro-step meaning cannot safely
  // be inferred from length, token overlap, or the full
  // component validator alone.
  //
  // The bounded semantic-evidence provider supplies the
  // governed micro-step evidence consumed by this assessment.
  //
  // JavaScript retains final authority by converting the
  // supplied evidence into one governed outcome here.
  // --------------------------------------------------

  if (!semanticEvidence) {
    return {
      assessmentStatus:
        "semanticEvidenceRequired",

      outcome:
        null,

      frameComponent:
        component,

      guidedConstructionStep:
        step,

      studentEvidence:
        text,

      rule:
        structuredClone(
          stepRules
        ),

      evidenceBasis: [
        "observableStudentEvidencePresent",
        "microStepMeaningRequiresBoundedSemanticEvidence",
      ],
    };
  }

  const semanticAssessmentEstablished =
    semanticEvidence
      ?.assessmentEstablished ===
      true;

  const sufficientForCurrentStep =
    semanticEvidence
      ?.sufficientForCurrentStep ===
      true;

  const usableForFinalStep =
    semanticEvidence
      ?.usableForFinalStep ===
      true;

  if (!semanticAssessmentEstablished) {
    return {
      assessmentStatus:
        "semanticEvidenceUnavailable",

      outcome:
        null,

      frameComponent:
        component,

      guidedConstructionStep:
        step,

      studentEvidence:
        text,

      rule:
        structuredClone(
          stepRules
        ),

      evidenceBasis: [
        "boundedSemanticAssessmentNotEstablished",
      ],
    };
  }

  // --------------------------------------------------
  // STEP 1 / STEP 2
  // --------------------------------------------------

  if (
    step === 1 ||
    step === 2
  ) {
    return {
      assessmentStatus:
        "established",

      outcome:
        sufficientForCurrentStep
          ? GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
              .SUFFICIENT_MICRO_STEP_EVIDENCE
          : GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
              .INSUFFICIENT_MICRO_STEP_EVIDENCE,

      frameComponent:
        component,

      guidedConstructionStep:
        step,

      studentEvidence:
        text,

      rule:
        structuredClone(
          stepRules
        ),

      evidenceBasis: [
        sufficientForCurrentStep
          ? "boundedSemanticEvidenceSupportsCurrentMicroStep"
          : "boundedSemanticEvidenceDoesNotSupportCurrentMicroStep",
      ],
    };
  }

  // --------------------------------------------------
  // STEP 3
  //
  // Full component acceptance was already checked above.
  //
  // At Step 3 the remaining question is whether the
  // student's response contains usable final-step
  // formulation or synthesis that Kaw can continue
  // coaching without supplying the missing thinking.
  // --------------------------------------------------

  return {
    assessmentStatus:
      "established",

    outcome:
      usableForFinalStep
        ? GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
            .USABLE_FINAL_STEP_EVIDENCE
        : GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
            .NO_USABLE_FINAL_STEP_EVIDENCE,

    frameComponent:
      component,

    guidedConstructionStep:
      step,

    studentEvidence:
      text,

    rule:
      structuredClone(
        stepRules
      ),

    evidenceBasis: [
      usableForFinalStep
        ? "boundedSemanticEvidenceSupportsUsableFinalStep"
        : "boundedSemanticEvidenceDoesNotSupportUsableFinalStep",
    ],
  };
}

// ======================================================
// GUIDED CONSTRUCTION SEMANTIC EVIDENCE
// ======================================================
//
// Provides bounded semantic evidence for one active
// Guided Construction micro-step.
//
// The teacher-authored GUIDED_CONSTRUCTION_RULES remain
// authoritative.
//
// AI may determine only whether observable student
// meaning supports each predetermined criterion for the
// current micro-step.
//
// AI does not:
//
// • determine whether Guided Construction begins;
// • determine the active Guided Construction step;
// • determine progression;
// • return STAY, ADVANCE, COMPLETE, or ENDPOINT;
// • validate the completed Frame component;
// • rewrite or improve student work;
// • generate missing student thinking;
// • mutate runtime state.
//
// JavaScript converts the bounded criterion evidence into
// the semantic artifact consumed by
// assessGuidedConstructionEvidence().
//
// ======================================================

async function getGuidedConstructionSemanticEvidence({
  state = null,
  response = "",
  frameComponent = "",
  guidedConstructionStep = null,
} = {}) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const studentResponse =
    cleanText(response);

  const component =
    cleanText(
      frameComponent
    );

  const step =
    Number(
      guidedConstructionStep
    );

  const componentRules =
    GUIDED_CONSTRUCTION_RULES
      ?.[component] || null;

  const stepRules =
    componentRules
      ?.steps
      ?.[step] || null;

  const sufficientCriteria =
    Array.isArray(
      stepRules
        ?.sufficientMicroStepEvidence
    )
      ? stepRules
          .sufficientMicroStepEvidence
          .map(cleanText)
          .filter(Boolean)
      : [];

  const insufficientCriteria =
    Array.isArray(
      stepRules
        ?.insufficientMicroStepEvidence
    )
      ? stepRules
          .insufficientMicroStepEvidence
          .map(cleanText)
          .filter(Boolean)
      : [];

  const validContext =
    Boolean(
      studentResponse &&
      componentRules &&
      stepRules &&
      Number.isInteger(step) &&
      step >= 1 &&
      step <= 3 &&
      sufficientCriteria.length > 0
    );

  if (!validContext) {
    return {
      assessmentEstablished:
        false,

      sufficientForCurrentStep:
        false,

      usableForFinalStep:
        false,

      criterionEvidence:
        [],

      confidence:
        0,

      source:
        "notRequested",
    };
  }

  // --------------------------------------------------
  // INSTRUCTIONAL CONTEXT
  //
  // Only accepted Frame content and student-owned Guided
  // Construction evidence may contextualize the current
  // micro-step.
  //
  // --------------------------------------------------

  const keyTopic =
    cleanText(
      safeState?.frame
        ?.keyTopic || ""
    );

  const isAbout =
    cleanText(
      safeState?.frame
        ?.isAbout || ""
    );

  const mainIdeas =
    getIdeaList(safeState)
      .map(cleanText)
      .filter(Boolean);

  const details =
    Array.isArray(
      safeState?.frame
        ?.details
    )
      ? safeState.frame.details.map(
          (bucket) =>
            Array.isArray(bucket)
              ? bucket
                  .map(cleanText)
                  .filter(Boolean)
              : []
        )
      : [];

  const pending =
    safeState?.pending &&
    typeof safeState.pending ===
      "object"
      ? safeState.pending
      : {};

  const currentMainIdea =
    component === "details" &&
    Number.isInteger(
      pending?.index
    )
      ? cleanText(
          mainIdeas[
            pending.index
          ] || ""
        )
      : "";

  const priorGuidedEvidence =
    pending
      ?.guidedConstructionEvidence &&
    typeof pending
      .guidedConstructionEvidence ===
      "object"
      ? structuredClone(
          pending
            .guidedConstructionEvidence
        )
      : {};

  const system = `You provide bounded semantic evidence for Guided Construction in Kaw Companion, which supports students using the KU Framing Routine.

Your responsibility is extremely narrow.

The student's current Guided Construction component, current micro-step, teacher-authored purpose, and predetermined evidence criteria will be supplied.

Evaluate only whether the student's current response provides observable semantic evidence for each supplied sufficient criterion.

You are an evidence observer only.

You do not determine:
- whether Guided Construction should begin;
- whether the student should stay on or advance from a step;
- whether the completed Frame component is valid;
- whether instruction should progress;
- what Kaw should ask next;
- what support the student needs.

Rules:
- Evaluate the student's actual words.
- Preserve imperfect grammar, spelling, fragments, words, and phrases when they communicate the required thinking.
- Do not impose a minimum length.
- Do not require sentence form unless the supplied criterion explicitly requires it.
- Use accepted Frame context only to understand the student's response.
- Use prior Guided Construction evidence only to determine whether the current response builds from, extends, relates to, or merely repeats the student's own earlier thinking.
- Never use context to supply an idea, relationship, organizer, explanation, significance, or synthesis the student did not express.
- Do not rewrite, improve, complete, combine, or paraphrase the student's thinking.
- Do not generate a possible answer.
- Do not judge factual accuracy unless factual relationship is explicitly required by the supplied criterion.
- Do not infer hidden understanding, intent, effort, motivation, or emotion.
- Treat each sufficient criterion independently.
- criterionEvidence must contain exactly one item for each supplied sufficient criterion, using its supplied zero-based criterionIndex.
- supported must be true only when the student's current response observably satisfies that criterion in the supplied context.
- confidence represents how clearly the student's words support your criterion judgments.
- Return semantic evidence only.
- Return only the required JSON object.`;

  const user = `Guided Construction component:
${component}

Guided Construction step:
${step}

Current thinking operation:
${cleanText(
  stepRules?.operation || ""
)}

Teacher-authored purpose:
${cleanText(
  stepRules?.purpose || ""
)}

Sufficient micro-step evidence criteria:
${JSON.stringify(
  sufficientCriteria.map(
    (criterion, criterionIndex) => ({
      criterionIndex,
      criterion,
    })
  ),
  null,
  2
)}

Teacher-authored insufficient-evidence descriptions:
${JSON.stringify(
  insufficientCriteria,
  null,
  2
)}

Accepted Frame context:

Key Topic:
${keyTopic || "(not available)"}

Is About:
${isAbout || "(not available)"}

Accepted Main Ideas:
${JSON.stringify(
  mainIdeas,
  null,
  2
)}

Accepted Essential Details:
${JSON.stringify(
  details,
  null,
  2
)}

Current accepted Main Idea for Essential Detail work:
${currentMainIdea || "(not applicable)"}

Prior student-owned Guided Construction evidence:
${JSON.stringify(
  priorGuidedEvidence,
  null,
  2
)}

Student's current response:
"${studentResponse}"

Report only whether the student's current words provide evidence for each predetermined sufficient criterion.`;

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
              "guided_construction_semantic_evidence",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                criterionEvidence: {
                  type:
                    "array",

                  items: {
                    type:
                      "object",

                    additionalProperties:
                      false,

                    properties: {
                      criterionIndex: {
                        type:
                          "integer",

                        minimum:
                          0,
                      },

                      supported: {
                        type:
                          "boolean",
                      },
                    },

                    required: [
                      "criterionIndex",
                      "supported",
                    ],
                  },
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
                "criterionEvidence",
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
        resp?.choices?.[0]
          ?.message?.content || "{}"
      );

    const rawCriterionEvidence =
      Array.isArray(
        parsed?.criterionEvidence
      )
        ? parsed.criterionEvidence
        : [];

    // --------------------------------------------------
    // JAVASCRIPT SANITIZATION
    //
    // AI may return evidence only for the criteria that
    // actually exist in the teacher-authored current
    // micro-step.
    //
    // Missing, duplicate, or out-of-range criterion
    // judgments cannot silently establish sufficiency.
    // --------------------------------------------------

    const criterionEvidence =
      sufficientCriteria.map(
        (
          criterion,
          criterionIndex
        ) => {
          const matchingEvidence =
            rawCriterionEvidence.find(
              (item) =>
                Number(
                  item?.criterionIndex
                ) ===
                criterionIndex
            );

          return {
            criterionIndex,

            criterion,

            supported:
              matchingEvidence
                ?.supported === true,
          };
        }
      );

    const confidence =
      Number(
        parsed?.confidence || 0
      );

    const normalizedConfidence =
      Number.isFinite(confidence)
        ? Math.max(
            0,
            Math.min(
              confidence,
              1
            )
          )
        : 0;

    // --------------------------------------------------
    // DETERMINISTIC MICRO-STEP EVIDENCE DECISION
    //
    // AI supplies criterion observations.
    //
    // JavaScript decides whether those observations are
    // sufficient to establish the current micro-step.
    //
    // Every teacher-authored sufficient criterion must
    // be observably supported.
    //
    // --------------------------------------------------

    const allCriteriaSupported =
      criterionEvidence.length ===
        sufficientCriteria.length &&

      criterionEvidence.length > 0 &&

      criterionEvidence.every(
        (item) =>
          item.supported === true
      );

    const assessmentEstablished =
      normalizedConfidence >=
      0.9;

    const sufficientForCurrentStep =
      assessmentEstablished &&
      allCriteriaSupported;

    const usableForFinalStep =
      step === 3 &&
      sufficientForCurrentStep;

    return {
      assessmentEstablished,

      sufficientForCurrentStep,

      usableForFinalStep,

      criterionEvidence,

      confidence:
        normalizedConfidence,

      source:
        "aiBoundedGuidedConstructionSemanticEvidence",
    };
  } catch (error) {
    console.error(
      "Guided Construction semantic evidence error:",
      error
    );

    return {
      assessmentEstablished:
        false,

      sufficientForCurrentStep:
        false,

      usableForFinalStep:
        false,

      criterionEvidence:
        [],

      confidence:
        0,

      source:
        "semanticEvidenceUnavailable",
    };
  }
}

// ======================================================
// GUIDED CONSTRUCTION PROGRESSION DECISION
// ======================================================
//
// Converts one established Guided Construction evidence
// assessment into one deterministic pathway decision.
//
// This is the progression-decision brain for Progressive
// Support Stage 3.
//
// It answers only:
//
// "Given the current Guided Construction step and the
// established evidence outcome, what should happen next
// inside Guided Construction?"
//
// The normal governed component validator has already
// received first authority through
// assessGuidedConstructionEvidence().
//
// This decision layer does not:
//
// • validate student work;
// • request semantic evidence;
// • determine whether Guided Construction begins;
// • mutate pending state;
// • save student evidence;
// • select an Instructional Contract;
// • advance the larger Frame;
// • generate student-facing communication.
//
// These decisions are applied to bounded Guided
// Construction state by applyGuidedConstructionProgression().
//
// ======================================================

const GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS =
  Object.freeze({
    STAY_CURRENT_STEP:
      "stayCurrentStep",

    ADVANCE_TO_NEXT_STEP:
      "advanceToNextStep",

    COMPONENT_COMPLETE:
      "componentComplete",

    REFINE_FINAL_STEP:
      "refineFinalStep",

    REPHRASE_FINAL_STEP:
      "rephraseFinalStep",

    ADDITIONAL_SUPPORT_ENDPOINT:
      "additionalSupportEndpoint",
  });

function buildGuidedConstructionProgressionDecision({
  evidenceAssessment = null,
  finalRephraseUsed = false,
} = {}) {
  const assessment =
    evidenceAssessment &&
    typeof evidenceAssessment ===
      "object"
      ? evidenceAssessment
      : null;

  if (!assessment) {
    return {
      decisionStatus:
        "unavailable",

      decision:
        null,

      currentStep:
        null,

      nextStep:
        null,

      frameComponent:
        null,

      saveCurrentEvidence:
        false,

      finalRephraseUsed:
        finalRephraseUsed === true,

      decisionBasis: [
        "guidedConstructionEvidenceAssessmentUnavailable",
      ],
    };
  }

  const frameComponent =
    cleanText(
      assessment
        ?.frameComponent || ""
    );

  const currentStep =
    Number(
      assessment
        ?.guidedConstructionStep
    );

  const outcome =
    cleanText(
      assessment
        ?.outcome || ""
    );

  const validStep =
    Number.isInteger(
      currentStep
    ) &&
    currentStep >= 1 &&
    currentStep <= 3;

  const assessmentEstablished =
    assessment
      ?.assessmentStatus ===
      "established" ||

    assessment
      ?.assessmentStatus ===
      "complete";

  if (
    !frameComponent ||
    !GUIDED_CONSTRUCTION_RULES
      ?.[frameComponent] ||
    !validStep ||
    !assessmentEstablished ||
    !outcome
  ) {
    return {
      decisionStatus:
        "unavailable",

      decision:
        null,

      currentStep:
        validStep
          ? currentStep
          : null,

      nextStep:
        null,

      frameComponent:
        frameComponent || null,

      saveCurrentEvidence:
        false,

      finalRephraseUsed:
        finalRephraseUsed === true,

      decisionBasis: [
        "guidedConstructionProgressionContextUnavailable",
      ],
    };
  }

  // --------------------------------------------------
  // COMPONENT COMPLETE
  //
  // Full governed component validation overrides the
  // Guided Construction pathway immediately.
  //
  // Guided Construction does not create an additional
  // instructional hoop after successful validation.
  // --------------------------------------------------

  if (
    outcome ===
    GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
      .COMPONENT_COMPLETE
  ) {
    return {
      decisionStatus:
        "established",

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .COMPONENT_COMPLETE,

      currentStep,

      nextStep:
        null,

      frameComponent,

      saveCurrentEvidence:
        false,

      finalRephraseUsed:
        finalRephraseUsed === true,

      decisionBasis: [
        "governedComponentValidationPassed",
        "normalComponentProgressionAuthoritative",
      ],
    };
  }

  // --------------------------------------------------
  // STEP 1 / STEP 2
  //
  // Insufficient micro-step evidence:
  // remain on the same thinking operation.
  //
  // Sufficient micro-step evidence:
  // preserve the student's evidence and advance exactly
  // one Guided Construction step.
  //
  // --------------------------------------------------

  if (
    currentStep === 1 ||
    currentStep === 2
  ) {
    if (
      outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .INSUFFICIENT_MICRO_STEP_EVIDENCE
    ) {
      return {
        decisionStatus:
          "established",

        decision:
          GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
            .STAY_CURRENT_STEP,

        currentStep,

        nextStep:
          currentStep,

        frameComponent,

        saveCurrentEvidence:
          false,

        finalRephraseUsed:
          finalRephraseUsed === true,

        decisionBasis: [
          "insufficientCurrentMicroStepEvidence",
          "sameThinkingOperationRemainsAuthoritative",
        ],
      };
    }

    if (
      outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .SUFFICIENT_MICRO_STEP_EVIDENCE
    ) {
      return {
        decisionStatus:
          "established",

        decision:
          GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
            .ADVANCE_TO_NEXT_STEP,

        currentStep,

        nextStep:
          currentStep + 1,

        frameComponent,

        saveCurrentEvidence:
          true,

        finalRephraseUsed:
          finalRephraseUsed === true,

        decisionBasis: [
          "sufficientCurrentMicroStepEvidence",
          "advanceExactlyOneGuidedConstructionStep",
        ],
      };
    }

    return {
      decisionStatus:
        "unavailable",

      decision:
        null,

      currentStep,

      nextStep:
        null,

      frameComponent,

      saveCurrentEvidence:
        false,

      finalRephraseUsed:
        finalRephraseUsed === true,

      decisionBasis: [
        `unexpectedEvidenceOutcome:${outcome}`,
      ],
    };
  }

  // --------------------------------------------------
  // STEP 3 — FINAL GUIDED CONSTRUCTION OPERATION
  //
  // There is no Guided Construction Step 4.
  //
  // The normal component validator already received
  // first authority above.
  //
  // Remaining possibilities:
  //
  // • usable but incomplete final-step thinking
  //     → continue coaching Step 3;
  //
  // • first response with no usable final-step evidence
  //     → rephrase the same Step-3 operation once;
  //
  // • persistent no usable evidence after that rephrase
  //     → additional-support endpoint.
  //
  // --------------------------------------------------

  if (
    outcome ===
    GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
      .USABLE_FINAL_STEP_EVIDENCE
  ) {
    return {
      decisionStatus:
        "established",

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .REFINE_FINAL_STEP,

      currentStep:
        3,

      nextStep:
        3,

      frameComponent,

      saveCurrentEvidence:
        true,

      finalRephraseUsed:
        finalRephraseUsed === true,

      decisionBasis: [
        "usableButIncompleteFinalStepEvidence",
        "continueSameFinalThinkingOperation",
      ],
    };
  }

  if (
    outcome ===
    GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
      .NO_USABLE_FINAL_STEP_EVIDENCE
  ) {
    if (
      finalRephraseUsed !== true
    ) {
      return {
        decisionStatus:
          "established",

        decision:
          GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
            .REPHRASE_FINAL_STEP,

        currentStep:
          3,

        nextStep:
          3,

        frameComponent,

        saveCurrentEvidence:
          false,

        finalRephraseUsed:
          true,

        decisionBasis: [
          "noUsableFinalStepEvidence",
          "finalStepRephraseNotYetUsed",
          "rephraseSameThinkingOperationOnce",
        ],
      };
    }

    return {
      decisionStatus:
        "established",

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .ADDITIONAL_SUPPORT_ENDPOINT,

      currentStep:
        3,

      nextStep:
        3,

      frameComponent,

      saveCurrentEvidence:
        false,

      finalRephraseUsed:
        true,

      decisionBasis: [
        "noUsableFinalStepEvidence",
        "finalStepRephraseAlreadyUsed",
        "guidedConstructionEndpointReached",
      ],
    };
  }

  return {
    decisionStatus:
      "unavailable",

    decision:
      null,

    currentStep:
      3,

    nextStep:
      null,

    frameComponent,

    saveCurrentEvidence:
      false,

    finalRephraseUsed:
      finalRephraseUsed === true,

    decisionBasis: [
      `unexpectedEvidenceOutcome:${outcome}`,
    ],
  };
}

// ======================================================
// GUIDED CONSTRUCTION STATE UPDATER
// ======================================================
//
// Applies one already-established Guided Construction
// progression decision to Guided Construction-owned
// pending-state metadata only.
//
// This updater may modify only Guided Construction-owned
// pending-state metadata:
//
// • guidedConstructionStep;
// • guidedConstructionEvidence;
// • guidedConstructionFinalRephraseUsed;
// • guidedConstructionLocation;
// • guidedConstructionAdditionalSupportEndpoint.
//
// It does not:
//
// • save a completed Frame component;
// • advance the larger Frame;
// • change pending.type;
// • change captureMode;
// • change interactionMode;
// • change Instructional Contract;
// • determine Instructional Situation;
// • determine evidence sufficiency;
// • generate student-facing communication.
//
// Full component completion remains owned by the normal
// governed component runtime.
//
// ======================================================

function applyGuidedConstructionProgression({
  state = null,
  progressionDecision = null,
  evidenceAssessment = null,
  instructionalLocation = null,
} = {}) {
  if (
    !state ||
    typeof state !== "object" ||
    !state?.pending ||
    typeof state.pending !== "object"
  ) {
    return {
      applied:
        false,

      reason:
        "pendingStateUnavailable",
    };
  }

  const decision =
    progressionDecision &&
    typeof progressionDecision ===
      "object"
      ? progressionDecision
      : null;

  const assessment =
    evidenceAssessment &&
    typeof evidenceAssessment ===
      "object"
      ? evidenceAssessment
      : null;

  const location =
    instructionalLocation &&
    typeof instructionalLocation ===
      "object"
      ? instructionalLocation
      : null;

  if (
    !decision ||
    decision?.decisionStatus !==
      "established" ||
    !decision?.decision
  ) {
    return {
      applied:
        false,

      reason:
        "progressionDecisionUnavailable",
    };
  }

  const pending =
    state.pending;

  const currentStep =
    Number(
      pending
        ?.guidedConstructionStep
    );

  const decisionCurrentStep =
    Number(
      decision
        ?.currentStep
    );

  const validCurrentStep =
    Number.isInteger(
      currentStep
    ) &&
    currentStep >= 1 &&
    currentStep <= 3;

  const validDecisionStep =
    Number.isInteger(
      decisionCurrentStep
    ) &&
    decisionCurrentStep >= 1 &&
    decisionCurrentStep <= 3;

  if (
    !validCurrentStep ||
    !validDecisionStep ||
    currentStep !==
      decisionCurrentStep
  ) {
    return {
      applied:
        false,

      reason:
        "guidedConstructionStepMismatch",
    };
  }

  // --------------------------------------------------
  // LOCATION PRESERVATION
  //
  // Store the exact Guided Construction location only
  // when a valid location artifact has been established.
  //
  // Later preservation logic will compare this stored
  // location against the student's current location.
  // --------------------------------------------------

  if (
    location?.locationEstablished ===
      true
  ) {
    pending.guidedConstructionLocation =
      structuredClone(
        location
      );
  }

  // --------------------------------------------------
  // STUDENT-OWNED GUIDED EVIDENCE
  //
  // Save only evidence that Step 6 explicitly authorized
  // for preservation.
  //
  // Evidence remains separate by Guided Construction
  // step so later coaching can reconnect only to actual
  // student thinking.
  // --------------------------------------------------

  if (
    decision
      ?.saveCurrentEvidence ===
      true
  ) {
    const studentEvidence =
      cleanText(
        assessment
          ?.studentEvidence || ""
      );

    if (studentEvidence) {
      const existingEvidence =
        pending
          ?.guidedConstructionEvidence &&
        typeof pending
          .guidedConstructionEvidence ===
          "object"
          ? pending
              .guidedConstructionEvidence
          : {};

      pending.guidedConstructionEvidence = {
        ...structuredClone(
          existingEvidence
        ),

        [String(currentStep)]: {
          step:
            currentStep,

          evidence:
            studentEvidence,
        },
      };
    }
  }

  // --------------------------------------------------
  // APPLY DETERMINISTIC PATHWAY DECISION
  // --------------------------------------------------

  switch (decision.decision) {
    case GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
      .STAY_CURRENT_STEP: {
      pending.guidedConstructionStep =
        currentStep;

      return {
        applied:
          true,

        decision:
          decision.decision,

        guidedConstructionStep:
          pending
            .guidedConstructionStep,
      };
    }

    case GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
      .ADVANCE_TO_NEXT_STEP: {
      const nextStep =
        Number(
          decision?.nextStep
        );

      if (
        !Number.isInteger(
          nextStep
        ) ||
        nextStep < 1 ||
        nextStep > 3 ||
        nextStep !==
          currentStep + 1
      ) {
        return {
          applied:
            false,

          reason:
            "invalidGuidedConstructionAdvance",
        };
      }

      pending.guidedConstructionStep =
        nextStep;

      return {
        applied:
          true,

        decision:
          decision.decision,

        guidedConstructionStep:
          nextStep,
      };
    }

    case GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
      .REFINE_FINAL_STEP: {
      pending.guidedConstructionStep =
        3;

      return {
        applied:
          true,

        decision:
          decision.decision,

        guidedConstructionStep:
          3,
      };
    }

    case GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
      .REPHRASE_FINAL_STEP: {
      pending.guidedConstructionStep =
        3;

      pending.guidedConstructionFinalRephraseUsed =
        true;

      return {
        applied:
          true,

        decision:
          decision.decision,

        guidedConstructionStep:
          3,

        guidedConstructionFinalRephraseUsed:
          true,
      };
    }

    case GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
  .ADDITIONAL_SUPPORT_ENDPOINT: {
  pending.guidedConstructionStep =
    3;

  pending.guidedConstructionFinalRephraseUsed =
    true;

  pending.guidedConstructionAdditionalSupportEndpoint =
    true;

  return {
    applied:
      true,

    decision:
      decision.decision,

    guidedConstructionStep:
      3,

    guidedConstructionFinalRephraseUsed:
      true,

    guidedConstructionAdditionalSupportEndpoint:
      true,

    endpointReached:
      true,
  };
}

    case GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
      .COMPONENT_COMPLETE: {
      // ------------------------------------------------
      // NO COMPLETION MUTATION HERE
      //
      // The normal component runtime owns acceptance,
      // saving, confirmation, and Frame progression.
      //
      // Guided Construction simply yields authority.
      // ------------------------------------------------

      return {
        applied:
          true,

        decision:
          decision.decision,

        guidedConstructionStep:
          currentStep,

        yieldsToNormalComponentProgression:
          true,
      };
    }

    default:
      return {
        applied:
          false,

        reason:
          "unsupportedGuidedConstructionDecision",
      };
  }
}

// ======================================================
// GUIDED CONSTRUCTION PENDING-STATE PRESERVATION
// ======================================================
//
// Rebuilds pending state while preserving Guided
// Construction-owned metadata only when the replacement
// pending object still represents the same exact
// instructional location.
//
// This prevents ordinary component failure branches from
// accidentally erasing an active Guided Construction
// pathway when they reconstruct pending state.
//
// It also prevents Guided Construction metadata from
// leaking into a different Main Idea, Essential Detail,
// component, capture mode, or interaction pathway.
//
// This helper does not:
//
// • determine whether Guided Construction begins;
// • determine Guided Construction progression;
// • validate student evidence;
// • advance the Frame;
// • select an Instructional Contract;
// • generate communication.
//
// ======================================================

function buildPendingWithGuidedConstructionPreservation(
  state,
  nextPending
) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const currentPending =
    safeState?.pending &&
    typeof safeState.pending === "object"
      ? safeState.pending
      : null;

  const replacementPending =
    nextPending &&
    typeof nextPending === "object"
      ? structuredClone(nextPending)
      : null;

  if (!replacementPending) {
    return replacementPending;
  }

  if (!currentPending) {
    return replacementPending;
  }

  const activeContext =
    getActiveGuidedConstructionContext(
      safeState
    );

  if (
    activeContext?.active !== true
  ) {
    return replacementPending;
  }

  const storedLocation =
    currentPending
      ?.guidedConstructionLocation &&
    typeof currentPending
      .guidedConstructionLocation ===
      "object"
      ? structuredClone(
          currentPending
            .guidedConstructionLocation
        )
      : buildGuidedConstructionInstructionalLocation(
          safeState
        );

  if (
    storedLocation
      ?.locationEstablished !== true
  ) {
    return replacementPending;
  }

  // --------------------------------------------------
  // BUILD CANDIDATE LOCATION
  //
  // Evaluate the replacement pending object as if it
  // were active, without mutating the real runtime yet.
  // --------------------------------------------------

  const candidateState = {
    ...safeState,

    pending:
      structuredClone(
        replacementPending
      ),
  };

  const candidateLocation =
    buildGuidedConstructionInstructionalLocation(
      candidateState
    );

  const sameInstructionalLocation =
    isSameGuidedConstructionInstructionalLocation(
      storedLocation,
      candidateLocation
    );

  if (!sameInstructionalLocation) {
    return replacementPending;
  }

  // --------------------------------------------------
  // PRESERVE ONLY GUIDED CONSTRUCTION-OWNED METADATA
  //
  // Do not spread the entire old pending object.
  // --------------------------------------------------

  const preservedPending = {
    ...replacementPending,

    progressiveSupportStage:
      currentPending
        .progressiveSupportStage,

    guidedConstructionStep:
      currentPending
        .guidedConstructionStep,

    guidedConstructionLocation:
      structuredClone(
        storedLocation
      ),
  };

  if (
    currentPending
      ?.guidedConstructionEvidence &&
    typeof currentPending
      .guidedConstructionEvidence ===
      "object"
  ) {
    preservedPending
      .guidedConstructionEvidence =
      structuredClone(
        currentPending
          .guidedConstructionEvidence
      );
  }

  if (
    currentPending
      ?.guidedConstructionFinalRephraseUsed ===
      true
  ) {
    preservedPending
      .guidedConstructionFinalRephraseUsed =
      true;
  }

  if (
  currentPending
    ?.guidedConstructionAdditionalSupportEndpoint ===
    true
) {
  preservedPending
    .guidedConstructionAdditionalSupportEndpoint =
    true;
}

  return preservedPending;
}

// ======================================================
// GUIDED CONSTRUCTION CONTINUATION FACADE
// ======================================================
//
// Provides one shared runtime entry point for continuing
// an already-active Guided Construction pathway.
//
// This façade coordinates the Guided Construction
// subsystem in one governed sequence:
//
// 1. confirm active Guided Construction context;
// 2. confirm exact instructional location;
// 3. preserve normal component validation authority;
// 4. establish bounded semantic evidence only when needed;
// 5. assess current micro-step evidence;
// 6. build the deterministic progression decision;
// 7. apply only Guided Construction-owned state changes.
//
// This façade does not:
//
// • determine whether Guided Construction begins;
// • perform normal component validation;
// • save a completed Frame component;
// • advance the larger Frame;
// • select or change an Instructional Contract;
// • change pending.type or captureMode;
// • generate student-facing communication.
//
// Component-specific runtime branches call this
// shared façade for active Guided Construction continuation.
//
// ======================================================

async function continueGuidedConstruction({
  state = null,
  response = "",
  componentValidation = null,
  finalRephraseUsed = false,
} = {}) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : null;

  const validation =
    componentValidation &&
    typeof componentValidation ===
      "object"
      ? componentValidation
      : null;

  // --------------------------------------------------
  // PERSISTENT FINAL-STEP REPHRASE STATE
  //
  // The Guided Construction pathway owns whether the one
  // permitted Step-3 rephrase has already been consumed.
  //
  // Persistent pending state is authoritative across
  // runtime turns. The explicit argument remains supported
  // for deterministic tests and bounded callers, but a
  // caller may never reset previously consumed pathway
  // state by supplying false.
  //
  // Exact-location protection is handled before this state
  // can survive pending reconstruction by
  // buildPendingWithGuidedConstructionPreservation().
  // --------------------------------------------------

  const effectiveFinalRephraseUsed =
    finalRephraseUsed === true ||
    safeState
      ?.pending
      ?.guidedConstructionFinalRephraseUsed ===
      true;

  if (!safeState) {
    return {
      continuationStatus:
        "unavailable",

      reason:
        "stateUnavailable",

      activeContext:
        null,

      instructionalLocation:
        null,

      semanticEvidence:
        null,

      evidenceAssessment:
        null,

      progressionDecision:
        null,

      stateUpdate:
        null,
    };
  }

  // --------------------------------------------------
  // ACTIVE GUIDED CONSTRUCTION CONTEXT
  // --------------------------------------------------

  const activeContext =
    getActiveGuidedConstructionContext(
      safeState
    );

  if (
    activeContext?.active !== true
  ) {
    return {
      continuationStatus:
        "inactive",

      reason:
        "guidedConstructionNotActive",

      activeContext,

      instructionalLocation:
        null,

      semanticEvidence:
        null,

      evidenceAssessment:
        null,

      progressionDecision:
        null,

      stateUpdate:
        null,
    };
  }

  // --------------------------------------------------
  // NORMAL COMPONENT VALIDATION REQUIRED
  //
  // Guided Construction may never assess micro-step
  // progression without first receiving the normal
  // governed component validator's result.
  //
  // --------------------------------------------------

  if (!validation) {
    return {
      continuationStatus:
        "unavailable",

      reason:
        "componentValidationUnavailable",

      activeContext,

      instructionalLocation:
        null,

      semanticEvidence:
        null,

      evidenceAssessment:
        null,

      progressionDecision:
        null,

      stateUpdate:
        null,
    };
  }

  const frameComponent =
    cleanText(
      activeContext
        ?.frameComponent || ""
    );

  const guidedConstructionStep =
    Number(
      activeContext
        ?.guidedConstructionStep
    );

  // --------------------------------------------------
  // EXACT INSTRUCTIONAL LOCATION
  //
  // Guided Construction may continue only at the exact
  // location where the active pathway was established.
  //
  // --------------------------------------------------

  const instructionalLocation =
    buildGuidedConstructionInstructionalLocation(
      safeState
    );

  const storedLocation =
    safeState?.pending
      ?.guidedConstructionLocation &&
    typeof safeState.pending
      .guidedConstructionLocation ===
      "object"
      ? safeState.pending
          .guidedConstructionLocation
      : null;

  const sameInstructionalLocation =
    storedLocation
      ?.locationEstablished === true &&
    instructionalLocation
      ?.locationEstablished === true &&
    isSameGuidedConstructionInstructionalLocation(
      storedLocation,
      instructionalLocation
    );

  if (!sameInstructionalLocation) {
    return {
      continuationStatus:
        "unavailable",

      reason:
        "guidedConstructionLocationMismatch",

      activeContext,

      instructionalLocation,

      semanticEvidence:
        null,

      evidenceAssessment:
        null,

      progressionDecision:
        null,

      stateUpdate:
        null,
    };
  }

  // --------------------------------------------------
  // FIRST ASSESSMENT PASS
  //
  // Deterministic assessment gets the first opportunity
  // to establish:
  //
  // • full component completion; or
  // • clear no-evidence conditions.
  //
  // Semantic evidence is requested only if this assessor
  // explicitly requires it.
  //
  // --------------------------------------------------

  let semanticEvidence =
    null;

  let evidenceAssessment =
    assessGuidedConstructionEvidence({
      state:
        safeState,

      response,

      frameComponent,

      guidedConstructionStep,

      componentValidation:
        validation,

      microStepSemanticEvidence:
        null,
    });

  // --------------------------------------------------
  // BOUNDED SEMANTIC EVIDENCE
  //
  // Only observable student evidence whose micro-step
  // meaning cannot be established deterministically
  // reaches the semantic evidence provider.
  //
  // --------------------------------------------------

  if (
    evidenceAssessment
      ?.assessmentStatus ===
      "semanticEvidenceRequired"
  ) {
    semanticEvidence =
      await getGuidedConstructionSemanticEvidence({
        state:
          safeState,

        response,

        frameComponent,

        guidedConstructionStep,
      });

    evidenceAssessment =
      assessGuidedConstructionEvidence({
        state:
          safeState,

        response,

        frameComponent,

        guidedConstructionStep,

        componentValidation:
          validation,

        microStepSemanticEvidence:
          semanticEvidence,
      });
  }

  // --------------------------------------------------
  // EVIDENCE ASSESSMENT MUST BE ESTABLISHED
  // --------------------------------------------------

  const evidenceAssessmentEstablished =
    evidenceAssessment
      ?.assessmentStatus ===
      "established" ||
    evidenceAssessment
      ?.assessmentStatus ===
      "complete";

  if (!evidenceAssessmentEstablished) {
    return {
      continuationStatus:
        "unavailable",

      reason:
        "guidedConstructionEvidenceAssessmentUnavailable",

      activeContext,

      instructionalLocation,

      semanticEvidence,

      evidenceAssessment,

      progressionDecision:
        null,

      stateUpdate:
        null,
    };
  }

  // --------------------------------------------------
  // DETERMINISTIC PROGRESSION DECISION
  // --------------------------------------------------

  const progressionDecision =
    buildGuidedConstructionProgressionDecision({
      evidenceAssessment,

      finalRephraseUsed:
        effectiveFinalRephraseUsed,
    });

  if (
    progressionDecision
      ?.decisionStatus !==
      "established"
  ) {
    return {
      continuationStatus:
        "unavailable",

      reason:
        "guidedConstructionProgressionDecisionUnavailable",

      activeContext,

      instructionalLocation,

      semanticEvidence,

      evidenceAssessment,

      progressionDecision,

      stateUpdate:
        null,
    };
  }

  // --------------------------------------------------
  // BOUNDED GUIDED CONSTRUCTION STATE UPDATE
  //
  // This may update Guided Construction-owned metadata
  // only.
  //
  // It may not save or progress the completed Frame.
  //
  // --------------------------------------------------

  const stateUpdate =
    applyGuidedConstructionProgression({
      state:
        safeState,

      progressionDecision,

      evidenceAssessment,

      instructionalLocation,
    });

  if (
    stateUpdate?.applied !== true
  ) {
    return {
      continuationStatus:
        "unavailable",

      reason:
        stateUpdate?.reason ||
        "guidedConstructionStateUpdateUnavailable",

      activeContext,

      instructionalLocation,

      semanticEvidence,

      evidenceAssessment,

      progressionDecision,

      stateUpdate,
    };
  }

 const endpointReached =
  stateUpdate
    ?.endpointReached ===
    true;

const additionalSupportEndpoint =
  endpointReached
    ? buildGuidedConstructionAdditionalSupportEndpoint(
        safeState
      )
    : null;

return {
  continuationStatus:
    "established",

  reason:
    null,

  activeContext,

  instructionalLocation,

  semanticEvidence,

  evidenceAssessment,

  progressionDecision,

  stateUpdate,

  yieldsToNormalComponentProgression:
    stateUpdate
      ?.yieldsToNormalComponentProgression ===
      true,

  endpointReached,

  additionalSupportEndpoint,
};
}

// ======================================================
// GUIDED CONSTRUCTION ADDITIONAL-SUPPORT ENDPOINT
// ======================================================
//
// Builds the deterministic instructional endpoint used
// when Guided Construction Step 3 has been exhausted.
//
// This endpoint does not create Stage 4 or Step 4.
//
// It preserves the student's current instructional
// location and directs the student toward external
// instructional resources without supplying the missing
// Frame thinking.
//
// JavaScript establishes this endpoint.
// AI does not decide whether teacher or resource support
// is needed.
//
// Endpoint resumption is governed separately.
//
// ======================================================

function buildGuidedConstructionAdditionalSupportEndpoint(
  state
) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const pending =
    safeState?.pending &&
    typeof safeState.pending === "object"
      ? safeState.pending
      : null;

  if (
    !pending ||
    pending
      ?.guidedConstructionAdditionalSupportEndpoint !==
      true
  ) {
    return null;
  }

  const activeContext =
    getActiveGuidedConstructionContext(
      safeState
    );

  const instructionalLocation =
    buildGuidedConstructionInstructionalLocation(
      safeState
    );

  const storedLocation =
    pending
      ?.guidedConstructionLocation &&
    typeof pending
      .guidedConstructionLocation ===
      "object"
      ? pending.guidedConstructionLocation
      : null;

  const sameInstructionalLocation =
    activeContext?.active === true &&
    storedLocation
      ?.locationEstablished === true &&
    instructionalLocation
      ?.locationEstablished === true &&
    isSameGuidedConstructionInstructionalLocation(
      storedLocation,
      instructionalLocation
    );

  if (!sameInstructionalLocation) {
    return null;
  }

  return {
    artifactType:
      "guidedConstructionAdditionalSupportEndpoint",

    endpointStatus:
      "established",

    frameComponent:
      activeContext
        ?.frameComponent ||
      null,

    guidedConstructionStep:
      activeContext
        ?.guidedConstructionStep ||
      null,

    instructionalLocation:
      structuredClone(
        instructionalLocation
      ),

    supportOptions: [
      "notes",
      "sourceMaterials",
      "assignmentMaterials",
      "teacherSupport",
    ],

    governance: {
      deterministicEndpoint:
        true,

      preserveExactInstructionalLocation:
        true,

      preserveAcceptedFrameContent:
        true,

      preserveGuidedConstructionEvidence:
        true,

      maySupplyStudentThinking:
        false,

      mayCreateAdditionalGuidedConstructionStep:
        false,

      mayCreateAdditionalProgressiveSupportStage:
        false,

      aiChoosesReferral:
        false,
    },
  };
}

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

async function getGuidedConstructionEndpointResumeObservation({
  state = null,
  message = "",
} = {}) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const studentMessage =
    cleanText(message);

  const endpoint =
    buildGuidedConstructionAdditionalSupportEndpoint(
      safeState
    );

  if (
    !studentMessage ||
    endpoint?.endpointStatus !==
      "established"
  ) {
    return {
      observationEstablished:
        false,

      resumeAcknowledgmentObserved:
        false,

      substantiveFrameContentObserved:
        false,

      confidence:
        0,

      source:
        "notRequested",
    };
  }

  const system = `You are a bounded observation layer for Kaw Companion.

The student has reached a governed Guided Construction additional-support endpoint.

Kaw previously directed the student to consult one of these external supports:
- notes
- source materials
- assignment materials
- teacher support

Your task is limited to making two independent observations about the student's current message.

OBSERVATION 1 — RESUME ACKNOWLEDGMENT

Determine whether the message communicates that the student is ready to return to the same instructional location after the support opportunity.

A resume acknowledgment may be expressed in many natural ways.

Examples of meaning that may count:
- the student says they are ready to continue;
- the student says they checked or reviewed a support;
- the student says they received help;
- the student otherwise clearly indicates they want to return to the work.

Do not require exact wording.

OBSERVATION 2 — SUBSTANTIVE FRAME CONTENT

Determine whether the message also contains an attempted contribution to the student's actual Frame content.

This is only a presence-or-absence observation.

Substantive Frame content means the student expresses subject-matter thinking that could be part of the Frame component they are currently working on.

Do not judge whether that thinking is:
- correct;
- sufficient;
- strong;
- relevant enough;
- complete;
- ready to progress;
- valid as a finished Frame component.

Statements such as:
- "I checked my notes";
- "I talked to my teacher";
- "I'm ready";
- "okay, let's continue"

are support acknowledgments by themselves and are not substantive Frame content.

A message may contain both observations at the same time.

For example, a student may say they checked their notes and then immediately offer their own Frame thinking. In that case:
- resumeAcknowledgmentObserved=true
- substantiveFrameContentObserved=true

Do not determine:
- whether Guided Construction should resume;
- whether the instructional location is still valid;
- whether the student completed the support successfully;
- whether the student now understands the content;
- whether their Frame thinking is correct or sufficient;
- what Kaw should teach next;
- whether progression should occur.

Return only the required JSON object.`;

  const user = `Student message:
"${studentMessage}"

Report only:

1. whether the message communicates readiness to return from the additional-support endpoint; and
2. whether the same message contains any attempted substantive Frame content.

Treat these as independent observations.`;

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
              "guided_construction_endpoint_resume_observation",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                resumeAcknowledgmentObserved: {
                  type:
                    "boolean",
                },

                substantiveFrameContentObserved: {
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
                "resumeAcknowledgmentObserved",
                "substantiveFrameContentObserved",
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
        response?.choices?.[0]
          ?.message?.content || "{}"
      );

    const confidence =
      Number(
        parsed?.confidence || 0
      );

    const normalizedConfidence =
      Number.isFinite(confidence)
        ? Math.max(
            0,
            Math.min(
              confidence,
              1
            )
          )
        : 0;

    const observationEstablished =
      normalizedConfidence >=
      0.9;

    return {
      observationEstablished,

      resumeAcknowledgmentObserved:
        observationEstablished &&
        parsed
          ?.resumeAcknowledgmentObserved ===
          true,

      substantiveFrameContentObserved:
        observationEstablished &&
        parsed
          ?.substantiveFrameContentObserved ===
          true,

      confidence:
        normalizedConfidence,

      source:
        "aiBoundedEndpointResumeObservation",
    };
  } catch (error) {
    console.error(
      "Guided Construction endpoint resume observation error:",
      error
    );

    return {
      observationEstablished:
        false,

      resumeAcknowledgmentObserved:
        false,

      substantiveFrameContentObserved:
        false,

      confidence:
        0,

      source:
        "endpointResumeObservationUnavailable",
    };
  }
}

function resumeGuidedConstructionAdditionalSupportEndpoint(
  state,
  resumeObservation
) {
  if (
    !state ||
    typeof state !== "object" ||
    !state?.pending ||
    typeof state.pending !== "object"
  ) {
    return {
      resumed:
        false,

      reason:
        "pendingStateUnavailable",
    };
  }

  const observation =
    resumeObservation &&
    typeof resumeObservation === "object"
      ? resumeObservation
      : null;

  if (
    !observation ||
    observation
      ?.observationEstablished !==
      true ||
    observation
      ?.resumeAcknowledgmentObserved !==
      true
  ) {
    return {
      resumed:
        false,

      reason:
        "resumeAcknowledgmentNotEstablished",
    };
  }

if (
  observation
    ?.substantiveFrameContentObserved !==
    false
) {
  return {
    resumed:
      false,

    reason:
      "substantiveFrameContentObserved",
  };
}
  
  const endpoint =
    buildGuidedConstructionAdditionalSupportEndpoint(
      state
    );

  if (
    endpoint?.endpointStatus !==
    "established"
  ) {
    return {
      resumed:
        false,

      reason:
        "additionalSupportEndpointNotEstablished",
    };
  }

  delete state.pending
    .guidedConstructionAdditionalSupportEndpoint;

  delete state.pending
    .guidedConstructionAdditionalSupportEndpointArtifact;

  return {
    resumed:
      true,

    reason:
      null,

    frameComponent:
      endpoint.frameComponent,

    guidedConstructionStep:
      endpoint.guidedConstructionStep,

    instructionalLocation:
      structuredClone(
        endpoint.instructionalLocation
      ),
  };
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
// governed Frame components in the current scope.
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
      componentContribution:
      observationReport
        ?.componentContribution &&
      typeof observationReport
        .componentContribution ===
        "object"
        ? structuredClone(
            observationReport
              .componentContribution
          )
        : {
            observed:
              false,

            evidenceText:
              "",
          },
  
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
// Current authoritative role:
//
// This finding deterministically interprets governed
// interaction evidence for downstream Instructional
// Situation reasoning.
//
// It does not itself select the Instructional Contract,
// determine support, progression, pending state, or
// generate communication.
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

   // --------------------------------------------------
  // OBSERVABLE COMPONENT CONTRIBUTION
  //
  // The AI Observation Layer may report only whether the
  // student's current words contain candidate content for
  // the active Frame component.
  //
  // It does not determine validity, sufficiency,
  // progression, or instructional situation.
  //
  // --------------------------------------------------

  const componentContribution =
    interactionAssessment
      ?.componentContribution &&
    typeof interactionAssessment
      .componentContribution ===
      "object"
      ? interactionAssessment
          .componentContribution
      : {
          observed:
            false,

          evidenceText:
            "",
        };

  const componentContributionObserved =
    componentContribution
      ?.observed === true;

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

    const interactionOnlyObservationPresent =
    observations.some(
      (observation) =>
        interactionOnlyCategories.has(
          cleanText(
            observation?.category
          )
        )
    );

  const responseFunctionsOnlyAsInteraction =
    componentCaptureActive &&
    interactionOnlyObservationPresent &&
    componentContributionObserved !==
      true;
  
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

      componentContribution:
        structuredClone(
          componentContribution
        ),

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

    progressionAuthority:
      false,
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
// Current authoritative role:
//
// The engine deterministically establishes and stores one
// governed Instructional Situation for downstream contract
// selection and progression authorization.
//
// It does not itself execute progression, mutate pending
// state, or generate communication.
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
  const progressiveSupportPreviouslyActive =
    Number.isInteger(
      pending?.progressiveSupportStage
  );
  
  const priorSupportActive =
    Boolean(
      priorFinding &&
      sameInstructionalComponent &&
      (
        progressiveSupportPreviouslyActive ||
        (
          governedContractPresent &&
          governedActivationPresent
      )
    )
  );

  const priorDiagnosis =
  cleanText(
    priorFinding?.diagnosis || ""
  );

const priorInstructionalSituation =
  cleanText(
    pending
      ?.instructionalContract
      ?.instructionalSituation || ""
  );

const priorNoEvidence =
  priorInstructionalSituation ===
    INSTRUCTIONAL_SITUATIONS
      .NO_COMPONENT_EVIDENCE ||
  priorDiagnosis ===
    "emptyResponse" ||
  priorDiagnosis ===
    "noComponentEvidence";

  return {
    priorSupportActive,
  
    progressiveSupportPreviouslyActive,
  
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
    (
      evidenceHistory
        .progressiveSupportPreviouslyActive ===
        true ||
  
      (
        evidenceHistory
          .priorSupportActive === true &&
  
        evidenceHistory
          .priorNoEvidence === true
    )
  );

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
//   governed Frame components in the current scope;
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
// Progression Layer obeys before progression may occur.
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

  // --------------------------------------------------
  // STUDENT-FACING COMMUNICATION STRUCTURE
  //
  // Formatting requirements come from the structure of
  // the already-authorized Thinking Move—not merely from
  // Progressive Support stage.
  //
  // AI still does not choose the structure.
  // --------------------------------------------------

  const requiredThinkingMove =
  cleanText(
    execution?.thinkingMove || ""
  );

const progressiveSupportStage =
  Number(
    execution?.progressiveSupportStage
  );

const progressiveSupportType =
  cleanText(
    execution?.progressiveSupportType ||
    ""
  );

const guidedConstructionStep =
  Number(
    execution?.guidedConstructionStep
  );

const frameComponent =
  cleanText(
    instructionalFinding
      ?.frameComponent ||
    execution?.context
      ?.frameComponent ||
    ""
  );

const guidedConstructionActive =
  progressiveSupportStage === 3 &&
  progressiveSupportType ===
    "guidedConstruction" &&
  [1, 2, 3].includes(
    guidedConstructionStep
  );

// --------------------------------------------------
// NO-COMPONENT-EVIDENCE VISUAL ARCHITECTURE
//
// A first no-evidence response remains a light,
// conversational nudge.
//
// When Kaw reconnects the student to accepted Frame
// context, that context is displayed visually using the
// established Framing Routine icons rather than being
// buried inside a dense sentence.
//
// This is not a thinking-options list and does not change
// the instructional move.
//
// Current scope:
// Is About — Key Topic.
// Main Idea — Key Topic + Is About.
// Essential Detail — current Main Idea.
// So What — accepted Frame context.
// --------------------------------------------------

const noComponentEvidenceVisualActive =
  execution?.contractId ===
    "IA-NCE-001" ||
  execution?.contractId ===
    "MI-NCE-001" ||
  execution?.contractId ===
    "ED-NCE-001" ||
  execution?.contractId ===
    "SW-NCE-001";

const noComponentEvidenceVisualArchitecture =
  !noComponentEvidenceVisualActive
    ? null

    : execution?.contractId ===
        "IA-NCE-001"
      ? {
          required:
            true,

          parentContexts: [
            {
              icon:
                "🧩",

              label:
                "Key Topic",

              value:
                cleanText(
                  execution?.context
                    ?.keyTopic || ""
                ),
            },
          ],

          requireVisualSeparation:
            true,

          requireSingleFinalQuestion:
            true,
        }

      : execution?.contractId ===
          "MI-NCE-001"
        ? {
            required:
              true,

            parentContexts: [
              {
                icon:
                  "🧩",

                label:
                  "Key Topic",

                value:
                  cleanText(
                    execution?.context
                      ?.keyTopic || ""
                  ),
              },

              {
                icon:
                  "💬",

                label:
                  "Is About",

                value:
                  cleanText(
                    execution?.context
                      ?.isAbout || ""
                  ),
              },
            ],

            requireVisualSeparation:
              true,

            requireSingleFinalQuestion:
              true,
          }

        : execution?.contractId ===
            "ED-NCE-001"
          ? {
              required:
                true,

              parentContexts: [
                {
                  icon:
                    "💡",

                  label:
                    "Main Idea",

                  value:
                    cleanText(
                      execution?.context
                        ?.currentMainIdea || ""
                    ),
                },
              ],

              requireVisualSeparation:
                true,

              requireSingleFinalQuestion:
                true,
            }

          : {
              required:
                true,

              parentContexts:
                Array.isArray(
                  execution?.context
                    ?.mainIdeas
                )
                  ? execution.context.mainIdeas
                      .map(
                        (idea, index) => ({
                          icon:
                            "💡",

                          label:
                            `Main Idea ${index + 1}`,

                          value:
                            cleanText(
                              idea || ""
                            ),
                        })
                      )
                      .filter(
                        (context) =>
                          context.value
                      )
                  : [],

              requireVisualSeparation:
                true,

              requireSingleFinalQuestion:
                true,
            };
  
// --------------------------------------------------
// STAGE 1 PROMPT VISUAL ARCHITECTURE
//
// Some Progressive Support Stage 1 prompts benefit from
// a small deterministic set of visually separated
// thinking lenses.
//
// These lenses reduce reading load without changing the
// instructional objective or supplying student thinking.
//
// AI may contextualize the prompt around accepted Frame
// content, but it may not choose, add, remove, or rewrite
// the authorized lenses.
//
// Current scope:
// Main Idea — Stage 1 Prompt.
// Essential Detail — Stage 1 Prompt.
//
//
// --------------------------------------------------

const stage1PromptVisualActive =
  progressiveSupportStage === 1 &&
  progressiveSupportType ===
    "prompt" &&
  (
    frameComponent ===
      "mainIdeas" ||
    frameComponent ===
      "details"
  );

const stage1PromptVisualArchitecture =
  !stage1PromptVisualActive
    ? null

    : frameComponent ===
        "mainIdeas"
      ? {
          required:
            true,

          componentIcon:
            "💡",

          componentLabel:
            "Main Idea",

          parentContexts: [
            {
              icon:
                "🧩",

              label:
                "Key Topic",
            },

            {
              icon:
                "💬",

              label:
                "Is About",
            },
          ],

          requireComponentInLeadIn:
            true,

          requireParentContext:
            true,

          requireBridgeLine:
            true,

          bridgeLine:
            "Think about one of these:",

          requireVisualSeparation:
            true,

          indentThinkingLenses:
            true,

          thinkingLenses: [
            {
              icon:
                "🧩",

              label:
                "an important category or part",
            },

            {
              icon:
                "🔄",

              label:
                "an important event or process",
            },

            {
              icon:
                "💭",

              label:
                "an important concept or idea",
            },
          ],

          finalQuestionTemplate:
            "Looking at your Key Topic and Is About, what is one important part that could become your Main Idea?",

          requireSingleFinalQuestion:
            true,
        }

      : {
          required:
            true,

          componentIcon:
            "✍️",

          componentLabel:
            "Essential Detail",

          leadIn:
            "I’ll help you build an ✍️ Essential Detail by connecting back to what you already have.",

          parentContextIcon:
            "💡",

          parentContextLabel:
            "Main Idea",

          requireComponentInLeadIn:
            true,

          requireParentContext:
            true,

          requireBridgeLine:
            true,

          bridgeLine:
            "Think about one of these:",

          requireVisualSeparation:
            true,

          indentThinkingLenses:
            true,

          thinkingLenses: [
            {
              icon:
                "📌",

              label:
                "a fact",
            },

            {
              icon:
                "💬",

              label:
                "an example",
            },

            {
              icon:
                "👀",

              label:
                "something you noticed or learned",
            },
          ],

          finalQuestionTemplate:
            "Looking at your Main Idea, what is one specific thing that could help explain or support it?",

          requireSingleFinalQuestion:
            true,
        };

// --------------------------------------------------
// GUIDED CONSTRUCTION VISUAL ARCHITECTURE
//
// Stage 3 uses one deterministic visual architecture
// across Is About, Main Idea, Essential Detail, and
// So What.
//
// The Guided Construction step determines the visual
// structure. AI does not choose the structure.
//
// Step 1 = single-step scaffold
// Step 2 = evidence-building scaffold
// Step 3 = evidence-assembly scaffold
//
// Icons and visual separation are instructional
// supports that reduce reading load for a student who
// is already experiencing struggle.
//
// --------------------------------------------------

const guidedConstructionComponentVisuals = {
  isAbout: {
    icon:
      "💬",

    label:
      "Is About",
  },

  mainIdeas: {
    icon:
      "💡",

    label:
      "Main Idea",
  },

  details: {
    icon:
      "✍️",

    label:
      "Essential Detail",
  },

  soWhat: {
    icon:
      "🎯",

    label:
      "So What",
  },
};

const guidedConstructionStepVisuals = {
  1: {
    mode:
      "singleStep",

    purpose:
      "Present one smaller thinking operation with clear visual anchors and one concise question.",
  },

  2: {
    mode:
      "evidenceBuilding",

    purpose:
      "Reconnect visually to the student-owned thinking from the previous guided step before asking for the next thinking operation.",
  },

  3: {
    mode:
      "evidenceAssembly",

    purpose:
      "Visually stack the student-owned pieces established in Guided Construction before inviting the student to formulate the Frame component in their own words.",
  },
};

const guidedConstructionEvidence =
  execution?.context
    ?.guidedConstructionEvidence &&
  typeof execution.context
    .guidedConstructionEvidence ===
    "object"
    ? execution.context
        .guidedConstructionEvidence
    : {};

const requiredGuidedConstructionEvidence =
  guidedConstructionStep === 2
    ? [
        cleanText(
          guidedConstructionEvidence
            ?.[1]
            ?.evidence || ""
        ),
      ].filter(Boolean)
    : guidedConstructionStep === 3
      ? [
          cleanText(
            guidedConstructionEvidence
              ?.[1]
              ?.evidence || ""
          ),
          cleanText(
            guidedConstructionEvidence
              ?.[2]
              ?.evidence || ""
          ),
        ].filter(Boolean)
      : [];
  
const guidedConstructionVisualArchitecture =
  guidedConstructionActive
    ? {
        required:
          true,

        step:
          guidedConstructionStep,

        mode:
          guidedConstructionStepVisuals
            ?.[guidedConstructionStep]
            ?.mode ||
          null,

        purpose:
          guidedConstructionStepVisuals
            ?.[guidedConstructionStep]
            ?.purpose ||
          "",

        componentIcon:
          guidedConstructionComponentVisuals
            ?.[frameComponent]
            ?.icon ||
          "",

        componentLabel:
          guidedConstructionComponentVisuals
            ?.[frameComponent]
            ?.label ||
          "",

        requireComponentHeader:
          true,

        requireIcons:
          true,

        requireVisualSeparation:
          true,

        requirePriorStudentEvidence:
          guidedConstructionStep >= 2,

        requireEvidenceStack:
          guidedConstructionStep === 3,

        requireSingleFinalQuestion:
          true,

        preserveStudentWording:
          true,
        
        requiredStudentEvidence:
          requiredGuidedConstructionEvidence,
      }
    : null;

  const modelContrastPresent =
    progressiveSupportStage === 2 &&
    progressiveSupportType ===
      "model";

  return {
    contractId:
      execution.contractId,

    instructionalGoal:
      execution.instructionalGoal,

    teachingMove:
      execution.teachingMove,

    requiredThinkingMove:
      execution.thinkingMove,

    progressiveSupport: {
      active:
        Number.isInteger(
          execution?.progressiveSupportStage
        ),

      stage:
        execution?.progressiveSupportStage ||
        null,

      guidedConstructionStep:
        execution?.guidedConstructionStep ||
        null,
      
      move:
        execution?.progressiveSupportMove ||
        null,
    },

    communicationPattern,

    studentFacingFormat: {
  noComponentEvidenceVisualArchitecture:
    noComponentEvidenceVisualArchitecture
      ? structuredClone(
          noComponentEvidenceVisualArchitecture
        )
      : null,

  stage1PromptVisualArchitecture:
    stage1PromptVisualArchitecture
      ? structuredClone(
          stage1PromptVisualArchitecture
        )
      : null,

  guidedConstructionVisualArchitecture:
    guidedConstructionVisualArchitecture
      ? structuredClone(
          guidedConstructionVisualArchitecture
        )
      : null,

  requireModelContrastSeparation:
    modelContrastPresent,

  requireSingleFinalQuestion:
    true,
},
    
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
    (response || "")
      .toString()
      .trim();

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

    // --------------------------------------------------
  // STUDENT-FACING FORMAT VALIDATION
  //
  // Guided Construction may require parallel cognitive
  // options to be rendered as a vertically scannable
  // list rather than compressed into one dense sentence.
  //
  // This validation checks presentation only.
  //
  // It does not:
  //
  // • change the Thinking Move;
  // • choose or generate instructional options;
  // • change Progressive Support stage;
  // • change Guided Construction step;
  // • change progression.
  //
  // --------------------------------------------------

  const studentFacingFormat =
    communicationLicense
      ?.studentFacingFormat &&
    typeof communicationLicense
      .studentFacingFormat ===
      "object"
      ? communicationLicense
          .studentFacingFormat
      : {};

  const noComponentEvidenceVisualArchitecture =
  studentFacingFormat
    ?.noComponentEvidenceVisualArchitecture &&
  typeof studentFacingFormat
    .noComponentEvidenceVisualArchitecture ===
      "object"
    ? studentFacingFormat
        .noComponentEvidenceVisualArchitecture
    : null;

  if (
  noComponentEvidenceVisualArchitecture
    ?.required === true
) {
  const parentContexts =
    Array.isArray(
      noComponentEvidenceVisualArchitecture
        ?.parentContexts
    )
      ? noComponentEvidenceVisualArchitecture
          .parentContexts
      : [];

  const nonEmptyLines =
    text
      .split(/\r?\n/)
      .map(
        (line) =>
          line.trim()
      )
      .filter(Boolean);

  const parentContextMissing =
    parentContexts.some(
      (context) => {
        const icon =
          cleanText(
            context?.icon || ""
          );

        const label =
          cleanText(
            context?.label || ""
          );

        const value =
          cleanText(
            context?.value || ""
          );

        return (
          !icon ||
          !label ||
          !value ||
          !text.includes(icon) ||
          !text.includes(label) ||
          !text.includes(value)
        );
      }
    );

  if (parentContextMissing) {
    violations.push(
      "noComponentEvidenceParentContextRequired"
    );
  }

  if (
    noComponentEvidenceVisualArchitecture
      ?.requireVisualSeparation === true &&
    nonEmptyLines.length < 3
  ) {
    violations.push(
      "noComponentEvidenceVisualSeparationRequired"
    );
  }

  if (
    noComponentEvidenceVisualArchitecture
      ?.requireSingleFinalQuestion === true &&
    questionCount !== 1
  ) {
    violations.push(
      "noComponentEvidenceSingleQuestionRequired"
    );
  }
}
  
  const stage1PromptVisualArchitecture =
  studentFacingFormat
    ?.stage1PromptVisualArchitecture &&
  typeof studentFacingFormat
    .stage1PromptVisualArchitecture ===
    "object"
    ? studentFacingFormat
        .stage1PromptVisualArchitecture
    : null;
  
  const guidedConstructionVisualArchitecture =
  studentFacingFormat
    ?.guidedConstructionVisualArchitecture &&
  typeof studentFacingFormat
    .guidedConstructionVisualArchitecture ===
    "object"
    ? studentFacingFormat
        .guidedConstructionVisualArchitecture
    : null;

if (
  stage1PromptVisualArchitecture
    ?.required === true
) {
  const componentIcon =
    cleanText(
      stage1PromptVisualArchitecture
        ?.componentIcon || ""
    );

  const componentLabel =
    cleanText(
      stage1PromptVisualArchitecture
        ?.componentLabel || ""
    );

  const leadIn =
  cleanText(
    stage1PromptVisualArchitecture
      ?.leadIn || ""
  );
  
  const parentContextIcon =
  cleanText(
    stage1PromptVisualArchitecture
      ?.parentContextIcon || ""
  );

const parentContextLabel =
  cleanText(
    stage1PromptVisualArchitecture
      ?.parentContextLabel || ""
  );

const parentContexts =
  Array.isArray(
    stage1PromptVisualArchitecture
      ?.parentContexts
  )
    ? stage1PromptVisualArchitecture
        .parentContexts
    : [];

  const thinkingLenses =
    Array.isArray(
      stage1PromptVisualArchitecture
        ?.thinkingLenses
    )
      ? stage1PromptVisualArchitecture
          .thinkingLenses
      : [];

  const nonEmptyLines =
    text
      .split(/\r?\n/)
      .map(
        (line) =>
          line.trim()
      )
      .filter(Boolean);

    if (
      leadIn &&
      !text.includes(leadIn)
  ) {
    violations.push(
      "stage1PromptLeadInRequired"
    );
  }
  
  if (
  stage1PromptVisualArchitecture
    ?.requireParentContext === true
) {
  const multipleParentContextsRequired =
    parentContexts.length > 0;

  const parentContextMissing =
    multipleParentContextsRequired
      ? parentContexts.some(
          (context) => {
            const icon =
              cleanText(
                context?.icon || ""
              );

            const label =
              cleanText(
                context?.label || ""
              );

            return (
              !icon ||
              !label ||
              !text.includes(icon) ||
              !text.includes(label)
            );
          }
        )
      : (
          !parentContextIcon ||
          !parentContextLabel ||
          !text.includes(parentContextIcon) ||
          !text.includes(parentContextLabel)
        );

  if (parentContextMissing) {
    violations.push(
      "stage1PromptParentContextRequired"
    );
  }
}
  
const bridgeLine =
  cleanText(
    stage1PromptVisualArchitecture
      ?.bridgeLine || ""
  );

if (
  stage1PromptVisualArchitecture
    ?.requireBridgeLine === true &&
  (
    !bridgeLine ||
    !text.includes(bridgeLine)
  )
) {
  violations.push(
    "stage1PromptBridgeLineRequired"
  );
}

  if (
    thinkingLenses.some(
      (lens) => {
        const icon =
          cleanText(
            lens?.icon || ""
          );

        const label =
          cleanText(
            lens?.label || ""
          );

        return (
          !icon ||
          !label ||
          !text.includes(icon) ||
          !text.includes(label)
        );
      }
    )
  ) {
    violations.push(
      "stage1PromptThinkingLensesRequired"
    );
  }

  if (
    stage1PromptVisualArchitecture
      ?.requireVisualSeparation === true &&
    nonEmptyLines.length < 6
  ) {
    violations.push(
      "stage1PromptVisualSeparationRequired"
    );
  }

  if (
    stage1PromptVisualArchitecture
      ?.requireSingleFinalQuestion === true &&
    questionCount !== 1
  ) {
    violations.push(
      "stage1PromptSingleQuestionRequired"
    );
  }
}

  const finalQuestionTemplate =
  cleanText(
    stage1PromptVisualArchitecture
      ?.finalQuestionTemplate || ""
  );

if (
  finalQuestionTemplate &&
  !text.endsWith(
    finalQuestionTemplate
  )
) {
  violations.push(
    "stage1PromptFinalQuestionRequired"
  );
}

  
if (
  guidedConstructionVisualArchitecture
    ?.required === true
) {
  const componentIcon =
    cleanText(
      guidedConstructionVisualArchitecture
        ?.componentIcon || ""
    );

  const componentLabel =
    cleanText(
      guidedConstructionVisualArchitecture
        ?.componentLabel || ""
    );

  const nonEmptyLines =
    text
      .split(/\r?\n/)
      .map(
        (line) =>
          line.trim()
      )
      .filter(Boolean);

  if (
    guidedConstructionVisualArchitecture
      ?.requireComponentHeader === true &&
    (
      !componentIcon ||
      !componentLabel ||
      !text.includes(componentIcon) ||
      !text.includes(componentLabel)
    )
  ) {
    violations.push(
      "guidedConstructionComponentHeaderRequired"
    );
  }

  if (
    guidedConstructionVisualArchitecture
      ?.requireVisualSeparation === true &&
    nonEmptyLines.length < 3
  ) {
    violations.push(
      "guidedConstructionVisualSeparationRequired"
    );
  }

  const requiredStudentEvidence =
  Array.isArray(
    guidedConstructionVisualArchitecture
      ?.requiredStudentEvidence
  )
    ? guidedConstructionVisualArchitecture
        .requiredStudentEvidence
        .map(cleanText)
        .filter(Boolean)
    : [];

if (
  guidedConstructionVisualArchitecture
    ?.requirePriorStudentEvidence ===
    true &&
  (
    requiredStudentEvidence.length === 0 ||
    requiredStudentEvidence.some(
      (evidence) =>
        !text.includes(evidence)
    )
  )
) {
  violations.push(
    "guidedConstructionStudentEvidenceRequired"
  );
}

  if (
    guidedConstructionVisualArchitecture
      ?.requireSingleFinalQuestion === true &&
    questionCount !== 1
  ) {
    violations.push(
      "guidedConstructionSingleQuestionRequired"
    );
  }
}

    // --------------------------------------------------
  // MODEL / CONTRAST VISUAL SEPARATION
  //
  // A content-distant model should not be compressed
  // into one dense paragraph when the license requires
  // the example and contrast to be visually separated.
  // --------------------------------------------------

  const requiresModelContrastSeparation =
    studentFacingFormat
      ?.requireModelContrastSeparation ===
      true;

  if (requiresModelContrastSeparation) {
  const nonEmptyLines =
    text
      .split(/\r?\n/)
      .map(
        (line) =>
          line.trim()
      )
      .filter(Boolean);

  if (nonEmptyLines.length < 2) {
    violations.push(
      "modelContrastSeparationRequired"
    );
  }
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

// ------------------------------------------------------
// PROGRESSIVE SUPPORT SCAFFOLD SELECTION
// ------------------------------------------------------
//
// Selects the predetermined Progressive Support scaffold
// from the already-selected Instructional Contract.
//
// Progressive Support uses three governed stages:
// Prompt, Model, and Guided Construction.
//
// This helper does not:
//
// • determine Genuine Struggle;
// • select an Instructional Contract;
// • change instructional location;
// • advance progression;
// • generate student-facing communication.
//
// Stage lifecycle is owned separately by Runtime
// Progression.
//
// The historical pending.supportLevel field remains a
// read-only compatibility fallback for previously stored
// Progressive Support state. Current runtime state uses
// pending.progressiveSupportStage.
//
// ------------------------------------------------------

function selectProgressiveSupportScaffold(
  contract,
  state
) {
  const scaffolds =
    Array.isArray(
      contract
        ?.progressiveSupport
        ?.scaffolds
    )
      ? contract
          .progressiveSupport
          .scaffolds
      : [];

  if (scaffolds.length === 0) {
    return null;
  }

  const requestedStage =
    Number(
      state?.pending
        ?.progressiveSupportStage ??
      state?.pending
        ?.supportLevel ??
      1
    );

  const progressiveSupportStage =
    Number.isFinite(
      requestedStage
    )
      ? Math.max(
          1,
          Math.min(
            requestedStage,
            3
          )
        )
      : 1;

  const selectedScaffold =
    scaffolds.find(
      (scaffold) =>
        Number(
          scaffold?.level
        ) ===
        progressiveSupportStage
    ) ||
    scaffolds[0] ||
    null;

  if (!selectedScaffold) {
    return null;
  }

  // --------------------------------------------------
  // GUIDED CONSTRUCTION STEP-AWARE THINKING MOVE
  //
  // Progressive Support Stage 3 is Guided Construction.
  //
  // Once Stage 3 is active, the current
  // guidedConstructionStep determines the smaller
  // teacher-authored Thinking Move.
  //
  // The Instructional Playbook declares which governed
  // Guided Construction rule belongs to each step.
  //
  // GUIDED_CONSTRUCTION_RULES remains the single source
  // of truth for the actual Thinking Move.
  //
  // Stage 1 Prompt and Stage 2 Model continue using the
  // scaffold's existing Thinking Move unchanged.
  //
  // --------------------------------------------------

  let guidedConstructionStep =
    null;

  let guidedConstructionStepDefinition =
    null;

  let guidedConstructionRule =
    null;

  let thinkingMove =
    selectedScaffold
      ?.thinkingMove ||
    contract?.thinkingMove ||
    null;

  if (
    progressiveSupportStage === 3
  ) {
    const requestedGuidedStep =
      Number(
        state?.pending
          ?.guidedConstructionStep
      );

    const validGuidedStep =
      Number.isInteger(
        requestedGuidedStep
      ) &&
      requestedGuidedStep >= 1 &&
      requestedGuidedStep <= 3;

    if (!validGuidedStep) {
      return null;
    }

    guidedConstructionStep =
      requestedGuidedStep;

    guidedConstructionStepDefinition =
      selectedScaffold
        ?.guidedSteps
        ?.[guidedConstructionStep] ||
      null;

    if (
      !guidedConstructionStepDefinition
    ) {
      return null;
    }

    const ruleComponent =
      cleanText(
        guidedConstructionStepDefinition
          ?.ruleComponent || ""
      );

    const ruleStep =
      Number(
        guidedConstructionStepDefinition
          ?.ruleStep
      );

    const operation =
      cleanText(
        guidedConstructionStepDefinition
          ?.operation || ""
      );

    const contractFrameComponent =
      cleanText(
        contract
          ?.frameComponent || ""
      );

    const ruleComponentMatchesContract =
      Boolean(
        ruleComponent &&
        ruleComponent ===
          contractFrameComponent
      );

    const ruleStepMatchesCurrentStep =
      Number.isInteger(ruleStep) &&
      ruleStep ===
        guidedConstructionStep;

    const componentRules =
      GUIDED_CONSTRUCTION_RULES
        ?.[ruleComponent] ||
      null;

    const currentRule =
      componentRules
        ?.steps
        ?.[ruleStep] ||
      null;

    const operationMatchesRule =
      Boolean(
        operation &&
        currentRule &&
        cleanText(
          currentRule
            ?.operation || ""
        ) === operation
      );

    if (
      !ruleComponentMatchesContract ||
      !ruleStepMatchesCurrentStep ||
      !currentRule ||
      !operationMatchesRule
    ) {
      return null;
    }

    guidedConstructionRule =
      currentRule;

    thinkingMove =
      cleanText(
        guidedConstructionRule
          ?.thinkingMove || ""
      ) ||
      null;

    if (!thinkingMove) {
      return null;
    }
  }

  return {
    progressiveSupportStage,

    guidedConstructionStep,

    move:
      selectedScaffold?.move ||
      null,

    supportType:
      selectedScaffold?.supportType ||
      null,

    purpose:
      selectedScaffold?.purpose ||
      null,

    cue:
      selectedScaffold?.cue ||
      null,

    modelRules:
      selectedScaffold?.modelRules
        ? structuredClone(
            selectedScaffold.modelRules
          )
        : null,

    guidedConstructionStepDefinition:
      guidedConstructionStepDefinition
        ? structuredClone(
            guidedConstructionStepDefinition
          )
        : null,

    guidedConstructionRule:
      guidedConstructionRule
        ? structuredClone(
            guidedConstructionRule
          )
        : null,

    thinkingMove,
  };
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

  const progressiveSupport =
    contract.contractId ===
      "IA-GS-001"
      ? selectProgressiveSupportScaffold(
          contract,
          state
        )
      : null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      progressiveSupport
        ?.thinkingMove ||
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    progressiveSupportStage:
      progressiveSupport
        ?.progressiveSupportStage ||
      null,

    guidedConstructionStep:
      progressiveSupport
        ?.guidedConstructionStep ||
      null,
    
    progressiveSupportMove:
      progressiveSupport?.move ||
      null,

    progressiveSupportType:
      progressiveSupport?.supportType ||
      null,

    progressiveSupportCue:
      progressiveSupport?.cue ||
      null,

    progressiveSupportModelRules:
      progressiveSupport?.modelRules
        ? structuredClone(
            progressiveSupport.modelRules
          )
        : null,

    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      guidedConstructionEvidence:
      state?.pending
        ?.guidedConstructionEvidence &&
      typeof state.pending
        .guidedConstructionEvidence ===
        "object"
        ? structuredClone(
            state.pending
              .guidedConstructionEvidence
      )
    : {},

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

  const progressiveSupport =
    contract.contractId ===
      "ED-GS-001"
      ? selectProgressiveSupportScaffold(
          contract,
          state
        )
      : null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      progressiveSupport
        ?.thinkingMove ||
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    progressiveSupportStage:
      progressiveSupport
        ?.progressiveSupportStage ||
      null,

    guidedConstructionStep:
      progressiveSupport
        ?.guidedConstructionStep ||
      null,
    
    progressiveSupportMove:
      progressiveSupport?.move ||
      null,

    progressiveSupportType:
  progressiveSupport?.supportType ||
  null,

    progressiveSupportCue:
      progressiveSupport?.cue ||
      null,
    
    progressiveSupportModelRules:
      progressiveSupport?.modelRules
        ? structuredClone(
            progressiveSupport.modelRules
          )
        : null,
    
    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      guidedConstructionEvidence:
        state?.pending
          ?.guidedConstructionEvidence &&
        typeof state.pending
          .guidedConstructionEvidence ===
          "object"
          ? structuredClone(
              state.pending
                .guidedConstructionEvidence
      )
    : {},

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

  const progressiveSupport =
    contract.contractId ===
      "MI-GS-001"
      ? selectProgressiveSupportScaffold(
          contract,
          state
        )
      : null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      progressiveSupport
        ?.thinkingMove ||
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    progressiveSupportStage:
      progressiveSupport
        ?.progressiveSupportStage ||
      null,

    guidedConstructionStep:
      progressiveSupport
        ?.guidedConstructionStep ||
      null,
    
    progressiveSupportMove:
      progressiveSupport?.move ||
      null,

    progressiveSupportType:
      progressiveSupport?.supportType ||
      null,

    progressiveSupportCue:
      progressiveSupport?.cue ||
      null,

    progressiveSupportModelRules:
      progressiveSupport?.modelRules
        ? structuredClone(
            progressiveSupport.modelRules
          )
        : null,
    
    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      guidedConstructionEvidence:
        state?.pending
          ?.guidedConstructionEvidence &&
        typeof state.pending
          .guidedConstructionEvidence ===
          "object"
          ? structuredClone(
              state.pending
                .guidedConstructionEvidence
      )
    : {},

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

  const progressiveSupport =
    contract.contractId ===
      "SW-GS-001"
      ? selectProgressiveSupportScaffold(
          contract,
          state
        )
      : null;

  return {
    contractId:
      contract.contractId,

    instructionalGoal:
      contract.instructionalGoal,

    teachingMove:
      contract.teachingMove,

    thinkingMove:
      progressiveSupport
        ?.thinkingMove ||
      contract.thinkingMove,

    communicationPattern:
      contract.communicationPattern ||
      "questionOnly",

    aiContextualizes:
      contract.aiContextualizes,

    instructionalFinding,

    progressiveSupportStage:
      progressiveSupport
        ?.progressiveSupportStage ||
      null,

    guidedConstructionStep:
      progressiveSupport
        ?.guidedConstructionStep ||
      null,
    
    progressiveSupportMove:
      progressiveSupport?.move ||
      null,

    progressiveSupportType:
      progressiveSupport?.supportType ||
      null,

    progressiveSupportCue:
      progressiveSupport?.cue ||
      null,
    
    progressiveSupportModelRules:
      progressiveSupport?.modelRules
        ? structuredClone(
            progressiveSupport.modelRules
          )
        : null,
    
    context: {
      assignmentContext:
        state?.frameMeta
          ?.assignmentContext || {},

      thinkingTask:
        state?.assignmentReasoning || {},

      frameComponent:
        contract.frameComponent,

      guidedConstructionEvidence:
        state?.pending
          ?.guidedConstructionEvidence &&
        typeof state.pending
          .guidedConstructionEvidence ===
          "object"
          ? structuredClone(
              state.pending
                .guidedConstructionEvidence
      )
    : {},

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

    progressiveSupportStage:
      execution?.progressiveSupportStage ||
      null,

    guidedConstructionStep:
      execution?.guidedConstructionStep ||
      null,
    
    progressiveSupportMove:
      execution?.progressiveSupportMove ||
      null,

    progressiveSupportType:
      execution?.progressiveSupportType ||
      null,

    progressiveSupportCue:
      execution?.progressiveSupportCue ||
      null,

    progressiveSupportModelRules:
      execution?.progressiveSupportModelRules ||
      null,
    
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

      guidedConstructionEvidence:
        execution?.context
          ?.guidedConstructionEvidence &&
        typeof execution.context
          .guidedConstructionEvidence ===
          "object"
          ? structuredClone(
              execution.context
                .guidedConstructionEvidence
      )
    : {},

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

  const guidedConstructionEvidence =
    payload?.context
      ?.guidedConstructionEvidence &&
    typeof payload.context
      .guidedConstructionEvidence ===
      "object"
      ? payload.context
          .guidedConstructionEvidence
      : {};

const noComponentEvidenceVisualArchitecture =
  communicationLicense
    ?.studentFacingFormat
    ?.noComponentEvidenceVisualArchitecture &&
  typeof communicationLicense
    .studentFacingFormat
    .noComponentEvidenceVisualArchitecture ===
    "object"
    ? communicationLicense
        .studentFacingFormat
        .noComponentEvidenceVisualArchitecture
    : null;
  
const stage1PromptVisualArchitecture =
  communicationLicense
    ?.studentFacingFormat
    ?.stage1PromptVisualArchitecture &&
  typeof communicationLicense
    .studentFacingFormat
    .stage1PromptVisualArchitecture ===
    "object"
    ? communicationLicense
        .studentFacingFormat
        .stage1PromptVisualArchitecture
    : null;

const stage1FinalQuestion =
  stage1PromptVisualArchitecture
    ?.finalQuestionTemplate
    ? cleanText(
        stage1PromptVisualArchitecture
          .finalQuestionTemplate
      )
    : "";

const guidedConstructionVisualArchitecture =
  communicationLicense
    ?.studentFacingFormat
    ?.guidedConstructionVisualArchitecture &&
  typeof communicationLicense
    .studentFacingFormat
    .guidedConstructionVisualArchitecture ===
    "object"
    ? communicationLicense
        .studentFacingFormat
        .guidedConstructionVisualArchitecture
    : null;
  
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
- Preserve every instructional distinction, comparison, cognitive cue, and constraint contained in the predetermined Thinking Move. Do not generalize it into a simpler or earlier-stage question.
- When a No-Component-Evidence Visual Architecture is supplied in the Communication License, it is mandatory.
- Render each supplied accepted parent context on its own visually separated line using the supplied icon, label, and exact value.
- Keep the parent-context lines visually separate from the brief lead-in and the final question.
- Do not turn these parent contexts into thinking options, examples, or suggested answers.
- When a Stage 1 Prompt Visual Architecture is supplied in the Communication License, it is mandatory.
- If an exact lead-in is supplied in the Stage 1 Prompt Visual Architecture, render that lead-in exactly as provided. Otherwise, use a brief teacher-like lead-in that explains how Kaw is helping the student and includes the supplied component icon and component label naturally in the sentence.
- Show every supplied accepted parent context on its own visually separated line using its supplied icon and label. When multiple parent contexts are supplied, display each one separately and preserve their supplied order.
- When a bridge line is supplied, render it exactly as provided before the thinking lenses.
- Render the authorized thinking lenses as separate, scannable lines beneath the bridge line using their supplied icons and labels.
- When indentation is required, visually indent the thinking lenses beneath the bridge line so they read as related thinking options.
- Do not add, remove, rename, combine, or replace the supplied thinking lenses.
- Do not turn the lenses into suggested student answers; they are categories that help the student think.
- If a final Stage 1 question is supplied, render it exactly as provided.
- End with exactly one concise question that performs the predetermined Thinking Move.
- When Progressive Support Stage 3 Guided Construction is active, the Guided Construction Visual Architecture supplied in the Communication License is mandatory.

- Guided Construction uses one standard visual architecture across Is About, Main Idea, Essential Detail, and So What. The current Guided Construction step determines the structure. Do not choose a different structure.

- Every Guided Construction response must:
  - begin with the supplied Framing Routine component icon and component name as the visual header;
  - use whitespace and short visually separated chunks rather than dense paragraphs;
  - preserve the student's wording exactly when displaying student-owned Guided Construction evidence;
  - display only accepted Frame context and student-owned evidence that are relevant to the current thinking operation;
  - end with exactly one concise student-facing question.

- Guided Construction Step 1 is a SINGLE-STEP SCAFFOLD:
  - show the active component header;
  - reconnect visually to the accepted Frame context needed for this step;
  - present only the one smaller thinking operation authorized by the Predetermined Thinking Move;
  - if the Predetermined Thinking Move contains several authorized thinking lenses or categories that genuinely help the student perform that operation, separate those choices vertically so they are easy to scan;
  - do not require a fixed number of choices and do not invent additional choices merely to create a list.

- Guided Construction Step 2 is an EVIDENCE-BUILDING SCAFFOLD:
  - show the active component header;
  - visibly reconnect to the relevant student-owned evidence established in Guided Construction Step 1;
  - label that evidence in natural student-facing language rather than internal architecture terminology;
  - reconnect to accepted Frame context when the Predetermined Thinking Move requires it;
  - visually separate the student's existing thinking from the next thinking operation;
  - ask only for the next thinking operation authorized for Step 2.

- Guided Construction Step 3 is an EVIDENCE-ASSEMBLY SCAFFOLD:
  - show the active component header;
  - visibly stack the relevant student-owned evidence established in the earlier Guided Construction steps;
  - use short, natural student-facing labels that explain what each displayed piece represents;
  - preserve each piece of student evidence without rewriting, combining, improving, interpreting, or completing it;
  - invite the student to formulate, assemble, or synthesize the component in their own words according to the Predetermined Thinking Move.

- Visual separation does not require generic bullet syntax. Use the supplied Framing Routine icons, short labels, line breaks, and whitespace to make the thinking structure easy to scan.

- When reconnecting to an accepted parent Frame component, use its established Framing Routine icon immediately before its name when appropriate:
  🧩 Key Topic
  💬 Is About
  💡 Main Idea
  ✍️ Essential Detail
  🎯 So What

- The visual architecture is presentation only. It may not change the Instructional Goal, Teaching Move, Thinking Move, Guided Construction step, or progression; add another teaching move; supply student thinking; or create another question.

- Keep Guided Construction communication visually light, concrete, and easy to scan for a student who is already experiencing struggle.
- When Progressive Support is active, preserve the exact instructional architecture selected by the deterministic runtime: Stage 1 is Prompt, Stage 2 is Model, and Stage 3 is Guided Construction. Stage 1 must provide only the light Prompt defined by the predetermined Thinking Move. Stage 2 must visibly model the required kind of thinking using one brief, clearly content-distant example that follows the supplied model rules, then return immediately to the student's own Frame. Stage 3 must visibly begin or continue Guided Construction by reducing the current component thinking into the smaller step identified by the predetermined Thinking Move and Guided Construction step. Never collapse Model or Guided Construction back into a general Prompt.
- Do not mention Progressive Support, stage numbers, or internal move names to the student.
- Do not reinterpret, expand, weaken, strengthen, or replace the established Instructional Finding.
- Do not infer student intent, understanding, confusion, emotion, effort, motivation, or meaning.
- Do not make claims about success, progress, correctness, relationships, or quality unless the established Instructional Finding and Communication License permit that claim.
- When relationship status is undetermined, preserve that uncertainty rather than resolving it yourself.
- Preserve student ownership at all times.
- When an Instructional Finding provides both attemptedDetail and displayAttemptedDetail, use displayAttemptedDetail whenever referencing that attempted Essential Detail to the student. attemptedDetail is raw internal evidence and must not be displayed when the corrected display version is available.
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

Progressive Support Stage:
${payload.progressiveSupportStage || "(not active)"}

Progressive Support Type:
${payload.progressiveSupportType || "(not active)"}

Progressive Support Cue:
${payload.progressiveSupportCue || "(not active)"}

Guided Construction Step:
${payload.guidedConstructionStep || "(not active)"}

Progressive Support Model Rules:
${JSON.stringify(
  payload.progressiveSupportModelRules || {},
  null,
  2
)}

Progressive Support Move:
${payload.progressiveSupportMove || "(not active)"}

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

STUDENT-FACING FORMAT REQUIREMENT:

No-Component-Evidence Visual Architecture:
${
  noComponentEvidenceVisualArchitecture
    ? JSON.stringify(
        noComponentEvidenceVisualArchitecture,
        null,
        2
      )
    : "(not active)"
}

Stage 1 Prompt Visual Architecture:
${
  stage1PromptVisualArchitecture
    ? JSON.stringify(
        stage1PromptVisualArchitecture,
        null,
        2
      )
    : "(not active)"
}

Stage 1 Final Question:
${stage1FinalQuestion || "(not supplied)"}

Guided Construction Visual Architecture:
${
  guidedConstructionVisualArchitecture
    ? JSON.stringify(
        guidedConstructionVisualArchitecture,
        null,
        2
      )
    : "(not active)"
}

Student-Owned Guided Construction Evidence:
${
  Object.keys(
    guidedConstructionEvidence || {}
  ).length
    ? JSON.stringify(
        guidedConstructionEvidence,
        null,
        2
      )
    : "(none established yet)"
}

When Guided Construction Visual Architecture is active, follow it exactly.

- Use the supplied component icon and component label as the visual header.
- Use short, visually separated chunks with whitespace rather than a dense paragraph.
- Preserve displayed student-owned evidence exactly as the student expressed it.
- Do not rewrite, improve, combine, interpret, or complete that evidence.
- Show only the prior student evidence relevant to the current Guided Construction step.
- Use natural student-facing labels rather than internal architecture terms.
- End with exactly one concise question.

For Guided Construction Step 1:
- use the required single-step scaffold;
- reconnect to only the accepted Frame context needed for the current smaller thinking operation;
- do not display student-owned Guided Construction evidence unless the Predetermined Thinking Move specifically requires prior student evidence.

For Guided Construction Step 2:
- use the required evidence-building scaffold;
- visibly display the relevant student-owned evidence established in Step 1;
- visually separate that evidence from the next thinking operation.

For Guided Construction Step 3:
- use the required evidence-assembly scaffold;
- visibly stack the relevant student-owned evidence established in the earlier Guided Construction steps;
- keep each student-owned piece separate;
- invite the student to formulate, assemble, or synthesize the component in their own words.

Do not create a generic bullet list merely because the Predetermined Thinking Move contains commas or the word "or".

When authorized thinking choices or lenses genuinely need to be shown, separate them visually for readability, but do not invent extra choices or require an arbitrary number of items.
If Progressive Support is active, preserve the exact instructional difference of the selected stage and move. Do not simplify a Stage 2 or Stage 3 prompt back into the generic Stage 1 question.
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

  // --------------------------------------------------
  // ONE GOVERNED FORMAT-ONLY REGENERATION ATTEMPT
  //
  // A communication response that fails only because
  // required student-facing formatting was not followed
  // may be regenerated once.
  //
  // The retry must preserve:
  //
  // • the same Instructional Contract;
  // • the same Instructional Goal;
  // • the same Teaching Move;
  // • the same Thinking Move;
  // • the same Progressive Support stage;
  // • the same Guided Construction step;
  // • the same student-work protections.
  //
  // This retry corrects presentation only.
  //
  // --------------------------------------------------

  const formattingOnlyViolation =
  communicationValidation
    .violations.length === 1 &&
  (
    communicationValidation
      .violations.includes(
        "guidedConstructionComponentHeaderRequired"
      ) ||
    communicationValidation
      .violations.includes(
        "guidedConstructionVisualSeparationRequired"
      ) ||
    communicationValidation
      .violations.includes(
        "guidedConstructionStudentEvidenceRequired"
      ) ||
    communicationValidation
      .violations.includes(
        "guidedConstructionSingleQuestionRequired"
      ) ||
    communicationValidation
      .violations.includes(
        "modelContrastSeparationRequired"
      )
  );

  if (formattingOnlyViolation) {
    const retryInstruction = `${user}

FORMAT CORRECTION REQUIRED:

Your previous response preserved the instructional move but failed the required student-facing presentation format.

Regenerate the SAME predetermined Thinking Move.

Do not change the instructional content, question, component, Progressive Support stage, or Guided Construction step.

Follow the Communication License exactly.

If Guided Construction Visual Architecture is active:
- preserve the SAME Guided Construction step and Thinking Move;
- use the required Framing Routine component icon and component name as the header;
- use short, visually separated chunks with whitespace;
- preserve any displayed student-owned Guided Construction evidence exactly;
- do not rewrite, combine, interpret, improve, or complete student evidence;
- follow the required Step 1 single-step, Step 2 evidence-building, or Step 3 evidence-assembly structure supplied in the Communication License;
- do not create a generic 3–5 item list merely to satisfy formatting;
- if genuine authorized thinking choices need to be shown, separate only those choices visually.

If the authorized move is a content-distant model:
- visually separate the model from the contrast and the return to the student's own Frame;
- do not compress the entire model into one paragraph;
- do not add another example or any student-specific answer content.

Ask exactly one final question.

Return only the corrected student-facing response.
`;

    const retryResp =
      await client.chat.completions.create({
        model:
          DEFAULT_MODEL,

        reasoning_effort:
          "none",

        temperature:
          0,

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
              retryInstruction,
          },
        ],
      });

    const retryResponse =
      retryResp?.choices?.[0]
        ?.message?.content || "";

    const retryValidation =
      validateInstructionalCommunicationResponse(
        retryResponse,
        communicationLicense
      );

    activation.communicationDebug = {
      ...activation.communicationDebug,

      retryResponse:
        cleanText(
          retryResponse
        ),

      retryValidation:
        structuredClone(
          retryValidation
        ),
    };

     if (retryValidation.valid) {
      return (retryResponse || "")
        .toString()
        .trim();
    }
  }

  return null;
}
    
    return (response || "")
      .toString()
      .trim();
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
      
      supportingRelationshipExpressed:
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
- makes the supporting relationship understandable from the student's response itself rather than requiring the reader to infer the connection from the surrounding Frame;
- adds concrete information that is more specific than the Main Idea;
- helps the reader understand how, why, when, where, what happened, what resulted, what example demonstrates the idea, or what evidence supports it;
- can function as a fact, example, observation, explanation, event, condition, action, result, or piece of evidence;
- does not merely repeat, shorten, or make a more general statement about the Main Idea;
- does not function primarily as a separate major organizing Main Idea.

Relationship test:
- supportsMainIdea must be true only when the student's response itself communicates how the proposed Essential Detail supports, explains, illustrates, demonstrates, or provides evidence for the accepted Main Idea.
- A response may be clearly related to the Main Idea while still leaving the supporting relationship unstated.
- Do not use the surrounding Frame context to supply a connection the student did not express.
- If a reasonable reader must infer why the detail supports the Main Idea, supportsMainIdea must be false.
- The student does not need to use a particular connector word such as "because," "shows," or "supports"; the relationship may be expressed naturally in any wording.

supportingRelationshipExpressed test:
- supportingRelationshipExpressed is true only when the student's own words communicate enough of the connection for a reasonable reader to understand how the proposed Essential Detail supports or explains the accepted Main Idea.
- It must be false when the response is merely relevant to the Main Idea or contains an internal cause-and-effect relationship but still requires an additional unstated bridge to the Main Idea.
- Do not infer or supply that missing bridge from the surrounding Frame.
- The supporting relationship may be expressed using different vocabulary from the Main Idea; exact word overlap is not required.
- Example: "Notifications and messages can make it hard to disconnect." contains a relationship inside the detail, but it does not yet explain how being unable to disconnect supports the Main Idea "Social media can increase anxiety and stress." Therefore supportingRelationshipExpressed must be false.
- Example: "Feeling like they always need to check notifications can keep teenagers tense and make it harder for them to relax." communicates the consequential connection in the student's own words. Therefore supportingRelationshipExpressed may be true even though it does not repeat the exact words "anxiety" or "stress."

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
- supportsMainIdea must be false because the student's response does not communicate how needing sunlight supports or explains the accepted Main Idea;
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
- Do not require a particular connector word such as "because," "shows," or "supports."
- Use the complete Frame context to understand what the student is discussing, but never use that context to supply an unstated supporting relationship.
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

              supportingRelationshipExpressed: {
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
              "supportingRelationshipExpressed",
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

      supportingRelationshipExpressed:
        parsed.supportingRelationshipExpressed ===
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

      supportingRelationshipExpressed:
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

const studentExpressedRelationship =
  deterministicResult
    ?.relationshipEvidence
    ?.hasRelationshipLanguage === true;

const relationshipEstablished =
  studentExpressedRelationship === true &&

  semanticEvidence
    .supportsMainIdea === true &&

  semanticEvidence
    .supportingRelationshipExpressed === true &&

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

        supportingRelationshipExpressed:
          semanticEvidence
            .supportingRelationshipExpressed,
        
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

      supportingRelationshipExpressed:
        semanticEvidence
          .supportingRelationshipExpressed,
      
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
- the response must express an actual takeaway, relationship, implication, significance, lesson, or conclusion that becomes clear from the ideas developed in the Frame;
- false when the response merely says the topic is important, has effects, matters, is interesting, is good, is bad, affects people, or requires someone to "be careful" without explaining the actual understanding behind that statement;
- a recommendation or warning is not automatically a meaningful So What; the response must communicate why that recommendation or warning follows from the completed Frame;
- false for broad statements such as "Social media has important effects on teenagers" because the reader still does not know what the important understanding is;
- false for broad caution statements such as "Social media has good and bad effects, so teenagers should be careful" because the reader still does not know what effects matter, what the Frame shows about them, or what understanding supports the caution.

specificEnoughToUnderstand:
- true when the reader can identify the student's actual takeaway, even when the response uses a metaphor, analogy, application, recommendation, warning, or broad life truth;
- false when the wording is so general that it could apply to many topics without meaningful change;
- false when general words such as "good," "bad," "important," "effects," "careful," or "responsible" carry the conclusion without making the underlying understanding clear;
- false when the reader would still need to ask "What effect?", "Why does it matter?", "Why should they be careful?", "What does this show?", or "What is the actual takeaway?";
- specificity does not require copying Main Ideas or Essential Details, but the meaning must be grounded enough that the relationship to the completed Frame is understandable.

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

  // --------------------------------------------------
  // GUIDED CONSTRUCTION — ESSENTIAL DETAIL
  // TARGETED VERIFICATION
  // --------------------------------------------------
  //
  // Confirms that Essential Detail uses the shared
  // Guided Construction runtime while preserving:
  //
  // • normal Essential Detail validation authority;
  // • exact Main Idea + Detail location identity;
  // • deterministic stay / advance behavior;
  // • student-owned guided evidence only;
  // • immediate yield when the full component is valid.
  //
  // --------------------------------------------------

  function createEssentialDetailGuidedTestState() {
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
      "Social Media and Teen Mental Health";

    state.frame.isAbout =
      "How social media can affect teen mental health.";

    state.frame.parentItems = [
      currentMainIdea,
      "Social media can affect self-esteem.",
    ];

    state.frame.details = [
      [],
      [],
    ];

    state.pending = {
      type:
        "collectAnotherDetail",

      index:
        0,

      captureMode:
        "required",

      progressiveSupportStage:
        3,

      guidedConstructionStep:
        1,
    };

    return state;
  }

  // --------------------------------------------------
  // ED GC TEST 1 — STEP-AWARE THINKING MOVE SELECTION
  // --------------------------------------------------

  const guidedSelectionContract =
    INSTRUCTIONAL_PLAYBOOK
      ?.details
      ?.genuineStruggle;

  const guidedSelectionResults =
    [1, 2, 3].map(
      (guidedConstructionStep) => {
        const testState =
          createEssentialDetailGuidedTestState();

        testState.pending
          .guidedConstructionStep =
          guidedConstructionStep;

        const selectedScaffold =
          selectProgressiveSupportScaffold(
            guidedSelectionContract,
            testState
          );

        const expectedRule =
          GUIDED_CONSTRUCTION_RULES
            ?.details
            ?.steps
            ?.[guidedConstructionStep];

        return {
          guidedConstructionStep,

          passed:
            selectedScaffold
              ?.progressiveSupportStage ===
              3 &&

            selectedScaffold
              ?.guidedConstructionStep ===
              guidedConstructionStep &&

            selectedScaffold
              ?.thinkingMove ===
              expectedRule
                ?.thinkingMove,

          actualThinkingMove:
            selectedScaffold
              ?.thinkingMove ||
            null,

          expectedThinkingMove:
            expectedRule
              ?.thinkingMove ||
            null,
        };
      }
    );

  const guidedSelectionPassed =
    guidedSelectionResults.every(
      (result) =>
        result.passed === true
    );

  results.push({
    name:
      "ED Guided Construction - Stage 3 selects the correct Step 1, 2, and 3 Thinking Moves",

    passed:
      guidedSelectionPassed,

    expected: {
      progressiveSupportStage:
        3,

      guidedSteps:
        [1, 2, 3],

      allThinkingMovesMatchRules:
        true,
    },

    actual: {
      allThinkingMovesMatchRules:
        guidedSelectionPassed,

      stepResults:
        guidedSelectionResults,
    },
  });

  // --------------------------------------------------
  // ED GC TEST 2 — EXACT TWO-COORDINATE LOCATION
  // --------------------------------------------------

  const guidedLocationState =
    createEssentialDetailGuidedTestState();

  const guidedLocationActual =
    buildGuidedConstructionInstructionalLocation(
      guidedLocationState
    );

  const guidedLocationPassed =
    guidedLocationActual
      ?.locationEstablished ===
      true &&

    guidedLocationActual
      ?.frameComponent ===
      "details" &&

    guidedLocationActual
      ?.detailMainIdeaIndex ===
      0 &&

    guidedLocationActual
      ?.detailIndex ===
      0;

  results.push({
    name:
      "ED Guided Construction - Exact location preserves Main Idea index and Detail index",

    passed:
      guidedLocationPassed,

    expected: {
      frameComponent:
        "details",

      detailMainIdeaIndex:
        0,

      detailIndex:
        0,
    },

    actual: {
      frameComponent:
        guidedLocationActual
          ?.frameComponent ||
        null,

      detailMainIdeaIndex:
        guidedLocationActual
          ?.detailMainIdeaIndex ??
        null,

      detailIndex:
        guidedLocationActual
          ?.detailIndex ??
        null,

      locationEstablished:
        guidedLocationActual
          ?.locationEstablished ===
        true,
    },
  });

  // --------------------------------------------------
  // ED GC TEST 3 — INSUFFICIENT STEP-1 EVIDENCE STAYS
  // --------------------------------------------------

  const guidedStayState =
    createEssentialDetailGuidedTestState();

  guidedStayState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedStayState
    );

  const guidedStayValidation =
    validateEssentialDetailResponse(
      "idk",
      currentMainIdea
    );

  const guidedStayActual =
    await continueGuidedConstruction({
      state:
        guidedStayState,

      response:
        "idk",

      componentValidation:
        guidedStayValidation,

      finalRephraseUsed:
        false,
    });

  const guidedStayPassed =
    guidedStayActual
      ?.continuationStatus ===
      "established" &&

    guidedStayActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .INSUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedStayActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .STAY_CURRENT_STEP &&

    guidedStayState
      ?.pending
      ?.guidedConstructionStep ===
      1 &&

    !guidedStayState
      ?.pending
      ?.guidedConstructionEvidence;

  results.push({
    name:
      "ED Guided Construction - Insufficient Step-1 evidence stays on Step 1",

    passed:
      guidedStayPassed,

    expected: {
      continuationStatus:
        "established",

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .INSUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .STAY_CURRENT_STEP,

      guidedConstructionStep:
        1,

      guidedEvidenceSaved:
        false,
    },

    actual: {
      continuationStatus:
        guidedStayActual
          ?.continuationStatus ||
        null,

      evidenceOutcome:
        guidedStayActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedStayActual
          ?.progressionDecision
          ?.decision ||
        null,

      guidedConstructionStep:
        guidedStayState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      guidedEvidenceSaved:
        Boolean(
          guidedStayState
            ?.pending
            ?.guidedConstructionEvidence
        ),
    },
  });

  // --------------------------------------------------
  // ED GC TEST 4 — SUFFICIENT STEP-1 EVIDENCE ADVANCES
  //
  // Bounded semantic evidence is supplied directly so
  // this test verifies the deterministic progression
  // brain without making an additional AI call.
  // --------------------------------------------------

  const guidedAdvanceState =
    createEssentialDetailGuidedTestState();

  const guidedAdvanceLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedAdvanceState
    );

  guidedAdvanceState
    .pending
    .guidedConstructionLocation =
    structuredClone(
      guidedAdvanceLocation
    );

  const guidedAdvanceResponse =
    "Teens compare themselves to influencers.";

  const guidedAdvanceValidation =
    validateEssentialDetailResponse(
      guidedAdvanceResponse,
      currentMainIdea
    );

  const guidedAdvanceAssessment =
    assessGuidedConstructionEvidence({
      state:
        guidedAdvanceState,

      response:
        guidedAdvanceResponse,

      frameComponent:
        "details",

      guidedConstructionStep:
        1,

      componentValidation:
        guidedAdvanceValidation,

      microStepSemanticEvidence: {
        assessmentEstablished:
          true,

        sufficientForCurrentStep:
          true,

        usableForFinalStep:
          false,

        criterionEvidence:
          [],

        confidence:
          1,

        source:
          "deterministicSelfTestSemanticEvidence",
      },
    });

  const guidedAdvanceDecision =
    buildGuidedConstructionProgressionDecision({
      evidenceAssessment:
        guidedAdvanceAssessment,

      finalRephraseUsed:
        false,
    });

  const guidedAdvanceUpdate =
    applyGuidedConstructionProgression({
      state:
        guidedAdvanceState,

      progressionDecision:
        guidedAdvanceDecision,

      evidenceAssessment:
        guidedAdvanceAssessment,

      instructionalLocation:
        guidedAdvanceLocation,
    });

  const guidedAdvancePassed =
    guidedAdvanceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .SUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedAdvanceDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .ADVANCE_TO_NEXT_STEP &&

    guidedAdvanceUpdate
      ?.applied ===
      true &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionStep ===
      2 &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionEvidence
      ?.[1]
      ?.evidence ===
      guidedAdvanceResponse;

  results.push({
    name:
      "ED Guided Construction - Sufficient Step-1 evidence advances exactly to Step 2",

    passed:
      guidedAdvancePassed,

    expected: {
      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .SUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .ADVANCE_TO_NEXT_STEP,

      guidedConstructionStep:
        2,

      savedEvidence:
        guidedAdvanceResponse,
    },

    actual: {
      evidenceOutcome:
        guidedAdvanceAssessment
          ?.outcome ||
        null,

      decision:
        guidedAdvanceDecision
          ?.decision ||
        null,

      applied:
        guidedAdvanceUpdate
          ?.applied ===
        true,

      guidedConstructionStep:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      savedEvidence:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionEvidence
          ?.[1]
          ?.evidence ||
        null,
    },
  });

  // --------------------------------------------------
  // ED GC TEST 5 — FULL COMPONENT VALIDATION WINS
  // --------------------------------------------------

  const guidedCompleteState =
    createEssentialDetailGuidedTestState();

  guidedCompleteState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedCompleteState
    );

  const guidedCompleteResponse =
    "Teens compare themselves to influencers, which can make them feel inadequate and increase anxiety.";

  const guidedCompleteValidation =
    validateEssentialDetailResponse(
      guidedCompleteResponse,
      currentMainIdea
    );

  const guidedCompleteActual =
    await continueGuidedConstruction({
      state:
        guidedCompleteState,

      response:
        guidedCompleteResponse,

      componentValidation:
        guidedCompleteValidation,

      finalRephraseUsed:
        false,
    });

  const guidedCompletePassed =
    guidedCompleteValidation
      ?.valid ===
      true &&

    guidedCompleteActual
      ?.continuationStatus ===
      "established" &&

    guidedCompleteActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.yieldsToNormalComponentProgression ===
      true &&

    guidedCompleteState
      ?.frame
      ?.details
      ?.[0]
      ?.length ===
      0;

  results.push({
    name:
      "ED Guided Construction - Full valid Essential Detail immediately yields to normal component progression",

    passed:
      guidedCompletePassed,

    expected: {
      governedValidationPassed:
        true,

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .COMPONENT_COMPLETE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .COMPONENT_COMPLETE,

      yieldsToNormalComponentProgression:
        true,

      guidedConstructionDoesNotSaveComponent:
        true,
    },

    actual: {
      governedValidationPassed:
        guidedCompleteValidation
          ?.valid ===
        true,

      evidenceOutcome:
        guidedCompleteActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedCompleteActual
          ?.progressionDecision
          ?.decision ||
        null,

      yieldsToNormalComponentProgression:
        guidedCompleteActual
          ?.yieldsToNormalComponentProgression ===
        true,

      guidedConstructionDoesNotSaveComponent:
        guidedCompleteState
          ?.frame
          ?.details
          ?.[0]
          ?.length ===
        0,
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

// LIVE RUNTIME + GOVERNED SITUATION TEST
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
    ?.progressionAuthorization
    ?.authorized === false &&

  repeatedTopicActual
    ?.progressionAuthorization
    ?.selectedContractId ===
    "IA-RNR-001";

  results.push({
    name:
      "IA Runtime - Repeated Key Topic produces relationship repair situation",

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

// LIVE RUNTIME + GOVERNED SITUATION TEST
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
      .COMPONENT_NEEDS_REVISION;

  results.push({
    name:
      "IA Runtime - Limited evidence produces component revision situation",

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
    },
  });

 // LIVE RUNTIME + GOVERNED SITUATION TEST 
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
        .NO_COMPONENT_EVIDENCE;

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
    },
  });

// LIVE RUNTIME + GOVERNED SITUATION TEST
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
      ?.progressionAuthorization
      ?.authorized === true &&

    validIsAboutActual
      ?.progressionAuthorization
      ?.selectedContractId ===
      "IA-RTP-001";

  results.push({
    name:
      "IA Runtime - Valid paraphrase produces ready-to-progress situation",

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
  // RUNTIME NORMALIZATION REGRESSION
  //
  // A student may naturally repeat the accepted Key Topic
  // as part of a complete "X is about Y" sentence.
  //
  // Runtime validation must evaluate the full student
  // response so the observable Key Topic relationship is
  // preserved, while Frame storage normalizes the saved
  // Is About by removing the repeated prefix.
  // --------------------------------------------------

  const prefixedIsAboutState =
    createIsAboutRuntimeTestState();

  prefixedIsAboutState.frame.keyTopic =
    "Social media";

  const prefixedIsAboutResponse =
    "Social media is about how online platforms affect teenagers' lives and well-being.";

  const expectedNormalizedIsAbout =
    "How online platforms affect teenagers' lives and well-being.";

  const prefixedIsAboutActual =
    await updateStateFromStudent(
      prefixedIsAboutState,
      prefixedIsAboutResponse
    );

  const prefixedIsAboutPassed =
    prefixedIsAboutActual
      ?.frame
      ?.isAbout ===
      expectedNormalizedIsAbout &&

    prefixedIsAboutActual
      ?.pending
      ?.type ===
      "confirmIsAbout" &&

    prefixedIsAboutActual
      ?.componentInstructionalFinding
      ?.componentCriteriaStatus ===
      "satisfied" &&

    prefixedIsAboutActual
      ?.componentInstructionalFinding
      ?.relationshipStatus ===
      "established" &&

    prefixedIsAboutActual
      ?.instructionalSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .READY_TO_PROGRESS &&

    prefixedIsAboutActual
      ?.progressionAuthorization
      ?.authorized === true;

  results.push({
    name:
      "IA Runtime - Full Key Topic prefix is validated before normalized storage",

    passed:
      prefixedIsAboutPassed,

    response:
      prefixedIsAboutResponse,

    expected: {
      savedIsAbout:
        expectedNormalizedIsAbout,

      pendingType:
        "confirmIsAbout",

      componentCriteriaStatus:
        "satisfied",

      relationshipStatus:
        "established",

      governedSituation:
        INSTRUCTIONAL_SITUATIONS
          .READY_TO_PROGRESS,

      progressionAuthorized:
        true,
    },

    actual: {
      savedIsAbout:
        prefixedIsAboutActual
          ?.frame
          ?.isAbout || null,

      pendingType:
        prefixedIsAboutActual
          ?.pending
          ?.type || null,

      componentCriteriaStatus:
        prefixedIsAboutActual
          ?.componentInstructionalFinding
          ?.componentCriteriaStatus ||
        null,

      relationshipStatus:
        prefixedIsAboutActual
          ?.componentInstructionalFinding
          ?.relationshipStatus ||
        null,

      governedSituation:
        prefixedIsAboutActual
          ?.instructionalSituation
          ?.instructionalSituation ||
        null,

      progressionAuthorized:
        prefixedIsAboutActual
          ?.progressionAuthorization
          ?.authorized === true,
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
  // This test exercises the governed refresh directly so
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
      false;
  
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
      true;

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
    },
  });

    // --------------------------------------------------
  // IA RUNTIME — INTERACTION-ONLY LANGUAGE MUST NOT
  // BECOME IS ABOUT EVIDENCE
  //
  // Protects the live bug discovered during physical
  // testing:
  //
  // "I understand it's about social media, but I can't
  // explain the whole thing yet."
  //
  // The Observation Layer may identify uncertainty, but
  // no actual Is About contribution is present.
  //
  // The deterministic Interaction Finding must therefore
  // classify the response as interaction-only and prevent
  // it from functioning as component evidence.
  // --------------------------------------------------

  const interactionOnlyEvidenceState =
    buildEvidenceState(
      {
        ...createIsAboutRuntimeTestState(),

        frame: {
          ...createIsAboutRuntimeTestState()
            .frame,

          keyTopic:
            "Social media",
        },
      },

      "I understand it's about social media, but I can't explain the whole thing yet.",

      {
        version:
          "1.0",

        source:
          "aiObservation",

        studentInteraction:
          "I understand it's about social media, but I can't explain the whole thing yet.",

        observations: [
          {
            category:
              "uncertaintyExpression",

            evidenceText:
              "I can't explain the whole thing yet",

            confidence:
              1,
          },
        ],

        componentContribution: {
          observed:
            false,

          evidenceText:
            "",
        },

        ambiguityPresent:
          false,
      }
    );

  const interactionOnlyAssessment =
    buildInstructionalAssessment(
      interactionOnlyEvidenceState
    );

  const interactionOnlyFinding =
    buildInteractionInstructionalFinding(
      interactionOnlyEvidenceState,
      interactionOnlyAssessment
    );

  const interactionOnlyPassed =
    interactionOnlyFinding
      ?.responseFunctionsOnlyAsInteraction ===
      true &&

    interactionOnlyFinding
      ?.componentEvidenceFinding ===
      "noComponentEvidenceObserved" &&

    interactionOnlyFinding
      ?.evidence
      ?.componentContribution
      ?.observed ===
      false;

  results.push({
    name:
      "IA Runtime - Topic-reference uncertainty does not become Is About evidence",

    passed:
      interactionOnlyPassed,

    expected: {
      responseFunctionsOnlyAsInteraction:
        true,

      componentEvidenceFinding:
        "noComponentEvidenceObserved",

      componentContributionObserved:
        false,
    },

    actual: {
      responseFunctionsOnlyAsInteraction:
        interactionOnlyFinding
          ?.responseFunctionsOnlyAsInteraction ??
        null,

      componentEvidenceFinding:
        interactionOnlyFinding
          ?.componentEvidenceFinding ||
        null,

      componentContributionObserved:
        interactionOnlyFinding
          ?.evidence
          ?.componentContribution
          ?.observed ??
        null,
    },
  });

  // --------------------------------------------------
  // IA RUNTIME — MIXED INTERACTION + COMPONENT CONTENT
  //
  // Protects the live case where a student response
  // contains both uncertainty language and genuine
  // student-owned Is About content.
  //
  // The interaction wrapper must not be saved as part of
  // the Frame component.
  //
  // Only the exact verbatim component contribution
  // identified by the governed Observation Report may be
  // validated and preserved.
  // --------------------------------------------------

  const mixedContributionState =
    createIsAboutRuntimeTestState();

  mixedContributionState.frame = {
    ...mixedContributionState.frame,

    keyTopic:
      "Social media",

    isAbout:
      "",
  };

  mixedContributionState.observationReport = {
    version:
      "1.0",

    source:
      "aiObservation",

    studentInteraction:
      "I'm kind of stuck, but I think social media can change how teens feel about themselves.",

    observations: [
      {
        category:
          "uncertaintyExpression",

        evidenceText:
          "I'm kind of stuck",

        confidence:
          1,
      },
    ],

    componentContribution: {
      observed:
        true,

      evidenceText:
        "social media can change how teens feel about themselves.",
    },

    ambiguityPresent:
      false,
  };

  const mixedContributionRawResponse =
    "I'm kind of stuck, but I think social media can change how teens feel about themselves.";

  const mixedContributionExpected =
    "Social media can change how teens feel about themselves.";

  await applyIsAboutCapture(
    mixedContributionState,
    mixedContributionRawResponse,
    {
      captureMode:
        "build",
    }
  );

  const mixedContributionPassed =
    mixedContributionState
      ?.frame
      ?.isAbout ===
      mixedContributionExpected &&

    mixedContributionState
      ?.frame
      ?.isAbout !==
      mixedContributionRawResponse &&

    mixedContributionState
      ?.pending
      ?.type ===
      "confirmIsAbout";

  results.push({
    name:
      "IA Runtime - Mixed uncertainty preserves only student-owned Is About contribution",

    passed:
      mixedContributionPassed,

    expected: {
      savedIsAbout:
        mixedContributionExpected,

      rawInteractionSaved:
        false,

      pendingType:
        "confirmIsAbout",
    },

    actual: {
      savedIsAbout:
        mixedContributionState
          ?.frame
          ?.isAbout || "",

      rawInteractionSaved:
        mixedContributionState
          ?.frame
          ?.isAbout ===
        mixedContributionRawResponse,

      pendingType:
        mixedContributionState
          ?.pending
          ?.type || null,
    },
  });

  // --------------------------------------------------
  // GUIDED CONSTRUCTION — IS ABOUT TARGETED VERIFICATION
  // --------------------------------------------------
  //
  // Verifies the newly integrated Is About Guided
  // Construction pathway without changing production
  // behavior.
  //
  // --------------------------------------------------

  // --------------------------------------------------
  // IA GC TEST 1 — STEP-AWARE THINKING MOVE SELECTION
  // --------------------------------------------------

  const guidedSelectionContract =
    INSTRUCTIONAL_PLAYBOOK
      ?.isAbout
      ?.genuineStruggle;

  const guidedSelectionResults =
    [1, 2, 3].map(
      (guidedConstructionStep) => {
        const testState =
          createIsAboutRuntimeTestState();

        testState.pending = {
          type:
            "reviseIsAbout",

          captureMode:
            "build",

          progressiveSupportStage:
            3,

          guidedConstructionStep,
        };

        const selectedScaffold =
          selectProgressiveSupportScaffold(
            guidedSelectionContract,
            testState
          );

        const expectedRule =
          GUIDED_CONSTRUCTION_RULES
            ?.isAbout
            ?.steps
            ?.[guidedConstructionStep];

        return {
          guidedConstructionStep,

          passed:
            selectedScaffold
              ?.progressiveSupportStage ===
              3 &&

            selectedScaffold
              ?.guidedConstructionStep ===
              guidedConstructionStep &&

            selectedScaffold
              ?.thinkingMove ===
              expectedRule
                ?.thinkingMove,

          actualThinkingMove:
            selectedScaffold
              ?.thinkingMove ||
            null,

          expectedThinkingMove:
            expectedRule
              ?.thinkingMove ||
            null,
        };
      }
    );

  const guidedSelectionPassed =
    guidedSelectionResults.every(
      (result) =>
        result.passed === true
    );

  results.push({
    name:
      "IA Guided Construction - Stage 3 selects the correct Step 1, 2, and 3 Thinking Moves",

    passed:
      guidedSelectionPassed,

    expected: {
      progressiveSupportStage:
        3,

      guidedSteps:
        [1, 2, 3],

      allThinkingMovesMatchRules:
        true,
    },

    actual: {
      allThinkingMovesMatchRules:
        guidedSelectionPassed,

      stepResults:
        guidedSelectionResults,
    },
  });

  // --------------------------------------------------
  // IA GC TEST 2 — INSUFFICIENT STEP-1 EVIDENCE STAYS
  // --------------------------------------------------

  const guidedStayState =
    createIsAboutRuntimeTestState();

  guidedStayState.pending = {
    type:
      "reviseIsAbout",

    captureMode:
      "build",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      1,

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
  };

  guidedStayState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedStayState
    );

  const guidedStayValidation =
    validateIsAboutResponse(
      "idk",
      keyTopic
    );

  const guidedStayActual =
    await continueGuidedConstruction({
      state:
        guidedStayState,

      response:
        "idk",

      componentValidation:
        guidedStayValidation,

      finalRephraseUsed:
        false,
    });

  const guidedStayPassed =
    guidedStayActual
      ?.continuationStatus ===
      "established" &&

    guidedStayActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .INSUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedStayActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .STAY_CURRENT_STEP &&

    guidedStayState
      ?.pending
      ?.guidedConstructionStep ===
      1 &&

    !guidedStayState
      ?.pending
      ?.guidedConstructionEvidence;

  results.push({
    name:
      "IA Guided Construction - Insufficient Step-1 evidence stays on Step 1",

    passed:
      guidedStayPassed,

    expected: {
      continuationStatus:
        "established",

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .INSUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .STAY_CURRENT_STEP,

      guidedConstructionStep:
        1,

      guidedEvidenceSaved:
        false,
    },

    actual: {
      continuationStatus:
        guidedStayActual
          ?.continuationStatus ||
        null,

      evidenceOutcome:
        guidedStayActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedStayActual
          ?.progressionDecision
          ?.decision ||
        null,

      guidedConstructionStep:
        guidedStayState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      guidedEvidenceSaved:
        Boolean(
          guidedStayState
            ?.pending
            ?.guidedConstructionEvidence
        ),
    },
  });

  // --------------------------------------------------
  // IA GC TEST 3 — SUFFICIENT STEP-1 EVIDENCE ADVANCES
  // --------------------------------------------------

  const guidedAdvanceState =
    createIsAboutRuntimeTestState();

  guidedAdvanceState.pending = {
    type:
      "reviseIsAbout",

    captureMode:
      "build",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      1,
  };

  const guidedAdvanceLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedAdvanceState
    );

  guidedAdvanceState
    .pending
    .guidedConstructionLocation =
    structuredClone(
      guidedAdvanceLocation
    );

  const guidedAdvanceResponse =
    "Plants use sunlight";

  const guidedAdvanceValidation =
    validateIsAboutResponse(
      guidedAdvanceResponse,
      keyTopic
    );

  const guidedAdvanceAssessment =
    assessGuidedConstructionEvidence({
      state:
        guidedAdvanceState,

      response:
        guidedAdvanceResponse,

      frameComponent:
        "isAbout",

      guidedConstructionStep:
        1,

      componentValidation:
        guidedAdvanceValidation,

      microStepSemanticEvidence: {
        assessmentEstablished:
          true,

        sufficientForCurrentStep:
          true,

        usableForFinalStep:
          false,

        criterionEvidence:
          [],

        confidence:
          1,

        source:
          "deterministicSelfTestSemanticEvidence",
      },
    });

  const guidedAdvanceDecision =
    buildGuidedConstructionProgressionDecision({
      evidenceAssessment:
        guidedAdvanceAssessment,

      finalRephraseUsed:
        false,
    });

  const guidedAdvanceUpdate =
    applyGuidedConstructionProgression({
      state:
        guidedAdvanceState,

      progressionDecision:
        guidedAdvanceDecision,

      evidenceAssessment:
        guidedAdvanceAssessment,

      instructionalLocation:
        guidedAdvanceLocation,
    });

  const guidedAdvancePassed =
    guidedAdvanceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .SUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedAdvanceDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .ADVANCE_TO_NEXT_STEP &&

    guidedAdvanceUpdate
      ?.applied ===
      true &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionStep ===
      2 &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionEvidence
      ?.[1]
      ?.evidence ===
      guidedAdvanceResponse;

  results.push({
    name:
      "IA Guided Construction - Sufficient Step-1 evidence advances exactly to Step 2",

    passed:
      guidedAdvancePassed,

    expected: {
      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .SUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .ADVANCE_TO_NEXT_STEP,

      guidedConstructionStep:
        2,

      savedEvidence:
        guidedAdvanceResponse,
    },

    actual: {
      evidenceOutcome:
        guidedAdvanceAssessment
          ?.outcome ||
        null,

      decision:
        guidedAdvanceDecision
          ?.decision ||
        null,

      applied:
        guidedAdvanceUpdate
          ?.applied ===
        true,

      guidedConstructionStep:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      savedEvidence:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionEvidence
          ?.[1]
          ?.evidence ||
        null,
    },
  });

  // --------------------------------------------------
  // IA GC TEST 4 — FULL COMPONENT VALIDATION WINS
  // --------------------------------------------------

  const guidedCompleteState =
    createIsAboutRuntimeTestState();

  guidedCompleteState.pending = {
    type:
      "reviseIsAbout",

    captureMode:
      "build",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      1,
  };

  guidedCompleteState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedCompleteState
    );

  const guidedCompleteValidation =
    validateIsAboutResponse(
      validIsAboutResponse,
      keyTopic
    );

  const guidedCompleteActual =
    await continueGuidedConstruction({
      state:
        guidedCompleteState,

      response:
        validIsAboutResponse,

      componentValidation:
        guidedCompleteValidation,

      finalRephraseUsed:
        false,
    });

  const guidedCompletePassed =
    guidedCompleteValidation
      ?.valid ===
      true &&

    guidedCompleteActual
      ?.continuationStatus ===
      "established" &&

    guidedCompleteActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.yieldsToNormalComponentProgression ===
      true &&

    guidedCompleteState
      ?.frame
      ?.isAbout ===
      "";

  results.push({
    name:
      "IA Guided Construction - Full valid Is About immediately yields to normal component progression",

    passed:
      guidedCompletePassed,

    expected: {
      governedValidationPassed:
        true,

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .COMPONENT_COMPLETE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .COMPONENT_COMPLETE,

      yieldsToNormalComponentProgression:
        true,

      guidedConstructionDoesNotSaveComponent:
        true,
    },

    actual: {
      governedValidationPassed:
        guidedCompleteValidation
          ?.valid ===
        true,

      evidenceOutcome:
        guidedCompleteActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedCompleteActual
          ?.progressionDecision
          ?.decision ||
        null,

      yieldsToNormalComponentProgression:
        guidedCompleteActual
          ?.yieldsToNormalComponentProgression ===
        true,

      guidedConstructionDoesNotSaveComponent:
        guidedCompleteState
          ?.frame
          ?.isAbout ===
        "",
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

    // --------------------------------------------------
  // MI RUNTIME — MIXED INTERACTION + COMPONENT CONTENT
  //
  // Protects the live case where a student response
  // contains both uncertainty language and genuine
  // student-owned Main Idea content.
  //
  // The interaction wrapper must not be saved as part of
  // the Frame component.
  //
  // Only the exact verbatim component contribution
  // identified by the governed Observation Report may be
  // validated and preserved.
  // --------------------------------------------------

  const mixedMainIdeaContributionState =
    createMainIdeaTestState();

  mixedMainIdeaContributionState
    .observationReport = {
      version:
        "1.0",

      source:
        "aiObservation",

      studentInteraction:
        "I'm kind of stuck, but social media can increase anxiety and stress.",

      observations: [
        {
          category:
            "uncertaintyExpression",

          evidenceText:
            "I'm kind of stuck",

          confidence:
            1,
        },
      ],

      componentContribution: {
        observed:
          true,

        evidenceText:
          "social media can increase anxiety and stress.",
      },

      ambiguityPresent:
        false,
    };

  const mixedMainIdeaRawResponse =
    "I'm kind of stuck, but social media can increase anxiety and stress.";

  const mixedMainIdeaExpected =
    "social media can increase anxiety and stress.";

  await applyMainIdeaCapture(
    mixedMainIdeaContributionState,
    mixedMainIdeaRawResponse,
    {
      captureMode:
        "required",
    }
  );

  const mixedMainIdeaSaved =
    mixedMainIdeaContributionState
      ?.frame
      ?.parentItems
      ?.[0] || "";

  const mixedMainIdeaAttempted =
    mixedMainIdeaContributionState
      ?.componentInstructionalFinding
      ?.evidence
      ?.attemptedMainIdea || "";

  const mixedMainIdeaContributionPassed =
    mixedMainIdeaSaved ===
      mixedMainIdeaExpected &&

    mixedMainIdeaSaved !==
      mixedMainIdeaRawResponse &&

    mixedMainIdeaAttempted ===
      mixedMainIdeaRawResponse &&

    mixedMainIdeaContributionState
      ?.pending
      ?.type ===
      "offerAnotherMainIdea";

  results.push({
    name:
      "MI Runtime - Mixed uncertainty preserves only student-owned Main Idea contribution",

    passed:
      mixedMainIdeaContributionPassed,

    expected: {
      savedMainIdea:
        mixedMainIdeaExpected,

      rawInteractionSaved:
        false,

      attemptedMainIdea:
        mixedMainIdeaRawResponse,

      pendingType:
        "offerAnotherMainIdea",
    },

    actual: {
      savedMainIdea:
        mixedMainIdeaSaved,

      rawInteractionSaved:
        mixedMainIdeaSaved ===
        mixedMainIdeaRawResponse,

      attemptedMainIdea:
        mixedMainIdeaAttempted,

      pendingType:
        mixedMainIdeaContributionState
          ?.pending
          ?.type || null,
    },
  });

  // --------------------------------------------------
  // GUIDED CONSTRUCTION — MAIN IDEA TARGETED VERIFICATION
  // --------------------------------------------------
  //
  // Confirms the Main Idea integration uses the shared
  // Guided Construction runtime without changing normal
  // Main Idea validation or progression authority.
  //
  // --------------------------------------------------

  // --------------------------------------------------
  // MI GC TEST 1 — STEP-AWARE THINKING MOVE SELECTION
  // --------------------------------------------------

  const guidedSelectionContract =
    INSTRUCTIONAL_PLAYBOOK
      ?.mainIdeas
      ?.genuineStruggle;

  const guidedSelectionResults =
    [1, 2, 3].map(
      (guidedConstructionStep) => {
        const testState =
          createMainIdeaTestState();

        testState.pending = {
          type:
            "collectMainIdea",

          captureMode:
            "required",

          progressiveSupportStage:
            3,

          guidedConstructionStep,
        };

        const selectedScaffold =
          selectProgressiveSupportScaffold(
            guidedSelectionContract,
            testState
          );

        const expectedRule =
          GUIDED_CONSTRUCTION_RULES
            ?.mainIdeas
            ?.steps
            ?.[guidedConstructionStep];

        return {
          guidedConstructionStep,

          passed:
            selectedScaffold
              ?.progressiveSupportStage ===
              3 &&

            selectedScaffold
              ?.guidedConstructionStep ===
              guidedConstructionStep &&

            selectedScaffold
              ?.thinkingMove ===
              expectedRule
                ?.thinkingMove,

          actualThinkingMove:
            selectedScaffold
              ?.thinkingMove ||
            null,

          expectedThinkingMove:
            expectedRule
              ?.thinkingMove ||
            null,
        };
      }
    );

  const guidedSelectionPassed =
    guidedSelectionResults.every(
      (result) =>
        result.passed === true
    );

  results.push({
    name:
      "MI Guided Construction - Stage 3 selects the correct Step 1, 2, and 3 Thinking Moves",

    passed:
      guidedSelectionPassed,

    expected: {
      progressiveSupportStage:
        3,

      guidedSteps:
        [1, 2, 3],

      allThinkingMovesMatchRules:
        true,
    },

    actual: {
      allThinkingMovesMatchRules:
        guidedSelectionPassed,

      stepResults:
        guidedSelectionResults,
    },
  });

  // --------------------------------------------------
  // MI GC TEST 2 — INSUFFICIENT STEP-1 EVIDENCE STAYS
  // --------------------------------------------------

  const guidedStayState =
    createMainIdeaTestState();

  guidedStayState.pending = {
    type:
      "collectMainIdea",

    captureMode:
      "required",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      1,

    instructionalFinding: {
      frameComponent:
        "mainIdeas",

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "noComponentEvidence",
    },
  };

  guidedStayState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedStayState
    );

  const guidedStayValidation =
    validateMainIdeaResponse(
      "idk",
      keyTopic,
      isAbout
    );

  const guidedStayActual =
    await continueGuidedConstruction({
      state:
        guidedStayState,

      response:
        "idk",

      componentValidation:
        guidedStayValidation,

      finalRephraseUsed:
        false,
    });

  const guidedStayPassed =
    guidedStayActual
      ?.continuationStatus ===
      "established" &&

    guidedStayActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .INSUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedStayActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .STAY_CURRENT_STEP &&

    guidedStayState
      ?.pending
      ?.guidedConstructionStep ===
      1 &&

    !guidedStayState
      ?.pending
      ?.guidedConstructionEvidence;

  results.push({
    name:
      "MI Guided Construction - Insufficient Step-1 evidence stays on Step 1",

    passed:
      guidedStayPassed,

    expected: {
      continuationStatus:
        "established",

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .INSUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .STAY_CURRENT_STEP,

      guidedConstructionStep:
        1,

      guidedEvidenceSaved:
        false,
    },

    actual: {
      continuationStatus:
        guidedStayActual
          ?.continuationStatus ||
        null,

      evidenceOutcome:
        guidedStayActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedStayActual
          ?.progressionDecision
          ?.decision ||
        null,

      guidedConstructionStep:
        guidedStayState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      guidedEvidenceSaved:
        Boolean(
          guidedStayState
            ?.pending
            ?.guidedConstructionEvidence
        ),
    },
  });

  // --------------------------------------------------
  // MI GC TEST 3 — SUFFICIENT STEP-1 EVIDENCE ADVANCES
  //
  // Supplies bounded semantic evidence directly so this
  // remains deterministic and does not make another AI
  // call merely to test the progression brain.
  // --------------------------------------------------

  const guidedAdvanceState =
    createMainIdeaTestState();

  guidedAdvanceState.pending = {
    type:
      "collectMainIdea",

    captureMode:
      "required",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      1,
  };

  const guidedAdvanceLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedAdvanceState
    );

  guidedAdvanceState
    .pending
    .guidedConstructionLocation =
    structuredClone(
      guidedAdvanceLocation
    );

  const guidedAdvanceResponse =
    "Effects on mental health";

  const guidedAdvanceValidation =
    validateMainIdeaResponse(
      guidedAdvanceResponse,
      keyTopic,
      isAbout
    );

  const guidedAdvanceAssessment =
    assessGuidedConstructionEvidence({
      state:
        guidedAdvanceState,

      response:
        guidedAdvanceResponse,

      frameComponent:
        "mainIdeas",

      guidedConstructionStep:
        1,

      componentValidation:
        guidedAdvanceValidation,

      microStepSemanticEvidence: {
        assessmentEstablished:
          true,

        sufficientForCurrentStep:
          true,

        usableForFinalStep:
          false,

        criterionEvidence:
          [],

        confidence:
          1,

        source:
          "deterministicSelfTestSemanticEvidence",
      },
    });

  const guidedAdvanceDecision =
    buildGuidedConstructionProgressionDecision({
      evidenceAssessment:
        guidedAdvanceAssessment,

      finalRephraseUsed:
        false,
    });

  const guidedAdvanceUpdate =
    applyGuidedConstructionProgression({
      state:
        guidedAdvanceState,

      progressionDecision:
        guidedAdvanceDecision,

      evidenceAssessment:
        guidedAdvanceAssessment,

      instructionalLocation:
        guidedAdvanceLocation,
    });

  const guidedAdvancePassed =
    guidedAdvanceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .SUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedAdvanceDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .ADVANCE_TO_NEXT_STEP &&

    guidedAdvanceUpdate
      ?.applied ===
      true &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionStep ===
      2 &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionEvidence
      ?.[1]
      ?.evidence ===
      guidedAdvanceResponse;

  results.push({
    name:
      "MI Guided Construction - Sufficient Step-1 evidence advances exactly to Step 2",

    passed:
      guidedAdvancePassed,

    expected: {
      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .SUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .ADVANCE_TO_NEXT_STEP,

      guidedConstructionStep:
        2,

      savedEvidence:
        guidedAdvanceResponse,
    },

    actual: {
      evidenceOutcome:
        guidedAdvanceAssessment
          ?.outcome ||
        null,

      decision:
        guidedAdvanceDecision
          ?.decision ||
        null,

      applied:
        guidedAdvanceUpdate
          ?.applied ===
        true,

      guidedConstructionStep:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      savedEvidence:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionEvidence
          ?.[1]
          ?.evidence ||
        null,
    },
  });

  // --------------------------------------------------
  // MI GC TEST 4 — FULL COMPONENT VALIDATION WINS
  // --------------------------------------------------

  const guidedCompleteState =
    createMainIdeaTestState();

  guidedCompleteState.pending = {
    type:
      "collectMainIdea",

    captureMode:
      "required",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      1,
  };

  guidedCompleteState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedCompleteState
    );

  // Reuse the already-established governed validation
  // result from the existing Main Idea test above.
  const guidedCompleteValidation =
    governedValidActual;

  const guidedCompleteActual =
    await continueGuidedConstruction({
      state:
        guidedCompleteState,

      response:
        validMainIdea,

      componentValidation:
        guidedCompleteValidation,

      finalRephraseUsed:
        false,
    });

  const guidedCompletePassed =
    guidedCompleteValidation
      ?.valid ===
      true &&

    guidedCompleteActual
      ?.continuationStatus ===
      "established" &&

    guidedCompleteActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.yieldsToNormalComponentProgression ===
      true &&

    getIdeaList(
      guidedCompleteState
    ).length ===
      0;

  results.push({
    name:
      "MI Guided Construction - Full valid Main Idea immediately yields to normal component progression",

    passed:
      guidedCompletePassed,

    expected: {
      governedValidationPassed:
        true,

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .COMPONENT_COMPLETE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .COMPONENT_COMPLETE,

      yieldsToNormalComponentProgression:
        true,

      guidedConstructionDoesNotSaveComponent:
        true,
    },

    actual: {
      governedValidationPassed:
        guidedCompleteValidation
          ?.valid ===
        true,

      evidenceOutcome:
        guidedCompleteActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedCompleteActual
          ?.progressionDecision
          ?.decision ||
        null,

      yieldsToNormalComponentProgression:
        guidedCompleteActual
          ?.yieldsToNormalComponentProgression ===
        true,

      guidedConstructionDoesNotSaveComponent:
        getIdeaList(
          guidedCompleteState
        ).length ===
        0,
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
// Dedicated runtime save-path coverage remains separate
// from this governed So What validation suite.
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
  // continuation, and revision pathways through
  // updateStateFromStudent().
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
    "continueSoWhat" &&

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
      "continueSoWhat",

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
  // advances directly to So What confirmation.
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
      "confirmSoWhat" &&
    
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
        "confirmSoWhat",
      
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

  // --------------------------------------------------
  // GUIDED CONSTRUCTION — SO WHAT
  // TARGETED VERIFICATION
  // --------------------------------------------------
  //
  // Confirms that So What uses the shared Guided
  // Construction runtime while preserving:
  //
  // • normal governed So What validation authority;
  // • Step 1 / Step 2 / Step 3 Thinking Moves;
  // • deterministic stay / advance behavior;
  // • student-owned Guided Construction evidence only;
  // • immediate yield when the full So What is valid.
  //
  // --------------------------------------------------

  function createSoWhatGuidedTestState() {
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

    state.pending = {
      type:
        "continueSoWhat",

      captureMode:
        "initial",

      progressiveSupportStage:
        3,

      guidedConstructionStep:
        1,
    };

    return state;
  }

  // --------------------------------------------------
  // SW GC TEST 1 — STEP-AWARE THINKING MOVE SELECTION
  // --------------------------------------------------

  const guidedSelectionContract =
    INSTRUCTIONAL_PLAYBOOK
      ?.soWhat
      ?.genuineStruggle;

  const guidedSelectionResults =
    [1, 2, 3].map(
      (guidedConstructionStep) => {
        const testState =
          createSoWhatGuidedTestState();

        testState.pending
          .guidedConstructionStep =
          guidedConstructionStep;

        const selectedScaffold =
          selectProgressiveSupportScaffold(
            guidedSelectionContract,
            testState
          );

        const expectedRule =
          GUIDED_CONSTRUCTION_RULES
            ?.soWhat
            ?.steps
            ?.[guidedConstructionStep];

        return {
          guidedConstructionStep,

          passed:
            selectedScaffold
              ?.progressiveSupportStage ===
              3 &&

            selectedScaffold
              ?.guidedConstructionStep ===
              guidedConstructionStep &&

            selectedScaffold
              ?.thinkingMove ===
              expectedRule
                ?.thinkingMove,

          actualThinkingMove:
            selectedScaffold
              ?.thinkingMove ||
            null,

          expectedThinkingMove:
            expectedRule
              ?.thinkingMove ||
            null,
        };
      }
    );

  const guidedSelectionPassed =
    guidedSelectionResults.every(
      (result) =>
        result.passed === true
    );

  results.push({
    name:
      "SW Guided Construction - Stage 3 selects the correct Step 1, 2, and 3 Thinking Moves",

    passed:
      guidedSelectionPassed,

    expected: {
      progressiveSupportStage:
        3,

      guidedSteps:
        [1, 2, 3],

      allThinkingMovesMatchRules:
        true,
    },

    actual: {
      allThinkingMovesMatchRules:
        guidedSelectionPassed,

      stepResults:
        guidedSelectionResults,
    },
  });

  // --------------------------------------------------
  // SW GC TEST 2 — INSUFFICIENT STEP-1 EVIDENCE STAYS
  // --------------------------------------------------

  const guidedStayState =
    createSoWhatGuidedTestState();

  guidedStayState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedStayState
    );

  const guidedStayValidation =
    validateSoWhatResponse(
      "idk",
      instructionalContext
    );

  const guidedStayActual =
    await continueGuidedConstruction({
      state:
        guidedStayState,

      response:
        "idk",

      componentValidation:
        guidedStayValidation,

      finalRephraseUsed:
        false,
    });

  const guidedStayPassed =
    guidedStayActual
      ?.continuationStatus ===
      "established" &&

    guidedStayActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .INSUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedStayActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .STAY_CURRENT_STEP &&

    guidedStayState
      ?.pending
      ?.guidedConstructionStep ===
      1 &&

    !guidedStayState
      ?.pending
      ?.guidedConstructionEvidence;

  results.push({
    name:
      "SW Guided Construction - Insufficient Step-1 evidence stays on Step 1",

    passed:
      guidedStayPassed,

    expected: {
      continuationStatus:
        "established",

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .INSUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .STAY_CURRENT_STEP,

      guidedConstructionStep:
        1,

      guidedEvidenceSaved:
        false,
    },

    actual: {
      continuationStatus:
        guidedStayActual
          ?.continuationStatus ||
        null,

      evidenceOutcome:
        guidedStayActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedStayActual
          ?.progressionDecision
          ?.decision ||
        null,

      guidedConstructionStep:
        guidedStayState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      guidedEvidenceSaved:
        Boolean(
          guidedStayState
            ?.pending
            ?.guidedConstructionEvidence
        ),
    },
  });

  // --------------------------------------------------
  // SW GC TEST 3 — SUFFICIENT STEP-1 EVIDENCE ADVANCES
  //
  // Bounded semantic evidence is supplied directly so
  // this test verifies the deterministic progression
  // brain without making an additional AI call.
  // --------------------------------------------------

  const guidedAdvanceState =
    createSoWhatGuidedTestState();

  const guidedAdvanceLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedAdvanceState
    );

  guidedAdvanceState
    .pending
    .guidedConstructionLocation =
    structuredClone(
      guidedAdvanceLocation
    );

  const guidedAdvanceResponse =
    "Online comparison and constant notifications both create pressure for teens.";

  const guidedAdvanceValidation =
    validateSoWhatResponse(
      guidedAdvanceResponse,
      instructionalContext
    );

  const guidedAdvanceAssessment =
    assessGuidedConstructionEvidence({
      state:
        guidedAdvanceState,

      response:
        guidedAdvanceResponse,

      frameComponent:
        "soWhat",

      guidedConstructionStep:
        1,

      componentValidation:
        guidedAdvanceValidation,

      microStepSemanticEvidence: {
        assessmentEstablished:
          true,

        sufficientForCurrentStep:
          true,

        usableForFinalStep:
          false,

        criterionEvidence:
          [],

        confidence:
          1,

        source:
          "deterministicSelfTestSemanticEvidence",
      },
    });

  const guidedAdvanceDecision =
    buildGuidedConstructionProgressionDecision({
      evidenceAssessment:
        guidedAdvanceAssessment,

      finalRephraseUsed:
        false,
    });

  const guidedAdvanceUpdate =
    applyGuidedConstructionProgression({
      state:
        guidedAdvanceState,

      progressionDecision:
        guidedAdvanceDecision,

      evidenceAssessment:
        guidedAdvanceAssessment,

      instructionalLocation:
        guidedAdvanceLocation,
    });

  const guidedAdvancePassed =
    guidedAdvanceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .SUFFICIENT_MICRO_STEP_EVIDENCE &&

    guidedAdvanceDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .ADVANCE_TO_NEXT_STEP &&

    guidedAdvanceUpdate
      ?.applied ===
      true &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionStep ===
      2 &&

    guidedAdvanceState
      ?.pending
      ?.guidedConstructionEvidence
      ?.[1]
      ?.evidence ===
      guidedAdvanceResponse;

  results.push({
    name:
      "SW Guided Construction - Sufficient Step-1 evidence advances exactly to Step 2",

    passed:
      guidedAdvancePassed,

    expected: {
      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .SUFFICIENT_MICRO_STEP_EVIDENCE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .ADVANCE_TO_NEXT_STEP,

      guidedConstructionStep:
        2,

      savedEvidence:
        guidedAdvanceResponse,
    },

    actual: {
      evidenceOutcome:
        guidedAdvanceAssessment
          ?.outcome ||
        null,

      decision:
        guidedAdvanceDecision
          ?.decision ||
        null,

      applied:
        guidedAdvanceUpdate
          ?.applied ===
        true,

      guidedConstructionStep:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionStep ||
        null,

      savedEvidence:
        guidedAdvanceState
          ?.pending
          ?.guidedConstructionEvidence
          ?.[1]
          ?.evidence ||
        null,
    },
  });

  // --------------------------------------------------
  // SW GC TEST 4 — FULL COMPONENT VALIDATION WINS
  //
  // A valid So What at any Guided Construction step must
  // immediately yield authority back to normal component
  // progression.
  //
  // Guided Construction may never become an extra hoop.
  // --------------------------------------------------

  const guidedCompleteState =
    createSoWhatGuidedTestState();

  guidedCompleteState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      guidedCompleteState
    );

  const guidedCompleteValidation =
    await validateSoWhatResponseGoverned(
      supportedSoWhat,
      instructionalContext
    );

  const guidedCompleteActual =
    await continueGuidedConstruction({
      state:
        guidedCompleteState,

      response:
        supportedSoWhat,

      componentValidation:
        guidedCompleteValidation,

      finalRephraseUsed:
        false,
    });

  const guidedCompletePassed =
    guidedCompleteValidation
      ?.valid ===
      true &&

    guidedCompleteActual
      ?.continuationStatus ===
      "established" &&

    guidedCompleteActual
      ?.evidenceAssessment
      ?.outcome ===
      GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.progressionDecision
      ?.decision ===
      GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
        .COMPONENT_COMPLETE &&

    guidedCompleteActual
      ?.yieldsToNormalComponentProgression ===
      true &&

    guidedCompleteState
      ?.frame
      ?.soWhat ===
      "";

  results.push({
    name:
      "SW Guided Construction - Full valid So What immediately yields to normal component progression",

    passed:
      guidedCompletePassed,

    expected: {
      governedValidationPassed:
        true,

      evidenceOutcome:
        GUIDED_CONSTRUCTION_EVIDENCE_OUTCOMES
          .COMPONENT_COMPLETE,

      decision:
        GUIDED_CONSTRUCTION_PROGRESSION_DECISIONS
          .COMPONENT_COMPLETE,

      yieldsToNormalComponentProgression:
        true,

      guidedConstructionDoesNotSaveComponent:
        true,
    },

    actual: {
      governedValidationPassed:
        guidedCompleteValidation
          ?.valid ===
        true,

      evidenceOutcome:
        guidedCompleteActual
          ?.evidenceAssessment
          ?.outcome ||
        null,

      decision:
        guidedCompleteActual
          ?.progressionDecision
          ?.decision ||
        null,

      yieldsToNormalComponentProgression:
        guidedCompleteActual
          ?.yieldsToNormalComponentProgression ===
        true,

      guidedConstructionDoesNotSaveComponent:
        guidedCompleteState
          ?.frame
          ?.soWhat ===
        "",
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
// STAGE 1 FINAL-QUESTION VALIDATION TESTS
//
// Verifies that a governed Stage-1 communication response
// must end with the exact deterministic final question
// supplied by the Communication License.
//
// These tests exercise the validator directly and do not
// call AI.
// ------------------------------------------------------

function runStage1FinalQuestionValidationSelfTests() {
  const finalQuestionTemplate =
    "Looking at your Main Idea, what is one specific thing that could help explain or support it?";

  const communicationLicense = {
    permissions: {
      maximumQuestions:
        1,
    },

    prohibitions: {
      mayClaimUnsupportedProgress:
        false,

      mayGenerateStudentWork:
        false,
    },

    relationshipStatus:
      "undetermined",

    studentFacingFormat: {
      stage1PromptVisualArchitecture: {
        required:
          true,

        componentIcon:
          "✍️",

        componentLabel:
          "Essential Detail",

        leadIn:
          "I’ll help you build an ✍️ Essential Detail by connecting back to what you already have.",

        parentContextIcon:
          "💡",

        parentContextLabel:
          "Main Idea",

        requireParentContext:
          true,

        requireBridgeLine:
          true,

        bridgeLine:
          "Think about one of these:",

        requireVisualSeparation:
          true,

        thinkingLenses: [
          {
            icon:
              "📌",

            label:
              "a fact",
          },

          {
            icon:
              "💬",

            label:
              "an example",
          },

          {
            icon:
              "👀",

            label:
              "something you noticed or learned",
          },
        ],

        finalQuestionTemplate,

        requireSingleFinalQuestion:
          true,
      },
    },
  };

  const validResponse = [
    "I’ll help you build an ✍️ Essential Detail by connecting back to what you already have.",
    "",
    "💡 Main Idea: Mental health",
    "",
    "Think about one of these:",
    "",
    "📌 a fact",
    "💬 an example",
    "👀 something you noticed or learned",
    "",
    finalQuestionTemplate,
  ].join("\n");

  const alteredResponse =
    validResponse.replace(
      finalQuestionTemplate,
      "What Essential Detail would you like to add?"
    );

  const validResult =
    validateInstructionalCommunicationResponse(
      validResponse,
      communicationLicense
    );

  const alteredResult =
    validateInstructionalCommunicationResponse(
      alteredResponse,
      communicationLicense
    );

  const passed =
    validResult.valid === true &&
    alteredResult.valid === false &&
    alteredResult.violations.includes(
      "stage1PromptFinalQuestionRequired"
    );

  return {
    passed,

    passedCount:
      passed
        ? 1
        : 0,

    failedCount:
      passed
        ? 0
        : 1,

    total:
      1,

    results: [
      {
        name:
          "Stage 1 - Exact deterministic final question is enforced",

        passed,

        expected: {
          exactQuestionAccepted:
            true,

          alteredQuestionRejected:
            true,

          violation:
            "stage1PromptFinalQuestionRequired",
        },

        actual: {
          exactQuestionAccepted:
            validResult.valid,

          alteredQuestionRejected:
            alteredResult.valid === false,

          violations:
            alteredResult.violations,
        },
      },
    ],
  };
}

  function formatStage1FinalQuestionValidationSelfTestResults(
    testResults
  ) {
    const lines = [
      "🎯 KAW STAGE 1 FINAL-QUESTION VALIDATION",
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
      "🚀 Stage 1 final-question enforcement is operating correctly."
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
// PROGRESSIVE SUPPORT LIFECYCLE SELF-TESTS
// ------------------------------------------------------
//
// Verifies the shared deterministic Stage lifecycle for
// Is About, Main Idea, Essential Detail, and So What.
//
// Each component must demonstrate:
//
// • first Genuine Struggle -> Stage 1;
// • continued struggle -> Stage 2;
// • continued struggle -> Stage 3;
// • further struggle remains capped at Stage 3;
// • non-struggle governed support clears the stage;
// • struggle after reset restarts at Stage 1;
// • a new exact instructional location starts at Stage 1.
//
// These tests exercise the shared lifecycle and each
// component's actual governed contract executor.
// ------------------------------------------------------

async function runProgressiveSupportSelfTests() {
  const results = [];

  const componentCases = [
    {
      label: "IA",
      frameComponent: "isAbout",
      pending: {
        type: "reviseIsAbout",
      },
      nextPending: {
        type: "strengthenReviseIsAbout",
      },
    },
    {
      label: "MI",
      frameComponent: "mainIdeas",
      pending: {
        type: "reviseMainIdea",
        index: 0,
      },
      nextPending: {
        type: "reviseMainIdea",
        index: 1,
      },
    },
    {
      label: "ED",
      frameComponent: "details",
      pending: {
        type: "collectAnotherDetail",
        index: 0,
      },
      nextPending: {
        type: "collectAnotherDetail",
        index: 1,
      },
    },
    {
      label: "SW",
      frameComponent: "soWhat",
      pending: {
        type: "continueSoWhat",
      },
      nextPending: {
        type: "strengthenCurrentSoWhat",
      },
    },
  ];

  for (const componentCase of componentCases) {
    const state = defaultState();

    state.frame.keyTopic =
      "Social Media and Teen Mental Health";

    state.frame.isAbout =
      "How social media can affect teen mental health.";

    state.frame.parentItems = [
      "Social media can increase anxiety and stress.",
      "Social media can affect self-esteem.",
    ];

    state.frame.details = [
      [
        "Comparing themselves to others can increase anxiety.",
      ],
      [
        "Negative feedback can affect how teens see themselves.",
      ],
    ];

    state.pending =
      structuredClone(
        componentCase.pending
      );

    const genuineStruggleContract =
      getInstructionalContract(
        componentCase.frameComponent,
        INSTRUCTIONAL_SITUATIONS
          .GENUINE_STRUGGLE
      );

    const noComponentEvidenceContract =
      getInstructionalContract(
        componentCase.frameComponent,
        INSTRUCTIONAL_SITUATIONS
          .NO_COMPONENT_EVIDENCE
      );

    const setGovernedSituation = (
      instructionalSituation,
      contract
    ) => {
      state.instructionalSituation = {
        instructionalSituation,
        frameComponent:
          componentCase.frameComponent,
      };

      state.instructionalContractSelection = {
        selectionStatus:
          "contractSelected",

        selectedContractId:
          contract?.contractId || null,

        selectedContract:
          contract,
      };
    };

    const instructionalFinding = {
      frameComponent:
        componentCase.frameComponent,

      componentEvidenceLevel:
        "none",

      componentCriteriaStatus:
        "notSatisfied",

      relationshipStatus:
        "undetermined",

      diagnosis:
        "noComponentEvidence",
    };

    const observedPendingStages = [];
    const observedExecutionStages = [];

    for (let attempt = 0; attempt < 4; attempt += 1) {
      setGovernedSituation(
        INSTRUCTIONAL_SITUATIONS
          .GENUINE_STRUGGLE,
        genuineStruggleContract
      );

      attachGovernedSupportToPending(
        state,
        "idk",
        {
          intent: "stuck",
          confidence: 1,
          source:
            `progressiveSupportSelfTest:${componentCase.label}`,
          instructionalFinding,
        }
      );

      observedPendingStages.push(
        state?.pending
          ?.progressiveSupportStage ??
        null
      );

      observedExecutionStages.push(
        state?.pending
          ?.instructionalActivation
          ?.execution
          ?.progressiveSupportStage ??
        null
      );
    }

    setGovernedSituation(
      INSTRUCTIONAL_SITUATIONS
        .NO_COMPONENT_EVIDENCE,
      noComponentEvidenceContract
    );

    attachGovernedSupportToPending(
      state,
      "idk",
      {
        intent: "stuck",
        confidence: 1,
        source:
          `progressiveSupportResetSelfTest:${componentCase.label}`,
        instructionalFinding,
      }
    );

    const stageCleared =
      !Object.prototype.hasOwnProperty.call(
        state.pending,
        "progressiveSupportStage"
      ) &&
      state?.pending
        ?.instructionalActivation
        ?.execution
        ?.progressiveSupportStage ===
        null;

    setGovernedSituation(
      INSTRUCTIONAL_SITUATIONS
        .GENUINE_STRUGGLE,
      genuineStruggleContract
    );

    attachGovernedSupportToPending(
      state,
      "idk",
      {
        intent: "stuck",
        confidence: 1,
        source:
          `progressiveSupportRestartSelfTest:${componentCase.label}`,
        instructionalFinding,
      }
    );

    const restartStage =
      state?.pending
        ?.progressiveSupportStage ??
      null;

    state.pending =
      structuredClone(
        componentCase.nextPending
      );

    setGovernedSituation(
      INSTRUCTIONAL_SITUATIONS
        .GENUINE_STRUGGLE,
      genuineStruggleContract
    );

    attachGovernedSupportToPending(
      state,
      "idk",
      {
        intent: "stuck",
        confidence: 1,
        source:
          `progressiveSupportLocationSelfTest:${componentCase.label}`,
        instructionalFinding,
      }
    );

    const newLocationStage =
      state?.pending
        ?.progressiveSupportStage ??
      null;

    const expectedStages = [
      1,
      2,
      3,
      3,
    ];

    const passed =
      JSON.stringify(
        observedPendingStages
      ) ===
        JSON.stringify(
          expectedStages
        ) &&

      JSON.stringify(
        observedExecutionStages
      ) ===
        JSON.stringify(
          expectedStages
        ) &&

      stageCleared === true &&
      restartStage === 1 &&
      newLocationStage === 1;

    results.push({
      name:
        `${componentCase.label} Progressive Support - Stage 1 -> 2 -> 3 lifecycle`,

      passed,

      expected: {
        pendingStages:
          expectedStages,

        executionStages:
          expectedStages,

        stageCleared:
          true,

        restartStage:
          1,

        newLocationStage:
          1,
      },

      actual: {
        pendingStages:
          observedPendingStages,

        executionStages:
          observedExecutionStages,

        stageCleared,

        restartStage,

        newLocationStage,
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

function formatProgressiveSupportSelfTestResults(
  testResults
) {
  const lines = [
    "🪜 KAW PROGRESSIVE SUPPORT SELF-TESTS",
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
      "🚀 Progressive Support lifecycle is operating correctly."
    );
  }

  return lines.join("\n");
}

// ======================================================
// REDIRECT NAVIGATION DETERMINISTIC SELF-TESTS
// ======================================================
//
// Verifies the deterministic redirect authority chain:
//
// interpretation artifact
// → deterministic validation
// → navigation preparation
// → atomic commit
//
// These tests do not call AI.
//
// They verify runtime authorization, target resolution,
// canonical Frame preservation, and location-owned
// artifact invalidation.
//
// ======================================================

function createRedirectNavigationTestState() {
  return {
    frame: {
      keyTopic:
        "Social media",

      isAbout:
        "How social media affects teenagers",

      parentItems: [
        "Mental health",
        "Relationships",
      ],

      details: [
        [
          "Social media can affect self-esteem.",
          "Comparison can affect confidence.",
        ],
        [
          "Social media changes how teens communicate.",
          "Online interactions can affect friendships.",
        ],
      ],

      soWhat:
        "Social media can shape both how teenagers feel and how they connect with others.",
    },

    frameMeta: {
      assignmentContext: {
        raw:
          "Explain how social media affects teenagers.",

        valid:
          true,

        confirmed:
          true,
      },
    },

    interactionMode:
      "build",

    pending: {
      type:
        "reviseDetailAt",

      index:
        1,

      detailIndex:
        0,

      captureMode:
        "revision",
    },

    observationReport: {
      stale:
        true,
    },

    instructionalAssessment: {
      stale:
        true,
    },

    componentInstructionalFinding: {
      stale:
        true,
    },

    instructionalSituation: {
      stale:
        true,
    },

    instructionalContractSelection: {
      stale:
        true,
    },

    progressionAuthorization: {
      stale:
        true,
    },
  };
}

function runRedirectNavigationSelfTests() {
  const results = [];

  // --------------------------------------------------
  // TEST 1 — EXISTING MAIN IDEA AUTHORIZES
  // --------------------------------------------------

  const mainIdeaState =
    createRedirectNavigationTestState();

  const mainIdeaSnapshot =
    structuredClone(
      mainIdeaState
    );

  const mainIdeaInterpretation = {
    artifactType:
      "redirectInterpretation",

    interpretationStatus:
      "redirectObserved",

    redirectIntent:
      "revisitTarget",

    requestedTarget: {
      component:
        "mainIdeas",

      mainIdeaReference:
        "ordinal1",

      detailReference:
        "unspecified",
    },

    requestedOperation:
      "workOn",

    currentPathDisposition:
      "unspecified",
  };

  const mainIdeaValidation =
    buildRedirectValidation(
      mainIdeaState,
      mainIdeaInterpretation
    );

  const mainIdeaPreparation =
    buildRedirectNavigationPreparation(
      mainIdeaState,
      mainIdeaInterpretation,
      mainIdeaValidation
    );

  const mainIdeaCommit =
    buildRedirectNavigationCommit(
      mainIdeaState,
      mainIdeaPreparation
    );

  const mainIdeaPassed =
    mainIdeaValidation
      ?.validationStatus ===
      "authorized" &&

    mainIdeaPreparation
      ?.preparationStatus ===
      "prepared" &&

    mainIdeaPreparation
      ?.verified === true &&

    mainIdeaCommit
      ?.commitStatus ===
      "committed" &&

    mainIdeaCommit
      ?.committed === true &&

    mainIdeaCommit
      ?.committedState
      ?.pending
      ?.type ===
      "reviseMainIdeaAt" &&

    mainIdeaCommit
      ?.committedState
      ?.pending
      ?.index ===
      0 &&

    JSON.stringify(
      mainIdeaState
    ) ===
    JSON.stringify(
      mainIdeaSnapshot
    );

  results.push({
    name:
      "Redirect - Existing Main Idea resolves, prepares, and commits atomically",

    passed:
      mainIdeaPassed,

    expected: {
      validationStatus:
        "authorized",

      preparationStatus:
        "prepared",

      commitStatus:
        "committed",

      pendingType:
        "reviseMainIdeaAt",

      index:
        0,

      sourceStateUnchanged:
        true,
    },

    actual: {
      validationStatus:
        mainIdeaValidation
          ?.validationStatus || null,

      preparationStatus:
        mainIdeaPreparation
          ?.preparationStatus || null,

      commitStatus:
        mainIdeaCommit
          ?.commitStatus || null,

      pendingType:
        mainIdeaCommit
          ?.committedState
          ?.pending
          ?.type || null,

      index:
        mainIdeaCommit
          ?.committedState
          ?.pending
          ?.index ?? null,

      sourceStateUnchanged:
        JSON.stringify(
          mainIdeaState
        ) ===
        JSON.stringify(
          mainIdeaSnapshot
        ),
    },
  });

  // --------------------------------------------------
  // TEST 2 — NONEXISTENT MAIN IDEA IS NOT AUTHORIZED
  // --------------------------------------------------

  const invalidState =
    createRedirectNavigationTestState();

  const invalidInterpretation = {
    artifactType:
      "redirectInterpretation",

    interpretationStatus:
      "redirectObserved",

    redirectIntent:
      "revisitTarget",

    requestedTarget: {
      component:
        "mainIdeas",

      mainIdeaReference:
        "ordinal5",

      detailReference:
        "unspecified",
    },

    requestedOperation:
      "workOn",

    currentPathDisposition:
      "unspecified",
  };

  const invalidValidation =
    buildRedirectValidation(
      invalidState,
      invalidInterpretation
    );

  const invalidPreparation =
    buildRedirectNavigationPreparation(
      invalidState,
      invalidInterpretation,
      invalidValidation
    );

  const invalidCommit =
    buildRedirectNavigationCommit(
      invalidState,
      invalidPreparation
    );

  const invalidPassed =
    invalidValidation
      ?.validationStatus ===
      "notAuthorized" &&

    invalidPreparation
      ?.preparationStatus ===
      "notApplicable" &&

    invalidCommit
      ?.commitStatus ===
      "notApplicable" &&

    invalidCommit
      ?.committed !== true;

  results.push({
    name:
      "Redirect - Nonexistent Main Idea cannot authorize navigation",

    passed:
      invalidPassed,

    expected: {
      validationStatus:
        "notAuthorized",

      preparationStatus:
        "notApplicable",

      commitStatus:
        "notApplicable",

      committed:
        false,
    },

    actual: {
      validationStatus:
        invalidValidation
          ?.validationStatus || null,

      preparationStatus:
        invalidPreparation
          ?.preparationStatus || null,

      commitStatus:
        invalidCommit
          ?.commitStatus || null,

      committed:
        invalidCommit
          ?.committed === true,
    },
  });

  // --------------------------------------------------
  // TEST 3 — ADD SUPPORTING DETAIL USES NEXT SLOT
  // --------------------------------------------------

  const detailState =
    createRedirectNavigationTestState();

  detailState.frame.details[0] = [
    "Social media can affect self-esteem.",
  ];

  const detailInterpretation = {
    artifactType:
      "redirectInterpretation",

    interpretationStatus:
      "redirectObserved",

    redirectIntent:
      "revisitTarget",

    requestedTarget: {
      component:
        "details",

      mainIdeaReference:
        "ordinal1",

      detailReference:
        "unspecified",
    },

    requestedOperation:
      "addSupportingContent",

    currentPathDisposition:
      "unspecified",
  };

  const detailValidation =
    buildRedirectValidation(
      detailState,
      detailInterpretation
    );

  const detailPreparation =
    buildRedirectNavigationPreparation(
      detailState,
      detailInterpretation,
      detailValidation
    );

  const detailPassed =
    detailValidation
      ?.validationStatus ===
      "authorized" &&

    detailPreparation
      ?.preparationStatus ===
      "prepared" &&

    detailPreparation
      ?.replacementPending
      ?.type ===
      "collectAnotherDetail" &&

    detailPreparation
      ?.replacementPending
      ?.index ===
      0 &&

    detailPreparation
      ?.replacementPending
      ?.detailIndex ===
      1 &&

    detailPreparation
      ?.replacementPending
      ?.captureMode ===
      "required";

  results.push({
    name:
      "Redirect - Add supporting content resolves to the next governed Detail slot",

    passed:
      detailPassed,

    expected: {
      validationStatus:
        "authorized",

      pendingType:
        "collectAnotherDetail",

      mainIdeaIndex:
        0,

      detailIndex:
        1,

      captureMode:
        "required",
    },

    actual: {
      validationStatus:
        detailValidation
          ?.validationStatus || null,

      pendingType:
        detailPreparation
          ?.replacementPending
          ?.type || null,

      mainIdeaIndex:
        detailPreparation
          ?.replacementPending
          ?.index ?? null,

      detailIndex:
        detailPreparation
          ?.replacementPending
          ?.detailIndex ?? null,

      captureMode:
        detailPreparation
          ?.replacementPending
          ?.captureMode || null,
    },
  });

  // --------------------------------------------------
  // TEST 4 — COMMIT PRESERVES FRAME, CLEARS OLD LOCATION
  // --------------------------------------------------

  const preservationState =
    createRedirectNavigationTestState();

  const preservationFrame =
    structuredClone(
      preservationState.frame
    );

  const preservationInterpretation = {
    artifactType:
      "redirectInterpretation",

    interpretationStatus:
      "redirectObserved",

    redirectIntent:
      "revisitTarget",

    requestedTarget: {
      component:
        "isAbout",

      mainIdeaReference:
        "unspecified",

      detailReference:
        "unspecified",
    },

    requestedOperation:
      "revise",

    currentPathDisposition:
      "unspecified",
  };

  const preservationValidation =
    buildRedirectValidation(
      preservationState,
      preservationInterpretation
    );

  const preservationPreparation =
    buildRedirectNavigationPreparation(
      preservationState,
      preservationInterpretation,
      preservationValidation
    );

  const preservationCommit =
    buildRedirectNavigationCommit(
      preservationState,
      preservationPreparation
    );

  const committedState =
    preservationCommit
      ?.committedState;

  const preservationPassed =
    preservationCommit
      ?.committed === true &&

    JSON.stringify(
      committedState?.frame
    ) ===
    JSON.stringify(
      preservationFrame
    ) &&

    committedState
      ?.pending
      ?.type ===
      "reviseIsAbout" &&

    committedState
      ?.observationReport ===
      undefined &&

    committedState
      ?.instructionalAssessment ===
      undefined &&

    committedState
      ?.componentInstructionalFinding ===
      undefined &&

    committedState
      ?.instructionalSituation ===
      undefined &&

    committedState
      ?.instructionalContractSelection ===
      undefined &&

    committedState
      ?.progressionAuthorization ===
      undefined;

  results.push({
    name:
      "Redirect - Atomic commit preserves canonical Frame and invalidates old location artifacts",

    passed:
      preservationPassed,

    expected: {
      canonicalFramePreserved:
        true,

      pendingType:
        "reviseIsAbout",

      staleLocationArtifactsCleared:
        true,
    },

    actual: {
      canonicalFramePreserved:
        JSON.stringify(
          committedState?.frame
        ) ===
        JSON.stringify(
          preservationFrame
        ),

      pendingType:
        committedState
          ?.pending
          ?.type || null,

      staleLocationArtifactsCleared:
        committedState
          ?.observationReport ===
          undefined &&
        committedState
          ?.instructionalAssessment ===
          undefined &&
        committedState
          ?.componentInstructionalFinding ===
          undefined &&
        committedState
          ?.instructionalSituation ===
          undefined &&
        committedState
          ?.instructionalContractSelection ===
          undefined &&
        committedState
          ?.progressionAuthorization ===
          undefined,
    },
  });

      // --------------------------------------------------
  // TEST 5 — SAME LOCATION PRESERVES GUIDED CONSTRUCTION
  // --------------------------------------------------

  const sameLocationState =
    createRedirectNavigationTestState();

  sameLocationState.pending = {
    type:
      "reviseMainIdeaAt",

    index:
      0,

    captureMode:
      "revision",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      2,

    guidedConstructionEvidence: {
      "1": {
        step:
          1,

        evidence:
          "Mental health",
      },
    },
  };

  sameLocationState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      sameLocationState
    );

  const sameLocationInterpretation = {
    artifactType:
      "redirectInterpretation",

    interpretationStatus:
      "redirectObserved",

    redirectIntent:
      "revisitTarget",

    requestedTarget: {
      component:
        "mainIdeas",

      mainIdeaReference:
        "ordinal1",

      detailReference:
        "unspecified",
    },

    requestedOperation:
      "workOn",

    currentPathDisposition:
      "unspecified",
  };

  const sameLocationValidation =
    buildRedirectValidation(
      sameLocationState,
      sameLocationInterpretation
    );

  const sameLocationPreparation =
    buildRedirectNavigationPreparation(
      sameLocationState,
      sameLocationInterpretation,
      sameLocationValidation
    );

  const sameLocationCommit =
    buildRedirectNavigationCommit(
      sameLocationState,
      sameLocationPreparation
    );

  const sameLocationPassed =
    sameLocationCommit
      ?.committed === true &&

    sameLocationCommit
      ?.commitStatus ===
      "sameLocation" &&

    sameLocationCommit
      ?.committedState
      ?.pending
      ?.progressiveSupportStage ===
      3 &&

    sameLocationCommit
      ?.committedState
      ?.pending
      ?.guidedConstructionStep ===
      2 &&

    sameLocationCommit
      ?.committedState
      ?.pending
      ?.guidedConstructionEvidence
      ?.[1]
      ?.evidence ===
      "Mental health";

  results.push({
    name:
      "Redirect - Same exact location preserves Progressive Support and Guided Construction",

    passed:
      sameLocationPassed,

    expected: {
      commitStatus:
        "sameLocation",

      progressiveSupportStage:
        3,

      guidedConstructionStep:
        2,

      guidedEvidencePreserved:
        true,
    },

    actual: {
      commitStatus:
        sameLocationCommit
          ?.commitStatus || null,

      progressiveSupportStage:
        sameLocationCommit
          ?.committedState
          ?.pending
          ?.progressiveSupportStage ?? null,

      guidedConstructionStep:
        sameLocationCommit
          ?.committedState
          ?.pending
          ?.guidedConstructionStep ?? null,

      guidedEvidencePreserved:
        sameLocationCommit
          ?.committedState
          ?.pending
          ?.guidedConstructionEvidence
          ?.[1]
          ?.evidence ===
        "Mental health",
    },
  });

  // --------------------------------------------------
  // TEST 6 — LOCATION CHANGE RESETS LOCATION-OWNED STATE
  // --------------------------------------------------

  const locationChangeState =
    createRedirectNavigationTestState();

  locationChangeState.pending = {
    type:
      "reviseMainIdeaAt",

    index:
      0,

    captureMode:
      "revision",

    progressiveSupportStage:
      3,

    guidedConstructionStep:
      2,

    guidedConstructionEvidence: {
      "1": {
        step:
          1,

        evidence:
          "Mental health",
      },
    },
  };

  locationChangeState
    .pending
    .guidedConstructionLocation =
    buildGuidedConstructionInstructionalLocation(
      locationChangeState
    );

  const locationChangeInterpretation = {
    artifactType:
      "redirectInterpretation",

    interpretationStatus:
      "redirectObserved",

    redirectIntent:
      "switchTarget",

    requestedTarget: {
      component:
        "mainIdeas",

      mainIdeaReference:
        "ordinal2",

      detailReference:
        "unspecified",
    },

    requestedOperation:
      "workOn",

    currentPathDisposition:
      "unspecified",
  };

  const locationChangeValidation =
    buildRedirectValidation(
      locationChangeState,
      locationChangeInterpretation
    );

  const locationChangePreparation =
    buildRedirectNavigationPreparation(
      locationChangeState,
      locationChangeInterpretation,
      locationChangeValidation
    );

  const locationChangeCommit =
    buildRedirectNavigationCommit(
      locationChangeState,
      locationChangePreparation
    );

  const locationChangePending =
    locationChangeCommit
      ?.committedState
      ?.pending;

  const locationChangePassed =
    locationChangeCommit
      ?.committed === true &&

    locationChangeCommit
      ?.commitStatus ===
      "committed" &&

    locationChangePending
      ?.type ===
      "reviseMainIdeaAt" &&

    locationChangePending
      ?.index ===
      1 &&

    locationChangePending
      ?.progressiveSupportStage ===
      undefined &&

    locationChangePending
      ?.guidedConstructionStep ===
      undefined &&

    locationChangePending
      ?.guidedConstructionEvidence ===
      undefined &&

    locationChangePending
      ?.guidedConstructionLocation ===
      undefined;

  results.push({
    name:
      "Redirect - Genuine location change clears prior Guided Construction state",

    passed:
      locationChangePassed,

    expected: {
      commitStatus:
        "committed",

      targetIndex:
        1,

      progressiveSupportCleared:
        true,

      guidedConstructionCleared:
        true,
    },

    actual: {
      commitStatus:
        locationChangeCommit
          ?.commitStatus || null,

      targetIndex:
        locationChangePending
          ?.index ?? null,

      progressiveSupportCleared:
        locationChangePending
          ?.progressiveSupportStage ===
        undefined,

      guidedConstructionCleared:
        locationChangePending
          ?.guidedConstructionStep ===
          undefined &&
        locationChangePending
          ?.guidedConstructionEvidence ===
          undefined &&
        locationChangePending
          ?.guidedConstructionLocation ===
          undefined,
    },
  });

// --------------------------------------------------
// TEST 7 — OPTIONAL MAIN IDEA DECLINE
// --------------------------------------------------

const mainIdeaDeclineState =
  createRedirectNavigationTestState();

mainIdeaDeclineState.pending = {
  type:
    "offerAnotherMainIdea",
};

const mainIdeaDeclineFrame =
  structuredClone(
    mainIdeaDeclineState.frame
  );

const mainIdeaDeclineSnapshot =
  structuredClone(
    mainIdeaDeclineState
  );

const mainIdeaDeclineInterpretation = {
  artifactType:
    "redirectInterpretation",

  interpretationStatus:
    "redirectObserved",

  redirectIntent:
    "leaveCurrentPath",

  requestedTarget: {
    component:
      "unspecified",

    mainIdeaReference:
      "unspecified",

    detailReference:
      "unspecified",
  },

  requestedOperation:
    "unspecified",

  currentPathDisposition:
    "decline",
};

const mainIdeaDeclineValidation =
  buildRedirectValidation(
    mainIdeaDeclineState,
    mainIdeaDeclineInterpretation
  );

const mainIdeaDeclinePreparation =
  buildRedirectNavigationPreparation(
    mainIdeaDeclineState,
    mainIdeaDeclineInterpretation,
    mainIdeaDeclineValidation
  );

const mainIdeaDeclineCommit =
  buildRedirectNavigationCommit(
    mainIdeaDeclineState,
    mainIdeaDeclinePreparation
  );

const mainIdeaDeclinePassed =
  mainIdeaDeclineValidation
    ?.validationStatus ===
    "authorized" &&

  mainIdeaDeclineValidation
    ?.resolvedTarget
    ?.operation ===
    "declineCurrentPath" &&

  mainIdeaDeclinePreparation
    ?.preparationStatus ===
    "prepared" &&

  mainIdeaDeclinePreparation
    ?.replacementPending
    ?.type ===
    "confirmMainIdeas" &&

  mainIdeaDeclineCommit
    ?.committed === true &&

  mainIdeaDeclineCommit
    ?.commitStatus ===
    "committed" &&

  mainIdeaDeclineCommit
    ?.committedState
    ?.pending
    ?.type ===
    "confirmMainIdeas" &&

  JSON.stringify(
    mainIdeaDeclineCommit
      ?.committedState
      ?.frame
  ) ===
  JSON.stringify(
    mainIdeaDeclineFrame
  ) &&

  JSON.stringify(
    mainIdeaDeclineState
  ) ===
  JSON.stringify(
    mainIdeaDeclineSnapshot
  );

results.push({
  name:
    "Redirect - Optional Main Idea decline is authorized and returns to Main Idea confirmation",

  passed:
    mainIdeaDeclinePassed,

  expected: {
    validationStatus:
      "authorized",

    operation:
      "declineCurrentPath",

    pendingType:
      "confirmMainIdeas",

    canonicalFramePreserved:
      true,

    sourceStateUnchanged:
      true,
  },

  actual: {
    validationStatus:
      mainIdeaDeclineValidation
        ?.validationStatus || null,

    operation:
      mainIdeaDeclineValidation
        ?.resolvedTarget
        ?.operation || null,

    pendingType:
      mainIdeaDeclineCommit
        ?.committedState
        ?.pending
        ?.type || null,

    canonicalFramePreserved:
      JSON.stringify(
        mainIdeaDeclineCommit
          ?.committedState
          ?.frame
      ) ===
      JSON.stringify(
        mainIdeaDeclineFrame
      ),

    sourceStateUnchanged:
      JSON.stringify(
        mainIdeaDeclineState
      ) ===
      JSON.stringify(
        mainIdeaDeclineSnapshot
      ),
  },
});

// --------------------------------------------------
// TEST 8 — OPTIONAL ESSENTIAL DETAIL DECLINE
// --------------------------------------------------

const detailDeclineState =
  createRedirectNavigationTestState();

detailDeclineState.pending = {
  type:
    "offerAnotherDetail",

  index:
    0,
};

const detailDeclineFrame =
  structuredClone(
    detailDeclineState.frame
  );

const detailDeclineSnapshot =
  structuredClone(
    detailDeclineState
  );

const detailDeclineInterpretation = {
  artifactType:
    "redirectInterpretation",

  interpretationStatus:
    "redirectObserved",

  redirectIntent:
    "leaveCurrentPath",

  requestedTarget: {
    component:
      "unspecified",

    mainIdeaReference:
      "unspecified",

    detailReference:
      "unspecified",
  },

  requestedOperation:
    "unspecified",

  currentPathDisposition:
    "decline",
};

const detailDeclineValidation =
  buildRedirectValidation(
    detailDeclineState,
    detailDeclineInterpretation
  );

const detailDeclinePreparation =
  buildRedirectNavigationPreparation(
    detailDeclineState,
    detailDeclineInterpretation,
    detailDeclineValidation
  );

const detailDeclineCommit =
  buildRedirectNavigationCommit(
    detailDeclineState,
    detailDeclinePreparation
  );

const detailDeclinePassed =
  detailDeclineValidation
    ?.validationStatus ===
    "authorized" &&

  detailDeclineValidation
    ?.resolvedTarget
    ?.operation ===
    "declineCurrentPath" &&

  detailDeclineValidation
    ?.resolvedTarget
    ?.mainIdeaIndex ===
    0 &&

  detailDeclinePreparation
    ?.preparationStatus ===
    "prepared" &&

  detailDeclinePreparation
    ?.replacementPending
    ?.type ===
    "confirmDetails" &&

  detailDeclinePreparation
    ?.replacementPending
    ?.index ===
    0 &&

  detailDeclineCommit
    ?.committed === true &&

  detailDeclineCommit
    ?.commitStatus ===
    "committed" &&

  detailDeclineCommit
    ?.committedState
    ?.pending
    ?.type ===
    "confirmDetails" &&

  detailDeclineCommit
    ?.committedState
    ?.pending
    ?.index ===
    0 &&

  JSON.stringify(
    detailDeclineCommit
      ?.committedState
      ?.frame
  ) ===
  JSON.stringify(
    detailDeclineFrame
  ) &&

  JSON.stringify(
    detailDeclineState
  ) ===
  JSON.stringify(
    detailDeclineSnapshot
  );

results.push({
  name:
    "Redirect - Optional Essential Detail decline is authorized and returns to Detail confirmation",

  passed:
    detailDeclinePassed,

  expected: {
    validationStatus:
      "authorized",

    operation:
      "declineCurrentPath",

    pendingType:
      "confirmDetails",

    mainIdeaIndex:
      0,

    canonicalFramePreserved:
      true,

    sourceStateUnchanged:
      true,
  },

  actual: {
    validationStatus:
      detailDeclineValidation
        ?.validationStatus || null,

    operation:
      detailDeclineValidation
        ?.resolvedTarget
        ?.operation || null,

    pendingType:
      detailDeclineCommit
        ?.committedState
        ?.pending
        ?.type || null,

    mainIdeaIndex:
      detailDeclineCommit
        ?.committedState
        ?.pending
        ?.index ?? null,

    canonicalFramePreserved:
      JSON.stringify(
        detailDeclineCommit
          ?.committedState
          ?.frame
      ) ===
      JSON.stringify(
        detailDeclineFrame
      ),

    sourceStateUnchanged:
      JSON.stringify(
        detailDeclineState
      ) ===
      JSON.stringify(
        detailDeclineSnapshot
      ),
  },
});

// --------------------------------------------------
// TEST 9 — REQUIRED PATH DECLINE IS NOT AUTHORIZED
// --------------------------------------------------

const requiredDeclineState =
  createRedirectNavigationTestState();

requiredDeclineState.frame.details[0] = [
  "Social media can affect self-esteem.",
];

requiredDeclineState.pending = {
  type:
    "collectAnotherDetail",

  index:
    0,

  detailIndex:
    1,

  captureMode:
    "required",
};

const requiredDeclineSnapshot =
  structuredClone(
    requiredDeclineState
  );

const requiredDeclineInterpretation = {
  artifactType:
    "redirectInterpretation",

  interpretationStatus:
    "redirectObserved",

  redirectIntent:
    "leaveCurrentPath",

  requestedTarget: {
    component:
      "unspecified",

    mainIdeaReference:
      "unspecified",

    detailReference:
      "unspecified",
  },

  requestedOperation:
    "unspecified",

  currentPathDisposition:
    "decline",
};

const requiredDeclineValidation =
  buildRedirectValidation(
    requiredDeclineState,
    requiredDeclineInterpretation
  );

const requiredDeclinePreparation =
  buildRedirectNavigationPreparation(
    requiredDeclineState,
    requiredDeclineInterpretation,
    requiredDeclineValidation
  );

const requiredDeclinePassed =
  requiredDeclineValidation
    ?.validationStatus ===
    "notAuthorized" &&

  requiredDeclineValidation
    ?.navigationAuthorized ===
    false &&

  requiredDeclineValidation
    ?.currentPathDispositionValidation
    ?.declineRequested ===
    true &&

  requiredDeclineValidation
    ?.currentPathDispositionValidation
    ?.declineAuthorized ===
    false &&

  requiredDeclinePreparation
    ?.preparationStatus !==
    "prepared" &&

  JSON.stringify(
    requiredDeclineState
  ) ===
  JSON.stringify(
    requiredDeclineSnapshot
  );

results.push({
  name:
    "Redirect - Required current path decline is blocked without changing state",

  passed:
    requiredDeclinePassed,

  expected: {
    validationStatus:
      "notAuthorized",

    navigationAuthorized:
      false,

    declineAuthorized:
      false,

    preparationStatus:
      "notApplicable",

    sourceStateUnchanged:
      true,
  },

  actual: {
    validationStatus:
      requiredDeclineValidation
        ?.validationStatus || null,

    navigationAuthorized:
      requiredDeclineValidation
        ?.navigationAuthorized === true,

    declineAuthorized:
      requiredDeclineValidation
        ?.currentPathDispositionValidation
        ?.declineAuthorized === true,

    preparationStatus:
      requiredDeclinePreparation
        ?.preparationStatus || null,

    sourceStateUnchanged:
      JSON.stringify(
        requiredDeclineState
      ) ===
      JSON.stringify(
        requiredDeclineSnapshot
      ),
  },
});

// --------------------------------------------------
// TEST 10 — OPTIONAL DETAIL COLLECTION AFTER 2 DETAILS
// --------------------------------------------------

const optionalCollectedDetailDeclineState =
  createRedirectNavigationTestState();

optionalCollectedDetailDeclineState.frame.details[0] = [
  "Social media can affect self-esteem.",
  "Online interactions can influence relationships.",
];

optionalCollectedDetailDeclineState.pending = {
  type:
    "collectAnotherDetail",

  index:
    0,

  detailIndex:
    2,

  captureMode:
    "optional",
};

const optionalCollectedDetailDeclineFrame =
  structuredClone(
    optionalCollectedDetailDeclineState.frame
  );

const optionalCollectedDetailDeclineSnapshot =
  structuredClone(
    optionalCollectedDetailDeclineState
  );

const optionalCollectedDetailDeclineInterpretation = {
  artifactType:
    "redirectInterpretation",

  interpretationStatus:
    "redirectObserved",

  redirectIntent:
    "leaveCurrentPath",

  requestedTarget: {
    component:
      "unspecified",

    mainIdeaReference:
      "unspecified",

    detailReference:
      "unspecified",
  },

  requestedOperation:
    "unspecified",

  currentPathDisposition:
    "decline",
};

const optionalCollectedDetailDeclineValidation =
  buildRedirectValidation(
    optionalCollectedDetailDeclineState,
    optionalCollectedDetailDeclineInterpretation
  );

const optionalCollectedDetailDeclinePreparation =
  buildRedirectNavigationPreparation(
    optionalCollectedDetailDeclineState,
    optionalCollectedDetailDeclineInterpretation,
    optionalCollectedDetailDeclineValidation
  );

const optionalCollectedDetailDeclineCommit =
  buildRedirectNavigationCommit(
    optionalCollectedDetailDeclineState,
    optionalCollectedDetailDeclinePreparation
  );

const optionalCollectedDetailDeclinePassed =
  optionalCollectedDetailDeclineValidation
    ?.validationStatus ===
    "authorized" &&

  optionalCollectedDetailDeclineValidation
    ?.currentPathDispositionValidation
    ?.declineAuthorized ===
    true &&

  optionalCollectedDetailDeclineValidation
    ?.resolvedTarget
    ?.operation ===
    "declineCurrentPath" &&

  optionalCollectedDetailDeclinePreparation
    ?.preparationStatus ===
    "prepared" &&

  optionalCollectedDetailDeclinePreparation
    ?.replacementPending
    ?.type ===
    "confirmDetails" &&

  optionalCollectedDetailDeclinePreparation
    ?.replacementPending
    ?.index ===
    0 &&

  optionalCollectedDetailDeclineCommit
    ?.committed === true &&

  optionalCollectedDetailDeclineCommit
    ?.commitStatus ===
    "committed" &&

  optionalCollectedDetailDeclineCommit
    ?.committedState
    ?.pending
    ?.type ===
    "confirmDetails" &&

  optionalCollectedDetailDeclineCommit
    ?.committedState
    ?.pending
    ?.index ===
    0 &&

  JSON.stringify(
    optionalCollectedDetailDeclineCommit
      ?.committedState
      ?.frame
  ) ===
  JSON.stringify(
    optionalCollectedDetailDeclineFrame
  ) &&

  JSON.stringify(
    optionalCollectedDetailDeclineState
  ) ===
  JSON.stringify(
    optionalCollectedDetailDeclineSnapshot
  );

results.push({
  name:
    "Redirect - Optional Detail collection after two details may be declined",

  passed:
    optionalCollectedDetailDeclinePassed,

  expected: {
    validationStatus:
      "authorized",

    declineAuthorized:
      true,

    operation:
      "declineCurrentPath",

    pendingType:
      "confirmDetails",

    mainIdeaIndex:
      0,

    canonicalFramePreserved:
      true,

    sourceStateUnchanged:
      true,
  },

  actual: {
    validationStatus:
      optionalCollectedDetailDeclineValidation
        ?.validationStatus || null,

    declineAuthorized:
      optionalCollectedDetailDeclineValidation
        ?.currentPathDispositionValidation
        ?.declineAuthorized === true,

    operation:
      optionalCollectedDetailDeclineValidation
        ?.resolvedTarget
        ?.operation || null,

    pendingType:
      optionalCollectedDetailDeclineCommit
        ?.committedState
        ?.pending
        ?.type || null,

    mainIdeaIndex:
      optionalCollectedDetailDeclineCommit
        ?.committedState
        ?.pending
        ?.index ?? null,

    canonicalFramePreserved:
      JSON.stringify(
        optionalCollectedDetailDeclineCommit
          ?.committedState
          ?.frame
      ) ===
      JSON.stringify(
        optionalCollectedDetailDeclineFrame
      ),

    sourceStateUnchanged:
      JSON.stringify(
        optionalCollectedDetailDeclineState
      ) ===
      JSON.stringify(
        optionalCollectedDetailDeclineSnapshot
      ),
  },
});

// --------------------------------------------------
// TEST 11 — DECLINE PLUS EXPLICIT TARGET
// --------------------------------------------------

const declineWithTargetState =
  createRedirectNavigationTestState();

declineWithTargetState.pending = {
  type:
    "offerAnotherDetail",

  index:
    0,
};

const declineWithTargetSnapshot =
  structuredClone(
    declineWithTargetState
  );

const declineWithTargetInterpretation = {
  artifactType:
    "redirectInterpretation",

  interpretationStatus:
    "redirectObserved",

  redirectIntent:
    "revisitTarget",

  requestedTarget: {
    component:
      "mainIdeas",

    mainIdeaReference:
      "ordinal1",

    detailReference:
      "unspecified",
  },

  requestedOperation:
    "workOn",

  currentPathDisposition:
    "decline",
};

const declineWithTargetValidation =
  buildRedirectValidation(
    declineWithTargetState,
    declineWithTargetInterpretation
  );

const declineWithTargetPreparation =
  buildRedirectNavigationPreparation(
    declineWithTargetState,
    declineWithTargetInterpretation,
    declineWithTargetValidation
  );

const declineWithTargetCommit =
  buildRedirectNavigationCommit(
    declineWithTargetState,
    declineWithTargetPreparation
  );

const declineWithTargetPassed =
  declineWithTargetValidation
    ?.validationStatus ===
    "authorized" &&

  declineWithTargetValidation
    ?.resolvedTarget
    ?.component ===
    "mainIdeas" &&

  declineWithTargetValidation
    ?.resolvedTarget
    ?.mainIdeaIndex ===
    0 &&

  declineWithTargetPreparation
    ?.preparationStatus ===
    "prepared" &&

  declineWithTargetPreparation
    ?.replacementPending
    ?.type ===
    "reviseMainIdeaAt" &&

  declineWithTargetPreparation
    ?.replacementPending
    ?.index ===
    0 &&

  declineWithTargetCommit
    ?.committed ===
    true &&

  declineWithTargetCommit
    ?.committedState
    ?.pending
    ?.type ===
    "reviseMainIdeaAt" &&

  declineWithTargetCommit
    ?.committedState
    ?.pending
    ?.index ===
    0 &&

  JSON.stringify(
    declineWithTargetState
  ) ===
  JSON.stringify(
    declineWithTargetSnapshot
  );

results.push({
  name:
    "Redirect - Explicit target remains authoritative when student also declines current optional path",

  passed:
    declineWithTargetPassed,

  expected: {
    validationStatus:
      "authorized",

    component:
      "mainIdeas",

    mainIdeaIndex:
      0,

    pendingType:
      "reviseMainIdeaAt",

    sourceStateUnchanged:
      true,
  },

  actual: {
    validationStatus:
      declineWithTargetValidation
        ?.validationStatus || null,

    operation:
      declineWithTargetValidation
        ?.resolvedTarget
        ?.operation || null,

    component:
      declineWithTargetValidation
        ?.resolvedTarget
        ?.component || null,

    mainIdeaIndex:
      declineWithTargetValidation
        ?.resolvedTarget
        ?.mainIdeaIndex ?? null,

    pendingType:
      declineWithTargetCommit
        ?.committedState
        ?.pending
        ?.type || null,

    sourceStateUnchanged:
      JSON.stringify(
        declineWithTargetState
      ) ===
      JSON.stringify(
        declineWithTargetSnapshot
      ),
  },
});

// --------------------------------------------------
// TEST 12 — RAW PENDING MATCH IS NOT ENOUGH FOR ACTIVE GC
// --------------------------------------------------

const canonicalGcMismatchState =
  createRedirectNavigationTestState();

canonicalGcMismatchState.pending = {
  type:
    "reviseDetailAt",

  index:
    1,

  detailIndex:
    0,

  captureMode:
    "revision",

  progressiveSupportStage:
    3,

  guidedConstructionStep:
    2,

  guidedConstructionEvidence: {
    1: {
      step:
        1,

      evidence:
        "Online interactions can affect friendships.",
    },
  },

  guidedConstructionLocation: {
    locationEstablished:
      true,

    interactionMode:
      "strengthen",

    frameComponent:
      "details",

    rawStage:
      "details",

    pendingType:
      "reviseDetailAt",

    captureMode:
      "revision",

    mainIdeaIndex:
      null,

    detailMainIdeaIndex:
      1,

    detailIndex:
      0,
  },
};

const canonicalGcMismatchSnapshot =
  structuredClone(
    canonicalGcMismatchState
  );

const canonicalGcMismatchPreparation = {
  artifactType:
    "redirectNavigationPreparation",

  preparationStatus:
    "prepared",

  verified:
    true,

  replacementPending: {
    type:
      "reviseDetailAt",

    index:
      1,

    detailIndex:
      0,

    captureMode:
      "revision",
  },

  resolvedTarget: {
    component:
      "details",

    mainIdeaIndex:
      1,

    detailIndex:
      0,

    operation:
      "workOn",
  },
};

const canonicalGcMismatchCommit =
  buildRedirectNavigationCommit(
    canonicalGcMismatchState,
    canonicalGcMismatchPreparation
  );

const canonicalGcMismatchPending =
  canonicalGcMismatchCommit
    ?.committedState
    ?.pending;

const canonicalGcMismatchPassed =
  canonicalGcMismatchCommit
    ?.committed === true &&

  canonicalGcMismatchCommit
    ?.commitStatus ===
    "committed" &&

  canonicalGcMismatchPending
    ?.type ===
    "reviseDetailAt" &&

  canonicalGcMismatchPending
    ?.index ===
    1 &&

  canonicalGcMismatchPending
    ?.detailIndex ===
    0 &&

  canonicalGcMismatchPending
    ?.progressiveSupportStage ===
    undefined &&

  canonicalGcMismatchPending
    ?.guidedConstructionStep ===
    undefined &&

  canonicalGcMismatchPending
    ?.guidedConstructionEvidence ===
    undefined &&

  canonicalGcMismatchPending
    ?.guidedConstructionLocation ===
    undefined &&

  JSON.stringify(
    canonicalGcMismatchState
  ) ===
  JSON.stringify(
    canonicalGcMismatchSnapshot
  );

results.push({
  name:
    "Redirect - Active Guided Construction requires canonical location match, not raw pending match",

  passed:
    canonicalGcMismatchPassed,

  expected: {
    commitStatus:
      "committed",

    rawPendingStillMatches:
      true,

    guidedConstructionCleared:
      true,

    sourceStateUnchanged:
      true,
  },

  actual: {
    commitStatus:
      canonicalGcMismatchCommit
        ?.commitStatus || null,

    rawPendingStillMatches:
      canonicalGcMismatchPending
        ?.type ===
        "reviseDetailAt" &&
      canonicalGcMismatchPending
        ?.index ===
        1 &&
      canonicalGcMismatchPending
        ?.detailIndex ===
        0,

    guidedConstructionCleared:
      canonicalGcMismatchPending
        ?.progressiveSupportStage ===
        undefined &&
      canonicalGcMismatchPending
        ?.guidedConstructionStep ===
        undefined &&
      canonicalGcMismatchPending
        ?.guidedConstructionEvidence ===
        undefined &&
      canonicalGcMismatchPending
        ?.guidedConstructionLocation ===
        undefined,

    sourceStateUnchanged:
      JSON.stringify(
        canonicalGcMismatchState
      ) ===
      JSON.stringify(
        canonicalGcMismatchSnapshot
      ),
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

function formatRedirectNavigationSelfTestResults(
  testResults
) {
  const lines = [
    "🧭 KAW REDIRECT NAVIGATION SELF-TESTS",
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
      "🚀 Redirect deterministic navigation is operating correctly."
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
    id:
      "redirectNavigation",

    label:
      "Redirect Navigation",

    run:
      runRedirectNavigationSelfTests,

    format:
      formatRedirectNavigationSelfTestResults,
  },
  {
    id: "evidenceState",
    label: "Evidence State",
    run: runEvidenceStateSelfTests,
    format: formatEvidenceStateSelfTestResults,
  },
      {
    id: "progressiveSupport",
    label: "Progressive Support Lifecycle",
    run: runProgressiveSupportSelfTests,
    format: formatProgressiveSupportSelfTestResults,
  },
  {
    id: "essentialDetail",
    label: "Essential Detail Validation",
    run: runEssentialDetailSelfTests,
    format: formatEssentialDetailSelfTestResults,
  },
  {
  id: "stage1FinalQuestion",
  label: "Stage 1 Final Question Validation",
  run: runStage1FinalQuestionValidationSelfTests,
  format: formatStage1FinalQuestionValidationSelfTestResults,
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
// Allows one registered suite to run independently
// without executing the full /run tests command.
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
// PARENT ANCHOR STRUCTURAL INTERPRETATION
// Provides read-only structural context for the current
// instructional moment.
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
 *   current pending flow, including overlay and saved-resume cases
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
 * This helper provides normalized Parent Anchor context for
 * downstream evidence, structural interpretation, and
* development verification.
* It does not change runtime behavior or progression.
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

const OBVIOUS_STUDENT_SPELLING_CORRECTIONS =
  Object.freeze({
    movimng:
      "moving",
  });

function correctObviousStudentSpelling(s) {
  const text =
    cleanText(s);

  return text.replace(
    /\b[A-Za-z]+\b/g,
    (word) => {
      const correction =
        OBVIOUS_STUDENT_SPELLING_CORRECTIONS[
          word.toLowerCase()
        ];

      if (!correction) {
        return word;
      }

      if (
        word ===
        word.toUpperCase()
      ) {
        return correction.toUpperCase();
      }

      if (
        word.charAt(0) ===
        word.charAt(0).toUpperCase()
      ) {
        return (
          correction.charAt(0).toUpperCase() +
          correction.slice(1)
        );
      }

      return correction;
    }
  );
}

function cleanFrameText(s) {
  let text = cleanText(s);

  // Apply bounded deterministic spelling correction
  // without changing student meaning or phrasing.
  text =
    correctObviousStudentSpelling(
      text
  );
  
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

// ======================================================
// GUIDED CONSTRUCTION ACTIVE PATHWAY AUTHORITY
// ======================================================
//
// Determines whether an already-entered Guided
// Construction pathway remains authoritative at the
// student's exact current instructional location.
//
// Genuine Struggle governs entry into Progressive Support
// Stage 3.
//
// Once Stage 3 Guided Construction is active at the same
// exact instructional location, Guided Construction owns
// continuation until:
//
// • the normal component validator accepts the component;
// • runtime progression moves to a different location; or
// • Guided Construction reaches its defined endpoint.
//
// A later componentNeedsRevision or
// relationshipNeedsRepair finding may describe the
// student's current evidence accurately, but it does not
// replace the active Guided Construction pathway.
//
// This helper is read-only.
//
// ======================================================

function buildGuidedConstructionPathwayAuthority(
  state
) {
  const safeState =
    state &&
    typeof state === "object"
      ? state
      : {};

  const pending =
    safeState?.pending &&
    typeof safeState.pending === "object"
      ? safeState.pending
      : null;

  if (!pending) {
    return {
      authoritative:
        false,

      reason:
        "pendingStateUnavailable",

      context:
        null,

      storedLocation:
        null,

      currentLocation:
        null,

      instructionalContract:
        null,
    };
  }

  const activeContext =
    getActiveGuidedConstructionContext(
      safeState
    );

  if (
    activeContext?.active !== true
  ) {
    return {
      authoritative:
        false,

      reason:
        "guidedConstructionNotActive",

      context:
        activeContext,

      storedLocation:
        null,

      currentLocation:
        null,

      instructionalContract:
        null,
    };
  }

  const storedLocation =
    pending
      ?.guidedConstructionLocation &&
    typeof pending
      .guidedConstructionLocation ===
      "object"
      ? structuredClone(
          pending.guidedConstructionLocation
        )
      : null;

  const currentLocation =
    buildGuidedConstructionInstructionalLocation(
      safeState
    );

  const sameInstructionalLocation =
    storedLocation
      ?.locationEstablished === true &&
    currentLocation
      ?.locationEstablished === true &&
    isSameGuidedConstructionInstructionalLocation(
      storedLocation,
      currentLocation
    );

  if (!sameInstructionalLocation) {
    return {
      authoritative:
        false,

      reason:
        "guidedConstructionLocationChanged",

      context:
        activeContext,

      storedLocation,

      currentLocation,

      instructionalContract:
        null,
    };
  }

  const instructionalContract =
    getInstructionalContract(
      activeContext.frameComponent,
      INSTRUCTIONAL_SITUATIONS
        .GENUINE_STRUGGLE
    );

  return {
    authoritative:
      instructionalContract !== null,

    reason:
      instructionalContract
        ? "activeGuidedConstructionOwnsContinuation"
        : "guidedConstructionContractUnavailable",

    context:
      activeContext,

    storedLocation,

    currentLocation,

    instructionalContract:
      instructionalContract
        ? structuredClone(
            instructionalContract
          )
        : null,
  };
}

// ------------------------------------------------------
// GOVERNED INSTRUCTIONAL SUPPORT ATTACHMENT
// ------------------------------------------------------
function attachGovernedSupportToPending(
  state,
  message,
  intentResult = {}
) {

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

  const selectedInstructionalContract =
    state?.instructionalContractSelection
      ?.selectedContract ||
    null;

  // --------------------------------------------------
  // ACTIVE PATHWAY AUTHORITY
  //
  // Genuine Struggle determines entry into Guided
  // Construction.
  //
  // Once Guided Construction is active at the same exact
  // instructional location, the Guided Construction
  // pathway remains authoritative even when the current
  // component finding would ordinarily select a revision
  // or relationship-repair contract.
  //
  // --------------------------------------------------

  const guidedConstructionAuthority =
    buildGuidedConstructionPathwayAuthority(
      state
    );

  const guidedConstructionContinuationActive =
    guidedConstructionAuthority
      ?.authoritative === true;

  const instructionalContract =
    guidedConstructionContinuationActive
      ? guidedConstructionAuthority
          .instructionalContract
      : selectedInstructionalContract;

  if (
    !instructionalSituation ||
    !instructionalContract
  ) {
    throw new Error(
      "Governed support requires an established Instructional Situation and authoritative Instructional Contract."
    );
  }

  // --------------------------------------------------
  // PROGRESSIVE SUPPORT STAGE LIFECYCLE
  // --------------------------------------------------
  //
  // BEFORE GUIDED CONSTRUCTION:
  //
  // Genuine Struggle advances:
  //
  // none → Stage 1 → Stage 2 → Stage 3
  //
  // AFTER GUIDED CONSTRUCTION ENTRY:
  //
  // Progressive Support Stage remains exactly 3.
  //
  // Guided Construction progression is controlled only
  // by the Guided Construction runtime.
  //
  // A componentNeedsRevision or relationshipNeedsRepair
  // finding does not reset or replace an active Guided
  // Construction pathway at the same exact location.
  //
  // --------------------------------------------------

  const genuineStruggleActive =
    instructionalSituation
      ?.instructionalSituation ===
      INSTRUCTIONAL_SITUATIONS
        .GENUINE_STRUGGLE;

  const previousProgressiveSupportStage =
    Number(
      currentPending
        ?.progressiveSupportStage ??
      currentPending
        ?.supportLevel ??
      0
    );

  const progressiveSupportStage =
    guidedConstructionContinuationActive
      ? 3
      : genuineStruggleActive
        ? Math.min(
            Math.max(
              Number.isFinite(
                previousProgressiveSupportStage
              )
                ? previousProgressiveSupportStage + 1
                : 1,
              1
            ),
            3
          )
        : null;

  const pendingForActivation = {
    ...currentPending,

    instructionalFinding,
  };

  const additionalSupportEndpoint =
  buildGuidedConstructionAdditionalSupportEndpoint(
    {
      ...state,

      pending:
        pendingForActivation,
    }
  );

  // Retire the historical numeric supportLevel field
  // whenever governed support is rewritten.
  delete pendingForActivation.supportLevel;

  // --------------------------------------------------
  // ACTIVE GUIDED CONSTRUCTION CONTINUATION
  //
  // Preserve the existing Stage-3 pathway exactly.
  //
  // Do not increment, reset, or reinterpret its current
  // Guided Construction step here.
  // --------------------------------------------------

  if (
    guidedConstructionContinuationActive
  ) {
    pendingForActivation
      .progressiveSupportStage = 3;

    if (
      !Number.isInteger(
        pendingForActivation
          ?.guidedConstructionStep
      ) ||
      pendingForActivation
        .guidedConstructionStep < 1 ||
      pendingForActivation
        .guidedConstructionStep > 3
    ) {
      throw new Error(
        "Active Guided Construction requires a valid Guided Construction step."
      );
    }

    pendingForActivation
      .guidedConstructionLocation =
      structuredClone(
        guidedConstructionAuthority
          .storedLocation
      );
  }

  // --------------------------------------------------
  // ORDINARY PROGRESSIVE SUPPORT ENTRY
  //
  // Genuine Struggle owns the pathway only until Stage 3
  // has been entered.
  // --------------------------------------------------

  else if (genuineStruggleActive) {
    pendingForActivation
      .progressiveSupportStage =
      progressiveSupportStage;

    if (
      progressiveSupportStage === 3
    ) {
      if (
        !Number.isInteger(
          pendingForActivation
            ?.guidedConstructionStep
        )
      ) {
        pendingForActivation
          .guidedConstructionStep = 1;
      }

      // A newly entered Guided Construction pathway must
      // begin with fresh Guided Construction-owned state.
      delete pendingForActivation
        .guidedConstructionEvidence;

      delete pendingForActivation
        .guidedConstructionFinalRephraseUsed;

      delete pendingForActivation
        .guidedConstructionAdditionalSupportEndpoint;
      
      const guidedConstructionEntryState = {
        ...state,

        pending:
          pendingForActivation,
      };

      const guidedConstructionLocation =
        buildGuidedConstructionInstructionalLocation(
          guidedConstructionEntryState
        );

      if (
        guidedConstructionLocation
          ?.locationEstablished !== true
      ) {
        throw new Error(
          "Guided Construction entry requires an established instructional location."
        );
      }

      pendingForActivation
        .guidedConstructionLocation =
        structuredClone(
          guidedConstructionLocation
        );
    } else {
      delete pendingForActivation
        .guidedConstructionStep;

      delete pendingForActivation
        .guidedConstructionEvidence;

      delete pendingForActivation
        .guidedConstructionAdditionalSupportEndpoint;
      
      delete pendingForActivation
        .guidedConstructionFinalRephraseUsed;

      delete pendingForActivation
        .guidedConstructionLocation;
    }
  }

  // --------------------------------------------------
  // ORDINARY NON-GUIDED PATHWAY
  //
  // Outside active Guided Construction, a non-Genuine
  // Struggle situation clears Progressive Support and all
  // Guided Construction-owned metadata.
  // --------------------------------------------------

  else {
    delete pendingForActivation
      .progressiveSupportStage;

    delete pendingForActivation
      .guidedConstructionStep;

    delete pendingForActivation
      .guidedConstructionEvidence;

    delete pendingForActivation
      .guidedConstructionFinalRephraseUsed;

    delete pendingForActivation
      .guidedConstructionAdditionalSupportEndpoint;
    
    delete pendingForActivation
      .guidedConstructionLocation;
  }

if (
  additionalSupportEndpoint
    ?.endpointStatus ===
    "established"
) {
  pendingForActivation
    .guidedConstructionAdditionalSupportEndpointArtifact =
    structuredClone(
      additionalSupportEndpoint
    );
} else {
  delete pendingForActivation
    .guidedConstructionAdditionalSupportEndpointArtifact;
}
  
  // --------------------------------------------------
  // CONTRACT ACTIVATION
  //
  // Active Guided Construction uses the component's
  // Genuine-Struggle contract because that contract owns
  // Progressive Support Stage 3.
  //
  // The current Instructional Finding remains available
  // as evidence, but it does not replace the active
  // pathway's instructional authority.
  // --------------------------------------------------

  const activationState = {
    ...state,

    pending:
      pendingForActivation,
  };

  const instructionalActivation =
    activateInstructionalContract(
      instructionalContract,
      activationState
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
    ...pendingForActivation,

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
  // This diagnosis records the first unestablished
  // Assignment Understanding gate.
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
// Parent Anchor stages map onto these through the
// Parent Anchor Bridge, while this engine remains the
// deterministic source of truth for Frame progression.

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

// PARENT ANCHOR ARCHITECTURAL BOUNDARY
// ------------------------------------
// Parent Anchor provides the structural interpretation of
// the Framing Routine without becoming a competing runtime
// progression controller.
//
// This layer is read-only.
//
// It must:
// - not change progression logic
// - not replace getStage()
// - not alter pending-state semantics
// - not become a competing controller
//
// Runtime control and state mutation remain with the
// governed runtime progression pathway.
//
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
// Stuck helper flows may use a saved resume stage to recover their
// underlying structural location.
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
  // These mappings reflect current runtime pending-state semantics.
  // They are structural interpretations, not independent instructional rules.
  confirmationStageByPending: {
    confirmIsAbout: "isAboutConfirm",

     offerAnotherMainIdea: "parentItemsConfirm",
     collectAnotherMainIdea: "parentItemsConfirm",
     confirmMainIdeas: "parentItemsConfirm",

    offerAnotherDetail: "detailsConfirmLoop",
    collectAnotherDetail: "detailsConfirmLoop",
    confirmDetails: "detailsConfirmLoop",

    // Governed continuation of the same So What attempt
    // before confirmation.
    continueSoWhat: "soWhatConfirm",
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
// Post-completion runtime stages are interpreted structurally
// as "export" so the Parent Anchor endpoint remains stable
// across completion and export flows.
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
 * Returns the current Parent Anchor structural stage.
 *
 * This helper is a read-only interpretation layer.
 * It does not advance stages, mutate state, or replace getStage().
 *
 * It interprets the governed runtime through the Parent Anchor
 * structural stage model: the invariant Framing Routine spine
 * Key Topic -> Is About -> Main Ideas -> Details -> So What.
 *
 * How it works:
 * 1) It checks state.pending?.type first.
 *    - confirmation/export pending states override raw getStage()
 *    - stuck helper flows may recover a saved structural location
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
 * Architectural guardrail:
 * This helper explains the current engine structurally.
 * It does not own or replace runtime progression.
 */

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
    cleanText(
      frame.keyTopic ||
      ""
  );

  base.frame.isAbout =
    cleanText(
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

    const rawCleanedIsAbout =
    cleanFrameText(msg);

  const observationReport =
    s?.observationReport &&
    typeof s.observationReport ===
      "object"
      ? s.observationReport
      : null;

  const componentContribution =
    observationReport
      ?.componentContribution &&
    typeof observationReport
      .componentContribution ===
      "object"
      ? observationReport
          .componentContribution
      : null;

  const interactionOnlyCategories =
    new Set([
      "uncertaintyExpression",
      "clarificationRequest",
      "answerSeeking",
      "frustrationExpression",
      "refusal",
      "offTaskShift",
    ]);

  const interactionObservationPresent =
    Array.isArray(
      observationReport?.observations
    ) &&
    observationReport.observations.some(
      (observation) =>
        interactionOnlyCategories.has(
          cleanText(
            observation?.category || ""
          )
        )
    );

  const observedContributionText =
    componentContribution
      ?.observed === true
      ? cleanFrameText(
          componentContribution
            ?.evidenceText || ""
        )
      : "";

  const cleanedIsAbout =
    interactionObservationPresent &&
    observedContributionText
      ? observedContributionText
      : rawCleanedIsAbout;

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
        cleanedIsAbout,
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
      cleanedIsAbout,

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

  // --------------------------------------------------
// GUIDED CONSTRUCTION — IS ABOUT CONTINUATION
// --------------------------------------------------
//
// Normal governed Is About validation has already
// received first authority.
//
// If Guided Construction is already active at this exact
// Is About location, allow the shared Guided Construction
// runtime to evaluate the student's current response and
// update only Guided Construction-owned pathway state.
//
// Full Is About acceptance still belongs to the normal
// validator and normal Frame progression below.
//
// --------------------------------------------------

const activeGuidedConstruction =
  getActiveGuidedConstructionContext(
    s
  );

if (
  activeGuidedConstruction?.active ===
    true &&
  activeGuidedConstruction
    ?.frameComponent ===
    "isAbout"
) {
  await continueGuidedConstruction({
      state:
        s,

      response:
        cleanedIsAbout,

      componentValidation:
        validation,

      finalRephraseUsed:
        false,
    });
}

    if (
    !validation.valid ||
    progressionAuthorization
      ?.authorized !== true
  ) {
      
const pendingType =
  isStrengthen
    ? "strengthenReviseIsAbout"
    : "reviseIsAbout";

s.pending = {
  ...(
    s?.pending &&
    typeof s.pending === "object"
      ? s.pending
      : {}
  ),

  type:
    pendingType,

  captureMode,
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
      `isAboutValidation:${validation.diagnosis}`,

    instructionalFinding,
  }
);
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
  const rawMainIdea =
  cleanText(msg);

const observationReport =
  s?.observationReport &&
  typeof s.observationReport ===
    "object"
    ? s.observationReport
    : null;

const componentContribution =
  observationReport
    ?.componentContribution &&
  typeof observationReport
    .componentContribution ===
    "object"
    ? observationReport
        .componentContribution
    : null;

const interactionOnlyCategories =
  new Set([
    "uncertaintyExpression",
    "clarificationRequest",
    "answerSeeking",
    "frustrationExpression",
    "refusal",
    "offTaskShift",
  ]);

const interactionObservationPresent =
  Array.isArray(
    observationReport?.observations
  ) &&
  observationReport.observations.some(
    (observation) =>
      interactionOnlyCategories.has(
        cleanText(
          observation?.category || ""
        )
      )
  );

const observedContributionText =
  componentContribution
    ?.observed === true
    ? cleanText(
        componentContribution
          ?.evidenceText || ""
      )
    : "";

const text =
  correctObviousStudentSpelling(
    interactionObservationPresent &&
    observedContributionText
      ? observedContributionText
      : rawMainIdea
  );

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
          rawMainIdea,

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

// --------------------------------------------------
// GUIDED CONSTRUCTION — MAIN IDEA CONTINUATION
// --------------------------------------------------
//
// Normal governed Main Idea validation has already
// received first authority.
//
// If Guided Construction is already active at this exact
// Main Idea location, allow the shared Guided Construction
// runtime to evaluate the student's current response and
// update only Guided Construction-owned pathway state.
//
// Full Main Idea acceptance still belongs to the normal
// validator and normal Frame progression below.
//
// --------------------------------------------------

const activeGuidedConstruction =
  getActiveGuidedConstructionContext(
    s
  );

if (
  activeGuidedConstruction?.active ===
    true &&
  activeGuidedConstruction
    ?.frameComponent ===
    "mainIdeas"
) {
  await continueGuidedConstruction({
      state:
        s,

      response:
        text,

      componentValidation:
        validation,
      
      finalRephraseUsed:
        false,
    });
}
  
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
  
s.pending = {
  ...(
    s?.pending &&
    typeof s.pending === "object"
      ? s.pending
      : {}
  ),

  ...pendingLocation,

  captureMode,
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

if (count === 1) {
  s.pending = {
    type:
      "collectAnotherMainIdea",
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

  const paStage =
    paContext.ownerStructuralStage;
    
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

  return getComponentPrompt(
    "isAbout",
    "confirmationPrompt",
    {
      keyTopic:
        s.frame.keyTopic,

      isAbout:
        isAboutDisplay,
    }
  );
}

if (
  s.pending?.type ===
  "reviseIsAbout"
) {
  return getComponentPrompt(
    "isAbout",
    "revisePrompt"
  );
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
  
if (s.pending?.type === "continueSoWhat") {
  return (
    `🎯 So What\n\n` +
    `Let's keep working on your So What.\n\n` +
    `Looking across your completed Frame, what is the most important thing someone should understand?`
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

// DETAILS LOOP
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

}

// ======================================================
// REDIRECT INTERPRETATION
// ======================================================
//
// Redirect interpretation is a side-effect-free,
// pre-instructional language layer.
//
// It may observe whether the student's message appears
// to request a change in instructional location or action.
//
// It does not:
//
// • validate whether a requested target exists;
// • authorize navigation;
// • mutate Frame or pending state;
// • change Build / Strengthen mode;
// • change Progressive Support or Guided Construction;
// • determine instructional strategy;
// • generate student-facing communication.
//
// Deterministic runtime validation will later decide
// whether an interpreted redirect may be acted upon.
//
// ======================================================

function buildRedirectInterpretationEligibility(
  state,
  message
) {
  const text =
    cleanText(message);

  if (!text) {
    return {
      eligible:
        false,

      reason:
        "emptyInput",
    };
  }

  const pendingType =
    cleanText(
      state?.pending?.type || ""
    );

  // Language confirmation is already owned by its
  // dedicated deterministic runtime pathway.
  if (
    pendingType ===
    "confirmLanguageSwitch"
  ) {
    return {
      eligible:
        false,

      reason:
        "languageConfirmation",
    };
  }

  // Explicit numeric menu selections already have
  // deterministic meaning in these pending contracts.
  // Do not ask AI to reinterpret them.
  const deterministicMenuChoices = {
    confirmAssignmentUnderstanding:
      new Set(["1", "2"]),

    assignmentReasoningIntro:
      new Set(["1", "2"]),

    confirmIsAbout:
      new Set(["1", "2"]),

    strengthenComponentSelection:
      new Set(["1", "2", "3", "4"]),

    strengthenComponentComplete:
      new Set(["1", "2", "3"]),

    chooseMainIdeaToRevise:
      new Set(["1", "2", "3", "4", "5"]),

    offerAnotherMainIdea:
      new Set(["1", "2"]),

    offerAnotherDetail:
      new Set(["1", "2"]),

    confirmMainIdeas:
      new Set(["1", "2"]),

    confirmDetails:
      new Set(["1", "2"]),

    chooseDetailToRevise:
      new Set(["1", "2", "3", "4", "5"]),

    confirmSoWhat:
      new Set(["1", "2"]),

    offerExport:
      new Set(["1", "2"]),

    chooseExportType:
      new Set(["1", "2", "3"]),
  };

  const allowedChoices =
    deterministicMenuChoices[
      pendingType
    ];

  if (
    allowedChoices instanceof Set &&
    allowedChoices.has(text)
  ) {
    return {
      eligible:
        false,

      reason:
        "deterministicMenuInput",
    };
  }

  return {
    eligible:
      true,

    reason:
      "eligibleForInterpretation",
  };
}

async function interpretRedirectIntent(
  state,
  message
) {
  const text =
    cleanText(message);

  const system = `You are the bounded Redirect Interpretation Layer for Kaw Companion.

Your responsibility is only to observe whether the student's current message is managing the tutoring interaction by requesting a change in where or how they want to work within their KU Framing Routine Frame.

You may interpret natural student language.

You do not decide whether a requested navigation is allowed.

You do not:
- validate whether a target exists;
- authorize navigation;
- determine progression;
- choose instructional strategy;
- choose a Teaching Move or Thinking Move;
- change Build or Strengthen mode;
- change Progressive Support;
- change Guided Construction;
- create pending state;
- mutate any runtime state;
- generate student-facing communication.

Distinguish interaction-management requests from ordinary subject-matter content.

For example:
- "Go back to my first Main Idea." is interaction management.
- "Going back to my example, social media affects relationships." is ordinary content.

Describe semantic references only. Do not invent trusted runtime indexes.

If there is no redirect request, return noRedirectObserved.

If the student clearly wants to change direction but the linguistic target is underspecified, return redirectPossiblyObserved.

evidenceText must be an exact excerpt from the student's current message when a redirect is observed or possibly observed. Otherwise return an empty string.

Return only the required JSON object.`;

  const user = `Current instructional context:

Interaction mode:
${cleanText(
  state?.interactionMode ||
  "build"
)}

Current stage:
${cleanText(
  getStage(state) || ""
)}

Current pending type:
${cleanText(
  state?.pending?.type || ""
)}

Accepted Main Ideas:
${JSON.stringify(
  getIdeaList(state),
  null,
  2
)}

Current student message:
"${text}"

Report only the student's apparent redirect/navigation meaning.`;

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
              "kaw_redirect_interpretation",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                interpretationStatus: {
                  type:
                    "string",

                  enum: [
                    "redirectObserved",
                    "redirectPossiblyObserved",
                    "noRedirectObserved",
                  ],
                },

                redirectIntent: {
                  type:
                    "string",

                  enum: [
                    "revisitTarget",
                    "switchTarget",
                    "leaveCurrentPath",
                    "requestForwardTarget",
                    "unspecified",
                  ],
                },

                requestedTarget: {
                  type:
                    "object",

                  additionalProperties:
                    false,

                  properties: {
                    component: {
                      type:
                        "string",

                      enum: [
                        "keyTopic",
                        "isAbout",
                        "mainIdeas",
                        "details",
                        "soWhat",
                        "unspecified",
                      ],
                    },

                    mainIdeaReference: {
                      type:
                        "string",

                      enum: [
                        "ordinal1",
                        "ordinal2",
                        "ordinal3",
                        "ordinal4",
                        "ordinal5",
                        "current",
                        "previous",
                        "other",
                        "unspecified",
                      ],
                    },

                    detailReference: {
                      type:
                        "string",

                      enum: [
                        "ordinal1",
                        "ordinal2",
                        "ordinal3",
                        "ordinal4",
                        "ordinal5",
                        "current",
                        "previous",
                        "other",
                        "unspecified",
                      ],
                    },
                  },

                  required: [
                    "component",
                    "mainIdeaReference",
                    "detailReference",
                  ],
                },

                requestedOperation: {
                  type:
                    "string",

                  enum: [
                    "workOn",
                    "revise",
                    "strengthen",
                    "addSupportingContent",
                    "unspecified",
                  ],
                },

                currentPathDisposition: {
                  type:
                    "string",

                  enum: [
                    "continue",
                    "decline",
                    "unspecified",
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
                "interpretationStatus",
                "redirectIntent",
                "requestedTarget",
                "requestedOperation",
                "currentPathDisposition",
                "evidenceText",
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
        response?.choices?.[0]
          ?.message?.content || "{}"
      );

    const confidence =
      Number(
        parsed?.confidence || 0
      );

    const normalizedConfidence =
      Number.isFinite(confidence)
        ? Math.max(
            0,
            Math.min(
              confidence,
              1
            )
          )
        : 0;

    const evidenceText =
      cleanText(
        parsed?.evidenceText || ""
      );

    const evidenceGrounded =
      !evidenceText ||
      text
        .toLowerCase()
        .includes(
          evidenceText.toLowerCase()
        );

    if (
      normalizedConfidence < 0.9 ||
      !evidenceGrounded
    ) {
      return {
        artifactType:
          "redirectInterpretation",

        interpretationStatus:
          "interpreterFailure",

        source:
          "redirectInterpretationUnavailable",
      };
    }

    return {
      artifactType:
        "redirectInterpretation",

      version:
        "1.0",

      source:
        "aiBoundedRedirectInterpretation",

      interpretationStatus:
        parsed.interpretationStatus,

      redirectIntent:
        parsed.redirectIntent,

      requestedTarget:
        structuredClone(
          parsed.requestedTarget
        ),

      requestedOperation:
        parsed.requestedOperation,

      currentPathDisposition:
        parsed.currentPathDisposition,

      evidenceText,

      confidence:
        normalizedConfidence,

      governance: {
        observationalOnly:
          true,

        controlsNavigation:
          false,

        controlsProgression:
          false,

        controlsPendingState:
          false,

        controlsInstructionalStrategy:
          false,
      },
    };
  } catch (error) {
    console.error(
      "Redirect interpretation error:",
      error
    );

    return {
      artifactType:
        "redirectInterpretation",

      interpretationStatus:
        "interpreterFailure",

      source:
        "redirectInterpretationUnavailable",
    };
  }
}

// ======================================================
// REDIRECT CLARIFICATION RESOLUTION
// ======================================================
//
// Resolves a student's answer to an already-established
// redirect clarification question.
//
// The original redirect interpretation remains the source
// of the requested operation and known target context.
//
// AI may resolve only the missing semantic reference.
//
// AI does not:
//
// • authorize navigation;
// • return trusted runtime indexes;
// • mutate state;
// • change instructional strategy;
// • generate student-facing communication.
//
// Deterministic redirect validation remains authoritative.
//
// ======================================================

async function interpretRedirectClarificationResolution(
  state,
  message,
  redirectNavigationBoundary
) {
  const text =
    cleanText(message);

  const boundary =
    redirectNavigationBoundary &&
    typeof redirectNavigationBoundary ===
      "object"
      ? redirectNavigationBoundary
      : null;

  const priorInterpretation =
    boundary?.interpretation &&
    typeof boundary.interpretation ===
      "object"
      ? boundary.interpretation
      : null;

  if (
    !text ||
    boundary?.status !==
      "clarificationRequired" ||
    !priorInterpretation
  ) {
    return {
      artifactType:
        "redirectInterpretation",

      interpretationStatus:
        "interpreterFailure",

      source:
        "redirectClarificationUnavailable",
    };
  }

  const priorTarget =
    priorInterpretation
      ?.requestedTarget &&
    typeof priorInterpretation
      .requestedTarget ===
      "object"
      ? priorInterpretation.requestedTarget
      : {};

  const system = `You are the bounded Redirect Clarification Resolution Layer for Kaw Companion.

The student previously made a navigation request inside their KU Framing Routine Frame.

Kaw asked one clarification question because the requested target could not be uniquely resolved.

Your only responsibility is to interpret the student's current answer as a possible resolution of that navigation clarification.

Preserve all already-established navigation meaning from the prior request.

You may resolve only semantic target references expressed by the student's current answer.

Use only these semantic references:
- ordinal1
- ordinal2
- ordinal3
- ordinal4
- ordinal5
- current
- previous
- other
- unspecified

Do not return runtime indexes.

Do not:
- authorize navigation;
- decide whether a target exists;
- decide whether the target is allowed;
- change the requested operation;
- change instructional mode;
- determine pedagogy;
- interpret the response as Frame content;
- generate student-facing communication.

If the current answer clearly resolves the requested target, return resolutionObserved.

If it still does not uniquely resolve the target, return resolutionUnclear.

evidenceText must be an exact excerpt from the student's current message.

Return only the required JSON object.`;

  const user = `Prior redirect interpretation:
${JSON.stringify(
  priorInterpretation,
  null,
  2
)}

Accepted Main Ideas:
${JSON.stringify(
  getIdeaList(state),
  null,
  2
)}

Accepted Essential Details:
${JSON.stringify(
  Array.isArray(
    state?.frame?.details
  )
    ? state.frame.details
    : [],
  null,
  2
)}

Student's clarification answer:
"${text}"

Resolve only the missing navigation target reference.`;

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
              "kaw_redirect_clarification_resolution",

            strict:
              true,

            schema: {
              type:
                "object",

              additionalProperties:
                false,

              properties: {
                resolutionStatus: {
                  type:
                    "string",

                  enum: [
                    "resolutionObserved",
                    "resolutionUnclear",
                  ],
                },

                mainIdeaReference: {
                  type:
                    "string",

                  enum: [
                    "ordinal1",
                    "ordinal2",
                    "ordinal3",
                    "ordinal4",
                    "ordinal5",
                    "current",
                    "previous",
                    "other",
                    "unspecified",
                  ],
                },

                detailReference: {
                  type:
                    "string",

                  enum: [
                    "ordinal1",
                    "ordinal2",
                    "ordinal3",
                    "ordinal4",
                    "ordinal5",
                    "current",
                    "previous",
                    "other",
                    "unspecified",
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
                "resolutionStatus",
                "mainIdeaReference",
                "detailReference",
                "evidenceText",
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
        response?.choices?.[0]
          ?.message?.content || "{}"
      );

    const confidence =
      Number(
        parsed?.confidence || 0
      );

    const normalizedConfidence =
      Number.isFinite(confidence)
        ? Math.max(
            0,
            Math.min(
              confidence,
              1
            )
          )
        : 0;

    const evidenceText =
      cleanText(
        parsed?.evidenceText || ""
      );

    const evidenceGrounded =
      Boolean(
        evidenceText &&
        text
          .toLowerCase()
          .includes(
            evidenceText
              .toLowerCase()
          )
      );

    if (
      parsed?.resolutionStatus !==
        "resolutionObserved" ||
      normalizedConfidence < 0.9 ||
      !evidenceGrounded
    ) {
      return {
        artifactType:
          "redirectInterpretation",

        interpretationStatus:
          "redirectPossiblyObserved",

        version:
          "1.0",

        source:
          "aiBoundedRedirectClarificationResolution",

        redirectIntent:
          priorInterpretation
            ?.redirectIntent ||
          "unspecified",

        requestedTarget: {
          component:
            priorTarget
              ?.component ||
            "unspecified",

          mainIdeaReference:
            priorTarget
              ?.mainIdeaReference ||
            "unspecified",

          detailReference:
            priorTarget
              ?.detailReference ||
            "unspecified",
        },

        requestedOperation:
          priorInterpretation
            ?.requestedOperation ||
          "unspecified",

        currentPathDisposition:
          priorInterpretation
            ?.currentPathDisposition ||
          "unspecified",

        evidenceText,

        confidence:
          normalizedConfidence,
      };
    }

    return {
      artifactType:
        "redirectInterpretation",

      version:
        "1.0",

      source:
        "aiBoundedRedirectClarificationResolution",

      interpretationStatus:
        "redirectObserved",

      redirectIntent:
        priorInterpretation
          ?.redirectIntent ||
        "unspecified",

      requestedTarget: {
        component:
          priorTarget
            ?.component ||
          "unspecified",

        mainIdeaReference:
          cleanText(
            parsed
              ?.mainIdeaReference ||
            "unspecified"
          ) !== "unspecified"
            ? parsed.mainIdeaReference
            : priorTarget
                ?.mainIdeaReference ||
              "unspecified",

        detailReference:
          cleanText(
            parsed
              ?.detailReference ||
            "unspecified"
          ) !== "unspecified"
            ? parsed.detailReference
            : priorTarget
                ?.detailReference ||
              "unspecified",
      },

      requestedOperation:
        priorInterpretation
          ?.requestedOperation ||
        "unspecified",

      currentPathDisposition:
        priorInterpretation
          ?.currentPathDisposition ||
        "unspecified",

      evidenceText,

      confidence:
        normalizedConfidence,

      governance: {
        observationalOnly:
          true,

        controlsNavigation:
          false,

        controlsProgression:
          false,

        controlsPendingState:
          false,

        controlsInstructionalStrategy:
          false,
      },
    };
  } catch (error) {
    console.error(
      "Redirect clarification resolution error:",
      error
    );

    return {
      artifactType:
        "redirectInterpretation",

      interpretationStatus:
        "interpreterFailure",

      source:
        "redirectClarificationUnavailable",
    };
  }
}

// ======================================================
// REDIRECT VALIDATION
// ======================================================
//
// Deterministically validates one established redirect
// interpretation against canonical Kaw runtime state.
//
// AI describes the student's apparent request.
// JavaScript determines whether the requested target can
// be resolved and whether the existing runtime currently
// has a supported re-entry pathway for it.
//
// This layer does not:
//
// • mutate Frame or pending state;
// • apply navigation;
// • clear instructional artifacts;
// • change Build / Strengthen mode;
// • change Progressive Support or Guided Construction;
// • generate student-facing communication.
//
// ======================================================

function resolveRedirectOrdinalReference(
  reference
) {
  const text =
    cleanText(reference);

  const match =
    text.match(
      /^ordinal([1-5])$/
    );

  if (!match) {
    return null;
  }

  return (
    Number(match[1]) - 1
  );
}

function buildRedirectCurrentPathDispositionValidation(
  state,
  redirectInterpretation
) {
  const disposition =
    cleanText(
      redirectInterpretation
        ?.currentPathDisposition ||
      "unspecified"
    );

  if (disposition !== "decline") {
    return {
      dispositionStatus:
        "notApplicable",

      declineRequested:
        false,

      declineAuthorized:
        false,

      reason:
        "declineNotRequested",
    };
  }

  const pending =
    state?.pending &&
    typeof state.pending === "object"
      ? state.pending
      : {};

  const pendingType =
    cleanText(
      pending?.type || ""
    );

  const captureMode =
    cleanText(
      pending?.captureMode || ""
    );

  const mainIdeaIndex =
    Number.isInteger(
      pending?.index
    )
      ? pending.index
      : null;

  const detailBucket =
    Number.isInteger(
      mainIdeaIndex
    ) &&
    Array.isArray(
      state?.frame
        ?.details
        ?.[mainIdeaIndex]
    )
      ? state.frame.details[
          mainIdeaIndex
        ]
      : [];

  const currentDetailCount =
    detailBucket.length;

  const declineAuthorized =
    captureMode === "optional" ||

    pendingType ===
      "offerAnotherMainIdea" ||

    pendingType ===
      "offerAnotherDetail" ||

    (
      pendingType ===
        "collectAnotherDetail" &&
      currentDetailCount >= 2
    );

  return {
    dispositionStatus:
      declineAuthorized
        ? "authorized"
        : "notAuthorized",

    declineRequested:
      true,

    declineAuthorized,

    pendingType:
      pendingType || null,

    currentDetailCount,

    reason:
      declineAuthorized
        ? "currentPathIsDeclinable"
        : "currentPathIsNotDeclinable",
  };
}

function buildRedirectValidation(
  state,
  redirectInterpretation
) {
  const interpretation =
    redirectInterpretation &&
    typeof redirectInterpretation ===
      "object"
      ? redirectInterpretation
      : null;

  const currentPathDispositionValidation =
  buildRedirectCurrentPathDispositionValidation(
    state,
    redirectInterpretation
  );

  const interpretationStatus =
    cleanText(
      interpretation
        ?.interpretationStatus || ""
    );

  if (
    !interpretation ||
    interpretationStatus ===
      "noRedirectObserved" ||
    interpretationStatus ===
      "interpreterFailure"
  ) {
    return {
      artifactType:
        "redirectValidation",

      version:
        "1.0",

      source:
        "deterministicRedirectValidator",

      validationStatus:
        "notApplicable",

      navigationAuthorized:
        false,

      resolvedTarget:
        null,

      validationEvidence: [
        interpretationStatus ||
          "redirectInterpretationUnavailable",
      ],
    };
  }


  if (
  currentPathDispositionValidation
    ?.declineRequested === true &&
  currentPathDispositionValidation
    ?.declineAuthorized !== true
) {
  return {
    artifactType:
      "redirectValidation",

    version:
      "1.0",

    source:
      "deterministicRedirectValidator",

    validationStatus:
      "notAuthorized",

    navigationAuthorized:
      false,

    resolvedTarget:
      null,

    currentPathDispositionValidation:
      structuredClone(
        currentPathDispositionValidation
      ),

    validationEvidence: [
      currentPathDispositionValidation
        ?.reason ||
      "currentPathDeclineNotAuthorized",
    ],
  };
}

  const hasExplicitRedirectTarget =
  cleanText(
    interpretation
      ?.requestedTarget
      ?.component || ""
  ) !== "" &&
  cleanText(
    interpretation
      ?.requestedTarget
      ?.component || ""
  ) !== "unspecified";

  if (
  currentPathDispositionValidation
    ?.declineRequested === true &&
  currentPathDispositionValidation
    ?.declineAuthorized === true &&
  hasExplicitRedirectTarget !== true
) {
  const currentPending =
    state?.pending &&
    typeof state.pending === "object"
      ? state.pending
      : {};

  return {
    artifactType:
      "redirectValidation",

    version:
      "1.0",

    source:
      "deterministicRedirectValidator",

    validationStatus:
      "authorized",

    navigationAuthorized:
      true,

    resolvedTarget: {
      operation:
        "declineCurrentPath",

      pendingType:
        cleanText(
          currentPending?.type || ""
        ) || null,

      mainIdeaIndex:
        Number.isInteger(
          currentPending?.index
        )
          ? currentPending.index
          : null,
    },

    currentPathDispositionValidation:
      structuredClone(
        currentPathDispositionValidation
      ),

    validationEvidence: [
      "currentPathDeclineRequested",
      "currentPathDeclineAuthorized",
    ],
  };
}

  const requestedTarget =
  interpretation
    ?.requestedTarget &&
  typeof interpretation
    .requestedTarget === "object"
    ? interpretation.requestedTarget
    : {};

  const component =
    cleanText(
      requestedTarget
        ?.component || ""
    );

  const mainIdeas =
    getIdeaList(state)
      .map(cleanText)
      .filter(Boolean);

  const details =
    Array.isArray(
      state?.frame?.details
    )
      ? state.frame.details
      : [];

  let mainIdeaIndex =
    resolveRedirectOrdinalReference(
      requestedTarget
        ?.mainIdeaReference
    );

  let detailIndex =
    resolveRedirectOrdinalReference(
      requestedTarget
        ?.detailReference
    );

  const mainIdeaReference =
    cleanText(
      requestedTarget
        ?.mainIdeaReference || ""
    );

  const detailReference =
    cleanText(
      requestedTarget
        ?.detailReference || ""
    );

  // --------------------------------------------------
  // RELATIVE MAIN IDEA REFERENCES
  // --------------------------------------------------

  const currentMainIdeaIndex =
    Number.isInteger(
      state?.pending?.index
    )
      ? state.pending.index
      : null;

  if (
    mainIdeaIndex === null &&
    mainIdeaReference ===
      "current" &&
    Number.isInteger(
      currentMainIdeaIndex
    )
  ) {
    mainIdeaIndex =
      currentMainIdeaIndex;
  }

  if (
    mainIdeaIndex === null &&
    mainIdeaReference ===
      "previous" &&
    Number.isInteger(
      currentMainIdeaIndex
    ) &&
    currentMainIdeaIndex > 0
  ) {
    mainIdeaIndex =
      currentMainIdeaIndex - 1;
  }

  if (
    mainIdeaIndex === null &&
    mainIdeaReference ===
      "other" &&
    Number.isInteger(
      currentMainIdeaIndex
    )
  ) {
    const otherIndexes =
      mainIdeas
        .map(
          (_, index) => index
        )
        .filter(
          (index) =>
            index !==
            currentMainIdeaIndex
        );

    if (
      otherIndexes.length === 1
    ) {
      mainIdeaIndex =
        otherIndexes[0];
    } else if (
      otherIndexes.length > 1
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "clarificationRequired",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "multipleMainIdeaCandidates",
        ],
      };
    }
  }

  // --------------------------------------------------
  // CURRENT DETAIL REFERENCE
  // --------------------------------------------------

  if (
    detailIndex === null &&
    detailReference ===
      "current" &&
    Number.isInteger(
      state?.pending?.detailIndex
    )
  ) {
    detailIndex =
      state.pending.detailIndex;
  }

  // --------------------------------------------------
  // COMPONENT-SPECIFIC VALIDATION
  // --------------------------------------------------

  if (
    component === "isAbout"
  ) {
    if (
      !cleanText(
        state?.frame?.isAbout || ""
      )
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "notAuthorized",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "acceptedIsAboutUnavailable",
        ],
      };
    }

    return {
      artifactType:
        "redirectValidation",

      version:
        "1.0",

      source:
        "deterministicRedirectValidator",

      validationStatus:
        "authorized",

      navigationAuthorized:
        true,

      resolvedTarget: {
        component:
          "isAbout",

        mainIdeaIndex:
          null,

        detailIndex:
          null,
      },

      validationEvidence: [
        "acceptedIsAboutExists",
        "supportedFreshReentryPathway",
      ],
    };
  }

  if (
    component === "mainIdeas"
  ) {
    if (
      !Number.isInteger(
        mainIdeaIndex
      )
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "clarificationRequired",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "mainIdeaTargetNotUniquelyResolved",
        ],
      };
    }

    if (
      mainIdeaIndex < 0 ||
      mainIdeaIndex >=
        mainIdeas.length
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "notAuthorized",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "requestedMainIdeaDoesNotExist",
        ],
      };
    }

    return {
      artifactType:
        "redirectValidation",

      version:
        "1.0",

      source:
        "deterministicRedirectValidator",

      validationStatus:
        "authorized",

      navigationAuthorized:
        true,

      resolvedTarget: {
        component:
          "mainIdeas",

        mainIdeaIndex,

        detailIndex:
          null,
      },

      validationEvidence: [
        "requestedMainIdeaExists",
        "supportedFreshReentryPathway",
      ],
    };
  }

  if (
    component === "details"
  ) {
    if (
      !Number.isInteger(
        mainIdeaIndex
      )
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "clarificationRequired",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "detailParentMainIdeaNotResolved",
        ],
      };
    }

    if (
      mainIdeaIndex < 0 ||
      mainIdeaIndex >=
        mainIdeas.length
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "notAuthorized",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "detailParentMainIdeaDoesNotExist",
        ],
      };
    }

    const detailBucket =
      Array.isArray(
        details?.[mainIdeaIndex]
      )
        ? details[mainIdeaIndex]
            .map(cleanText)
            .filter(Boolean)
        : [];

    const requestedOperation =
      cleanText(
        interpretation
          ?.requestedOperation || ""
      );

    if (
      requestedOperation ===
        "addSupportingContent"
    ) {
      if (
        detailBucket.length >= 5
      ) {
        return {
          artifactType:
            "redirectValidation",

          version:
            "1.0",

          source:
            "deterministicRedirectValidator",

          validationStatus:
            "notAuthorized",

          navigationAuthorized:
            false,

          resolvedTarget:
            null,

          validationEvidence: [
            "detailLimitReached",
          ],
        };
      }

      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "authorized",

        navigationAuthorized:
          true,

        resolvedTarget: {
          component:
            "details",

          mainIdeaIndex,

          detailIndex:
            detailBucket.length,
        },

        validationEvidence: [
          "parentMainIdeaExists",
          "additionalDetailSlotAvailable",
        ],
      };
    }

    if (
      !Number.isInteger(
        detailIndex
      )
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "clarificationRequired",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "detailTargetNotUniquelyResolved",
        ],
      };
    }

    if (
      detailIndex < 0 ||
      detailIndex >=
        detailBucket.length
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "notAuthorized",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "requestedDetailDoesNotExist",
        ],
      };
    }

    return {
      artifactType:
        "redirectValidation",

      version:
        "1.0",

      source:
        "deterministicRedirectValidator",

      validationStatus:
        "authorized",

      navigationAuthorized:
        true,

      resolvedTarget: {
        component:
          "details",

        mainIdeaIndex,

        detailIndex,
      },

      validationEvidence: [
        "requestedDetailExists",
        "supportedFreshReentryPathway",
      ],
    };
  }

  if (
    component === "soWhat"
  ) {
    if (
      !cleanText(
        state?.frame?.soWhat || ""
      )
    ) {
      return {
        artifactType:
          "redirectValidation",

        version:
          "1.0",

        source:
          "deterministicRedirectValidator",

        validationStatus:
          "notAuthorized",

        navigationAuthorized:
          false,

        resolvedTarget:
          null,

        validationEvidence: [
          "freshSoWhatReentryNotYetSupported",
        ],
      };
    }

    return {
      artifactType:
        "redirectValidation",

      version:
        "1.0",

      source:
        "deterministicRedirectValidator",

      validationStatus:
        "authorized",

      navigationAuthorized:
        true,

      resolvedTarget: {
        component:
          "soWhat",

        mainIdeaIndex:
          null,

        detailIndex:
          null,
      },

      validationEvidence: [
        "acceptedSoWhatExists",
        "existingSoWhatRevisionAvailable",
      ],
    };
  }

  // Accepted Key Topic revision is one of the Phase-4
  // compatibility gaps and is not authorized yet.
  if (
    component === "keyTopic"
  ) {
    return {
      artifactType:
        "redirectValidation",

      version:
        "1.0",

      source:
        "deterministicRedirectValidator",

      validationStatus:
        "notAuthorized",

      navigationAuthorized:
        false,

      resolvedTarget:
        null,

      validationEvidence: [
        "freshKeyTopicReentryNotYetSupported",
      ],
    };
  }

  return {
    artifactType:
      "redirectValidation",

    version:
      "1.0",

    source:
      "deterministicRedirectValidator",

    validationStatus:
      "clarificationRequired",

    navigationAuthorized:
      false,

    resolvedTarget:
      null,

    validationEvidence: [
      "redirectTargetNotResolved",
    ],
  };
}

// ======================================================
// REDIRECT NAVIGATION PREPARATION
// ======================================================
//
// Converts one deterministically authorized redirect into
// a candidate existing Kaw re-entry state.
//
// This is preparation only.
//
// It does not:
//
// • mutate the live runtime state;
// • clear the current pending location;
// • clear Progressive Support or Guided Construction;
// • change interactionMode;
// • save or modify Frame content;
// • apply navigation;
// • generate student-facing communication.
//
// Navigation may be committed only after this replacement
// state has been completely prepared and verified.
//
// ======================================================

function buildRedirectNavigationPreparation(
  state,
  redirectInterpretation,
  redirectValidation
) {
  const validation =
    redirectValidation &&
    typeof redirectValidation === "object"
      ? redirectValidation
      : null;

  if (
    validation?.validationStatus !==
      "authorized" ||
    validation?.navigationAuthorized !==
      true
  ) {
    return {
      artifactType:
        "redirectNavigationPreparation",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationPreparation",

      preparationStatus:
        "notApplicable",

      replacementPending:
        null,

      verified:
        false,
    };
  }

  const target =
    validation?.resolvedTarget &&
    typeof validation.resolvedTarget ===
      "object"
      ? validation.resolvedTarget
      : null;

  if (!target) {
    return {
      artifactType:
        "redirectNavigationPreparation",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationPreparation",

      preparationStatus:
        "failed",

      replacementPending:
        null,

      verified:
        false,

      reason:
        "resolvedTargetUnavailable",
    };
  }

  const component =
    cleanText(
      target?.component || ""
    );

  const requestedOperation =
    cleanText(
      redirectInterpretation
        ?.requestedOperation || ""
    );

  let replacementPending =
    null;

  const resolvedOperation =
  cleanText(
    target?.operation || ""
  );

if (
  resolvedOperation ===
    "declineCurrentPath"
) {
  const currentPendingType =
    cleanText(
      target?.pendingType || ""
    );

  if (
    currentPendingType ===
      "offerAnotherMainIdea"
  ) {
    replacementPending = {
      type:
        "confirmMainIdeas",
    };
  } else if (
    currentPendingType ===
      "offerAnotherDetail" ||
    currentPendingType ===
      "collectAnotherDetail"
  ) {
    if (
      Number.isInteger(
        target?.mainIdeaIndex
      )
    ) {
      replacementPending = {
        type:
          "confirmDetails",

        index:
          target.mainIdeaIndex,
      };
    }
  }
}

  // --------------------------------------------------
  // IS ABOUT REVISION
  // --------------------------------------------------

    if (
      !replacementPending &&
      component === "isAbout"
) {
    replacementPending = {
      type:
        "reviseIsAbout",

      captureMode:
        "revision",
    };
  }

  // --------------------------------------------------
  // MAIN IDEA REVISION
  // --------------------------------------------------

    else if (
    !replacementPending &&
    component === "mainIdeas" &&
    Number.isInteger(
      target?.mainIdeaIndex
    )
  ) {
    replacementPending = {
      type:
        "reviseMainIdeaAt",

      index:
        target.mainIdeaIndex,

      captureMode:
        "revision",
    };
  }

  // --------------------------------------------------
  // ESSENTIAL DETAIL
  //
  // addSupportingContent enters the existing
  // collect-another-detail pathway.
  //
  // Other authorized Detail requests revise one accepted
  // Detail at the resolved coordinates.
  // --------------------------------------------------

    else if (
      !replacementPending &&
      component === "details" &&
      Number.isInteger(
      target?.mainIdeaIndex
  )
) {
  if (
    requestedOperation ===
      "addSupportingContent"
  ) {
    const existingDetailBucket =
      Array.isArray(
        state?.frame
          ?.details
          ?.[target.mainIdeaIndex]
      )
        ? state.frame.details[
            target.mainIdeaIndex
          ]
        : [];

    const nextDetailIndex =
      existingDetailBucket.length;

    const nextCaptureMode =
      nextDetailIndex < 2
        ? "required"
        : "optional";

    replacementPending = {
      type:
        "collectAnotherDetail",

      index:
        target.mainIdeaIndex,

      detailIndex:
        nextDetailIndex,

      captureMode:
        nextCaptureMode,
    };
  } else if (
    Number.isInteger(
      target?.detailIndex
    )
  ) {
    replacementPending = {
      type:
        "reviseDetailAt",

      index:
        target.mainIdeaIndex,

      detailIndex:
        target.detailIndex,

      captureMode:
        "revision",
    };
  }
}

  // --------------------------------------------------
  // EXISTING SO WHAT REVISION
  // --------------------------------------------------

  else if (
    !replacementPending &&
    component === "soWhat"
  ) {
    replacementPending = {
      type:
        "confirmSoWhat",
  };
}

  if (!replacementPending) {
    return {
      artifactType:
        "redirectNavigationPreparation",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationPreparation",

      preparationStatus:
        "failed",

      replacementPending:
        null,

      verified:
        false,

      reason:
        "supportedReplacementPendingUnavailable",
    };
  }

  // --------------------------------------------------
  // VERIFY WITHOUT MUTATING LIVE STATE
  //
  // The candidate is evaluated in a cloned state.
  // Nothing has been cleared or committed.
  // --------------------------------------------------

  const candidateState =
    structuredClone(state);

  candidateState.pending =
    structuredClone(
      replacementPending
    );

  const candidatePendingValid =
    candidateState?.pending &&
    typeof candidateState.pending ===
      "object" &&
    cleanText(
      candidateState.pending.type || ""
    );

  if (!candidatePendingValid) {
    return {
      artifactType:
        "redirectNavigationPreparation",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationPreparation",

      preparationStatus:
        "failed",

      replacementPending:
        null,

      verified:
        false,

      reason:
        "replacementPendingVerificationFailed",
    };
  }

  return {
    artifactType:
      "redirectNavigationPreparation",

    version:
      "1.0",

    source:
      "deterministicRedirectNavigationPreparation",

    preparationStatus:
      "prepared",

    replacementPending:
      structuredClone(
        replacementPending
      ),

    verified:
      true,

    resolvedTarget:
      structuredClone(
        target
      ),

    governance: {
      deterministicPreparation:
        true,

      mutatesLiveState:
        false,

      navigationCommitted:
        false,
    },
  };
}

// ======================================================
// REDIRECT NAVIGATION COMMIT
// ======================================================
//
// Applies one fully prepared and verified redirect as an
// all-or-nothing state transaction.
//
// The live source state is never modified while the
// replacement is being constructed.
//
// Commit sequence:
//
// prepare replacement → verify → create candidate state
// → clear stale location-owned artifacts → verify
// → return committed candidate
//
// Canonical Frame and session-level state remain intact.
//
// ======================================================

function buildRedirectNavigationCommit(
  state,
  redirectNavigationPreparation
) {
  const preparation =
    redirectNavigationPreparation &&
    typeof redirectNavigationPreparation ===
      "object"
      ? redirectNavigationPreparation
      : null;

  if (
    preparation?.preparationStatus !==
      "prepared" ||
    preparation?.verified !== true ||
    !preparation?.replacementPending ||
    typeof preparation.replacementPending !==
      "object"
  ) {
    return {
      artifactType:
        "redirectNavigationCommit",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationCommit",

      commitStatus:
        "notApplicable",

      committed:
        false,

      committedState:
        null,
    };
  }

  const replacementPending =
    structuredClone(
      preparation.replacementPending
    );

  const currentGuidedConstructionStep =
  Number(
    state?.pending
      ?.guidedConstructionStep
  );

const guidedConstructionActive =
  Number(
    state?.pending
      ?.progressiveSupportStage
  ) === 3 &&

  Number.isInteger(
    currentGuidedConstructionStep
  ) &&

  currentGuidedConstructionStep >= 1 &&
  currentGuidedConstructionStep <= 3 &&

  state?.pending
    ?.guidedConstructionLocation
    ?.locationEstablished === true;
  
  // --------------------------------------------------
  // SAME EXACT RE-ENTRY LOCATION
  //
  // Do not destructively rebuild an instructional
  // location the student is already occupying.
  // --------------------------------------------------

  const currentPending =
    state?.pending &&
    typeof state.pending === "object"
      ? state.pending
      : null;

  const samePendingType =
    cleanText(
      currentPending?.type || ""
    ) ===
    cleanText(
      replacementPending?.type || ""
    );

  const sameIndex =
    (
      !Number.isInteger(
        replacementPending?.index
      ) &&
      !Number.isInteger(
        currentPending?.index
      )
    ) ||
    (
      Number.isInteger(
        replacementPending?.index
      ) &&
      Number.isInteger(
        currentPending?.index
      ) &&
      replacementPending.index ===
        currentPending.index
    );

  const sameDetailIndex =
    (
      !Number.isInteger(
        replacementPending?.detailIndex
      ) &&
      !Number.isInteger(
        currentPending?.detailIndex
      )
    ) ||
    (
      Number.isInteger(
        replacementPending?.detailIndex
      ) &&
      Number.isInteger(
        currentPending?.detailIndex
      ) &&
      replacementPending.detailIndex ===
        currentPending.detailIndex
    );

  const replacementCaptureMode =
    cleanText(
      replacementPending?.captureMode || ""
    );

  const currentCaptureMode =
    cleanText(
      currentPending?.captureMode || ""
    );

  const sameCaptureMode =
    !replacementCaptureMode ||
    !currentCaptureMode ||
    replacementCaptureMode ===
      currentCaptureMode;

  const redirectGuidedConstructionCandidateState =
  guidedConstructionActive
    ? {
        ...structuredClone(state),

        pending:
          structuredClone(
            replacementPending
          ),

        componentInstructionalFinding:
          null,
      }
    : null;

const redirectGuidedConstructionCandidateLocation =
  redirectGuidedConstructionCandidateState
    ? buildGuidedConstructionInstructionalLocation(
        redirectGuidedConstructionCandidateState
      )
    : null;

  const currentGuidedConstructionLocation =
  guidedConstructionActive
    ? (
        currentPending
          ?.guidedConstructionLocation
          ?.locationEstablished === true
          ? structuredClone(
              currentPending
                .guidedConstructionLocation
            )
          : buildGuidedConstructionInstructionalLocation(
              state
            )
      )
    : null;

const sameGuidedConstructionLocation =
  guidedConstructionActive &&
  currentGuidedConstructionLocation
    ?.locationEstablished === true &&
  redirectGuidedConstructionCandidateLocation
    ?.locationEstablished === true &&
  isSameGuidedConstructionInstructionalLocation(
    currentGuidedConstructionLocation,
    redirectGuidedConstructionCandidateLocation
  );
  
  const sameRawPendingLocation =
  Boolean(
    currentPending &&
    samePendingType &&
    sameIndex &&
    sameDetailIndex &&
    sameCaptureMode
  );

const sameExactLocation =
  guidedConstructionActive
    ? sameGuidedConstructionLocation
    : sameRawPendingLocation;

  if (sameExactLocation) {
    return {
      artifactType:
        "redirectNavigationCommit",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationCommit",

      commitStatus:
        "sameLocation",

      committed:
        true,

      committedState:
        structuredClone(state),

      resolvedTarget:
        preparation?.resolvedTarget
          ? structuredClone(
              preparation.resolvedTarget
            )
          : null,

      governance: {
        deterministicCommit:
          true,

        destructiveResetApplied:
          false,

        canonicalFramePreserved:
          true,
      },
    };
  }

  // --------------------------------------------------
  // BUILD COMPLETE CANDIDATE FIRST
  //
  // Nothing in the source state has been cleared yet.
  // --------------------------------------------------

  const candidateState =
    structuredClone(state);

  candidateState.pending =
    structuredClone(
      replacementPending
    );

  // --------------------------------------------------
  // INVALIDATE LOCATION-OWNED ARTIFACTS
  //
  // These artifacts describe the previous instructional
  // location and must be recomputed at the new target.
  //
  // Do not clear Frame content, assignment context,
  // transcript, interactionMode, or other durable
  // session state.
  // --------------------------------------------------

  delete candidateState
    .observationReport;

  delete candidateState
    .instructionalAssessment;

  delete candidateState
    .componentInstructionalFinding;

  delete candidateState
    .instructionalSituation;

  delete candidateState
    .instructionalContractSelection;

  delete candidateState
    .progressionAuthorization;

  // --------------------------------------------------
  // FINAL VERIFICATION
  // --------------------------------------------------

  const candidateVerified =
    candidateState?.pending &&
    typeof candidateState.pending ===
      "object" &&
    cleanText(
      candidateState.pending.type || ""
    );

  if (!candidateVerified) {
    return {
      artifactType:
        "redirectNavigationCommit",

      version:
        "1.0",

      source:
        "deterministicRedirectNavigationCommit",

      commitStatus:
        "failed",

      committed:
        false,

      committedState:
        null,

      reason:
        "candidateCommitVerificationFailed",
    };
  }

  return {
    artifactType:
      "redirectNavigationCommit",

    version:
      "1.0",

    source:
      "deterministicRedirectNavigationCommit",

    commitStatus:
      "committed",

    committed:
      true,

    committedState:
      candidateState,

    resolvedTarget:
      preparation?.resolvedTarget
        ? structuredClone(
            preparation.resolvedTarget
          )
        : null,

    governance: {
      deterministicCommit:
        true,

      atomic:
        true,

      canonicalFramePreserved:
        true,

      oldLocationArtifactsInvalidated:
        true,
    },
  };
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
        clarificationCount: 0,
    },
  };
}

  if (!s.frameMeta.assignmentContext) {
    s.frameMeta.assignmentContext = {
      raw: "",
      understanding: "",
      confidence: "low",
      clarificationCount: 0,
    };
  }

    const activeRedirectNavigationBoundary =
    s?.redirectNavigationBoundary &&
    typeof s
      .redirectNavigationBoundary ===
      "object"
      ? s.redirectNavigationBoundary
      : null;

  if (
    activeRedirectNavigationBoundary
      ?.status ===
      "clarificationRequired"
  ) {
    const clarificationInterpretation =
      await interpretRedirectClarificationResolution(
        s,
        msg,
        activeRedirectNavigationBoundary
      );

    const clarificationValidation =
      buildRedirectValidation(
        s,
        clarificationInterpretation
      );

    const clarificationPreparation =
      buildRedirectNavigationPreparation(
        s,
        clarificationInterpretation,
        clarificationValidation
      );

    const clarificationCommit =
      buildRedirectNavigationCommit(
        s,
        clarificationPreparation
      );

    if (
  clarificationCommit
    ?.committed === true &&
  clarificationCommit
    ?.committedState &&
  typeof clarificationCommit
    .committedState ===
    "object"
) {
  const resolvedState =
    structuredClone(
      clarificationCommit
        .committedState
    );

  delete resolvedState
    .redirectNavigationBoundary;

  if (
    clarificationCommit
      ?.commitStatus ===
      "committed"
  ) {
    resolvedState.redirectNavigationOutcome = {
      artifactType:
        "redirectNavigationOutcome",

      version:
        "1.0",

      status:
        "committed",

      resolvedTarget:
        clarificationCommit
          ?.resolvedTarget
          ? structuredClone(
              clarificationCommit
                .resolvedTarget
            )
          : null,
    };
  }

  return resolvedState;
}

    const unresolvedState =
      structuredClone(s);

    unresolvedState
      .redirectNavigationBoundary = {
      artifactType:
        "redirectNavigationBoundary",

      version:
        "1.0",

      status:
        clarificationValidation
          ?.validationStatus ===
          "notAuthorized"
            ? "notAuthorized"
            : "clarificationRequired",

      interpretation:
        structuredClone(
          clarificationInterpretation
        ),

      validation:
        structuredClone(
          clarificationValidation
        ),
    };

    return unresolvedState;
  }

  const redirectEligibility =
    buildRedirectInterpretationEligibility(
      s,
      msg
    );

  const redirectInterpretation =
    redirectEligibility
      ?.eligible === true
      ? await interpretRedirectIntent(
          s,
          msg
        )
      : null;

    const redirectValidation =
    buildRedirectValidation(
      s,
      redirectInterpretation
    );

  const redirectNavigationPreparation =
    buildRedirectNavigationPreparation(
      s,
      redirectInterpretation,
      redirectValidation
    );

    const redirectNavigationCommit =
    buildRedirectNavigationCommit(
      s,
      redirectNavigationPreparation
    );

  if (
  redirectNavigationCommit
    ?.committed === true &&
  redirectNavigationCommit
    ?.committedState &&
  typeof redirectNavigationCommit
    .committedState === "object"
) {
  const committedState =
    structuredClone(
      redirectNavigationCommit
        .committedState
    );

  if (
    redirectNavigationCommit
      ?.commitStatus ===
      "committed"
  ) {
    committedState.redirectNavigationOutcome = {
      artifactType:
        "redirectNavigationOutcome",

      version:
        "1.0",

      status:
        "committed",

      resolvedTarget:
        redirectNavigationCommit
          ?.resolvedTarget
          ? structuredClone(
              redirectNavigationCommit
                .resolvedTarget
            )
          : null,
    };
  }

  return committedState;
}

    const redirectValidationStatus =
    cleanText(
      redirectValidation
        ?.validationStatus || ""
    );

  if (
    redirectValidationStatus ===
      "clarificationRequired" ||
    redirectValidationStatus ===
      "notAuthorized"
  ) {
    const boundaryState =
      structuredClone(s);

    boundaryState.redirectNavigationBoundary = {
      artifactType:
        "redirectNavigationBoundary",

      version:
        "1.0",

      status:
        redirectValidationStatus,

      interpretation:
        redirectInterpretation
          ? structuredClone(
              redirectInterpretation
            )
          : null,

      validation:
        structuredClone(
          redirectValidation
        ),
    };

    return boundaryState;
  }
  
const endpointResumeObservation =
  await getGuidedConstructionEndpointResumeObservation({
    state:
      s,

    message:
      msg,
  });

const endpointResumption =
  resumeGuidedConstructionAdditionalSupportEndpoint(
    s,
    endpointResumeObservation
  );

if (
  endpointResumption?.resumed === true
) {
  return s;
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

// Store the current governed Observation Report so it
// remains available to downstream instructional layers.
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
// control progression. For governed Frame components in the
// current scope, its refreshed result supports deterministic
// contract selection and governed communication.

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

  const {
  soWhatValidation,
  instructionalFinding,
  progressionAuthorization,
  capturedSoWhat,
} =
  await applySoWhatCapture(
    s,
    currentSoWhat,
    {
      captureMode:
        "strengthen",
    }
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
    capturedSoWhat;

   s.pending = {
  type:
    "strengthenComponentComplete",

  component:
    "soWhat",

  componentLabel:
    "So What",

  completedWork:
    capturedSoWhat,

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
  const instructionalContract =
    s?.instructionalContractSelection
      ?.selectedContract ||
    null;

  s.pending = {
    type:
      "collectAnotherDetail",

    index:
      i,

    detailIndex:
      arr.length,

    captureMode:
      "required",

    instructionalFinding,

    instructionalContract:
      instructionalContract
        ? structuredClone(
            instructionalContract
          )
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
  
  const {
  detailValidation,
  instructionalFinding,
  progressionAuthorization,
  capturedDetail,
} =
  await applyEssentialDetailCapture(
    s,
    msg,
    {
      index:
        idx,

      detailIndex:
        detailIndex,

      captureMode:
        "optionalDirectEntry",
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
  
  s.frame.details[idx] = [
    ...arr,
    capturedDetail,
];
  
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

const currentDetailIndex =
  s.frame.details[idx].length;

const captureMode =
  currentDetailIndex < 2
    ? "required"
    : "optional";

const {
  detailValidation,
  instructionalFinding,
  progressionAuthorization,
  capturedDetail,
} =
  await applyEssentialDetailCapture(
    s,
    msg,
    {
      index:
        idx,

      detailIndex:
        currentDetailIndex,

      captureMode,
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
    
  s.frame.details[idx] = [
    ...s.frame.details[idx],
    capturedDetail,
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
    const rawDetail =
    cleanText(msg);

  const priorRelationshipRepairActive =
  s?.pending
    ?.instructionalContract
    ?.contractId ===
    "ED-RNR-001";

  const originalAttemptedDetail =
    priorRelationshipRepairActive
      ? correctObviousStudentSpelling(
          s?.pending
            ?.instructionalFinding
            ?.evidence
            ?.attemptedDetail || ""
      )
    : "";
    
  const observationReport =
    s?.observationReport &&
    typeof s.observationReport ===
      "object"
      ? s.observationReport
      : null;

  const componentContribution =
    observationReport
      ?.componentContribution &&
    typeof observationReport
      .componentContribution ===
      "object"
      ? observationReport
          .componentContribution
      : null;

  const interactionOnlyCategories =
    new Set([
      "uncertaintyExpression",
      "clarificationRequest",
      "answerSeeking",
      "frustrationExpression",
      "refusal",
      "offTaskShift",
    ]);

  const interactionObservationPresent =
    Array.isArray(
      observationReport?.observations
    ) &&
    observationReport.observations.some(
      (observation) =>
        interactionOnlyCategories.has(
          cleanText(
            observation?.category || ""
          )
        )
    );

  const observedContributionText =
    componentContribution
      ?.observed === true
      ? cleanText(
          componentContribution
            ?.evidenceText || ""
        )
      : "";

  const text =
    correctObviousStudentSpelling(
      interactionObservationPresent &&
      observedContributionText
        ? observedContributionText
        : rawDetail
  );
    
  const currentMainIdea =
    getIdeaList(s)[index] || "";

  const detailValidation =
    await validateEssentialDetailResponseGoverned(
      text,
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
          rawDetail,

        displayAttemptedDetail:
          correctObviousStudentSpelling(
            rawDetail
          ),
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
      text,

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

  // --------------------------------------------------
// GUIDED CONSTRUCTION — ESSENTIAL DETAIL CONTINUATION
// --------------------------------------------------

const activeGuidedConstruction =
  getActiveGuidedConstructionContext(
    s
  );

if (
  activeGuidedConstruction?.active ===
    true &&
  activeGuidedConstruction
    ?.frameComponent ===
    "details"
) {
  await continueGuidedConstruction({
      state:
        s,

      response:
        text,

      componentValidation:
        detailValidation,

      finalRephraseUsed:
        false,
    });
}

   return {
    detailValidation,
    instructionalFinding,
    progressionAuthorization,

  capturedDetail:
    priorRelationshipRepairActive &&
    detailValidation.valid &&
    originalAttemptedDetail
      ? originalAttemptedDetail
      : text,
};
};

  async function applySoWhatCapture(
  s,
  msg,
  {
    captureMode = "revision",
    previousSoWhat = "",
  } = {}
) {
  const rawSoWhat =
    cleanText(msg);

  const observationReport =
    s?.observationReport &&
    typeof s.observationReport ===
      "object"
      ? s.observationReport
      : null;

  const componentContribution =
    observationReport
      ?.componentContribution &&
    typeof observationReport
      .componentContribution ===
      "object"
      ? observationReport
          .componentContribution
      : null;

  const interactionOnlyCategories =
    new Set([
      "uncertaintyExpression",
      "clarificationRequest",
      "answerSeeking",
      "frustrationExpression",
      "refusal",
      "offTaskShift",
    ]);

  const interactionObservationPresent =
    Array.isArray(
      observationReport?.observations
    ) &&
    observationReport.observations.some(
      (observation) =>
        interactionOnlyCategories.has(
          cleanText(
            observation?.category || ""
          )
        )
    );

  const observedContributionText =
    componentContribution
      ?.observed === true
      ? cleanText(
          componentContribution
            ?.evidenceText || ""
        )
      : "";

  const text =
    correctObviousStudentSpelling(
      interactionObservationPresent &&
      observedContributionText
        ? observedContributionText
        : rawSoWhat
  );

  const soWhatValidation =
    await validateSoWhatResponseGoverned(
      text,
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
          cleanText(previousSoWhat),

        attemptedSoWhat:
          rawSoWhat,

        captureMode,
      },
    }),

    synthesisState:
      soWhatValidation
        .synthesisState || null,

    validationSource:
      soWhatValidation
        .validationSource || null,

    captureMode,
  };

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
          "soWhat",

        expectedContractId:
          "SW-RTP-001",
      }
    );

  s.progressionAuthorization =
    structuredClone(
      progressionAuthorization
    );

  // --------------------------------------------------
  // GUIDED CONSTRUCTION — SO WHAT CONTINUATION
  // --------------------------------------------------

  const activeGuidedConstruction =
    getActiveGuidedConstructionContext(
      s
    );

  if (
    activeGuidedConstruction?.active ===
      true &&
    activeGuidedConstruction
      ?.frameComponent ===
      "soWhat"
  ) {
    await continueGuidedConstruction({
        state:
          s,

        response:
          text,

        componentValidation:
          soWhatValidation,

        finalRephraseUsed:
          false,
      });
  }

  return {
    soWhatValidation,
    instructionalFinding,
    progressionAuthorization,
    capturedSoWhat:
      text,
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
    capturedDetail,
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
    capturedDetail;
}

  // Return to the Detail confirmation checkpoint.
  s.pending = { type: "confirmDetails", index: idx };
  return s;
}

// --------------------------------------------------
// SO WHAT GOVERNED CONTINUATION
//
// This pending location is used only when an initial
// So What has not yet passed governed validation.
//
// The student's next response is treated as another
// attempt at the same So What—not as additional content
// appended to an accepted So What.
// --------------------------------------------------

if (s.pending?.type === "continueSoWhat") {
  const {
    soWhatValidation,
    instructionalFinding,
    progressionAuthorization,
    capturedSoWhat,
  } =
    await applySoWhatCapture(
      s,
      msg,
      {
        captureMode:
          "initial",
      }
    );

  if (
    !soWhatValidation.valid ||
    progressionAuthorization
      ?.authorized !== true
  ) {
    const previousProgressiveSupportStage =
  s?.pending?.progressiveSupportStage;

s.pending =
  buildPendingWithGuidedConstructionPreservation(
    s,
    {
      type:
        "continueSoWhat",

      captureMode:
        "initial",

      instructionalFinding,

      progressiveSupportStage:
        previousProgressiveSupportStage,
    }
  );

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
    capturedSoWhat;

  s.pending = {
    type:
      "confirmSoWhat",
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

const {
  soWhatValidation,
  instructionalFinding,
  progressionAuthorization,
  capturedSoWhat,
} =
  await applySoWhatCapture(
    s,
    msg,
    {
      captureMode:
        "revision",

      previousSoWhat,
    }
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
      capturedSoWhat;
    
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
    const choice =
      normalized === "3" ||
      normalized.includes("both")
        ? "both"
        : normalized === "1" ||
          normalized.includes("frame")
          ? "frame"
          : normalized === "2" ||
            normalized.includes("transcript")
            ? "transcript"
            : null;
    
   if (!choice) {
  return s;
}

s.flags.exportChoice =
  choice;

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
    await applyIsAboutCapture(
      s,
      s.frame.keyTopic
        ? msg
        : parsed.isAbout
);
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
    const {
  detailValidation,
  instructionalFinding,
  progressionAuthorization,
  capturedDetail,
} =
  await applyEssentialDetailCapture(
    s,
    msg,
    {
      index:
        i,

      detailIndex:
        arr.length,

      captureMode:
        "required",
    }
  );

if (
  !detailValidation.valid ||
  progressionAuthorization
    ?.authorized !== true
) {
  s.pending = {
  type:
    "collectAnotherDetail",

  index:
    i,

  detailIndex:
    arr.length,

  captureMode:
    "required",

  instructionalFinding,
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
          capturedDetail,
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

const {
  soWhatValidation,
  instructionalFinding,
  progressionAuthorization,
  capturedSoWhat,
} =
  await applySoWhatCapture(
    s,
    msg,
    {
      captureMode:
        "initial",
    }
  );
    
if (
  !soWhatValidation.valid ||
  progressionAuthorization
    ?.authorized !== true
) {
  s.pending =
    buildPendingWithGuidedConstructionPreservation(
      s,
      {
        type:
          "continueSoWhat",
      }
    );

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
  capturedSoWhat;

s.pending = {
  type:
    "confirmSoWhat",
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
// /run ps
// /run core
//
// These commands do not modify the student's active Frame.
// ------------------------------------------------------
const componentTestCommandMap = {
  "/run redirect":
  "redirectNavigation",
  
  "/run ia":
    "isAbout",

  "/run mi":
    "mainIdeas",

  "/run ed":
    "essentialDetail",

  "/run sw":
    "soWhat",

  "/run ps":
    "progressiveSupport",

  "/run core":
    "evidenceState",

  "/run h1":
    "stage1FinalQuestion",
  
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

const additionalSupportEndpoint =
  state?.pending
    ?.guidedConstructionAdditionalSupportEndpointArtifact ||
  null;

const additionalSupportResponse =
  additionalSupportEndpoint
    ?.endpointStatus ===
    "established"
      ? `🧭 You've reached the point where another Kaw scaffold could start supplying the thinking for you.

Use one of the supports already available to you:

- your notes
- your source materials
- your assignment materials
- your teacher

When you're ready, we'll return to this same spot.

Are you ready to check one of those supports before we continue?`
      : null;

const instructionalResponse =
  !additionalSupportResponse &&
  instructionalActivation
    ? await getInstructionalResponse(
        instructionalActivation
      )
    : null;

// A selected Kaw 2.5 Instructional Contract remains the
// instructional authority unless Guided Construction has
// deterministically reached its governed additional-
// support endpoint.
if (
  instructionalActivation &&
  !additionalSupportResponse &&
  !instructionalResponse
) {
  throw new Error(
    "Governed instructional communication failed."
  );
}

  const redirectNavigationBoundary =
  state?.redirectNavigationBoundary &&
  typeof state
    .redirectNavigationBoundary ===
    "object"
    ? state.redirectNavigationBoundary
    : null;

const normalNextQ =
  additionalSupportResponse ||
  (
    instructionalActivation
      ? instructionalResponse
      : computeNextQuestion(state)
  );

let redirectBoundaryResponse =
  null;

if (
  redirectNavigationBoundary
    ?.status ===
    "clarificationRequired"
) {
  const requestedComponent =
    cleanText(
      redirectNavigationBoundary
        ?.interpretation
        ?.requestedTarget
        ?.component || ""
    );

  const resolvedTarget =
    redirectNavigationBoundary
      ?.validation
      ?.resolvedTarget || null;

  if (
    requestedComponent ===
      "details" &&
    Number.isInteger(
      resolvedTarget?.mainIdeaIndex
    )
  ) {
    redirectBoundaryResponse =
      "Which Essential Detail do you want to work on?";
  } else if (
    requestedComponent ===
      "details"
  ) {
    redirectBoundaryResponse =
      "Which Main Idea has the Essential Detail you want to work on?";
  } else if (
    requestedComponent ===
      "mainIdeas"
  ) {
    redirectBoundaryResponse =
      "Which Main Idea do you want to work on?";
  } else {
    redirectBoundaryResponse =
      "Which part of your Frame do you want to work on: Is About, a Main Idea, an Essential Detail, or So What?";
  }
}

if (
  redirectNavigationBoundary
    ?.status ===
    "notAuthorized"
) {
  redirectBoundaryResponse =
    `I understand that you want to move to a different part of your Frame, but that part isn't available to work on right now.\n\n${normalNextQ}`;
}

  const redirectNavigationOutcome =
  state?.redirectNavigationOutcome &&
  typeof state
    .redirectNavigationOutcome ===
    "object"
    ? state.redirectNavigationOutcome
    : null;

let redirectSuccessAcknowledgment =
  null;

if (
  redirectNavigationOutcome
    ?.status ===
    "committed"
) {
  const resolvedTarget =
    redirectNavigationOutcome
      ?.resolvedTarget || {};

  const operation =
  cleanText(
    resolvedTarget?.operation || ""
  );

const pendingType =
  cleanText(
    resolvedTarget?.pendingType || ""
  );
  
  const component =
    cleanText(
      resolvedTarget?.component || ""
    );

   if (
  operation ===
    "declineCurrentPath"
) {
  if (
    pendingType ===
      "offerAnotherMainIdea"
  ) {
    redirectSuccessAcknowledgment =
      "That's okay — you can keep the Main Ideas you have.";
  } else if (
    pendingType ===
      "offerAnotherDetail" ||
    pendingType ===
      "collectAnotherDetail"
  ) {
    redirectSuccessAcknowledgment =
      "That's okay — you can keep the Essential Details you have for this Main Idea.";
  }
} else if (
  component === "isAbout"
) {
    redirectSuccessAcknowledgment =
      "Sure — let's work on your Is About.";
  } else if (
    component === "mainIdeas" &&
    Number.isInteger(
      resolvedTarget?.mainIdeaIndex
    )
  ) {
    redirectSuccessAcknowledgment =
      `Sure — let's work on Main Idea ${resolvedTarget.mainIdeaIndex + 1}.`;
  } else if (
    component === "details" &&
    Number.isInteger(
      resolvedTarget?.mainIdeaIndex
    ) &&
    Number.isInteger(
      resolvedTarget?.detailIndex
    )
  ) {
    redirectSuccessAcknowledgment =
      `Sure — let's work on Essential Detail ${resolvedTarget.detailIndex + 1} for Main Idea ${resolvedTarget.mainIdeaIndex + 1}.`;
  } else if (
    component === "soWhat"
  ) {
    redirectSuccessAcknowledgment =
      "Sure — let's work on your So What.";
  }
}

const nextQ =
  redirectBoundaryResponse ||
  (
    redirectSuccessAcknowledgment
      ? `${redirectSuccessAcknowledgment}\n\n${normalNextQ}`
      : normalNextQ
  );

delete state
  .redirectNavigationOutcome;

if (
  redirectNavigationBoundary
    ?.status ===
    "notAuthorized"
) {
  delete state
    .redirectNavigationBoundary;
}
    
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
