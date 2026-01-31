import { NextResponse } from "next/server";

/**
 * Kaw Companion — Frame-Driven Socratic Tutor (API)
 * Key upgrades:
 * - 3-part sufficiency check for "Key Topic → is about"
 * - prevents looping on Key Topic
 * - JSON-only responses for Wix
 */

type Stage = "FOCUS_TOPIC" | "MAIN_IDEAS" | "DETAILS" | "SO_WHAT" | "NEXT";

type KawRequest = {
  stage: Stage;

  // Student text from Wix
  student_last: string;

  // Optional, maintained client-side (Wix) and sent each turn
  known?: {
    keyTopic?: string | null;
    isAbout?: string | null;
    reaskCounts?: Record<string, number>; // e.g. { focus: 1 }
  };

  // Optional: what class/content area (helps examples stay relevant)
  context?: {
    subject?: string; // "Civics", "ELA", "Biology"
    grade?: string;
  };
};

type KawResponse = {
  next_stage: Stage;
  assistant_message: string;
  frames: string[];
  quick_choices: string[];
  field_updates: Record<string, any>;
  needs: string[];
};

const MAX_REASK = 2;

/** --- 1) Instructional sufficiency check (3-part) --- */
function sufficiencyCheck(keyTopic?: string | null, isAbout?: string | null) {
  const kt = (keyTopic ?? "").trim();
  const ia = (isAbout ?? "").trim();

  const hasClearLabel =
    kt.length >= 3 &&
    kt.length <= 80 &&
    !/^(topic|stuff|things|history|government|science|math)$/i.test(kt);

  const hasMeaningfulDirection =
    ia.length >= 8 &&
    ia.length <= 220 &&
    !/^(it is about|about|this is about)?\s*(stuff|things|a topic|government|history|science)\.?$/i.test(
      ia
    );

  const canLeadToMainIdeas =
    // simple heuristic: contains a relationship/process cue or a describable angle
    /(because|so that|so|how|why|to|caused by|leads to|results in|changed|influenced|prevents|helps|shows|explains|compares|contrasts)/i.test(
      ia
    ) || ia.split(" ").length >= 8;

  const instructionallySufficient = hasClearLabel && hasMeaningfulDirection && canLeadToMainIdeas;

  return {
    instructionallySufficient,
    hasClearLabel,
    hasMeaningfulDirection,
    canLeadToMainIdeas,
  };
}

/** --- 2) Very light safety gate to avoid confusing model pauses --- */
function quickSafetyGate(text: string) {
  const t = (text || "").toLowerCase();
  const selfHarm = /(kill myself|suicide|self harm|hurt myself)/i.test(t);
  const threat = /(shoot up|bomb|kill them|stab them)/i.test(t);
  const minorSex = /(child porn|underage sex|minor nude)/i.test(t);
  const block = selfHarm || threat || minorSex;
  return { block, reason: selfHarm ? "self_harm" : threat ? "threat" : minorSex ? "minor_sex" : null };
}

/** --- 3) Helper: increment reask count safely --- */
function bumpReask(known: KawRequest["known"], key: string) {
  const counts = { ...(known?.reaskCounts ?? {}) };
  counts[key] = (counts[key] ?? 0) + 1;
  return counts;
}

/** --- 4) OpenAI call (fill in with your existing client) ---
 * IMPORTANT: Replace this stub with your actual call.
 * The rest of the logic works regardless of Responses API / Chat Completions.
 */
async function callModel(_payload: any): Promise<KawResponse> {
  // TODO: Replace with your OpenAI call.
  // Must return KawResponse JSON.
  return {
    next_stage: "FOCUS_TOPIC",
    assistant_message: "Model not wired yet.",
    frames: [],
    quick_choices: [],
    field_updates: {},
    needs: ["keyTopic", "isAbout"],
  };
}

/** --- 5) System prompt & schema constraint --- */
const SYSTEM_PROMPT = `
You are Kaw Companion, a frame-driven Socratic tutor.
Goal: help students complete the Framing Routine. You are NOT grading; you are like a teacher circulating.
Key rule: Advance once the student's Key Topic → "is about" statement is instructionally sufficient.
Instructional sufficiency = (1) clear label, (2) meaningful direction, (3) can lead to main ideas.
Ask ONE question per turn. Always provide 2–4 sentence frames and 3 quick choices.
Never loop on the same question more than twice. If stuck, scaffold in smaller parts.
Return ONLY valid JSON matching the response schema.
`;

export async function POST(req: Request) {
  const body = (await req.json()) as KawRequest;

  const safe = quickSafetyGate(body.student_last || "");
  if (safe.block) {
    const resp: KawResponse = {
      next_stage: body.stage,
      assistant_message:
        "I can’t help with that directly. If you’re in danger or thinking about harming yourself or someone else, please contact a trusted adult right now or call local emergency services. If you want, tell me the school-safe assignment you’re working on and I’ll help with the writing.",
      frames: [
        "I’m working on ______ and I need help with ______.",
        "A safer way to phrase my idea for class is ______."
      ],
      quick_choices: ["Switch topic", "Return to assignment", "Get help resources"],
      field_updates: {},
      needs: []
    };
    return NextResponse.json(resp);
  }

  // Pull known fields from client
  const known = body.known ?? {};
  const keyTopic = known.keyTopic ?? null;
  const isAbout = known.isAbout ?? null;

  // If we are in the focus step, apply sufficiency logic server-side
  if (body.stage === "FOCUS_TOPIC") {
    const check = sufficiencyCheck(keyTopic, isAbout);

    // ✅ If sufficient → advance immediately (don’t over-polish)
    if (check.instructionallySufficient) {
      const resp: KawResponse = {
        next_stage: "MAIN_IDEAS",
        assistant_message:
          "Nice — that Key Topic → “is about” statement is clear enough to move forward. Now, what are 2–3 main ideas someone needs to understand to explain your topic?",
        frames: [
          "One main idea is ______.",
          "Another main idea is ______.",
          "A third main idea is ______."
        ],
        quick_choices: ["I have 2 main ideas", "I have 3 main ideas", "I’m not sure yet"],
        field_updates: { focus_ready: true },
        needs: ["mainIdeas"]
      };
      return NextResponse.json(resp);
    }

    // 🔴 If not sufficient, scaffold WITHOUT looping endlessly
    const reaskCounts = bumpReask(known, "focus");
    const reasks = reaskCounts["focus"];

    // Case C: student stuck / missing pieces — split the move
    if (!check.hasClearLabel) {
      const resp: KawResponse = {
        next_stage: "FOCUS_TOPIC",
        assistant_message:
          reasks <= MAX_REASK
            ? "No worries — first just name your **Key Topic** (the title). What’s your topic called?"
            : "Let’s pick a clear Key Topic title so we can move forward. Which one fits best?",
        frames: [
          "My Key Topic is ______.",
          "The topic/title is ______."
        ],
        quick_choices: [
          "Three Branches of Government",
          "Checks and Balances",
          "Separation of Powers"
        ],
        field_updates: { reaskCounts },
        needs: ["keyTopic"]
      };
      return NextResponse.json(resp);
    }

    // Has label, but “is about” is weak → prompt the “is about” sentence
    if (!check.hasMeaningfulDirection || !check.canLeadToMainIdeas) {
      const resp: KawResponse = {
        next_stage: "FOCUS_TOPIC",
        assistant_message:
          reasks <= MAX_REASK
            ? `Great — your Key Topic is **${keyTopic}**. Now finish this: **“${keyTopic} is about…”**`
            : `You’ve got the Key Topic. Let’s lock in a workable “is about” so we can move on. Pick one to start, then you can tweak it.`,
        frames: [
          `${keyTopic} is about how ______.`,
          `${keyTopic} is about why ______ happens.`,
          `${keyTopic} is about the relationship between ______ and ______.`
        ],
        quick_choices: [
          "…how power is divided to prevent abuse",
          "…how each branch checks the others",
          "…why balance of power matters"
        ],
        field_updates: { reaskCounts },
        needs: ["isAbout"]
      };
      return NextResponse.json(resp);
    }
  }

  // For other stages, hand off to model (once you wire it)
  // (Still keep your routine constraints in the prompt/payload.)
  const modelPayload = {
    system: SYSTEM_PROMPT,
    input: {
      stage: body.stage,
      student_last: body.student_last,
      known,
      context: body.context ?? {}
    }
  };

  const modelResp = await callModel(modelPayload);
  return NextResponse.json(modelResp);
}
