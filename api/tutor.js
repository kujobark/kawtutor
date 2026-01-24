import OpenAI from "openai";
import { SYSTEM_PROMPT_STEP_7 } from "../lib/systemPrompt.js";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// For demo + Wix Preview, allow all origins.
// If you want to lock it down later, we can.
const ALLOWED_ORIGIN = "*";

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

export default async function handler(req, res) {
  setCors(res);

  // 1) Preflight support (THIS fixes the Wix CORS block)
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  // 2) Reject anything that's not POST
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  try {
    const { message = "" } = req.body || {};
    const trimmed = String(message).trim();

    if (!trimmed) {
      return res.status(400).json({ error: "Missing 'message' in body" });
    }

    // Optional safety layer (keep it simple for now)
    const safety = await classifyMessage(trimmed);
    if (safety?.flagged) {
      const reply =
        SAFETY_RESPONSES?.[safety.category] ||
        "I can’t help with that, but I can help you reframe your question in a safe way.";
      return res.status(200).json({
        reply,
        flagged: true,
        flagCategory: safety.category || "unknown",
        severity: safety.severity || "unknown",
        safetyMode: true,
      });
    }

    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT_STEP_7 },
        { role: "user", content: trimmed },
      ],
      temperature: 0.4,
    });

    const reply =
      completion?.choices?.[0]?.message?.content?.trim() ||
      "Could you say a bit more about what you're thinking?";

    return res.status(200).json({
      reply,
      flagged: false,
      flagCategory: "",
      severity: "",
      safetyMode: false,
    });
  } catch (err) {
    return res.status(500).json({
      error: "Server error",
      details: String(err?.message || err),
    });
  }
}
