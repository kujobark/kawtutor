import OpenAI from "openai";
import { SYSTEM_PROMPT_STEP_7 } from "../lib/systemPrompt.js";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ---- CORS CONFIG ----
// Allow Wix Preview + Published sites
const ALLOWED_ORIGIN = "*";

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

export default async function handler(req, res) {
  setCors(res);

  // ---- 1️⃣ Handle CORS preflight (REQUIRED for Wix) ----
  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  // ---- 2️⃣ Only allow POST ----
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  try {
    // ---- 3️⃣ Read input ----
    const { message = "" } = req.body || {};
    const trimmed = String(message).trim();

    if (!trimmed) {
      return res.status(400).json({
        error: "Missing 'message' in request body",
      });
    }

    // ---- 4️⃣ Safety check ----
    const safety = await classifyMessage(trimmed);

    if (safety?.flagged) {
      return res.status(200).json({
        reply:
          SAFETY_RESPONSES[safety.category] ||
          "Let’s pause for a moment and try approaching this in a different way.",
        flagged: true,
        flagCategory: safety.category || "unknown",
        severity: safety.severity || "low",
        safetyMode: true,
      });
    }

    // ---- 5️⃣ OpenAI call ----
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.4,
      messages: [
        { role: "system", content: SYSTEM_PROMPT_STEP_7 },
        { role: "user", content: trimmed },
      ],
    });

    const reply =
      completion?.choices?.[0]?.message?.content?.trim() ||
      "Could you say a bit more about what you’re thinking?";

    // ---- 6️⃣ Return response to Wix ----
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
