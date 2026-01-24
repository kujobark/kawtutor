import OpenAI from "openai";
import { SYSTEM_PROMPT_STEP_7 } from "../lib/systemPrompt.js";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// In-memory sessions (will reset on cold starts / redeploys)
const sessions = new Map();

// If you want to lock this down later, replace "*" with your Wix domain.
// Example: "https://www.kawtutor.com"
const ALLOWED_ORIGIN = "*";

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", ALLOWED_ORIGIN);
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

function getSession(sessionId) {
  if (!sessions.has(sessionId)) sessions.set(sessionId, { safetyMode: false });
  return sessions.get(sessionId);
}

export default async function handler(req, res) {
  setCors(res);

  // Preflight
  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  try {
    const body = req.body || {};
    const message = (body.message || "").toString().trim();
    const sessionId = (body.sessionId || "default").toString();

    if (!message) {
      return res.status(400).json({
        reply: "Please send a message.",
        flagged: false,
        flagCategory: "",
        severity: "",
        safetyMode: false,
      });
    }

    const session = getSession(sessionId);

    // --- Safety check ---
    const safety = await classifyMessage(message);
    // Expected safety shape (typical): { flagged, category, severity }
    // If yours differs, tell me what it returns and I'll align it.
    if (safety?.flagged) {
      session.safetyMode = true;

      const category = safety.category || "general";
      const response =
        SAFETY_RESPONSES?.[category] ||
        SAFETY_RESPONSES?.general ||
        "I can’t help with that, but I *can* help you in a safer direction. What are you trying to accomplish?";

      return res.status(200).json({
        reply: response,
        flagged: true,
        flagCategory: category,
        severity: safety.severity || "",
        safetyMode: true,
      });
    }

    session.safetyMode = false;

    // --- OpenAI call ---
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: SYSTEM_PROMPT_STEP_7 },
        { role: "user", content: message },
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
