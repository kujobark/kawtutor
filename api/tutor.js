import OpenAI from "openai";
import { SYSTEM_PROMPT_STEP_7 } from "../lib/systemPrompt.js";
import { SAFETY_RESPONSES } from "../lib/safetyResponses.js";
import { classifyMessage } from "../lib/safetyCheck.js";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const sessions = new Map();
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
  res.status(200).json({ reply: "Hello fro Kaw"});
    }

    session.safetyMode = false;
    
    const completion = await client.chat.completions.create({
      model: "gpt-4o-mini", // Corrected model name from 'gpt-4.1-mini' based on standard naming or use specific version from PDF if available
      messages: [
        { role: "system", content: SYSTEM_PROMPT_STEP_7 },
        { role: "user", content: message }
      ],
      temperature: 0.4
    });

    const reply = completion?.choices?.[0]?.message?.content?.trim() || "Could you say a bit more about what you're thinking?";

    return res.status(200).json({
      reply,
      flagged: false,
      flagCategory: "",
      severity: "",
      safetyMode: false
    });

  } catch (err) {
    return res.status(500).json({
      error: "Server error",
      details: String(err?.message || err)
    });
  }
}