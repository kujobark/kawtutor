import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      return res.status(405).json({ error: "Method not allowed" });
    }

    return res.status(200).json({
      status: "Tutor function is running clean.",
    });

  } catch (error) {
    console.error("Tutor error:", error);
    return res.status(500).json({ error: "Internal server error" });
  }
}
