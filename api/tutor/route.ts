import { NextResponse } from "next/server";
import systemPrompt from "@/lib/systemprompt";
import safetyCheck from "@/lib/safetycheck";

export async function POST(req: Request) {
  const body = await req.json();

  // basic sanity check
  if (!body || !body.message) {
    return NextResponse.json({ reply: "Missing message" }, { status: 400 });
  }

  // placeholder response for now (Wix just needs JSON)
  return NextResponse.json({
    reply: "Endpoint reachable. Ready for GPT wiring."
  });
}
