export function classifyMessage(message) {
  const text = (message || "").toLowerCase();
  
  const selfHarmSignals = ["kill myself", "hurt myself", "end my life", "suicide"];
  if (selfHarmSignals.some(s => text.includes(s))) {
    return { flagged: true, severity: "A", flagCategory: "SELF_HARM" };
  }

  const violenceSignals = ["bring a weapon", "shoot", "stab", "hurt someone", "kill him", "kill her"];
  if (violenceSignals.some(s => text.includes(s))) {
    return { flagged: true, severity: "A", flagCategory: "VIOLENCE" };
  }

  const severeNegSignals = ["i'm worthless", "i am worthless", "i hate myself", "nothing matters", "i want to disappear"];
  if (severeNegSignals.some(s => text.includes(s))) {
    return { flagged: true, severity: "B", flagCategory: "SEVERE_NEGATIVE_SELF_TALK" };
  }

  const bullyingSignals = ["you're stupid", "i hate you", "go die"];
  if (bullyingSignals.some(s => text.includes(s))) {
    return { flagged: true, severity: "B", flagCategory: "BULLYING" };
  }

  return { flagged: false, severity: "", flagCategory: "" };
}