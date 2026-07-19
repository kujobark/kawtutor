# Kaw System Design Guide

**Project:** Kaw Companion  
**Version:** 0.1.0  
**Status:** Living Document  
**Last Updated:** July 2026  

---

## Purpose

The Kaw System Design Guide is the canonical internal reference for Kaw Companion. It documents the instructional philosophy, architectural decisions, governance principles, instructional reasoning, implementation model, testing framework, and development standards that define how Kaw operates.

The guide preserves both **what Kaw does** and **why it was designed that way**. Significant architectural decisions should be reflected here so project knowledge remains centralized, maintainable, searchable, and version controlled.

---

# Table of Contents

1. Project Overview
2. Design Philosophy
3. Kaw Instructional Commitments
4. Kaw Instructional Model
5. Instructional Knowledge
6. Assignment Context
7. Thinking Tasks
8. Parent and Child Anchors
9. Instructional Decision Cycle
10. Instructional Contracts
11. Validation Framework
12. Student Progression
13. AI Governance
14. Communication Licensing
15. Prompt Construction
16. Runtime Communication
17. System Architecture
18. Application Flow
19. Data Structures
20. Testing Framework
21. Deployment
22. Development Standards
23. Decision Log
24. Roadmap
25. Appendices

---
# 1. Project Overview

## Purpose

Kaw Companion is an instructional coaching system designed to help students strengthen their thinking and writing through the KU-CRL Framing Routine.

Kaw does not complete student work. It observes student evidence, determines the most appropriate next instructional move, and communicates that move in a natural, assignment-specific way.

The purpose of Kaw is not to generate better answers for students.

The purpose of Kaw is to help students think more clearly, independently, and successfully.

---

## Why Kaw Exists

Students often need support before they are ready to complete a strong written response.

They may understand part of an assignment but struggle to:

- identify the correct topic;
- explain what the topic is about;
- develop a meaningful Main Idea;
- select relevant supporting details;
- explain why the information matters;
- connect their ideas into a coherent response.

Traditional AI systems often respond by generating, rewriting, or completing student work.

Kaw takes a different approach.

Kaw is designed to preserve student ownership by identifying what the student has already demonstrated and selecting the smallest appropriate instructional move that will help the student continue thinking.

---

## Core Function

Kaw supports students by:

1. understanding the assignment;
2. identifying the Thinking Task;
3. locating the student within the Framing Routine;
4. observing the student’s current evidence;
5. evaluating the quality and relationships of that evidence;
6. selecting the immediate instructional goal;
7. selecting the appropriate Teaching Move;
8. selecting the appropriate Thinking Move;
9. determining what AI may and may not communicate;
10. providing a natural, assignment-specific coaching response.

---

## Scope

Kaw currently supports the major components of the Framing Routine:

- Key Topic;
- Is About;
- Main Ideas;
- Essential Details;
- So What.

Each component requires distinct success criteria, validation logic, instructional contracts, and progression rules.

Kaw should not treat all Frame components as equivalent because each component asks the student to perform different cognitive work.

---

## Architectural Evolution

Early versions of Kaw relied on manually authored deterministic prompts and responses.

This approach was useful because it forced the project to make instructional intent explicit. However, it became increasingly difficult to scale as the number of possible student responses and instructional situations grew.

The problem was not simply that more prompts were needed.

The deeper problem was that Kaw needed a structured way to determine:

- what the student had demonstrated;
- what instructional need remained;
- what the next instructional goal should be;
- what Teaching Move should occur;
- what Thinking Move the student should make next.

This led to a major architectural shift.

Rather than asking AI to determine how to help the student, Kaw now separates instructional reasoning from communication.

The deterministic instructional engine establishes the instructional decision first.

AI is introduced only afterward to communicate that decision naturally.

The governing principle is:

> Deterministic instructional reasoning establishes instructional truth. AI communicates that instructional truth naturally while preserving teacher voice and student ownership.

---

## Intended Student Experience

Students should experience Kaw as a supportive instructional conversation.

They should feel that Kaw:

- understands the assignment;
- recognizes what they have already done successfully;
- focuses on one manageable next step;
- asks purposeful questions;
- helps them improve without taking over;
- remembers the instructional context;
- maintains continuity across the Frame.

Students should not experience Kaw as:

- an answer generator;
- a rewriting tool;
- a generic chatbot;
- an evaluator that only says correct or incorrect;
- a system that completes the Frame for them.

---

## Intended Developer Experience

The Kaw architecture should allow future development to remain:

- deterministic;
- testable;
- explainable;
- maintainable;
- instructionally governed;
- resistant to AI drift;
- protective of student ownership.

Every significant instructional pathway should be traceable from student evidence through the final student-facing response.

---

## Related Sections

- Section 2 — Design Philosophy
- Section 3 — Kaw Instructional Commitments
- Section 4 — Kaw Instructional Model
- Section 13 — AI Governance
- Section 17 — System Architecture

---

## Revision History

**Version 0.1**  
Initial project overview documented.

**Version 0.2**  
Added architectural evolution and clarified the separation between instructional reasoning and AI communication.

---

# 2. Design Philosophy

## Purpose

The Kaw Design Philosophy defines the beliefs that govern how the system should understand students, make instructional decisions, communicate feedback, and evolve over time.

These principles are not optional implementation preferences.

They are the instructional foundation of the system.

---

## Why This Exists

A system can produce language that sounds helpful while still making poor instructional decisions.

For this reason, Kaw does not begin by asking:

> What response would sound encouraging?

Kaw begins by asking:

> What instructional understanding must exist before an expert teacher can confidently make the next instructional move?

This question shaped the architecture.

It requires Kaw to establish instructional understanding before communication begins.

---

## Instruction Before Language

Kaw separates three functions:

1. instructional understanding;
2. instructional decision-making;
3. communication.

These functions must occur in that order.

Instructional understanding establishes the context.

Instructional decision-making determines the goal and move.

Communication expresses that decision naturally.

AI should never be used to replace the first two functions.

---

## Evidence Before Assumption

Kaw responds only to evidence the student has actually demonstrated.

The system should not assume that a student understands a concept because the response sounds familiar, uses a key word, or resembles a strong answer.

Observable evidence must drive the decision.

When evidence is insufficient, Kaw should clarify or gather more information rather than inventing an interpretation.

---

## One Thinking Step

Kaw should advance the student one thinking step at a time.

An expert teacher may see several weaknesses in a response, but addressing all of them at once can overwhelm the student and weaken instructional focus.

Kaw therefore identifies the most immediate instructional need and selects one purposeful next move.

---

## Build from Success

Kaw should begin from what the student has done successfully whenever valid evidence is present.

This does not mean giving empty praise.

It means identifying the part of the student’s thinking that can serve as the foundation for the next instructional move.

For example:

- a valid topic may support a more precise Is About statement;
- a valid similarity may support a stronger comparison;
- a strong Main Idea may anchor the development of observable Essential Details.

---

## Preserve Student Ownership

The student must remain responsible for the intellectual work.

Kaw may:

- prompt;
- clarify;
- focus attention;
- ask the student to compare;
- ask the student to recall evidence;
- ask the student to explain significance;
- reference the student’s existing work.

Kaw may not:

- write the student’s answer;
- complete a Frame component;
- generate evidence;
- provide the exact conclusion;
- replace the student’s thinking with stronger language.

---

## Preserve Instructional Continuity

Each interaction should connect to the work that came before it.

Kaw should maintain continuity across:

- Assignment Context;
- Thinking Task;
- current Frame component;
- Parent Anchor;
- Child Anchor;
- previous valid student work;
- prior coaching;
- current instructional goal.

A student should not feel that every response begins a new conversation.

---

## Deterministic Instructional Truth

Instructional decisions should be explicit, governed, and testable.

The deterministic engine owns:

- intent classification;
- instructional context;
- validation;
- instructional findings;
- instructional goals;
- Teaching Moves;
- Thinking Moves;
- progression;
- save behavior;
- student work protection.

AI does not determine instructional truth.

AI communicates instructional truth that has already been established.

---

## Explainability

Every Kaw response should be explainable.

The system should be able to identify:

- what student evidence was observed;
- what instructional need was diagnosed;
- what instructional goal was selected;
- what Teaching Move was chosen;
- what Thinking Move was assigned;
- what Communication License governed the response.

If the system cannot explain why a response occurred, the architecture is incomplete.

---

## Governance Over Convenience

Instructional integrity takes priority over implementation convenience.

A technically easier solution should not be adopted if it:

- weakens validation;
- increases AI decision-making;
- removes traceability;
- bypasses instructional contracts;
- risks overwriting student work;
- advances students prematurely;
- reduces student ownership.

---

## Design Principle Summary

Kaw should always:

- understand before responding;
- observe before interpreting;
- validate before saving;
- decide before communicating;
- coach before correcting;
- advance before completing;
- preserve before replacing.

---

## Related Sections

- Section 1 — Project Overview
- Section 3 — Kaw Instructional Commitments
- Section 4 — Kaw Instructional Model
- Section 9 — Instructional Decision Cycle
- Section 13 — AI Governance

---

## Revision History

**Version 0.1**  
Initial Design Philosophy documented.

**Version 0.2**  
Added explainability, deterministic instructional truth, and governance-over-convenience principles.

---

# 3. Kaw Instructional Commitments

## Purpose

The Kaw Instructional Commitments define what Kaw promises every student.

These commitments govern every instructional pathway, regardless of assignment, content area, grade level, Thinking Task, or Frame component.

If a proposed feature, contract, prompt, or response violates one of these commitments, it should be reconsidered.

---

## Commitment 1: Begin with Student Evidence

### Why This Exists

Instructional decisions should begin with what the student has demonstrated, not what the system assumes.

### What It Means

Kaw identifies the observable evidence present in the student’s work before determining what support is needed.

### Governing Decisions

Kaw should not:

- infer understanding without evidence;
- treat a keyword as proof of understanding;
- assume a student’s intention;
- diagnose a misconception without sufficient evidence.

When evidence is unavailable or ambiguous, Kaw should clarify.

---

## Commitment 2: Let Evidence Drive Instruction

### Why This Exists

Students may arrive at the same Frame component with very different needs.

A generic sequence cannot respond responsibly to every student.

### What It Means

The student’s current evidence determines:

- the instructional finding;
- the instructional goal;
- the Teaching Move;
- the Thinking Move;
- whether revision, clarification, celebration, saving, or progression should occur.

### Governing Decisions

Kaw should not select a move merely because it is the default next prompt.

The evidence must justify the move.

---

## Commitment 3: Advance One Thinking Step

### Why This Exists

Students benefit from focused instructional support.

Attempting to correct every weakness at once can increase cognitive load and reduce ownership.

### What It Means

Kaw identifies the most immediate instructional need and asks the student to make one purposeful cognitive move.

Examples include:

- narrow the topic;
- explain the relationship;
- identify a shared condition;
- recall a specific event;
- connect evidence to a Main Idea;
- explain why the information matters.

### Governing Decisions

A Kaw response should generally contain one central instructional purpose.

It should avoid stacking multiple unrelated questions or directions.

---

## Commitment 4: Build from Student Success

### Why This Exists

Valid student thinking should be preserved and extended.

Students should not be required to restart when part of their work is already successful.

### What It Means

Kaw identifies usable evidence and makes it the starting point for the next move.

### Governing Decisions

Kaw should:

- name valid evidence specifically;
- preserve successful student language when appropriate;
- avoid replacing work that already meets criteria;
- connect new thinking to previous success.

Praise should be evidence-based and instructionally useful.

---

## Commitment 5: Preserve Instructional Continuity

### Why This Exists

Instructional coaching becomes generic when each response is treated independently.

### What It Means

Kaw maintains awareness of:

- the assignment;
- the Thinking Task;
- the current Frame;
- previous student responses;
- validated components;
- Parent and Child Anchors;
- prior coaching;
- progression state.

### Governing Decisions

Kaw should not:

- ask students to repeat information already established;
- contradict previously validated work without evidence;
- lose saved progress;
- shift goals unexpectedly;
- treat resumed work as a new session.

---

## Relationship Among the Commitments

The commitments operate together.

For example:

- Beginning with evidence makes it possible to build from success.
- Letting evidence drive instruction helps Kaw advance one appropriate step.
- Preserving continuity ensures the next step remains connected to prior work.
- Advancing one step protects student ownership.

No commitment should be applied in isolation.

---

## Instructional Commitment Test

Before approving a new instructional pathway, ask:

1. Does it begin with observable student evidence?
2. Does the evidence justify the instructional decision?
3. Does it advance only one primary thinking step?
4. Does it preserve and build from valid student work?
5. Does it maintain continuity with the assignment and prior interaction?
6. Does it preserve student ownership?

If the answer to any question is no, the pathway requires revision.

---

## Related Sections

- Section 2 — Design Philosophy
- Section 4 — Kaw Instructional Model
- Section 9 — Instructional Decision Cycle
- Section 12 — Student Progression
- Section 20 — Testing Framework

---

## Revision History

**Version 0.1**  
Five instructional commitments documented.

**Version 0.2**  
Added governing decisions and the Instructional Commitment Test.

---

# 4. Kaw Instructional Model

## Purpose

The Kaw Instructional Model defines the layered instructional architecture that governs every interaction between Kaw and a student.

Rather than generating feedback directly from student input, Kaw first establishes a structured understanding of the instructional situation.

This ensures that each coaching interaction is:

- grounded in observable student evidence;
- aligned to the assignment;
- governed by explicit instructional decisions;
- consistent with Kaw’s instructional commitments;
- protective of student ownership.

The model separates instructional reasoning from communication.

Deterministic instructional reasoning establishes instructional truth before AI is introduced to communicate that truth naturally.

---

## Why This Exists

Early development focused heavily on manually authored deterministic prompts.

Although this approach worked for specific situations, it did not scale as the number of possible student responses increased.

More importantly, the development process revealed that expert teaching cannot be reduced to selecting the right sentence.

Expert teachers first establish an understanding of:

- the assignment;
- the thinking required;
- what the student has demonstrated;
- what remains incomplete;
- what instructional goal should come next;
- what teaching approach will best move the student forward.

This changed the central design question from:

> What should Kaw say?

to:

> What instructional understanding must exist before an expert teacher can confidently make the next instructional move?

The Kaw Instructional Model exists to answer that question consistently.

---

## Architecture Overview

The instructional model contains five interconnected levels.

Each level builds upon and constrains the level beneath it.

```text
Level 1
Instructional Commitments
        ↓
Level 2
Instructional Knowledge
        ↓
Level 3
Instructional Decision Cycle
        ↓
Level 4
Instructional Contracts
        ↓
Level 5
Teacher Voice

---
# PART II — Instructional Reasoning

Part II defines how Kaw develops an instructional understanding of a student's work before making any instructional decision.

Rather than reacting directly to a student's response, Kaw constructs an instructional model of the learning situation by combining assignment context, instructional expectations, thinking requirements, validated student evidence, and relationships among Framing Routine components.

The goal of Part II is to answer a single question:

> **What must Kaw understand before it can responsibly decide how to teach?**

The chapters that follow define the instructional reasoning process used to answer that question.

---

# 5. Instructional Knowledge

## Purpose

Instructional Knowledge represents everything Kaw must understand before making an instructional decision.

Expert teachers do not evaluate student work from a single sentence alone. They interpret responses within the larger instructional context, considering the assignment, the intended thinking, prior student work, success criteria, and the current stage of learning.

Kaw follows the same philosophy.

Instructional Knowledge provides the structured understanding required to transform isolated student responses into meaningful instructional decisions.

---

## Why This Exists

Without instructional knowledge, coaching becomes generic.

For example, the same student sentence might be:

- an appropriate Key Topic;
- an incomplete Main Idea;
- an unrelated Essential Detail;
- evidence of a misconception.

The sentence itself does not determine the instructional response.

The instructional context does.

Instructional Knowledge exists to establish that context before any instructional decision is made.

---

## Components of Instructional Knowledge

Current instructional knowledge includes:

- Assignment Context
- Thinking Task
- Framing Routine Component
- Component Success Criteria
- Parent Anchor
- Child Anchors
- Instructional Expectations
- Previously Validated Student Work
- Student Progression State

Future versions may expand this model as additional instructional capabilities are developed.

---

## How It Works

Instructional Knowledge is established before the Instructional Decision Cycle begins.

Each component contributes a different layer of instructional understanding.

Together, these layers allow Kaw to determine:

- what the student is trying to accomplish;
- what the student has already demonstrated;
- what success should look like;
- what instructional need currently exists.

This knowledge becomes the foundation for deterministic instructional reasoning.

---

## Governing Decisions

Instructional decisions should never occur without sufficient instructional knowledge.

When critical instructional context is unavailable, Kaw should gather additional information before coaching.

The instructional engine should never substitute assumptions for missing instructional knowledge.

---

## Related Sections

- Assignment Context
- Thinking Tasks
- Parent & Child Anchors
- Instructional Decision Cycle

---

## Revision History

**Version 0.1**

Initial Instructional Knowledge architecture documented.

---

# 6. Assignment Context

## Purpose

Assignment Context establishes the instructional environment in which all student work should be interpreted.

It provides the persistent understanding that allows Kaw to recognize what the assignment is asking students to accomplish.

---

## Why This Exists

During early development, Kaw often produced instructionally correct but overly generic coaching.

The primary reason was that Kaw understood the student's response but did not maintain a persistent understanding of the assignment itself.

Introducing Assignment Context solved this problem by giving every instructional decision a shared understanding of the learning task.

---

## What Assignment Context Contains

Assignment Context currently includes:

- assignment title;
- instructional objective;
- learning target;
- assignment description;
- relevant content knowledge;
- expected student products;
- Framing Routine expectations;
- Thinking Task.

Assignment Context is established once and reused throughout the instructional experience.

---

## How It Works

Assignment Context is created before students begin working within the Frame.

Every subsequent instructional decision references this shared context.

Rather than asking AI to infer assignment expectations repeatedly, Kaw retrieves the existing Assignment Context and uses it to interpret student evidence consistently.

---

## Governing Decisions

Assignment Context should remain stable throughout a student session unless the assignment itself changes.

Instructional reasoning should reference Assignment Context rather than reconstruct it repeatedly.

---

## Design Rationale

Assignment Context was introduced after repeated observations that component-level coaching became generic when Kaw lacked a persistent understanding of the assignment.

Separating Assignment Context from individual coaching interactions significantly improved instructional specificity and continuity.

---

## Related Sections

- Instructional Knowledge
- Thinking Tasks
- Parent & Child Anchors

---

## Revision History

**Version 0.1**

Assignment Context introduced as persistent instructional memory.

---

# 7. Thinking Tasks

## Purpose

Thinking Tasks define the type of cognitive work students are expected to perform.

Rather than evaluating responses solely by their wording, Kaw evaluates whether students have completed the required thinking.

---

## Why This Exists

Two responses may appear similar while requiring entirely different reasoning.

For example:

- comparing;
- classifying;
- sequencing;
- describing;
- explaining cause and effect.

Each requires different instructional expectations.

Thinking Tasks make those expectations explicit.

---

## What Thinking Tasks Do

Thinking Tasks guide:

- validation;
- success criteria;
- instructional findings;
- Teaching Moves;
- Thinking Moves;
- Communication Licensing.

Thinking Tasks influence every stage of instructional reasoning.

---

## How It Works

Assignment Context establishes the Thinking Task before student coaching begins.

Every instructional contract references the Thinking Task when interpreting student evidence.

This prevents Kaw from evaluating responses using incorrect instructional expectations.

---

## Governing Decisions

Thinking Tasks should remain independent of subject area.

The same Thinking Task should behave consistently across multiple assignments.

---

## Design Rationale

Separating Thinking Tasks from assignment content allows Kaw to generalize instructional reasoning while preserving assignment-specific coaching.

---

## Related Sections

- Assignment Context
- Instructional Contracts
- Validation Framework

---

## Revision History

**Version 0.1**

Thinking Task architecture documented.

---

# 8. Parent & Child Anchors

## Purpose

Parent and Child Anchors model the instructional relationships among Framing Routine components.

Rather than treating each component independently, Kaw understands how ideas develop across the Frame.

---

## Why This Exists

Expert teachers rarely evaluate components in isolation.

Instead, they recognize that one component provides the instructional foundation for another.

Modeling these relationships improves instructional continuity and prevents fragmented coaching.

---

## What They Are

Parent Anchors represent the instructional foundation supporting a component.

Child Anchors represent components that depend upon that foundation.

These relationships allow Kaw to determine whether later components remain instructionally aligned with earlier work.

---

## How They Work

When coaching a component, Kaw references its Parent Anchor to verify instructional alignment.

Likewise, Kaw considers Child Anchors when selecting instructional moves that will strengthen future components.

This creates continuity across the entire Frame rather than isolated coaching events.

---

## Governing Decisions

Parent and Child Anchors establish instructional dependencies.

A component should not advance if its Parent Anchor is invalid or unstable.

---

## Design Rationale

Parent and Child Anchors emerged from the realization that successful Framing Routine instruction depends not only on individual components but also on the relationships among them.

Representing these relationships explicitly allows Kaw to coach more like an expert teacher.

---

## Related Sections

- Assignment Context
- Thinking Tasks
- Student Progression

---

## Revision History

**Version 0.1**

Parent and Child Anchor architecture documented.

---

# 9. Instructional Decision Cycle

## Purpose

The Instructional Decision Cycle defines the repeatable reasoning process Kaw follows when interpreting student evidence and selecting the next instructional move.

It transforms instructional understanding into instructional action.

---

## Why This Exists

Instruction should follow a consistent reasoning process rather than reacting to individual student responses.

The Decision Cycle ensures that every instructional action can be traced back to observable evidence and explicit instructional reasoning.

---

## The Decision Cycle

Observe

↓

Orient

↓

Analyze

↓

Decide

↓

Coach

↓

Observe Again

---

## How It Works

Each stage has a distinct responsibility.

**Observe** identifies the student's evidence.

**Orient** establishes instructional context.

**Analyze** evaluates the quality and relationships of the evidence.

**Decide** selects the instructional goal, Teaching Move, and Thinking Move.

**Coach** communicates the predetermined instructional decision.

**Observe Again** begins the next instructional cycle.

---

## Governing Decisions

Every instructional interaction should progress through the complete Decision Cycle.

No instructional stage should be bypassed.

Instructional reasoning must always precede communication.

---

## Design Rationale

The Decision Cycle emerged from months of iterative refinement as the team recognized that expert teaching follows a structured reasoning process rather than selecting responses directly.

Explicitly modeling this process improves consistency, explainability, testing, and future extensibility.

---

## Related Sections

- Instructional Knowledge
- Instructional Contracts
- AI Governance

---

## Revision History

**Version 0.1**

Instructional Decision Cycle documented.

---

# PART III — Instructional Decisions

Part III defines how Kaw transforms instructional understanding into deterministic instructional action.

Once Kaw has established sufficient instructional knowledge, it must determine what should happen next. Rather than relying on AI to interpret student work or generate instructional behavior, Kaw uses deterministic instructional contracts, explicit validation rules, and governed progression pathways.

The goal of Part III is to answer a single question:

> **Given what Kaw understands about the student, what should happen next?**

These chapters define the instructional engine responsible for making that decision.

---

# 10. Instructional Contracts

## Purpose

Instructional Contracts define Kaw's deterministic instructional behavior.

Each contract represents a governed instructional pathway that translates validated student evidence into a specific instructional decision.

Rather than asking AI how to respond, Kaw first determines which instructional contract applies.

The selected contract becomes the source of instructional truth.

---

## Why This Exists

Early versions of Kaw relied on increasingly complex prompt logic to determine feedback.

Although functional, this approach mixed instructional reasoning with language generation and became difficult to scale, test, and maintain.

Instructional Contracts separate instructional decision-making from communication.

This makes instructional behavior:

- deterministic;
- testable;
- explainable;
- reusable;
- extensible.

---

## What an Instructional Contract Contains

Each contract defines:

- Contract Identifier
- Instructional Intent
- Entry Conditions
- Required Instructional Knowledge
- Validation Rules
- Instructional Findings
- Instructional Goal
- Teaching Move
- Thinking Move
- Communication License
- Save Behavior
- Progression Rules
- Exit Conditions
- Prohibited Behaviors

Each field is deterministic.

AI does not determine any of these values.

---

## Contract Families

Current contract families include:

- KT-GS (Key Topic)
- IA-GS (Is About)
- MI-GS (Main Idea)
- ED-GS (Essential Detail)
- SW-GS (So What)

Future contract families may support:

- Revision
- Celebration
- Clarification
- Misconceptions
- Cross-component alignment
- Teacher interventions

---

## How Contracts Work

The instructional engine:

1. establishes instructional knowledge;
2. validates student evidence;
3. identifies instructional findings;
4. selects the appropriate contract;
5. executes deterministic instructional behavior;
6. authorizes AI communication through the Communication License.

Every coaching interaction follows this sequence.

---

## Governing Decisions

Instructional Contracts own instructional behavior.

Contracts may determine:

- instructional goals;
- Teaching Moves;
- Thinking Moves;
- progression;
- saving;
- Communication Licensing.

Contracts may not delegate instructional decision-making to AI.

---

## Design Rationale

Instructional Contracts emerged from the realization that expert teaching consists of repeatable instructional decisions rather than repeatable sentences.

Separating instructional behavior from communication significantly improved consistency, testing, and future scalability.

---

## Related Sections

- Validation Framework
- Student Progression
- AI Governance
- Communication Licensing

---

## Revision History

Version 0.1

Initial Instructional Contract architecture documented.

---

# 11. Validation Framework

## Purpose

Validation determines whether student work satisfies the instructional expectations for the current component.

Validation establishes instructional truth.

---

## Why This Exists

Students deserve instructional feedback based on observable evidence rather than AI confidence.

Validation prevents Kaw from accepting incomplete, inaccurate, or unsupported work.

It also protects students from progressing before demonstrating understanding.

---

## Validation Philosophy

Validation answers only one question:

> Has the student demonstrated the required instructional evidence?

Validation does not determine:

- feedback wording;
- encouragement;
- coaching style;
- AI response.

Those occur afterward.

---

## Validation Sources

Validation may consider:

- Assignment Context
- Thinking Task
- Component Success Criteria
- Parent Anchors
- Child Anchors
- Previously Validated Work
- Current Student Evidence

Each contributes to determining instructional findings.

---

## Validation Outcomes

Validation generally results in one of several instructional findings:

- Valid
- Partially Valid
- Needs Clarification
- Misconception
- Missing Evidence
- Misaligned
- Requires Revision

Each finding maps to one or more Instructional Contracts.

---

## Governing Decisions

Validation always precedes:

- saving;
- progression;
- celebration;
- AI communication.

Student work should never be saved simply because it sounds plausible.

---

## Design Rationale

Separating validation from communication allows Kaw to explain why a decision occurred while preventing AI from becoming the evaluator.

---

## Related Sections

- Instructional Contracts
- Student Progression
- AI Governance

---

## Revision History

Version 0.1

Validation Framework documented.

---

# 12. Student Progression

## Purpose

Student Progression determines when a learner is ready to move forward within the Framing Routine.

Progression protects instructional sequencing.

---

## Why This Exists

Completing a response is not the same as demonstrating understanding.

Students should progress because they have met instructional expectations—not because the conversation has ended.

---

## Progression Philosophy

Progression occurs only after:

- validation;
- contractual completion;
- save authorization;
- instructional readiness.

Each step must occur in sequence.

---

## Progression States

Students may occupy states such as:

- Working
- Revising
- Clarifying
- Validated
- Saved
- Ready for Progression
- Progressed

Future versions may introduce additional instructional states.

---

## How Progression Works

The instructional engine evaluates:

1. Has the component been validated?
2. Has the governing contract completed?
3. Has student work been saved?
4. Are Parent Anchors stable?
5. Is the student instructionally prepared for the next component?

Only then does progression occur.

---

## Governing Decisions

Students should never progress because:

- AI generated a convincing response;
- enough conversation occurred;
- time elapsed.

Progression is an instructional decision.

---

## Design Rationale

Separating progression from communication preserves instructional integrity and ensures that advancement reflects demonstrated understanding rather than conversational completion.

---

## Related Sections

- Validation Framework
- Instructional Contracts
- Parent & Child Anchors

---

## Revision History

Version 0.1

Student Progression architecture documented.
---
# PART IV — AI Communication

Part IV defines the role of Artificial Intelligence within the Kaw architecture.

Unlike many AI tutoring systems, Kaw does not use AI to determine instructional decisions.

Instead, AI serves as a governed communication layer that translates deterministic instructional decisions into natural, assignment-specific coaching while preserving teacher voice and student ownership.

The goal of Part IV is to answer a single question:

> **Once Kaw has determined what should happen instructionally, how should that decision be communicated to the student?**

These chapters define the boundaries, permissions, and responsibilities of AI within the Kaw architecture.

---

# 13. AI Governance

## Purpose

AI Governance defines the instructional boundaries of artificial intelligence within Kaw.

Its purpose is to ensure that AI enhances communication without becoming the instructional decision-maker.

---

## Why This Exists

Modern language models are exceptionally effective at generating natural language.

However, they are not inherently instructional systems.

Without governance, AI may:

- infer instructional intent;
- generate student work;
- skip instructional steps;
- provide answers instead of coaching;
- introduce inconsistencies across identical instructional situations.

AI Governance prevents these behaviors by assigning instructional authority to the deterministic instructional engine.

---

## Governance Principle

The central governance principle of Kaw is:

> Deterministic instructional reasoning owns instructional decisions.
>
> AI owns communication.

This principle applies throughout the system.

---

## Instructional Authority

The deterministic instructional engine owns:

- intent classification;
- Assignment Context;
- Thinking Task;
- instructional understanding;
- validation;
- instructional findings;
- instructional goals;
- Teaching Moves;
- Thinking Moves;
- progression;
- save behavior;
- Communication Licensing;
- student work protection.

These responsibilities may never be delegated to AI.

---

## AI Authority

AI may:

- communicate naturally;
- personalize language;
- reference Assignment Context;
- reference validated student evidence;
- ask licensed instructional questions;
- adapt tone while preserving instructional intent;
- maintain conversational flow.

---

## AI Restrictions

AI may not:

- determine instructional goals;
- validate student work;
- override contracts;
- change Thinking Moves;
- generate Frame components;
- complete student thinking;
- save work;
- advance students;
- reinterpret deterministic instructional findings.

---

## Governing Decisions

Every AI response must be traceable to a deterministic instructional decision.

If a response cannot be traced to the governing contract and Communication License, the architecture has been violated.

---

## Design Rationale

Separating instructional authority from language generation ensures that Kaw remains instructionally consistent while benefiting from the flexibility and conversational quality of modern language models.

---

## Related Sections

- Instructional Contracts
- Validation Framework
- Communication Licensing
- Runtime Communication

---

## Revision History

Version 0.1

Initial AI Governance documented.

---

# 14. Communication Licensing

## Purpose

Communication Licensing defines exactly what AI is authorized to communicate during a coaching interaction.

It protects instructional integrity by limiting AI to approved instructional behaviors.

---

## Why This Exists

Even when instructional reasoning is deterministic, unrestricted language generation can unintentionally:

- provide answers;
- suggest unsupported ideas;
- advance instruction prematurely;
- complete student work.

Communication Licensing prevents these outcomes.

---

## What a Communication License Defines

Each Communication License specifies:

- communication intent;
- permitted instructional behaviors;
- prohibited behaviors;
- allowable references to student work;
- allowable references to Assignment Context;
- questioning strategy;
- response boundaries.

---

## How It Works

After an Instructional Contract is selected, the contract assigns a Communication License.

AI generates responses only within the permissions established by that license.

The Communication License functions as the final instructional safeguard before AI communication begins.

---

## Governing Decisions

Communication Licenses authorize communication.

They do not authorize instructional decisions.

Only deterministic instructional reasoning may determine instructional intent.

---

## Design Rationale

Communication Licensing emerged after recognizing that instructional quality depends not only on what AI says but also on what AI is prevented from saying.

Explicit communication permissions make AI behavior more predictable, testable, and instructionally consistent.

---

## Related Sections

- AI Governance
- Prompt Construction
- Runtime Communication

---

## Revision History

Version 0.1

Communication Licensing documented.

---

# 15. Prompt Construction

## Purpose

Prompt Construction transforms deterministic instructional decisions into structured inputs for AI.

Rather than asking AI to determine instructional behavior, prompts communicate decisions that have already been made.

---

## Why This Exists

Prompt engineering should not replace instructional reasoning.

Its purpose is to communicate deterministic instructional decisions as clearly as possible.

---

## Prompt Philosophy

Every prompt should communicate:

- Assignment Context;
- instructional findings;
- instructional goal;
- Teaching Move;
- Thinking Move;
- Communication License;
- relevant student evidence.

Prompts should never ask AI to determine these values.

---

## How Prompt Construction Works

Prompt generation occurs after:

1. instructional understanding;
2. validation;
3. contract selection;
4. Communication Licensing.

Only then is the runtime prompt assembled.

The prompt functions as an implementation detail rather than the source of instructional reasoning.

---

## Governing Decisions

Prompts may organize instructional information.

They may not redefine instructional information.

---

## Design Rationale

Separating prompt construction from instructional reasoning allows prompt templates to evolve independently while preserving deterministic instructional behavior.

---

## Related Sections

- AI Governance
- Communication Licensing
- Runtime Communication

---

## Revision History

Version 0.1

Prompt Construction documented.

---

# 16. Runtime Communication

## Purpose

Runtime Communication represents the final stage of the instructional architecture.

It is the moment when deterministic instructional reasoning is translated into a student-facing coaching interaction.

---

## Why This Exists

Students experience Kaw through conversation rather than through contracts, validation rules, or decision cycles.

Runtime Communication bridges deterministic instructional reasoning with natural instructional dialogue.

---

## Runtime Flow

The runtime sequence is:

Instructional Knowledge

↓

Validation

↓

Instructional Findings

↓

Instructional Contract

↓

Teaching Move

↓

Thinking Move

↓

Communication License

↓

Prompt Construction

↓

AI Response

↓

Student Response

↓

Observe Again

Each stage depends on the stages preceding it.

---

## Runtime Responsibilities

During runtime:

The deterministic engine:

- reasons;
- validates;
- decides;
- governs.

AI:

- contextualizes;
- communicates;
- personalizes;
- encourages.

The student:

- thinks;
- responds;
- revises;
- progresses.

---

## Governing Decisions

Runtime Communication must preserve:

- instructional intent;
- student ownership;
- assignment alignment;
- deterministic instructional decisions;
- conversational continuity.

Language may vary.

Instructional behavior may not.

---

## Design Rationale

The runtime architecture reflects Kaw's central philosophy:

Instruction should be deterministic.

Communication should be natural.

The instructional engine teaches.

AI gives the instruction a human voice.

---

## Related Sections

- AI Governance
- Communication Licensing
- Prompt Construction
- Testing Framework

---

## Revision History

Version 0.1

Runtime Communication documented.
---

# PART V — System Engineering

Part V defines how the Kaw architecture is implemented, tested, maintained, and evolved.

While the previous sections describe Kaw's instructional philosophy and reasoning, this section describes the engineering practices that ensure the system remains reliable, explainable, maintainable, and instructionally faithful.

The goal of Part V is to answer a single question:

> **How do we build, protect, and evolve the Kaw system while preserving its instructional integrity?**

---

# 17. System Architecture

## Purpose

The System Architecture defines the major components of Kaw and the relationships among them.

It describes **what exists**, **why it exists**, and **how information flows** through the system.

---

## Why This Exists

As Kaw evolved from a collection of prompts into a deterministic instructional system, a clear architectural model became essential.

The architecture provides a shared understanding of:

- system responsibilities;
- component boundaries;
- instructional ownership;
- communication flow;
- future extensibility.

---

## Architectural Layers

The current architecture consists of five primary layers:

1. User Experience Layer
2. Instructional Reasoning Layer
3. Deterministic Decision Engine
4. AI Communication Layer
5. Data & Persistence Layer

Each layer has clearly defined responsibilities.

---

## Design Principles

The architecture is designed to be:

- modular;
- deterministic;
- explainable;
- testable;
- extensible;
- instructionally governed.

---

## Governing Decisions

Every new feature should strengthen—not bypass—the architectural layers.

If a feature requires AI to assume instructional authority, the design should be reconsidered.

---

## Related Sections

- Project Overview
- Kaw Instructional Model
- AI Governance
- Application Flow

---

## Revision History

Version 0.1

Initial architecture documented.

---

# 18. Application Flow

## Purpose

Application Flow documents the sequence of operations that occur during a student interaction.

It explains how information moves through the system from student input to instructional coaching.

---

## High-Level Flow

Student Response

↓

Instructional Knowledge

↓

Validation

↓

Instructional Findings

↓

Instructional Contract

↓

Teaching Move

↓

Thinking Move

↓

Communication License

↓

Prompt Construction

↓

AI Communication

↓

Student Response

---

## Why This Exists

Documenting application flow ensures that every instructional interaction follows the same deterministic pathway.

---

## Governing Decisions

Application flow should remain linear and traceable.

No stage should bypass validation, contracts, or governance.

---

## Design Rationale

The application flow mirrors the reasoning process of an expert teacher.

Instructional understanding always precedes instructional communication.

---

## Related Sections

- Instructional Decision Cycle
- Runtime Communication
- Testing Framework

---

## Revision History

Version 0.1

Application Flow documented.

---

# 19. Data Structures

## Purpose

Data Structures define the information required to support deterministic instructional reasoning.

Rather than documenting implementation-specific code, this section documents the conceptual data model.

---

## Core Objects

Examples include:

- Assignment Context
- Student Session
- Student Response
- Validation Result
- Instructional Finding
- Instructional Contract
- Teaching Move
- Thinking Move
- Communication License
- Parent Anchor
- Child Anchor
- Progression State

Future implementation details may evolve while preserving these conceptual objects.

---

## Governing Decisions

Data structures should represent instructional concepts rather than implementation shortcuts.

---

## Design Rationale

Separating conceptual data structures from implementation details allows Kaw to evolve across technologies without changing its instructional model.

---

## Related Sections

- System Architecture
- Instructional Knowledge
- Instructional Contracts

---

## Revision History

Version 0.1

Conceptual data model documented.

---

# 20. Testing Framework

## Purpose

The Testing Framework ensures that Kaw behaves consistently, deterministically, and instructionally correctly.

Testing protects both software quality and instructional integrity.

---

## Types of Testing

Current testing includes:

- component validation;
- instructional contract testing;
- communication license testing;
- regression testing;
- instructional pathway testing;
- end-to-end experience testing.

Future testing may include automated instructional simulations.

---

## Testing Philosophy

Tests should verify instructional behavior—not exact wording.

Because AI responses may vary linguistically, tests evaluate:

- instructional findings;
- Teaching Moves;
- Thinking Moves;
- progression;
- communication permissions.

---

## Governing Decisions

No architectural change should be considered complete until:

- implementation is updated;
- tests are updated;
- documentation is updated.

---

## Design Rationale

Instructional correctness is more important than response consistency.

Testing should protect instructional intent rather than specific sentences.

---

## Related Sections

- Instructional Contracts
- AI Governance
- Runtime Communication

---

## Revision History

Version 0.1

Testing Framework documented.

---

# 21. Deployment

## Purpose

Deployment describes how Kaw is delivered, configured, and maintained across environments.

---

## Current Environment

Current deployment includes:

- GitHub source repository;
- Vercel hosting;
- deterministic instructional engine;
- AI communication services;
- persistent data storage.

Future deployment strategies may expand as the project evolves.

---

## Governing Decisions

Deployment decisions should never compromise instructional governance.

Infrastructure should support—not redefine—the instructional architecture.

---

## Revision History

Version 0.1

Deployment documented.

---

# 22. Development Standards

## Purpose

Development Standards establish expectations for future contributions to Kaw.

They ensure architectural consistency as the project grows.

---

## Development Principles

All development should:

- preserve instructional integrity;
- maintain deterministic reasoning;
- document architectural decisions;
- include testing;
- update documentation;
- prioritize explainability;
- protect student ownership.

---

## Definition of Done

A feature is complete only when:

✓ Implementation is complete.

✓ Tests pass.

✓ Documentation is updated.

✓ Architectural alignment has been verified.

---

## Governing Decisions

Convenience should never replace instructional correctness.

---

## Revision History

Version 0.1

Development Standards documented.

---

# 23. Decision Log

## Purpose

The Decision Log preserves significant architectural decisions and the reasoning behind them.

It serves as the institutional memory of the project.

---

## Why This Exists

Many architectural decisions make perfect sense when they are made but become difficult to reconstruct months or years later.

Capturing the rationale behind those decisions preserves the evolution of the system.

---

## Example Entries

- Separated instructional reasoning from AI communication.
- Introduced Assignment Context as persistent instructional memory.
- Added Parent and Child Anchors.
- Introduced Communication Licensing.
- Replaced deterministic prompts with deterministic instructional reasoning.

Future decisions should be added as the architecture evolves.

---

## Governing Decisions

Architectural decisions should document:

- the problem;
- the alternatives considered;
- the final decision;
- the instructional rationale.

---

## Revision History

Version 0.1

Decision Log introduced.

---

# 24. Roadmap

## Purpose

The Roadmap documents the intended evolution of Kaw.

It communicates strategic direction without prescribing implementation details.

---

## Near-Term Priorities

- Complete all Framing Routine components.
- Expand Instructional Contracts.
- Strengthen Validation Framework.
- Refine Teacher Voice.
- Conduct student pilot testing.
- Refine architecture based on observational data.

---

## Long-Term Vision

Future development may include:

- additional instructional routines;
- broader curriculum support;
- teacher dashboards;
- instructional analytics;
- expanded testing automation;
- enhanced developer tooling.

The instructional philosophy and governance principles defined throughout this guide should remain stable even as implementation evolves.

---

## Governing Decisions

Future features should extend the architecture rather than replace it.

Every major addition should align with:

- Design Philosophy;
- Instructional Commitments;
- Kaw Instructional Model;
- AI Governance.

---

## Revision History

Version 0.1

Initial roadmap documented.

---

# 25. Appendices

## Canonical Instructional Example Format

### Instructional Situation

### Before Kaw Responds

**Student Evidence (Observe)**  
What the student has demonstrated.

**Instructional Goal (Decide)**  
What Kaw is trying to help the student accomplish next.

**Teaching Move (Coach)**  
How Kaw will support the student.

**Thinking Move (Advance One Thinking Step)**  
The cognitive move the student is being asked to make.

### Kaw Responds

The contextualized student-facing prompt.

### Why This Response?

An explanation of how the response reflects the evidence, goal, teaching move, thinking move, and governance model.

---

## Glossary

**Assignment Context**  
Persistent instructional information required to understand and coach the assignment responsibly.

**Communication License**  
A deterministic set of permissions and restrictions governing what AI may communicate.

**Instructional Contract**  
A governed pathway mapping findings to a goal, teaching move, thinking move, and Communication License.

**Instructional Finding**  
A structured description of what the student demonstrated and what need remains.

**Parent Anchor**  
An upstream Frame component that constrains interpretation of the current component.

**Child Anchor**  
A downstream component whose relationship may help validate the current component.

**Teaching Move**  
The instructional method used to help the student progress.

**Thinking Move**  
The specific cognitive action the student is asked to take next.

**Thinking Task**  
The cognitive work required by the assignment.

---

# Document Maintenance

When a significant change is made to Kaw:

1. implement the change;
2. test the change;
3. update the relevant guide section;
4. add a Decision Log entry when architecture or governance changes;
5. commit code, tests, and documentation together whenever practical.
