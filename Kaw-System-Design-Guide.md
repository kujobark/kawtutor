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

Kaw Companion is an instructional coaching system designed to help students strengthen their thinking and writing through the KU-CRL Framing Routine.

Kaw does not complete student work. It observes student evidence, determines the most appropriate next instructional move, and communicates that move in a natural, assignment-specific way.

Kaw is designed to:

- understand the student’s assignment;
- identify the thinking task embedded in that assignment;
- determine the student’s current location within the Frame;
- evaluate the evidence the student has already demonstrated;
- select an appropriate instructional goal and teaching move;
- communicate that move without supplying the student’s thinking.

## Architectural Evolution

Early versions relied on manually authored deterministic prompts and responses. That clarified instructional intent but did not scale across the growing number of student situations.

Kaw therefore evolved toward a layered architecture in which deterministic instructional reasoning continues to make every instructional decision while AI is used only to communicate the predetermined decision naturally.

> Deterministic instructional reasoning establishes instructional truth. AI communicates that instructional truth naturally while preserving teacher voice and student ownership.

---

# 2. Design Philosophy

Kaw begins with evidence rather than assumption.

The system does not start by asking what response would sound helpful. It starts by asking:

> What instructional understanding must exist before an expert teacher can confidently make the next instructional move?

Kaw separates three functions:

1. instructional understanding;
2. instructional decision-making;
3. communication.

AI is introduced only after the first two are complete.

---

# 3. Kaw Instructional Commitments

## Begin with Student Evidence

Kaw starts with what the student has actually demonstrated. It does not infer understanding that is not observable.

## Let Evidence Drive Instruction

The student’s evidence determines the next instructional goal.

## Advance One Thinking Step

Kaw moves the student forward one manageable instructional step rather than collapsing several thinking moves into one prompt.

## Build from Student Success

When the student demonstrates partial success, Kaw names and uses that success as the foundation for the next move.

## Preserve Instructional Continuity

Kaw maintains continuity across the assignment, Frame component, previous student work, prior coaching, and current instructional goal.

---

# 4. Kaw Instructional Model

## Level 1: Instructional Commitments

What Kaw promises every student:

- Begin with Student Evidence
- Let Evidence Drive Instruction
- Advance One Thinking Step
- Build from Student Success
- Preserve Instructional Continuity

## Level 2: Instructional Knowledge

What Kaw understands:

- Assignment Context
- Thinking Task
- Framing Routine
- Component Success Criteria
- Parent and Child Anchors
- Instructional Expectations

## Level 3: Instructional Decision Cycle

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

## Level 4: Instructional Contracts

What instructional move should occur:

- KT-GS-001
- IA-GS-001
- MI-GS-001
- ED-GS-001
- SW-GS-001
- Revision
- Celebrate
- Clarify
- Misconception

## Level 5: Teacher Voice

How the predetermined move is communicated naturally, clearly, and in context.

---

# 5. Instructional Knowledge

Before coaching, Kaw establishes the knowledge required to interpret student work responsibly:

- assignment;
- expected product;
- content or text;
- thinking task;
- current Frame component;
- relationships among components;
- component success criteria;
- prior student evidence;
- instructional expectations.

Instructional knowledge directly constrains the decisions Kaw may make.

---

# 6. Assignment Context

Assignment Context is persistent instructional memory. It may include:

- assignment directions;
- content area;
- source text or topic;
- grade level;
- expected response type;
- required thinking task;
- teacher-provided criteria;
- relevant vocabulary;
- instructional constraints.

Assignment Context must be established before component-level coaching whenever the student’s response cannot be interpreted responsibly without it.

---

# 7. Thinking Tasks

The Thinking Task identifies the cognitive work required by the assignment.

Examples include:

- compare;
- explain;
- describe;
- analyze;
- evaluate;
- identify cause and effect;
- determine importance;
- synthesize.

The Thinking Task affects what counts as successful evidence, which relationships should appear in the Frame, and what type of instructional prompt is appropriate.

---

# 8. Parent and Child Anchors

Frame components are evaluated relationally, not only in isolation.

Examples:

- Is About is anchored to Key Topic.
- Main Idea is anchored to Key Topic and Is About.
- Essential Detail is anchored to its Main Idea.
- So What is anchored to the completed Frame.

Anchors support relationship validation, continuity, progression, resume behavior, and prevention of disconnected or circular responses.

---

# 9. Instructional Decision Cycle

## Observe

Identify the evidence actually present.

## Orient

Locate the student within the assignment, Thinking Task, Framing Routine, and progression state.

## Analyze

Evaluate quality, completeness, precision, and relationships.

## Decide

Select the immediate instructional goal.

## Coach

Select the teaching move and thinking move that advance the student one step.

## Observe Again

Evaluate the student’s next response and repeat the cycle.

---

# 10. Instructional Contracts

Instructional Contracts convert validated evidence into governed instructional behavior.

Each contract should define:

- component;
- entry conditions;
- instructional finding;
- instructional goal;
- teaching move;
- thinking move;
- communication license;
- save or progression rules;
- validation expectations;
- prohibited behaviors.

## KT-GS-001: Key Topic

Supports the student in identifying a focused topic that accurately represents the assignment.

## IA-GS-001: Is About

Supports the student in explaining what the Key Topic is specifically about within the assignment.

## MI-GS-001: Main Idea

Supports the student in developing a meaningful organizing idea.

## ED-GS-001: Essential Detail

Supports the student in developing observable evidence that supports a Main Idea. This is currently the most mature reference implementation.

## SW-GS-001: So What

Supports the student in synthesizing the meaning, significance, implication, or larger importance of the completed Frame.

---

# 11. Validation Framework

Validation occurs before AI communication.

## Component Validation

Possible dimensions:

- componentEvidenceLevel;
- componentCriteriaStatus;
- specificity;
- completeness;
- observability;
- duplication;
- circularity.

## Relationship Validation

Possible dimensions:

- relationshipStatus;
- parent alignment;
- child support;
- assignment alignment;
- Thinking Task alignment;
- progression consistency.

## Instructional Findings

Validators should return structured findings rather than only pass or fail.

Potential fields:

- componentEvidenceLevel
- componentCriteriaStatus
- relationshipStatus
- diagnosis
- recommended contract
- save eligibility
- progression eligibility

## Runtime Validation

Runtime validation ensures that invalid work is not saved, valid work is preserved, resume state is maintained, and progression does not occur prematurely.

---

# 12. Student Progression

Progression is an instructional decision, not a navigation convenience.

Kaw should progress only when the current component meets required criteria. Progression rules should preserve valid work, support revision, allow celebration, retain prior context, and avoid forcing unnecessary rework.

The system advances one thinking step, not merely one screen.

---

# 13. AI Governance

AI never owns instructional decisions.

The deterministic instructional engine owns:

- intent classification;
- assignment understanding;
- instructional goals;
- teaching moves;
- thinking moves;
- progression;
- validation;
- save decisions;
- student work protection.

AI is limited to contextualizing the predetermined instructional move into a natural, assignment-specific response.

AI must not change the goal, select a different move, infer unsupported understanding, generate student work, override validation, or advance the student.

---

# 14. Communication Licensing

A Communication License defines what AI may and may not do.

## AI May

- contextualize the predetermined move;
- reference the assignment;
- reference observable student work;
- adapt wording to context;
- use a natural teacher-like voice;
- build from identified success;
- ask the licensed thinking question.

## AI May Not

- alter instructional intent;
- change the instructional goal;
- introduce a new teaching move;
- provide the answer;
- write the student’s Frame component;
- infer evidence not present;
- make progression or save decisions.

---

# 15. Prompt Construction

Prompt construction occurs only after deterministic reasoning is complete.

Assignment Context  
↓  
Thinking Task  
↓  
Student Evidence  
↓  
Validation  
↓  
Instructional Finding  
↓  
Instructional Contract  
↓  
Instructional Goal  
↓  
Teaching Move  
↓  
Thinking Move  
↓  
Communication License  
↓  
AI Contextualization

---

# 16. Runtime Communication

Runtime communication should:

- sound natural rather than templated;
- remain assignment-specific;
- reference actual student evidence;
- preserve continuity;
- avoid overexplaining;
- ask one purposeful thinking question;
- avoid competing directions;
- maintain the predetermined goal.

Wording may vary while remaining instructionally equivalent.

---

# 17. System Architecture

## Earlier Architecture

Assignment  
↓  
KU Frame  
↓  
Deterministic Prompt  
↓  
Student

## Current Architecture

Assignment  
↓  
Assignment Context  
↓  
Thinking Task  
↓  
KU Frame  
(Key Topic • Is About • Main Ideas • Essential Details • So What)  
↓  
Student Evidence Analysis  
↓  
Instructional Decision  
↓  
Communication License  
↓  
AI Communication  
↓  
Student

---

# 18. Application Flow

1. Load or establish Assignment Context.
2. Determine the Thinking Task.
3. Locate the student within the Framing Routine.
4. Retrieve Parent and Child Anchors.
5. Observe current evidence.
6. Run component and relationship validation.
7. Produce an instructional finding.
8. Select the governing contract.
9. Determine the instructional goal.
10. Select the teaching move.
11. Select the thinking move.
12. Issue a Communication License.
13. Build the constrained AI prompt.
14. Generate the contextual response.
15. Validate runtime behavior.
16. Save or block student work.
17. Progress, revise, celebrate, or clarify.
18. Observe again.

---

# 19. Data Structures

This section remains under development.

Future documentation should include:

- Assignment Context schema;
- Thinking Task schema;
- Frame state;
- component state;
- Parent and Child Anchor references;
- instructional findings;
- contract identifiers;
- communication licenses;
- runtime history;
- save and resume state;
- test fixtures.

---

# 20. Testing Framework

## Deterministic Tests

Verify that Kaw reasons correctly:

- component validation;
- relationship validation;
- instructional findings;
- contract selection;
- save prevention;
- progression;
- resume behavior;
- circularity;
- no-evidence cases.

## Runtime Tests

Verify actual application behavior, including save blocking, valid saves, second-detail behavior, and preserved resume state.

## AI Communication Tests

Verify that AI follows its license. Tests should evaluate response characteristics rather than exact wording.

## Instructional Experience Tests

A future suite should evaluate end-to-end student experiences across Key Topic, Is About, Main Idea, Essential Detail, and So What.

---

# 21. Deployment

Current implementation uses:

- Wix for the student-facing interface;
- Vercel for backend deployment;
- GitHub for source control and documentation.

Future documentation should include environment variables, API endpoints, deployment workflow, rollback procedure, Wix integration, mobile and tablet considerations, and production versus preview environments.

---

# 22. Development Standards

1. Instructional decisions remain deterministic.
2. AI behavior must be licensed.
3. Every new pathway includes validation.
4. Significant pathways include tests.
5. Invalid student work is not saved.
6. Existing valid work is protected.
7. Changes preserve resume behavior.
8. Architectural decisions are documented here.
9. Code, tests, and documentation are updated together.
10. New work must not weaken student ownership.

---

# 23. Decision Log

## July 2026: AI Limited to Communication

**Decision:** AI will not own instructional decisions.

**Reason:** Instructional intent, goals, teaching moves, progression, validation, and student work protection must remain predictable and governable.

## July 2026: Essential Detail as Reference Pathway

**Decision:** ED-GS-001 serves as the reference implementation for the contract architecture.

**Reason:** Essential Detail exposed the need for observable evidence, relationship validation, circularity checks, save protection, and contextual communication.

## July 2026: Assignment Context as Persistent Memory

**Decision:** Assignment Context is established before component-level coaching.

**Reason:** Student responses cannot be interpreted responsibly without understanding the assignment and required thinking.

## July 2026: Parent and Child Anchors

**Decision:** Components are evaluated relationally, not only in isolation.

**Reason:** A component may appear valid independently while failing to support or align with surrounding components.

---

# 24. Roadmap

## Immediate Priority

Prepare the full Framing Routine pathway for the first student pilot:

- Key Topic;
- Is About;
- Main Idea;
- Essential Detail;
- So What.

The goal is a usable, coherent experience that reveals where students become confused, successful, or stalled.

## Near-Term Priorities

- strengthen Key Topic;
- strengthen Is About;
- validate Main Idea;
- maintain Essential Detail;
- develop So What synthesis;
- add end-to-end experience tests;
- document data structures;
- document deployment;
- formalize development standards;
- later create the external teacher-facing Kaw Instructional Guide.

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
