**BEGINNING OF OFFICIAL MANUSCRIPT**

*Source basis: This draft preserves and builds from the uploaded Version 1.1 architecture guide while incorporating the constitutional governance decisions we finalized, including the updated governance principles, Communication Governance, and the new Communication Validation layer. *

---

# Kaw Companion Governance & Architecture Guide

### Constitutional Principles, Instructional Architecture, and System Design

**Version 2.0 — Draft 1**

---

**Kaw Companion**

A Governed Instructional Engine

---

*"The student owns the thinking. Kaw owns the instruction."*

---

© 2026

Working Constitutional Draft

---

# Revision History

| Version           | Date        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.0               | Spring 2026 | Initial instructional architecture documenting the Framing Routine implementation.                                                                                                                                                                                                                                                                                                                                                                                           |
| 1.1               | July 2026   | Introduced the Two-Gate Progression Model, expanded student ownership protections, formalized Communication Governance, and clarified instructional sufficiency.                                                                                                                                                                                                                                                                                                             |
| **2.0 (Draft 1)** | July 2026   | Reorganized the project as a constitutional governance guide. Established deterministic instructional governance as the foundation of the system, elevated Communication Governance into an architectural layer, introduced Communication Validation following AI contextualization, replaced "instructional truth" with "instructional intent," and generalized the architecture so the Framing Routine becomes the current implementation rather than the defining system. |

---

# Preface

Every mature engineering discipline eventually reaches a point where implementation alone is no longer sufficient.

Software evolves.

Algorithms evolve.

Individual features evolve.

Without an enduring constitutional framework, however, each improvement introduces the possibility of architectural drift. Systems slowly become collections of individually reasonable decisions that no longer operate according to a coherent design philosophy.

The Kaw Companion project reached precisely this point during the development of Version 1.1.

What began as an instructional companion for the KU-CRL Framing Routine gradually revealed a deeper architectural discovery. The project was not merely building an intelligent tutoring system. It was uncovering the constitutional principles required for a governed instructional engine capable of combining deterministic instructional reasoning with adaptive artificial intelligence.

Version 2.0 established the constitutional foundation for that discovery.

It defined the permanent separation between instructional reasoning and instructional communication, introduced deterministic instructional intent as the governing center of the architecture, and established the constitutional principles that every future implementation must preserve.

Continued development of the runtime, however, revealed that another architectural distinction was equally important.

Determining instruction and communicating instruction are not the same activity.

Likewise, communicating instruction and teaching are not the same activity.

Instruction determines **what** should occur.

Communication determines **how** that predetermined instruction is expressed.

Teaching emerges from the successful integration of both.

This realization significantly expanded the role of the Communication Architecture.

Rather than functioning solely as a governed interface between deterministic reasoning and artificial intelligence, the Communication Architecture became the architectural description of how predetermined instructional decisions are transformed into authentic instructional experiences while preserving constitutional governance.

At the same time, implementation revealed another foundational requirement.

Before instructional reasoning can determine instructional intent, the instructional engine must first establish sufficient understanding of the student's assignment.

Every subsequent instructional inference—including Thinking Task identification, instructional interpretation, instructional contracts, validation, progression, and communication—depends upon this initial understanding.

For this reason, Assignment Understanding now occupies a permanent architectural role as the instructional gateway into the governed instructional engine.

Instruction does not begin when Kaw believes it understands the assignment.

Instruction begins only after Kaw and the student establish a shared understanding of the assignment supported by sufficient instructional evidence.

This guide represents the continuing evolution of the Kaw Companion architecture.

Rather than documenting a specific implementation, this manuscript defines the governing principles and architectural layers that every implementation must obey. The Framing Routine remains the current instructional framework through which these principles are expressed, but it is no longer the architectural boundary of the system itself.

Throughout this guide, one distinction remains paramount.

Instruction and communication are fundamentally different operations.

Instruction establishes instructional intent through deterministic reasoning.

Communication expresses that predetermined intent through governed instructional interactions.

Artificial intelligence participates only within those governed communication boundaries.

This separation is not simply a software design preference.

It is the constitutional principle upon which every subsequent architectural decision depends.

The pages that follow define those principles, describe the instructional architecture that carries them out, and explain how governed communication transforms deterministic instructional reasoning into authentic teaching while preserving student ownership and instructional integrity.

Constitutional principles are intentionally difficult to change because every architectural layer and every implementation depends upon them.

Implementations evolve.

Architecture matures.

The Constitution changes only with exceptional justification.

---

## One recommendation before we move on...

I would make **one small change** to the hierarchy diagram immediately following the Preface.

Right now it says:

```text
Constitution
        ↓
Instructional Architecture
        ↓
Communication Architecture
        ↓
System Design
```

I actually **would not** insert Assignment Understanding as another layer there.

After thinking about it, I believe that's incorrect architecturally.

The **Assignment Understanding Validator belongs inside the Instructional Architecture**, not alongside it.

So instead, I would leave the hierarchy exactly as it is and add a new diagram in the Instructional Architecture section showing:

```text
Student Assignment
        ↓
Assignment Understanding
        ↓
Instructional Reasoning
        ↓
Instructional Intent
        ↓
Communication Architecture
```

                     CONSTITUTION

          Constitutional Governance
                 (Permanent)

                      ↓

         Instructional Architecture
                 (Permanent)

                      ↓

         Communication Architecture
                 (Permanent)

                      ↓

               System Design
          (Reference Implementation)

                      ↓

         Instructional Framework
      (Framing Routine – Current)

                      ↓

            Runtime Implementation
         (tutor.js and Supporting Code)

---
# Table of Contents

**PART I — Constitutional Governance**

Chapter 1 — Why Governance Comes First

Chapter 2 — Constitutional Principles

Chapter 3 — Deterministic Instructional Intent

Chapter 4 — Student Ownership and Instructional Responsibility

Chapter 5 — Governing Artificial Intelligence

---

**PART II — Instructional Architecture**

Chapter 6 — The Governed Instructional Engine

Chapter 7 — Assignment Understanding

Chapter 8 — Instructional Knowledge

Chapter 9 — Instructional Reasoning

Chapter 10 — Instructional Contracts

Chapter 11 — The Two-Gate Progression Model

---

**PART III — Communication Architecture**

Chapter 12 — Communication Governance

Chapter 13 — Communication Goals and Communication Taxonomy

Chapter 14 — AI Contextualization

Chapter 15 — Communication Validation

Chapter 16 — Recovery Architecture

Chapter 17 — Teacher Voice

---

**PART IV — System Design**

Chapter 18 — Instructional State

Chapter 19 — Evidence Processing

Chapter 20 — Assignment Context

Chapter 21 — Extending the Architecture

Chapter 22 — Future Instructional Models

---

**PART V — Appendices**

Appendix A — Constitutional Summary

Appendix B — Complete Architectural Flow

Appendix C — Deterministic and Non-Deterministic Responsibilities

Appendix D — Immutable Layer Contracts

Appendix E — Design Principles

Appendix F — Looking Forward

---

# PART I

## Constitutional Governance

---

# Chapter 1

## Why Governance Comes First

Every instructional system answers two fundamental questions.

**What should happen next?**

**How should that decision be communicated?**

Traditional tutoring systems often treat these questions as one. The same mechanism that determines instructional direction also generates the instructional response.

Kaw intentionally rejects this approach.

The architecture begins with a constitutional separation between instructional reasoning and instructional communication.

Instructional reasoning determines the appropriate instructional action.

Communication expresses that predetermined action.

Although closely related, these are fundamentally different responsibilities requiring different forms of governance.

Instructional reasoning must remain completely deterministic.

Every student response should produce an instructional decision that is explainable, reproducible, and consistent with established instructional principles.

Communication, by contrast, benefits from flexibility.

Natural language varies.

Teacher voice varies.

Assignments vary.

Students vary.

Artificial intelligence can adapt to these differences, provided that every adaptation remains bounded by deterministic governance.

This distinction transforms AI from an instructional decision-maker into a governed communication partner.

Rather than deciding what instruction should occur, AI communicates instruction that has already been determined.

This constitutional separation protects instructional consistency while preserving the conversational qualities that make individualized instruction effective.

Everything that follows in this guide builds upon this foundational distinction.

---

# Chapter 2

## Constitutional Principles

Every architectural component described throughout this guide derives from five constitutional principles.

These principles are not implementation details.

They are permanent governance rules that every future implementation must obey.

**Principle One — Deterministic instructional reasoning establishes instructional intent.**

Instructional decisions originate exclusively through deterministic reasoning.

The instructional engine determines instructional location, instructional objective, progression, teaching move, thinking move, validation requirements, and recovery strategy before any communication occurs.

Artificial intelligence never establishes instructional intent.

It communicates instructional intent.

---

**Principle Two — Instruction precedes communication.**

Communication may never determine instruction.

Before any instructional response is generated, the instructional engine has already established the precise instructional action to be communicated.

Instruction is therefore an architectural prerequisite for communication.

---

**Principle Three — The student owns the thinking. Kaw owns the instruction.**

The student remains responsible for every intellectual contribution.

Kaw remains responsible for instructional guidance, progression, validation, clarification, and coaching.

This principle governs every interaction within the system.

Student ownership is never exchanged for instructional efficiency.

---

**Principle Four — Recovery means smaller thinking—not different thinking.**

When students struggle, the instructional objective does not change.

Instead, instructional support becomes smaller, more explicit, or more scaffolded until the student can successfully perform the intended thinking.

Recovery therefore preserves instructional intent while reducing cognitive demand.

Instructional direction remains unchanged.

---

**Principle Five — Every non-deterministic operation must be bounded by deterministic governance.**

Artificial intelligence introduces flexibility into communication.

Flexibility without governance creates inconsistency.

Therefore every non-deterministic operation must operate inside deterministic boundaries established before AI is ever invoked.

These boundaries define what AI may communicate, what it may modify, and what it may never change.

Together, these five principles constitute the constitutional foundation of the Kaw Companion architecture.

Every architectural decision presented in later chapters exists to preserve one or more of these governing principles.

---

# Chapter 3

## Deterministic Instructional Intent

The purpose of deterministic reasoning is not merely consistency.

Its purpose is instructional integrity.

Every instructional decision should be explainable without reference to probability, language generation, or statistical inference.

Instead, each decision should be traceable through explicit instructional evidence.

Student evidence leads to instructional interpretation.

Instructional interpretation identifies the current instructional situation.

The instructional situation determines the appropriate instructional contract.

The instructional contract establishes instructional intent.

Only after instructional intent has been fully established may communication begin.

This sequence guarantees that instruction remains stable regardless of how many different ways the same guidance could be expressed.

It also creates complete architectural traceability.

Every instructional response can be traced backward through deterministic reasoning to the evidence that justified it.

Instruction therefore becomes transparent, reproducible, and governable.

The result is an instructional engine whose behavior is defined by constitutional principles rather than emergent language generation.

---
# Chapter 4

Assignment Understanding Engine   ⭐ NEW

        ↓

Chapter 5
Framing Routine

        ↓

Chapter 6
Evidence State

        ↓

Chapter 7
Instructional Assessment

        ↓

Chapter 8
Instructional Contracts

        ↓

Chapter 9
Teacher Voice

---

# Chapter 5

## Student Ownership and Instructional Responsibility

The defining characteristic of Kaw is not artificial intelligence.

It is protected student ownership.

Students remain the authors of their own thinking.

Kaw never completes intellectual work on a student's behalf.

Instead, Kaw performs the responsibilities traditionally held by an expert teacher.

These responsibilities include observing evidence, diagnosing instructional need, selecting appropriate teaching moves, determining progression, validating readiness, providing feedback, and adjusting instructional support.

Student responsibilities remain equally clear.

Students generate ideas.

Students explain reasoning.

Students construct understanding.

Students revise their own work.

Students ultimately produce the intellectual products required by instruction.

Because these responsibilities remain separated, instructional support never becomes intellectual substitution.

Normalization may improve readability.

Clarification may improve understanding.

Organization may improve coherence.

None of these actions may strengthen incomplete reasoning or introduce ideas the student has not produced.

Whenever student work remains instructionally insufficient after permissible normalization, Kaw continues coaching rather than rewriting.

Instruction continues.

Student ownership remains intact.

---

# Chapter 6

## Governing Artificial Intelligence

Artificial intelligence occupies an essential role within the Kaw architecture.

It is also the most carefully governed component of the system.

AI is intentionally positioned after deterministic instructional reasoning because its purpose is expressive rather than instructional.

Its responsibilities include adapting language to assignment context, maintaining natural teacher voice, varying explanation, improving conversational flow, personalizing encouragement, and contextualizing predetermined instructional moves for the student's current work.

These capabilities create instructional conversations that feel authentic while remaining architecturally constrained.

Equally important are the responsibilities AI does not possess.

Artificial intelligence may never alter instructional goals.

It may never redirect instructional progression.

It may never select a different instructional contract.

It may never advance or delay instructional location.

It may never invent student thinking.

It may never replace student ownership.

Every AI-generated response therefore operates within deterministic boundaries established before communication begins.

In later sections of this guide, these boundaries will be formalized through two complementary architectural layers.

**Communication Governance** defines what AI is permitted to communicate.

**Communication Validation** verifies that the generated communication faithfully preserves the deterministic instructional intent established by the instructional engine.

Together, these layers transform artificial intelligence from an autonomous instructional system into a governed instructional communication system—one that preserves instructional integrity while delivering the flexibility and natural interaction expected from a modern educational companion.

---

**END OF PART I**

**PART II**

# Instructional Architecture

---
Excellent. This is where Version 2.1 really starts.

I'd replace the entire existing **Chapter 6 – The Governed Instructional Engine** with the following.

---

# Chapter 6

## The Governed Instructional Engine

The constitutional principles established in Part I define the permanent rules that govern Kaw Companion.

Instructional Architecture defines how those constitutional principles are carried out.

The Kaw Companion is not fundamentally a conversational AI, a tutoring chatbot, or a language model application. It is a governed instructional engine whose purpose is to execute deterministic instructional reasoning while preserving instructional integrity, student ownership, and constitutional governance.

Every instructional interaction follows the same architectural process.

The system observes student evidence, determines the current instructional situation, selects the appropriate instructional response, and only then constructs the communication through which that predetermined instruction is delivered.

This distinction is foundational.

Instruction determines **what** should occur.

Communication determines **how** that predetermined instruction is expressed.

Artificial intelligence participates only within the governed communication process.

---

## The Purpose of the Instructional Engine

Every instructional model must answer the same fundamental questions.

* What is the student trying to accomplish?
* What evidence has the student provided?
* What does that evidence demonstrate?
* What should happen next instructionally?
* How should that instruction be communicated?

Different instructional frameworks may answer these questions differently.

The governed instructional engine does not.

Its architectural responsibilities remain constant regardless of the instructional framework being implemented.

For the Framing Routine, those responsibilities include:

* establishing assignment understanding,
* interpreting student evidence,
* identifying the current instructional situation,
* selecting the appropriate instructional contract,
* determining instructional intent,
* constructing governed communication, and
* validating instructional continuity throughout the interaction.

Future instructional models may replace the Framing Routine entirely while continuing to use the same governed instructional architecture.

---

## The Governed Instructional Flow

Every instructional interaction follows the same deterministic progression.

```text
Student Assignment
        ↓
Assignment Understanding
        ↓
Student Evidence
        ↓
Accumulated Evidence
        ↓
Instructional Framework Knowledge
        ↓
Instructional Situation
        ↓
Instructional Contract
        ↓
Instructional Intent
        ↓
Communication Specification
        ↓
AI Contextualization
        ↓
Student Communication
        ↓
New Student Evidence
```

Each layer performs one architectural responsibility.

Each layer produces deterministic output for the next.

No layer bypasses another.

Because responsibilities remain separated, every instructional decision remains explainable, testable, and traceable.

---

## Architectural Layers

The instructional engine is composed of distinct architectural layers.

Each layer answers one instructional question before passing responsibility to the next.

| Architectural Layer               | Governing Question                                                                       |
| --------------------------------- | ---------------------------------------------------------------------------------------- |
| Assignment Understanding          | Do we understand the assignment well enough to begin instruction?                        |
| Student Evidence                  | What evidence has the student provided?                                                  |
| Accumulated Evidence              | What instructional understanding already exists?                                         |
| Instructional Framework Knowledge | How is this instructional framework organized?                                           |
| Instructional Situation           | What instructional situation currently exists?                                           |
| Instructional Contract            | What instructional response is required?                                                 |
| Instructional Intent              | What should happen next instructionally?                                                 |
| Communication Specification       | How should that predetermined instruction be communicated?                               |
| AI Contextualization              | How can that communication be expressed naturally while preserving instructional intent? |

Each answer becomes the deterministic input for the next architectural layer.

---

## Deterministic Instruction Before Communication

One of the defining characteristics of the Kaw architecture is that instructional reasoning always concludes before communication begins.

By the time communication is constructed, the instructional engine has already determined:

* the instructional objective,
* the student's current instructional situation,
* the appropriate instructional contract,
* the required teaching move,
* the required thinking move,
* progression requirements,
* validation requirements,
* recovery requirements, when necessary, and
* the instructional intent that communication must preserve.

Communication never determines instruction.

Communication faithfully expresses instruction that has already been determined.

This separation ensures instructional consistency while allowing communication to remain flexible, adaptive, and conversational.

---

## Communication as an Architectural Responsibility

Version 2.1 expands the role of communication within the architecture.

Communication is not simply the generation of instructional language.

It is the governed process through which deterministic instructional intent becomes authentic instructional interaction.

This process begins with a **Communication Specification**.

The Communication Specification captures the instructional intent, instructional constraints, instructional context, validation requirements, and communication goals established by deterministic reasoning.

Artificial intelligence does not receive an instructional problem to solve.

It receives a governed communication specification describing how predetermined instruction should be expressed.

This distinction preserves constitutional governance while allowing communication to remain natural and contextually responsive.

---

## Continuous Instructional Reasoning

Instruction is not a linear sequence of prompts.

It is a continuous cycle of observation, interpretation, instruction, communication, and renewed observation.

Every student response becomes new instructional evidence.

That evidence updates accumulated understanding, informs deterministic instructional reasoning, and produces the next instructional decision.

```text
Student Evidence
        ↓
Instructional Reasoning
        ↓
Instructional Intent
        ↓
Governed Communication
        ↓
Student Response
        ↓
New Student Evidence
```

The cycle continues until the instructional objective has been achieved.

Throughout this process, constitutional governance remains unchanged.

Instruction continues to determine what should occur.

Communication continues to determine how that instruction is expressed.

Student ownership remains protected.

Artificial intelligence remains constitutionally governed.

Together, these architectural responsibilities define the governed instructional engine upon which every present and future implementation of Kaw Companion is built.

---
# Chapter 7

## Assignment Understanding

Before deterministic instructional reasoning can begin, the instructional engine must first establish sufficient understanding of the student's assignment.

This responsibility belongs to the Assignment Understanding layer.

Assignment Understanding serves as the instructional gateway into the governed instructional engine. Every subsequent architectural decision depends upon the quality of understanding established at this stage.

Without sufficient assignment understanding, the instructional engine cannot reliably determine instructional intent. Rather than making instructional assumptions, Kaw continues gathering assignment evidence until sufficient understanding has been established.

Instruction does not begin because a student has submitted work.

Instruction begins because the instructional engine has established enough shared understanding of the assignment to reason deterministically about what should happen next.

---

## Why Assignment Understanding Matters

Every instructional decision depends upon understanding the assignment being taught.

Before Kaw can determine how to help a student, it must understand what the student is trying to accomplish.

This understanding extends beyond simply recognizing the assignment topic.

Assignment Understanding establishes instructional context, identifies the intended learning experience, and provides the evidence necessary for later instructional inference.

Without this context, subsequent architectural layers cannot reliably determine:

- instructional objectives,
- expected student thinking,
- Thinking Tasks,
- Parent and Child Anchor relationships,
- instructional contracts,
- progression requirements,
- validation expectations,
- communication requirements.

Rather than risking incorrect instruction, the architecture requires additional evidence until sufficient understanding exists.

This protects instructional integrity while preserving deterministic reasoning.

---

## Shared Assignment Understanding

Assignment Understanding is not a process of Kaw independently interpreting an assignment.

It is the process through which Kaw and the student establish a shared instructional understanding of the task.

This distinction is fundamental.

The objective is not perfect assignment interpretation.

The objective is sufficient shared understanding for deterministic instructional reasoning to begin.

Students frequently describe assignments incompletely.

Teachers communicate assignments differently.

Assignments themselves vary considerably across classrooms and disciplines.

Rather than assuming missing information, Kaw gathers additional evidence until the instructional objective becomes sufficiently clear.

Instruction begins only after this shared understanding has been established.

---

## Assignment Understanding as an Architectural Layer

Assignment Understanding intentionally performs one architectural responsibility.

It does not determine instructional intent.

It does not select instructional contracts.

It does not evaluate student thinking.

Instead, it answers one constitutional question:

> **Do we understand the assignment well enough to begin deterministic instruction?**

Once that question can be answered affirmatively, responsibility passes to the remainder of the Instructional Architecture.

```text
Student Assignment
        ↓
Assignment Understanding
        ↓
Instructional Knowledge
        ↓
Instructional Reasoning
```

This separation preserves the single responsibility of each architectural layer while ensuring that instructional reasoning always begins with sufficient instructional context.

---

## The Assignment Understanding Validator

Within the current reference implementation, Assignment Understanding is carried out by the **Assignment Understanding Validator (AUV).**

The Assignment Understanding Validator evaluates whether sufficient instructional evidence has been collected to begin deterministic instructional reasoning.

Its responsibility is intentionally limited.

The validator is **not** responsible for perfectly understanding every assignment.

It is responsible for determining whether sufficient instructional evidence exists to justify the next instructional decision.

When sufficient understanding has not yet been established, Kaw continues gathering assignment evidence rather than prematurely beginning instruction.

Future implementations may replace the Assignment Understanding Validator with different mechanisms while preserving this same architectural responsibility.

---

## Evidence State

The Assignment Understanding Validator evaluates several dimensions of instructional readiness.

### Assignment Context

Has Kaw established what the assignment is about?

---

### Assignment Demand

Has Kaw established what the student has been asked to think about, explain, analyze, compare, evaluate, summarize, or otherwise accomplish?

---

### Summary Readiness

Can Kaw accurately summarize the assignment back to the student without introducing unsupported instructional assumptions?

Together, these dimensions determine whether sufficient instructional understanding exists to begin instruction.

---

## Instructional Assessment

Rather than searching for predefined keywords or patterns, the validator evaluates instructional readiness.

Questions include:

- Is the assignment context sufficiently understood?
- Is the instructional demand sufficiently understood?
- Can Kaw accurately summarize the assignment?
- Would subsequent instruction require unsupported assumptions?

These questions focus on instructional sufficiency rather than perfect understanding.

---

## Decision Logic

When sufficient evidence exists:

```text
Assignment Understanding
        ↓
Shared Assignment Summary
        ↓
Student Confirmation
        ↓
Instructional Reasoning Begins
```

When sufficient evidence does not exist:

```text
Incomplete Assignment Understanding
        ↓
Clarification Question
        ↓
Additional Student Evidence
        ↓
Reassessment
```

Clarification continues until sufficient instructional evidence has been collected.

---

## Architectural Role

Assignment Understanding permanently occupies the instructional gateway into the governed instructional engine.

```text
Student Assignment
        ↓
Assignment Understanding
        ↓
Student Evidence
        ↓
Accumulated Evidence
        ↓
Instructional Knowledge
        ↓
Instructional Reasoning
        ↓
Instructional Contract
        ↓
Instructional Intent
        ↓
Communication Specification
        ↓
AI Contextualization
```

No instructional reasoning occurs before this gateway has been satisfied.

---

## Constitutional Significance

Assignment Understanding reinforces several constitutional principles established throughout this guide.

Instruction continues to precede communication.

Deterministic reasoning continues to determine instructional intent.

Student ownership remains protected because Kaw seeks understanding of the student's assignment rather than supplying the student's thinking.

Artificial intelligence remains constitutionally governed because communication cannot begin until deterministic instructional reasoning has been authorized.

An incorrect understanding of the assignment can propagate through every subsequent instructional process, including:

- Thinking Task inference,
- Parent Anchor inference,
- Child Anchor inference,
- Is About validation,
- Main Idea validation,
- Essential Detail validation,
- So What validation,
- instructional contracts,
- instructional communication.

For this reason, Assignment Understanding serves as the constitutional gateway into the governed instructional engine.

Like every governed architectural layer within Kaw Companion, Assignment Understanding answers a single constitutional question:

> **Has sufficient instructional evidence been collected to justify the next instructional decision?**
---

# Chapter 8

## Instructional Knowledge

Assignment Understanding establishes what the student is trying to accomplish.

Instructional Knowledge provides everything the governed instructional engine must know in order to teach that assignment effectively.

Unlike student evidence, which changes continuously throughout an interaction, Instructional Knowledge represents stable instructional expertise.

It describes how an instructional framework is organized, what students are expected to learn, how student evidence should be interpreted, and which instructional responses are appropriate under different instructional situations.

Instructional Knowledge serves as the permanent instructional reference upon which deterministic reasoning depends.

---

## The Role of Instructional Knowledge

Every instructional framework embodies a body of instructional knowledge.

For the Framing Routine, this includes understanding:

- the instructional components,
- the relationships among those components,
- progression expectations,
- validation criteria,
- instructional contracts,
- teaching moves,
- thinking moves, and
- instructional goals.

This knowledge is independent of any particular student.

It exists before instruction begins and remains stable throughout instruction.

As new instructional frameworks are introduced, each contributes its own instructional knowledge while continuing to operate within the same constitutional architecture.

---

## Knowledge Before Reasoning

Instructional reasoning cannot evaluate student evidence without first knowing how the instructional framework is organized.

For example, within the Framing Routine the instructional engine already understands:

- what a Key Topic represents,
- what an Is About statement accomplishes,
- the purpose of Main Ideas,
- how Essential Details support Main Ideas,
- the role of the So What statement,
- acceptable progression paths,
- common misconceptions,
- validation expectations, and
- instructional dependencies between components.

This knowledge exists independently of any individual instructional interaction.

Reasoning applies this knowledge to the evidence provided by the student.

---

## Categories of Instructional Knowledge

Instructional Knowledge can be organized into several categories.

### Structural Knowledge

The organizational structure of the instructional framework.

Examples include:

- instructional components,
- component relationships,
- progression order,
- parent-child dependencies,
- required instructional sequences.

---

### Semantic Knowledge

The instructional meaning of each component.

Examples include:

- component purposes,
- expected thinking,
- instructional objectives,
- conceptual distinctions,
- evidence expectations.

---

### Validation Knowledge

The criteria used to evaluate student evidence.

Examples include:

- success criteria,
- misconceptions,
- acceptable variation,
- required evidence,
- confidence expectations.

---

### Instructional Knowledge

The instructional actions available to the governed instructional engine.

Examples include:

- instructional contracts,
- teaching moves,
- thinking moves,
- celebration opportunities,
- revision strategies,
- recovery pathways.

---

## Instructional Knowledge Supports Deterministic Reasoning

Instructional Knowledge does not determine instruction.

Instead, it provides the knowledge required for deterministic reasoning to evaluate instructional evidence.

The relationship can be understood as:

```text
Instructional Knowledge
            +
Student Evidence
            ↓
Instructional Reasoning
            ↓
Instructional Intent
```

Knowledge provides the instructional framework.

Evidence provides the current instructional situation.

Reasoning combines both to determine instructional intent.

---

## Framework Independence

One of the strengths of the governed instructional architecture is that Instructional Knowledge is framework specific while the architecture itself remains framework independent.

Today, Kaw implements the KU-CRL Framing Routine.

Tomorrow, another instructional model may replace it.

The constitutional architecture remains unchanged.

Only the instructional knowledge changes.

This separation allows new instructional frameworks to be introduced without redesigning the governed instructional engine itself.

---

## Relationship to Subsequent Architecture

Instructional Knowledge prepares deterministic reasoning.

Once Assignment Understanding has established sufficient instructional context, Instructional Knowledge provides the reference against which student evidence can be interpreted.

Only then can the instructional engine determine:

- the current instructional situation,
- the appropriate instructional contract,
- instructional intent,
- and the governed communication required to continue instruction.

Instructional Knowledge therefore serves as the permanent instructional foundation upon which deterministic reasoning depends.
---
# Chapter 9

## Instructional Reasoning

Assignment Understanding establishes instructional context.

Instructional Knowledge provides instructional expertise.

Instructional Reasoning transforms both into deterministic instructional intent.

Instructional Reasoning represents the decision-making process of the governed instructional engine. It evaluates current student evidence, interprets that evidence within the instructional framework, determines the student's present instructional situation, and selects the appropriate instructional response.

Unlike communication, reasoning never attempts to generate instructional language.

Its sole responsibility is determining what should happen next instructionally.

---

## Reasoning as Deterministic Decision Making

Instructional Reasoning exists to answer one question:

> **Given everything currently known, what should happen next instructionally?**

Every instructional decision follows this same deterministic process.

The engine evaluates:

- the current student evidence,
- accumulated instructional understanding,
- the instructional framework,
- progression requirements,
- instructional contracts,
- and constitutional constraints.

From these inputs, it determines one instructional intent.

Because every decision follows explicit instructional rules, reasoning remains explainable, testable, and reproducible.

---

## Evidence Before Reasoning

Instructional Reasoning never begins with assumptions.

It begins with evidence.

Current student responses represent only part of that evidence.

The instructional engine also considers accumulated instructional understanding developed throughout the interaction.

Together, these form the Evidence State.

```text
Current Student Evidence
            +
Accumulated Evidence
            ↓
Evidence State
```

The Evidence State represents the complete instructional picture available at the moment reasoning occurs.

---

## Instructional Assessment

Reasoning first evaluates the Evidence State.

This assessment determines what the evidence demonstrates instructionally.

Assessment includes questions such as:

- Has sufficient evidence been provided?
- Has the instructional objective been achieved?
- Is the student's thinking progressing?
- Is revision required?
- Is clarification needed?
- Has a misconception emerged?
- Is recovery required?
- Is celebration appropriate?

Assessment describes the instructional situation.

It does not yet determine the instructional response.

---

## Instructional Strategy

Once the instructional situation has been established, the instructional engine determines the instructional strategy required to continue learning.

Examples include:

- continue progression,
- request additional evidence,
- clarify misunderstanding,
- revise student thinking,
- celebrate success,
- recover from misconceptions,
- reinforce learning,
- transition to the next instructional objective.

This strategy remains entirely deterministic.

No communication has yet occurred.

---

## Instructional Intent

Instructional Strategy produces Instructional Intent.

Instructional Intent represents the complete instructional decision.

It defines:

- the instructional objective,
- the instructional contract,
- the required teaching move,
- the required thinking move,
- progression requirements,
- communication goals,
- validation requirements,
- and instructional constraints.

Once Instructional Intent has been established, deterministic reasoning is complete.

Responsibility passes to the Communication Architecture.

---

## The Reasoning Flow

Instructional Reasoning can be understood as the following architectural progression.

```text
Evidence State
        ↓
Instructional Assessment
        ↓
Instructional Strategy
        ↓
Instructional Intent
        ↓
Communication Specification
```

Each stage narrows the instructional decision.

Assessment determines what the evidence means.

Strategy determines how instruction should respond.

Instructional Intent defines precisely what communication must preserve.

---

## Separation from Communication

Instructional Reasoning intentionally ends before communication begins.

This separation is one of the defining characteristics of the governed instructional architecture.

Reasoning determines:

- what instruction should occur,
- why it should occur,
- and what constraints communication must preserve.

Communication determines only how that predetermined instruction should be expressed.

Because these responsibilities remain separate, improvements in communication never alter instructional decision making.

Likewise, improvements in instructional reasoning never require changes to communication.

The two architectures evolve independently while remaining constitutionally aligned.

---

## Continuous Instructional Decision Making

Instructional Reasoning operates continuously throughout every instructional interaction.

Each student response produces new evidence.

That evidence updates the Evidence State.

The instructional engine reassesses the instructional situation, determines the next instructional strategy, establishes a new instructional intent, and returns responsibility to the Communication Architecture.

The cycle repeats until the instructional objective has been achieved.

In this way, Instructional Reasoning serves as the deterministic decision-making core of the governed instructional engine.

---
# Chapter 10

## Instructional Contracts

Instructional Reasoning determines what should happen next instructionally.

Instructional Contracts define how that instructional decision is carried out while preserving constitutional governance.

An Instructional Contract is a deterministic instructional policy that specifies the appropriate instructional response for a particular instructional situation. Contracts translate instructional reasoning into governed instructional intent while ensuring that identical instructional situations produce consistent instructional outcomes.

Instructional Contracts therefore serve as the bridge between instructional reasoning and instructional communication.

---

## Why Instructional Contracts Exist

Instructional reasoning identifies the student's instructional situation.

Instructional Contracts determine the appropriate instructional response.

Without contracts, instructional reasoning would need to generate instructional behavior directly.

Instead, reasoning identifies *what* instructional situation exists, and the corresponding contract defines *how the governed instructional engine should respond.*

This separation produces a system that is explainable, testable, extensible, and constitutionally governed.

---

## Deterministic Instructional Responses

Every contract represents one deterministic instructional response.

Examples include:

- gathering additional evidence,
- celebrating successful thinking,
- coaching revision,
- clarifying misconceptions,
- supporting recovery,
- transitioning to the next instructional objective,
- reinforcing successful reasoning.

Each contract defines the instructional purpose of the interaction.

Communication later determines how that purpose is expressed.

---

## Responsibilities of an Instructional Contract

Every Instructional Contract specifies:

- the instructional objective,
- the required teaching move,
- the required thinking move,
- progression expectations,
- validation requirements,
- communication goals,
- instructional constraints,
- completion conditions.

The contract does not generate language.

It generates instructional intent.

---

## Relationship to Instructional Intent

Instructional Contracts produce Instructional Intent.

Instructional Intent represents the complete deterministic description of what communication must preserve.

```text
Instructional Situation
        ↓
Instructional Contract
        ↓
Instructional Intent
        ↓
Communication Specification
```

By separating contracts from communication, the architecture allows communication to evolve independently without altering instructional decision making.

---

## Framework Independence

Instructional Contracts belong to the instructional framework rather than the constitutional architecture.

The Framing Routine defines one collection of contracts.

Future instructional frameworks may define different contracts while preserving the same constitutional principles.

Because contracts remain framework specific, new instructional models can be introduced without redesigning the governed instructional engine.

---

## Extensibility

One of the strengths of the contract architecture is its extensibility.

New instructional situations can be supported by introducing additional contracts without modifying existing deterministic reasoning.

Likewise, communication improvements do not require changes to contractual behavior.

This separation allows the instructional engine to mature incrementally while maintaining constitutional stability.

---

## Constitutional Role

Instructional Contracts reinforce one of the central constitutional principles of the Kaw architecture:

Instruction precedes communication.

By the time communication begins, the instructional engine has already determined:

- why instruction is occurring,
- what instructional objective should be pursued,
- what instructional constraints must be preserved,
- and what instructional outcome communication should achieve.

Communication remains adaptive.

Instruction remains deterministic.

Instructional Contracts provide the constitutional bridge between these two architectural responsibilities.

---

# Chapter 10

## The Two-Gate Progression Model

Progression represents one of the most important responsibilities of the instructional engine.

Students should advance only when instructionally appropriate.

They should also avoid unnecessary repetition once sufficient understanding has been demonstrated.

Version 1.1 introduced the Two-Gate Progression Model to formalize this decision. That model remains a foundational element of the constitutional architecture. 

Rather than asking a single question—

"Is the student's answer correct?"

—the architecture asks two separate instructional questions.

These questions protect both instructional accuracy and instructional continuity.

---

**Gate One — Validation**

The first gate determines whether the student has demonstrated the required instructional evidence for the current instructional contract.

Validation evaluates whether required expectations have been satisfied.

If validation fails, instruction remains at the current instructional location.

Appropriate recovery strategies are selected while preserving instructional intent.

---

**Gate Two — Instructional Sufficiency**

Passing validation alone is not sufficient for progression.

The second gate determines whether accepted student work provides a stable instructional foundation for the next instructional objective.

Instructional sufficiency does not require perfection.

Instead, it asks whether the student's work:

• communicates a coherent idea;

• preserves the student's intended meaning;

• provides a stable instructional foundation capable of supporting subsequent instruction.

This distinction allows students to continue learning without requiring unnecessary refinement while simultaneously protecting future instructional success. 

---

## When Either Gate Fails

Failure at either gate produces the same progression decision.

The instructional engine does not advance.

Instead, Kaw remains at the current instructional location while adjusting instructional support.

Recovery therefore modifies instructional support rather than instructional direction.

Students continue working toward the same instructional objective.

Instructional continuity remains intact.

---

## Progression as Governance

The Two-Gate Progression Model illustrates the central philosophy of the Kaw architecture.

Progression is not determined by conversational quality.

It is not determined by AI confidence.

It is not determined by linguistic elegance.

Progression is governed exclusively through deterministic instructional reasoning.

Only after both gates have been satisfied may the instructional engine authorize movement to the next instructional objective.

In this way, progression remains one of the strongest constitutional protections within the entire system.

---

**END OF PART II**

**PART III**

# Communication Architecture

---
# Chapter 12

## Communication Governance

Instructional Reasoning concludes when Instructional Intent has been established.

From that point forward, responsibility transfers to the Communication Architecture.

The purpose of the Communication Architecture is not to determine instruction.

Its purpose is to faithfully express predetermined instructional intent while preserving constitutional governance.

Communication therefore becomes an architectural responsibility rather than a language generation task.

Every communication produced by Kaw must satisfy one requirement:

**It must preserve the instructional intent established by deterministic reasoning.**

---

## Communication as a Governed Process

Communication is intentionally separated from instructional reasoning.

Reasoning determines:

- what should happen,
- why it should happen,
- instructional constraints,
- progression,
- validation,
- and instructional goals.

Communication determines only how those predetermined decisions are expressed.

This separation allows communication to remain flexible without allowing it to alter instruction.

---

## Responsibilities of Communication Governance

Communication Governance ensures that communication always remains faithful to instructional intent.

Its responsibilities include:

- preserving instructional objectives,
- preserving student ownership,
- preserving constitutional constraints,
- preserving instructional progression,
- expressing appropriate teacher voice,
- adapting communication to context,
- preventing instructional drift.

Communication therefore serves instruction.

It never replaces it.

---

## Communication Begins With Deterministic Intent

Communication never begins with a blank page.

It begins with deterministic instructional intent already established by the instructional engine.

Communication therefore becomes an act of translation rather than decision making.

The instructional engine decides.

Communication expresses.

---

## Constitutional Role

Communication Governance preserves one of the central constitutional principles of Kaw Companion.

Instruction determines **what** should occur.

Communication determines **how** that predetermined instruction is expressed.

Artificial intelligence participates only within these governed communication boundaries.

---

# Chapter 13

## Communication Goals and Communication Taxonomy

Every instructional interaction serves a communication goal.

Communication Goals describe the instructional purpose communication is intended to achieve.

They do not determine instruction.

They organize how predetermined instruction should be expressed.

---

## Communication Goals

Common Communication Goals include:

- Introduce
- Orient
- Prompt
- Clarify
- Gather Evidence
- Coach Revision
- Reinforce Thinking
- Celebrate Progress
- Transition
- Check Understanding
- Recover
- Conclude

These goals describe instructional communication patterns rather than instructional decisions.

---

## Communication Taxonomy

Communication within Kaw Companion can be organized into several categories.

### Instructional Communication

Communication that advances student thinking.

Examples include:

- asking instructional questions,
- guiding revision,
- prompting reflection,
- extending understanding.

---

### Supportive Communication

Communication that encourages persistence while maintaining instructional focus.

Examples include:

- celebrating progress,
- reinforcing effort,
- acknowledging improvement.

---

### Transitional Communication

Communication that moves students between instructional stages.

Examples include:

- introducing a new component,
- transitioning to the next thinking task,
- summarizing progress.

---

### Recovery Communication

Communication used when instructional recovery is required.

Examples include:

- simplifying thinking,
- gathering additional evidence,
- repairing misconceptions,
- restoring instructional continuity.

---

## Communication Goals Support Instruction

Communication Goals never replace instructional reasoning.

Instead they organize how predetermined instructional intent is communicated.

Multiple instructional contracts may share the same Communication Goal.

Likewise, one Communication Goal may support many instructional situations.

This separation allows instructional reasoning and communication to evolve independently.

---

# Chapter 14

## Communication Specification

The Communication Specification represents the formal boundary between deterministic instructional reasoning and adaptive communication.

Once Instructional Intent has been established, the instructional engine constructs a Communication Specification describing everything communication must preserve.

Artificial intelligence never determines instructional intent directly.

It receives a Communication Specification.

---

## Purpose

The Communication Specification ensures that adaptive communication remains constitutionally governed.

Rather than asking artificial intelligence to determine instruction, the instructional engine provides a complete description of the instructional communication that should occur.

This preserves deterministic instructional decision making while allowing communication to remain natural, flexible, and contextually responsive.

---

## Components of the Communication Specification

A Communication Specification typically contains:

- instructional intent,
- communication goal,
- instructional objective,
- teaching move,
- thinking move,
- instructional constraints,
- validation requirements,
- support level,
- accumulated instructional context,
- current student evidence,
- communication context.

Future implementations may expand these elements while preserving the same architectural responsibility.

---

## Architectural Role

The Communication Specification separates reasoning from communication.

```text
Instructional Reasoning
        ↓
Instructional Intent
        ↓
Communication Specification
        ↓
AI Contextualization
        ↓
Student Communication
```

Once the specification has been created, deterministic reasoning is complete.

Communication becomes responsible for expressing—not changing—the instructional decision.

---

## Benefits

Introducing the Communication Specification provides several architectural advantages.

It:

- preserves deterministic instructional reasoning,
- creates a stable boundary between reasoning and communication,
- improves explainability,
- improves traceability,
- supports validation,
- allows communication to evolve independently,
- simplifies future implementations.

---

## Constitutional Significance

The Communication Specification reinforces the constitutional separation between instruction and communication.

Instruction remains deterministic.

Communication remains adaptive.

Artificial intelligence never determines what instruction should occur.

Instead, it faithfully contextualizes instruction that has already been determined.

The Communication Specification therefore serves as the constitutional bridge between the Instructional Architecture and the Communication Architecture.
---
# Chapter 15

## AI Contextualization

Once a Communication Specification has been constructed, responsibility passes to artificial intelligence.

At this point, instructional reasoning has concluded.

Artificial intelligence does not determine instructional intent, modify instructional objectives, or select instructional strategies.

Its sole responsibility is to transform the governed Communication Specification into authentic instructional communication while faithfully preserving every instructional constraint established by the deterministic instructional engine.

Artificial intelligence therefore serves as an expressive component of the architecture rather than an instructional decision maker.

---

## Contextualization Rather Than Instruction

Artificial intelligence receives communication that has already been determined.

Its responsibility is to contextualize—not create—the instructional interaction.

This includes adapting communication to:

- the student's assignment,
- the student's current evidence,
- accumulated instructional context,
- support level,
- instructional history,
- teacher voice,
- conversational flow.

Although the wording may differ from one interaction to another, the instructional purpose always remains identical.

---

## Adaptive Expression

Because instructional intent has already been established, communication may adapt naturally without risking instructional inconsistency.

Examples of adaptive communication include:

- selecting appropriate examples,
- adjusting explanation depth,
- varying sentence structure,
- responding naturally to student wording,
- maintaining conversational continuity.

These adaptations improve communication without altering instruction.

---

## Constitutional Constraints

Artificial intelligence must never:

- change instructional intent,
- skip instructional progression,
- alter validation requirements,
- replace instructional contracts,
- assume student thinking,
- violate student ownership.

Every generated response remains bounded by the Communication Specification.

These constitutional boundaries preserve instructional consistency while allowing authentic conversation.

---

## Architectural Role

AI Contextualization represents the final adaptive step before communication reaches the student.

```text
Instructional Intent
        ↓
Communication Specification
        ↓
AI Contextualization
        ↓
Teacher Communication
```

Instruction remains deterministic.

Communication remains adaptive.

The separation between these responsibilities preserves both instructional integrity and conversational authenticity.

---

# Chapter 16

## Communication Validation

Producing communication is not the final responsibility of the Communication Architecture.

Every communication must also be validated.

Communication Validation verifies that instructional communication faithfully preserves the Communication Specification established by deterministic reasoning.

Validation therefore protects constitutional governance after communication has been generated.

---

## Purpose

Communication Validation answers one question:

**Does this communication faithfully express the predetermined instructional intent?**

Validation evaluates communication independently of its writing quality.

A beautifully written response that violates instructional intent fails validation.

Likewise, a simple response that faithfully preserves instructional intent succeeds.

---

## Validation Responsibilities

Communication Validation verifies that communication:

- preserves instructional intent,
- preserves instructional objectives,
- preserves teaching moves,
- preserves thinking moves,
- preserves instructional constraints,
- preserves student ownership,
- preserves instructional progression,
- preserves constitutional governance.

Only communication satisfying these requirements should reach the student.

---

## Communication Quality

Validation also evaluates communication quality.

Examples include:

- clarity,
- coherence,
- instructional continuity,
- contextual relevance,
- support-level fidelity,
- consistency of teacher voice.

These qualities improve instructional communication while remaining secondary to instructional fidelity.

---

## Continuous Validation

Communication Validation operates continuously.

Every instructional interaction produces a new Communication Specification.

Every Communication Specification produces new communication.

Every communication is evaluated before responsibility returns to the instructional engine.

This continuous validation preserves instructional consistency throughout the entire instructional cycle.

---

## Constitutional Role

Communication Validation provides the final safeguard protecting deterministic instructional reasoning.

It ensures that adaptive communication never becomes adaptive instruction.

Instruction remains governed.

Communication remains expressive.

The constitutional separation established in Part I therefore remains intact throughout every instructional interaction.

---

# Chapter 17

## Teacher Voice

Teacher Voice represents the observable expression of the governed instructional architecture.

Students experience Kaw primarily through its communication.

Teacher Voice therefore determines how governed instruction feels without determining what instruction occurs.

Teacher Voice is an outcome of the architecture rather than an independent instructional system.

---

## Purpose

The purpose of Teacher Voice is to communicate deterministic instructional intent in a manner that feels supportive, authentic, and instructionally purposeful.

Every response should sound like an experienced teacher guiding student thinking.

Communication should remain encouraging without becoming permissive.

Supportive without becoming dependent.

Conversational without sacrificing instructional precision.

---

## Characteristics of Teacher Voice

Teacher Voice consistently demonstrates:

- instructional clarity,
- warmth,
- encouragement,
- professionalism,
- respect for student ownership,
- confidence,
- instructional consistency.

The voice should remain recognizable regardless of assignment, support level, or instructional framework.

---

## Preserving Student Ownership

Teacher Voice never completes student thinking.

Instead, it supports students as they perform their own thinking.

Effective Teacher Voice:

- asks purposeful questions,
- guides reflection,
- encourages revision,
- celebrates authentic progress,
- reinforces successful thinking,
- maintains appropriate cognitive demand.

Students remain responsible for producing instructional work.

---

## Support-Level Expression

Teacher Voice expresses the Support Level determined by deterministic reasoning.

High Support provides more explanation, modeling, and guidance.

Moderate Support provides concise reminders and instructional cues.

Low Support provides brief prompts that encourage independent thinking.

Support Level changes the communication.

It never changes the instructional objective.

---

## Instructional Continuity

Teacher Voice maintains continuity throughout instruction.

Communication should:

- acknowledge previous work,
- build upon accumulated evidence,
- explain why the next thinking step matters,
- reinforce instructional progress,
- preserve the terminology of the instructional framework.

Students should experience one continuous instructional conversation rather than isolated prompts.

---

## Teacher Voice as Architectural Expression

Teacher Voice is the final expression of every architectural layer that precedes it.

Constitutional Governance protects instructional principles.

Instructional Architecture determines instructional intent.

Communication Architecture governs instructional communication.

Teacher Voice brings that governed communication to life.

When students interact with Kaw, they encounter the architecture through Teacher Voice.

For that reason, Teacher Voice is not merely a stylistic choice.

It is the visible expression of constitutional governance, deterministic instructional reasoning, and governed instructional communication working together as a single instructional system.
---
# Chapter 18

## Instructional State

Instructional State represents the complete instructional understanding maintained throughout an interaction.

Unlike individual student responses, Instructional State accumulates instructional understanding over time. It preserves continuity, supports deterministic reasoning, and enables instruction to build upon previous evidence rather than treating each interaction independently.

Instructional State includes:

- Assignment Understanding
- Current Student Evidence
- Accumulated Evidence
- Progression Status
- Instructional Situation
- Active Instructional Contract
- Current Instructional Intent
- Communication Context

Instructional State is continuously updated as new evidence is observed and serves as the single instructional reference for deterministic reasoning.

---

# Chapter 19

## Evidence Processing

Evidence Processing transforms student responses into instructional evidence.

Rather than evaluating responses in isolation, Evidence Processing integrates current evidence with accumulated instructional understanding.

Evidence Processing supports:

- evidence collection,
- evidence interpretation,
- evidence accumulation,
- validation,
- instructional assessment.

Evidence Processing does not determine instruction.

It prepares evidence for deterministic instructional reasoning.

---

# Chapter 20

## Assignment Context

Assignment Context provides the instructional background required to interpret student work.

Assignment Context may include:

- assignment description,
- learning objectives,
- instructional framework,
- teacher expectations,
- supporting materials,
- prior instructional history.

Assignment Context supports Assignment Understanding but remains distinct from student evidence.

It describes the instructional environment rather than student thinking.

---

# Chapter 21

## Extending the Architecture

The constitutional architecture intentionally separates permanent governance from instructional implementation.

Future instructional frameworks may introduce:

- new instructional models,
- new instructional contracts,
- new validation systems,
- new communication strategies,
- new instructional domains.

Because constitutional governance remains unchanged, these additions require only framework-specific knowledge rather than architectural redesign.

This separation allows Kaw Companion to evolve while preserving constitutional stability.

---

# Chapter 22

## Future Instructional Models

The Framing Routine represents the first complete instructional implementation of the governed instructional engine.

Future instructional frameworks may include:

- writing instruction,
- mathematics,
- scientific inquiry,
- historical analysis,
- project-based learning,
- disciplinary literacy,
- other structured instructional models.

Each framework contributes new instructional knowledge while operating within the same constitutional architecture.

The governed instructional engine therefore serves as a permanent instructional platform rather than an implementation tied to a single instructional framework.
---


**END OF PART IV**

**PART V**

# Appendices

---

The Kaw Companion Constitution establishes five permanent principles.

1. Deterministic instructional reasoning establishes instructional intent.

2. Instruction precedes communication.

3. The student owns the thinking.
   Kaw owns the instruction.

4. Recovery means smaller thinking—not different thinking.

5. Every adaptive operation remains bounded by deterministic governance.

These principles define the permanent constitutional foundation upon which every implementation depends.
---

**END OF PART V**

---

# END OF OFFICIAL MANUSCRIPT


