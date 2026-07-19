KAW COMPANION
Architecture & System Design Guide
Instructional Reasoning, AI Governance, and System Engineering

Version: 1.0
Status: Official Architecture Specification
Project: Kaw Companion
Maintained By: KU-CRL AI Research Project
Last Updated: July 2026

Computer science provides the architecture. Instructional science provides the intelligence. Artificial intelligence provides the voice.

Guiding Principle

Deterministic instructional reasoning establishes instructional truth.

Artificial Intelligence communicates that instructional truth naturally while preserving teacher voice and student ownership.

Everything within the Kaw architecture flows from this principle.

The instructional engine—not artificial intelligence—is responsible for understanding assignments, interpreting student evidence, selecting instructional goals, determining Teaching Moves and Thinking Moves, validating learning, protecting student work, and governing progression.

Artificial intelligence serves a different purpose.

Its responsibility is to communicate deterministic instructional decisions in a natural, conversational manner that reflects teacher voice, maintains assignment context, and supports student thinking without replacing it.

This separation between instructional reasoning and language generation defines the architecture of Kaw and distinguishes it from traditional AI tutoring systems.

Executive Summary
Purpose

Kaw Companion is an instructional reasoning system designed to support student thinking through the KU-CRL Framing Routine.

Unlike traditional AI tutoring systems, Kaw does not rely on artificial intelligence to determine instructional decisions. Instead, it separates instructional reasoning from language generation through a deterministic instructional engine that evaluates student evidence, determines the next instructional move, and authorizes AI to communicate that decision.

The result is an instructional system that remains explainable, testable, consistent, and grounded in established instructional practice while still providing students with natural, personalized coaching.

Architectural Philosophy

The architecture is founded on a simple principle:

Instruction should be deterministic. Communication should be natural.

Rather than asking an AI model to decide how to teach, Kaw first establishes instructional understanding through structured reasoning.

Only after instructional decisions have been made does AI generate the student-facing response.

This sequence preserves instructional integrity while leveraging the strengths of modern language models.

Architectural Layers

Each layer depends on the integrity of the layers preceding it.

Instructional understanding always precedes instructional communication.

Kaw at a Glance
End-to-End Architecture
Instructional Flow
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

This instructional cycle governs every interaction within Kaw.

Core Principles

Every architectural decision should strengthen these principles.

Instruction Before Language

Instructional reasoning always precedes AI communication.

Evidence Before Assumption

Observable student evidence is the foundation of every instructional decision.

One Thinking Step

Kaw advances one meaningful cognitive step at a time.

Build from Student Success

Valid student thinking is preserved and extended rather than replaced.

Preserve Student Ownership

Students remain responsible for the intellectual work.

Preserve Instructional Continuity

Instruction builds across Assignment Context, Parent Anchors, Child Anchors, and validated student work.

Deterministic Instructional Truth

Instructional decisions belong to the deterministic instructional engine.

Explainability

Every instructional decision should be traceable and explainable.

Governance over Convenience

Architectural convenience should never compromise instructional integrity.

Table of Contents
Part I — Foundations
Project Overview
Design Philosophy
Kaw Instructional Commitments
Kaw Instructional Model
Part II — Instructional Reasoning
Instructional Knowledge
Assignment Context
Thinking Tasks
Parent and Child Anchors
Instructional Decision Cycle
Part III — Instructional Decisions
Instructional Contracts
Validation Framework
Student Progression
Part IV — AI Communication
AI Governance
Communication Licensing
Prompt Construction
Runtime Communication
Part V — System Engineering
System Architecture
Application Flow
Data Structures
Testing Framework
Deployment
Development Standards
Architecture Decision Records
Roadmap
Appendices

A. Complete Student Walkthrough

B. Example Instructional Contracts

C. Glossary

D. Repository Structure

E. Contributor Workflow
