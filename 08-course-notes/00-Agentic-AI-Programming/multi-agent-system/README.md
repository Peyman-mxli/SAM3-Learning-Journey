# Multi-Agent AI System

**Course:** SAM3 — Computer Vision with Segment Anything Model 3  
**Topic:** Agentic AI Programming  
**Section:** Multi-Agent System Architecture

---

## Overview

A **Multi-Agent System** consists of multiple specialized AI agents working together to accomplish a larger objective.

Instead of asking one AI agent to perform every task, responsibilities can be divided among specialized agents.

Each agent can have its own:

- **Role** — What the agent is responsible for
- **Goal** — What the agent must accomplish
- **Traits** — Knowledge, expertise, and expected behavior
- **Constraints** — Rules and limitations the agent must follow
- **Tools** — Capabilities the agent can use

The agents can collaborate as part of a larger workflow, with the output of one agent becoming the input of another.

### General Architecture

```text
                    USER / OBJECTIVE
                           │
                           ▼
                      ARCHITECT
                           │
                    Designs the plan
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    SIMULATION          MACHINE            QA &
     ENGINEER           LEARNING        EVALUATION
                        ENGINEER         SPECIALIST
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                       FINAL RESULT
```

---

## Agents in the System

The example introduced in class contains several specialized agents:

1. **The Architect**
2. **The Simulation Engineer (`@sim_engineer`)**
3. **The Machine Learning Engineer (`@ml_engineer`)**
4. **The QA & Evaluation Specialist (`@qa`)**

Each agent will be documented below according to the specifications presented during the class.

---
