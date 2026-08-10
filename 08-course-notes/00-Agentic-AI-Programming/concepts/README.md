# Agentic AI — Key Concepts

This document contains the key concepts and terminology introduced
during Topic 0 of the SAM3 course.

---

## 1. Agent

### Definition

An **AI Agent** is an AI-based system that receives an objective,
interprets the context, decides which actions to perform, and uses
tools to advance toward that objective with a certain degree of
autonomy.

### Original Class Definition

> Un sistema basado en IA que recibe un objetivo, interpreta el
> contexto, decide qué acciones realizar y utiliza herramientas para
> avanzar hacia ese objetivo con cierto grado de autonomía.

### In Simple Terms

An AI agent does more than simply answer a question.

It can:

- Receive a goal
- Understand the available context
- Decide what should happen next
- Use available tools
- Perform actions
- Evaluate results
- Continue working toward the objective

A simplified workflow is:

```text
Goal
  ↓
Understand Context
  ↓
Reason / Decide
  ↓
Select an Action
  ↓
Use a Tool
  ↓
Observe the Result
  ↓
Continue Toward the Goal

Example

A Computer Vision AI agent could receive the objective:

"Analyze this image and identify the objects."

The agent could then:

Receive the image.
Determine which Computer Vision tool is needed.
Run the appropriate model.
Analyze the model output.
Generate or save the results.
Report the final answer.


## 2. MCP — Model Context Protocol

### Definition

**MCP (Model Context Protocol)** is a standard protocol that allows AI models and agents to connect with external tools through a common interface.

It enables an AI agent to **discover and use available capabilities without requiring a custom integration for every individual tool**.

### Original Class Definition

> Model Context Protocol. Es un protocolo estándar para conectar modelos/agentes con herramientas mediante una interfaz común. Permite que un agente descubra y use capacidades sin tener una integración específica para cada herramienta.

### In Simple Terms

MCP provides a standardized way for an AI agent to communicate with external tools.

Instead of creating a completely different connection for every service:

```text
AI Agent ─── Custom Integration ─── Tool A
AI Agent ─── Custom Integration ─── Tool B
AI Agent ─── Custom Integration ─── Tool C
```

MCP provides a common communication layer:

```text
                 ┌── Tool A
                 │
AI Agent ── MCP ─┼── Tool B
                 │
                 └── Tool C
```

### Why Is MCP Useful?

MCP can allow an AI agent to:

- Discover available tools
- Access external capabilities
- Retrieve information
- Execute supported actions
- Work with different services through a standardized interface
- Reduce the need for separate custom integrations

### Example

Imagine an AI agent working on a Computer Vision project.

The agent may need access to:

```text
AI Agent
   ↓
MCP
   ↓
┌──────────────┬──────────────┬──────────────┐
│              │              │              │
Dataset      Files         Database      CV Tools
│              │              │              │
└──────────────┴──────────────┴──────────────┘
                       ↓
                    Result
```

Instead of manually building a unique connection for every tool, MCP can provide a standardized mechanism for exposing those capabilities to the AI application.

### Key Idea

> **Agent = decides what needs to be done.**  
> **MCP = provides a standardized way to connect the agent with tools and capabilities.**

---
## 3. Skill

### Definition

A **Skill** is a specialized and reusable capability that teaches an AI agent how to perform a particular type of task.

A skill can contain instructions, procedures, and rules that guide the agent in deciding **how to complete the task and which tools to use**.

### Original Class Definition

> Una capacidad especializada y reutilizable que le enseña al agente cómo realizar cierto tipo de tarea. Puede contener instrucciones, procedimientos y reglas sobre qué herramientas utilizar.

### In Simple Terms

A skill can be thought of as a set of specialized instructions that teaches an AI agent **how to do something correctly**.

The agent may be capable of reasoning and making decisions, but a skill provides specific knowledge about how a particular task should be performed.

```text
AI Agent
   ↓
Selects a Skill
   ↓
Skill provides:
   ├── Instructions
   ├── Procedures
   ├── Rules
   └── Tool Guidance
   ↓
Agent performs the task
```

### Why Are Skills Useful?

Skills can help AI agents:

- Perform specialized tasks
- Follow predefined procedures
- Apply consistent rules
- Select appropriate tools
- Reuse knowledge across similar tasks
- Complete complex workflows more reliably

### Example

Imagine an AI agent working on a Computer Vision project.

The agent receives the objective:

> "Analyze this image and identify the objects."

The agent could use a **Computer Vision Skill** containing instructions such as:

```text
Computer Vision Skill
        ↓
1. Load the image
        ↓
2. Validate the image format
        ↓
3. Select the appropriate model
        ↓
4. Run inference
        ↓
5. Process predictions
        ↓
6. Visualize the results
        ↓
7. Return the final output
```

The skill provides the specialized procedure while the agent decides when and why to use it.

### Relationship Between Agent, MCP, and Skill

The three concepts introduced so far work together:

```text
                    AI AGENT
                        │
              Decides what to do
                        │
                        ▼
                      SKILL
                        │
             Explains how to do it
                        │
                        ▼
                       MCP
                        │
          Connects the agent to tools
                        │
                        ▼
                     TOOLS
```

### Key Idea

> **Agent = decides what needs to be done.**  
> **Skill = provides specialized knowledge about how to do it.**  
> **MCP = provides a standardized way to connect with the tools needed to do it.**

---

## 4. Guardrail

### Definition

A **Guardrail** is a rule or control mechanism that restricts, validates, or guides the behavior of an AI agent.

Guardrails help ensure that the agent operates within defined boundaries, such as security requirements, permissions, safety rules, organizational policies, or business rules.

### Original Class Definition

> Una regla o mecanismo de control que restringe o valida el comportamiento del agente para mantenerlo dentro de ciertos límites, políticas de negocio, etc.

### In Simple Terms

A guardrail defines **what an AI agent is allowed or not allowed to do**.

The agent may be capable of performing many actions, but guardrails establish boundaries around those actions.

```text
             AI Agent
                 │
                 ▼
            Proposed Action
                 │
                 ▼
             GUARDRAIL
                 │
          ┌──────┴──────┐
          │             │
       Allowed        Blocked
          │             │
          ▼             ▼
     Execute Action   Stop / Reject
```

### Why Are Guardrails Important?

Guardrails can help:

- Prevent unauthorized actions
- Enforce security requirements
- Validate agent decisions
- Protect sensitive information
- Enforce business policies
- Control access to tools
- Reduce unsafe or unintended behavior
- Define operational boundaries

### Example

Imagine an AI agent that can access a company's database.

The agent receives this request:

> "Delete all customer records."

Before performing the action, a guardrail could check whether the agent has permission to delete data.

```text
User Request
     ↓
AI Agent
     ↓
Wants to delete records
     ↓
Guardrail Check
     ↓
Does the agent have permission?
     ↓
 ┌───┴───┐
 │       │
YES      NO
 │       │
 ▼       ▼
Allow   Block
```

If the agent does not have the required permission, the guardrail prevents the action.

### Computer Vision Example

Suppose a Computer Vision agent can analyze images and upload results to an external service.

A guardrail could specify:

```text
IF image contains sensitive information:
    DO NOT upload the image externally.
ELSE:
    Continue with the workflow.
```

This allows the agent to operate autonomously while still respecting defined restrictions.

### Relationship Between Agent, Skill, MCP, and Guardrail

```text
                    AI AGENT
                        │
                 Decides what to do
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
           SKILL               GUARDRAIL
      How to perform          What is allowed
         the task             or restricted
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                       MCP
                        │
              Connects to tools
                        │
                        ▼
                      TOOLS
```

### Key Idea

> **Agent = decides what needs to be done.**  
> **Skill = provides specialized knowledge about how to do it.**  
> **MCP = provides a standardized way to connect with tools.**  
> **Guardrail = defines the rules and boundaries the agent must respect.**

---

## 5. Workflow

### Definition

A **Workflow** is a defined sequence of steps, decisions, and actions designed to complete a process.

Unlike the more autonomous behavior of an AI agent, a workflow follows a **more explicit, structured, and predictable path**.

### Original Class Definition

> Una secuencia definida de pasos, decisiones y acciones para completar un proceso. A diferencia del comportamiento de un agente, un workflow sigue una estructura más explícita y predecible.

### In Simple Terms

A workflow describes **the sequence in which tasks should happen**.

Instead of allowing the AI agent to independently decide every next step, the workflow establishes a predefined process.

```text
Start
  ↓
Step 1
  ↓
Step 2
  ↓
Decision
  ↓
┌───────┴───────┐
│               │
YES             NO
│               │
▼               ▼
Step 3        Step 4
│               │
└───────┬───────┘
        ↓
      Result
```

### Why Are Workflows Useful?

Workflows help:

- Organize complex processes
- Define the order of operations
- Create predictable behavior
- Coordinate multiple tasks
- Define decision points
- Automate repetitive processes
- Make systems easier to understand and debug
- Combine AI agents with traditional automation

### Example

Imagine a Computer Vision application that processes uploaded images.

The workflow could be:

```text
User Uploads Image
        ↓
Validate Image
        ↓
Preprocess Image
        ↓
Run Computer Vision Model
        ↓
Process Predictions
        ↓
Generate Visualization
        ↓
Save Results
        ↓
Return Results to User
```

Each step has a specific purpose and follows a defined order.

### Agent vs. Workflow

An important distinction is the level of autonomy.

```text
WORKFLOW                         AI AGENT

Predefined steps                Receives a goal
      ↓                              ↓
Follow sequence                 Analyzes context
      ↓                              ↓
Execute actions                 Decides next action
      ↓                              ↓
Return result                   Uses tools/skills
                                     ↓
                                Evaluates result
                                     ↓
                                Continues toward goal
```

A **workflow** generally determines the path in advance.

An **agent** has greater flexibility to decide what action should happen next based on the current context.

### Relationship Between the Concepts

```text
                     AI SYSTEM
                         │
             ┌───────────┴───────────┐
             │                       │
          WORKFLOW                 AGENT
     Defines the process      Makes decisions
                                     │
                         ┌───────────┼───────────┐
                         │           │           │
                       SKILL     GUARDRAIL      MCP
                         │           │           │
                    How to do    Boundaries   Connection
                    the task      & rules      to tools
                                                 │
                                                 ▼
                                               TOOLS
```

### Key Idea

> **Agent = decides what needs to be done.**  
> **Skill = provides specialized knowledge about how to do it.**  
> **MCP = provides a standardized way to connect with tools.**  
> **Guardrail = defines the rules and boundaries.**  
> **Workflow = defines a structured sequence of steps, decisions, and actions.**

---

## 6. Memory / Context

### Definition

**Context** is the information available to an AI agent during a specific execution. It can include instructions, conversation history, data, tool results, and other information needed to understand the current task.

**Memory** is information that can persist and be retrieved across different interactions or executions.

### Original Class Definition

> Context es la información disponible para el agente durante una ejecución: instrucciones, conversación, etc. Memory es información que puede persistir y recuperarse entre interacciones o ejecuciones.

### In Simple Terms

Although **Context** and **Memory** are related, they are not exactly the same.

```text
CONTEXT
"What does the agent know right now?"

Examples:
├── Current instructions
├── Current conversation
├── User request
├── Tool results
└── Current task information
```

Memory is information that can be saved and used again later:

```text
MEMORY
"What can the agent remember later?"

Interaction #1
     ↓
Information is stored
     ↓
    Memory
     ↓
Interaction #2
     ↓
Information is retrieved
     ↓
Agent uses previous information
```

### Context Example

Imagine that a user tells a Computer Vision agent:

> "Analyze this image using SAM3 and identify the objects."

During that execution, the agent's context could contain:

```text
Current Context
│
├── User instruction
├── Uploaded image
├── SAM3 instructions
├── Available tools
├── Previous messages
└── Current model results
```

The agent uses this information to determine what it should do.

### Memory Example

Suppose the user previously told the system:

> "For my Computer Vision projects, I prefer results exported as PNG files."

If that preference is stored as memory, a future interaction could retrieve it:

```text
Previous Interaction
        ↓
"Prefer PNG output"
        ↓
      MEMORY
        ↓
Future Interaction
        ↓
Memory Retrieved
        ↓
Agent generates PNG output
```

The user does not necessarily need to repeat the preference every time.

### Context vs. Memory

| Context | Memory |
|---|---|
| Available during the current execution | Can persist between executions |
| Helps understand the current task | Helps retain useful information |
| May contain conversation and instructions | Can contain previously stored information |
| Usually focused on what is needed now | Can be retrieved later |
| Changes as the interaction progresses | May persist across interactions |

### Why Are Context and Memory Important?

They help AI agents:

- Understand the current task
- Maintain continuity during a conversation
- Use relevant instructions
- Incorporate tool results
- Remember useful information when appropriate
- Avoid requiring the same information repeatedly
- Adapt future actions based on previously stored information

### Relationship Between the Concepts

```text
                         AI AGENT
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
       CONTEXT           MEMORY           GUARDRAIL
   What it knows now   What can persist   What is allowed
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                          SKILL
                     How to perform
                        the task
                            │
                            ▼
                           MCP
                            │
                     Access to tools
                            │
                            ▼
                          TOOLS
```

A **Workflow** can organize these components into a structured sequence of operations.

### Key Idea

> **Context = information available to the agent during the current execution.**  
> **Memory = information that can persist and be retrieved across interactions or executions.**


---

## 7. Observability

### Definition

**Observability** is the ability to understand what an AI system or agent is doing during its execution.

It helps us examine:

- What decisions the agent made
- Which tools it used
- How long the process took
- What results it produced
- Whether the system behaved as expected
- Whether the final result was successful

Observability can include **logs, traces, metrics, and evaluations**.

### Original Class Definition

> La capacidad de entender qué está haciendo el sistema: qué decisiones tomó, qué herramientas utilizó, cuánto tardó y cuál fue el resultado. Incluye logs, traces, métricas y evaluaciones.

The central questions are:

> **¿Qué hizo y cómo sabemos si funcionó?**

In English:

> **What did it do, and how do we know whether it worked?**

### In Simple Terms

Observability allows developers to **see and understand what happened inside an AI system**.

Without observability, we might only see:

```text
Input
  ↓
AI Agent
  ↓
Output
```

We know what went in and what came out, but we may not understand what happened in between.

With observability:

```text
User Request
     ↓
AI Agent
     │
     ├── Decision #1
     │
     ├── Tool Call #1
     │
     ├── Tool Result
     │
     ├── Decision #2
     │
     ├── Tool Call #2
     │
     └── Final Decision
     ↓
Final Result
```

Developers can inspect the execution process instead of seeing only the final answer.

### Main Components of Observability

#### Logs

Logs record important events that occur while the system is running.

```text
10:30:01 - Agent started
10:30:02 - Image loaded
10:30:03 - SAM3 model selected
10:30:05 - Inference started
10:30:08 - Inference completed
10:30:09 - Results generated
```

#### Traces

Traces show the sequence of operations performed during an execution.

```text
User Request
     ↓
Agent
     ↓
Tool A
     ↓
Agent
     ↓
Tool B
     ↓
Final Response
```

They are especially useful for understanding complex agent workflows involving multiple tools.

#### Metrics

Metrics provide numerical information about system performance.

Examples include:

```text
Execution time:      4.2 seconds
Tool calls:          3
Successful calls:    3
Failed calls:        0
Images processed:    10
```

#### Evaluations

Evaluations help determine whether the system produced a correct or useful result.

They can answer questions such as:

- Was the result accurate?
- Did the agent complete the objective?
- Did it use the correct tool?
- Did it follow the required instructions?
- Did it respect the guardrails?

### Computer Vision Example

Imagine a Computer Vision agent receives:

> "Segment all objects in this image using SAM3."

Observability could record:

```text
Request Received
       ↓
Image Loaded
       ↓
SAM3 Selected
       ↓
Inference Started
       ↓
Segmentation Generated
       ↓
Masks Processed
       ↓
Result Evaluated
       ↓
Output Returned
```

The system might also record:

```text
Model:              SAM3
Processing time:    2.8 seconds
Objects detected:   6
Masks generated:    6
Errors:             0
Status:              Successful
```

If something goes wrong, these records help developers identify **where and why the failure occurred**.

### Why Is Observability Important?

Observability helps developers:

- Debug AI agents
- Understand agent decisions
- Track tool usage
- Detect failures
- Measure performance
- Evaluate output quality
- Identify bottlenecks
- Monitor workflows
- Improve reliability
- Verify that guardrails are working

### Relationship Between the Concepts

```text
                         AI AGENT
                            │
                    Receives Context
                            │
                            ▼
                     Makes Decisions
                            │
               ┌────────────┼────────────┐
               │            │            │
             SKILL      GUARDRAIL       MCP
               │            │            │
               │        Checks rules     │
               │                         ▼
               │                       TOOLS
               │                         │
               └────────────┬────────────┘
                            │
                            ▼
                          RESULT
                            │
                            ▼
                    OBSERVABILITY
                            │
              ┌─────────────┼─────────────┐
              │             │             │
             Logs         Traces        Metrics
                                            │
                                      Evaluations
```

### Key Idea

> **Observability = understanding what the AI system did, how it did it, how it performed, and whether it worked correctly.**

In simple terms:

> **If an agent does something, observability helps us understand what happened.**

