Multi-Agent AI System

Course: SAM3 — Computer Vision with Segment Anything Model 3Topic: Agentic AI ProgrammingSection: Multi-Agent System Architecture

Overview

A Multi-Agent System consists of multiple specialized AI agents working together to accomplish a larger objective.

Instead of asking one AI agent to perform every task, responsibilities can be divided among specialized agents.

Each agent can have its own:

Role — What the agent is responsible for

Goal — What the agent must accomplish

Traits — Knowledge, expertise, and expected behavior

Constraints — Rules and limitations the agent must follow

Tools — Capabilities the agent can use

The agents can collaborate as part of a larger workflow, with the output of one agent becoming the input of another.

General Architecture

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

Agents in the System

The example introduced in class contains several specialized agents:

The Pipeline Architect (@architect)

The Simulation Engineer (@sim_engineer)

The Machine Learning Engineer (@ml_engineer)

The QA & Evaluation Specialist (@qa)

Each agent will be documented below according to the specifications presented during the class.

1. The Pipeline Architect (@architect)

Role

The Pipeline Architect is the lead systems integrator of the multi-agent team.

In the class example, this agent is described as a visionary Pipeline Architect and Lead Systems Integrator with extensive experience.

Its primary responsibility is to design and coordinate the complete technical pipeline while defining how the specialized agents interact with one another.

Goal

The Pipeline Architect orchestrates the end-to-end data flow across the Computer Vision and Synthetic Data Generation pipeline.

The class example connects technologies including:

Houdini
   ↓
Isaac Sim
   ↓
Synthetic Dataset
   ↓
YOLO Training
   ↓
Machine Learning
   ↓
Evaluation

The Architect is responsible for making sure these different stages work together as one coherent system.

Traits

The Pipeline Architect is expected to be:

Highly analytical

Structured

Focused on cross-platform data integrity

Capable of designing automation logic

Capable of defining how agents interact

Capable of defining how data is passed between agents

A major responsibility is determining exactly how the agents communicate and exchange data.

Constraint

The class specification places an important human-control requirement on the Architect.

The Architect must pause for explicit user approval before considering the pipeline finalized.

This means the architecture should not simply be generated and automatically treated as complete.

Instead:

Architect Designs Pipeline
          ↓
Present Architecture
          ↓
Wait for User Review
          ↓
     User Approval?
       /       \
     YES        NO
      │          │
      ▼          ▼
 Continue     Redesign
                 │
                 ▼
          Review Comments
                 │
                 ▼
          Update Workflow

The Architect can redesign workflows based on feedback and inline comments.

In Simple Terms

The Pipeline Architect acts like the technical leader of the AI-agent team.

It does not necessarily perform every specialized task itself.

Instead, it determines:

What needs to happen, which specialist should perform it, how information moves between specialists, and how the complete system fits together.

For example:

                    PIPELINE ARCHITECT
                           │
                   Designs the system
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Simulation        ML Engineer      QA Specialist
      Engineer
          │                │                │
          ▼                ▼                ▼
 Synthetic Data       Train Model      Evaluate Model
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                       Final System

Responsibilities

The Pipeline Architect may be responsible for:

Designing the overall pipeline

Coordinating specialized AI agents

Defining agent responsibilities

Designing automation logic

Maintaining cross-platform data integrity

Defining how data moves between stages

Defining how agents communicate

Reviewing the overall system architecture

Incorporating user feedback

Requesting explicit user approval before finalizing the pipeline

Relationship With the Other Agents

In this example, the Pipeline Architect operates above the specialized engineering agents.

                     USER
                       │
                       ▼
              PIPELINE ARCHITECT
                  @architect
                       │
           Designs & coordinates
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Simulation       Machine        QA &
    Engineer        Learning     Evaluation
 @sim_engineer      Engineer     Specialist
                  @ml_engineer      @qa

The specialized agents perform their respective technical responsibilities while following the architecture and specifications defined by the Pipeline Architect.

Human-in-the-Loop Principle

One particularly important concept demonstrated by this agent is human-in-the-loop control.

Even though the system uses autonomous AI agents, the human user retains authority over important decisions.

AI proposes
     ↓
Human reviews
     ↓
Human provides feedback
     ↓
AI modifies
     ↓
Human approves
     ↓
Pipeline continues

This combines AI automation with human supervision.

Key Idea

The Pipeline Architect is the coordinator of the multi-agent system. It designs the overall pipeline, defines how specialized agents interact and exchange data, and keeps the human user involved in approving the architecture.

2. The Simulation Engineer (@sim_engineer)

Role

The Simulation Engineer is the Synthetic Data Generation specialist in the multi-agent system.

The class describes this agent as an expert in:

Houdini

Isaac Sim

Synthetic Data Generation (SDG)

Houdini Python

USD manipulation

Simulation automation

This agent is responsible for creating and preparing the synthetic data that will later be consumed by the Machine Learning Engineer.

Goal

The Simulation Engineer works with an existing Houdini pipeline and then transfers the resulting assets into Isaac Sim.

The general process presented in class is:

Existing Houdini Script
        │
        ▼
Execute Houdini Processing
        │
        ▼
Generate / Prepare USD
        │
        ▼
Open USD in Isaac Sim
        │
        ▼
Apply Semantic Labels
        │
        ▼
Configure Simulation
        │
        ▼
Generate Randomized
Synthetic Dataset

One of the scripts referenced in the class example is:

houdini_scans_prepration.py

The generated USD asset is then opened in Isaac Sim, where semantic labels can be programmatically applied to imported primitives before producing the randomized synthetic dataset.

Traits

The Simulation Engineer is expected to have deep technical knowledge of areas such as:

Houdini Python (hython)

USD manipulation

Python-based simulation automation

Synthetic Data Generation

Semantic labeling

Dataset generation

The agent should also care strongly about:

Dataset variance

Realism

Eliminating biases

These characteristics are particularly important when generating synthetic training data for Computer Vision models.

Constraint

The Simulation Engineer must strictly follow the specifications provided by the Pipeline Architect.

Pipeline Architect
        │
        │ Specification
        ▼
Simulation Engineer
        │
        ▼
Generate Synthetic Data
        │
        ▼
Validate Output
        │
        ▼
dataset/

Before handing its work to another agent, the Simulation Engineer must ensure that the generated output data is properly prepared and saved in the designated:

dataset/

directory.

This establishes a clear handoff point between agents.

In Simple Terms

The Simulation Engineer's job is essentially:

Create high-quality synthetic training data according to the architecture defined by the Pipeline Architect.

Instead of manually collecting thousands of real-world images, simulation tools can be used to generate controlled synthetic data.

For example:

             SIMULATION ENGINEER
                     │
             Creates 3D Scene
                     │
                     ▼
               Isaac Sim
                     │
          ┌──────────┼──────────┐
          │          │          │
       Objects     Lighting   Camera
          │          │          │
          └──────────┼──────────┘
                     │
                     ▼
              Randomization
                     │
                     ▼
              Semantic Labels
                     │
                     ▼
             Synthetic Images
                     +
                  Labels
                     │
                     ▼
                 dataset/

Why Synthetic Data?

Synthetic Data Generation can be useful when real-world training data is:

Difficult to collect

Expensive to label

Limited in variety

Missing rare scenarios

Difficult to reproduce

Biased toward particular environments

Simulation allows developers to control factors such as:

Lighting
Camera position
Object position
Object orientation
Background
Environment
Scene composition

Changing these variables can create a more varied dataset.

Responsibilities

The Simulation Engineer is responsible for tasks such as:

Working with Houdini

Executing Houdini automation

Preparing USD assets

Working with Isaac Sim

Applying semantic labels

Generating synthetic datasets

Introducing dataset variance

Improving simulation realism

Reducing dataset bias

Following the Architect's specifications

Preparing data for downstream ML training

Saving generated data in the correct directory

Data Handoff

This agent demonstrates an important principle in multi-agent systems:

one agent's output becomes another agent's input.

In this example:

PIPELINE ARCHITECT
        │
        │ Specifications
        ▼
SIMULATION ENGINEER
   @sim_engineer
        │
        │ Generates
        ▼
    dataset/
        │
        │ Consumed by
        ▼
MACHINE LEARNING ENGINEER
   @ml_engineer

The Simulation Engineer does not train the final Computer Vision model.

Its primary responsibility is to create the data needed by the Machine Learning Engineer.

Relationship With the Architect

The Architect defines the overall system requirements:

Architect
   │
   ├── Defines pipeline
   ├── Defines data flow
   ├── Defines requirements
   │
   ▼
Simulation Engineer
   │
   ├── Implements simulation
   ├── Generates data
   └── Prepares dataset

The Simulation Engineer therefore operates as a specialized implementation agent inside the larger architecture.

Key Idea

The Simulation Engineer transforms the Architect's specifications into synthetic training data using technologies such as Houdini, USD, and Isaac Sim, then prepares that data for the Machine Learning Engineer.

3. The Machine Learning Engineer (@ml_engineer)

Role

The Machine Learning Engineer is the Computer Vision and model-training specialist in the multi-agent system.

The class describes this agent as a:

Senior Computer Vision Specialist specializing in object detection and YOLO architectures.

Its main responsibility is to consume the synthetic dataset generated by the Simulation Engineer and use it to configure, train, optimize, and evaluate a Computer Vision model.

Goal

The Machine Learning Engineer receives the synthetic dataset generated by:

@sim_engineer

The general workflow is:

Synthetic Dataset
       │
       ▼
Machine Learning Engineer
       │
       ├── Configure Model
       ├── Train Model
       ├── Tune Hyperparameters
       ├── Monitor Convergence
       └── Evaluate Performance
       │
       ▼
Trained Model + Logs

In the class example, this agent specializes in object detection and YOLO architectures.

Traits

The Machine Learning Engineer is expected to:

Write clean and scalable PyTorch/Python code

Understand Computer Vision and object detection

Work with YOLO architectures

Perform hyperparameter tuning

Analyze training behavior and loss

Monitor model convergence

Produce reproducible training results

The agent should approach model development systematically rather than simply starting training and accepting whatever result is produced.

Constraint

An important constraint shown in the class is:

The Machine Learning Engineer does not modify the dataset directly.

If the data is faulty or insufficient, the agent should report the problem rather than silently changing the dataset.

Simulation Engineer
       │
       │ Generates
       ▼
    dataset/
       │
       │ Consumed by
       ▼
Machine Learning Engineer
       │
       ├── Train Model
       ├── Save Logs
       └── Produce Evaluation Summary
       │
       ▼
    models/

This separation of responsibilities keeps the workflow controlled and makes it clear which agent owns each stage.

Model Output

The Machine Learning Engineer saves model artifacts and training logs in the designated:

models/

directory.

It also produces a clear evaluation summary for the QA agent.

ML Engineer
     │
     ├── Trained Model
     ├── Training Logs
     ├── Metrics
     └── Evaluation Summary
              │
              ▼
        QA Specialist
             @qa

In Simple Terms

The Machine Learning Engineer receives training data and transforms it into a trained Computer Vision model.

Receive Dataset
      ↓
Configure YOLO
      ↓
Train Model
      ↓
Tune Hyperparameters
      ↓
Analyze Convergence
      ↓
Evaluate Model
      ↓
Save Model + Logs
      ↓
Send Results to QA

Example

Suppose the Simulation Engineer generates a synthetic object-detection dataset.

The Machine Learning Engineer could then:

Load the dataset configuration.

Configure the YOLO training environment.

Select appropriate training parameters.

Train the object-detection model.

Monitor loss and convergence.

Evaluate model performance.

Save the trained model and logs.

Produce an evaluation summary.

Pass the results to the QA & Evaluation Specialist.

Separation of Responsibilities

One of the important architectural ideas demonstrated in the class is that each agent has a specific responsibility.

Simulation Engineer
       │
       │ Creates
       ▼
     DATA
       │
       ▼
Machine Learning Engineer
       │
       │ Creates
       ▼
     MODEL
       │
       ▼
QA & Evaluation Specialist
       │
       │ Evaluates
       ▼
    QUALITY

If a dataset problem is detected during training:

Dataset Problem Detected
          ↓
   ML Engineer
          ↓
Report the Problem
          ↓
Simulation / Pipeline Feedback
          ↓
Corrected Dataset
          ↓
Resume Training

This makes the multi-agent system easier to understand, control, and debug.

Relationship With the Other Agents

PIPELINE ARCHITECT
    @architect
        │
        ▼
Defines overall pipeline
        │
        ▼
SIMULATION ENGINEER
   @sim_engineer
        │
        ▼
Synthetic Dataset
        │
        ▼
MACHINE LEARNING ENGINEER
   @ml_engineer
        │
        ▼
Trained Model + Evaluation
        │
        ▼
QA & EVALUATION SPECIALIST
       @qa

The Machine Learning Engineer acts as the bridge between data generation and model quality evaluation.

Key Idea

The Machine Learning Engineer transforms the synthetic dataset into a trained Computer Vision model, manages the training process and model artifacts, and passes the results to the QA agent for independent evaluation.

4. The QA & Evaluation Specialist (@qa)

Role

The QA & Evaluation Specialist is responsible for evaluating the quality of both the trained Computer Vision model and the data used to train it.

The class describes this agent as a:

Meticulous Model Evaluator and Data Quality Assurance auditor.

This agent acts as the quality-control layer of the multi-agent system.

It does not simply check whether the pipeline executed successfully. It examines whether the resulting model actually performs well and whether the dataset is sufficiently reliable, diverse, and representative.

Goal

The QA & Evaluation Specialist reviews two major outputs from the previous agents:

Simulation Engineer
        │
        ▼
Synthetic Dataset ─────────┐
                           │
                           ▼
                    QA & Evaluation
                           ▲
                           │
Trained Model ─────────────┘
        ▲
        │
Machine Learning Engineer

Its goal is to scrutinize:

The synthetic dataset generated by the Simulation Engineer

The model trained by the Machine Learning Engineer

Model performance

Data quality

Potential biases

Failure cases

Poor inference results

The QA agent determines whether the current system is performing well enough or whether another iteration is required.

Traits

The QA & Evaluation Specialist is expected to be:

Detail-oriented

Highly data-driven

Meticulous

Critical of model performance

Focused on finding edge cases

Sensitive to dataset bias

Focused on identifying weak inference results

Capable of interpreting model evaluation metrics

The purpose of this agent is not simply to confirm that the system works.

Its job is to actively search for situations where the system does not work correctly.

Focus Areas

The class specifically emphasizes looking for poor inference results.

This means the QA agent should examine cases such as:

Model Prediction
       │
       ▼
Is the prediction reliable?
       │
   ┌───┴────┐
   │        │
  YES       NO
   │        │
   ▼        ▼
Accept    Investigate
            │
            ├── Dataset problem?
            ├── Model problem?
            ├── Bias?
            ├── Edge case?
            └── Insufficient variation?

The agent can use evaluation results to determine where the pipeline needs improvement.

Model Evaluation

The QA agent can examine relevant Computer Vision metrics and evaluation results.

Depending on the model and task, these may include:

Precision

Recall

mAP

IoU

Validation loss

Training loss

False positives

False negatives

Per-class performance

Inference quality

The exact metrics depend on the Computer Vision task being evaluated.

Dataset Quality Evaluation

The QA agent should also inspect the dataset itself.

Potential problems can include:

Dataset
   │
   ├── Insufficient variation
   ├── Class imbalance
   ├── Poor labels
   ├── Unrealistic synthetic images
   ├── Missing edge cases
   ├── Dataset bias
   └── Insufficient examples

This is important because poor model performance may originate from the training data, not necessarily from the model architecture.

Feedback Loop

One of the most important responsibilities shown in the class example is the ability to trigger a feedback loop.

If the QA agent discovers that the model performs poorly because the dataset is insufficient, it can send the problem back to the Simulation Engineer.

        SIMULATION ENGINEER
           @sim_engineer
                 │
                 │ Generate data
                 ▼
             dataset/
                 │
                 ▼
        ML ENGINEER
         @ml_engineer
                 │
                 │ Train model
                 ▼
             models/
                 │
                 ▼
          QA SPECIALIST
              @qa
                 │
            Evaluate
                 │
          Is quality good?
             /       \
           YES        NO
            │          │
            ▼          ▼
         Accept     Identify Problem
                       │
                       ▼
                 More / Better Data
                       │
                       ▼
                @sim_engineer
                       │
                       └───────────┐
                                   │
                                   ▼
                              New Dataset

This creates an iterative improvement cycle rather than a simple one-way pipeline.

Example

Suppose the Machine Learning Engineer trains a YOLO model using synthetic images generated by the Simulation Engineer.

The overall model performs well, but the QA agent discovers that detection performance becomes poor when objects are partially hidden.

The QA agent could identify this as an edge case:

QA Evaluation
      ↓
Poor detection of
partially occluded objects
      ↓
Possible dataset weakness
      ↓
Request additional synthetic data
      ↓
Simulation Engineer
      ↓
Generate scenes with
more object occlusion
      ↓
New Dataset
      ↓
ML Engineer retrains model
      ↓
QA evaluates again

This demonstrates why QA is an active component of the agentic system rather than simply the final step.

Relationship With the Other Agents

The QA Specialist closes the loop between all of the specialized agents:

                 PIPELINE ARCHITECT
                    @architect
                        │
                        ▼
               Defines Architecture
                        │
                        ▼
              SIMULATION ENGINEER
                 @sim_engineer
                        │
                 Synthetic Dataset
                        │
                        ▼
                 ML ENGINEER
                 @ml_engineer
                        │
                   Trained Model
                        │
                        ▼
                 QA SPECIALIST
                     @qa
                        │
                     Evaluate
                        │
                ┌───────┴───────┐
                │               │
              PASS             FAIL
                │               │
                ▼               ▼
          Accept Result     Feedback Loop
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
           @sim_engineer                @ml_engineer
           Improve Data                 Improve Model

Why This Agent Is Important

Without QA, an automated pipeline could successfully:

Generate Data
     ↓
Train Model
     ↓
Produce Predictions

and still produce a poor model.

QA adds another question:

Generate Data
     ↓
Train Model
     ↓
Produce Predictions
     ↓
       ?
"Are the results actually good?"
     ↓
Evaluate

This makes the pipeline focused not only on completion, but also on quality.

Key Idea

The QA & Evaluation Specialist evaluates the dataset and trained model, searches for poor inference results, biases and edge cases, and can trigger a feedback loop when additional data or model improvements are required.

---

## Author

**Peyman Miyandashti**  
SAM3 Learning Journey  
Computer Vision & Artificial Intelligence  

GitHub: [Peyman-mxli](https://github.com/Peyman-mxli)

---
