# Topic 0 — Agentic AI Programming Methodologies

> Learning how to use AI agents as development assistants for Computer Vision and modern software engineering.

**Course:** SAM3 — Computer Vision with Segment Anything Model 3  
**Topic:** Agentic AI Programming Methodologies  
**Estimated Duration:** 1 hour

---

## Table of Contents

1. [Overview](#overview)
2. [The Agentic AI Paradigm](#1-the-agentic-ai-paradigm)
3. [Advantages in Computer Vision](#2-advantages-in-computer-vision)
4. [Prompt Engineering for Code](#3-prompt-engineering-for-code)
5. [Exercise 1 — AI-Assisted Code Generation](#4-exercise-1--ai-assisted-code-generation)
6. [Interactive Debugging and Refinement](#5-interactive-debugging-and-refinement)
7. [Exercise 2 — AI-Assisted Debugging](#6-exercise-2--ai-assisted-debugging)
8. [Recommended Agentic Development Workflow](#7-recommended-agentic-development-workflow)
9. [MCP and Roboflow](#8-mcp-and-roboflow)
10. [Key Takeaways](#9-key-takeaways)
11. [Next Steps](#10-next-steps)

---

# Overview

Welcome to **Topic 0 of my SAM3 Learning Journey**.

Modern software development is changing rapidly with the introduction of Artificial Intelligence tools capable of assisting developers throughout the development process.

In this lesson, the goal is not only to learn how to write code, but also to understand how to **orchestrate AI agents** that can help generate, debug, explain, and optimize code.

This approach is especially useful in **Computer Vision**, where applications frequently require repetitive code for loading models, processing images, manipulating tensors, calculating metrics, and visualizing results.

The developer remains responsible for understanding the problem, designing the solution, reviewing the generated code, and validating the final result.

---

# 1. The Agentic AI Paradigm

Traditionally, programming required developers to remember syntax, libraries, APIs, functions, and implementation details.

With modern AI-assisted development, this workflow is evolving.

The developer increasingly takes the role of:

- Software Architect
- System Designer
- Problem Solver
- Code Reviewer
- Technical Decision Maker

AI development assistants can help perform tasks such as:

- Generating code
- Explaining code
- Finding bugs
- Refactoring functions
- Writing documentation
- Suggesting optimizations
- Creating tests
- Explaining error messages

Examples of AI-assisted development tools include:

- GitHub Copilot
- Cursor
- Gemini
- ChatGPT
- Other AI coding assistants

The objective is not to blindly accept AI-generated code.

Instead, the developer should **guide, review, test, and validate** the work produced by the AI assistant.

---

# 2. Advantages in Computer Vision

Computer Vision applications often require significant amounts of repetitive or boilerplate code.

For example, a Computer Vision pipeline may need to:

```text
Load Image
    ↓
Preprocess Image
    ↓
Load Model
    ↓
Run Inference
    ↓
Process Predictions
    ↓
Calculate Metrics
    ↓
Visualize Results
    ↓
Export Results
```

AI assistants can help generate many of these repetitive components.

This allows the developer to spend more time focusing on:

- Application architecture
- Model behavior
- Data quality
- Experimentation
- Performance
- Business logic
- Result validation

This becomes particularly useful when working with technologies used throughout the SAM3 course, including:

```text
Python
OpenCV
NumPy
PyTorch
SAM3
Supervision
Roboflow
Hugging Face
Google Colab
```

---

# 3. Prompt Engineering for Code

The quality of AI-generated code depends heavily on the quality of the instructions provided to the AI.

This process is commonly known as **Prompt Engineering**.

A strong programming prompt should contain four important components.

---

## 3.1 Context

Tell the AI which technology, programming language, framework, or library you are using.

For example:

```text
I am using Python, OpenCV, and NumPy.
```

Other examples might include:

```text
OpenCV
Supervision
Ultralytics
PyTorch
SAM3
Roboflow
```

Providing context helps the AI select the appropriate APIs and programming techniques.

---

## 3.2 Clear Objective

Clearly explain what you want the program to accomplish.

For example:

```text
Load an image and draw a bounding box around the detected object.
```

A vague instruction such as:

```text
Make an image program.
```

provides much less useful information.

---

## 3.3 Constraints

Constraints tell the AI how the solution should or should not be implemented.

For example:

```text
Do not use a Python for loop.
Vectorize the operation using NumPy.
```

Other constraints could include:

```text
Use OpenCV only.

Do not save the image to disk.

Return a NumPy array.

The function must work in Google Colab.
```

---

## 3.4 Input and Output

Clearly describe what information enters the program and what the program should return.

For example:

```text
Input:
A URL containing an image.

Output:
A NumPy array containing the grayscale version of the image.
```

A well-structured prompt therefore follows this general pattern:

```text
Context + Objective + Constraints + Input/Output
```

---

# 4. Exercise 1 — AI-Assisted Code Generation

## Objective

Use an AI coding assistant to generate a Python function that:

1. Receives an image URL.
2. Downloads the image into memory.
3. Reads the image using OpenCV.
4. Converts the image to grayscale.
5. Returns the processed image.

---

## Example Prompt

```text
Write a function called `download_and_process` that receives a URL,
downloads the image into memory, reads it using OpenCV,
converts it to grayscale, and returns the processed image.
```

---

## Starter Code

```python
import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt


def download_and_process(url):
    # TODO: AI-generated implementation
    pass
```

Instead of manually implementing every line immediately, the objective of this exercise is to practice giving an AI coding assistant clear instructions.

---

## What Are We Practicing?

This exercise demonstrates the transition from:

```text
Developer writes everything manually
```

to:

```text
Developer defines the problem
        ↓
AI proposes implementation
        ↓
Developer reviews implementation
        ↓
Developer tests the code
        ↓
Developer improves the solution
```

The developer is still responsible for the final result.

---

# 5. Interactive Debugging and Refinement

The first code generated by an AI assistant will not always be correct.

AI-generated code may contain:

- Incorrect function names
- Invalid parameters
- Outdated APIs
- Incorrect assumptions
- Logical errors
- Compatibility problems
- Missing imports
- Incorrect data types

For example, an AI assistant could suggest an OpenCV function that does not exist or use parameters from an older version of a library.

Therefore, AI-assisted development should be an **iterative process**.

---

## Recommended Debugging Workflow

```text
1. Generate
      ↓
2. Execute
      ↓
3. Inspect the error
      ↓
4. Copy the traceback
      ↓
5. Give the error to the AI
      ↓
6. Ask for an explanation
      ↓
7. Review the proposed solution
      ↓
8. Execute again
      ↓
9. Validate the result
```

One of the most useful techniques is to provide the **actual traceback** to the AI assistant.

Instead of saying:

```text
My code doesn't work.
```

provide:

```text
Here is my Python code.

Here is the complete traceback.

Explain why the error occurs and show me how to correct it.
Do not change unrelated parts of the program.
```

This gives the AI significantly more information about the problem.

---

# 6. Exercise 2 — AI-Assisted Debugging

The following example demonstrates how AI can assist with debugging.

```python
image = np.zeros((100, 100, 3), dtype=np.uint8)

cv2.rectangle(
    image,
    (10, 10),
    (50, 50),
    (255, 0, 0),
    2
)
```

When experimenting with different argument types or values, OpenCV may generate errors if the provided parameters do not match what the function expects.

---

## Example Debugging Prompt

```text
The following OpenCV code produces an error.

Explain what the error means, identify which argument is incorrect,
and show me the Pythonic way to fix the problem.

Do not simply provide the corrected code.
Explain why the correction works.
```

This last instruction is particularly important.

The goal is not simply to obtain working code.

The goal is to **understand why it works**.

---

# 7. Recommended Agentic Development Workflow

A practical workflow for AI-assisted programming is:

```text
DEFINE
   ↓
PROMPT
   ↓
GENERATE
   ↓
EXECUTE
   ↓
INSPECT
   ↓
DEBUG
   ↓
REFINE
   ↓
VALIDATE
```

### Define

Understand the problem before asking the AI to solve it.

### Prompt

Provide clear context, objectives, constraints, and expected input/output.

### Generate

Allow the AI assistant to propose an implementation.

### Execute

Run the generated code in the actual development environment.

### Inspect

Read the output, warnings, and error messages.

### Debug

Provide errors and tracebacks back to the AI when necessary.

### Refine

Improve the implementation through additional prompts and manual review.

### Validate

Confirm that the final code actually solves the original problem.

---

# 8. MCP and Roboflow

The course also introduces **MCP** in combination with tools such as **Roboflow**.

MCP-based integrations can allow AI systems to interact with external tools and development environments through structured interfaces.

This can expand an AI assistant beyond simple text generation and allow it to participate in more complex development workflows.

This section will be expanded as MCP and Roboflow are explored further during the SAM3 course.

---

# 9. Key Takeaways

After completing this topic, the most important concepts are:

- AI can assist throughout the software development process.
- Developers should remain responsible for architecture and validation.
- Good prompts lead to better generated code.
- Context is essential when requesting programming assistance.
- Constraints help control how AI generates solutions.
- Input and output requirements should be clearly defined.
- Computer Vision contains many repetitive tasks that AI can help automate.
- Tracebacks are valuable information for AI-assisted debugging.
- Generated code should always be executed and tested.
- AI-generated explanations should also be verified.
- AI should augment technical understanding rather than replace it.

---

# 10. Next Steps

The techniques introduced in this topic will be used throughout the SAM3 Learning Journey.

They will become especially useful while working with:

- Python
- Google Colab
- OpenCV
- NumPy
- Hugging Face
- Roboflow
- Supervision
- Computer Vision pipelines
- SAM3

As the course progresses, these tools can be combined to build increasingly sophisticated Computer Vision applications.

---

## Learning Philosophy

> **Use AI to accelerate development, but always understand, review, test, and validate the code you build.**

---

## Author

**Peyman Miyandashti**

SAM3 Learning Journey  
Computer Vision | Artificial Intelligence | Software Development
