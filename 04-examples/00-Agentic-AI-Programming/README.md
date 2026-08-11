# Agentic AI Programming — Examples

This folder contains practical examples from the **Agentic AI Programming Methodologies** session of the SAM3 course.

The purpose of these examples is to demonstrate how Artificial Intelligence can assist developers in generating, reviewing, debugging, and improving computer vision code.

---

## Examples

### 01 — AI-Generated Image Processing

**File:** `01-ai-generated-function.py`

This example demonstrates how an AI assistant can help generate a Python function that:

- Receives an image URL
- Downloads the image
- Loads it into memory
- Processes it using OpenCV
- Converts it to grayscale
- Returns the processed image

---

### 02 — OpenCV Debugging

**File:** `02-opencv-debugging.py`

This example demonstrates the AI-assisted debugging workflow introduced during the class.

```text
Write / Generate Code
        ↓
Run the Code
        ↓
Observe the Error
        ↓
Analyze the Error
        ↓
Ask the AI for Assistance
        ↓
Review the Proposed Solution
        ↓
Fix the Code
        ↓
Run Again
```

---

## Agentic Development Workflow

The main idea behind this session is that the developer does not simply ask AI to write code and accept the result.

Instead, the developer acts as the **software architect and code reviewer**.

```text
Developer
    ↓
Define Context
    ↓
Define Objective
    ↓
Define Restrictions
    ↓
Describe Input / Output
    ↓
AI Assistant
    ↓
Generate Code
    ↓
Developer Review
    ↓
Execute
    ↓
Debug
    ↓
Refine
```

---

## Prompt Engineering for Code

A good coding prompt should clearly define:

1. **Context** — Which library or technology is being used?
2. **Objective** — What should the code accomplish?
3. **Restrictions** — Are there specific implementation requirements?
4. **Input / Output** — What data enters the program and what result should be returned?

Example:

```text
Using OpenCV, create a Python function called descargar_y_procesar
that receives an image URL, downloads the image into memory,
converts it to grayscale, and returns the processed image.
```

---

## Technologies

The examples in this folder use:

- Python
- OpenCV
- NumPy
- Matplotlib
- AI-assisted programming

---

## Learning Objective

The goal of these examples is to practice a development workflow where AI assists with coding while the developer remains responsible for:

- Understanding the generated code
- Reviewing its correctness
- Executing the program
- Identifying errors
- Debugging problems
- Improving the solution

This workflow will be used throughout the SAM3 learning journey.
