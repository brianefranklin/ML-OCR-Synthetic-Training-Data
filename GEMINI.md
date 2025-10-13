# Project: Synthetic Training Data for OCR

## Core Principles

This project adheres to the following principles:

1.  **Universality First**: The primary goal is to generate data for any writing system. All code must be language-agnostic and make no assumptions about character sets, directionality (e.g., left-to-right, right-to-left, top-to-bottom, bottom-to-top), or glyphs.
2.  **Documentation-Driven**: We write documentation as part of the development process, not as an afterthought. Our docs, hosted on Read the Docs, are as important as our code.
3.  **Test for Reliability**: We prefer Test-Driven Development (TDD) to ensure our code is reliable, robust, and correct.
4.  **Dependency Minimalism**: We avoid adding external dependencies unless they provide a significant, clear benefit that outweighs the cost of a new dependency.

---

## Development Workflow

1.  **TDD is the Standard**: When implementing a new feature, the preferred workflow is to write failing tests first, then write the code to make them pass, and finally refactor. If TDD is not practical, you must still write or update tests *before* the feature is considered complete.
2.  **Update Documentation**: As you develop, update or create the relevant Markdown files in the `./docs` directory. Every new function or feature should have corresponding documentation.

---

## Coding Standards

-   **Docstrings & Type Hints**: All new and refactored Python functions, classes, and modules must include a **Google-style docstring** and **PEP 484 type hints**.

---

## Documentation for Read the Docs

-   **Format**: All documentation is written in **Markdown** and located in the `./docs` directory.
-   **Structure**: The documentation should be organized to help users at all levels. When adding documentation, try to fit it into one of these categories:
    -   **Tutorials**: Step-by-step guides for common use cases.
    -   **How-To Guides**: Detailed recipes for accomplishing specific tasks.
    -   **API Reference**: (Often auto-generated from docstrings) Detailed explanation of classes and functions.
    -   **Conceptual Explanations**: High-level discussions of the project's architecture and design.

---

## Dependency Management

-   **Approval**: Confirm with the user before adding any new external dependencies.
-   **Python**: If a new Python dependency is approved, add it to `./requirements.txt`.
-   **System**: If a new system-level dependency is approved, add it to the project's `./Dockerfile`.

---

## Git & Version Control

-   **Git Integration**: Do not directly commit to git. Instruct the user to make a git commit. Provide git message and description. 
-   **Commit Messages**: Please use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/). For example:
    -   `feat: add support for custom font loading`
    -   `fix: resolve bug in image rotation logic`
    -   `docs: update tutorial for background generation`
-   **Commit Message Formatting**:
    When you provide a commit message, you must format it as plain text only. Do not wrap it in Markdown code blocks or any other formatting. The output should be clean and ready to be copied directly into a terminal.