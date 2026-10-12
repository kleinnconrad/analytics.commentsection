# Repository Guidelines

## Table of Contents
- [1. Directory Structure](#1-directory-structure)
- [2. Dependencies](#2-dependencies)
- [3. Environment Execution](#3-environment-execution)
- [4. Code Quality & Formatting](#4-code-quality--formatting)
- [5. Language & Documentation Style](#5-language--documentation-style)
- [6. Linting & Validation](#6-linting--validation)
- [7. Security Best Practices](#7-security-best-practices)
- [8. Error Handling](#8-error-handling)
- [9. Version Control & Committing](#9-version-control--committing)

This file contains foundational instructions and conventions for developing in this repository. AI agents and developers must strictly adhere to these rules.

## 1. Directory Structure
- The `src` directory is strictly reserved for the core logic of the application. 
- Do not place auxiliary code in `src`. Place supporting files in meaningfully named directories (e.g., `utilities`, `scripts`, `data_generation`, `sql`).
- Maintain a separate `tests` directory for unit and integration tests to keep the core codebase clean.

## 2. Dependencies
- Always maintain a `requirements.txt` file at the root of the project.
- Any time a new Python dependency is introduced, it must be explicitly added to `requirements.txt` with appropriate version pinning to ensure reproducible builds.

## 3. Environment Execution
- Execute local scripts using the Conda `myenv` environment to ensure consistency across local development setups.
- Provide clear instructions in the `README.md` if the Conda environment requires updating due to new dependencies.

## 4. Code Quality & Formatting
- Avoid hardcoding paths without environment checks; use relative paths or standard library path resolution (e.g., `pathlib`).
- Use the standard Python `logging` module instead of `print()` statements for proper log leveling, observability, and debugging.
- Implement Python type hinting (PEP 484) across all function signatures and class definitions to improve readability and catch type errors early.
- Follow PEP 8 style guidelines for Python code.

## 5. Language & Documentation Style
- **English Only:** All code, comments, documentation, and filenames in this repository must be strictly in English.
- **No Emojis:** Never use emojis anywhere in the repository, including code, documentation, and commit messages.
- **Tone:** Maintain a sober, objective, and professional tone across all documentation (including `README.md`).
- **Vocabulary:** Use only as many adjectives as strictly required. Do not use superlatives (e.g., "fastest", "best", "most") or filler words.
- **Docstrings:** All modules, classes, and public functions must include clear, structured docstrings (e.g., Google-style) detailing their purpose, arguments, and return types.
- **Table of Content:** All markdown files should have a table of content if they contain more than one paragraph.

## 6. Linting & Validation
- Code must pass standard Python linting and formatting checks (e.g., `ruff`, `flake8`, or `black`) before being finalized.
- Always verify that configuration files (e.g., `.yml` or `.json` files) remain structurally compliant and correctly reflect any changes made to the codebase (such as script renames or moving files).

## 7. Security Best Practices
- Never hardcode secrets, passwords, or API keys in the repository. Always rely on environment variables or a `.env` file (ensure `.env` is included in `.gitignore`).

## 8. Error Handling
- Use specific exception handling (`try...except` blocks) instead of catching generic `Exception`s where possible, and provide informative error messages.

## 9. Version Control & Committing
- Never run `git commit` or `git push`, and never commit or push through GitHub tools or APIs, without the user's explicit approval of that specific commit. Prepare the changes locally, show the diff and the proposed commit message, and wait for approval. Approval of one commit does not extend to later commits.
