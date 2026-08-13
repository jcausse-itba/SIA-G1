# AGENTS.md

Standardized instructions and operational guidelines for AI Coding Agents (Antigravity, Claude Code, OpenAI Codex, Cursor, Windsurf, Aider, etc.) collaborating on this project.

---

## 1. Project Context & Environment

* **Course**: Sistemas de Inteligencia Artificial (SIA) — Assignment / TP-1
* **Repository Structure**: This folder (`tp-1`) is a project module inside a monorepo.
  * **Git Configuration**: `.gitignore` and `.gitattributes` are managed at the monorepo root. **Do NOT** create local `.gitignore` or `.gitattributes` files inside this subfolder unless explicitly instructed by a human maintainer.
* **Primary Language**: Python 3.14+ (`requires-python >= 3.14`)

---

## 2. Package & Environment Management

* **Package Manager**: **`uv`** is the mandatory environment and dependency tool.
  * Always execute scripts and tools using `uv run`:
    ```bash
    uv run python main.py
    ```
  * Add dependencies using `uv add <package>`:
    ```bash
    uv add pandas plotly
    ```
  * Do **NOT** use `pip`, `poetry`, `conda`, or `pipenv` directly.

---

## 3. Allowed & Prohibited Libraries

### Allowed & Preferred Stack
* **Data Processing & Analysis**: `pandas`, `numpy`
* **Visualization**: `plotly` (interactive HTML/notebook charts and export utilities)
* **Environment/CLI**: `uv`

### Strictly Prohibited Frameworks 🚫
The following libraries are **strictly forbidden** by course rules. No AI agent may import, install, or reference them in code or configuration:
* ❌ **PyTorch** (`torch`, `torchvision`, `torchaudio`, etc.)
* ❌ **TensorFlow** (`tensorflow`, `tf`, etc.)
* ❌ **Keras** (`keras`, `tensorflow.keras`)

> **Note for AI Agents**: Algorithms (search strategies, neural networks, genetic algorithms, decision trees, etc.) must be implemented from scratch using pure Python, standard library modules, NumPy, or Pandas as required by the assignment guidelines.

---

## 4. Multi-Agent AI Guidelines & Code Standards

All AI assistants (regardless of vendor or IDE platform) must adhere to these directives:

### A. Pre-Task Protocol
1. Read this `AGENTS.md` file before generating or modifying code.
2. Inspect existing project structure and imports before introducing new files or modules.
3. Validate dependencies in `pyproject.toml` managed via `uv`.

### B. Coding Standards
* **Python Style**: Adhere to PEP 8 principles.
* **Type Hints**: Use explicit type annotations for function signatures and public APIs.
* **Documentation**: Include concise docstrings (Google or NumPy style) for functions, classes, and modules.
* **Modularity**: Separate core algorithms (domain logic) from CLI scripts, data loading, and visualization code.

### C. Execution & Verification Rules
* **No Unverified Declarations**: Never report a task as complete without running execution/verification commands (e.g., `uv run python ...` or `uv run pytest`).
* **Error Log Inspection**: If an execution fails, inspect un-truncated logs completely before attempting fixes.
* **No Fallback Masking**: Do not swallow exceptions with bare `except:` clauses or mock data to bypass test failures. Fix the root algorithmic cause.

### D. File & Workspace Hygiene
* Keep generated outputs, scratch scripts, or report artifacts isolated from core source directories.
* Maintain clean git status; do not generate unintended config files.

---

## 5. Recommended Project Layout

```text
tp-1/
├── AGENTS.md             # Multi-agent AI rules and workspace guide (this file)
├── README.md             # Human-readable project overview & usage documentation
├── pyproject.toml        # uv package configuration and dependency definitions
├── uv.lock               # Deterministic dependency lockfile
├── src/                  # Core algorithms and business logic
│   └── __init__.py
├── scripts/              # Execution scripts / CLI entry points
├── doc/                  # Reports, diagrams, and assignment specs
└── tests/                # Unit and integration tests
```
