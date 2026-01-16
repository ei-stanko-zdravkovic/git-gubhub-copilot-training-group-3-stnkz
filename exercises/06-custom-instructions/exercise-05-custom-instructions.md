# Exercise 05 — Custom Instructions and Saved Prompts

## Learning Objectives
- Create reusable custom instructions for specific contexts (folders, file types)
- Build custom Copilot Chat agents using `.prompt.md` files
- Automate instruction generation across the codebase
- Share consistent Copilot configuration through version control

---

## Why Custom Instructions Matter

As projects grow, different parts of the codebase require different coding standards, patterns, and constraints:

- **Models** need validation rules, type hints, and no database imports
- **Tests** require fixtures, mocks, and deterministic behavior
- **Notebooks** need proper path setup and clear documentation
- **Data layers** must follow specific connection patterns

**Problem:** Manually creating `.instructions.md` files for every folder is tedious and error-prone.

**Solution:** Build a custom Copilot agent that generates context-aware instructions automatically.

---

## Task 1: Create the Custom Instructions Tool

**Goal:** Build a reusable Copilot Chat command that generates folder-specific instructions.

### Instructions

**Step 1: Create the prompts directory**

```bash
mkdir -p .github/prompts
cd .github/prompts
```

**Step 2: Create the custom instructions prompt file**

Create `.github/prompts/custom-instructions.prompt.md` with the following content:

```markdown
---
description: Generates custom GitHub Copilot instructions based on provided context (folder, files, or file types)
mode: agent
---

You are a GitHub Copilot instruction generator. Your job is to create context-aware coding instructions for specific parts of a codebase.

## Instructions

1. **Analyze the provided context** (folder, files, or file type)
   - Identify the purpose: models, tests, notebooks, data access, agents, etc.
   - Look at existing code patterns and conventions
   - Note any special requirements (e.g., no external dependencies, type hints required)

2. **Generate appropriate instructions** based on context type:

   **For Python scripts/modules:**
   - Coding standards (type hints, docstrings, error handling)
   - Import rules (what's allowed/forbidden)
   - Patterns to follow (e.g., "Models never import from data layer")
   - Testing requirements

   **For Jupyter notebooks:**
   - Setup requirements (sys.path, imports)
   - Documentation standards
   - Execution order expectations

   **For test files:**
   - Fixture patterns
   - Mocking requirements
   - Deterministic behavior (seeds, no I/O)
   - Coverage expectations

   **For data access layers:**
   - Connection patterns
   - Transaction handling
   - Error handling
   - Security considerations

3. **Output format:**
   - Write to: `.github/instructions/<context-name>.instructions.md`
   - Use kebab-case for filename (parent folder name)
   - Include front matter:
     ```yaml
     ---
     description: <Brief description of what this instruction applies to>
     applyTo: <glob pattern, e.g., "src/models/**/*.py">
     ---
     ```
   - Keep instructions concise, actionable, and specific
   - Use bullet points and examples
   - Reference existing patterns from the codebase

4. **Context handling:**
   - If context is provided (e.g., `#folder:src/models`), analyze it
   - If no context: ask user which folder/files to generate instructions for
   - Never guess or generate instructions without context

5. **Examples of good instructions:**

   **For models folder:**
   ```markdown
   ---
   description: Instructions for ML model implementations
   applyTo: src/models/**/*.py
   ---
   
   ## Model Implementation Rules
   
   - NEVER import from `src.data.database` - models are pure functions
   - All models must accept DataFrames as input
   - Include type hints: `def fit(X: pd.DataFrame, y: pd.Series) -> None`
   - Set `random_state=42` for reproducibility
   - Validate inputs: raise `ValueError` on empty DataFrame or missing columns
   - Implement save/load methods using joblib
   ```

   **For notebooks folder:**
   ```markdown
   ---
   description: Instructions for Jupyter notebooks
   applyTo: notebooks/**/*.ipynb
   ---
   
   ## Notebook Standards
   
   - First cell must add parent to sys.path:
     ```python
     import sys
     from pathlib import Path
     sys.path.insert(0, str(Path().resolve().parent))
     ```
   - Use markdown cells to explain each analysis step
   - Import from `src.*` modules (e.g., `from src.data.database import get_connection`)
   - Clear outputs before committing
   ```

## Behavior

- **Command:** `/custom-instructions`
- **With context:** `/custom-instructions #folder:src/models` → generates `models.instructions.md`
- **Without context:** Ask user: "Which folder or files should I generate instructions for?"
- **Multiple contexts:** Generate one file per context
- **Existing file:** Ask if user wants to overwrite or merge

## Remember

- Instructions are version-controlled → consistency across team
- Instructions are context-aware → different rules for different code
- Instructions are reusable → run command anytime patterns evolve
```

**Step 3: Reload VS Code**

Press `Cmd + Shift + P` (Mac) or `Ctrl + Shift + P` (Windows/Linux) and select:
```
Developer: Reload Window
```

**⚠️ Important:** This step is required! VS Code won't recognize the new agent until you reload.

**Step 4: Verify the tool is available**

1. Open Copilot Chat
2. Type `/custom-instructions` (with the slash)
3. You should see the command appear in autocomplete
4. If you don't see it, check:
   - File is saved in `.github/prompts/`
   - Front matter includes `mode: agent`
   - You reloaded the window

### Acceptance Criteria

- [ ] `.github/prompts/custom-instructions.prompt.md` exists with correct front matter
- [ ] VS Code window has been reloaded
- [ ] `/custom-instructions` command appears in Copilot Chat autocomplete
- [ ] Running `/custom-instructions` without context asks which folder to target

---

## Task 2: Generate Instructions for Notebooks

**Goal:** Use the custom tool to create context-specific instructions for the notebooks folder.

### Instructions

**Step 1: Run the custom instructions command**

Open Copilot Chat and use:
```
/custom-instructions #folder:notebooks
```

**Alternative with file context:**
```
/custom-instructions #file:exploratory_analysis.ipynb

Generate instructions for all Jupyter notebooks in the notebooks/ folder.
Include:
- Required sys.path setup in first cell
- Import patterns (src.* modules)
- Documentation standards
- Kernel selection (use "Python (airline-discount-ml)" kernel)
```

**Step 2: Review the generated file**

The agent should create: `.github/instructions/notebooks.instructions.md`

**Expected content should include:**
- Front matter with `applyTo: notebooks/**/*.ipynb` or `notebooks/**`
- sys.path setup requirement
- Import patterns from `src.*`
- Documentation standards
- Kernel selection guidance

**Step 3: Test the instructions**

1. Open `airline-discount-ml/notebooks/exploratory_analysis.ipynb`
2. Add the notebook as context to Copilot Chat: `#file:exploratory_analysis.ipynb`
3. Ask Copilot: "What setup code do I need in the first cell?"
4. Copilot should reference the instructions you just generated

**Step 4: Refine if needed**

If the generated instructions are too generic or missing patterns:
- Provide more context examples
- Specify what patterns to include
- Ask the agent to analyze existing notebooks first

### Acceptance Criteria

- [ ] `.github/instructions/notebooks.instructions.md` exists
- [ ] Front matter includes correct `applyTo` pattern
- [ ] Instructions include sys.path setup requirement
- [ ] Instructions reference correct import patterns (`src.*`)
- [ ] Copilot Chat uses these instructions when working with notebooks

---

## Task 3: Generate Instructions for Multiple Contexts

**Goal:** Automate instruction generation for all major code areas.

### Instructions

**Step 1: Generate instructions for the training scripts**

```
/custom-instructions #folder:src/training

Generate instructions for training and evaluation scripts.
Include:
- Database connection patterns (use get_connection())
- JSON parsing requirements (trips, total_spend fields)
- Model persistence patterns (save/load .pkl files)
- Metrics to report (MAE, RMSE, R²)
```

**Step 2: Generate instructions for agents**

```
/custom-instructions #folder:src/agents

Generate instructions for agent implementations.
Include:
- Can call models and database (unlike models themselves)
- Business logic layer between API and ML
- Input validation requirements
- Return type patterns
```

**Step 3: Generate instructions for the data layer**

```
/custom-instructions #folder:src/data

Generate instructions for database access code.
Include:
- Use Database class or get_connection() helper
- JSON field parsing patterns (travel_history → trips, total_spend)
- Connection lifecycle (connect, use, close)
- Error handling patterns
```

**Step 4: Verify all instructions**

Check that `.github/instructions/` now contains:
- `notebooks.instructions.md`
- `training.instructions.md`
- `agents.instructions.md`
- `data.instructions.md`
- `models.instructions.md` (from earlier exercises)
- `tests.instructions.md` (from earlier exercises)

**Step 5: Test the instructions in action**

1. Open a file in `src/data/`
2. Ask Copilot: "How should I query the database?"
3. Copilot should follow the patterns from `data.instructions.md`

### Why This Works

- **Context-aware:** Each folder gets instructions tailored to its purpose
- **Consistent:** Same patterns enforced across all similar code
- **Version-controlled:** Team shares the same Copilot behavior
- **Maintainable:** Update one instruction file → affects all relevant code
- **Scalable:** Add new folders without manual instruction writing

### Acceptance Criteria

- [ ] At least 4 new instruction files generated
- [ ] Each file has correct `applyTo` glob pattern in front matter
- [ ] Instructions are specific to the context (not generic)
- [ ] Copilot Chat respects instructions when working with those files
- [ ] All instruction files are committed to git

---

## Task 4: Create an Auto-Update Tool (Advanced)

**Goal:** Build a tool that automatically updates instructions when code patterns change.

### Instructions

**Step 1: Create an update-instructions prompt**

Create `.github/prompts/update-instructions.prompt.md`:

```markdown
---
description: Updates existing GitHub Copilot instructions based on code changes
mode: agent
---

You are an instruction updater. When code patterns evolve, you update the corresponding `.instructions.md` files.

## Task

1. **Analyze recent changes** in the provided context
2. **Compare with existing instructions** in `.github/instructions/`
3. **Identify mismatches** between actual code and documented patterns
4. **Update instructions** to reflect current best practices
5. **Preserve valid rules** that still apply

## Process

1. Read the provided context (files, diffs, or folders)
2. Find the corresponding `.instructions.md` file
3. Check if documented patterns match actual code
4. If mismatches exist:
   - Update patterns to match current code
   - Preserve other valid instructions
   - Note what changed in commit message

## Output

- Update the existing `.instructions.md` file
- Show a diff of changes
- Explain why each change was made

## Command Usage

- `/update-instructions #folder:src/models` → update models.instructions.md
- `/update-instructions #file:discount_predictor.py` → update parent folder's instructions
```

**Step 2: Reload VS Code window**

**Step 3: Test the update tool**

Suppose you changed the database connection pattern in `src/data/database.py`:

```
/update-instructions #folder:src/data

The Database class now supports context manager (with statement).
Update data.instructions.md to reflect this pattern change.
```

The agent should update the instructions to document the new pattern.

**Step 4: Create a git hook (optional)**

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Auto-update instructions on commit

changed_files=$(git diff --cached --name-only)

if echo "$changed_files" | grep -q "src/models/"; then
    echo "Models changed - consider updating models.instructions.md"
fi

if echo "$changed_files" | grep -q "src/data/"; then
    echo "Data layer changed - consider updating data.instructions.md"
fi
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

### Acceptance Criteria

- [ ] `update-instructions.prompt.md` created
- [ ] Tool can detect pattern mismatches
- [ ] Tool updates instructions while preserving valid rules
- [ ] Optional: Git hook reminds developers to update instructions

---

## Task 5: Share Instructions with the Team

**Goal:** Commit all instruction files to version control so the team uses consistent Copilot behavior.

### Instructions

**Step 1: Review all instruction files**

```bash
cd .github/instructions
ls -la
```

You should see:
- `models.instructions.md`
- `tests.instructions.md`
- `notebooks.instructions.md`
- `training.instructions.md`
- `agents.instructions.md`
- `data.instructions.md`

**Step 2: Create a README documenting the instruction system**

Create `.github/instructions/README.md`:

```markdown
# GitHub Copilot Custom Instructions

This folder contains context-specific coding instructions for GitHub Copilot.

## How It Works

- Each `.instructions.md` file applies to specific folders/files (see `applyTo` in front matter)
- Copilot automatically uses these instructions when editing matching files
- Instructions are version-controlled → consistent behavior across the team

## Instruction Files

| File | Applies To | Purpose |
|------|------------|---------|
| `models.instructions.md` | `src/models/**/*.py` | ML model implementation rules |
| `tests.instructions.md` | `tests/**/*.py` | Unit test standards |
| `notebooks.instructions.md` | `notebooks/**/*.ipynb` | Jupyter notebook conventions |
| `training.instructions.md` | `src/training/**/*.py` | Training pipeline patterns |
| `agents.instructions.md` | `src/agents/**/*.py` | Agent layer guidelines |
| `data.instructions.md` | `src/data/**/*.py` | Database access patterns |

## Generating New Instructions

Use the custom `/custom-instructions` command in Copilot Chat:

```
/custom-instructions #folder:src/new_module
```

This will generate a new `.instructions.md` file automatically.

## Updating Instructions

When code patterns change:

```
/update-instructions #folder:src/models
```

This updates the corresponding instruction file to match current code.

## For New Team Members

1. Clone the repository
2. Open in VS Code with GitHub Copilot extension
3. Instructions are automatically applied - no manual setup needed!

## Best Practices

- Keep instructions specific to their context (not generic advice)
- Use examples from actual codebase
- Update instructions when patterns evolve
- Review instruction PRs just like code PRs
```

**Step 3: Commit everything**

```bash
cd /Users/mariasukhareva/VSCode/airlst-github-copilot-training

# Add all instruction files
git add .github/instructions/
git add .github/prompts/

# Commit
git commit -m "Add custom instruction system with auto-generation tools

- Created /custom-instructions agent for generating context-aware instructions
- Created /update-instructions agent for maintaining instruction files
- Generated instructions for: models, tests, notebooks, training, agents, data
- Added README documenting the instruction system

This enables consistent Copilot behavior across the team without manual configuration."

# Push
git push origin main
```

**Step 4: Test with a teammate (or in a fresh clone)**

1. Clone the repo in a new location (or ask a teammate to)
2. Open in VS Code with Copilot
3. Edit a file in `src/models/`
4. Ask Copilot for help
5. Verify it follows the `models.instructions.md` patterns

### Acceptance Criteria

- [ ] All instruction files committed to git
- [ ] README documents the instruction system
- [ ] Custom prompt files committed
- [ ] Instructions work for any team member who clones the repo
- [ ] No manual configuration required (besides having Copilot extension)

---

## Summary: What You Built

You now have a **scalable, team-wide instruction system**:

1. **Custom agent** that generates instructions automatically
2. **Context-aware instructions** for every part of the codebase
3. **Version-controlled configuration** shared across the team
4. **Maintenance tools** to keep instructions up-to-date
5. **Zero manual setup** for new team members

### Key Benefits

- ✅ **Consistency:** Everyone's Copilot follows the same rules
- ✅ **Automation:** Generate instructions instead of writing manually
- ✅ **Maintainability:** Update patterns in one place
- ✅ **Scalability:** Add new contexts without reinventing the wheel
- ✅ **Onboarding:** New devs get correct patterns instantly

### Next Steps

- Add more custom agents (e.g., `/generate-docstrings`, `/analyze-coverage`)
- Create project-specific agents for common tasks
- Build CI checks that validate code against instructions
- Extend to other tools (linters, formatters) using the same patterns

---

## Troubleshooting

### Agent doesn't appear in autocomplete

- Check front matter includes `mode: agent`
- Verify file is in `.github/prompts/`
- Reload VS Code window (`Cmd+Shift+P` → "Reload Window")

### Instructions not being used

- Check `applyTo` glob pattern matches your files
- Verify front matter is valid YAML
- Try referencing the file explicitly: `#file:models.instructions.md`

### Generated instructions are too generic

- Provide more specific context (multiple file examples)
- Ask agent to analyze existing code first
- Explicitly list patterns you want included

### Can't push to git

- Ensure you're on the correct branch
- Pull latest changes first: `git pull --rebase origin main`
- Resolve any conflicts in instruction files
