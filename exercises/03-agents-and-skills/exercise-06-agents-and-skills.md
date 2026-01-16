# Exercise 06 — Custom Agents and Skills

## Learning Objectives
- Understand the difference between Copilot Chat **agents** and **skills**
- Create reusable agents that delegate to skill files for specialized tasks
- Build a Synth CLI installer agent using the skills pattern
- Test and invoke custom agents in VS Code

---

## What Are Agents and Skills?

**Copilot Chat Agents** are specialized AI assistants with a focused purpose. They:
- Live in `.github/agents/` with the `.agent.md` extension
- Have a specific persona, constraints, and capabilities
- Can invoke agents using `@agent-name` in Copilot Chat

**Skills** are reusable knowledge modules stored in directories with a `SKILL.md` file. VS Code supports:

**Project skills** (stored in your repository):
- `.github/skills/` (recommended)

**Personal skills** (stored in your user profile):
- `~/.copilot/skills/` (recommended) - we are not doing these ones and it is a new feature so use with caution. 

Each skill:
- Has its own subdirectory (e.g., `.github/skills/install-synth/`)
- Contains a `SKILL.md` file with YAML frontmatter defining `name` (required) and `description` (required)
- Can include additional files like scripts, templates, and examples
- Provides step-by-step procedures that prevent AI from "hallucinating" commands

### Why This Pattern?

| Problem | Solution |
|---------|----------|
| Agents might invent wrong commands | Skills provide verified, tested commands |
| Same procedure repeated across prompts | Single source of truth in skill files |
| Hard to update install instructions | Update one skill file, all agents benefit |
| No guardrails on agent behavior | Constraints in agent file prevent drift |

---

## Task 1: Understand the Agent + Skill Structure

**Goal:** Examine the existing `install_synth` agent and its associated skill to understand the pattern.

### Instructions

**Step 1:** Open and examine the agent file:

```bash
cat .github/agents/install_synth.agent.md
```

Notice:
- The `---` YAML frontmatter with required and optional fields:
  - `description` - Brief description shown in chat
  - `name` - Agent identifier (defaults to filename if omitted)
  - `argument-hint` - Optional hint for users
  - `tools` - Array of available tools (`run_in_terminal`, `read_file`)
  - `model` - Optional AI model specification (e.g., `Claude Sonnet 4`)
  - `handoffs` - Optional buttons to transition to other agents
- The agent's persona and constraints in the body
- The reference to `../skills/install-synth/SKILL.md`

**Step 1b:** Compare with the template:

Open `exercises/06-agents-and-skills/agent-template.md` and compare:
- Front matter structure
- How `handoffs` enable agent chaining
- The balance between agent (persona/constraints) and skill (procedures)

**Step 1c:** Try modifying and calling the agent:

1. Make a copy of the install_synth agent:
   ```bash
   cp .github/agents/install_synth.agent.md .github/agents/install_synth_backup.agent.md
   ```

2. Edit `.github/agents/install_synth.agent.md` and change the `model` field:
   ```yaml
   model: GPT-5.2
   ```

3. Reload VS Code (`Cmd+Shift+P` → "Developer: Reload Window")

4. Test the agent in Copilot Chat:
   ```
   @install_synth Check if Synth is installed
   ```

5. Observe how it follows the skill file procedures

**Step 2:** Open and examine the skill file:

```bash
cat .github/skills/install-synth/SKILL.md
```

Notice:
- YAML frontmatter with `name` (required) and `description` (required)
- Step-by-step commands (no room for interpretation)
- Platform-specific notes (macOS vs Windows)
- Troubleshooting section
- Additional files in the skill directory (like `install_synth.py`)

**Step 3:** List all existing agents and skills:

```bash
ls -la .github/agents/
ls -la .github/skills/
```

### Acceptance Criteria

- [ ] You understand that agents are "who" (persona) and skills are "how" (procedure)
- [ ] You can identify the link between `install_synth.agent.md` and `install-synth/SKILL.md`
- [ ] You recognize the constraints preventing the agent from hallucinating commands

---

## Task 2: Create Your Own Agent

**Goal:** Create a new agent that helps developers generate synthetic data using Synth CLI.

### Instructions

**Step 1:** Create a new agent file

Create `.github/agents/generate_data.agent.md`:

```bash
mkdir -p .github/agents
touch .github/agents/generate_data.agent.md
```

**Step 2:** Define the agent's persona

Add the following content to `generate_data.agent.md`:

```markdown
---
description: 'Generates synthetic datasets for the airline-discount-ml project using Synth CLI.'
tools: ['run_in_terminal', 'read_file']
---
# Synthetic Data Generator Agent

You are the **Data Generator** for the airline-discount-ml project. Your goal is to help developers create realistic synthetic datasets for testing and development.

## Your Capabilities (Skills)
You have access to specialized procedures for data generation.
**When the user asks to generate data, you MUST read and follow:**
`../skills/generate-data/SKILL.md`

## Your Constraints
1. **No Hallucinations**: Never invent synth commands. Always use the skill file.
2. **Project Scope**: Only generate data into `airline-discount-ml/data/synthetic_output/`.
3. **Schema Awareness**: Always use schemas from `airline-discount-ml/synth_models/airline_data/`.
4. **Verify First**: Check if Synth is installed before attempting generation.

## Your Personality
- **Helpful**: Explain what data will be generated before running commands.
- **Safe**: Never overwrite existing data without asking.
- **Educational**: Explain the schema structure when asked.
```

### Acceptance Criteria

- [ ] Agent file exists at `.github/agents/generate_data.agent.md`
- [ ] Front matter includes `description` and `tools` array
- [ ] Agent references a skill file path
- [ ] Constraints section prevents hallucination

---

## Task 3: Create the Associated Skill

**Goal:** Create a skill file with step-by-step data generation procedures.

### Instructions

**Step 1:** Create the skill directory:

```bash
mkdir -p .github/skills/generate-data
```

**Step 2:** Create the skill file

Create `.github/skills/generate-data/SKILL.md` with:

```markdown
---
name: generate-data
description: Generate synthetic airline data using Synth CLI. Use when developers need test data for passengers, routes, or discounts tables.
---

# Synthetic Data Generation

When the user wants to generate synthetic data, follow these verified steps.

## Prerequisites Check

Before generating data, verify Synth is installed:

```bash
synth version
```

Expected output: `synth 0.6.9`

If not installed, tell the user to run: `@install_synth`

## Standard Generation Procedure

1. **Navigate to Project Root**:
   ```bash
   cd airline-discount-ml
   ```

2. **Check Available Schemas**:
   ```bash
   ls synth_models/airline_data/
   ```
   
   You should see:
   - `passengers.json` - Passenger profile schema
   - `routes.json` - Flight route schema
   - `discounts.json` - Discount rules schema

3. **Generate Data** (default: 1000 records):
   ```bash
   synth generate synth_models/airline_data/ --size 1000 > data/synthetic_output/generated_data.json
   ```

4. **Generate Specific Collection** (e.g., passengers only):
   ```bash
   synth generate synth_models/airline_data/ --collection passengers --size 500
   ```

5. **Verify Output**:
   ```bash
   head -20 data/synthetic_output/generated_data.json
   ```

## Common Customizations

### Different Output Sizes
```bash
# Small dataset for quick tests
synth generate synth_models/airline_data/ --size 100

# Large dataset for load testing  
synth generate synth_models/airline_data/ --size 10000
```

### Specific Seed for Reproducibility
```bash
synth generate synth_models/airline_data/ --size 1000 --seed 42
```

## Troubleshooting

### synth: command not found
Synth is not installed. Tell the user to invoke `@install_synth`.

### No such file or directory: synth_models/
Ensure you are in the `airline-discount-ml` directory.

### JSON parse error in output
Check that the schema files in `synth_models/airline_data/` are valid JSON.
```

### Acceptance Criteria

- [ ] Skill directory exists at `.github/skills/generate-data/`
- [ ] `SKILL.md` includes prerequisite checks
- [ ] Step-by-step commands with expected outputs
- [ ] Troubleshooting section covers common errors

---

## Task 4: Test Your Agent

**Goal:** Invoke your new agent in Copilot Chat and verify it follows the skill.

### Instructions

**Step 1:** Reload VS Code to pick up new agent files:

- Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows)
- Type "Developer: Reload Window" and press Enter

**Step 2:** Invoke the agent in Copilot Chat:

Open Copilot Chat and type:

```
@generate_data Generate 500 passenger records for testing
```

**Step 3:** Verify the agent:

- Did it reference the skill file?
- Did it check if Synth is installed first?
- Did it use the exact commands from the skill (not invented ones)?

**Step 4:** Test the install agent too:

```
@install_synth Install Synth on my machine
```

Verify it uses the Python install script, not raw curl commands.

### Acceptance Criteria

- [ ] Agent appears in Copilot Chat suggestions when typing `@gen`
- [ ] Agent follows the skill file procedures
- [ ] Agent respects its constraints (doesn't hallucinate commands)
- [ ] Generated data appears in `data/synthetic_output/`

---

## Task 5: Explore Existing Agents

**Goal:** Understand how multiple agents work together in a project.

### Instructions

**Step 1:** List all available agents:

In Copilot Chat, type `@` to see available agents.

You should see:
- `@install_synth` - Installs Synth CLI
- `@setup` or `@setup_env` - Sets up Python environment
- `@generate_data` (if you created it)

**Step 2:** Try chaining agents:

```
First run setup to create my environment, then install_synth to install Synth, finally generate_data to create test data
```

**Step 3:** Examine how agents delegate to skills:

Each agent is specialized and delegates complex procedures to skill files. This keeps agents small and maintainable while skills contain the detailed "how-to".

### Acceptance Criteria

- [ ] You can invoke multiple agents in sequence
- [ ] You understand the agent → skill delegation pattern
- [ ] You can identify which skill file each agent uses

---

## Summary

| Concept | Location | Purpose |
|---------|----------|---------|
| **Agent** | `.github/agents/*.agent.md` | Persona, constraints, references skill |
| **Skill** | `.github/skills/<name>/SKILL.md` | Step-by-step verified procedures |
| **Tools** | Front matter `tools: [...]` | What actions the agent can perform |
| **Constraints** | Agent body | Guardrails preventing hallucination |

## Best Practices

1. **Keep agents focused** - One agent, one job
2. **Skills are source of truth** - Never put commands directly in agents
3. **Include troubleshooting** - Skills should handle common failures
4. **Test with real scenarios** - Invoke agents and verify behavior
5. **Version control everything** - Agents and skills are code, commit them

---

## Bonus Challenge

Create an agent for **evaluating trained models** that:
- Lives at `.github/agents/evaluate_model.agent.md`
- References `.github/skills/evaluate-model/SKILL.md`
- Uses commands from `src/training/evaluate.py`
- Has constraints about what metrics to report
- Includes troubleshooting for common evaluation errors
