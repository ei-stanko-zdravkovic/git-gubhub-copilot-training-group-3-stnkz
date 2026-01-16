# Agent and Skill Templates

This folder contains templates and examples for creating Copilot Chat agents and skills.

## Files in This Exercise

| File | Description |
|------|-------------|
| `exercise-06-agents-and-skills.md` | Main exercise instructions |
| `agent-template.md` | Template for creating new agents |
| `skill-template.md` | Template for creating new skills |

## Quick Reference

### Agent File Location
```
.github/agents/<agent-name>.agent.md
```

### Skill File Location
```
.github/skills/<skill-name>/SKILL.md
```

### Invoking an Agent
```
@agent-name <your request>
```

## Existing Agents in This Repo

| Agent | Purpose | Skill |
|-------|---------|-------|
| `@install_synth` | Install Synth CLI | `install-synth/SKILL.md` |
| `@setup` | Python environment setup | `setup-environment/SKILL.md` |
| `@setup_env` | Python environment setup | `setup-environment/SKILL.md` |
| `@generate_dataset_synth` | Generate synthetic data | `generate-dataset-synth/SKILL.md` |
