# GitHub Copilot Custom Instructions

This folder contains context-specific coding instructions for GitHub Copilot.

## How It Works

- Each `.instructions.md` file applies to specific folders/files (see `applyTo` in front matter)
- Copilot automatically uses these instructions when editing matching files
- Instructions are version-controlled → consistent behavior across the team

## Instruction Files

| File | Applies To | Purpose |
|------|------------|---------|
| `models.instructions.md` | `airline-discount-ml/src/models/**/*.py` | ML model implementation rules |
| `tests.instructions.md` | `airline-discount-ml/tests/**/*.py` | Unit test standards |
| `notebooks.instructions.md` | `airline-discount-ml/notebooks/**` | Jupyter notebook conventions |
| `training.instructions.md` | `airline-discount-ml/src/training/**/*.py` | Training pipeline patterns |
| `agents.instructions.md` | `airline-discount-ml/src/agents/**/*.py` | Agent layer guidelines |
| `data.instructions.md` | `airline-discount-ml/src/data/**/*.py` | Database access patterns |

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

## Custom Agents

This repository includes custom Copilot agents (in `.github/prompts/`):

- `/custom-instructions` - Generate context-aware instruction files
- `/update-instructions` - Update instructions when code patterns change

To use these agents:
1. Make sure the files exist in `.github/prompts/`
2. Reload VS Code window (`Cmd+Shift+P` → "Developer: Reload Window")
3. Type `/custom-instructions` or `/update-instructions` in Copilot Chat

## Troubleshooting

**Agent doesn't appear in autocomplete:**
- Check front matter includes `mode: agent`
- Verify file is in `.github/prompts/`
- Reload VS Code window

**Instructions not being used:**
- Check `applyTo` glob pattern matches your files
- Verify front matter is valid YAML
- Try referencing the file explicitly with context

**Generated instructions are too generic:**
- Provide more specific context (multiple file examples)
- Ask agent to analyze existing code first
- Explicitly list patterns you want included
