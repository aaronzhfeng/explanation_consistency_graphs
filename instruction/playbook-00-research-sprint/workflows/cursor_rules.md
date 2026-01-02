# Cursor IDE Configuration for Speedrun

> Set up Cursor for effective AI-assisted research

---

## Cursor Rules File

Create `.cursorrules` or `cursorrules.txt` in your project root:

```
# PROJECT_NAME Rules

## Context
- This is a research project targeting [VENUE]
- Deadline: [DATE]
- Core contribution: [ONE SENTENCE]

## Training Commands
- DO NOT automatically run training commands
- Only provide commands for the user to run manually
- User will execute training when ready

## Documentation
- Keep docs/ files numerically ordered (00_, 01_, 02_, etc.)
- Update 00_index.md when adding new docs
- Log all sessions in session_log.md

## Hardware
- Available GPUs: [LIST]
- Use CUDA_VISIBLE_DEVICES to select GPUs

## Key Paths
- Model outputs: outputs/
- Training data: data/processed/
- Test data: data/processed/

## Code Style
- Use type hints
- Include docstrings
- Follow existing patterns in codebase

## Paper
- LaTeX source in paper/
- Use existing style file
- Keep bibliography updated
```

---

## Recommended Settings

### Editor Settings
```json
{
  "editor.formatOnSave": true,
  "python.linting.enabled": true,
  "python.formatting.provider": "black"
}
```

### Keyboard Shortcuts
| Action | Shortcut |
|--------|----------|
| Open AI chat | Cmd+L |
| Inline edit | Cmd+K |
| Accept suggestion | Tab |
| Reject suggestion | Esc |

---

## Effective Cursor Workflows

### 1. Context Window Management

Always include relevant files when asking questions:

```
@file1.py @file2.py 
What's wrong with this code?
```

### 2. Use @ References

| Reference | Purpose |
|-----------|---------|
| `@filename.py` | Include specific file |
| `@folder/` | Include folder contents |
| `@codebase` | Search entire codebase |
| `@docs` | Include documentation |

### 3. Terminal Integration

Run commands and share output:
```
@terminal (lines 45-60)
What does this error mean?
```

### 4. Image Understanding

For debugging visualizations:
```
@screenshot.png
The plot looks wrong. What should I change?
```

---

## Project Setup Checklist

- [ ] Create `.cursorrules` file
- [ ] Set up Python environment
- [ ] Configure linting
- [ ] Add project-specific rules
- [ ] Create initial documentation structure

