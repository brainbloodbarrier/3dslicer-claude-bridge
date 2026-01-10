# /todo - PR Review TODO List

Display and manage the PR #22 review items.

## Description

This command reads the TODO.md file and provides a summary of pending items from the PR review analysis (silent-failure-hunter and pr-test-analyzer agents).

## Usage

```
/todo           # Show summary and next recommended action
/todo status    # Count pending vs completed items
/todo next      # Show the next highest-priority item to work on
```

## Instructions

When the user invokes this command:

1. **Read TODO.md** from the project root
2. **Count items** by parsing checkbox state:
   - `- [ ]` = pending
   - `- [x]` = completed
3. **Categorize by priority**:
   - Critical (search for "Critical")
   - High Priority (search for "High")
   - Medium Priority (search for "Medium")
   - Low Priority (search for "Low")
4. **Display summary** in this format:

```
📋 PR #22 Review Items

Status: X/Y completed

🔴 Critical: N pending
🟠 High: N pending
🟡 Medium: N pending
🟢 Low: N pending

Next recommended action:
→ [First unchecked Critical item, or first High if no Critical]

Quick commands:
- Edit TODO.md to mark items complete
- Run: uv run pytest -v to verify fixes
```

5. **If `$ARGUMENTS` is "next"**: Show only the next item to work on with its full context (file path, line numbers, fix instructions)

6. **If `$ARGUMENTS` is "status"**: Show just the counts without recommendations

## Context

This TODO list was generated from PR review agents analyzing the `refactor/code-quality-improvements` branch. Items include:
- Silent failure patterns that mask errors
- Missing test coverage for new code
- Error handling improvements

## Related Files

- `TODO.md` - The actual todo list
- `.claude/plans/deep-plotting-gem.md` - Full implementation plan
- `ref/resilience-patterns.md` - Documentation on error handling patterns
