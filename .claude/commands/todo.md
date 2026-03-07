# /todo - Project TODO List

Display and manage the active project TODO list.

## Description

This command reads the root `TODO.md` file and provides a summary of active
work, with special attention to the current v2 migration items.

## Usage

```text
/todo           # Show summary and next recommended action
/todo status    # Count pending vs completed items
/todo next      # Show the next highest-priority item to work on
```

## Instructions

When the user invokes this command:

1. **Read `TODO.md`** from the project root
1. **Count items** by parsing checkbox state:
   - `- [ ]` = pending
   - `- [x]` = completed
1. **Categorize by priority**:
   - Critical (search for "Critical")
   - High Priority (search for "High")
   - Medium Priority (search for "Medium")
   - Low Priority (search for "Low")
1. **Display summary** in this format:

```text
📋 Project TODO Summary

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

1. **If `$ARGUMENTS` is "next"**: Show only the next item to work on with its full context (file path, line numbers, fix instructions)

1. **If `$ARGUMENTS` is "status"**: Show just the counts without recommendations

## Context

This TODO list is the lightweight execution tracker for the repository. It can
be used for v2 planning, migration work, documentation cleanup, or other active
project tasks.

## Related Files

- `TODO.md` - The actual todo list
- `docs/plans/2026-03-07-v2-roadmap.md` - Current lightweight v2 roadmap
- `CLAUDE.md` - Core repo guidance and links to active project docs
