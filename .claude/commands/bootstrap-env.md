# /bootstrap-env - Setup Claude Code Environment

Bootstrap your Claude Code environment with required plugins and configuration.

## Description

This command helps you replicate a fully-configured Claude Code environment on any device or remote environment (like "Vibing"). It lists all the plugins that need to be installed and provides step-by-step guidance.

## Usage

```
/bootstrap-env          # Full setup guide
/bootstrap-env plugins  # Just list plugins to install
/bootstrap-env verify   # Check what's missing
```

## Instructions

When the user invokes this command:

### 1. Show Environment Status

First, display what's currently available:
```
📋 Claude Code Environment Bootstrap

Current environment:
- Platform: [detect from system]
- Project: NC_Slicer-Claude-Bridge
```

### 2. List Required Plugins (23 total)

Display the plugins grouped by category:

**Core Development:**
```bash
/plugin install context7
/plugin install feature-dev
/plugin install code-review
/plugin install commit-commands
/plugin install pr-review-toolkit
/plugin install plugin-dev
```

**Output Styles:**
```bash
/plugin install explanatory-output-style
/plugin install learning-output-style
```

**Integrations:**
```bash
/plugin install greptile
/plugin install supabase
/plugin install figma
/plugin install slack
```

**Language Support:**
```bash
/plugin install typescript-lsp
/plugin install pyright-lsp
/plugin install rust-analyzer-lsp
/plugin install jdtls-lsp
```

**Testing & Quality:**
```bash
/plugin install playwright
/plugin install security-guidance
/plugin install agent-sdk-dev
/plugin install hookify
```

**Specialized:**
```bash
/plugin install frontend-design
/plugin install serena
/plugin install ralph-wiggum
/plugin install ralph-loop
```

### 3. If `$ARGUMENTS` is "plugins"

Just output the plugin install commands as a copyable block:
```bash
# Core
/plugin install context7 feature-dev code-review commit-commands pr-review-toolkit plugin-dev

# Styles
/plugin install explanatory-output-style learning-output-style

# Integrations
/plugin install greptile supabase figma slack

# Languages
/plugin install typescript-lsp pyright-lsp rust-analyzer-lsp jdtls-lsp

# Testing
/plugin install playwright security-guidance agent-sdk-dev hookify

# Specialized
/plugin install frontend-design serena ralph-wiggum ralph-loop
```

### 4. If `$ARGUMENTS` is "verify"

Check which plugins are currently installed and report:
```
✅ Installed: context7, feature-dev, ...
❌ Missing: greptile, supabase, ...

Run /bootstrap-env plugins to see install commands.
```

### 5. Post-Setup Verification

After plugin installation, guide user to verify:
```bash
/help                    # Should show all commands
/todo                    # Project-specific todo list
uv run pytest -v         # Verify project works
```

## Notes

- Plugins must be installed individually on each device
- Remote environments (Vibing) don't inherit local plugins
- Project-scoped commands in `.claude/commands/` are automatically available
- Run this command whenever setting up a new environment
