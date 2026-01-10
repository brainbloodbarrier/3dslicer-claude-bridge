#!/bin/bash
# Claude Code Environment Bootstrap Script
#
# This script helps replicate your Claude Code dev environment across devices.
# Run it to see which plugins to install on a new machine or remote environment.
#
# Usage:
#   ./scripts/claude-setup.sh          # Show all plugins
#   ./scripts/claude-setup.sh core     # Show only core plugins
#   ./scripts/claude-setup.sh --copy   # Output commands ready to paste

set -e

# Plugin categories
CORE_PLUGINS=(
    "context7"
    "feature-dev"
    "code-review"
    "commit-commands"
    "pr-review-toolkit"
    "plugin-dev"
)

STYLE_PLUGINS=(
    "explanatory-output-style"
    "learning-output-style"
)

INTEGRATION_PLUGINS=(
    "greptile"
    "supabase"
    "figma"
    "slack"
)

LANGUAGE_PLUGINS=(
    "typescript-lsp"
    "pyright-lsp"
    "rust-analyzer-lsp"
    "jdtls-lsp"
)

TESTING_PLUGINS=(
    "playwright"
    "security-guidance"
    "agent-sdk-dev"
    "hookify"
)

SPECIALIZED_PLUGINS=(
    "frontend-design"
    "serena"
    "ralph-wiggum"
    "ralph-loop"
)

print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Claude Code Environment Bootstrap"
    echo "  Project: NC_Slicer-Claude-Bridge"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
}

print_category() {
    local name="$1"
    shift
    local plugins=("$@")

    echo "▸ $name (${#plugins[@]} plugins)"
    for plugin in "${plugins[@]}"; do
        if [[ "$COPY_MODE" == "true" ]]; then
            echo "/plugin install $plugin"
        else
            echo "  • $plugin"
        fi
    done
    echo ""
}

# Parse arguments
COPY_MODE="false"
FILTER=""

for arg in "$@"; do
    case $arg in
        --copy)
            COPY_MODE="true"
            ;;
        core|styles|integrations|languages|testing|specialized)
            FILTER="$arg"
            ;;
        --help|-h)
            echo "Usage: $0 [category] [--copy]"
            echo ""
            echo "Categories: core, styles, integrations, languages, testing, specialized"
            echo "Options:"
            echo "  --copy    Output as /plugin install commands"
            echo "  --help    Show this help"
            exit 0
            ;;
    esac
done

if [[ "$COPY_MODE" != "true" ]]; then
    print_header
fi

# Print categories based on filter
case $FILTER in
    core)
        print_category "Core Development" "${CORE_PLUGINS[@]}"
        ;;
    styles)
        print_category "Output Styles" "${STYLE_PLUGINS[@]}"
        ;;
    integrations)
        print_category "Integrations" "${INTEGRATION_PLUGINS[@]}"
        ;;
    languages)
        print_category "Language Support" "${LANGUAGE_PLUGINS[@]}"
        ;;
    testing)
        print_category "Testing & Quality" "${TESTING_PLUGINS[@]}"
        ;;
    specialized)
        print_category "Specialized" "${SPECIALIZED_PLUGINS[@]}"
        ;;
    *)
        # Show all categories
        print_category "Core Development" "${CORE_PLUGINS[@]}"
        print_category "Output Styles" "${STYLE_PLUGINS[@]}"
        print_category "Integrations" "${INTEGRATION_PLUGINS[@]}"
        print_category "Language Support" "${LANGUAGE_PLUGINS[@]}"
        print_category "Testing & Quality" "${TESTING_PLUGINS[@]}"
        print_category "Specialized" "${SPECIALIZED_PLUGINS[@]}"
        ;;
esac

if [[ "$COPY_MODE" != "true" ]]; then
    TOTAL=$((${#CORE_PLUGINS[@]} + ${#STYLE_PLUGINS[@]} + ${#INTEGRATION_PLUGINS[@]} + ${#LANGUAGE_PLUGINS[@]} + ${#TESTING_PLUGINS[@]} + ${#SPECIALIZED_PLUGINS[@]}))
    echo "───────────────────────────────────────────────────────────"
    echo "Total: $TOTAL plugins"
    echo ""
    echo "Quick install (core only):"
    echo "  Run: $0 core --copy | pbcopy"
    echo ""
    echo "After installing plugins:"
    echo "  /help           # Verify commands available"
    echo "  /todo           # Check project TODO list"
    echo "  /bootstrap-env  # Full environment guide"
fi
