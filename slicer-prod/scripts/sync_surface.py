#!/usr/bin/env python3
"""Synchronize slicer-prod command surface from canonical project files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _prod_dir(root_dir: Path) -> Path:
    return root_dir / "slicer-prod"


def _source_commands_dir(root_dir: Path) -> Path:
    return root_dir / ".claude" / "commands"


def _prod_claude_commands_path(prod_dir: Path) -> Path:
    return prod_dir / ".claude" / "commands"


def _prod_opencode_commands_dir(prod_dir: Path) -> Path:
    return prod_dir / ".opencode" / "commands"


def _symlink_target_for_prod_claude_commands() -> Path:
    return Path("../../.claude/commands")


def _yaml_single_quoted(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _extract_description(command_text: str, fallback: str) -> str:
    for line in command_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return fallback


def _render_opencode_command(command_name: str, source_text: str) -> str:
    description = _extract_description(source_text, command_name)
    return f"---\ndescription: {_yaml_single_quoted(description)}\n---\n\n{source_text.rstrip()}\n"


def _ensure_claude_symlink(prod_dir: Path, check_only: bool) -> list[str]:
    issues: list[str] = []
    link_path = _prod_claude_commands_path(prod_dir)
    expected_target = _symlink_target_for_prod_claude_commands()

    if link_path.is_symlink():
        current_target = Path(link_path.readlink())
        if current_target != expected_target:
            issues.append(
                f"Claude commands symlink points to {current_target}, expected {expected_target}"
            )
            if not check_only:
                link_path.unlink()
                link_path.symlink_to(expected_target)
                issues.pop()
        return issues

    if link_path.exists():
        issues.append(f"Claude commands path exists but is not a symlink: {link_path}")
        if check_only:
            return issues
        if link_path.is_dir():
            raise RuntimeError(f"Refusing to replace non-symlink directory: {link_path}")
        link_path.unlink()

    if check_only:
        issues.append(f"Missing Claude commands symlink: {link_path}")
        return issues

    link_path.symlink_to(expected_target)
    return issues


def _sync_opencode_commands(root_dir: Path, prod_dir: Path, check_only: bool) -> list[str]:
    issues: list[str] = []
    source_dir = _source_commands_dir(root_dir)
    target_dir = _prod_opencode_commands_dir(prod_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob("*.md"))
    expected_names = {path.name for path in source_files}
    current_names = {path.name for path in target_dir.glob("*.md")}

    unexpected = sorted(current_names - expected_names)
    missing = sorted(expected_names - current_names)

    if unexpected:
        issues.append(f"Unexpected OpenCode command files: {', '.join(unexpected)}")
    if missing:
        issues.append(f"Missing OpenCode command files: {', '.join(missing)}")

    for source_file in source_files:
        expected_text = _render_opencode_command(source_file.stem, source_file.read_text())
        target_file = target_dir / source_file.name

        if not target_file.exists():
            if check_only:
                continue
            target_file.write_text(expected_text)
            continue

        current_text = target_file.read_text()
        if current_text != expected_text:
            issues.append(f"Out-of-sync OpenCode command: {source_file.name}")
            if not check_only:
                target_file.write_text(expected_text)

    if not check_only:
        for extra_name in unexpected:
            (target_dir / extra_name).unlink()
        issues = [
            issue for issue in issues if not issue.startswith("Unexpected OpenCode command files")
        ]
        issues = [
            issue for issue in issues if not issue.startswith("Missing OpenCode command files")
        ]
        issues = [issue for issue in issues if not issue.startswith("Out-of-sync OpenCode command")]

    return issues


def run(check_only: bool) -> int:
    root_dir = _root_dir()
    prod_dir = _prod_dir(root_dir)

    issues: list[str] = []
    issues.extend(_ensure_claude_symlink(prod_dir, check_only=check_only))
    issues.extend(_sync_opencode_commands(root_dir, prod_dir, check_only=check_only))

    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1

    mode = "check" if check_only else "sync"
    print(f"slicer-prod {mode} OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of rewriting if slicer-prod is out of sync",
    )
    args = parser.parse_args()
    return run(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
