from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SLICER_PROD_DIR = ROOT_DIR / "slicer-prod"


def test_claude_commands_are_symlinked_to_canonical_directory() -> None:
    commands_path = SLICER_PROD_DIR / ".claude" / "commands"

    assert commands_path.is_symlink()
    assert commands_path.readlink() == Path("../../.claude/commands")


def test_claude_code_project_mcp_config_points_to_parent_repo() -> None:
    config = json.loads((SLICER_PROD_DIR / ".mcp.json").read_text())

    assert config == {
        "mcpServers": {
            "slicer-bridge": {
                "command": "uv",
                "args": ["--directory", "..", "run", "slicer-mcp"],
                "env": {"SLICER_URL": "http://localhost:2016"},
            }
        }
    }


def test_opencode_project_config_points_to_parent_repo() -> None:
    config = json.loads((SLICER_PROD_DIR / "opencode.json").read_text())

    assert config == {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "slicer-bridge": {
                "type": "local",
                "enabled": True,
                "command": ["uv", "--directory", "..", "run", "slicer-mcp"],
                "environment": {"SLICER_URL": "http://localhost:2016"},
                "timeout": 10000,
            }
        },
    }


def test_sync_surface_check_mode_passes() -> None:
    script_path = SLICER_PROD_DIR / "scripts" / "sync_surface.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--check"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT_DIR,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "slicer-prod check OK" in result.stdout
