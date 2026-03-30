"""Shared fixtures for integration tests requiring a running 3D Slicer instance."""

import os

import pytest
import requests

from slicer_mcp.core.slicer_client import SlicerClient


@pytest.fixture(scope="session")
def slicer_url() -> str:
    """Return the Slicer WebServer URL from environment or default.

    Reads SLICER_URL env var, falling back to http://localhost:2016.
    """
    return os.environ.get("SLICER_URL", "http://localhost:2016")


@pytest.fixture(scope="session")
def slicer_available(slicer_url: str) -> bool:
    """Check whether 3D Slicer WebServer is reachable.

    Makes a lightweight GET to /slicer/mrml to verify connectivity.
    If Slicer is not running, all dependent tests are skipped.
    """
    try:
        resp = requests.get(f"{slicer_url}/slicer/mrml", timeout=5)
        resp.raise_for_status()
        return True
    except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
        pytest.skip(
            f"3D Slicer WebServer not reachable at {slicer_url}. "
            "Start Slicer with WebServer enabled to run integration tests."
        )
        return False  # unreachable, keeps type checker happy


@pytest.fixture
def live_client(slicer_available: bool, slicer_url: str) -> SlicerClient:
    """Return a real SlicerClient connected to the running Slicer instance.

    Depends on slicer_available so tests are automatically skipped
    when Slicer is not reachable.
    """
    return SlicerClient(base_url=slicer_url, timeout=30)
