"""Shared fixtures for behavior-capture tests.
CRITICAL: isaacgym must be imported before torch.
"""

from __future__ import annotations

import glob
import os

import pytest

_ISAACGYM_AVAILABLE = True
try:
    import isaacgym  # noqa: F401
except (ImportError, RuntimeError):
    _ISAACGYM_AVAILABLE = False

if not _ISAACGYM_AVAILABLE:
    # Prevent pytest from importing test files that require isaacgym.
    # collect_ignore is evaluated before module import, avoiding RuntimeError.
    _test_dir = os.path.dirname(__file__)
    collect_ignore = glob.glob(os.path.join(_test_dir, "test_*.py"))


@pytest.fixture
def device() -> str:
    return "cpu"


@pytest.fixture
def num_envs() -> int:
    return 4
