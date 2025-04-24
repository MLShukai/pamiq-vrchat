"""Tests package top level features."""

import tomllib
from pathlib import Path

import python_uv_template

PROJECT_ROOT = Path(__file__).parent.parent


def test_version():
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    assert python_uv_template.__version__ == pyproject["project"]["version"]
