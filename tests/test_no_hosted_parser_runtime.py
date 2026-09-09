from __future__ import annotations

import importlib.metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _prohibited_markers() -> tuple[str, ...]:
    return (
        "llama" + "parse",
        "llama" + "_parse",
        "llama" + "cloud",
        "llama" + "_cloud",
        "api." + "cloud." + "llamaindex",
        "@llamaindex/" + "cloud",
    )


def test_dependencies_and_runtime_have_no_hosted_parser_package_or_fallback():
    dependency_text = (ROOT / "pyproject.toml").read_text().casefold()
    runtime_text = "\n".join(
        path.read_text(errors="ignore")
        for path in sorted((ROOT / "src").rglob("*.py"))
    ).casefold()
    for marker in _prohibited_markers():
        assert marker not in dependency_text
        assert marker not in runtime_text

    installed = {
        distribution.metadata["Name"].casefold()
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    assert not any(
        marker in package
        for package in installed
        for marker in _prohibited_markers()
    )


def test_runtime_configuration_has_no_hosted_parser_secret_or_endpoint():
    config_text = (ROOT / "src/geek_crawler_rag/config.py").read_text().casefold()
    env_text = (ROOT / ".env.example").read_text().casefold()
    combined = config_text + env_text
    assert not any(marker in combined for marker in _prohibited_markers())
