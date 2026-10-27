from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from renslate.config import Config
from renslate.errors import ConfigError


def test_defaults_match_the_design_document() -> None:
    config = Config()
    assert config.target_lang == "vietnamese"
    assert config.llm.base_url == "http://127.0.0.1:8080/v1"
    assert config.llm.n_ctx == 6144
    assert config.batch.max_lines == 25
    assert config.tm.fuzzy_threshold == 0.88
    assert config.qa.max_length_ratio == 1.8


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    config = Config(target_lang="vi_VN", sdk="/opt/renpy")
    path = tmp_path / "nested" / "config.json"
    config.save(path)
    assert Config.load(path) == config
    assert "tự nhiên" in path.read_text(encoding="utf-8")


def test_partial_sections_keep_other_defaults() -> None:
    config = Config.from_dict({"llm": {"temperature": 1}, "batch": {"workers": 2}})
    assert config.llm.temperature == 1.0
    assert isinstance(config.llm.temperature, float)
    assert config.llm.n_ctx == 6144
    assert config.batch.workers == 2


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"unknown": 1},
        {"llm": {"unknown": 1}},
        {"llm": []},
        {"target_lang": 5},
        {"target_lang": "None"},
        {"target_lang": "bad name"},
        {"tl_dir": "/abs"},
        {"tl_dir": "../escape"},
        {"sdk": 3},
        {"llm": {"n_ctx": 12.5}},
        {"llm": {"n_ctx": True}},
        {"llm": {"n_ctx": 10}},
        {"llm": {"temperature": 3}},
        {"llm": {"base_url": "ftp://x"}},
        {"llm": {"model": ""}},
        {"batch": {"workers": 0}},
        {"tm": {"fuzzy_threshold": 1.5}},
        {"qa": {"max_length_ratio": 0.5}},
    ],
)
def test_invalid_configuration_is_rejected(raw: Any) -> None:
    with pytest.raises(ConfigError):
        Config.from_dict(raw)


def test_load_reports_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        Config.load(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        Config.load(broken)
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"target_lang": "vietnamese"}), encoding="utf-8")
    assert Config.load(valid).target_lang == "vietnamese"
