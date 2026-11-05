"""Tests for ergonomic CLI argument parser and 2026 literary quality flags."""

from __future__ import annotations

import pytest

from renslate.cli.parser import build_parser


def test_parser_version() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0


def test_parser_modern_2026_translate_flags() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "translate",
        "--deliberate",
        "--calque-threshold", "0.42",
        "--exemplars", "3",
        "--narrative-memory", "8",
        "--transgraph",
        "--studio",
    ])
    assert args.command == "translate"
    assert args.deliberate is True
    assert args.calque_threshold == 0.42
    assert args.exemplars == 3
    assert args.narrative_memory == 8
    assert args.transgraph is True
    assert args.studio is True


def test_parser_no_deliberate_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["translate", "--no-deliberate", "--no-transgraph"])
    assert args.command == "translate"
    assert args.no_deliberate is True
    assert args.no_transgraph is True


def test_parser_backward_aliases_run_auto() -> None:
    parser = build_parser()
    args_run = parser.parse_args(["run", "--chunk-size", "50", "--retry-failed"])
    assert args_run.command == "run"
    assert args_run.chunk_size == 50
    assert args_run.retry_failed is True

    args_auto = parser.parse_args(["auto", "--chunk-size", "50", "--retry-failed"])
    assert args_auto.command == "auto"
    assert args_auto.chunk_size == 50
    assert args_auto.retry_failed is True


def test_parser_backward_aliases_wizard_interactive() -> None:
    parser = build_parser()
    args_wiz = parser.parse_args(["wizard", "--studio"])
    assert args_wiz.command == "wizard"
    assert args_wiz.menu is True

    args_ia = parser.parse_args(["interactive", "--classic"])
    assert args_ia.command == "interactive"
    assert args_ia.classic is True
