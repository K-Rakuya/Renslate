"""Tests for centralized CLI console output formatting and ANSI abstraction."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from renslate.cli.formatter import ConsoleFormatter, print_extract_report, print_issues
from renslate.renpy.extract import ExtractReport
from renslate.renpy.parser import Issue, Severity


def test_console_formatter_plain_mode(capsys: pytest.CaptureFixture[str]) -> None:
    fmt = ConsoleFormatter(plain=True)
    fmt.info("Hello world")
    fmt.success("Operation succeeded")
    fmt.warning("Check something")
    fmt.error("Fatal error")

    captured = capsys.readouterr()
    assert "Hello world\n" in captured.out
    assert "✓ Operation succeeded\n" in captured.out
    assert "▲ Check something\n" in captured.err
    assert "✗ Fatal error\n" in captured.err
    # In plain mode, no ANSI escape sequences
    assert "\033[" not in captured.out
    assert "\033[" not in captured.err


def test_console_formatter_color_mode() -> None:
    out = io.StringIO()
    err = io.StringIO()
    fmt = ConsoleFormatter(plain=False, force_color=True, out_stream=out, err_stream=err)
    fmt.success("Success in color")
    fmt.error("Error in color")

    assert "\033[" in out.getvalue()
    assert "\033[" in err.getvalue()
    assert "Success in color" in out.getvalue()
    assert "Error in color" in err.getvalue()


def test_print_issues_truncation(capsys: pytest.CaptureFixture[str]) -> None:
    issues = [
        Issue(
            file=Path("test.rpy"),
            line=i,
            code="test_code",
            message=f"Message {i}",
            severity=Severity.WARNING,
        )
        for i in range(25)
    ]
    print_issues(issues, max_shown=10)
    captured = capsys.readouterr()
    assert "test.rpy:0: warning [test_code] Message 0" in captured.err
    assert "test.rpy:9: warning [test_code] Message 9" in captured.err
    assert "... và 15 vấn đề khác" in captured.err


def test_print_extract_report(capsys: pytest.CaptureFixture[str]) -> None:
    report = ExtractReport(
        files=5,
        units=120,
        slots=150,
        created=100,
        source_changed=10,
        imported=5,
        restored=3,
        obsolete=2,
    )
    print_extract_report(report)
    captured = capsys.readouterr()
    assert "Đã quét 5 tệp, 120 đơn vị dịch, 150 vị trí ghi" in captured.out
    assert "mới 100, đổi nguồn 10, nhập bản dịch có sẵn 5" in captured.out
