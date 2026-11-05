"""Command handlers registry."""

from __future__ import annotations

from renslate.cli.handlers.data import cmd_export, cmd_import
from renslate.cli.handlers.discourse import cmd_analyze, cmd_bootstrap
from renslate.cli.handlers.project import (
    cmd_apply,
    cmd_extract,
    cmd_generate,
    cmd_init,
    get_project_root,
)
from renslate.cli.handlers.qa import cmd_clean_db, cmd_qa
from renslate.cli.handlers.tools import (
    cmd_dashboard,
    cmd_decompile,
    cmd_font,
    cmd_status,
    cmd_wizard,
)
from renslate.cli.handlers.translation import cmd_preview, cmd_run, cmd_translate

__all__ = [
    "cmd_analyze",
    "cmd_apply",
    "cmd_bootstrap",
    "cmd_clean_db",
    "cmd_dashboard",
    "cmd_decompile",
    "cmd_export",
    "cmd_extract",
    "cmd_font",
    "cmd_generate",
    "cmd_import",
    "cmd_init",
    "cmd_preview",
    "cmd_qa",
    "cmd_run",
    "cmd_status",
    "cmd_translate",
    "cmd_wizard",
    "get_project_root",
]
