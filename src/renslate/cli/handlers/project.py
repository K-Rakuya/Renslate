"""Project lifecycle command handlers: init, generate, extract, apply."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from renslate.cli.formatter import print_apply_report, print_extract_report
from renslate.core.project import Project
from renslate.renpy.extract import extract
from renslate.renpy.sdk import generate_translations
from renslate.renpy.writer import apply_translations, setup_vietnamese_support

_LOG = logging.getLogger("renslate.cli.project")


def get_project_root(args: argparse.Namespace) -> Path:
    """Resolve target project root directory from CLI arguments."""
    return Path(args.project) if getattr(args, "project", None) else Path.cwd()


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new translation project for a Ren'Py game."""
    project = Project.create(
        get_project_root(args),
        target_lang=args.lang,
        sdk=args.sdk,
        force=args.force,
    )
    print(f"Đã khởi tạo dự án tại {project.root}")
    print(f"  cấu hình       : {project.relative(project.config_path)}")
    print(f"  cơ sở dữ liệu  : {project.relative(project.db_path)}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate translation skeleton files using Ren'Py SDK."""
    project = Project.open(get_project_root(args))
    sdk = args.sdk or project.config.sdk
    sdk_path = Path(sdk).expanduser() if sdk else None
    output = generate_translations(sdk_path, project.root, project.config.target_lang)
    _LOG.info("%s", output)
    print(f"Đã sinh khung dịch trong {project.relative(project.tl_root)}")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract translatable strings from Ren'Py scripts into project database."""
    project = Project.open(get_project_root(args))
    with project.open_db() as database:
        report = extract(project, database)
    print_extract_report(report)
    return 1 if report.errors else 0


def cmd_apply(args: argparse.Namespace) -> int:
    """Apply translated strings back into Ren'Py game script files."""
    project = Project.open(get_project_root(args))
    if (
        getattr(args, "setup_font", False)
        or getattr(args, "sans_font", None)
        or getattr(args, "serif_font", None)
    ):
        font_path = setup_vietnamese_support(
            project,
            font_file=getattr(args, "sans_font", None),
            serif_font=getattr(args, "serif_font", None),
        )
        print(f"Đã tạo cấu hình phông tiếng Việt: {project.relative(font_path)}")

    with project.open_db() as database:
        report = apply_translations(project, database, dry_run=args.dry_run)
    print_apply_report(report)
    return 0
