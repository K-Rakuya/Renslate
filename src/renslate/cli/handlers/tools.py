"""Standalone utility command handlers: decompile, font, dashboard, wizard, status."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from renslate.cli.handlers.project import get_project_root
from renslate.core.project import Project


def cmd_decompile(args: argparse.Namespace) -> int:
    """Decompile Ren'Py bytecode (.rpyc) files into readable script (.rpy)."""
    from renslate.renpy.rpyc import decompile_rpyc

    raw_target = getattr(args, "target", None)
    out_arg = getattr(args, "output", None)
    overwrite = getattr(args, "overwrite", False)

    if raw_target:
        target_path = Path(raw_target).resolve()
    else:
        root = get_project_root(args)
        target_path = (root / "game").resolve() if (root / "game").is_dir() else root.resolve()

    if target_path.is_file():
        if target_path.suffix != ".rpyc":
            print(f"Lỗi: Tệp không phải đuôi .rpyc: {target_path}", file=sys.stderr)
            return 1
        out_path: Path
        if out_arg:
            p_out = Path(out_arg)
            if p_out.is_dir() or str(out_arg).endswith(("/", "\\")):
                out_path = p_out / target_path.with_suffix(".rpy").name
            else:
                out_path = p_out
        else:
            out_path = target_path.with_suffix(".rpy")

        if out_path.exists() and not overwrite:
            print(
                f"Bỏ qua: Tệp {out_path} đã tồn tại (dùng --overwrite để ghi đè)",
                file=sys.stderr,
            )
            return 1

        try:
            decompile_rpyc(target_path, out_path)
        except Exception as e:
            print(f"Lỗi khi dịch ngược {target_path}: {e}", file=sys.stderr)
            return 1
        print(f"Đã dịch ngược: {target_path.name} -> {out_path}")
        return 0

    if target_path.is_dir():
        rpyc_files = sorted(
            [p for p in target_path.rglob("*.rpyc") if "tl" not in p.parts],
            key=lambda p: str(p),
        )
        if not rpyc_files:
            print(f"Không tìm thấy tệp .rpyc nào trong thư mục: {target_path}", file=sys.stderr)
            return 1

        out_dir = Path(out_arg).resolve() if out_arg else target_path
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        failed_count = 0
        for rpyc_file in rpyc_files:
            rel = rpyc_file.relative_to(target_path)
            dest = (out_dir / rel).with_suffix(".rpy")
            if dest.exists() and not overwrite:
                continue
            try:
                decompile_rpyc(rpyc_file, dest)
                count += 1
            except Exception as e:
                failed_count += 1
                if getattr(args, "verbose", False):
                    print(f"Lỗi khi dịch ngược {rpyc_file}: {e}", file=sys.stderr)

        if count == 0 and failed_count > 0:
            print(f"Lỗi: Không thể dịch ngược {failed_count} tệp .rpyc", file=sys.stderr)
            return 1

        print(f"Đã dịch ngược {count}/{len(rpyc_files)} tệp .rpyc vào {out_dir}")
        return 0

    print(f"Lỗi: Đường dẫn không tồn tại: {target_path}", file=sys.stderr)
    return 1


def cmd_font(args: argparse.Namespace) -> int:
    """Discover fonts in game directory and configure automated Vietnamese font mapping."""
    project = Project.open(get_project_root(args))
    from renslate.renpy.fonts import discover_game_fonts, setup_vietnamese_support

    detected = discover_game_fonts(project)
    print(f"Quét phông chữ cho dự án: {project.root}")
    if detected:
        print(f"Đã phát hiện {len(detected)} phông chữ trong game:")
        for f in detected:
            if f.supports_vi:
                status = "✓ Hỗ trợ đầy đủ tiếng Việt"
            else:
                status = f"✗ Thiếu {f.missing_count} ký tự dấu tiếng Việt"
            print(f"  - [{f.style.upper():<7}] {f.rel_path} -> {status}")
    else:
        print("  Không tìm thấy tệp phông chữ tùy biến trong game/ (dùng phông mặc định của Ren'Py).")

    if not getattr(args, "inspect", False):
        font_path = setup_vietnamese_support(
            project,
            font_file=getattr(args, "sans_font", None),
            serif_font=getattr(args, "serif_font", None),
            auto_replace_game_fonts=not getattr(args, "no_replace", False),
        )
        print(f"\nĐã tạo cấu hình phông tiếng Việt tự động tại: {project.relative(font_path)}")
        print("  - Giao diện GUI: Tự động dùng Be Vietnam Pro / Nunito / DejaVuSans")
        print("  - Thẻ {font=...}: Tự động thay thế qua config.font_replacement_map")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch real-time progress dashboard in terminal."""
    project = Project.open(get_project_root(args))
    from renslate.ui.dashboard import run_live_dashboard

    interval = float(getattr(args, "interval", 2.0))
    once = bool(getattr(args, "once", False))
    plain = bool(getattr(args, "plain", False))
    limit = int(getattr(args, "limit", 5))
    filter_mode = str(getattr(args, "filter", "all"))

    return run_live_dashboard(
        project,
        interval=interval,
        once=once,
        plain=plain,
        limit=limit,
        initial_filter=filter_mode,
    )


def cmd_status(args: argparse.Namespace) -> int:
    """Display translation metrics and status matrix for project."""
    if getattr(args, "dashboard", False):
        args.once = True
        return cmd_dashboard(args)

    project = Project.open(get_project_root(args))
    with project.open_db() as database:
        rows = database.stats()
        slots = database.count_slots()
    print(f"Dự án: {project.root}")
    print(f"Ngôn ngữ đích: {project.config.target_lang}")
    if not rows:
        print("Chưa có dữ liệu, hãy chạy 'renslate extract'.")
        return 0
    print(f"{'loại':<10} {'trạng thái':<12} {'số lượng':>8}")
    for kind, status, total in rows:
        print(f"{kind.value:<10} {status.value:<12} {total:>8}")
    print(f"Tổng: {sum(total for _, _, total in rows)} đơn vị, {slots} vị trí ghi")
    return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    """Launch interactive setup wizard or studio menu."""
    from renslate.ui.wizard import run_interactive_wizard

    classic = getattr(args, "classic", False)
    use_menu = getattr(args, "menu", False) or (not classic and sys.stdin.isatty())
    return run_interactive_wizard(get_project_root(args), interactive_menu=use_menu)


def cmd_board(args: argparse.Namespace) -> int:
    """Launch interactive TUI Control Board for 2026 quality flags."""
    root = get_project_root(args)
    try:
        project = Project.open(root)
    except Exception:
        project = Project.create(root) if (root / "game").is_dir() else None

    from renslate.ui.control_board import run_control_board

    cfg_dict, action = run_control_board(project)
    if action == "START" and cfg_dict is not None and project is not None:
        from renslate.ui.runner import run_unattended_pipeline

        chunk = int(cfg_dict.get("batch_size", 8))
        return run_unattended_pipeline(
            project.root,
            batch_chunk=chunk,
            guard=bool(cfg_dict.get("guard", True)),
            concise_tags=bool(cfg_dict.get("concise_tags", False)),
            auto_apply=bool(cfg_dict.get("auto_apply", False)),
        )
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    """Launch real-time bilingual translation studio and in-place editor."""
    root = get_project_root(args)
    try:
        project = Project.open(root)
    except Exception:
        if (root / "game").is_dir():
            project = Project.create(root)
        else:
            print(f"Lỗi: Thư mục không phải dự án Ren'Py hợp lệ: {root}", file=sys.stderr)
            return 1

    from renslate.core.models import LiveDialogueItem
    from renslate.ui.studio import StudioState, run_studio_session

    state = StudioState()
    with project.open_db() as db:
        recent = db.all_units()
        for u in recent:
            if u.target:
                state.add_or_update_unit(
                    LiveDialogueItem(
                        uid=u.uid,
                        seq=u.seq,
                        speaker=u.speaker or "",
                        listener=None,
                        source=u.source,
                        target=u.target,
                        status=u.status.value,
                        origin=u.origin.value if u.origin else "manual",
                        qa_issues=u.qa_flags,
                    )
                )

        run_studio_session(project, state=state, db=db)
    return 0
