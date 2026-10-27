from __future__ import annotations

from vitrans.project import Project


def read_tree(project: Project) -> dict[str, bytes]:
    """Snapshot of all translation files, keyed by project-relative path."""
    return {
        project.relative(path): path.read_bytes() for path in sorted(project.tl_root.rglob("*.rpy"))
    }
