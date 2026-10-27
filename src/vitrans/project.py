"""Project layout: where configuration, store, backups and translation files live."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vitrans.config import Config
from vitrans.db import Database
from vitrans.errors import ProjectError

WORK_DIR_NAME = ".vitrans"


@dataclass(frozen=True, slots=True)
class Project:
    root: Path
    config: Config

    @property
    def work_dir(self) -> Path:
        return self.root / WORK_DIR_NAME

    @property
    def config_path(self) -> Path:
        return self.work_dir / "config.json"

    @property
    def db_path(self) -> Path:
        return self.work_dir / "project.db"

    @property
    def backup_root(self) -> Path:
        return self.work_dir / "backup"

    @property
    def tl_root(self) -> Path:
        return self.root / self.config.tl_dir / self.config.target_lang

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        target_lang: str = "vietnamese",
        sdk: str | None = None,
        force: bool = False,
    ) -> Project:
        root = root.resolve()
        if not (root / "game").is_dir():
            raise ProjectError(f"không phải dự án Ren'Py (thiếu thư mục 'game'): {root}")
        project = cls(root, Config(target_lang=target_lang, sdk=sdk))
        if project.config_path.exists() and not force:
            raise ProjectError(
                f"dự án đã được khởi tạo: {project.config_path} (dùng --force để ghi đè)"
            )
        project.config.save(project.config_path)
        project.open_db(create=True).close()
        return project

    @classmethod
    def open(cls, root: Path) -> Project:
        root = root.resolve()
        config_path = root / WORK_DIR_NAME / "config.json"
        if not config_path.is_file():
            raise ProjectError(f"dự án chưa được khởi tạo, hãy chạy 'vitrans init' trong {root}")
        return cls(root, Config.load(config_path))

    def open_db(self, *, create: bool = False) -> Database:
        return Database.open(self.db_path, create=create)

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
