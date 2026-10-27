from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from vitrans.db import Database
from vitrans.project import Project

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def project(tmp_path: Path) -> Project:
    root = tmp_path / "game_root"
    shutil.copytree(DATA_DIR / "tl", root / "game" / "tl")
    return Project.create(root)


@pytest.fixture
def database(project: Project) -> Iterator[Database]:
    with project.open_db() as db:
        yield db
