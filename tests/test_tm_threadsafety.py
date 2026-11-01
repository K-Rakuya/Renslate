"""Regression: the TM index must survive concurrent fuzzy lookups and recording.

The translation pipeline prepares the next batch (fuzzy lookups) on a worker thread while the
main thread records fresh translations, which used to raise
``RuntimeError: Set changed size during iteration`` intermittently.
"""

from __future__ import annotations

import threading
from pathlib import Path

from renslate.core.project import Project
from renslate.engine.tm import TranslationMemory


def test_concurrent_lookup_fuzzy_and_record(tmp_path: Path) -> None:
    root = tmp_path / "game_root"
    (root / "game").mkdir(parents=True)
    project = Project.create(root)
    errors: list[BaseException] = []

    with project.open_db() as db:
        tm = TranslationMemory(db)
        for i in range(200):
            tm.record(f"The quick brown fox number {i} jumps", f"Cáo nâu số {i} nhảy")

        def reader() -> None:
            try:
                for _ in range(150):
                    tm.lookup_fuzzy("The quick brown fox number 7 jumps over", threshold=0.5)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for i in range(200, 500):
            tm.record(f"The quick brown fox number {i} jumps higher", f"Cáo {i} nhảy cao")
        for t in threads:
            t.join(timeout=60)

    assert not errors, errors
