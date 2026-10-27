"""Tier 5 Adversarial Stress Harness for Renslate Translation Pipeline Resiliency & Continuity.

Empirically validates:
1. Partial batch recovery across varying batch sizes (5, 20) with failures at index 0,
   middle, last, and multiple positions caused by QA markup errors, unmask exceptions,
   missing keys, and length ratio overflows. Confirms valid units commit to SQLite with
   Status.TRANSLATED without re-evaluation, while failing units retry individually.
2. Bounded lookback deque buffer with FIFO eviction over 100 units under lookback=4.
3. Cross-chunk dialogue context continuity across pipeline instances via SQLite persistence.
4. Lookback slicing safety across lookback configs (0, 1, 5, 100) and deque inputs.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from renslate.config import Config
from renslate.context import ContextBuilder, LookbackItem
from renslate.errors import LLMError
from renslate.models import Status, Unit, UnitKind
from renslate.project import Project
from renslate.translate import TranslationPipeline


def _create_test_project(tmp_path: Path, name: str = "stress_game") -> Project:
    root = tmp_path / name
    (root / "game").mkdir(parents=True, exist_ok=True)
    return Project.create(root)


# ==============================================================================
# SECTION 1: PARTIAL BATCH RECOVERY STRESS TESTS
# ==============================================================================


@pytest.mark.parametrize("batch_size", [5, 20])
@pytest.mark.parametrize("fail_pos", ["first", "middle", "last", "first_and_last"])
@pytest.mark.parametrize(
    "failure_type", ["qa_tag_mismatch", "unmask_corruption", "qa_ratio_overflow"]
)
def test_partial_batch_recovery_empirical_matrix(
    tmp_path: Path,
    batch_size: int,
    fail_pos: str,
    failure_type: str,
) -> None:
    """Stress-test partial batch recovery matrix across batch sizes, positions, and causes.

    Invariants verified:
    1. Valid units are committed to SQLite with Status.TRANSLATED immediately.
    2. Valid units are NEVER re-evaluated in subsequent LLM calls.
    3. Failing units are isolated and retried individually in fallback single-unit calls.
    4. Database state accurately reflects successful retries or Status.FAILED.
    """
    proj_name = f"proj_bs{batch_size}_{fail_pos}_{failure_type}"
    project = _create_test_project(tmp_path, proj_name)
    cfg = Config.from_dict({"batch": {"max_lines": batch_size, "lookback": 4}})

    # Determine failing indices
    failing_indices: set[int] = set()
    if fail_pos == "first":
        failing_indices.add(0)
    elif fail_pos == "middle":
        failing_indices.add(batch_size // 2)
    elif fail_pos == "last":
        failing_indices.add(batch_size - 1)
    elif fail_pos == "first_and_last":
        failing_indices.add(0)
        failing_indices.add(batch_size - 1)

    units: list[Unit] = []
    with project.open_db() as db:
        for i in range(batch_size):
            if failure_type == "unmask_corruption":
                src = f"Speaker {i}: Line {i} with {{b}}bold{{/b}} formatting."
            elif failure_type == "qa_tag_mismatch":
                src = f"Speaker {i}: Line {i} with {{i}}italic{{/i}} styling."
            else:
                src = f"Speaker {i}: This is normal sentence {i} for testing."

            u = Unit(
                uid=f"u_{batch_size}_{fail_pos}_{i}",
                kind=UnitKind.DIALOGUE,
                seq=i,
                source=src,
                status=Status.PENDING,
                speaker=f"spk_{i}",
                label="ch1",
            )
            db.save_unit(u)
            units.append(u)

    call_records: list[dict[str, Any]] = []

    def mock_complete_json(
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, str], int, int]:
        schema_props = list(schema.get("properties", {}).keys()) if schema else []
        call_records.append(
            {
                "requested_uids": schema_props,
                "is_single": len(schema_props) == 1,
                "prompt": user_prompt,
            }
        )

        results: dict[str, str] = {}
        for uid in schema_props:
            idx = int(uid.rsplit("_", 1)[-1])
            is_fail_unit = idx in failing_indices

            if is_fail_unit and len(schema_props) > 1:
                if failure_type == "qa_tag_mismatch":
                    results[uid] = f"Người nói {idx}: Dòng {idx} với {{i}}nghiêng{{/b}} phong cách."
                elif failure_type == "unmask_corruption":
                    results[uid] = f"Người nói {idx}: Dòng {idx} với ⟦V99_INVALID⟧ định dạng."
                elif failure_type == "qa_ratio_overflow":
                    results[uid] = f"Người nói {idx}: " + ("Quá dài vượt tỉ lệ! " * 20)
            elif is_fail_unit and len(schema_props) == 1:
                if failure_type == "qa_tag_mismatch":
                    results[uid] = f"Người nói {idx}: Dòng {idx} với {{i}}nghiêng{{/i}} phong cách."
                elif failure_type == "unmask_corruption":
                    results[uid] = f"Người nói {idx}: Dòng {idx} với {{b}}đậm{{/b}} định dạng."
                elif failure_type == "qa_ratio_overflow":
                    results[uid] = f"Người nói {idx}: Câu bình thường {idx}."
            else:
                if failure_type == "qa_tag_mismatch":
                    results[uid] = f"Người nói {idx}: Dòng {idx} với {{i}}nghiêng{{/i}} phong cách."
                elif failure_type == "unmask_corruption":
                    results[uid] = f"Người nói {idx}: Dòng {idx} với {{b}}đậm{{/b}} định dạng."
                else:
                    results[uid] = f"Người nói {idx}: Câu bình thường {idx}."

        return (results, len(schema_props) * 20, len(schema_props) * 10)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        # Invariant 1: First call was the full batch
        expected_calls = 1 + len(failing_indices)
        assert len(call_records) == expected_calls, (
            f"Expected {expected_calls} calls, got {len(call_records)}"
        )
        assert set(call_records[0]["requested_uids"]) == {u.uid for u in units}

        # Invariant 2: Subsequent calls strictly for failing units (NO valid re-eval)
        valid_uids = {u.uid for idx, u in enumerate(units) if idx not in failing_indices}
        for retry_call in call_records[1:]:
            assert retry_call["is_single"] is True
            retry_uid = retry_call["requested_uids"][0]
            assert retry_uid not in valid_uids, f"Valid unit {retry_uid} was re-evaluated!"
            retry_idx = int(retry_uid.rsplit("_", 1)[-1])
            assert retry_idx in failing_indices

        # Invariant 3: All units are Status.TRANSLATED
        assert stats.llm_translated == batch_size
        assert stats.failed == 0

        for u in units:
            db_u = db.get_unit(u.uid)
            assert db_u is not None
            assert db_u.status is Status.TRANSLATED, f"Unit {u.uid} has status {db_u.status}"
            assert db_u.target is not None and len(db_u.target) > 0


def test_partial_batch_recovery_permanent_failure(tmp_path: Path) -> None:
    """Empirically verify that units failing retry are marked Status.FAILED without hang."""
    project = _create_test_project(tmp_path, "permanent_fail_proj")
    cfg = Config.from_dict({"batch": {"max_lines": 5, "lookback": 4}})

    with project.open_db() as db:
        for i in range(5):
            db.save_unit(
                Unit(f"u_perm_{i}", UnitKind.DIALOGUE, i, f"Line {i} content", Status.PENDING)
            )

    # Unit 2 fails on batch call AND fails on fallback retry call
    def mock_complete_json(
        sys: str, usr: str, schema: dict[str, Any] | None = None, **kw: Any
    ) -> tuple[dict[str, str], int, int]:
        props = list(schema.get("properties", {}).keys()) if schema else []
        res = {}
        for uid in props:
            if uid == "u_perm_2":
                res[uid] = "Bản dịch lỗi {/b}thứ tự{b}"
            else:
                res[uid] = f"Bản dịch tốt cho {uid}"
        return (res, 50, 25)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        # 4 succeeded, 1 failed permanently
        assert stats.llm_translated == 4
        assert stats.failed == 1

        u_fail = db.get_unit("u_perm_2")
        assert u_fail is not None
        assert u_fail.status is Status.FAILED

        # Other 4 units are cleanly translated
        for i in (0, 1, 3, 4):
            u_ok = db.get_unit(f"u_perm_{i}")
            assert u_ok is not None
            assert u_ok.status is Status.TRANSLATED


def test_partial_batch_missing_key_in_response(tmp_path: Path) -> None:
    """Adversarial test: LLM returns response missing one unit's key completely."""
    project = _create_test_project(tmp_path, "missing_key_proj")
    cfg = Config.from_dict({"batch": {"max_lines": 4, "lookback": 2}})

    with project.open_db() as db:
        for i in range(4):
            db.save_unit(Unit(f"u_miss_{i}", UnitKind.DIALOGUE, i, f"Source {i}", Status.PENDING))

    calls = 0

    def mock_complete_json(
        sys: str, usr: str, schema: dict[str, Any] | None = None, **kw: Any
    ) -> tuple[dict[str, str], int, int]:
        nonlocal calls
        calls += 1
        props = list(schema.get("properties", {}).keys()) if schema else []
        res = {}
        for uid in props:
            if calls == 1 and uid == "u_miss_1":
                continue
            res[uid] = f"Dịch {uid}"
        return (res, 40, 20)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert calls == 2  # 1 batch + 1 retry for missing key
        assert stats.llm_translated == 4
        assert stats.failed == 0

        for i in range(4):
            u = db.get_unit(f"u_miss_{i}")
            assert u is not None and u.status is Status.TRANSLATED


# ==============================================================================
# SECTION 2: BOUNDED LOOKBACK BUFFER STRESS TESTS
# ==============================================================================


def test_bounded_lookback_buffer_100_units(tmp_path: Path) -> None:
    """Feed 100 translated units into pipeline with lookback=4 (capacity 16).

    Assert:
    1. len(lookback_buffer) == 16 at completion.
    2. Oldest items (0-83) were evicted in strict FIFO order.
    3. Retained items in lookback buffer are units 84 through 99 in order.
    4. SQLite db.recent_translated_units(limit=16) returns exact same 16 units.
    """
    project = _create_test_project(tmp_path, "lookback_100_proj")
    # lookback = 4 -> lookback_capacity = 16. max_lines = 10 -> 10 batches of 10
    cfg = Config.from_dict({"batch": {"lookback": 4, "max_lines": 10}})

    total_units = 100
    with project.open_db() as db:
        for i in range(total_units):
            db.save_unit(
                Unit(
                    f"u_lb_{i:03d}",
                    UnitKind.DIALOGUE,
                    i,
                    f"English source line {i}",
                    Status.PENDING,
                    speaker="hero",
                )
            )

    captured_buffers: list[list[LookbackItem]] = []

    def mock_complete_json(
        sys: str, usr: str, schema: dict[str, Any] | None = None, **kw: Any
    ) -> tuple[dict[str, str], int, int]:
        props = list(schema.get("properties", {}).keys()) if schema else []
        res = {uid: f"Tiếng Việt dòng {uid.split('_')[-1]}" for uid in props}
        return (res, len(props) * 15, len(props) * 8)

    original_translate_batch = TranslationPipeline._translate_batch

    def tracked_translate_batch(self: Any, batch: Any, lookback: Any, stats: Any) -> Any:
        res = original_translate_batch(self, batch, lookback, stats)
        captured_buffers.append(list(lookback))
        return res

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
        patch.object(TranslationPipeline, "_translate_batch", tracked_translate_batch),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert stats.llm_translated == 100
        assert stats.batches_count == 10

        # Check final buffer state after all 10 batches
        final_buffer = captured_buffers[-1]
        assert len(final_buffer) == 16, f"Expected 16 items in buffer, got {len(final_buffer)}"

        # Verify FIFO eviction: units 0-83 evicted, 84-99 retained in exact order
        expected_sources = [f"English source line {i}" for i in range(84, 100)]
        expected_targets = [f"Tiếng Việt dòng {i:03d}" for i in range(84, 100)]

        actual_sources = [item.source for item in final_buffer]
        actual_targets = [item.target for item in final_buffer]

        assert actual_sources == expected_sources, "FIFO eviction mismatch"
        assert actual_targets == expected_targets

        # Verify SQLite synchronization: recent_translated_units returns same 16 units
        recent = db.recent_translated_units(limit=16)
        assert len(recent) == 16
        assert [u.uid for u in recent] == [f"u_lb_{i:03d}" for i in range(84, 100)]


# ==============================================================================
# SECTION 3: CROSS-CHUNK CONTINUITY STRESS TESTS
# ==============================================================================


def test_cross_chunk_dialogue_continuity(tmp_path: Path) -> None:
    """Run simulated chunks of 10 lines.

    Verify chunk 2's user prompt contains the preceding dialogue from chunk 1
    retrieved from SQLite.
    """
    project = _create_test_project(tmp_path, "cross_chunk_proj")
    cfg = Config.from_dict({"batch": {"lookback": 3, "max_lines": 10}})

    total_lines = 20
    with project.open_db() as db:
        for i in range(total_lines):
            speaker = "Alice" if i % 2 == 0 else "Bob"
            db.save_unit(
                Unit(
                    f"c_unit_{i:02d}",
                    UnitKind.DIALOGUE,
                    i,
                    f"Dialogue line {i} from character",
                    Status.PENDING,
                    speaker=speaker,
                    label="scene_main",
                )
            )

    # 1. Run Chunk 1 (limit=10)
    with (
        project.open_db() as db,
        patch(
            "renslate.llm.LLMClient.complete_json",
            side_effect=lambda sys, usr, schema, **kw: (
                {uid: f"Lời thoại dịch {uid}" for uid in schema["properties"]},
                100,
                50,
            ),
        ),
    ):
        p1 = TranslationPipeline(project, db, cfg)
        stats1 = p1.run(limit=10)
        assert stats1.llm_translated == 10

    # 2. Run Chunk 2 (limit=10) in a NEW pipeline instance to simulate separate execution
    chunk2_prompts: list[str] = []

    def mock_chunk2_call(
        sys: str, usr: str, schema: dict[str, Any] | None = None, **kw: Any
    ) -> tuple[dict[str, str], int, int]:
        chunk2_prompts.append(usr)
        props = list(schema.get("properties", {}).keys()) if schema else []
        return ({uid: f"Lời thoại dịch {uid}" for uid in props}, 100, 50)

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_chunk2_call),
    ):
        p2 = TranslationPipeline(project, db, cfg)
        stats2 = p2.run(limit=10)
        assert stats2.llm_translated == 10

    # 3. Assert chunk 2's prompt contains preceding dialogue from chunk 1 in lookback
    assert len(chunk2_prompts) == 1
    prompt = chunk2_prompts[0]

    assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC (để nối mạch):" in prompt
    lookback_section = prompt.split("NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC (để nối mạch):")[1]
    lookback_content = lookback_section.split("\n\n")[0]

    # For lookback=3, preceding lines from chunk 1 are lines 7, 8, 9
    expected_lb7 = '[Bob] "Dialogue line 7 from character" => "Lời thoại dịch c_unit_07"'
    expected_lb8 = '[Alice] "Dialogue line 8 from character" => "Lời thoại dịch c_unit_08"'
    expected_lb9 = '[Bob] "Dialogue line 9 from character" => "Lời thoại dịch c_unit_09"'
    assert expected_lb7 in lookback_content
    assert expected_lb8 in lookback_content
    assert expected_lb9 in lookback_content

    # Line 6 should NOT be in the lookback section (as lookback is 3)
    assert "c_unit_06" not in lookback_content


# ==============================================================================
# SECTION 4: LOOKBACK SLICING SAFETY STRESS TESTS
# ==============================================================================


@pytest.mark.parametrize("lookback_val", [0, 1, 5, 100])
@pytest.mark.parametrize("buffer_size", [0, 1, 3, 20])
def test_lookback_slicing_safety_with_deque(lookback_val: int, buffer_size: int) -> None:
    """Verify deque buffer passed to ContextBuilder never raises TypeError.

    Invariants:
    1. deque slicing does not crash with TypeError (index must be int, not slice).
    2. When lookback=0, lookback section is omitted (no negative zero slice bug).
    3. When lookback > 0 and items exist, at most min(lookback, buffer_size) items appear.
    """
    cfg = Config.from_dict({"batch": {"lookback": lookback_val}})
    builder = ContextBuilder(cfg)

    items = deque(
        [
            LookbackItem(f"Speaker_{i}", f"Source line {i}", f"Target line {i}")
            for i in range(buffer_size)
        ]
    )

    batch_units = [
        Unit(f"u_{j}", UnitKind.DIALOGUE, j, f"Batch unit {j}", Status.PENDING) for j in range(3)
    ]

    # Must execute cleanly without exception
    prompt = builder.build_user_prompt(batch_units=batch_units, lookback=items)

    assert isinstance(prompt, str)
    assert "CÁC DÒNG CẦN DỊCH" in prompt

    if lookback_val == 0 or buffer_size == 0:
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" not in prompt
    else:
        assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in prompt
        expected_count = min(lookback_val, buffer_size)
        for i in range(buffer_size - expected_count, buffer_size):
            assert f'"Source line {i}"' in prompt
            assert f'"Target line {i}"' in prompt
        for i in range(0, buffer_size - expected_count):
            assert f'"Source line {i}"' not in prompt


# ==============================================================================
# SECTION 5: ADVERSARIAL LLM ERROR IN FALLBACK RETRY
# ==============================================================================


def test_fallback_retry_llm_network_error(tmp_path: Path) -> None:
    """Adversarial stress: batch fails QA, and single-unit fallback retry raises LLMError."""
    project = _create_test_project(tmp_path, "retry_err_proj")
    cfg = Config.from_dict({"batch": {"max_lines": 3, "lookback": 2}})

    with project.open_db() as db:
        for i in range(3):
            db.save_unit(Unit(f"u_net_{i}", UnitKind.DIALOGUE, i, f"Line {i}", Status.PENDING))

    calls = 0

    def mock_complete_json(
        sys: str, usr: str, schema: dict[str, Any] | None = None, **kw: Any
    ) -> tuple[dict[str, str], int, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                {
                    "u_net_0": "Bản dịch 0",
                    "u_net_1": "Bản dịch lỗi {/b}tag{b}",
                    "u_net_2": "Bản dịch 2",
                },
                60,
                30,
            )
        raise LLMError("Connection reset by peer during fallback retry")

    with (
        project.open_db() as db,
        patch("renslate.llm.LLMClient.complete_json", side_effect=mock_complete_json),
    ):
        pipeline = TranslationPipeline(project, db, cfg)
        stats = pipeline.run()

        assert stats.llm_translated == 2
        assert stats.failed == 1
        assert len(stats.errors) >= 1

        u1 = db.get_unit("u_net_1")
        assert u1 is not None and u1.status is Status.FAILED
        assert any("Connection reset" in err or "LLM error" in err for err in stats.errors)
        assert any("Connection reset" in flag or "LLM error" in flag for flag in u1.qa_flags)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
