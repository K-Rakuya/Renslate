"""Tier 5 White-Box Adversarial Hardening Suite: Engine Context & Hallucination Guard.

Adversarial stress harness targeting:
1. Multi-script token math under extreme unicode combinations (Devanagari, CJK, Emojis, Math, Cyrillic).
2. Knapsack token budgeting under extreme pressure (near-zero/negative budgets, massive lookback, hundreds of profiles).
3. InterlocutorResolver 12-turn deque memory across narrative gaps, scene boundary resets, complex vocatives.
4. Symmetrical and asymmetrical pronoun pair resolution under graph and character profile rules.
5. Continuous LLM Hallucination Guard resilience: 7-category taxonomy, corrupt/truncated LLM outputs.
6. Empirical vulnerability verification: Fabricated LLM UIDs and NaN confidence in audit persistence.
7. End-to-end feedback bridge from HallucinationGuard to RetryLadder and ContextBuilder.
8. Multi-threaded concurrent SQLite qa_audit_log persistence under WAL mode.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

from renslate.core.config import Config, LLMConfig
from renslate.core.db import Database
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import (
    ContextBuilder,
    InterlocutorResolver,
    LookbackItem,
    _analyze_monologue_and_dialogue,
    estimate_tokens,
)
from renslate.engine.graph import RelationshipGraph
from renslate.engine.hallucination import (
    ALL_CATEGORIES,
    CONTEXT_LEAK,
    FOREIGN_ARTIFACT,
    PRONOUN_ERROR,
    SEMANTIC_INVERSION,
    SPEAKER_MISATTRIBUTION,
    TONE_REGISTER_VIOLATION,
    HallucinationGuard,
    HallucinationScanResult,
    format_hallucination_feedback,
    is_valid_category,
    normalize_category,
    persist_hallucination_audit,
    scan_results_to_unit_failures,
)


def _make_unit(
    uid: str,
    seq: int = 1,
    source: str = "Test dialogue line",
    *,
    speaker: str | None = None,
    target: str | None = None,
    label: str | None = "scene_1",
    ref: str | None = "script.rpy:100",
    status: Status = Status.PENDING,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        target=target,
        label=label,
        ref=ref,
        status=status,
    )


# ==============================================================================
# 1. MULTI-SCRIPT TOKEN MATH UNDER EXTREME UNICODE & ASTRAL PLANES
# ==============================================================================


def test_token_math_extreme_multiscript_and_astral_planes() -> None:
    """Stress test multi-script token math with diverse Unicode scripts and extremes."""
    # 1. Boundary empty and minimal strings
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("   ") >= 1

    # 2. CJK characters (approx 1.2 chars/token)
    cjk_text = "桜の花びらが舞い散る春の日に、私たちは再会した。"
    tokens_cjk = estimate_tokens(cjk_text)
    assert tokens_cjk == math.ceil(len(cjk_text) / 1.2)

    # 3. Vietnamese text with full diacritics
    vi_text = "Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau."
    tokens_vi = estimate_tokens(vi_text, is_vietnamese=True)
    assert tokens_vi > 0

    # 4. Devanagari, Math symbols, and Emojis
    devanagari = "ॐ नमः शिवाय शान्तिः शान्तिः शान्तिः"
    assert estimate_tokens(devanagari) > 0

    math_symbols = "∀x ∈ ℝ, ∃y ∈ ℂ: ∫_0^∞ e^{-x^2} dx = √π/2"  # noqa: RUF001
    assert estimate_tokens(math_symbols) > 0

    emojis = "🎉🚀🔥✨🌟🎭🎮👾🎲"
    assert estimate_tokens(emojis) > 0

    # 5. Massive mixed-script string (10,000 characters)
    mixed = (
        "Hello World! "
        + "こんにちは世界 "
        + "Xin chào thế giới! "
        + "नमस्ते दुनिया "
        + "🌍🚀 "
    ) * 200
    assert len(mixed) > 10000
    tokens_mixed = estimate_tokens(mixed)
    assert tokens_mixed > 2000
    assert isinstance(tokens_mixed, int)


# ==============================================================================
# 2. KNAPSACK TOKEN BUDGETING UNDER EXTREME COMPACTION & OVERFLOW
# ==============================================================================


def test_context_builder_extreme_budget_pressure_resilience() -> None:
    """Test ContextBuilder when available token budget is zero or negative."""
    cfg = Config(llm=LLMConfig(n_ctx=150))  # Extremely small context window
    builder = ContextBuilder(cfg)

    units = [_make_unit(f"d:{i}", seq=i, speaker="alice", source=f"Line {i} of dialogue") for i in range(10)]
    chars = {
        f"c{i}": CharacterProfile(
            tag=f"c{i}",
            display_name=f"Char {i}",
            gender="female",
            role="Heroine with a very long elaborate description " * 5,
            pronoun_map={f"c{j}": "em - anh" for j in range(5)},
        )
        for i in range(20)
    }
    lookback = [LookbackItem(speaker="bob", source="Previous source", target="Bản dịch trước") for _ in range(15)]
    macro = SceneMacroContext(
        scene_id="s1",
        label="dungeon_midnight",
        summary="Elaborate scene summary describing the dark dungeon battle in great detail." * 5,
    )

    # Prompt generation must not crash even though budget is heavily negative
    prompt = builder.build_user_prompt(
        batch_units=units,
        characters=chars,
        lookback=lookback,
        scene_macro=macro,
    )

    assert "CÁC DÒNG CẦN DỊCH (bắt buộc dịch đầy đủ đúng id):" in prompt
    for u in units:
        assert u.uid in prompt
    # Optional layers must be pruned due to zero/negative budget
    assert "HỒ SƠ NHÂN VẬT & QUY TẮC XƯNG HÔ:" not in prompt
    assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" not in prompt


def test_context_builder_sliding_lookback_compaction() -> None:
    """Verify that Lookback compaction includes the most recent items up to available budget."""
    cfg = Config(llm=LLMConfig(n_ctx=4096))
    builder = ContextBuilder(cfg)

    # Construct 10 lookback items with sequence identifiers
    lookback = [
        LookbackItem(speaker="Alice", source=f"Turn {i}", target=f"Lượt {i}")
        for i in range(1, 11)
    ]

    # Compact into a tight budget that only fits 2-3 lines
    block, rem = builder._compact_lookback(lookback, available_budget=85)
    assert block is not None
    assert "NGỮ CẢNH CÁC CÂU THOẠI LIỀN TRƯỚC" in block
    # Most recent turn 10 must be included
    assert '"Turn 10"' in block
    # Old turn 1 must have been pruned out
    assert '"Turn 1"' not in block
    assert rem >= 0


# ==============================================================================
# 3. INTERLOCUTOR RESOLVER: LONG NARATION, BOUNDARIES, VOCATIVES & PRONOUNS
# ==============================================================================


def test_interlocutor_resolver_narration_memory_expiry_100_turns() -> None:
    """Verify interlocutor deque memory expires after narrative gap > history_window."""
    resolver = InterlocutorResolver(history_window=12)

    # Turn 1: Alice talks to Bob (vocative)
    u1 = _make_unit("d:1", seq=1, speaker="alice", source="Hey Bob, wake up!")
    r1 = resolver.resolve(u1, active_speakers={"alice", "bob", "charlie"})
    assert r1.listener == "bob"
    assert r1.resolution_method == "vocative"

    # Turns 2 to 120 (119 narration turns with no active dialogue speaker)
    for s in range(2, 121):
        u_narr = _make_unit(f"d:{s}", seq=s, speaker=None, source="The dust settled slowly.")
        resolver.resolve(u_narr, active_speakers={"alice", "bob", "charlie"})

    # Turn 121: Bob speaks without vocative in a 3-character scene
    u121 = _make_unit("d:121", seq=121, speaker="bob", source="Where are we now?")
    r121 = resolver.resolve(u121, active_speakers={"alice", "bob", "charlie"})

    # Alice must NOT be falsely retained because 120 turns have elapsed (> 12)
    assert r121.listener is None
    assert r121.resolution_method == "default"


def test_interlocutor_resolver_cross_narrative_boundary_reset() -> None:
    """Verify that crossing file or label scene boundaries purges dynamic alias memory."""
    resolver = InterlocutorResolver(history_window=12)

    # Scene 1: Prologue, Alice addresses Bob as 'Papa'
    u1 = _make_unit("d:1", seq=1, speaker="alice", source="Papa, please wait!", label="prologue", ref="prologue.rpy:10")
    r1 = resolver.resolve(u1, active_speakers={"alice", "bob"})
    assert r1.listener == "bob"

    # Scene 2: Chapter 1 in a different file, Alice addresses the King as 'Papa'
    u2 = _make_unit("d:2", seq=2, speaker="alice", source="Papa, I have returned.", label="chapter_1", ref="ch1.rpy:5")
    r2 = resolver.resolve(u2, active_speakers={"alice", "king"})

    # In chapter 1, Bob is no longer present; alias was reset across scene boundary
    assert r2.listener == "king"
    assert r2.resolution_method == "vocative"


def test_interlocutor_resolver_complex_stutter_and_brackets() -> None:
    """Verify vocative extraction across Japanese brackets, stage directions and stutters."""
    resolver = InterlocutorResolver()
    name_map = {"liana": "liana", "arthur": "arthur"}

    # 1. Japanese brackets with stage direction and hyphenated stutter
    t1 = "「(Gasping) H-Hey, Liana... are you okay?」"
    is_mono, clean1 = _analyze_monologue_and_dialogue(t1)
    assert not is_mono
    assert clean1 == "H-Hey, Liana... are you okay?"
    tgt1, m1 = resolver._extract_vocative(clean1, name_map, "arthur")
    assert tgt1 == "liana"
    assert m1 == "vocative"

    # 2. Self-address must be rejected (speaker cannot vocatively address self)
    t2 = "Arthur, you fool, get a grip!"
    tgt2, _ = resolver._extract_vocative(t2, name_map, speaker="arthur")
    assert tgt2 is None

    # 3. Off-screen character rejected when allowed_targets specified
    tgt3, _ = resolver._extract_vocative(
        "Arthur, where are you?", name_map, speaker="liana", allowed_targets={"charlie"}
    )
    assert tgt3 is None


def test_pronoun_symmetry_and_asymmetry_resolution() -> None:
    """Verify reciprocal pronoun inversion across RelationshipGraph and CharacterProfile."""
    resolver = InterlocutorResolver()

    # 1. RelationshipGraph direct and reciprocal
    graph = RelationshipGraph()
    graph.add_node("lord")
    graph.add_node("knight")
    # Lord speaks to Knight: ta - ngươi
    graph.add_edge(
        RelationshipEdge(
            source_tag="lord",
            target_tag="knight",
            kind=RelationshipKind.SUPERIOR_SUBORDINATE,
            speaker_pronoun="ta",
            listener_pronoun="ngươi",
        )
    )

    # Direct: Lord -> Knight
    p_direct = resolver.resolve_pronouns("lord", "knight", graph=graph)
    assert p_direct == ("ta", "ngươi")

    # Reciprocal: Knight -> Lord (must invert to: ngươi - ta)
    p_recip = resolver.resolve_pronouns("knight", "lord", graph=graph)
    assert p_recip == ("ngươi", "ta")

    # 2. CharacterProfile pronoun_map reciprocal
    char_map = {
        "alice": CharacterProfile(
            tag="alice",
            display_name="Alice",
            pronoun_map={"bob": "em - anh"},
        ),
        "bob": CharacterProfile(
            tag="bob",
            display_name="Bob",
            pronoun_map={},  # Bob has no rule for Alice
        ),
    }

    # Alice -> Bob
    assert resolver.resolve_pronouns("alice", "bob", char_map=char_map) == ("em", "anh")
    # Bob -> Alice (inferred from Alice's profile)
    assert resolver.resolve_pronouns("bob", "alice", char_map=char_map) == ("anh", "em")


# ==============================================================================
# 4. CONTINUOUS LLM HALLUCINATION GUARD: TAXONOMY, RESILIENCE & PAYLOADS
# ==============================================================================


def test_hallucination_taxonomy_complete_canonical_and_aliases() -> None:
    """Verify all 7 canonical categories and synonym aliases normalize correctly."""
    assert len(ALL_CATEGORIES) == 7
    expected = {
        "SPEAKER_MISATTRIBUTION",
        "SEMANTIC_INVERSION",
        "PRONOUN_ERROR",
        "TONE_REGISTER_VIOLATION",
        "TERMINOLOGY_DRIFT",
        "CONTEXT_LEAK",
        "FOREIGN_ARTIFACT",
    }
    assert expected == ALL_CATEGORIES

    # Test synonym mappings
    assert normalize_category("factual_hallucination") == SEMANTIC_INVERSION
    assert normalize_category("critical_omission") == SEMANTIC_INVERSION
    assert normalize_category("pronoun_inversion") == PRONOUN_ERROR
    assert normalize_category("register_breakdown") == TONE_REGISTER_VIOLATION
    assert normalize_category("emotional_distortion") == TONE_REGISTER_VIOLATION
    assert normalize_category("continuity_breakdown") == SPEAKER_MISATTRIBUTION
    assert normalize_category("leaked_artifacts") == FOREIGN_ARTIFACT
    assert normalize_category("leaked_artifact") == CONTEXT_LEAK
    assert normalize_category("pronoun_drift") == PRONOUN_ERROR

    assert is_valid_category("factual_hallucination")
    assert is_valid_category("SPEAKER_MISATTRIBUTION")
    assert not is_valid_category("random_hallucination_tag")


def test_hallucination_guard_fail_open_under_corrupted_payloads() -> None:
    """Verify that HallucinationGuard fails open under truncated, non-dict, or garbage LLM payloads."""
    guard = HallucinationGuard()
    units = [_make_unit("d:1", source="Hello"), _make_unit("d:2", source="World")]

    # 1. Truncated JSON
    truncated_json = '{"scanned_units": [{"uid": "d:1", "verdict": "FAIL"'
    res_trunc = guard.parse_guard_response(truncated_json, units=units)
    assert len(res_trunc) == 2
    assert all(not r.has_hallucination for r in res_trunc)
    assert all("JSON parse error" in r.explanation for r in res_trunc)

    # 2. Non-dict root (e.g. JSON array)
    res_array = guard.parse_guard_response('[{"uid": "d:1"}]', units=units)
    assert len(res_array) == 2
    assert all(not r.has_hallucination for r in res_array)

    # 3. Missing scanned_units array
    res_missing = guard.parse_guard_response('{"status": "ok"}', units=units)
    assert len(res_missing) == 2
    assert all(not r.has_hallucination for r in res_missing)

    # 4. Partial units in scanned_units (d:1 present, d:2 omitted by LLM)
    partial_json = {
        "scanned_units": [
            {
                "uid": "d:1",
                "verdict": "FAIL",
                "issue_type": "PRONOUN_ERROR",
                "confidence": 0.95,
                "explanation": "Wrong pronoun",
            }
        ]
    }
    res_partial = guard.parse_guard_response(partial_json, units=units)
    assert len(res_partial) == 2
    by_id = {r.unit_id: r for r in res_partial}
    assert by_id["d:1"].has_hallucination is True
    assert by_id["d:1"].category == PRONOUN_ERROR
    # d:2 was omitted, must be passed by default
    assert by_id["d:2"].has_hallucination is False
    assert "omitted" in by_id["d:2"].explanation


# ==============================================================================
# 5. EMPIRICAL VULNERABILITY REPRODUCTION: FABRICATED UIDS & NAN CONFIDENCE
# ==============================================================================


def test_empirical_vulnerability_fabricated_llm_uid_breaks_foreign_key(tmp_path: Path) -> None:
    """Adversarial test: Fabricated LLM UIDs are filtered out by HallucinationGuard

    and database audit persistence gracefully ignores stray foreign key violations.
    """
    db_path = tmp_path / "fk_vuln.db"
    with Database.open(db_path, create=True) as db:
        u = _make_unit("d:real_1", source="Real dialogue")
        db.save_units([u])

        guard = HallucinationGuard()
        # LLM hallucinates an arbitrary UID "d:fabricated_999" not in the batch
        raw_llm_output = {
            "scanned_units": [
                {
                    "uid": "d:fabricated_999",
                    "verdict": "FAIL",
                    "issue_type": "SEMANTIC_INVERSION",
                    "confidence": 0.9,
                    "explanation": "Fabricated unit detected",
                }
            ]
        }
        results = guard.parse_guard_response(raw_llm_output, units=[u])
        assert not any(r.unit_id == "d:fabricated_999" for r in results)
        assert any(r.unit_id == "d:real_1" for r in results)

        # Even if a fabricated result is directly supplied to persist_hallucination_audit,
        # it is caught safely without raising an unhandled IntegrityError
        fabricated_result = HallucinationScanResult(
            unit_id="d:fabricated_999",
            has_hallucination=True,
            category="SEMANTIC_INVERSION",
            confidence=0.9,
        )
        inserted = persist_hallucination_audit(db, [fabricated_result])
        assert inserted == 0


def test_empirical_vulnerability_nan_confidence_violates_not_null(tmp_path: Path) -> None:
    """Adversarial test: NaN confidence from LLM is sanitized to float,

    preventing SQLite NOT NULL constraint failures.
    """
    db_path = tmp_path / "nan_vuln.db"
    with Database.open(db_path, create=True) as db:
        u = _make_unit("d:real_1", source="Real dialogue")
        db.save_units([u])

        guard = HallucinationGuard()
        # LLM outputs "NaN" for confidence string
        raw_llm_output = {
            "scanned_units": [
                {
                    "uid": "d:real_1",
                    "verdict": "FAIL",
                    "issue_type": "SEMANTIC_INVERSION",
                    "confidence": "NaN",
                    "explanation": "NaN confidence test",
                }
            ]
        }
        results = guard.parse_guard_response(raw_llm_output, units=[u])
        assert not math.isnan(results[0].confidence)
        assert results[0].confidence == 1.0

        # Persistence succeeds cleanly without violating NOT NULL constraint
        inserted = persist_hallucination_audit(db, results)
        assert inserted == 1


# ==============================================================================
# 6. HALLUCINATION FEEDBACK BRIDGE TO RETRY LADDER
# ==============================================================================


def test_hallucination_feedback_bridge_to_retry_prompt() -> None:
    """Verify detections meeting threshold convert to UnitFailures and format cleanly."""
    u1 = _make_unit("d:10", source="Do you want to die?", target="Em muốn chết sao?")
    u2 = _make_unit("d:11", source="I will protect you.", target="Anh sẽ bảo vệ em.")
    units_by_id = {"d:10": u1, "d:11": u2}

    results = [
        HallucinationScanResult(
            unit_id="d:10",
            has_hallucination=True,
            category="PRONOUN_ERROR",
            confidence=0.92,
            explanation="Tự xưng 'em' làm chủ ngữ cho câu hỏi hướng về đối phương",
            suggested_correction="Anh muốn chết sao?",
        ),
        HallucinationScanResult(
            unit_id="d:11",
            has_hallucination=False,
            category="",
            confidence=0.98,
        ),
        HallucinationScanResult(
            unit_id="d:10",
            has_hallucination=True,
            category="TONE_REGISTER_VIOLATION",
            confidence=0.50,  # Below threshold 0.70
            explanation="Low confidence finding",
        ),
    ]

    # Convert to UnitFailures with min_confidence=0.70
    failures = scan_results_to_unit_failures(results, units_by_id, min_confidence=0.70)
    assert len(failures) == 1
    assert failures[0].unit.uid == "d:10"
    assert failures[0].code == "PRONOUN_ERROR"

    feedback = format_hallucination_feedback(failures)
    assert "LỖI Ở LẦN DỊCH TRƯỚC" in feedback
    assert "- d:10: PRONOUN_ERROR" in feedback
    assert "Gợi ý: Anh muốn chết sao?" in feedback
    assert 'bản dịch bị loại: "Em muốn chết sao?"' in feedback


# ==============================================================================
# 7. MULTI-THREADED CONCURRENT AUDIT LOGGING
# ==============================================================================


def test_concurrent_audit_logging_wal_resilience(tmp_path: Path) -> None:
    """Stress test 10 concurrent threads persisting audit records under SQLite WAL mode."""
    db_path = tmp_path / "concurrent_audit.db"
    with Database.open(db_path, create=True) as db:
        units = [_make_unit(f"d:{i}", seq=i, source=f"Line {i}") for i in range(50)]
        db.save_units(units)

    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        try:
            with Database.open(db_path, create=False) as tdb:
                for rep in range(5):
                    results = [
                        HallucinationScanResult(
                            unit_id=f"d:{worker_id * 5 + j}",
                            has_hallucination=(j % 2 == 0),
                            category=PRONOUN_ERROR if (j % 2 == 0) else "",
                            confidence=0.88,
                            explanation=f"Worker {worker_id} rep {rep}",
                        )
                        for j in range(5)
                    ]
                    persist_hallucination_audit(tdb, results)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent logging encountered errors: {errors}"

    # Verify all records persisted correctly
    with Database.open(db_path, create=False) as db:
        logs = db.get_qa_audit_logs("d:0")
        assert len(logs) == 5
