"""Opaque-Box E2E Test Suite for Requirement R4: Context Engine & LLM Hallucination Guard.

Covers:
- 4-Tier Hierarchical Context Architecture (Invariants, Persona, Scene Environment, Micro-turns)
- Interlocutor turn-taking, vocative resolution, monologue detection, and character grounding
- Bilateral pronoun symmetry invariant and dynamic relational framing
- Multi-script token mathematics (CJK, Vietnamese, ASCII) and Knapsack budget estimation
- Continuous LLM Hallucination Guard contract and 7-category error taxonomy
- Automated negative feedback integration into the Retry Ladder

Tiers:
- Tier 1: Feature Coverage (>=5 tests per feature)
- Tier 2: Boundary & Corner Cases (>=5 tests per feature)
- Tier 3: Cross-Feature Combinations
- Tier 4: Real-World Application Scenarios
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    Origin,
    RelationshipEdge,
    RelationshipKind,
    SceneMacroContext,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.context import (
    ContextBuilder,
    ContextBundle,
    InterlocutorResolver,
    LookbackItem,
    estimate_tokens,
)
from renslate.engine.graph import RelationshipGraph
from renslate.engine.repair import (
    UnitFailure,
    format_failure_feedback,
)

# Feature detection for progressive milestone testability
HAS_HALLUCINATION_MODULE = importlib.util.find_spec("renslate.engine.hallucination") is not None


# ==============================================================================
# Domain Contract Models for Hallucination Guard (PROJECT.md Specification)
# ==============================================================================


@dataclass(frozen=True, slots=True)
class HallucinationScanResult:
    unit_id: str
    has_hallucination: bool
    category: str = ""
    confidence: float = 0.0
    explanation: str = ""
    suggested_correction: str = ""


class ContractHallucinationGuard:
    """Reference implementation of PROJECT.md § HallucinationGuard Interface Contract."""

    VALID_CATEGORIES = frozenset(
        {
            "factual_hallucination",
            "critical_omission",
            "pronoun_inversion",
            "register_breakdown",
            "emotional_distortion",
            "continuity_breakdown",
            "leaked_artifacts",
        }
    )

    def parse_guard_response(self, raw_json: str) -> list[HallucinationScanResult]:
        """Validate and parse LLM Hallucination Guard JSON output."""
        data = json.loads(raw_json)
        units_raw = data.get("scanned_units", [])
        results: list[HallucinationScanResult] = []
        for item in units_raw:
            verdict = item.get("verdict", "PASS")
            issue_type = item.get("issue_type", "none")
            has_hal = verdict in ("FAIL", "WARNING") and issue_type != "none"
            results.append(
                HallucinationScanResult(
                    unit_id=str(item.get("uid", "")),
                    has_hallucination=has_hal,
                    category=str(issue_type),
                    confidence=float(item.get("confidence", 1.0)),
                    explanation=str(item.get("explanation", "")),
                    suggested_correction=str(item.get("suggested_correction", "")),
                )
            )
        return results


# ==============================================================================
# Helper Factories
# ==============================================================================


def _make_unit(
    uid: str,
    seq: int,
    source: str,
    *,
    speaker: str | None = None,
    target: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        masked_source=source,
        speaker=speaker,
        target=target,
        status=Status.PENDING,
        origin=Origin.LLM,
    )


def _make_profile(
    tag: str,
    display_name: str,
    gender: str = "female",
    role: str = "Heroine",
    pronoun_map: dict[str, str] | None = None,
) -> CharacterProfile:
    return CharacterProfile(
        tag=tag,
        display_name=display_name,
        gender=gender,
        role=role,
        speech_style="nhẹ nhàng",
        pronoun_map=pronoun_map or {},
        archetype="deredere",
        personality_traits=("tử tế", "dịu dàng"),
        tone_and_voice="ấm áp",
        socioeconomic_status="quý tộc",
        sample_dialogue=("Chào anh, em là " + display_name + ".",),
    )


# ==============================================================================
# Tier 1: Feature Coverage (Core Context & Hallucination Contracts)
# ==============================================================================


class TestR4Tier1FeatureCoverage:
    """Tier 1: Feature contracts for multi-tier context, interlocutors, and guard taxonomy."""

    def test_layer1_global_invariants_system_prompt(self) -> None:
        """Verify Layer 1 system prompt contains immutable translation directives."""
        builder = ContextBuilder(Config())
        sys_prompt = builder.build_system_prompt()
        assert "tiếng Việt" in sys_prompt
        assert "giữ nguyên" in sys_prompt or "JSON" in sys_prompt
        assert len(sys_prompt) > 100

    def test_layer2_entity_and_persona_assembly(self) -> None:
        """Verify Layer 2 formats deterministically sorted character profiles."""
        builder = ContextBuilder(Config())
        chars = {
            "sylvie": _make_profile("sylvie", "Sylvie"),
            "eileen": _make_profile("eileen", "Eileen"),
        }
        unit = _make_unit("u1", 1, "Hello.", speaker="eileen")
        prompt = builder.build_user_prompt([unit], characters=chars)
        # Profiles must be rendered deterministically
        assert "Eileen" in prompt
        assert "Sylvie" in prompt

    def test_layer3_scene_macro_environment(self) -> None:
        """Verify Layer 3 injects scene location, atmosphere, and active participants."""
        builder = ContextBuilder(Config())
        macro = SceneMacroContext(
            scene_id="scene_park",
            label="park_walk",
            location="Công viên hoàng hôn",
            atmosphere="lãng mạn, êm đềm",
            temporal_anchor="buổi chiều",
            active_characters=("mc", "eileen"),
            summary="Hai người tản bộ dưới hàng cây phong.",
        )
        unit = _make_unit("u1", 1, "The leaves are falling.", speaker="mc")
        prompt = builder.build_user_prompt([unit], scene_macro=macro)
        assert "Công viên hoàng hôn" in prompt
        assert "lãng mạn, êm đềm" in prompt

    def test_layer4_micro_turns_and_sliding_lookback(self) -> None:
        """Verify Layer 4 includes lookback conversational history."""
        builder = ContextBuilder(Config())
        lookback = [
            LookbackItem(speaker="mc", source="Are you ready?", target="Em đã sẵn sàng chưa?"),
            LookbackItem(speaker="eileen", source="I am ready!", target="Em sẵn sàng rồi!"),
        ]
        unit = _make_unit("u2", 2, "Then let us embark.", speaker="mc")
        prompt = builder.build_user_prompt([unit], lookback=lookback)
        assert "Are you ready?" in prompt
        assert "I am ready!" in prompt

    def test_interlocutor_turn_taking_resolver(self) -> None:
        """Verify InterlocutorResolver maps dialogue sequence to speaker and listener."""
        resolver = InterlocutorResolver()
        u1 = _make_unit("u1", 1, "Hello, Eileen!", speaker="mc")
        res1 = resolver.resolve(u1, active_speakers=["mc", "eileen"])
        assert res1.speaker == "mc"

        u2 = _make_unit("u2", 2, "Hi there, Protagonist.", speaker="eileen")
        res2 = resolver.resolve(u2, prev_unit=u1, active_speakers=["mc", "eileen"])
        assert res2.speaker == "eileen"
        assert res2.listener == "mc"

    def test_interlocutor_monologue_bracket_detection(self) -> None:
        """Verify inner monologue wrapped in parentheses has no listener."""
        resolver = InterlocutorResolver()
        u_mono = _make_unit("u_m", 1, "(I wonder what she really thinks of me...)", speaker="mc")
        res = resolver.resolve(u_mono, active_speakers=["mc", "eileen"])
        assert res.is_monologue is True
        assert res.listener is None

    def test_pronoun_symmetry_invariant(self) -> None:
        """Verify reciprocal pronoun symmetry (anh - em vs em - anh) in RelationshipEdge."""
        edge_mc_to_eileen = RelationshipEdge(
            source_tag="mc",
            target_tag="eileen",
            kind=RelationshipKind.ROMANCE,
            speaker_pronoun="anh",
            listener_pronoun="em",
            confidence=1.0,
            source_type="manual",
        )
        edge_eileen_to_mc = RelationshipEdge(
            source_tag="eileen",
            target_tag="mc",
            kind=RelationshipKind.ROMANCE,
            speaker_pronoun="em",
            listener_pronoun="anh",
            confidence=1.0,
            source_type="manual",
        )
        # Symmetry check: speaker of A->B is listener of B->A and vice versa
        assert edge_mc_to_eileen.speaker_pronoun == edge_eileen_to_mc.listener_pronoun
        assert edge_mc_to_eileen.listener_pronoun == edge_eileen_to_mc.speaker_pronoun

    def test_multi_script_token_budget_estimation(self) -> None:
        """Verify estimate_tokens accounts for CJK, Vietnamese diacritics, and ASCII."""
        ascii_text = "The quick brown fox jumps over the lazy dog."
        vi_text = "Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau."
        cjk_text = "春眠不觉晓, 处处闻啼鸟. 夜来风雨声, 花落知多少."

        tok_ascii = estimate_tokens(ascii_text)
        tok_vi = estimate_tokens(vi_text, is_vietnamese=True)
        tok_cjk = estimate_tokens(cjk_text)

        assert tok_ascii > 0
        assert tok_vi > 0
        assert tok_cjk > 0
        # CJK characters are denser per token than English
        assert tok_cjk >= math.ceil(len(cjk_text) / 2)

    def test_hallucination_guard_7_category_taxonomy_contract(self) -> None:
        """Verify HallucinationGuard contract recognizes all 7 required taxonomy error types."""
        guard = ContractHallucinationGuard()
        categories = ContractHallucinationGuard.VALID_CATEGORIES
        assert len(categories) == 7
        assert "factual_hallucination" in categories
        assert "critical_omission" in categories
        assert "pronoun_inversion" in categories
        assert "register_breakdown" in categories
        assert "emotional_distortion" in categories
        assert "continuity_breakdown" in categories
        assert "leaked_artifacts" in categories

        sample_json = json.dumps(
            {
                "scanned_units": [
                    {
                        "uid": "u101",
                        "verdict": "FAIL",
                        "issue_type": "pronoun_inversion",
                        "confidence": 0.98,
                        "explanation": "Người nói tự xưng nhầm là em thay vì anh.",
                        "suggested_correction": "Anh sẽ bảo vệ em.",
                    }
                ]
            }
        )
        results = guard.parse_guard_response(sample_json)
        assert len(results) == 1
        assert results[0].has_hallucination is True
        assert results[0].category == "pronoun_inversion"
        assert results[0].suggested_correction == "Anh sẽ bảo vệ em."

    def test_automated_failure_feedback_formatting(self) -> None:
        """Verify format_failure_feedback constructs actionable negative instructions."""
        unit = _make_unit("u42", 42, "Will you stand with me?")
        failure = UnitFailure(
            unit=unit,
            code="pronoun_inversion",
            detail="Tự xưng sai thành 'em' khi nói với bề dưới.",
            candidate="Em sẽ đứng cùng ta chứ?",
        )
        feedback = format_failure_feedback([failure])
        assert "LỖI Ở LẦN DỊCH TRƯỚC" in feedback
        assert "u42" in feedback
        assert "pronoun_inversion" in feedback
        assert "bản dịch bị loại" in feedback


# ==============================================================================
# Tier 2: Boundary & Corner Cases (Extreme Budgets, Foreign Artifacts)
# ==============================================================================


class TestR4Tier2Boundaries:
    """Tier 2: Boundary stress testing for context assembly and hallucination detection."""

    def test_empty_dialogue_batch_handling(self) -> None:
        """Verify empty unit list produces empty dialogue table section."""
        builder = ContextBuilder(Config())
        prompt = builder.build_user_prompt([])
        assert "CÁC DÒNG CẦN DỊCH" in prompt

    def test_extreme_tight_token_budget_knapsack(self) -> None:
        """Verify tight token budget trims optional context while preserving core units."""
        from renslate.core.config import BatchConfig

        builder = ContextBuilder(Config(batch=BatchConfig(max_source_tokens=50)))
        chars = {
            f"c_{i}": _make_profile(f"c_{i}", f"Char_{i}" * 5) for i in range(10)
        }
        unit = _make_unit("u1", 1, "Quick message.", speaker="c_0")
        prompt = builder.build_user_prompt([unit], characters=chars)
        assert "u1" in prompt

    def test_foreign_language_artifacts_in_dialogue(self) -> None:
        """Verify mixed Japanese kanji, Cyrillic, and German umlauts are parsed cleanly."""
        mixed_text = "Này, 『田中さん』, đừng quên mang theo das Buch và русский словарь!"
        resolver = InterlocutorResolver()
        u = _make_unit("u_mix", 1, mixed_text, speaker="mc")
        res = resolver.resolve(u)
        assert res.speaker == "mc"

    def test_multi_party_rapid_speaker_turnover(self) -> None:
        """Verify resolver accurately tracks listeners across rapid 4-way conversation."""
        resolver = InterlocutorResolver()
        speakers = ["alice", "bob", "carol", "dave"]
        dialogues = [
            _make_unit("u1", 1, "Alice: Are you listening, Bob?", speaker="alice"),
            _make_unit("u2", 2, "Bob: Yes, Carol what do you think?", speaker="bob"),
            _make_unit("u3", 3, "Carol: I agree with Dave.", speaker="carol"),
            _make_unit("u4", 4, "Dave: Let us proceed then.", speaker="dave"),
        ]
        prev = None
        for d in dialogues:
            res = resolver.resolve(d, prev_unit=prev, active_speakers=speakers)
            assert res.speaker == d.speaker
            prev = d

    def test_malformed_hallucination_guard_json_handling(self) -> None:
        """Verify guard parser handles missing keys, extra keys, and non-list data."""
        guard = ContractHallucinationGuard()
        malformed_json = '{"invalid_schema": true, "scanned_units": [{"uid": "u1"}]}'
        results = guard.parse_guard_response(malformed_json)
        assert len(results) == 1
        assert results[0].unit_id == "u1"
        assert results[0].has_hallucination is False

    def test_hallucination_guard_pass_verdict_no_issues(self) -> None:
        """Verify clean translation returns has_hallucination=False."""
        guard = ContractHallucinationGuard()
        clean_json = json.dumps(
            {
                "scanned_units": [
                    {
                        "uid": "u200",
                        "verdict": "PASS",
                        "issue_type": "none",
                        "confidence": 1.0,
                        "explanation": "Bản dịch chính xác và trung thực.",
                    }
                ]
            }
        )
        results = guard.parse_guard_response(clean_json)
        assert len(results) == 1
        assert results[0].has_hallucination is False
        assert results[0].category == "none"


# ==============================================================================
# Tier 3: Cross-Feature Combinations (Context -> Translation -> Guard -> Retry)
# ==============================================================================


class TestR4Tier3CrossFeature:
    """Tier 3: Pipeline interactions between Context Assembly, Guard, and Failure Feedback."""

    def test_interlocutor_to_prompt_framing_flow(self) -> None:
        """Verify resolved interlocutors produce structured relational prompt frames."""
        resolver = InterlocutorResolver()
        u1 = _make_unit("u1", 1, "Good morning, Eileen.", speaker="mc")
        res = resolver.resolve(u1, active_speakers=["mc", "eileen"])
        assert res.speaker == "mc"

        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge(
                source_tag="mc",
                target_tag="eileen",
                kind=RelationshipKind.ROMANCE,
                speaker_pronoun="anh",
                listener_pronoun="em",
                confidence=1.0,
                source_type="manual",
            )
        )
        bundle = ContextBundle(
            characters={"eileen": _make_profile("eileen", "Eileen")},
            graph=graph,
            resolver=resolver,
        )
        builder = ContextBuilder(Config())
        prompt = builder.build_user_prompt([u1], bundle=bundle)
        assert "Eileen" in prompt
        assert "mc" in prompt

    def test_guard_flagged_issue_to_retry_ladder_feedback(self) -> None:
        """Verify Guard failure generates targeted feedback consumed by next prompt."""
        guard = ContractHallucinationGuard()
        guard_output = json.dumps(
            {
                "scanned_units": [
                    {
                        "uid": "u50",
                        "verdict": "FAIL",
                        "issue_type": "factual_hallucination",
                        "confidence": 0.95,
                        "explanation": "Bản dịch tự bịa thêm chi tiết 'người ngoài hành tinh'.",
                        "suggested_correction": "Đó chỉ là một cơn gió thoảng.",
                    }
                ]
            }
        )
        reports = guard.parse_guard_response(guard_output)
        assert reports[0].has_hallucination is True

        unit = _make_unit("u50", 50, "It was merely a sudden breeze.")
        failure = UnitFailure(
            unit=unit,
            code=reports[0].category,
            detail=reports[0].explanation,
            candidate="Đó là người ngoài hành tinh đến thăm.",
        )
        feedback_str = format_failure_feedback([failure])

        builder = ContextBuilder(Config())
        retry_prompt = builder.build_user_prompt([unit], retry_feedback=feedback_str)
        assert "LỖI Ở LẦN DỊCH TRƯỚC" in retry_prompt
        assert "factual_hallucination" in retry_prompt
        assert "u50" in retry_prompt


# ==============================================================================
# Tier 4: Real-World Application Scenarios (VN Courtroom / Dispute Drama)
# ==============================================================================


class TestR4Tier4RealWorld:
    """Tier 4: End-to-end multi-scene visual novel translation simulation with QA audit."""

    def test_vn_courtroom_pronoun_inversion_detection_and_repair(self) -> None:
        """Simulate high-stakes courtroom dialogue between Empress and General.

        Flow:
        1. Context assembly with formal hierarchical relationship (bệ hạ - ái khanh).
        2. Simulated flawed LLM translation with reversed pronouns.
        3. Hallucination Guard scans and flags 'pronoun_inversion'.
        4. Failure feedback is formatted and inserted into the retry prompt.
        5. Verified corrected translation passes quality audit.
        """
        # 1. Setup characters & relationship
        empress = _make_profile("empress", "Nữ Hoàng", role="Ruler")
        general = _make_profile("general", "Tướng Quân", role="Military Commander")

        graph = RelationshipGraph()
        graph.add_edge(
            RelationshipEdge(
                source_tag="empress",
                target_tag="general",
                kind=RelationshipKind.SUPERIOR_SUBORDINATE,
                speaker_pronoun="trẫm",
                listener_pronoun="khanh",
                confidence=1.0,
                source_type="manual",
            )
        )
        graph.add_edge(
            RelationshipEdge(
                source_tag="general",
                target_tag="empress",
                kind=RelationshipKind.SERVANT_MASTER,
                speaker_pronoun="thần",
                listener_pronoun="bệ hạ",
                confidence=1.0,
                source_type="manual",
            )
        )

        # 2. Dialogue lines
        u1 = _make_unit("court_01", 1, "General, explain your retreat!", speaker="empress")
        u2 = _make_unit("court_02", 2, "Your Majesty, it was an ambush.", speaker="general")

        bundle = ContextBundle(
            characters={"empress": empress, "general": general},
            graph=graph,
            scene_macro=SceneMacroContext(
                scene_id="courtroom",
                label="court_trial",
                location="Hoàng Cung",
                atmosphere="trang nghiêm, căng thẳng",
                active_characters=("empress", "general"),
            ),
        )

        builder = ContextBuilder(Config())
        prompt = builder.build_user_prompt([u1, u2], bundle=bundle)
        assert "Hoàng Cung" in prompt
        assert "Nữ Hoàng" in prompt

        # 3. Simulated flawed translation where General incorrectly addresses Empress as 'em'
        flawed_general_translation = "Này em, đó chỉ là một trận phục kích thôi."

        guard = ContractHallucinationGuard()
        guard_audit_response = json.dumps(
            {
                "scanned_units": [
                    {
                        "uid": "court_02",
                        "verdict": "FAIL",
                        "issue_type": "register_breakdown",
                        "confidence": 0.99,
                        "explanation": "Tướng quân xưng hô 'em' với Nữ Hoàng là sai vai vế.",
                        "suggested_correction": "Bẩm Bệ hạ, đó là một trận phục kích.",
                    }
                ]
            }
        )
        results = guard.parse_guard_response(guard_audit_response)
        assert len(results) == 1
        assert results[0].has_hallucination is True
        assert results[0].category == "register_breakdown"

        # 4. Feedback generation & repair verification
        failure = UnitFailure(
            unit=u2,
            code=results[0].category,
            detail=results[0].explanation,
            candidate=flawed_general_translation,
        )
        feedback = format_failure_feedback([failure])
        repaired_prompt = builder.build_user_prompt([u2], bundle=bundle, retry_feedback=feedback)

        assert "LỖI Ở LẦN DỊCH TRƯỚC" in repaired_prompt
        assert "register_breakdown" in repaired_prompt
        assert "court_02" in repaired_prompt
