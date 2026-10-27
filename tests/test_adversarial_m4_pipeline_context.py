"""Adversarial stress harness & empirical verification for Milestone 4 TranslationPipeline.

Probing:
1. Multi-party dialogue (3+ characters) verifying that interlocutors are accurately resolved
   and reciprocal pronouns are matched to the specific active interlocutor.
2. Alternating turns, leading/trailing/medial vocatives, and inner thoughts mixed in a batch.
3. Rapid scene transitions: ensure self.resolver.reset() clears scene-specific aliases.
4. Auto-hydration edge cases: corrupted JSON on disk, missing tables in SQLite, empty graphs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

from renslate.core.config import BatchConfig, Config
from renslate.core.models import (
    CharacterProfile,
    RelationshipEdge,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.core.project import Project
from renslate.engine.graph import RelationshipGraph
from renslate.engine.translate import TranslationPipeline


def _setup_test_project(tmp_path: Path, name: str = "m4_adv_proj") -> Project:
    root = tmp_path / name
    (root / "game").mkdir(parents=True, exist_ok=True)
    return Project.create(root, target_lang="vietnamese")


def _make_unit(
    uid: str,
    source: str,
    speaker: str | None = None,
    seq: int = 1,
    label: str | None = None,
) -> Unit:
    return Unit(
        uid=uid,
        kind=UnitKind.DIALOGUE,
        seq=seq,
        source=source,
        status=Status.PENDING,
        speaker=speaker,
        label=label,
    )


# ==============================================================================
# 1. Multi-Party Dialogue & Specific Active Interlocutor Reciprocal Pronouns
# ==============================================================================


class TestAdversarialMultiPartyDialogueAndReciprocalPronouns:
    """Stress tests verifying dynamic interlocutor resolution in 3+ party scenes."""

    def test_tripartite_dialogue_distinct_relational_frames(self, tmp_path: Path) -> None:
        """Verify that in a 3+ character dialogue batch, each line receives the exact

        relational frame and reciprocal pronouns matching its specific active interlocutor.
        """
        project = _setup_test_project(tmp_path, "tripartite_prompt")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male", role="Protagonist"))
            db.save_character(CharacterProfile("eileen", "Eileen", "female", role="Heroine"))
            db.save_character(CharacterProfile("vane", "Lord Vane", "male", role="Elder"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    "ren",
                    "eileen",
                    RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "eileen",
                    "ren",
                    RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "ren",
                    "vane",
                    RelationshipKind.FAMILY_ELDER,
                    speaker_pronoun="con",
                    listener_pronoun="cha",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "vane",
                    "ren",
                    RelationshipKind.FAMILY_ELDER,
                    speaker_pronoun="ta",
                    listener_pronoun="con",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "vane",
                    "eileen",
                    RelationshipKind.SUPERIOR_SUBORDINATE,
                    speaker_pronoun="ta",
                    listener_pronoun="tiểu thư",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "eileen",
                    "vane",
                    RelationshipKind.SUPERIOR_SUBORDINATE,
                    speaker_pronoun="con",
                    listener_pronoun="đại nhân",
                )
            )

            units = [
                _make_unit("d:1", "Eileen, are you ready?", speaker="Ren", seq=1),
                _make_unit("d:2", "I am ready, Ren.", speaker="Eileen", seq=2),
                _make_unit("d:3", "Do not overdo it, Eileen.", speaker="vane", seq=3),
                _make_unit("d:4", "Lord Vane, I am ready.", speaker="Eileen", seq=4),
                _make_unit("d:5", "Ren, what of your duty?", speaker="vane", seq=5),
                _make_unit("d:6", "Father, all is done.", speaker="Ren", seq=6),
            ]
            for u in units:
                db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=graph)

            captured_prompts: list[str] = []

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                return (
                    {
                        "d:1": "Eileen, em đã sẵn sàng chưa?",
                        "d:2": "Em đã sẵn sàng rồi, Ren.",
                        "d:3": "Đừng quá tự phụ, tiểu thư Eileen.",
                        "d:4": "Thưa đại nhân Vane, con đã sẵn sàng.",
                        "d:5": "Ren, nhiệm vụ của con thế nào rồi?",
                        "d:6": "Thưa cha, con đã hoàn thành tất cả rồi.",
                    },
                    200,
                    100,
                )

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 6
                assert stats.failed == 0

            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]

            # 1. Ren -> Eileen: ROMANCE (Ren xưng 'anh', gọi 'em')
            assert (
                "[Người nói: Ren -> Người nghe: Eileen | "
                "Quan hệ: ROMANCE (Ren xưng 'anh', gọi 'em')"
                in prompt
            )
            # 2. Eileen -> Ren: ROMANCE (Eileen xưng 'em', gọi 'anh')
            assert (
                "[Người nói: Eileen -> Người nghe: Ren | "
                "Quan hệ: ROMANCE (Eileen xưng 'em', gọi 'anh')]"
                in prompt
            )
            # 3. Vane -> Eileen: SUPERIOR_SUBORDINATE (Lord Vane xưng 'ta', gọi 'tiểu thư')
            assert (
                "[Người nói: Lord Vane -> Người nghe: Eileen | "
                "Quan hệ: SUPERIOR_SUBORDINATE (Lord Vane xưng 'ta', gọi 'tiểu thư')]"
                in prompt
            )
            # 4. Eileen -> Vane: SUPERIOR_SUBORDINATE (Eileen xưng 'con', gọi 'đại nhân')
            assert (
                "[Người nói: Eileen -> Người nghe: Lord Vane | "
                "Quan hệ: SUPERIOR_SUBORDINATE (Eileen xưng 'con', gọi 'đại nhân')]"
                in prompt
            )
            # 5. Vane -> Ren: FAMILY_ELDER (Lord Vane xưng 'ta', gọi 'con')
            assert (
                "[Người nói: Lord Vane -> Người nghe: Ren | "
                "Quan hệ: FAMILY_ELDER (Lord Vane xưng 'ta', gọi 'con')"
                in prompt
            )
            # 6. Ren -> Vane: FAMILY_ELDER (Ren xưng 'con', gọi 'cha')
            assert (
                "[Người nói: Ren -> Người nghe: Lord Vane | "
                "Quan hệ: FAMILY_ELDER (Ren xưng 'con', gọi 'cha')]"
                in prompt
            )

    def test_multi_party_pronoun_inversion_per_active_interlocutor(self, tmp_path: Path) -> None:
        """Verify that in a 3-party batch, pronoun inversion QA checks enforce the specific

        active interlocutor's rules rather than leaking rules from other speakers in the batch.
        """
        project = _setup_test_project(tmp_path, "tripartite_pronoun_inversion")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            db.save_character(CharacterProfile("vane", "Lord Vane", "male"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    "ren",
                    "eileen",
                    RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "eileen",
                    "ren",
                    RelationshipKind.ROMANCE,
                    speaker_pronoun="em",
                    listener_pronoun="anh",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "ren",
                    "vane",
                    RelationshipKind.FAMILY_ELDER,
                    speaker_pronoun="con",
                    listener_pronoun="cha",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "vane",
                    "ren",
                    RelationshipKind.FAMILY_ELDER,
                    speaker_pronoun="ta",
                    listener_pronoun="con",
                )
            )

            u1 = _make_unit("d:1", "Eileen, do you understand?", speaker="Ren", seq=1)
            u2 = _make_unit("d:2", "I understand, Ren.", speaker="Eileen", seq=2)
            u3 = _make_unit("d:3", "Father, do you understand?", speaker="Ren", seq=3)
            u4 = _make_unit("d:4", "Watch your tone, Ren.", speaker="vane", seq=4)
            db.save_unit(u1)
            db.save_unit(u2)
            db.save_unit(u3)
            db.save_unit(u4)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=graph)

            mock_responses = {
                "d:1": "Eileen, em có hiểu không?",
                "d:2": "Em hiểu rồi, Ren.",
                "d:3": "Cha, con có hiểu không?",  # Inverted! Calling Father 'con' instead of 'cha'
                "d:4": "Hãy cẩn thận lời nói của con, Ren.",
            }

            with patch.object(
                pipeline.llm, "complete_json", return_value=(mock_responses, 200, 100)
            ):
                stats = pipeline.run()

            # Unit 1, 2, 4 passed QA and committed
            assert db.get_unit("d:1").status == Status.TRANSLATED  # type: ignore[union-attr]
            assert db.get_unit("d:2").status == Status.TRANSLATED  # type: ignore[union-attr]
            assert db.get_unit("d:4").status == Status.TRANSLATED  # type: ignore[union-attr]

            # Unit 3 failed QA due to pronoun inversion with Father and was quarantined
            saved3 = db.get_unit("d:3")
            assert saved3 is not None
            assert saved3.status == Status.FAILED
            assert stats.failed == 1
            assert stats.llm_translated == 3

    def test_quadripartite_cross_talk_with_unregistered_edge(self, tmp_path: Path) -> None:
        """Verify that when a 4th character without an established relationship edge joins,

        the pipeline gracefully formats the prompt frame without crashing and translates safely.
        """
        project = _setup_test_project(tmp_path, "quadripartite")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            db.save_character(CharacterProfile("eileen", "Eileen", "female"))
            db.save_character(CharacterProfile("vane", "Lord Vane", "male"))
            db.save_character(CharacterProfile("kael", "Kael", "male", role="Guard"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    "ren",
                    "eileen",
                    RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                )
            )

            u1 = _make_unit("d:k1", "Ren, halt where you are!", speaker="kael", seq=1)
            u2 = _make_unit("d:k2", "Kael, step aside.", speaker="Ren", seq=2)
            db.save_unit(u1)
            db.save_unit(u2)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=graph)

            captured_prompts: list[str] = []

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                return (
                    {
                        "d:k1": "Ren, đứng lại đó!",
                        "d:k2": "Kael, tránh ra.",
                    },
                    80,
                    40,
                )

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 2

            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]
            assert "[Người nói: Kael -> Người nghe: Ren" in prompt
            assert "[Người nói: Ren -> Người nghe: Kael] Kael, step aside." in prompt


# ==============================================================================
# 2. Alternating Turns, Vocative Placements & Inner Thoughts Mixed in a Batch
# ==============================================================================


class TestAdversarialMixedTurnTypesAndInnerThoughts:
    """Stress tests probing complex batches with mixed turn types, vocatives, and thoughts."""

    def test_complex_mixed_turn_batch_assembly_and_peeling(self, tmp_path: Path) -> None:
        """Verify prompt assembly and multi-pass frame peeling across:

        - Leading, trailing, medial vocatives
        - Alternating turns (turn-taking adjacency)
        - Parenthesized stage direction with spoken dialogue
        - True inner thoughts and narrator tags
        """
        project = _setup_test_project(tmp_path, "mixed_batch")
        with project.open_db() as db:
            db.save_character(CharacterProfile("alice", "Alice", "female"))
            db.save_character(CharacterProfile("bob", "Bob", "male"))

            graph = RelationshipGraph()
            graph.add_edge(
                RelationshipEdge(
                    "alice",
                    "bob",
                    RelationshipKind.FRIENDSHIP_CLOSE,
                    speaker_pronoun="tớ",
                    listener_pronoun="cậu",
                )
            )
            graph.add_edge(
                RelationshipEdge(
                    "bob",
                    "alice",
                    RelationshipKind.FRIENDSHIP_CLOSE,
                    speaker_pronoun="tớ",
                    listener_pronoun="cậu",
                )
            )

            units = [
                _make_unit("d:1", "Alice, did you see the flare?", speaker="Bob", seq=1),
                _make_unit("d:2", "I saw it clearly, Bob.", speaker="Alice", seq=2),
                _make_unit("d:3", "Wait, Alice, look over there!", speaker="Bob", seq=3),
                _make_unit("d:4", "What creature is that?", speaker="Alice", seq=4),
                _make_unit("d:5", "(Draws sword) Bob, behind me!", speaker="Alice", seq=5),
                _make_unit("d:6", "(Can we survive this...)", speaker="Bob", seq=6),
                _make_unit("d:7", "『Find the weak point.』", speaker="Alice", seq=7),
                _make_unit("d:8", "A chill ran down my spine.", speaker="th", seq=8),
                _make_unit("d:9", "The beast let out a roar.", speaker="narrator", seq=9),
                _make_unit("d:10", "Alice, strike now!", speaker="Bob", seq=10),
            ]
            for u in units:
                db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=graph)

            captured_prompts: list[str] = []

            mock_responses = {
                "d:1": (
                    '[Người nói: Bob -> Người nghe: Alice | Quan hệ: FRIENDSHIP_CLOSE] '
                    '"Alice, cậu có nhìn thấy pháo hiệu không?"'
                ),
                "d:2": "Tớ nhìn thấy rõ ràng, Bob.",
                "d:3": "Chờ đã, Alice, nhìn đằng kia kìa!",
                "d:4": "Đó là sinh vật gì vậy?",
                "d:5": "(Rút kiếm) Bob, đứng ra sau tớ!",
                "d:6": "[Độc thoại/Nội tâm: Bob] (Liệu chúng ta có sống sót...)",
                "d:7": "『Hãy tìm điểm yếu.』",
                "d:8": "Một cơn ớn lạnh chạy dọc sống lưng tôi.",
                "d:9": "Con quái thú gầm lên một tiếng.",
                "d:10": "Alice, tấn công ngay!",
            }

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                return (mock_responses, 300, 150)

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 10
                assert stats.failed == 0

            assert len(captured_prompts) == 1
            prompt = captured_prompts[0]

            assert "[Người nói: Bob -> Người nghe: Alice" in prompt
            assert "[Người nói: Alice -> Người nghe: Bob" in prompt
            assert (
                "[Người nói: Alice -> Người nghe: Bob | "
                "Quan hệ: FRIENDSHIP_CLOSE (Alice xưng 'tớ', gọi 'cậu')"
                in prompt
            )
            assert "(Draws sword) Bob, behind me!" in prompt
            assert "[Độc thoại/Nội tâm: Bob" in prompt
            assert "(Can we survive this...)" in prompt
            assert "[Độc thoại/Nội tâm: Alice] 『Find the weak point.』" in prompt
            assert "[Độc thoại/Nội tâm: th] A chill ran down my spine." in prompt
            assert "[Độc thoại/Nội tâm: narrator] The beast let out a roar." in prompt

            for uid, expected_snippet in [
                ("d:1", "Alice, cậu có nhìn thấy pháo hiệu không?"),
                ("d:6", "(Liệu chúng ta có sống sót...)"),
            ]:
                saved = db.get_unit(uid)
                assert saved is not None
                assert saved.status == Status.TRANSLATED
                assert saved.target == expected_snippet
                assert "[Người nói:" not in (saved.target or "")
                assert "[Độc thoại/Nội tâm:" not in (saved.target or "")

    def test_monologue_does_not_poison_subsequent_turn_taking(self, tmp_path: Path) -> None:
        """Verify that an inner thought sandwiched between dialogue turns does not disrupt

        interlocutor resolution or cause erroneous self-listening.
        """
        project = _setup_test_project(tmp_path, "monologue_sandwich")
        with project.open_db() as db:
            db.save_character(CharacterProfile("alice", "Alice", "female"))
            db.save_character(CharacterProfile("bob", "Bob", "male"))

            u1 = _make_unit("d:m1", "Bob, where is the key?", speaker="Alice", seq=1)
            u2 = _make_unit("d:m2", "(Should I really tell her?)", speaker="Bob", seq=2)
            u3 = _make_unit("d:m3", "It is under the board.", speaker="Bob", seq=3)
            db.save_unit(u1)
            db.save_unit(u2)
            db.save_unit(u3)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            captured_prompts: list[str] = []

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                return (
                    {
                        "d:m1": "Bob, chìa khóa ở đâu vậy?",
                        "d:m2": "(Mình có thực sự nên nói cho cô ấy?)",
                        "d:m3": "Nó ở dưới tấm ván đấy.",
                    },
                    120,
                    60,
                )

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 3

            prompt = captured_prompts[0]
            assert "[Người nói: Alice -> Người nghe: Bob" in prompt
            assert "[Độc thoại/Nội tâm: Bob]" in prompt
            assert "[Người nói: Bob -> Người nghe: Bob" not in prompt


# ==============================================================================
# 3. Rapid Scene Transitions & Dynamic Alias Isolation
# ==============================================================================


class TestAdversarialSceneTransitionsAndAliasIsolation:
    """Stress tests verifying resolver resets clear scene-specific aliases on transitions."""

    def test_scene_transition_clears_kinship_aliases(self, tmp_path: Path) -> None:
        """Verify that a kinship title alias learned in Scene 1 does not bleed into Scene 2,

        allowing the title to resolve to a completely different character in Scene 2.
        """
        project = _setup_test_project(tmp_path, "scene_alias_isolation")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            db.save_character(CharacterProfile("vane", "Lord Vane", "male"))
            db.save_character(CharacterProfile("jerome", "Jerome", "male"))

            u1 = _make_unit(
                "d:s1_1", "Where were you, Ren?", speaker="vane", seq=1, label="scene_castle"
            )
            u2 = _make_unit(
                "d:s1_2", "I am here, Father.", speaker="Ren", seq=2, label="scene_castle"
            )

            u3 = _make_unit(
                "d:s2_1", "Welcome, my son.", speaker="jerome", seq=3, label="scene_church"
            )
            u4 = _make_unit(
                "d:s2_2", "I need guidance, Father.", speaker="Ren", seq=4, label="scene_church"
            )

            db.save_unit(u1)
            db.save_unit(u2)
            db.save_unit(u3)
            db.save_unit(u4)

            cfg = Config(batch=BatchConfig(max_lines=2))
            pipeline = TranslationPipeline(project, db, cfg)

            captured_prompts: list[str] = []

            def mock_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                captured_prompts.append(user_prompt)
                res = {}
                for line in user_prompt.split("\n"):
                    if line.startswith(("d:s1_", "d:s2_")):
                        uid = line.split(" | ")[0].strip()
                        res[uid] = f"Bản dịch cho {uid}"
                return (res, 50, 20)

            with patch.object(pipeline.llm, "complete_json", side_effect=mock_complete):
                stats = pipeline.run()
                assert stats.llm_translated == 4

            assert len(captured_prompts) == 2
            prompt1, prompt2 = captured_prompts[0], captured_prompts[1]

            assert "[Người nói: Ren -> Người nghe: Lord Vane] I am here, Father." in prompt1
            assert "[Người nói: Ren -> Người nghe: Lord Vane]" not in prompt2
            assert "[Người nói: Ren -> Người nghe: Jerome] I need guidance, Father." in prompt2

    def test_rapid_multi_scene_churn(self, tmp_path: Path) -> None:
        """Verify rapid alternating scene labels across 5 batches properly reset the resolver

        and update _current_scene_label on each transition.
        """
        project = _setup_test_project(tmp_path, "rapid_churn")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))

            labels = [
                "scene_prologue",
                "scene_forest",
                "scene_cave",
                "scene_forest",
                None,
                "scene_epilogue",
            ]

            for i, lbl in enumerate(labels):
                db.save_unit(
                    _make_unit(
                        f"d:churn_{i}",
                        f"Line {i} in scene {lbl}",
                        speaker="Ren",
                        seq=i + 1,
                        label=lbl,
                    )
                )

            cfg = Config(batch=BatchConfig(max_lines=1))
            pipeline = TranslationPipeline(project, db, cfg)

            reset_labels_seen: list[str | None] = []
            orig_reset = pipeline.resolver.reset

            def tracking_reset() -> None:
                orig_reset()
                reset_labels_seen.append(pipeline._current_scene_label)

            def mock_churn_complete(
                sys_prompt: str, user_prompt: str, **kwargs: Any
            ) -> tuple[dict[str, str], int, int]:
                uid = user_prompt.split(" | ")[0].split("\n")[-1].strip()
                return ({uid: "Bản dịch."}, 30, 10)

            with (
                patch.object(pipeline.resolver, "reset", side_effect=tracking_reset),
                patch.object(pipeline.llm, "complete_json", side_effect=mock_churn_complete),
            ):
                stats = pipeline.run()
                assert stats.llm_translated == 6

            assert len(reset_labels_seen) == 6
            assert pipeline._current_scene_label == "scene_epilogue"


# ==============================================================================
# 4. Auto-Hydration Edge Cases: Corrupted JSON, Missing Tables, Empty Graphs
# ==============================================================================


class TestAdversarialAutoHydrationAndFallbackResilience:
    """Stress tests probing auto-hydration edge cases and graceful degradation."""

    def test_auto_hydrate_corrupted_json_falls_back_to_sqlite(self, tmp_path: Path) -> None:
        """Verify that when relationships.json is corrupted syntax, the pipeline gracefully

        catches the JSON error and falls back to loading valid relationships from SQLite.
        """
        project = _setup_test_project(tmp_path, "corrupted_json_fallback")
        rel_file = project.work_dir / "relationships.json"
        rel_file.write_text("{ this is corrupted [ json !!!", encoding="utf-8")

        with project.open_db() as db:
            db_graph = RelationshipGraph()
            db_graph.add_edge(
                RelationshipEdge(
                    "hero",
                    "heroine",
                    RelationshipKind.ROMANCE,
                    speaker_pronoun="anh",
                    listener_pronoun="em",
                )
            )
            db_graph.save_to_db(db)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            assert pipeline.graph is not None
            assert pipeline.graph.has_edge("hero", "heroine")
            edge = pipeline.graph.get_edge("hero", "heroine")
            assert edge is not None
            assert edge.kind == RelationshipKind.ROMANCE
            assert edge.speaker_pronoun == "anh"

    def test_auto_hydrate_corrupted_json_and_missing_sqlite_table(self, tmp_path: Path) -> None:
        """Verify that when BOTH relationships.json is corrupted AND the SQLite table is missing,

        the pipeline catches all exceptions and safely initializes an empty RelationshipGraph.
        """
        project = _setup_test_project(tmp_path, "corrupted_json_and_missing_table")
        rel_file = project.work_dir / "relationships.json"
        rel_file.write_text("<<<INVALID BINARY GARBAGE>>>", encoding="utf-8")

        with project.open_db() as db:
            db._connection.execute("DROP TABLE IF EXISTS character_relationships")

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)

            assert isinstance(pipeline.graph, RelationshipGraph)
            assert pipeline.graph.edge_count == 0

    def test_auto_hydrate_missing_json_and_missing_sqlite_table(self, tmp_path: Path) -> None:
        """Verify clean fallback to empty graph when no JSON exists and SQLite table is dropped."""
        project = _setup_test_project(tmp_path, "missing_all")
        with project.open_db() as db:
            db._connection.execute("DROP TABLE IF EXISTS character_relationships")

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)
            assert isinstance(pipeline.graph, RelationshipGraph)
            assert pipeline.graph.edge_count == 0

    def test_auto_hydrate_zero_byte_json_file(self, tmp_path: Path) -> None:
        """Verify that a 0-byte relationships.json does not raise JSONDecodeError."""
        project = _setup_test_project(tmp_path, "zero_byte_json")
        rel_file = project.work_dir / "relationships.json"
        rel_file.write_text("", encoding="utf-8")

        with project.open_db() as db:
            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg)
            assert isinstance(pipeline.graph, RelationshipGraph)
            assert pipeline.graph.edge_count == 0

    def test_auto_hydrate_malformed_json_schema_types(self, tmp_path: Path) -> None:
        """Verify that JSON containing unexpected types degrades gracefully to empty graph."""
        for i, bad_payload in enumerate(
            [
                json.dumps([1, 2, 3]),
                json.dumps("a plain string"),
                json.dumps({"edges": "not_a_list"}),
                json.dumps({"edges": [{"source_tag": 123}]}),
            ]
        ):
            project = _setup_test_project(tmp_path, f"bad_schema_{i}")
            rel_file = project.work_dir / "relationships.json"
            rel_file.write_text(bad_payload, encoding="utf-8")

            with project.open_db() as db:
                cfg = Config()
                pipeline = TranslationPipeline(project, db, cfg)
                assert isinstance(pipeline.graph, RelationshipGraph)

    def test_empty_graph_pipeline_execution(self, tmp_path: Path) -> None:
        """Verify TranslationPipeline runs end-to-end without errors when graph is empty."""
        project = _setup_test_project(tmp_path, "empty_graph_run")
        with project.open_db() as db:
            db.save_character(CharacterProfile("ren", "Ren", "male"))
            u = _make_unit("d:empty_1", "The sky is blue.", speaker="Ren")
            db.save_unit(u)

            cfg = Config()
            pipeline = TranslationPipeline(project, db, cfg, graph=RelationshipGraph())

            with patch.object(
                pipeline.llm,
                "complete_json",
                return_value=({"d:empty_1": "Bầu trời màu xanh."}, 40, 15),
            ):
                stats = pipeline.run()
                assert stats.llm_translated == 1
                assert stats.failed == 0

            saved = db.get_unit("d:empty_1")
            assert saved is not None
            assert saved.status == Status.TRANSLATED
            assert saved.target == "Bầu trời màu xanh."

    def test_sqlite_operational_error_during_from_db(self, tmp_path: Path) -> None:
        """Verify that an unexpected database operational error during from_db is safely caught."""
        project = _setup_test_project(tmp_path, "db_error_resilience")
        with project.open_db() as db:
            cfg = Config()

            with patch.object(
                RelationshipGraph,
                "from_db",
                side_effect=sqlite3.DatabaseError("database disk image is malformed"),
            ):
                pipeline = TranslationPipeline(project, db, cfg)
                assert isinstance(pipeline.graph, RelationshipGraph)
                assert pipeline.graph.edge_count == 0
