"""Tests for DiscourseAttentionPolicy and MobRoleClassifier.

Verifies:
1. Strict isolated vocative detection vs 3rd-person reference exclusion.
2. Group dialogue disambiguation (>2 characters: Alice, Bob, Keiko).
3. Stage presence filtering of absent characters.
4. Question-Answer adjacency turn fallback.
5. Mob / Anonymous NPC social role taxonomy and prompt formatting.
"""

from renslate.core.models import CharacterProfile, Status, Unit, UnitKind
from renslate.domain.models import SpriteState, StageSnapshot
from renslate.domain.policies.attention import DiscourseAttentionPolicy
from renslate.domain.policies.mob_roles import MobRoleCategory, MobRoleClassifier
from renslate.engine.context.prompt_frame import format_rich_dialogue_line
from renslate.engine.context.resolver import InterlocutorResolver


def test_third_person_reference_detection() -> None:
    # 3rd-person references should be detected
    assert DiscourseAttentionPolicy.is_third_person_reference("Did you talk to Keiko?", "Keiko")
    assert DiscourseAttentionPolicy.is_third_person_reference("Did you see Keiko?", "Keiko")
    assert DiscourseAttentionPolicy.is_third_person_reference("I went with Keiko yesterday.", "Keiko")
    assert DiscourseAttentionPolicy.is_third_person_reference("Keiko said she was tired.", "Keiko")
    assert DiscourseAttentionPolicy.is_third_person_reference("That is Keiko's book.", "Keiko")
    assert DiscourseAttentionPolicy.is_third_person_reference("I think that Keiko knows the truth.", "Keiko")

    # Direct addresses should NOT be detected as 3rd person
    assert not DiscourseAttentionPolicy.is_third_person_reference("Keiko, what do you think?", "Keiko")
    assert not DiscourseAttentionPolicy.is_third_person_reference("What do you think, Keiko?", "Keiko")
    assert not DiscourseAttentionPolicy.is_third_person_reference("Hey Keiko, listen to me.", "Keiko")


def test_strict_isolated_vocative_detection() -> None:
    # Isolated vocatives
    assert DiscourseAttentionPolicy.is_strict_isolated_vocative("Keiko, what do you think?", "Keiko")
    assert DiscourseAttentionPolicy.is_strict_isolated_vocative("What do you think, Keiko?", "Keiko")
    assert DiscourseAttentionPolicy.is_strict_isolated_vocative("Wait, Keiko, listen to me.", "Keiko")
    assert DiscourseAttentionPolicy.is_strict_isolated_vocative("Hey Keiko! Look over here.", "Keiko")

    # 3rd-person mentions should be rejected
    assert not DiscourseAttentionPolicy.is_strict_isolated_vocative("Did you see Keiko?", "Keiko")
    assert not DiscourseAttentionPolicy.is_strict_isolated_vocative("Keiko is coming tomorrow.", "Keiko")
    assert not DiscourseAttentionPolicy.is_strict_isolated_vocative("Because of Keiko, we lost.", "Keiko")


def test_group_dialogue_disambiguation() -> None:
    resolver = InterlocutorResolver()
    active_speakers = {"alice", "bob", "keiko"}
    known_names = {"alice": "Alice", "bob": "Bob", "keiko": "Keiko"}

    # Turn 1: Alice addresses Bob while mentioning Keiko in 3rd person
    u1 = Unit(
        uid="d:1",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="Bob, did you see Keiko earlier?",
        status=Status.PENDING,
        speaker="alice",
    )
    r1 = resolver.resolve(u1, active_speakers=active_speakers, known_names=known_names)
    assert r1.listener == "bob"

    # Turn 2: Alice addresses Bob at trailing position while mentioning Keiko in 3rd person
    u2 = Unit(
        uid="d:2",
        kind=UnitKind.DIALOGUE,
        seq=2,
        source="Did you talk with Keiko about the plan, Bob?",
        status=Status.PENDING,
        speaker="alice",
    )
    r2 = resolver.resolve(u2, active_speakers=active_speakers, known_names=known_names)
    assert r2.listener == "bob"

    # Turn 3: Alice addresses Keiko directly
    u3 = Unit(
        uid="d:3",
        kind=UnitKind.DIALOGUE,
        seq=3,
        source="Keiko, can you help me with this?",
        status=Status.PENDING,
        speaker="alice",
    )
    r3 = resolver.resolve(u3, active_speakers=active_speakers, known_names=known_names)
    assert r3.listener == "keiko"


def test_stage_presence_filter() -> None:
    resolver = InterlocutorResolver()
    active_speakers = {"alice", "bob", "keiko"}
    known_names = {"alice": "Alice", "bob": "Bob", "keiko": "Keiko"}

    # Stage only has Alice and Bob
    stage_snap = StageSnapshot(
        scene_bg="bg room",
        active_sprites=(
            SpriteState(tag="alice", position="left"),
            SpriteState(tag="bob", position="right"),
        ),
    )

    # Unit calls Keiko, but Keiko is not on stage
    u = Unit(
        uid="d:1",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="Keiko, answer me!",
        status=Status.PENDING,
        speaker="alice",
        stage=stage_snap,
    )
    r = resolver.resolve(u, active_speakers=active_speakers, known_names=known_names)
    # Keiko is absent from stage, so Keiko cannot be the addressee
    assert r.listener != "keiko"


def test_question_answer_adjacency_turn() -> None:
    resolver = InterlocutorResolver()
    active_speakers = {"alice", "bob", "keiko"}
    known_names = {"alice": "Alice", "bob": "Bob", "keiko": "Keiko"}

    # Turn 1: Alice asks a question without vocative
    u1 = Unit(
        uid="d:1",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="Where did everyone go?",
        status=Status.PENDING,
        speaker="alice",
    )
    resolver.resolve(u1, active_speakers=active_speakers, known_names=known_names)

    # Turn 2: Bob answers immediately
    u2 = Unit(
        uid="d:2",
        kind=UnitKind.DIALOGUE,
        seq=2,
        source="They went to the cafeteria.",
        status=Status.PENDING,
        speaker="bob",
    )
    r2 = resolver.resolve(u2, prev_unit=u1, active_speakers=active_speakers, known_names=known_names)
    assert r2.listener == "alice"
    assert r2.resolution_method in ("turn_taking", "question_adjacency")


def test_mob_role_classifier_taxonomy() -> None:
    assert MobRoleClassifier.classify("waiter") is MobRoleCategory.SERVICE
    assert MobRoleClassifier.classify("maid_anna") is MobRoleCategory.SERVICE
    assert MobRoleClassifier.classify("police_officer") is MobRoleCategory.AUTHORITY
    assert MobRoleClassifier.classify("doctor") is MobRoleCategory.AUTHORITY
    assert MobRoleClassifier.classify("student_1") is MobRoleCategory.PEER
    assert MobRoleClassifier.classify("villager_boy") is MobRoleCategory.PEER
    assert MobRoleClassifier.classify("unknown_creature") is None


def test_mob_role_pronoun_pair_inferencing() -> None:
    # Service to customer
    assert MobRoleClassifier.infer_pronoun_pair("waiter", "customer") == ("tôi", "quý khách")
    # Customer to service
    assert MobRoleClassifier.infer_pronoun_pair("customer", "waiter") == ("tôi", "cậu")
    # Authority to citizen
    assert MobRoleClassifier.infer_pronoun_pair("police", "student") == ("tôi", "anh/chị")
    # Citizen to authority
    assert MobRoleClassifier.infer_pronoun_pair("student", "doctor") == ("tôi", "ngài")
    # Peer to peer
    assert MobRoleClassifier.infer_pronoun_pair("student_a", "classmate_b") == ("tôi", "bạn")


def test_mob_role_prompt_frame_integration() -> None:
    resolver = InterlocutorResolver()
    active_speakers = {"waiter", "mc"}

    u = Unit(
        uid="d:1",
        kind=UnitKind.DIALOGUE,
        seq=1,
        source="May I take your order, sir?",
        status=Status.PENDING,
        speaker="waiter",
    )
    r = resolver.resolve(u, active_speakers=active_speakers)
    assert r.listener == "mc"

    char_map = {
        "waiter": CharacterProfile(tag="waiter", display_name="Bồi bàn", role="service"),
        "mc": CharacterProfile(tag="mc", display_name="Nhân vật chính", role="protagonist"),
    }

    formatted = format_rich_dialogue_line(u, r, char_map, None)
    assert "Bồi bàn" in formatted
    assert "Nhân vật chính" in formatted
    assert "Vai trò xã hội: Bồi bàn xưng 'tôi', gọi 'quý khách'" in formatted
