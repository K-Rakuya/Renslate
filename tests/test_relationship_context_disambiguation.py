"""Comprehensive synthetic tests verifying character persona and relationship context disambiguation.

Validates that:
1. ContextBuilder does not construct Cartesian all-pairs direct-dialogue rules for unrelated
   speakers in multi-party scenes, eliminating LLM relationship context confusion.
2. infer_semantic_relationship does not falsely classify unrelated third-party NPCs (e.g. Guard, Merchant)
   as family elders, siblings, or romantic partners when interacting with characters whose roles
   mention family terms relative to the MC.
3. InterlocutorResolver respects negative scoring for 3rd-person mentions even when only one candidate
   remains, and maintains speaker-scoped kinship isolation without alias pollution.
4. CharacterProfile pronoun_map sanitizes self-referential character tags without stripping 'self' defaults.
5. collect_pair_evidence canonicalizes character tags and avoids self-loop evidence pairs.
"""

from __future__ import annotations

from renslate.core.config import Config
from renslate.core.models import (
    CharacterProfile,
    RelationshipKind,
    Status,
    Unit,
    UnitKind,
)
from renslate.engine.analyze.persona import infer_semantic_relationship
from renslate.engine.context.builder import ContextBuilder
from renslate.engine.context.prompt_frame import format_rich_dialogue_line
from renslate.engine.context.resolver import (
    InterlocutorResolver,
    ResolvedInterlocutor,
    _gender_conflict,
    invert_reciprocal_pronoun_rule,
)
from renslate.engine.relationships import collect_pair_evidence


def _make_unit(
    uid: str,
    source: str,
    speaker: str | None = None,
    seq: int = 1,
    label: str | None = "scene_court",
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


class TestRelationshipContextDisambiguation:
    """Test suite for relationship disambiguation across multi-party discourse contexts."""

    def test_multi_party_context_builder_avoids_cartesian_all_pairs_confusion(self) -> None:
        """Verify that in multi-party batches with an active resolver, ContextBuilder

        only generates dialogue rules for pairs that actually interact, rather than generating
        an O(N^2) Cartesian product asserting every speaker is directly talking to every other speaker.
        """
        cfg = Config()
        builder = ContextBuilder(cfg)
        resolver = InterlocutorResolver()

        chars = {
            "hero": CharacterProfile(
                "hero",
                "Hero",
                pronoun_map={"liana": "anh - em", "guard": "ta - ngươi"},
            ),
            "liana": CharacterProfile(
                "liana",
                "Liana",
                pronoun_map={"hero": "em - anh", "guard": "tôi - anh"},
            ),
            "guard": CharacterProfile(
                "guard",
                "Gate Guard",
                pronoun_map={"hero": "thần - bệ hạ", "merchant": "ta - ngươi"},
            ),
            "merchant": CharacterProfile(
                "merchant",
                "Potion Merchant",
                pronoun_map={"guard": "thảo dân - đại nhân", "hero": "thảo dân - ngài"},
            ),
        }

        # Sub-dialogue 1: Hero talking to Liana
        # Sub-dialogue 2: Guard talking to Merchant
        units = [
            _make_unit("d:1", "Liana, are you ready to leave?", speaker="hero", seq=1),
            _make_unit("d:2", "Yes, Hero. Let us make haste.", speaker="liana", seq=2),
            _make_unit("d:3", "Show me your trading permit, merchant.", speaker="guard", seq=3),
            _make_unit("d:4", "Here it is, honored guard.", speaker="merchant", seq=4),
        ]

        prompt = builder.build_user_prompt(units, characters=chars, resolver=resolver)

        # Interacting pairs must be recognized
        assert "- Hero và Liana đang đối thoại trực tiếp" in prompt
        assert "- Gate Guard và Potion Merchant đang đối thoại trực tiếp" in prompt

        # Non-interacting cross-pairs must NOT be asserted as "đang đối thoại trực tiếp"
        assert "- Hero và Gate Guard đang đối thoại trực tiếp" not in prompt
        assert "- Hero và Potion Merchant đang đối thoại trực tiếp" not in prompt
        assert "- Liana và Gate Guard đang đối thoại trực tiếp" not in prompt
        assert "- Liana và Potion Merchant đang đối thoại trực tiếp" not in prompt

    def test_infer_semantic_relationship_unrelated_npc_does_not_inherit_family_elder(self) -> None:
        """Verify that a third-party NPC (e.g. Guard) interacting with 'Mother of the MC'

        does NOT receive FAMILY_ELDER relationship simply because the partner role contains 'mother'.
        """
        guard_tag = "guard"
        mother_tag = "queen_mother"

        guard_info = {"role": "Captain of the Royal Guard", "display_name": "Guard"}
        mother_info = {"role": "Queen Dowager, mother of the Hero", "display_name": "Queen Mother"}

        # Guard speaking to Queen Mother (neither is MC): Should NOT infer Family Elder
        res_guard = infer_semantic_relationship(
            guard_tag, guard_info, mother_tag, mother_info, is_mc=False, partner_is_mc=False
        )
        assert res_guard is None or res_guard[0] != RelationshipKind.FAMILY_ELDER, (
            f"Expected unrelated guard not to inherit FAMILY_ELDER, got {res_guard}"
        )

        # But MC speaking to Queen Mother MUST infer Family Elder
        hero_info = {"role": "Protagonist / Crown Prince", "display_name": "Hero"}
        res_mc = infer_semantic_relationship(
            "hero", hero_info, mother_tag, mother_info, is_mc=True, partner_is_mc=False
        )
        assert res_mc is not None
        assert res_mc[0] == RelationshipKind.FAMILY_ELDER
        assert res_mc[1] is False  # Mother is elder/senior relative to MC

    def test_infer_semantic_relationship_unrelated_npc_does_not_inherit_sibling(self) -> None:
        """Verify that an unrelated NPC interacting with 'Sister of the MC'

        does NOT receive FAMILY_SIBLING relationship.
        """
        merchant_tag = "merchant"
        sister_tag = "evelyn"

        merchant_info = {"role": "Wandering potion merchant", "display_name": "Merchant"}
        sister_info = {"role": "Eldest sister of Luke", "display_name": "Evelyn"}

        res = infer_semantic_relationship(
            merchant_tag, merchant_info, sister_tag, sister_info, is_mc=False, partner_is_mc=False
        )
        assert res is None or res[0] != RelationshipKind.FAMILY_SIBLING

    def test_infer_semantic_relationship_unrelated_npc_does_not_inherit_romance(self) -> None:
        """Verify that an unrelated NPC interacting with 'Wife of the MC'

        does NOT receive ROMANCE relationship.
        """
        bandit_tag = "bandit"
        wife_tag = "liana"

        bandit_info = {"role": "Forest Bandit", "display_name": "Bandit"}
        wife_info = {"role": "Beloved wife and consort of Hero", "display_name": "Liana"}

        res_bandit = infer_semantic_relationship(
            bandit_tag, bandit_info, wife_tag, wife_info, is_mc=False, partner_is_mc=False
        )
        assert res_bandit is None or res_bandit[0] != RelationshipKind.ROMANCE

        # MC with wife MUST receive Romance
        hero_info = {"role": "Hero", "display_name": "Hero"}
        res_mc = infer_semantic_relationship(
            "hero", hero_info, wife_tag, wife_info, is_mc=True, partner_is_mc=False
        )
        assert res_mc is not None
        assert res_mc[0] == RelationshipKind.ROMANCE

    def test_resolver_negative_scoring_on_third_person_reference_with_single_candidate(self) -> None:
        """Verify that when only one candidate is in the pool, but that candidate is explicitly

        referenced in the third person within the source text, InterlocutorResolver does not
        mistake them for the vocative addressee.
        """
        resolver = InterlocutorResolver()
        active = {"bob", "alice"}
        names = {"bob": "Bob", "alice": "Alice"}

        # Alice is talking, Bob is the only other participant in active speakers.
        # But Bob is explicitly mentioned in 3rd person: 'Where did Bob go?'
        u = _make_unit("d:1", "Where did Bob go?", speaker="alice")
        res = resolver.resolve(u, active_speakers=active, known_names=names)

        # Bob must NOT be resolved as the listener of a vocative
        assert res.resolution_method != "vocative"

    def test_character_profile_strips_self_referential_tag_in_prompt_building(self) -> None:
        """Verify that ContextBuilder Section 3.1 does not print self-referential rules

        like 'hero: tôi - tôi' in the character's profile block.
        """
        cfg = Config()
        builder = ContextBuilder(cfg)

        chars = {
            "hero": CharacterProfile(
                "hero",
                "Hero",
                pronoun_map={
                    "hero": "tôi - tôi",
                    "Hero": "tôi - tôi",
                    "liana": "anh - em",
                    "self": "tôi",
                },
            ),
        }

        units = [_make_unit("d:1", "I will protect everyone.", speaker="hero")]
        prompt = builder.build_user_prompt(units, characters=chars)

        # Profile block should format external target rules, not self-loops
        assert "với liana (xưng 'anh', gọi 'em')" in prompt
        assert "với hero" not in prompt
        assert "với Hero" not in prompt

    def test_collect_pair_evidence_canonicalization_and_self_loop_prevention(self) -> None:
        """Verify collect_pair_evidence canonicalizes tags and ignores self-turns."""
        chars = {
            "hero": CharacterProfile("hero", "Hero"),
            "villain": CharacterProfile("villain", "Lord Malakor"),
        }

        # Case variations in speakers
        units = [
            _make_unit("d:1", "Face me, villain!", speaker="Hero", seq=1),
            _make_unit("d:2", "You are too late, hero.", speaker="villain", seq=2),
            # Self-turn monologue
            _make_unit("d:3", "I cannot give up now...", speaker="hero", seq=3),
            _make_unit("d:4", "I must draw strength from within.", speaker="Hero", seq=4),
        ]

        evidence = collect_pair_evidence(units, chars)

        # Expected canonical pair is ('hero', 'villain')
        assert ("hero", "villain") in evidence
        assert len(evidence) == 1  # No self-loops like ('hero', 'hero') or duplicate ('Hero', 'villain')

    def test_builder_does_not_invert_non_reciprocal_pronoun_pairs(self) -> None:
        """Verify ContextBuilder does NOT blindly swap asymmetric pairs like 'ta - ngươi'.

        When King has pronoun_map={'guard': 'ta - ngươi'}, Guard must NOT receive an inverted
        rule telling Guard to self-reference as 'ngươi' (2nd person) and address King as 'ta' (1st person).
        However, reciprocal kinship pairs like 'anh - em' must cleanly invert to 'em - anh'.
        """
        cfg = Config()
        builder = ContextBuilder(cfg)
        chars = {
            "king": CharacterProfile("king", "King", pronoun_map={"guard": "ta - ngươi"}),
            "guard": CharacterProfile("guard", "Guard", pronoun_map={}),
            "hero": CharacterProfile("hero", "Hero", pronoun_map={"liana": "anh - em"}),
            "liana": CharacterProfile("liana", "Liana", pronoun_map={}),
        }

        # Asymmetric pair: King & Guard
        units_asym = [
            _make_unit("d:1", "Where is the merchant?", speaker="king", seq=1),
            _make_unit("d:2", "He fled, Your Majesty.", speaker="guard", seq=2),
        ]
        prompt_asym = builder.build_user_prompt(units_asym, characters=chars)
        assert "Tự xưng (I/me/my) là 'ngươi'" not in prompt_asym, (
            "Guard must not be instructed to self-refer as derogatory/2nd-person 'ngươi'!"
        )

        # Reciprocal pair: Hero & Liana
        units_sym = [
            _make_unit("d:3", "Liana, follow me.", speaker="hero", seq=3),
            _make_unit("d:4", "Right behind you, Hero.", speaker="liana", seq=4),
        ]
        prompt_sym = builder.build_user_prompt(units_sym, characters=chars)
        assert "Khi Liana (liana) nói với Hero (hero): Tự xưng (I/me/my) là 'em', gọi đối phương (you/your) là 'anh'" in prompt_sym

    def test_infer_semantic_relationship_external_relation_in_role_does_not_taint_mc(self) -> None:
        """Verify NPC roles mentioning third-party family (e.g. 'husband to Mary', 'father of two')

        are not erroneously inferred as MC's romance, elder, or sibling.
        """
        mc_info = {"role": "Protagonist", "display_name": "Hero", "gender": "male"}

        # 1. Guard with external wife and children
        guard_info = {
            "role": "Captain of the guard, devoted husband to Mary and father of two",
            "display_name": "Guard",
            "gender": "male",
        }
        res_guard = infer_semantic_relationship("mc", mc_info, "guard", guard_info, is_mc=True, partner_is_mc=False)
        assert res_guard is not None
        assert res_guard[0] != RelationshipKind.ROMANCE
        assert res_guard[0] != RelationshipKind.FAMILY_ELDER

        # 2. Blacksmith husband to Mary
        bs_info = {
            "role": "Village blacksmith, devoted husband to Mary",
            "display_name": "Blacksmith",
            "gender": "male",
        }
        res_bs = infer_semantic_relationship("mc", mc_info, "blacksmith", bs_info, is_mc=True, partner_is_mc=False)
        assert res_bs is not None
        assert res_bs[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 3. Orphan who lost his sister
        orphan_info = {
            "role": "A young boy who lost his sister in the war",
            "display_name": "Orphan",
            "gender": "male",
        }
        res_orphan = infer_semantic_relationship("mc", mc_info, "orphan", orphan_info, is_mc=True, partner_is_mc=False)
        assert res_orphan is not None
        assert res_orphan[0] != RelationshipKind.FAMILY_SIBLING

    def test_infer_semantic_relationship_word_boundary_avoids_substring_false_positive(self) -> None:
        """Verify tag substring matching does not falsely fire inside unrelated words (e.g. 'al' in 'royal')."""
        al_info = {"role": "Royal alchemist", "display_name": "Al", "gender": "male"}
        queen_info = {
            "role": "Queen mother of the royal prince",
            "display_name": "Elizabeth",
            "gender": "female",
        }
        res = infer_semantic_relationship("al", al_info, "elizabeth", queen_info, is_mc=False, partner_is_mc=False)
        assert res is not None
        assert res[0] != RelationshipKind.FAMILY_ELDER

    def test_resolver_rejects_kinship_on_zero_evidence_and_gender_mismatch(self) -> None:
        """Verify InterlocutorResolver does not resolve male candidate as 'Mother' on zero evidence."""
        resolver = InterlocutorResolver()
        active = {"kevin", "alice"}
        names = {"kevin": "Kevin", "alice": "Alice"}

        u = _make_unit("d:1", "Mother, help me!", speaker="alice")
        res = resolver.resolve(u, active_speakers=active, known_names=names)

        # Kevin must not be resolved as Mother via vocative, and alias table must remain unpoisoned
        assert res.listener != "kevin" or res.resolution_method != "vocative"
        assert "mother" not in resolver._aliases

    def test_prompt_frame_mob_role_classification_with_speaker_having_other_pronoun_rules(self) -> None:
        """Verify MobRoleClassifier fires even when speaker has a pronoun_map with rules for other characters."""
        char_map = {
            "mc": CharacterProfile("mc", "Hero", role="protagonist", pronoun_map={"liana": "anh - em"}),
            "waiter": CharacterProfile("waiter", "Bồi bàn", role="service"),
        }
        resolver = InterlocutorResolver()
        u = _make_unit("d:1", "May I have the menu?", speaker="mc")
        res = resolver.resolve(u, active_speakers={"mc", "waiter"}, known_names={"mc": "Hero", "waiter": "Bồi bàn"})

        formatted = format_rich_dialogue_line(u, res, char_map, None)
        assert "Vai trò xã hội: Hero xưng 'tôi', gọi 'cậu'" in formatted

    def test_resolver_trailing_vocative_turn_taking_does_not_poison_aliases(self) -> None:
        """Verify trailing vocatives that fail kinship validation do NOT get backdoored into aliases via turn-taking."""
        resolver = InterlocutorResolver()
        active = {"kevin", "alice"}
        names = {"kevin": "Kevin", "alice": "Alice"}

        u_prev = _make_unit("d:1", "Are you ready?", speaker="kevin", seq=1)
        u_curr = _make_unit("d:2", "I do not know, mother.", speaker="alice", seq=2)

        res = resolver.resolve(u_curr, prev_unit=u_prev, active_speakers=active, known_names=names)
        assert res.listener == "kevin"
        assert res.resolution_method == "turn_taking"
        # Kevin must NEVER be aliased to 'mother'!
        assert "mother" not in resolver._aliases
        assert ("alice", "mother") not in resolver._speaker_aliases

    def test_resolver_zero_evidence_prev_spk_and_pronoun_does_not_alias_kinship(self) -> None:
        """Verify that a previous speaker with zero kinship evidence is not resolved as 'mother' when 3rd-person pronoun appears."""
        resolver = InterlocutorResolver()
        active = {"luke", "alice"}
        names = {"luke": "Luke", "alice": "Alice"}

        u_prev = _make_unit("d:1", "Are you okay?", speaker="luke", seq=1)
        u_curr = _make_unit("d:2", "Mother, where did he go?", speaker="alice", seq=2)

        res = resolver.resolve(u_curr, prev_unit=u_prev, active_speakers=active, known_names=names)
        assert res.resolution_method != "vocative"
        assert "mother" not in resolver._aliases

    def test_resolve_pronouns_does_not_invert_asymmetric_authority_pairs(self) -> None:
        """Verify resolve_pronouns does NOT invert asymmetric authority pairs like 'ta - ngươi'."""
        resolver = InterlocutorResolver()
        char_map = {
            "king": CharacterProfile("king", "King", pronoun_map={"guard": "ta - ngươi"}),
            "guard": CharacterProfile("guard", "Guard", pronoun_map={}),
        }

        # Guard speaking to King: must NOT receive ('ngươi', 'ta')
        pair = resolver.resolve_pronouns("guard", "king", char_map=char_map)
        assert pair != ("ngươi", "ta")

    def test_resolve_pronouns_inverts_reciprocal_pairs_cleanly(self) -> None:
        """Verify resolve_pronouns cleanly inverts reciprocal pairs like 'anh - em' -> 'em - anh'."""
        resolver = InterlocutorResolver()
        char_map = {
            "hero": CharacterProfile("hero", "Hero", pronoun_map={"liana": "anh - em"}),
            "liana": CharacterProfile("liana", "Liana", pronoun_map={}),
        }

        pair = resolver.resolve_pronouns("liana", "hero", char_map=char_map)
        assert pair == ("em", "anh")

    def test_infer_semantic_relationship_individual_backstory_descriptions_not_falsely_bound_to_mc(self) -> None:
        """Verify NPCs with individual backstories (grieving father, widow, missing boyfriend) are not inferred as MC's family or romance."""
        mc_info = {"role": "Protagonist", "display_name": "Hero", "gender": "male"}

        # 1. Grieving father
        grieving_father = {"role": "A grieving father seeking justice for his village", "display_name": "Old Man", "gender": "male"}
        res1 = infer_semantic_relationship("mc", mc_info, "old_man", grieving_father, is_mc=True, partner_is_mc=False)
        assert res1 is not None and res1[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 2. Woman searching for missing boyfriend
        searcher = {"role": "A young woman searching for her missing boyfriend", "display_name": "Anna", "gender": "female"}
        res2 = infer_semantic_relationship("mc", mc_info, "anna", searcher, is_mc=True, partner_is_mc=False)
        assert res2 is not None and res2[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 3. Widow mourning late husband
        widow = {"role": "A widow mourning her late husband", "display_name": "Martha", "gender": "female"}
        res3 = infer_semantic_relationship("mc", mc_info, "martha", widow, is_mc=True, partner_is_mc=False)
        assert res3 is not None and res3[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 4. Devoted husband
        husband = {"role": "A devoted husband and brave warrior", "display_name": "Goran", "gender": "male"}
        res4 = infer_semantic_relationship("mc", mc_info, "goran", husband, is_mc=True, partner_is_mc=False)
        assert res4 is not None and res4[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 5. Brother fighting for his home
        brother = {"role": "A protective older brother who fights for his hometown", "display_name": "Leo", "gender": "male"}
        res5 = infer_semantic_relationship("mc", mc_info, "leo", brother, is_mc=True, partner_is_mc=False)
        assert res5 is not None and res5[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    def test_infer_semantic_relationship_supreme_royalty_senior_to_officers(self) -> None:
        """Verify supreme royalty (King) is senior to subordinate officer (Captain), and two sovereign kings are peers."""
        king_info = {"role": "King of the realm", "display_name": "King Arthur"}
        captain_info = {"role": "Captain of the Guard", "display_name": "Lancelot"}

        # King to Captain: King is senior
        res_kc = infer_semantic_relationship("king", king_info, "captain", captain_info)
        assert res_kc is not None
        assert res_kc[0] == RelationshipKind.SUPERIOR_SUBORDINATE
        assert res_kc[1] is True  # Speaker is senior

        # Captain to King: Captain is subordinate
        res_ck = infer_semantic_relationship("captain", captain_info, "king", king_info)
        assert res_ck is not None
        assert res_ck[0] == RelationshipKind.SUPERIOR_SUBORDINATE
        assert res_ck[1] is False  # Speaker is subordinate

        # Two allied sovereign Kings: Peers
        king2_info = {"role": "King of the northern lands", "display_name": "King Charles"}
        res_kk = infer_semantic_relationship("king1", king_info, "king2", king2_info)
        assert res_kk is not None
        assert res_kk[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    def test_infer_semantic_relationship_romance_overrides_royalty(self) -> None:
        """Verify intimate romance cues take precedence over court royalty hierarchy."""
        mc_info = {"role": "Knight / Protagonist", "display_name": "Hero", "gender": "male"}
        princess_info = {"role": "Princess of the realm, beloved fiancee of the Hero", "display_name": "Liana", "gender": "female"}

        res = infer_semantic_relationship("mc", mc_info, "liana", princess_info, is_mc=True)
        assert res is not None
        assert res[0] == RelationshipKind.ROMANCE

    def test_context_builder_direct_dialogue_pair_resolves_protagonist_mc_rule(self) -> None:
        """Verify ContextBuilder Section 3.2 resolves direct dialogue rules when partner rule is keyed by 'mc'."""
        cfg = Config()
        builder = ContextBuilder(cfg)
        chars = {
            "hero": CharacterProfile("hero", "Hero", role="Protagonist"),
            "liana": CharacterProfile("liana", "Liana", pronoun_map={"mc": "em - anh"}),
        }
        units = [
            _make_unit("d:1", "Hero, wait for me!", speaker="liana", seq=1),
            _make_unit("d:2", "I am here, Liana.", speaker="hero", seq=2),
        ]
        prompt = builder.build_user_prompt(units, characters=chars)
        assert "- Liana và Hero đang đối thoại trực tiếp" in prompt
        assert "Tự xưng (I/me/my) là 'em', gọi đối phương (you/your) là 'anh'" in prompt


    def test_infer_semantic_relationship_mentor_backstory_does_not_falsely_bind_to_mc(self) -> None:
        """Verify NPC with narrative backstory mentioning master is NOT inferred as MC's mentor."""
        mc_info = {"role": "Protagonist", "display_name": "Hero", "gender": "male"}
        rikku_info = {
            "role": "A young warrior searching for his missing master",
            "display_name": "Rikku",
            "gender": "male",
        }
        res = infer_semantic_relationship("mc", mc_info, "rikku", rikku_info, is_mc=True)
        assert res is not None
        assert res[0] != RelationshipKind.MENTOR_DISCIPLE, (
            f"Expected formal acquaintance, got {res[0]} (MC falsely calling warrior master)"
        )

        # True mentor (Master Wu) must be inferred as Mentor
        wu_info = {"role": "Master Wu, mentor of the Hero", "display_name": "Master Wu", "gender": "male"}
        res_wu = infer_semantic_relationship("mc", mc_info, "wu", wu_info, is_mc=True)
        assert res_wu is not None
        assert res_wu[0] == RelationshipKind.MENTOR_DISCIPLE
        assert res_wu[1] is False  # MC is disciple (junior)

        # True apprentice (Tim) must be inferred as Disciple
        tim_info = {"role": "Inquisitive apprentice of the Hero", "display_name": "Tim", "gender": "male"}
        res_tim = infer_semantic_relationship("mc", mc_info, "tim", tim_info, is_mc=True)
        assert res_tim is not None
        assert res_tim[0] == RelationshipKind.MENTOR_DISCIPLE
        assert res_tim[1] is True  # MC is mentor (senior)

    def test_infer_semantic_relationship_child_kinship_recognized_accurately(self) -> None:
        """Verify child/daughter/son cues are recognized as FAMILY_ELDER with correct relative seniority."""
        mc_info = {"role": "Protagonist / Hero", "display_name": "Hero", "gender": "male"}
        lily_info = {"role": "Daughter of the Hero", "display_name": "Lily", "gender": "female"}

        # MC to Daughter: MC is elder
        res_mc_lily = infer_semantic_relationship("mc", mc_info, "lily", lily_info, is_mc=True)
        assert res_mc_lily is not None
        assert res_mc_lily[0] == RelationshipKind.FAMILY_ELDER
        assert res_mc_lily[1] is True  # Speaker is senior

        # Daughter to MC: Daughter is junior
        res_lily_mc = infer_semantic_relationship("lily", lily_info, "mc", mc_info, partner_is_mc=True)
        assert res_lily_mc is not None
        assert res_lily_mc[0] == RelationshipKind.FAMILY_ELDER
        assert res_lily_mc[1] is False  # Speaker is junior

        # Non-MC family pair: John ("Father of Lily") and Lily ("Daughter of John")
        john_info = {"role": "Father of Lily", "display_name": "John", "gender": "male"}
        res_jl = infer_semantic_relationship("john", john_info, "lily", lily_info)
        assert res_jl is not None
        assert res_jl[0] == RelationshipKind.FAMILY_ELDER
        assert res_jl[1] is True

        res_lj = infer_semantic_relationship("lily", lily_info, "john", john_info)
        assert res_lj is not None
        assert res_lj[0] == RelationshipKind.FAMILY_ELDER
        assert res_lj[1] is False

    def test_infer_semantic_relationship_younger_sibling_is_junior(self) -> None:
        """Verify little sister / younger sibling is classified as junior relative to the elder protagonist."""
        mc_info = {"role": "Protagonist / Hero", "display_name": "Hero", "gender": "male"}
        iris_info = {"role": "Little sister of the Hero", "display_name": "Iris", "gender": "female"}

        # Iris to Hero: Iris is junior sibling
        res_iris = infer_semantic_relationship("iris", iris_info, "mc", mc_info, partner_is_mc=True)
        assert res_iris is not None
        assert res_iris[0] == RelationshipKind.FAMILY_SIBLING
        assert res_iris[1] is False  # Iris must NOT be senior!

        # Hero to Iris: Hero is senior sibling
        res_hero = infer_semantic_relationship("mc", mc_info, "iris", iris_info, is_mc=True)
        assert res_hero is not None
        assert res_hero[0] == RelationshipKind.FAMILY_SIBLING
        assert res_hero[1] is True  # Hero is senior

    def test_format_rich_dialogue_line_reciprocal_pronoun_map_and_graph(self) -> None:
        """Verify format_rich_dialogue_line displays relational context when reciprocal rule exists in listener pronoun_map or graph."""
        hero = CharacterProfile(
            tag="hero", display_name="Hero", role="Protagonist", pronoun_map={"liana": "anh - em"}
        )
        liana = CharacterProfile(tag="liana", display_name="Liana", role="Heroine", pronoun_map={})
        char_map = {"hero": hero, "liana": liana}

        u = _make_unit("u1", "Are you feeling all right?", speaker="liana")
        res = ResolvedInterlocutor(speaker="liana", listener="hero")

        # Liana speaking to Hero should receive reciprocal rule ('em', 'anh')
        formatted = format_rich_dialogue_line(u, res, char_map)
        assert "Quan hệ: Liana xưng 'em', gọi 'anh'" in formatted

    def test_resolver_gender_conflict_vietnamese_honorifics(self) -> None:
        """Verify _gender_conflict identifies Vietnamese masculine honorifics ('Chú', 'Bác') conflicting with feminine vocatives ('Mẹ')."""
        assert _gender_conflict("mẹ", "Chú Hùng", "chu_hung") is True
        assert _gender_conflict("bố", "Cô Lan", "co_lan") is True
        assert _gender_conflict("mẹ", "Bác Ba", "bac_ba") is True

    def test_builder_skips_mc_self_referential_pronoun_rule(self) -> None:
        """Verify ContextBuilder Section 3.1 does not format self-referential rules when MC has 'mc' key in pronoun_map."""
        cfg = Config()
        builder = ContextBuilder(cfg)
        chars = {
            "hero": CharacterProfile(
                "hero",
                "Hero",
                role="Protagonist",
                pronoun_map={"mc": "anh - em", "liana": "anh - em"},
            ),
            "liana": CharacterProfile("liana", "Liana", role="Heroine"),
        }
        units = [_make_unit("d:1", "I will do my best.", speaker="hero")]
        prompt = builder.build_user_prompt(units, characters=chars)
        assert "với Hero (mc)" not in prompt
        assert "với liana" in prompt

    def test_infer_semantic_relationship_external_possessive_saxon_genitive(self) -> None:
        """Verify Saxon genitive roles ('Bob's wife', 'Arthur's daughter', 'John's mother')

        speaking to MC are NOT inferred as MC's family, romance, or apprentice.
        """
        mc_info = {"role": "Protagonist", "display_name": "Hero", "gender": "male"}

        # 1. Bob's wife
        sarah_info = {"role": "Bob's wife, a quiet herbalist", "display_name": "Sarah", "gender": "female"}
        res1 = infer_semantic_relationship("mc", mc_info, "sarah", sarah_info, is_mc=True)
        assert res1 is not None and res1[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 2. Arthur's daughter
        gwen_info = {"role": "Arthur's daughter", "display_name": "Gwen", "gender": "female"}
        res2 = infer_semantic_relationship("mc", mc_info, "gwen", gwen_info, is_mc=True)
        assert res2 is not None and res2[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 3. John's mother
        mary_info = {"role": "John's mother", "display_name": "Mary", "gender": "female"}
        res3 = infer_semantic_relationship("mc", mc_info, "mary", mary_info, is_mc=True)
        assert res3 is not None and res3[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 4. Kevin's little sister
        chloe_info = {"role": "Kevin's little sister", "display_name": "Chloe", "gender": "female"}
        res4 = infer_semantic_relationship("mc", mc_info, "chloe", chloe_info, is_mc=True)
        assert res4 is not None and res4[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 5. Peter's apprentice
        tom_info = {"role": "Peter's apprentice", "display_name": "Tom", "gender": "male"}
        res5 = infer_semantic_relationship("mc", mc_info, "tom", tom_info, is_mc=True)
        assert res5 is not None and res5[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    def test_infer_semantic_relationship_vietnamese_post_nominal_modifiers_recognized(self) -> None:
        """Verify Vietnamese kinship with post-nominal modifiers bound to the protagonist are accurately recognized."""
        mc_info = {"role": "Nhân vật chính", "display_name": "Lâm", "gender": "male"}

        # 1. Cha ruột của nhân vật chính
        minh_info = {"role": "Cha ruột của nhân vật chính", "display_name": "Minh", "gender": "male"}
        res1 = infer_semantic_relationship("mc", mc_info, "minh", minh_info, is_mc=True)
        assert res1 is not None
        assert res1[0] == RelationshipKind.FAMILY_ELDER
        assert res1[1] is False  # MC is junior

        # 2. Mẹ kế của nhân vật chính
        lan_info = {"role": "Mẹ kế của nhân vật chính", "display_name": "Lan", "gender": "female"}
        res2 = infer_semantic_relationship("mc", mc_info, "lan", lan_info, is_mc=True)
        assert res2 is not None
        assert res2[0] == RelationshipKind.FAMILY_ELDER
        assert res2[1] is False

        # 3. Con gái ruột của nhân vật chính
        hoa_info = {"role": "Con gái ruột của nhân vật chính", "display_name": "Hoa", "gender": "female"}
        res3 = infer_semantic_relationship("mc", mc_info, "hoa", hoa_info, is_mc=True)
        assert res3 is not None
        assert res3[0] == RelationshipKind.FAMILY_ELDER
        assert res3[1] is True  # MC is senior

        # 4. Vợ chưa cưới của nhân vật chính
        nga_info = {"role": "Vợ chưa cưới của nhân vật chính", "display_name": "Nga", "gender": "female"}
        res4 = infer_semantic_relationship("mc", mc_info, "nga", nga_info, is_mc=True)
        assert res4 is not None
        assert res4[0] == RelationshipKind.ROMANCE

        # 5. Em gái ruột của nhân vật chính
        mai_info = {"role": "Em gái ruột của nhân vật chính", "display_name": "Mai", "gender": "female"}
        res5 = infer_semantic_relationship("mc", mc_info, "mai", mai_info, is_mc=True)
        assert res5 is not None
        assert res5[0] == RelationshipKind.FAMILY_SIBLING
        assert res5[1] is True  # MC is senior brother

    def test_infer_semantic_relationship_vietnamese_external_possessives_not_falsely_bound(self) -> None:
        """Verify Vietnamese external possessive phrases ('Bố ruột John', 'Vợ Nam') do NOT bind to MC."""
        mc_info = {"role": "Nhân vật chính", "display_name": "Lâm", "gender": "male"}

        # 1. Bố ruột John
        father_john = {"role": "Bố ruột John", "display_name": "Minh", "gender": "male"}
        res1 = infer_semantic_relationship("mc", mc_info, "minh", father_john, is_mc=True)
        assert res1 is not None and res1[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 2. Bố nuôi Huy
        father_huy = {"role": "Bố nuôi Huy", "display_name": "Hải", "gender": "male"}
        res2 = infer_semantic_relationship("mc", mc_info, "hai", father_huy, is_mc=True)
        assert res2 is not None and res2[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 3. Vợ Nam
        wife_nam = {"role": "Vợ Nam", "display_name": "Linh", "gender": "female"}
        res3 = infer_semantic_relationship("mc", mc_info, "linh", wife_nam, is_mc=True)
        assert res3 is not None and res3[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    def test_infer_semantic_relationship_servant_friend_rival_not_bound_to_unrelated_third_parties(self) -> None:
        """Verify unrelated NPCs (Guard, Merchant) interacting with NPCs having servant, friend, or rival cues

        do NOT inherit those relationships when neither is the designated target.
        """
        guard_info = {"role": "Gate guard", "display_name": "Guard"}
        merchant_info = {"role": "Potion merchant", "display_name": "Merchant"}

        # 1. Guard interacting with Alice ('Childhood friend of Bob')
        alice_friend = {"role": "Childhood friend of Bob", "display_name": "Alice"}
        res_friend = infer_semantic_relationship("guard", guard_info, "alice", alice_friend)
        assert res_friend is not None and res_friend[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 2. Merchant interacting with Alice ('Rival of Bob')
        alice_rival = {"role": "Rival of Bob", "display_name": "Alice"}
        res_rival = infer_semantic_relationship("merchant", merchant_info, "alice", alice_rival)
        assert res_rival is not None and res_rival[0] == RelationshipKind.FORMAL_ACQUAINTANCE

        # 3. Merchant interacting with Butler of Bob
        butler_bob = {"role": "Butler to Bob", "display_name": "Sebastian"}
        res_butler = infer_semantic_relationship("merchant", merchant_info, "sebastian", butler_bob)
        assert res_butler is not None and res_butler[0] == RelationshipKind.FORMAL_ACQUAINTANCE

    def test_resolver_gender_conflict_uncle_aunt_son_daughter(self) -> None:
        """Verify _gender_conflict identifies conflicts on candidates titled Uncle, Aunt, Son, Daughter."""
        assert _gender_conflict("mother", "Uncle Vance", "vance") is True
        assert _gender_conflict("father", "Aunt Beatrice", "beatrice") is True
        assert _gender_conflict("father", "Con gái Mai", "mai") is True
        assert _gender_conflict("mother", "Con trai Nam", "nam") is True

    def test_resolver_reciprocal_pairs_martial_arts_de_tu(self) -> None:
        """Verify reciprocal pair inversion supports martial arts 'sư phụ - đệ tử'."""
        assert invert_reciprocal_pronoun_rule("sư phụ - đệ tử") == "đệ tử - sư phụ"
        assert invert_reciprocal_pronoun_rule("đệ tử - sư phụ") == "sư phụ - đệ tử"

        resolver = InterlocutorResolver()
        char_map = {
            "master": CharacterProfile("master", "Sư phụ", pronoun_map={"linh": "sư phụ - đệ tử"}),
            "linh": CharacterProfile("linh", "Linh", pronoun_map={}),
        }
        pair = resolver.resolve_pronouns("linh", "master", char_map=char_map)
        assert pair == ("đệ tử", "sư phụ")


