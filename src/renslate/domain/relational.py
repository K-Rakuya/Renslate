"""Multi-dimensional relational state vector and algebraic pronoun projection."""

from __future__ import annotations

from typing import Any

from renslate.core.models import KinshipKind, RelationalVector


def _project_canonical(
    vec: RelationalVector,
    ga: str,
    gb: str,
) -> tuple[str, str]:
    """Directed canonical projection for non-negative hierarchy (delta_hierarchy >= 0)."""
    # 1. Sovereign-Subject
    if vec.kinship == KinshipKind.SOVEREIGN_SUBJECT:
        if vec.delta_hierarchy > 0:
            return ("trẫm", "khanh") if vec.register >= 0.8 else ("ta", "ngươi")
        return ("khanh", "trẫm") if vec.register >= 0.8 else ("ngươi", "ta")

    # 2. Parent-Child
    if vec.kinship == KinshipKind.PARENT_CHILD:
        if vec.delta_hierarchy > 0:
            p_term = "cha" if ga == "male" else ("mẹ" if ga == "female" else "bố")
            return (p_term, "con")
        p_term = "cha" if gb == "male" else ("mẹ" if gb == "female" else "bố")
        return ("con", p_term)

    # 3. Romance
    if vec.kinship == KinshipKind.ROMANTIC_COUPLE:
        if vec.register >= 0.8:
            return ("chàng", "nàng") if ga == "male" else ("nàng", "chàng")
        if ga == "male" and gb == "female":
            return ("anh", "em")
        if ga == "female" and gb == "male":
            return ("em", "anh")
        return ("anh", "em") if vec.delta_hierarchy >= 0 else ("em", "anh")

    # 4. Sibling
    if vec.kinship == KinshipKind.SIBLING:
        if vec.delta_hierarchy > 0:
            elder = "anh" if ga == "male" else "chị"
            return (elder, "em")
        if vec.delta_hierarchy < 0:
            elder = "anh" if gb == "male" else "chị"
            return ("em", elder)
        return ("tớ", "cậu")

    # 5. Mentor-Disciple
    if vec.kinship == KinshipKind.MENTOR_DISCIPLE:
        if vec.delta_hierarchy > 0:
            return ("thầy", "trò") if vec.register >= 0.7 else ("ta", "con")
        lis = "thầy" if gb == "male" else ("cô" if gb == "female" else "thầy")
        return ("trò", lis) if vec.register >= 0.7 else ("con", lis)

    # 6. Avuncular
    if vec.kinship == KinshipKind.AVUNCULAR:
        if vec.delta_hierarchy > 0:
            elder = "chú" if ga == "male" else ("cô" if ga == "female" else "bác")
            return (elder, "cháu")
        elder = "chú" if gb == "male" else ("cô" if gb == "female" else "bác")
        return ("cháu", elder)

    # 7. Grand-Relation
    if vec.kinship == KinshipKind.GRAND_RELATION:
        if vec.delta_hierarchy > 0:
            elder = "ông" if ga == "male" else ("bà" if ga == "female" else "ông")
            return (elder, "cháu")
        elder = "ông" if gb == "male" else ("bà" if gb == "female" else "ông")
        return ("cháu", elder)

    # 8. Sworn-Alliance
    if vec.kinship == KinshipKind.SWORN_ALLIANCE:
        if vec.delta_hierarchy > 0:
            elder = "anh" if ga == "male" else "chị"
            return (elder, "hiền đệ") if vec.register >= 0.7 else (elder, "em")
        return ("huynh đệ", "huynh đệ") if vec.register >= 0.7 else ("tớ", "cậu")

    # 9. Non-kinship partitions:
    # Antagonistic / Hostility
    if vec.affinity <= -0.3 and vec.register <= 0.4:
        if vec.delta_hierarchy >= 0:
            return ("tao", "mày")
        return ("mày", "tao")

    if vec.affinity <= -0.4:
        if vec.delta_hierarchy >= 0:
            return ("ta", "ngươi")
        return ("ngươi", "ta")

    # Hierarchy driven non-kinship
    if abs(vec.delta_hierarchy) >= 0.2:
        if vec.delta_hierarchy > 0:
            if ga == "male" and gb == "female":
                return ("anh", "em")
            if ga == "female" and gb == "male":
                return ("chị", "em")
            if ga == "female":
                return ("chị", "em")
            if ga == "male":
                return ("anh", "em")
            return ("tôi", "cậu")
        if gb == "male" and ga == "female":
            return ("em", "anh")
        if gb == "female" and ga == "male":
            return ("em", "chị")
        return ("cậu", "tôi")

    # Vulgar / Street intimacy partition (informal male peers)
    if vec.register < 0.25 and vec.affinity >= 0.3 and abs(vec.delta_hierarchy) <= 0.2 and (ga == "male" or gb == "male"):
        return ("tao", "mày")

    # Close friends
    if vec.affinity >= 0.4 and vec.register < 0.5:
        return ("tớ", "cậu") if vec.delta_hierarchy >= 0 else ("cậu", "tớ")

    # High register / Courtly
    if vec.register >= 0.7:
        if vec.delta_hierarchy >= 0:
            return ("tôi", "cô") if gb == "female" else ("tôi", "anh")
        return ("cô", "tôi") if ga == "female" else ("anh", "tôi")

    # Default neutral
    return ("tôi", "bạn") if vec.delta_hierarchy >= 0 else ("bạn", "tôi")


def project_relational_vector(
    R: RelationalVector,
    gender_a: str | Any,
    gender_b: str | Any,
) -> tuple[str, str]:
    """Deterministically project 4D relational vector to natural Vietnamese pronouns.

    Algebraic Reciprocal Invariant Closure:
        Pi(R.inverted(), gender_b, gender_a) == (p2, p1)
        whenever (p1, p2) = Pi(R, gender_a, gender_b) for asymmetric states,
        and Pi(R.inverted(), gender_b, gender_a) == (p1, p2) for symmetric states.
    """
    ga = (gender_a.value if hasattr(gender_a, "value") else str(gender_a or "unknown")).strip().lower()
    gb = (gender_b.value if hasattr(gender_b, "value") else str(gender_b or "unknown")).strip().lower()

    if R.delta_hierarchy < 0:
        inv_vec = R.inverted()
        p1, p2 = _project_canonical(inv_vec, gb, ga)
        return (p2, p1)

    return _project_canonical(R, ga, gb)


__all__ = [
    "KinshipKind",
    "RelationalVector",
    "project_relational_vector",
]
