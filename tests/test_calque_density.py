"""Tests for reference-free syntactic calque density scoring."""

from __future__ import annotations

from renslate.engine.qa.calque_density import compute_calque_density


def test_calque_density_flags_stiff_formulaic_sentences() -> None:
    source = "Well, I was told by him that you failed."
    stiff_target = "Chà, tôi đã bị nói bởi anh ấy rằng bạn đã thất bại."
    score_stiff = compute_calque_density(source, stiff_target)

    natural_target = "Anh ấy có bảo với tôi là cậu trượt rồi đấy."
    score_natural = compute_calque_density(source, natural_target)

    assert score_stiff > 0.4
    assert score_natural < 0.15
    assert score_stiff > score_natural


def test_calque_density_does_not_penalize_legitimate_passives() -> None:
    cases = [
        ("He was injured in the battle.", "Anh ấy bị thương trong trận chiến."),
        ("He was rewarded for his bravery.", "Cậu ấy được thưởng một chuyến du lịch."),
        ("Students are punished for being late.", "Học sinh bị phạt vì đi học muộn."),
        ("Farmers had a good harvest this year.", "Năm nay bà con được mùa."),
    ]
    for src, tgt in cases:
        score = compute_calque_density(src, tgt)
        assert score < 0.2, f"Legitimate passive penalized too high ({score}): {tgt}"


def test_calque_density_detects_formulaic_starters() -> None:
    starters = [
        ("Well, that is complicated.", "Chà, chuyện này khó đấy."),
        ("Well, let me see.", "Chà... để tôi xem nào."),
        ("You see, we have no time.", "Bạn thấy đấy, chúng ta không có nhiều thời gian."),
        ("As you know, exams start tomorrow.", "Như bạn biết, kỳ thi sắp bắt đầu."),
        ("In other words, the plan changed.", "Nói cách khác, kế hoạch đã thay đổi."),
    ]
    for src, tgt in starters:
        score = compute_calque_density(src, tgt)
        assert score >= 0.2, f"Starter not flagged with sufficient penalty ({score}): {tgt}"

    natural_starters = [
        ("Well, that is complicated.", "Chuyện này hơi khó đấy."),
        ("You see, we have no time.", "Chúng mình không còn nhiều thời gian đâu."),
        ("As you know, exams start tomorrow.", "Mai là thi rồi đấy."),
        ("In other words, the plan changed.", "Vậy là kế hoạch đổi rồi."),
    ]
    for src, tgt in natural_starters:
        score = compute_calque_density(src, tgt)
        assert score < 0.15, f"Natural translation falsely flagged ({score}): {tgt}"


def test_calque_density_detects_agentive_passive_calques() -> None:
    passive_cases = [
        (
            "The report was approved by the manager.",
            "Bản báo cáo đã được duyệt bởi người quản lý.",
        ),
        (
            "I was surprised by your arrival.",
            "Tôi đã bị làm cho ngạc nhiên bởi sự đến của bạn.",
        ),
        (
            "The secret was revealed by her.",
            "Bí mật đã bị tiết lộ bởi cô ấy.",
        ),
    ]
    for src, tgt in passive_cases:
        score = compute_calque_density(src, tgt)
        assert score >= 0.35, f"Agentive passive calque not flagged ({score}): {tgt}"


def test_calque_density_does_not_flag_boi_vi_conjunction() -> None:
    src = "He was reprimanded because he was late."
    tgt = "Anh ấy bị khiển trách bởi vì đến trễ."
    score = compute_calque_density(src, tgt)
    assert score < 0.2, f"'Bởi vì' conjunction falsely treated as agentive 'bởi' ({score}): {tgt}"


def test_calque_density_detects_excessive_possessive_density() -> None:
    src = "This is the car of the brother of my friend."
    tgt = "Đây là xe của anh trai của bạn của tôi."
    score = compute_calque_density(src, tgt)
    assert score >= 0.35, f"Excessive possessive 'của' density not flagged ({score}): {tgt}"

    src_natural = "This is my choice."
    tgt_natural = "Đây là lựa chọn của tôi."
    score_natural = compute_calque_density(src_natural, tgt_natural)
    assert score_natural < 0.15, (
        f"Natural single possessive falsely flagged ({score_natural}): {tgt_natural}"
    )


def test_calque_density_edge_cases_and_immutability() -> None:
    assert compute_calque_density("", "") == 0.0
    assert compute_calque_density("   ", "   \t\n") == 0.0
    assert compute_calque_density("I agree.", "Tôi đồng ý.") == 0.0

    original_tgt = "Chà, tôi đã bị nói bởi anh ấy."
    tgt_copy = str(original_tgt)
    score = compute_calque_density("Well, I was told by him.", original_tgt)
    assert 0.0 <= score <= 1.0
    assert original_tgt == tgt_copy
