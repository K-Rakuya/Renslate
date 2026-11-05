"""Unit tests for Real-Time Full Translation Studio and In-Place Editor."""

from __future__ import annotations

from pathlib import Path

from renslate.ui.studio import (
    LiveDialogueItem,
    StudioState,
    render_dialogue_card,
    wrap_text_to_lines,
)


def test_wrap_text_to_lines_preserves_full_content_without_ellipsis() -> None:
    text = (
        "Nhưng em gái tôi vẫn còn đang bị mắc kẹt bên trong gian phòng thánh, "
        "tôi tuyệt đối không thể bỏ mặc muội muội một mình ở nơi nguy hiểm như thế này!"
    )
    # Wrap at 40 chars width
    lines = wrap_text_to_lines(text, max_width=40)
    assert len(lines) >= 3
    # Verify 100% full content preserved, NO '...' truncation!
    reconstructed = " ".join(lines)
    assert "tuyệt đối không thể" in reconstructed
    assert "nguy hiểm như thế này!" in reconstructed
    assert "..." not in lines[-1]


def test_live_dialogue_item_state_and_filter() -> None:
    state = StudioState()
    u1 = LiveDialogueItem(
        uid="d:1",
        seq=1,
        speaker="hero",
        listener="sage",
        source="Master, what should we do?",
        target="Thưa sư phụ, chúng ta nên làm gì?",
        status="translated",
        origin="llm",
        calque_score=0.10,
    )
    u2 = LiveDialogueItem(
        uid="d:2",
        seq=2,
        speaker="sage",
        listener="hero",
        source="Stay calm.",
        target="Hãy bình tĩnh.",
        status="qa_flagged",
        origin="reflected",
        calque_score=0.45,
        qa_issues=("pronoun_warning",),
    )
    state.add_or_update_unit(u1)
    state.add_or_update_unit(u2)

    assert len(state.units) == 2
    assert state.selected_unit() == u1

    # Navigation
    state.move_down()
    assert state.selected_unit() == u2
    state.move_down()  # clamp at bottom
    assert state.selected_unit() == u2
    state.move_up()
    assert state.selected_unit() == u1

    # Filtering
    state.filter_mode = "flagged"
    filtered = state.filtered_units()
    assert len(filtered) == 1
    assert filtered[0].uid == "d:2"

    state.filter_mode = "reflected"
    filtered = state.filtered_units()
    assert len(filtered) == 1
    assert filtered[0].uid == "d:2"


def test_render_dialogue_card_full_display() -> None:
    item = LiveDialogueItem(
        uid="d:scene1_0042",
        seq=142,
        speaker="hero",
        listener="sage",
        source="Master, the eastern gate has fallen! What should we do now?",
        target="Thưa sư phụ, cổng đông đã thất thủ rồi! Chúng ta phải làm gì bây giờ?",
        status="translated",
        origin="llm",
        calque_score=0.12,
    )
    rendered_lines = render_dialogue_card(item, width=80, is_selected=True, plain=True)
    card_text = "\n".join(rendered_lines)

    assert "hero" in card_text
    assert "sage" in card_text
    assert "d:scene1_0042" in card_text
    assert "Master, the eastern gate has fallen!" in card_text
    assert "Thưa sư phụ, cổng đông đã thất thủ rồi!" in card_text
    assert "..." not in card_text


def test_commit_in_place_edit_updates_db_and_tm(tmp_path: Path) -> None:
    from renslate.core.db import Database
    from renslate.core.models import Origin, Status, Unit, UnitKind
    from renslate.ui.studio import commit_in_place_edit

    db_path = tmp_path / "test.db"
    with Database.open(db_path, create=True) as db:
        unit = Unit(
            uid="d:100",
            kind=UnitKind.DIALOGUE,
            seq=100,
            source="Look out, Eileen!",
            speaker="mc",
            status=Status.TRANSLATED,
            target="Coi chừng, Eileen!",
        )
        db.save_unit(unit)

        # 1. Edit with valid new Vietnamese translation
        success, _msg, issues = commit_in_place_edit(
            db,
            unit_uid="d:100",
            new_target="Cẩn thận đấy, Eileen!",
        )
        assert success is True
        assert not issues

        # Verify DB updated
        saved_u = db.get_unit("d:100")
        assert saved_u is not None
        assert saved_u.target == "Cẩn thận đấy, Eileen!"
        assert saved_u.origin == Origin.USER
        assert saved_u.status == Status.REVIEWED

        # Verify TM updated
        tm_entry = db.get_tm("Look out, Eileen!")
        assert tm_entry is not None
        assert tm_entry.target == "Cẩn thận đấy, Eileen!"

        # 2. Edit with invalid placeholder syntax -> rejected safely
        unit_var = Unit(
            uid="d:101",
            kind=UnitKind.DIALOGUE,
            seq=101,
            source="Hello [player]!",
            speaker="eileen",
            status=Status.TRANSLATED,
            target="Chào [player]!",
        )
        db.save_unit(unit_var)

        bad_success, _bad_msg, bad_issues = commit_in_place_edit(
            db,
            unit_uid="d:101",
            new_target="Chào bạn!",  # missing [player]
        )
        assert bad_success is False
        assert len(bad_issues) > 0


def test_render_edit_modal_formatting() -> None:
    from renslate.ui.studio import render_edit_modal

    item = LiveDialogueItem(
        uid="d:test",
        seq=1,
        speaker="mc",
        source="Watch out!",
        target="Coi chừng!",
    )
    modal_lines = render_edit_modal(item, current_input="Cẩn thận!", width=70, plain=True)
    modal_text = "\n".join(modal_lines)

    assert "CHỈNH SỬA BẢN DỊCH TẠI CHỖ" in modal_text
    assert "d:test" in modal_text
    assert "Watch out!" in modal_text
    assert "Cẩn thận!" in modal_text

    assert "..." not in modal_text


def test_render_studio_screen_full() -> None:
    from renslate.ui.studio import render_studio_screen

    state = StudioState()
    state.add_or_update_unit(
        LiveDialogueItem(
            uid="d:101",
            seq=101,
            speaker="hero",
            listener="sage",
            source="We must protect the villagers!",
            target="Chúng ta phải bảo vệ dân làng!",
            status="translated",
            origin="llm",
            calque_score=0.15,
        )
    )

    screen = render_studio_screen(
        state,
        header_info="Lô #5/20 • Tốc độ: 5.1 dòng/s",
        width=80,
        plain=True,
    )
    assert "RENSLATE LIVE TRANSLATION STUDIO" in screen
    assert "Lô #5/20" in screen
    assert "hero" in screen
    assert "sage" in screen
    assert "We must protect the villagers!" in screen
    assert "Chúng ta phải bảo vệ dân làng!" in screen
    assert "[E] Sửa bản dịch" in screen

