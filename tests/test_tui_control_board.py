"""Unit tests for Interactive TUI Control Board state machine and rendering."""

from __future__ import annotations

from renslate.ui.control_board import (
    ControlBoardState,
    ControlField,
    OptionType,
    create_default_control_fields,
)


def test_control_field_boolean_toggle() -> None:
    field = ControlField(
        key="deliberate",
        label="2-Pass Deliberative Reflector",
        field_type=OptionType.BOOLEAN,
        value=True,
    )
    assert field.value is True
    assert "[● BẬT]" in field.format_display_value()

    field.toggle()
    assert field.value is False
    assert "[○ TẮT]" in field.format_display_value()

    field.toggle()
    assert field.value is True


def test_control_field_number_range_stepping() -> None:
    field = ControlField(
        key="calque_threshold",
        label="Ngưỡng Calque kích hoạt",
        field_type=OptionType.NUMBER_RANGE,
        value=0.35,
        min_val=0.20,
        max_val=0.60,
        step=0.05,
    )
    # Step next
    field.step_next()
    assert round(field.value, 2) == 0.40

    # Step prev twice
    field.step_prev()
    field.step_prev()
    assert round(field.value, 2) == 0.30

    # Clamping at min_val
    for _ in range(10):
        field.step_prev()
    assert round(field.value, 2) == 0.20

    # Clamping at max_val
    for _ in range(20):
        field.step_next()
    assert round(field.value, 2) == 0.60


def test_control_field_choice_cycling() -> None:
    field = ControlField(
        key="filter_mode",
        label="Bộ lọc dòng thoại",
        field_type=OptionType.CHOICE,
        value="all",
        options=["all", "dialogue", "flagged"],
    )
    assert field.value == "all"
    field.step_next()
    assert field.value == "dialogue"
    field.step_next()
    assert field.value == "flagged"
    # Wrap-around
    field.step_next()
    assert field.value == "all"
    field.step_prev()
    assert field.value == "flagged"


def test_control_board_state_navigation_and_export() -> None:
    fields = create_default_control_fields()
    state = ControlBoardState(fields=fields)

    assert state.active_idx == 0
    first_key = state.current_field.key

    state.move_down()
    assert state.active_idx == 1
    assert state.current_field.key != first_key

    state.move_up()
    assert state.active_idx == 0

    # Test top clamp or wrap
    state.move_up()
    assert state.active_idx == len(fields) - 1

    state.move_down()
    assert state.active_idx == 0

    cfg_dict = state.to_config_dict()
    assert "deliberate" in cfg_dict
    assert "calque_threshold" in cfg_dict
    assert "batch_size" in cfg_dict
    assert "exemplars" in cfg_dict
    assert "narrative_memory" in cfg_dict


def test_render_control_board_layout() -> None:
    from renslate.ui.control_board import render_control_board

    fields = create_default_control_fields()
    state = ControlBoardState(fields=fields)

    rendered = render_control_board(
        state,
        project_name="CommercialVN-Beta",
        progress_info="1,420 / 3,850 dòng (36.8%)",
        width=84,
        plain=True,
    )
    assert "CommercialVN-Beta" in rendered
    assert "BẢNG ĐIỀU KHIỂN CẤU HÌNH" in rendered
    assert "2-Pass Deliberative Reflector" in rendered
    assert "THÔNG TIN & KHUYẾN NGHỊ" in rendered
    assert "Deliberative" in rendered
    assert "[ENTER] Bắt đầu Dịch" in rendered


def test_run_control_board_event_loop_mocked() -> None:
    from unittest.mock import MagicMock

    from renslate.ui.control_board import run_control_board

    project = MagicMock()
    project.root.name = "TestVN"

    # Simulate: Move DOWN, press SPACE (toggle calque or second option), then ENTER
    keys = ["DOWN", "SPACE", "ENTER"]

    def mock_reader() -> str:
        return keys.pop(0) if keys else "ENTER"

    cfg, action = run_control_board(project, key_reader=mock_reader)
    assert action == "START"
    assert isinstance(cfg, dict)
    assert "deliberate" in cfg


