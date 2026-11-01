"""Backward compatibility shim forwarding to renslate.ui.wizard package."""

from __future__ import annotations

from renslate.ui.wizard import (
    DEFAULT_TRANSITIONS,
    GameInspectionReport,
    GameInspector,
    InteractiveWizard,
    RenpyVersion,
    ScriptMode,
    WizardContext,
    WizardEvent,
    WizardPhase,
    WizardState,
    inspect_game,
    read_single_key,
    render_inspection_card,
    run_interactive_wizard,
    run_wizard,
    select_menu,
)

__all__ = [
    "DEFAULT_TRANSITIONS",
    "GameInspectionReport",
    "GameInspector",
    "InteractiveWizard",
    "RenpyVersion",
    "ScriptMode",
    "WizardContext",
    "WizardEvent",
    "WizardPhase",
    "WizardState",
    "inspect_game",
    "read_single_key",
    "render_inspection_card",
    "run_interactive_wizard",
    "run_wizard",
    "select_menu",
]
