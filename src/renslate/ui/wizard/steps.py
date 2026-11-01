"""Phase definitions, state transitions, and execution context for the interactive wizard."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from renslate.core.config import AppConfig, Config
from renslate.core.project import Project

if TYPE_CHECKING:
    from renslate.ui.wizard.config_flow import GameInspectionReport


class WizardPhase(enum.Enum):
    """The 6 phases of the localization studio state machine."""

    INIT = "INIT"
    DETECT_GAME = "DETECT_GAME"
    EXTRACT_OR_REPAIR = "EXTRACT_OR_REPAIR"
    LLM_PROBE = "LLM_PROBE"
    TRANSLATE_DASHBOARD = "TRANSLATE_DASHBOARD"
    APPLY_VERIFY = "APPLY_VERIFY"
    EXIT = "EXIT"


WizardState = WizardPhase


class WizardEvent(enum.Enum):
    """Event triggers transitioning the wizard state machine."""

    PROCEED = "PROCEED"
    BACK = "BACK"
    RETRY = "RETRY"
    CANCEL = "CANCEL"
    CUSTOM = "CUSTOM"


DEFAULT_TRANSITIONS: dict[WizardPhase, dict[WizardEvent, WizardPhase]] = {
    WizardPhase.INIT: {
        WizardEvent.PROCEED: WizardPhase.DETECT_GAME,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.DETECT_GAME: {
        WizardEvent.PROCEED: WizardPhase.EXTRACT_OR_REPAIR,
        WizardEvent.BACK: WizardPhase.INIT,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.EXTRACT_OR_REPAIR: {
        WizardEvent.PROCEED: WizardPhase.LLM_PROBE,
        WizardEvent.BACK: WizardPhase.DETECT_GAME,
        WizardEvent.RETRY: WizardPhase.EXTRACT_OR_REPAIR,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.LLM_PROBE: {
        WizardEvent.PROCEED: WizardPhase.TRANSLATE_DASHBOARD,
        WizardEvent.BACK: WizardPhase.EXTRACT_OR_REPAIR,
        WizardEvent.RETRY: WizardPhase.LLM_PROBE,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.TRANSLATE_DASHBOARD: {
        WizardEvent.PROCEED: WizardPhase.APPLY_VERIFY,
        WizardEvent.BACK: WizardPhase.LLM_PROBE,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
    WizardPhase.APPLY_VERIFY: {
        WizardEvent.PROCEED: WizardPhase.EXIT,
        WizardEvent.BACK: WizardPhase.TRANSLATE_DASHBOARD,
        WizardEvent.CANCEL: WizardPhase.EXIT,
    },
}


@dataclass
class WizardContext:
    """Mutable runtime state passed across wizard phases."""

    root: Path
    project: Project | None = None
    config: Config = field(default_factory=Config)
    inspection: GameInspectionReport | None = None
    server_url: str = field(default_factory=lambda: AppConfig.server_base_url)
    detected_model: str = ""
    chunk_size: int = 100
    retry_failed: bool = False
    auto_apply: bool = True
    dry_run: bool = False
    extracted_count: int = 0
    qa_issues_count: int = 0
