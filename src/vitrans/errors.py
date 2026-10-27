"""Exception hierarchy. Every error raised on purpose derives from VitransError."""

from __future__ import annotations


class VitransError(Exception):
    """Base class for all expected, user-reportable failures."""


class ConfigError(VitransError):
    """The configuration file is missing, malformed or contains invalid values."""


class ProjectError(VitransError):
    """The project layout or its database is missing or incompatible."""


class MarkupError(VitransError):
    """A text contains malformed Ren'Py markup (unterminated tag or interpolation)."""


class ApplyError(VitransError):
    """Translations cannot be written safely to the game files."""


class SdkError(VitransError):
    """The Ren'Py SDK is missing or the translate command failed."""


class LLMError(VitransError):
    """Failure communicating with or parsing response from the LLM backend."""


class TranslationError(VitransError):
    """Failure occurred during the translation pipeline."""


class QAError(VitransError):
    """Failure occurred during quality assurance verification."""
