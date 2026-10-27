"""Exception hierarchy. Every error raised on purpose derives from RenslateError."""

from __future__ import annotations


class RenslateError(Exception):
    """Base class for all expected, user-reportable failures."""


class ConfigError(RenslateError):
    """The configuration file is missing, malformed or contains invalid values."""


class ProjectError(RenslateError):
    """The project layout or its database is missing or incompatible."""


class MarkupError(RenslateError):
    """A text contains malformed Ren'Py markup (unterminated tag or interpolation)."""


class ApplyError(RenslateError):
    """Translations cannot be written safely to the game files."""


class SdkError(RenslateError):
    """The Ren'Py SDK is missing or the translate command failed."""


class LLMError(RenslateError):
    """Failure communicating with or parsing response from the LLM backend."""


class TranslationError(RenslateError):
    """Failure occurred during the translation pipeline."""


class QAError(RenslateError):
    """Failure occurred during quality assurance verification."""
