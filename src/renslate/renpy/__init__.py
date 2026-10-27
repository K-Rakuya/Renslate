"""Ren'Py AST parsing, lexical codecs, dialogue tag masking, extraction, and code generation."""

from __future__ import annotations

# Submodules
from renslate.renpy import (
    builtin_ui,
    extract,
    literals,
    mask,
    parser,
    rpyc,
    sdk,
    writer,
)
from renslate.renpy.builtin_ui import (
    BUILTIN_UI_SEEDS,
    get_builtin_ui_translation,
)
from renslate.renpy.extract import (
    FLAG_MARKUP_INVALID,
    FLAG_SOURCE_CHANGED,
    ExtractReport,
    extract_project,
    scan_project,
)
from renslate.renpy.literals import (
    decode_say_text,
    encode_say_text,
    parse_python_string,
    quote_python_text,
)
from renslate.renpy.mask import (
    CLOSE_MARK,
    OPEN_MARK,
    MaskedText,
    MaskToken,
    PlaceholderError,
    PlaceholderIssue,
    Token,
    TokenKind,
    auto_repair_placeholders,
    find_markup_error,
    reconcile_translation,
    tokenize,
    unmask,
    validate,
    validate_raw,
)
from renslate.renpy.parser import (
    Issue,
    ParsedUnit,
    SayStatement,
    Severity,
    TranslationScanner,
    fingerprint,
    parse_say,
    parse_string_line,
    string_uid,
)
from renslate.renpy.rpyc import (
    decompile_rpyc,
    extract_character_definitions_from_rpyc,
    extract_dialogue_from_rpyc,
    synthesize_translation_skeletons,
)
from renslate.renpy.sdk import (
    Runner,
    build_command,
    find_launcher,
    generate_translations,
)
from renslate.renpy.writer import (
    ApplyReport,
    apply_translations,
    setup_vietnamese_support,
)

__all__ = [
    "BUILTIN_UI_SEEDS",
    "CLOSE_MARK",
    "FLAG_MARKUP_INVALID",
    "FLAG_SOURCE_CHANGED",
    "OPEN_MARK",
    "ApplyReport",
    "ExtractReport",
    "Issue",
    "MaskToken",
    "MaskedText",
    "ParsedUnit",
    "PlaceholderError",
    "PlaceholderIssue",
    "Runner",
    "SayStatement",
    "Severity",
    "Token",
    "TokenKind",
    "TranslationScanner",
    "apply_translations",
    "auto_repair_placeholders",
    "build_command",
    "builtin_ui",
    "decode_say_text",
    "decompile_rpyc",
    "encode_say_text",
    "extract",
    "extract_character_definitions_from_rpyc",
    "extract_dialogue_from_rpyc",
    "extract_project",
    "find_launcher",
    "find_markup_error",
    "fingerprint",
    "generate_translations",
    "get_builtin_ui_translation",
    "literals",
    "mask",
    "parse_python_string",
    "parse_say",
    "parse_string_line",
    "parser",
    "quote_python_text",
    "reconcile_translation",
    "rpyc",
    "scan_project",
    "sdk",
    "setup_vietnamese_support",
    "string_uid",
    "synthesize_translation_skeletons",
    "tokenize",
    "unmask",
    "validate",
    "validate_raw",
    "writer",
]
