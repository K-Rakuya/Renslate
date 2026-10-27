"""Protection of Ren'Py markup while text travels through a language model.

Interpolations (``[name]``), text tags (``{i}``), escaped brackets (``[[``, ``{{``), percent
sequences and line breaks are replaced by opaque placeholders such as ``⟦V1⟧`` before
translation and restored afterwards. Restoration is validated: every placeholder must come
back exactly once and paired tags must stay ordered, so a damaged translation is rejected
instead of silently breaking the game script.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

from vitrans.errors import MarkupError

OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"

_PLACEHOLDER = re.compile(rf"{OPEN_MARK}(/?)([A-Z][0-9]+){CLOSE_MARK}")
_PERCENT_VARIABLE = re.compile(r"%\([A-Za-z_]\w*\)[sdifrg]")
_TAG_NAME = re.compile(r"\{/?(#|[A-Za-z]+)")


class TokenKind(enum.Enum):
    VARIABLE = "V"
    TAG = "T"
    ESCAPE = "E"
    NEWLINE = "L"


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    raw: str


@dataclass(frozen=True, slots=True)
class MaskToken:
    """One protected element: a placeholder key, its raw text and, for paired tags, the closer."""

    key: str
    kind: TokenKind
    raw: str
    close_raw: str | None = None


@dataclass(frozen=True, slots=True)
class MaskedText:
    text: str
    tokens: tuple[MaskToken, ...]


@dataclass(frozen=True, slots=True)
class PlaceholderIssue:
    code: str
    detail: str


class PlaceholderError(MarkupError):
    """A translation does not preserve the placeholders of its source."""

    def __init__(self, issues: list[PlaceholderIssue]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{issue.code}: {issue.detail}" for issue in issues))


def _skip_quoted(text: str, start: int) -> int:
    quote = text[start]
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    raise MarkupError(f"chuỗi trong markup chưa đóng tại vị trí {start}")


def _scan_interpolation(text: str, start: int) -> int:
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char in "'\"":
            index = _skip_quoted(text, index)
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise MarkupError(f"'[' chưa đóng tại vị trí {start}")


def _scan_tag(text: str, start: int) -> int:
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "[" and not text.startswith("[[", index):
            index = _scan_interpolation(text, index)
            continue
        if char == "}":
            return index + 1
        index += 1
    raise MarkupError(f"'{{' chưa đóng tại vị trí {start}")


def tokenize(text: str) -> list[str | Token]:
    """Split text into plain strings and markup tokens; raise MarkupError if malformed."""
    parts: list[str | Token] = []
    plain: list[str] = []
    index = 0

    def emit(kind: TokenKind, end: int) -> None:
        nonlocal index
        if plain:
            parts.append("".join(plain))
            plain.clear()
        parts.append(Token(kind, text[index:end]))
        index = end

    while index < len(text):
        char = text[index]
        if char == "\n":
            emit(TokenKind.NEWLINE, index + 1)
        elif char == "[":
            if text.startswith("[[", index):
                emit(TokenKind.ESCAPE, index + 2)
            else:
                emit(TokenKind.VARIABLE, _scan_interpolation(text, index))
        elif char == "{":
            if text.startswith("{{", index):
                emit(TokenKind.ESCAPE, index + 2)
            else:
                emit(TokenKind.TAG, _scan_tag(text, index))
        elif char == "%" and text.startswith("%%", index):
            emit(TokenKind.ESCAPE, index + 2)
        elif char == "%" and (match := _PERCENT_VARIABLE.match(text, index)):
            emit(TokenKind.VARIABLE, match.end())
        else:
            plain.append(char)
            index += 1
    if plain:
        parts.append("".join(plain))
    return parts


def find_markup_error(text: str) -> str | None:
    """Return a description of the first markup defect in text, or None if it is well formed."""
    try:
        tokenize(text)
    except MarkupError as error:
        return str(error)
    return None


def _tag_name(raw: str) -> str:
    match = _TAG_NAME.match(raw)
    return match.group(1) if match else raw


def _pair_tags(tokens: list[Token]) -> dict[int, int]:
    """Map the index of each opening tag to the index of its closing tag."""
    pairs: dict[int, int] = {}
    stack: list[tuple[str, int]] = []
    for index, token in enumerate(tokens):
        if token.kind is not TokenKind.TAG:
            continue
        name = _tag_name(token.raw)
        if token.raw.startswith("{/"):
            for depth in range(len(stack) - 1, -1, -1):
                if stack[depth][0] == name:
                    pairs[stack[depth][1]] = index
                    del stack[depth]
                    break
        else:
            stack.append((name, index))
    return pairs


def mask(text: str) -> MaskedText:
    """Replace markup in text with placeholders; raise MarkupError if text is malformed."""
    parts = tokenize(text)
    tokens = [part for part in parts if isinstance(part, Token)]
    pairs = _pair_tags(tokens)
    closers = set(pairs.values())
    keys: dict[int, str] = {}
    counters: dict[TokenKind, int] = {}
    mask_tokens: list[MaskToken] = []

    for index, token in enumerate(tokens):
        if index in closers:
            continue
        counters[token.kind] = counters.get(token.kind, 0) + 1
        key = f"{token.kind.value}{counters[token.kind]}"
        keys[index] = key
        close = tokens[pairs[index]].raw if index in pairs else None
        mask_tokens.append(MaskToken(key, token.kind, token.raw, close))
        if index in pairs:
            keys[pairs[index]] = f"/{key}"

    pieces: list[str] = []
    token_index = 0
    for part in parts:
        if isinstance(part, Token):
            pieces.append(f"{OPEN_MARK}{keys[token_index]}{CLOSE_MARK}")
            token_index += 1
        else:
            pieces.append(part)
    return MaskedText("".join(pieces), tuple(mask_tokens))


def _show(key: str) -> str:
    return f"{OPEN_MARK}{key}{CLOSE_MARK}"


def validate(masked: MaskedText, translated: str) -> list[PlaceholderIssue]:
    """Check that translated text preserves every placeholder of its masked source."""
    expected: dict[str, MaskToken] = {}
    order: dict[str, int] = {}
    for token in masked.tokens:
        expected[token.key] = token
        if token.close_raw is not None:
            expected[f"/{token.key}"] = token

    found: dict[str, list[int]] = {}
    for match in _PLACEHOLDER.finditer(translated):
        key = f"{match.group(1)}{match.group(2)}"
        found.setdefault(key, []).append(match.start())
        order.setdefault(key, match.start())

    issues: list[PlaceholderIssue] = []
    for key in expected:
        count = len(found.get(key, ()))
        if count == 0:
            issues.append(PlaceholderIssue("missing_placeholder", _show(key)))
        elif count > 1:
            issues.append(PlaceholderIssue("duplicate_placeholder", _show(key)))
    for key in found:
        if key not in expected:
            issues.append(PlaceholderIssue("unknown_placeholder", _show(key)))
    for token in masked.tokens:
        closing = f"/{token.key}"
        if closing in order and token.key in order and order[closing] < order[token.key]:
            issues.append(PlaceholderIssue("tag_order", _show(token.key)))

    leftover = _PLACEHOLDER.sub("", translated)
    if OPEN_MARK in leftover or CLOSE_MARK in leftover:
        issues.append(PlaceholderIssue("malformed_placeholder", "dấu phân cách lạc chỗ"))
    return issues


def unmask(masked: MaskedText, translated: str) -> str:
    """Restore markup in translated text; raise PlaceholderError or MarkupError if invalid."""
    issues = validate(masked, translated)
    if issues:
        raise PlaceholderError(issues)

    raw_by_key: dict[str, str] = {}
    for token in masked.tokens:
        raw_by_key[token.key] = token.raw
        if token.close_raw is not None:
            raw_by_key[f"/{token.key}"] = token.close_raw

    restored = _PLACEHOLDER.sub(lambda m: raw_by_key[f"{m.group(1)}{m.group(2)}"], translated)
    error = find_markup_error(restored)
    if error is not None:
        raise MarkupError(error)
    return restored
