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
from collections import Counter
from dataclasses import dataclass

from vitrans.errors import MarkupError

OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"

_PLACEHOLDER = re.compile(rf"{OPEN_MARK}(/?)([A-Z][0-9]+){CLOSE_MARK}")
_PERCENT_VARIABLE = re.compile(r"%\([A-Za-z_]\w*\)[sdifrg]")
_TAG_NAME = re.compile(r"\{/?(#|[A-Za-z0-9_]+)")


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


_SYMMETRIC_STYLE_TAGS = frozenset({"i", "b", "u", "s", "plain"})


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
        raw = token.raw
        if raw == "{/}":
            if stack:
                pairs[stack[-1][1]] = index
                stack.pop()
            continue

        name = _tag_name(raw)
        if raw.startswith("{/"):
            for depth in range(len(stack) - 1, -1, -1):
                if stack[depth][0] == name:
                    pairs[stack[depth][1]] = index
                    del stack[depth]
                    break
        elif name in _SYMMETRIC_STYLE_TAGS and stack and stack[-1][0] == name:
            pairs[stack[-1][1]] = index
            stack.pop()
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


def auto_repair_placeholders(masked: MaskedText, translated: str) -> str:
    """Auto-repair minor placeholder formatting imperfections from LLM outputs.

    Heals:
    1. Spacing inside placeholder delimiters: e.g. '⟦ V1 ⟧' or '⟦ /T1 ⟧' -> '⟦V1⟧', '⟦/T1⟧'.
    2. Bracket alterations: e.g. '[V1]', '{V1}', or '【V1】' when token key is from masked source.
    3. Converted newlines: converts literal '\\n' or '\n' back to missing '⟦LX⟧' tokens.
    4. Unclosed paired tags: appends missing '⟦/TX⟧' when opening '⟦TX⟧' is present.
    """
    if not masked.tokens:
        return translated

    result = translated

    # 1. Normalize spaces inside standard placeholder marks ⟦ ... ⟧
    result = re.sub(
        rf"{OPEN_MARK}\s*(/?)\s*([A-Za-z][0-9]+)\s*{CLOSE_MARK}",
        rf"{OPEN_MARK}\1\2{CLOSE_MARK}",
        result,
    )

    # 2. Repair bracket variants if model mutated ⟦...⟧ into [...] or {...} or 【...】
    for token in masked.tokens:
        k = token.key
        result = re.sub(
            rf"[\[\{{\u3010\u3014\uff3b]\s*{re.escape(k)}\s*[\]\}}\u3011\u3015\uff3d]",
            f"{OPEN_MARK}{k}{CLOSE_MARK}",
            result,
        )
        if token.close_raw is not None:
            result = re.sub(
                rf"[\[\{{\u3010\u3014\uff3b]\s*/{re.escape(k)}\s*[\]\}}\u3011\u3015\uff3d]",
                f"{OPEN_MARK}/{k}{CLOSE_MARK}",
                result,
            )

    # 3. Restore newline tokens (⟦LX⟧) if converted to literal newlines or stripped
    for token in masked.tokens:
        if token.kind is TokenKind.NEWLINE:
            nl_mark = f"{OPEN_MARK}{token.key}{CLOSE_MARK}"
            if nl_mark not in result:
                if "\n" in result:
                    result = result.replace("\n", nl_mark, 1)
                elif "\\n" in result:
                    result = result.replace("\\n", nl_mark, 1)
                else:
                    result = result + nl_mark

    # 4. Auto-close unclosed paired tags (e.g. ⟦T1⟧ ... without ⟦/T1⟧)
    for token in masked.tokens:
        if token.close_raw is not None:
            open_mark = f"{OPEN_MARK}{token.key}{CLOSE_MARK}"
            close_mark = f"{OPEN_MARK}/{token.key}{CLOSE_MARK}"
            if open_mark in result and close_mark not in result:
                result = result + close_mark

    # 5. Restore trailing tags from masked source if omitted by LLM (e.g. {nw}, {fast}, {w})
    for token in masked.tokens:
        if token.close_raw is None:
            mark = f"{OPEN_MARK}{token.key}{CLOSE_MARK}"
            if mark not in result and masked.text.rstrip().endswith(mark):
                result = result + mark

    # 6. Restore leading tags from masked source if omitted by LLM (e.g. {cps=...})
    for token in masked.tokens:
        if token.close_raw is None:
            mark = f"{OPEN_MARK}{token.key}{CLOSE_MARK}"
            if mark not in result and masked.text.lstrip().startswith(mark):
                result = mark + result

    return result


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


_STANDALONE_TAGS = frozenset({"w", "p", "nw", "fast", "space", "vspace", "image", "done"})


def _is_openable_tag(token: Token, tokens: list[Token], index: int) -> bool:
    if token.kind is not TokenKind.TAG:
        return False
    raw = token.raw
    if raw.startswith("{/") or raw.endswith("/}") or raw == "{/}":
        return False
    name = _tag_name(raw)
    if name in _STANDALONE_TAGS:
        return False
    for j in range(index + 1, len(tokens)):
        t2 = tokens[j]
        if t2.kind is TokenKind.TAG:
            if t2.raw == f"{{/{name}}}" or t2.raw == "{/}":
                return True
            if (
                name in _SYMMETRIC_STYLE_TAGS
                and _tag_name(t2.raw) == name
                and not t2.raw.startswith("{/")
            ):
                return True
    return False


def validate_raw(source: str, target: str) -> list[PlaceholderIssue]:
    """Check that a target string preserves all Ren'Py markup and variables from source."""
    issues: list[PlaceholderIssue] = []

    markup_err = find_markup_error(target)
    if markup_err is not None:
        issues.append(PlaceholderIssue("bad_markup", markup_err))
        return issues

    try:
        src_parts = tokenize(source)
        tgt_parts = tokenize(target)
    except MarkupError as error:
        issues.append(PlaceholderIssue("bad_markup", str(error)))
        return issues

    src_tokens = [p for p in src_parts if isinstance(p, Token)]
    tgt_tokens = [p for p in tgt_parts if isinstance(p, Token)]

    # 1. Variables
    src_vars = Counter(t.raw for t in src_tokens if t.kind is TokenKind.VARIABLE)
    tgt_vars = Counter(t.raw for t in tgt_tokens if t.kind is TokenKind.VARIABLE)

    for var, count in src_vars.items():
        tgt_count = tgt_vars.get(var, 0)
        if tgt_count < count:
            issues.append(PlaceholderIssue("missing_placeholder", var))
        elif tgt_count > count:
            issues.append(PlaceholderIssue("duplicate_placeholder", var))

    for var in tgt_vars:
        if var not in src_vars:
            issues.append(PlaceholderIssue("unknown_placeholder", var))

    # 2. Tags & Ordering (Semantic Matching with Strict LIFO)
    src_openers: list[str] = []
    src_standalone: list[str] = []
    tgt_openers: list[str] = []
    tgt_standalone: list[str] = []

    # Target: verify tag pairing and enforce strict LIFO
    tgt_stack: list[tuple[str, str]] = []
    for index, token in enumerate(tgt_tokens):
        if token.kind is not TokenKind.TAG:
            continue
        raw = token.raw
        name = _tag_name(raw)
        if raw == "{/}":
            if tgt_stack:
                tgt_openers.append(tgt_stack.pop()[1])
            else:
                issues.append(PlaceholderIssue("tag_order", raw))
        elif raw.startswith("{/"):
            if tgt_stack and tgt_stack[-1][0] == name:
                tgt_openers.append(tgt_stack.pop()[1])
            else:
                issues.append(PlaceholderIssue("tag_order", raw))
        elif name in _SYMMETRIC_STYLE_TAGS and tgt_stack and tgt_stack[-1][0] == name:
            tgt_openers.append(tgt_stack.pop()[1])
        elif _is_openable_tag(token, tgt_tokens, index):
            tgt_stack.append((name, raw))
        else:
            tgt_standalone.append(raw)

    for _, raw in tgt_stack:
        tgt_standalone.append(raw)

    # Source: extract paired openers and standalone tags
    src_stack: list[tuple[str, str]] = []
    for index, token in enumerate(src_tokens):
        if token.kind is not TokenKind.TAG:
            continue
        raw = token.raw
        name = _tag_name(raw)
        if raw == "{/}":
            if src_stack:
                src_openers.append(src_stack.pop()[1])
        elif raw.startswith("{/"):
            for depth in range(len(src_stack) - 1, -1, -1):
                if src_stack[depth][0] == name:
                    src_openers.append(src_stack.pop(depth)[1])
                    break
        elif name in _SYMMETRIC_STYLE_TAGS and src_stack and src_stack[-1][0] == name:
            src_openers.append(src_stack.pop()[1])
        elif _is_openable_tag(token, src_tokens, index):
            src_stack.append((name, raw))
        else:
            src_standalone.append(raw)

    for _, raw in src_stack:
        src_standalone.append(raw)

    # Compare paired openers (e.g. {i}, {color=#fff})
    src_op_c = Counter(src_openers)
    tgt_op_c = Counter(tgt_openers)
    for op, count in src_op_c.items():
        if tgt_op_c.get(op, 0) < count:
            issues.append(PlaceholderIssue("missing_placeholder", op))
        elif tgt_op_c.get(op, 0) > count:
            issues.append(PlaceholderIssue("duplicate_placeholder", op))
    for op in tgt_op_c:
        if op not in src_op_c:
            issues.append(PlaceholderIssue("unknown_placeholder", op))

    # Compare standalone tags (e.g. {nw}, {w})
    src_st_c = Counter(src_standalone)
    tgt_st_c = Counter(tgt_standalone)
    for st, count in src_st_c.items():
        if tgt_st_c.get(st, 0) < count:
            issues.append(PlaceholderIssue("missing_placeholder", st))
        elif tgt_st_c.get(st, 0) > count:
            issues.append(PlaceholderIssue("duplicate_placeholder", st))
    for st in tgt_st_c:
        if st not in src_st_c:
            issues.append(PlaceholderIssue("unknown_placeholder", st))

    # 4. Escapes
    src_escapes = Counter(t.raw for t in src_tokens if t.kind is TokenKind.ESCAPE)
    tgt_escapes = Counter(t.raw for t in tgt_tokens if t.kind is TokenKind.ESCAPE)
    for esc, count in src_escapes.items():
        if tgt_escapes.get(esc, 0) < count:
            issues.append(PlaceholderIssue("missing_placeholder", esc))

    # 5. Newline preservation if present
    src_newlines = sum(1 for t in src_tokens if t.kind is TokenKind.NEWLINE)
    tgt_newlines = sum(1 for t in tgt_tokens if t.kind is TokenKind.NEWLINE)
    if src_newlines > 0 and tgt_newlines < src_newlines:
        issues.append(PlaceholderIssue("missing_placeholder", "\\n"))

    return issues


def reconcile_translation(source: str, target: str) -> tuple[str, list[PlaceholderIssue]]:
    """Validate and unmask (if needed) target translation against source.

    Supports both synthetic placeholder tokens (e.g. ⟦V1⟧) and natural Ren'Py markup.
    Returns (cleaned_target, issues).
    """
    if OPEN_MARK in target:
        masked = mask(source)
        issues = validate(masked, target)
        if issues:
            return target, issues
        try:
            return unmask(masked, target), []
        except (PlaceholderError, MarkupError) as err:
            return target, [PlaceholderIssue("unmask_error", str(err))]
    else:
        issues = validate_raw(source, target)
        return target, issues
