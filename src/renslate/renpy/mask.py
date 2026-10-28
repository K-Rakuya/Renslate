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

from renslate.core.errors import MarkupError
from renslate.core.textio import normalize_nfc

OPEN_MARK = "⟦"
CLOSE_MARK = "⟧"

_PLACEHOLDER = re.compile(rf"{OPEN_MARK}(/?)([A-Z][0-9]+){CLOSE_MARK}")
_PERCENT_VARIABLE = re.compile(r"%\([A-Za-z_]\w*\)[sdifrg]")
_TAG_NAME = re.compile(r"\{/?(#|[A-Za-z0-9_]+)")

_REPAIR_DELIMITER_OPEN = r"[\[\{\u27e6\u3010\u3014\uff3b]"
_REPAIR_DELIMITER_CLOSE = r"[\]\}\u27e7\u3011\u3015\uff3d]"
_REPAIR_PLACEHOLDER_RE = re.compile(
    rf"({_REPAIR_DELIMITER_OPEN})\s*(/?)\s*([A-Za-z0-9_]+)\s*({_REPAIR_DELIMITER_CLOSE})"
)
_OUTER_CURLY_RE = re.compile(r"^(?:\{[^\}]+\})+\s*(⟦.+?⟧[\s\S]*?)\s*(?:\{[^\}]+\})+$")
_LEADING_TOKEN_CLUSTER_RE = re.compile(rf"^(?:{OPEN_MARK}[A-Za-z0-9/]+{CLOSE_MARK})+")
_TRAILING_TOKEN_CLUSTER_RE = re.compile(rf"(?:{OPEN_MARK}[A-Za-z0-9/]+{CLOSE_MARK})+$")
_TOKEN_CLUSTER_RE = re.compile(rf"{OPEN_MARK}[A-Za-z0-9/]+{CLOSE_MARK}")
_PREFIX_KEY_RE = re.compile(rf"^{OPEN_MARK}([A-Za-z0-9]+){CLOSE_MARK}")
_SUFFIX_KEY_RE = re.compile(rf"{OPEN_MARK}/([A-Za-z0-9]+){CLOSE_MARK}$")


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
    length = len(text)
    index = 0
    last_text_start = 0

    while index < length:
        char = text[index]
        if char == "\n":
            if index > last_text_start:
                parts.append(text[last_text_start:index])
            parts.append(Token(TokenKind.NEWLINE, "\n"))
            index += 1
            last_text_start = index
        elif char == "[":
            if index + 1 < length and text[index + 1] == "[":
                if index > last_text_start:
                    parts.append(text[last_text_start:index])
                parts.append(Token(TokenKind.ESCAPE, "[["))
                index += 2
                last_text_start = index
            else:
                if index > last_text_start:
                    parts.append(text[last_text_start:index])
                end = _scan_interpolation(text, index)
                parts.append(Token(TokenKind.VARIABLE, text[index:end]))
                index = end
                last_text_start = index
        elif char == "{":
            if index + 1 < length and text[index + 1] == "{":
                if index > last_text_start:
                    parts.append(text[last_text_start:index])
                parts.append(Token(TokenKind.ESCAPE, "{{"))
                index += 2
                last_text_start = index
            else:
                if index > last_text_start:
                    parts.append(text[last_text_start:index])
                end = _scan_tag(text, index)
                parts.append(Token(TokenKind.TAG, text[index:end]))
                index = end
                last_text_start = index
        elif char == "%":
            if index + 1 < length and text[index + 1] == "%":
                if index > last_text_start:
                    parts.append(text[last_text_start:index])
                parts.append(Token(TokenKind.ESCAPE, "%%"))
                index += 2
                last_text_start = index
            elif match := _PERCENT_VARIABLE.match(text, index):
                if index > last_text_start:
                    parts.append(text[last_text_start:index])
                end = match.end()
                parts.append(Token(TokenKind.VARIABLE, text[index:end]))
                index = end
                last_text_start = index
            else:
                index += 1
        else:
            index += 1

    if last_text_start < length:
        parts.append(text[last_text_start:length])

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
    """Zero-regex tag name parser matching Ren'Py tag syntax."""
    if not raw or raw[0] != "{":
        return raw
    idx = 1
    if idx < len(raw) and raw[idx] == "/":
        idx = 2
    if idx >= len(raw):
        return raw
    if raw[idx] == "#":
        return "#"
    start = idx
    length = len(raw)
    while idx < length:
        c = raw[idx]
        if ("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9") or (c == "_"):
            idx += 1
        else:
            break
    if idx > start:
        return raw[start:idx]
    return raw


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
    text = normalize_nfc(text)
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
    5. Hallucinated outer styling: strips extraneous outer curly tags wrapping placeholder blocks.
    6. Enclosing tag pair restoration: wraps translation with dropped outer formatting pairs.
    """
    if not masked.tokens:
        return translated

    result = normalize_nfc(translated)

    if "⟦" in result and ("{" in result or "}" in result):
        outer_curly = _OUTER_CURLY_RE.match(result)
        if outer_curly:
            result = outer_curly.group(1).strip()

    valid_tokens: set[str] = {token.key for token in masked.tokens}
    valid_tokens.update(f"/{token.key}" for token in masked.tokens if token.close_raw is not None)

    def _replace_placeholder(match: re.Match[str]) -> str:
        open_d, slash, key, close_d = match.groups()
        candidate = f"{slash}{key}"
        if candidate in valid_tokens:
            return f"{OPEN_MARK}{candidate}{CLOSE_MARK}"
        if open_d == OPEN_MARK and close_d == CLOSE_MARK:
            return f"{OPEN_MARK}{candidate}{CLOSE_MARK}"
        return match.group(0)

    result = _REPAIR_PLACEHOLDER_RE.sub(_replace_placeholder, result)

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

    for token in masked.tokens:
        if token.close_raw is not None:
            open_mark = f"{OPEN_MARK}{token.key}{CLOSE_MARK}"
            close_mark = f"{OPEN_MARK}/{token.key}{CLOSE_MARK}"
            if open_mark in result and close_mark not in result:
                result = result + close_mark

    has_raw_curly = ("{" in result or "}" in result) and (OPEN_MARK not in result)
    if not has_raw_curly:
        # Restore leading standalone/opening token sequence if partially or fully dropped
        m_lead = _LEADING_TOKEN_CLUSTER_RE.match(masked.text.lstrip())
        if m_lead:
            leading_tokens = _TOKEN_CLUSTER_RE.findall(m_lead.group(0))
            m_cur_lead = _LEADING_TOKEN_CLUSTER_RE.match(result.lstrip())
            cur_lead_str = m_cur_lead.group(0) if m_cur_lead else ""
            cur_lead_tokens = _TOKEN_CLUSTER_RE.findall(cur_lead_str)
            if any(tok not in cur_lead_tokens for tok in leading_tokens):
                result = result[len(cur_lead_str) :]
                result = "".join(leading_tokens) + result

        # Restore trailing standalone/closing token sequence if partially or fully dropped
        m_trail = _TRAILING_TOKEN_CLUSTER_RE.search(masked.text.rstrip())
        if m_trail:
            trailing_tokens = _TOKEN_CLUSTER_RE.findall(m_trail.group(0))
            m_cur_trail = _TRAILING_TOKEN_CLUSTER_RE.search(result.rstrip())
            cur_trail_str = m_cur_trail.group(0) if m_cur_trail else ""
            cur_trail_tokens = _TOKEN_CLUSTER_RE.findall(cur_trail_str)
            if any(tok not in cur_trail_tokens for tok in trailing_tokens):
                result = result[: len(result) - len(cur_trail_str)]
                result = result + "".join(trailing_tokens)

        # Restore any remaining standalone tokens dropped in body
        for token in masked.tokens:
            if token.close_raw is None:
                mark = f"{OPEN_MARK}{token.key}{CLOSE_MARK}"
                if mark not in result:
                    result = result + mark

    if "{" not in result and "[" not in result:
        text_strip = masked.text.strip()
        idx = 0
        token_by_key = {t.key: t for t in masked.tokens if t.close_raw is not None}
        prefix_keys: list[str] = []
        while idx < len(text_strip):
            m = _PREFIX_KEY_RE.match(text_strip[idx:])
            if not m:
                break
            k = m.group(1)
            if k in token_by_key:
                prefix_keys.append(k)
            idx += m.end()

        suffix_keys: list[str] = []
        s_text = text_strip[idx:].rstrip()
        while s_text:
            m = _SUFFIX_KEY_RE.search(s_text)
            if not m:
                break
            k = m.group(1)
            if k in token_by_key:
                suffix_keys.append(k)
            s_text = s_text[: m.start()].rstrip()

        enclosing = [k for k in prefix_keys if k in suffix_keys]
        if enclosing:
            missing = [
                k
                for k in enclosing
                if f"{OPEN_MARK}{k}{CLOSE_MARK}" not in result
                and f"{OPEN_MARK}/{k}{CLOSE_MARK}" not in result
            ]
            if missing:
                open_str = "".join(f"{OPEN_MARK}{k}{CLOSE_MARK}" for k in missing)
                close_str = "".join(f"{OPEN_MARK}/{k}{CLOSE_MARK}" for k in reversed(missing))
                result = f"{open_str}{result}{close_str}"

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
_FORMATTING_TAGS = frozenset(
    {
        "size",
        "font",
        "color",
        "alpha",
        "outlinecolor",
        "cps",
        "k",
        "space",
        "vspace",
    }
)


@dataclass(slots=True)
class _CloserIndex:
    last_universal: int
    last_named: dict[str, int]
    last_symmetric: dict[str, int]

    def has_closer(self, name: str, index: int) -> bool:
        return (
            self.last_universal > index
            or self.last_named.get(name, -1) > index
            or (name in _SYMMETRIC_STYLE_TAGS and self.last_symmetric.get(name, -1) > index)
        )


def _build_closer_index(tokens: list[Token]) -> _CloserIndex:
    last_univ = -1
    last_named: dict[str, int] = {}
    last_sym: dict[str, int] = {}
    for i, t in enumerate(tokens):
        if t.kind is not TokenKind.TAG:
            continue
        raw = t.raw
        if raw == "{/}":
            last_univ = i
        elif raw.startswith("{/"):
            name = _tag_name(raw)
            last_named[name] = i
        else:
            name = _tag_name(raw)
            if name in _SYMMETRIC_STYLE_TAGS:
                last_sym[name] = i
    return _CloserIndex(last_univ, last_named, last_sym)


def _is_openable_tag(
    token: Token,
    tokens: list[Token],
    index: int,
    closer_index: _CloserIndex | None = None,
) -> bool:
    if token.kind is not TokenKind.TAG:
        return False
    raw = token.raw
    if raw.startswith("{/") or raw.endswith("/}") or raw == "{/}":
        return False
    name = _tag_name(raw)
    if name in _STANDALONE_TAGS:
        return False
    if closer_index is not None:
        return closer_index.has_closer(name, index)
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

    src_openers: list[str] = []
    src_standalone: list[str] = []
    tgt_openers: list[str] = []
    tgt_standalone: list[str] = []

    tgt_closer_index = _build_closer_index(tgt_tokens)
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
        elif _is_openable_tag(token, tgt_tokens, index, tgt_closer_index):
            tgt_stack.append((name, raw))
        else:
            tgt_standalone.append(raw)

    for _, raw in tgt_stack:
        tgt_standalone.append(raw)

    src_closer_index = _build_closer_index(src_tokens)
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
        elif _is_openable_tag(token, src_tokens, index, src_closer_index):
            src_stack.append((name, raw))
        else:
            src_standalone.append(raw)

    for _, raw in src_stack:
        src_standalone.append(raw)

    src_op_c = Counter(src_openers)
    tgt_op_c = Counter(tgt_openers)
    for op, count in src_op_c.items():
        if tgt_op_c.get(op, 0) < count:
            issues.append(PlaceholderIssue("missing_placeholder", op))
        elif tgt_op_c.get(op, 0) > count:
            issues.append(PlaceholderIssue("duplicate_placeholder", op))
    src_tag_names = {_tag_name(t.raw) for t in src_tokens if t.kind is TokenKind.TAG}
    for op in tgt_op_c:
        if op not in src_op_c:
            name = _tag_name(op)
            if name in _FORMATTING_TAGS and name not in src_tag_names:
                continue
            issues.append(PlaceholderIssue("unknown_placeholder", op))

    src_st_c = Counter(src_standalone)
    tgt_st_c = Counter(tgt_standalone)
    for st, count in src_st_c.items():
        if tgt_st_c.get(st, 0) < count:
            issues.append(PlaceholderIssue("missing_placeholder", st))
        elif tgt_st_c.get(st, 0) > count:
            issues.append(PlaceholderIssue("duplicate_placeholder", st))
    for st in tgt_st_c:
        if st not in src_st_c:
            name = _tag_name(st)
            if name in _FORMATTING_TAGS and name not in src_tag_names:
                continue
            issues.append(PlaceholderIssue("unknown_placeholder", st))

    src_escapes = Counter(t.raw for t in src_tokens if t.kind is TokenKind.ESCAPE)
    tgt_escapes = Counter(t.raw for t in tgt_tokens if t.kind is TokenKind.ESCAPE)
    for esc, count in src_escapes.items():
        if tgt_escapes.get(esc, 0) < count:
            issues.append(PlaceholderIssue("missing_placeholder", esc))

    src_newlines = sum(1 for t in src_tokens if t.kind is TokenKind.NEWLINE)
    tgt_newlines = sum(1 for t in tgt_tokens if t.kind is TokenKind.NEWLINE)
    if src_newlines > 0 and tgt_newlines < src_newlines:
        issues.append(PlaceholderIssue("missing_placeholder", "\\n"))

    return issues


def enforce_tag_invariants(source: str, target: str) -> str:
    """Guarantee that target text only retains text formatting tags present in source:
    Tags(Target) ⊆ Tags(Source).

    Hallucinated formatting tags (e.g. {size}, {font}, {color}) invented by the LLM
    that were not present in the source text are cleanly stripped, while preserving valid
    source tags and escaped brackets ([[ and ]]).
    """
    try:
        src_parts = tokenize(source)
        src_tags = {
            _tag_name(p.raw)
            for p in src_parts
            if isinstance(p, Token) and p.kind is TokenKind.TAG and p.raw != "{/}"
        }
    except Exception:
        src_tags = set()

    try:
        tgt_parts = tokenize(target)
    except Exception:
        return target

    result_pieces: list[str] = []
    tag_stack: list[tuple[str, bool]] = []

    for part in tgt_parts:
        if isinstance(part, str):
            result_pieces.append(part)
        elif part.kind is not TokenKind.TAG:
            result_pieces.append(part.raw)
        else:
            raw = part.raw
            name = _tag_name(raw)

            if raw == "{/}":
                if tag_stack:
                    _, is_kept = tag_stack.pop()
                    if is_kept:
                        result_pieces.append(raw)
                elif "{/}" in source:
                    result_pieces.append(raw)
            elif raw.startswith("{/"):
                if name in _FORMATTING_TAGS and name not in src_tags:
                    for idx in range(len(tag_stack) - 1, -1, -1):
                        if tag_stack[idx][0] == name:
                            del tag_stack[idx]
                            break
                else:
                    result_pieces.append(raw)
            elif name in _SYMMETRIC_STYLE_TAGS and tag_stack and tag_stack[-1][0] == name:
                _, is_kept = tag_stack.pop()
                if is_kept:
                    result_pieces.append(raw)
            else:
                if name in _FORMATTING_TAGS and name not in src_tags:
                    tag_stack.append((name, False))
                else:
                    tag_stack.append((name, True))
                    result_pieces.append(raw)

    return "".join(result_pieces)


def reconcile_translation(source: str, target: str) -> tuple[str, list[PlaceholderIssue]]:
    """Validate and unmask (if needed) target translation against source.

    Supports both synthetic placeholder tokens (e.g. ⟦V1⟧) and natural Ren'Py markup.
    Returns (cleaned_target, issues).
    """
    target = normalize_nfc(target)
    if OPEN_MARK in target:
        masked = mask(source)
        issues = validate(masked, target)
        if issues:
            return target, issues
        try:
            unmasked = unmask(masked, target)
            cleaned = enforce_tag_invariants(source, unmasked)
            return cleaned, []
        except (PlaceholderError, MarkupError) as err:
            return target, [PlaceholderIssue("unmask_error", str(err))]
    else:
        cleaned = enforce_tag_invariants(source, target)
        issues = validate_raw(source, cleaned)
        return cleaned, issues


mask.mask = mask  # type: ignore[attr-defined]
