"""Quality assurance verification and post-translation sanitization."""

from __future__ import annotations

import re
from collections.abc import Sequence

from renslate.core.models import GlossaryEntry, QAIssue
from renslate.core.textio import normalize_nfc
from renslate.renpy.mask import MaskedText, find_markup_error, validate

_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_ENGLISH_WORDS = frozenset(
    {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
    }
)
_WORD_TOKEN = re.compile(r"[A-Za-z]+")
_PREFIX_JUNK = re.compile(
    r"^(?:bản dịch|dịch|translation|vietnamese|tiếng việt)\s*:\s*"
    r"|^(?:[dsu]:[A-Za-z0-9_.\-]+|\d+)\s*\|\s*",
    re.IGNORECASE,
)
_SPEAKER_METADATA = re.compile(
    r"^\s*\[\s*(?:"
    r"(?:Người nói|Speaker|Độc thoại/Nội tâm|Độc thoại|Nội tâm|Suy nghĩ|"
    r"Monologue(?:/(?:Internal|Thought))?|Thought)"
    r"|(?:"
    r"(?:Người nghe|Listener|Quan hệ|Relation(?:ship)?|Cảm xúc|Emotion)\s*:"
    r")"
    r")[^\]]*\]\s*:?\s*",
    re.IGNORECASE,
)
_METADATA_SUFFIX = re.compile(
    r"\s*\[\s*(?:"
    r"(?:Người nói|Speaker|Độc thoại/Nội tâm|Độc thoại|Nội tâm|Suy nghĩ|"
    r"Monologue(?:/(?:Internal|Thought))?|Thought)"
    r"|(?:"
    r"(?:Người nghe|Listener|Quan hệ|Relation(?:ship)?|Cảm xúc|Emotion)\s*:"
    r")"
    r")[^\]]*\]\s*$",
    re.IGNORECASE,
)
_STANDALONE_SPEAKER = re.compile(r"^\[[A-Za-z0-9_ '\u2019\-]{1,40}\]$")
_SPEAKER_METADATA_LEAK = re.compile(
    r"\[\s*(?:Người nói|Speaker|Độc thoại/Nội tâm|Độc thoại|Nội tâm|Suy nghĩ|"
    r"Monologue(?:/(?:Internal|Thought))?|Thought)\s*:[^\]]*\]",
    re.IGNORECASE,
)
_RELATION_METADATA_LEAK = re.compile(
    r"\[[^\]]*(?:Quan hệ|Relation(?:ship)?|Người nghe|Listener|Cảm xúc|Emotion)\s*:[^\]]*\]"
    r"|\[[^\]]*(?:xưng\s+['\"][^'\"]+['\"]\s*,\s*gọi\s+['\"][^'\"]+['\"])[^\]]*\]",
    re.IGNORECASE,
)
_BRACKET_SPEAKER = re.compile(
    r"^\[(?!/?(?:V|T|E|L)\d+\])[^\]]+\]\s*(?::\s*|(?=[\"'\u201c\u2018\u300c\u300e])|\s+)"
)
_PAREN_SPEAKER = re.compile(r"^\([^\)]+\)\s*(?::\s*|(?=[\"'\u201c\u2018\u300c\u300e]))")
_COLON_SPEAKER = re.compile(r"^[A-Za-z0-9_ '\u2019\-]{1,30}\s*:\s*")
_TAG_STRIP_RE = re.compile(r"\{[^\}]*\}")
_PLACEHOLDER_STRIP_RE = re.compile(r"⟦[^⟧]*⟧")
_SOUND_EFFECT_RE = re.compile(
    r"(?:[Hh]aa+|[Aa]ah*|[Hh]m+|[Gg]hg|[Uu]gh+|[Oo]oh*|[Nn]g+m*|[Mm]+nh*|[\.\?! \-,~])+"
)
_STUTTER_NAME_RE = re.compile(r"^[A-Za-z]-[A-Za-z0-9_'\"]+[!,\.\?]*$")
_RE_YOU = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b", re.IGNORECASE)
_RE_I = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.IGNORECASE)


def is_non_translatable_text(
    stripped_text: str,
    glossary_terms: set[str] | None = None,
) -> bool:
    """Check if stripped dialogue consists of onomatopoeia, sound effects, or character names."""
    s = stripped_text.strip()
    if not any(c.isalpha() for c in s):
        return True
    if len(s) <= 4:
        return True
    if "~" in s:
        return True
    if re.search(r"([a-zA-Z])\1{2,}", s):
        return True
    if _SOUND_EFFECT_RE.fullmatch(s):
        return True
    if _STUTTER_NAME_RE.match(s):
        return True
    return bool(glossary_terms and s.strip("!.,?\"'").lower() in glossary_terms)


def clean_translation(
    target: str,
    speaker: str | None = None,
    source: str | None = None,
) -> str:
    """Sanitize model translation: remove hallucinated speaker labels and unwanted outer quotes.

    Safely preserves dialogue enclosed in parentheses or brackets (e.g. thoughts, choices,
    dialogue actions like `[Sigh]`, ellipses `[...]`, or Ren'Py variable interpolations)
    when the source string contains matching bracketed elements.
    If the target consists solely of speaker or relation metadata tags, returns an empty string
    so QA validation flags it as empty and forces re-translation.
    """
    cleaned = normalize_nfc(target).strip()
    if not cleaned:
        return ""

    src_strip = source.strip() if source else ""
    is_source_paren = bool(src_strip.startswith("(") and src_strip.endswith(")"))
    is_source_bracket = bool(src_strip.startswith("[") and src_strip.endswith("]"))
    is_source_bracket_start = bool(src_strip.startswith("["))
    is_source_paren_start = bool(src_strip.startswith("("))
    is_source_quoted = bool(
        len(src_strip) >= 2
        and src_strip.startswith(('"', "'", "\u201c", "\u2018"))
        and src_strip.endswith(('"', "'", "\u201d", "\u2019"))
    )

    # Multi-pass peeling to handle layered artifacts e.g. [Dyna] "Bản dịch: 'Hello'"
    for _ in range(4):
        prev = cleaned

        cleaned = _PREFIX_JUNK.sub("", cleaned).strip()
        cleaned = _SPEAKER_METADATA.sub("", cleaned).strip()
        cleaned = _METADATA_SUFFIX.sub("", cleaned).strip()

        has_source_matching_bracket = False
        if source and "[" in source:
            bm = re.match(r"^\[([^\]]+)\]", cleaned)
            if bm and bm.group(0) in source:
                has_source_matching_bracket = True

        protect_bracket = (
            is_source_bracket or is_source_bracket_start or has_source_matching_bracket
        )

        if speaker and not protect_bracket:
            spk_norm = speaker.strip()
            if cleaned.lower().startswith(f"[{spk_norm.lower()}]"):
                cleaned = cleaned[len(f"[{spk_norm}]") :].lstrip(" :").strip()
            elif cleaned.lower().startswith(f"{spk_norm.lower()}:"):
                cleaned = cleaned[len(f"{spk_norm}:") :].strip()

        if not protect_bracket:
            cleaned = _BRACKET_SPEAKER.sub("", cleaned).strip()
            if _STANDALONE_SPEAKER.match(cleaned):
                cleaned = ""

        if not (is_source_paren or is_source_paren_start):
            cleaned = _PAREN_SPEAKER.sub("", cleaned).strip()

        cleaned = _COLON_SPEAKER.sub("", cleaned).strip()

        if (
            not is_source_quoted
            and len(cleaned) >= 2
            and (
                (cleaned.startswith('"') and cleaned.endswith('"'))
                or (cleaned.startswith("\u201c") and cleaned.endswith("\u201d"))
                or (cleaned.startswith("'") and cleaned.endswith("'"))
                or (cleaned.startswith("\u2018") and cleaned.endswith("\u2019"))
            )
        ):
            cleaned = cleaned[1:-1].strip()

        if cleaned == prev or not cleaned:
            break

    return cleaned


class QAChecker:
    """Runs linguistic, structural, and consistency QA checks on translations."""

    def __init__(self, max_length_ratio: float = 1.8) -> None:
        self.max_length_ratio = max_length_ratio

    def check_unit(
        self,
        uid: str,
        source: str,
        masked: MaskedText | None,
        target: str,
        glossary_matches: Sequence[GlossaryEntry] = (),
        speaker: str | None = None,
        pronoun_rules: tuple[str, str] | None = None,
    ) -> list[QAIssue]:
        """Perform all QA validations on a translated unit."""
        issues: list[QAIssue] = []
        target = target.strip()

        if not target:
            issues.append(QAIssue(uid, "empty_target", "error", "Bản dịch bị rỗng."))
            return issues

        if _SPEAKER_METADATA_LEAK.search(target):
            issues.append(
                QAIssue(
                    uid,
                    "speaker_tag_leakage",
                    "error",
                    "Bản dịch chứa nhãn người nói hoặc metadata nhắc lệnh ([Người nói: ...]).",
                )
            )

        if _RELATION_METADATA_LEAK.search(target):
            issues.append(
                QAIssue(
                    uid,
                    "relation_metadata_leakage",
                    "error",
                    "Bản dịch chứa nhãn quan hệ, người nghe hoặc cảm xúc "
                    "nhắc lệnh ([Quan hệ: ...]).",
                )
            )

        src_strip = source.strip()
        is_src_bracket = src_strip.startswith("[") and src_strip.endswith("]")
        if not is_src_bracket:
            if speaker:
                spk_clean = speaker.strip().lower()
                tgt_clean = target.strip(" []:'\"").lower()
                if tgt_clean == spk_clean and len(src_strip) > len(speaker) + 3:
                    issues.append(
                        QAIssue(
                            uid,
                            "speaker_tag_leakage",
                            "error",
                            f"Bản dịch chỉ chứa tên người nói '{speaker}' "
                            "mà không có nội dung lời thoại.",
                        )
                    )
            if _STANDALONE_SPEAKER.fullmatch(target):
                issues.append(
                    QAIssue(
                        uid,
                        "speaker_tag_leakage",
                        "error",
                        "Bản dịch chỉ chứa nhãn tên trong ngoặc vuông ([...]) "
                        "mà không có lời thoại.",
                    )
                )

        markup_err = find_markup_error(target)
        if markup_err is not None:
            issues.append(QAIssue(uid, "bad_markup", "error", f"Lỗi cú pháp markup: {markup_err}"))

        if masked and masked.tokens:
            placeholder_issues = validate(masked, target)
            for pi in placeholder_issues:
                issues.append(
                    QAIssue(
                        uid,
                        pi.code,
                        "error",
                        f"Lỗi placeholder: {pi.code} ({pi.detail})",
                    )
                )

        raw_src = _PLACEHOLDER_STRIP_RE.sub("", _TAG_STRIP_RE.sub("", source)).strip()
        raw_tgt = _PLACEHOLDER_STRIP_RE.sub("", _TAG_STRIP_RE.sub("", target)).strip()
        if raw_src.lower() == raw_tgt.lower() and not is_non_translatable_text(
            raw_src, {g.term.lower() for g in glossary_matches}
        ):
            is_glossary = any(g.term.lower() == raw_src.lower() for g in glossary_matches)
            if not is_glossary:
                issues.append(
                    QAIssue(
                        uid,
                        "identical_to_source",
                        "warning",
                        "Bản dịch trùng nguyên văn với văn bản gốc tiếng Anh.",
                    )
                )

        src_len = len(source)
        tgt_len = len(target)
        if src_len >= 15:
            ratio = tgt_len / src_len
            if ratio > self.max_length_ratio:
                issues.append(
                    QAIssue(
                        uid,
                        "length_ratio_exceeded",
                        "warning",
                        f"Bản dịch quá dài ({tgt_len} ký tự vs {src_len} gốc, "
                        f"tỷ lệ {ratio:.2f} > {self.max_length_ratio}).",
                    )
                )

        if _CJK_PATTERN.search(target):
            issues.append(
                QAIssue(
                    uid,
                    "cjk_leakage",
                    "warning",
                    "Bản dịch chứa ký tự tiếng Trung/Nhật/Hàn (CJK) chưa được dịch.",
                )
            )

        words = [w.lower() for w in _WORD_TOKEN.findall(target)]
        if len(words) >= 6:
            english_count = sum(1 for w in words if w in _ENGLISH_WORDS)
            if english_count / len(words) >= 0.4:
                issues.append(
                    QAIssue(
                        uid,
                        "english_leakage",
                        "warning",
                        f"Nhiều từ tiếng Anh chưa dịch ({english_count}/{len(words)} từ).",
                    )
                )

        target_norm = normalize_nfc(target).lower()
        for g in glossary_matches:
            if not g.translation.strip():
                continue

            candidates = [g.term] + [a for a in g.aliases if a.strip()]
            flags = 0 if g.case_sensitive else re.IGNORECASE
            source_has_term = any(
                re.search(rf"(?<!\w){re.escape(cand)}(?!\w)", source, flags) for cand in candidates
            )
            if not source_has_term:
                continue

            g_trans = normalize_nfc(g.translation).lower()
            if g_trans not in target_norm:
                issues.append(
                    QAIssue(
                        uid,
                        "glossary_violation",
                        "warning",
                        f"Thuật ngữ '{g.term}' thiếu bản dịch chuẩn '{g.translation}'.",
                    )
                )

            for cand in candidates:
                cand_lower = cand.lower()
                if cand_lower == g_trans:
                    continue
                flags = 0 if g.case_sensitive else re.IGNORECASE
                if re.search(rf"(?<!\w){re.escape(cand)}(?!\w)", target, flags):
                    issues.append(
                        QAIssue(
                            uid,
                            "untranslated_glossary_leakage",
                            "warning",
                            f"Thuật ngữ tiếng Anh '{cand}' bị rò rỉ nguyên văn trong bản dịch.",
                        )
                    )
                    break

        has_ta = bool(re.search(r"(?<!\w)ta(?!\w)", target_norm))
        has_toi = bool(re.search(r"(?<!\w)tôi(?!\w)", target_norm))
        has_minh = bool(re.search(r"(?<!\w)mình(?!\w)", target_norm))
        if has_ta and (has_toi or has_minh):
            issues.append(
                QAIssue(
                    uid,
                    "pronoun_register_clash",
                    "warning",
                    "Lẫn lộn đại từ ngôi thứ nhất ('ta' và 'tôi/mình') trong cùng một câu.",
                )
            )

        has_nguoi = bool(re.search(r"(?<!\w)ngươi(?!\w)", target_norm))
        has_cau = bool(re.search(r"(?<!\w)cậu(?!\w)", target_norm))
        has_ban = bool(re.search(r"(?<!\w)bạn(?!\w)", target_norm))
        if has_nguoi and (has_cau or has_ban):
            issues.append(
                QAIssue(
                    uid,
                    "pronoun_register_clash",
                    "warning",
                    "Lẫn lộn đại từ ngôi thứ hai ('ngươi' và 'cậu/bạn') trong cùng một câu.",
                )
            )

        if pronoun_rules and len(pronoun_rules) == 2:
            self_p, other_p = pronoun_rules[0].strip(), pronoun_rules[1].strip()
            if self_p and other_p and self_p.lower() != other_p.lower():
                tgt_clean = target.lower().strip(" \"'\u201c\u201d")
                core_tgt = re.sub(r"^(?:[\w\-]+[!?,.~]+\s*)+", "", tgt_clean)

                if _RE_YOU.search(source) and not _RE_I.search(source):
                    pat = (
                        rf"^(?:sao\s+|tại sao\s+|liệu\s+|có phải\s+|phải chăng\s+)?"
                        rf"{re.escape(self_p.lower())}\b"
                    )
                    if re.search(pat, core_tgt):
                        issues.append(
                            QAIssue(
                                uid,
                                "pronoun_inversion",
                                "error",
                                f"Đảo ngược đại từ: câu thoại ngôi thứ hai (you) bị dịch "
                                f"thành đại từ tự xưng '{self_p}' thay vì "
                                f"gọi đối phương '{other_p}'.",
                            )
                        )

                elif _RE_I.search(source) and not _RE_YOU.search(source):
                    pat = (
                        rf"^(?:sao\s+|tại sao\s+|liệu\s+|có phải\s+|phải chăng\s+)?"
                        rf"{re.escape(other_p.lower())}\b"
                    )
                    if re.search(pat, core_tgt):
                        issues.append(
                            QAIssue(
                                uid,
                                "pronoun_inversion",
                                "error",
                                f"Đảo ngược đại từ: câu thoại ngôi thứ nhất (I) bị dịch "
                                f"thành đại từ gọi đối phương '{other_p}' thay vì "
                                f"tự xưng '{self_p}'.",
                            )
                        )

        return issues


_MARKUP_OR_PLACEHOLDER = re.compile(r"(\{[^{}]*\}|\[[^\[\]]*\]|⟦[^⟦⟧]*⟧)")


def repair_glossary_leakage(text: str, glossary_matches: Sequence[GlossaryEntry]) -> str:
    """Auto-replace leaked English glossary words with their designated translations.

    Leaves all Ren'Py tags, variable interpolations, and placeholder tokens intact.
    """
    if not text or not glossary_matches:
        return text

    valid_entries = [g for g in glossary_matches if g.translation.strip()]
    if not valid_entries:
        return text

    replacements: list[tuple[re.Pattern[str], str, int]] = []
    for g in valid_entries:
        for cand in [g.term] + [a for a in g.aliases if a.strip()]:
            if cand.lower() == g.translation.strip().lower():
                continue
            flags = 0 if g.case_sensitive else re.IGNORECASE
            pat = re.compile(rf"(?<!\w){re.escape(cand)}(?!\w)", flags)
            replacements.append((pat, g.translation, len(cand)))

    if not replacements:
        return text

    # Sort replacements by term length descending so longer phrases match first
    replacements.sort(key=lambda x: x[2], reverse=True)

    parts = _MARKUP_OR_PLACEHOLDER.split(text)
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if not seg:
            continue
        for pat, trans, _ in replacements:
            if pat.search(seg):
                seg = pat.sub(trans, seg)
        parts[i] = seg

    return "".join(parts)
