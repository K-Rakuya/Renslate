"""Quality assurance verification and post-translation sanitization."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from vitrans.mask import MaskedText, find_markup_error, validate
from vitrans.models import GlossaryEntry, QAIssue

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
    r"^(?:bản dịch|dịch|translation|vietnamese|tiếng việt)\s*:\s*", re.IGNORECASE
)
_SPEAKER_METADATA = re.compile(
    r"^\s*\[(?:Người nói|Speaker)\s*:\s*[^\]]+\]\s*:?\s*", re.IGNORECASE
)
_STANDALONE_SPEAKER = re.compile(r"^\[[A-Za-z0-9_ '\u2019\-]{1,40}\]$")
_SPEAKER_METADATA_LEAK = re.compile(r"\[(?:Người nói|Speaker)\s*:[^\]]*\]", re.IGNORECASE)
_BRACKET_SPEAKER = re.compile(
    r"^\[(?!/?(?:V|T|E|L)\d+\])[^\]]+\]\s*(?::\s*|(?=[\"'\u201c\u2018\u300c\u300e])|\s+)"
)
_PAREN_SPEAKER = re.compile(
    r"^\([^\)]+\)\s*(?::\s*|(?=[\"'\u201c\u2018\u300c\u300e]))"
)
_COLON_SPEAKER = re.compile(r"^[A-Za-z0-9_ '\u2019\-]{1,30}\s*:\s*")


def clean_translation(
    target: str,
    speaker: str | None = None,
    source: str | None = None,
) -> str:
    """Sanitize model translation: remove hallucinated speaker labels and unwanted outer quotes.

    Safely preserves dialogue enclosed in parentheses or brackets (e.g. thoughts, choices,
    or system strings like `(statement)`) when the source string itself is so enclosed.
    If the target consists solely of speaker metadata tags (e.g. `[Người nói: MC]`),
    returns an empty string so QA validation flags it as empty and forces re-translation.
    """
    cleaned = unicodedata.normalize("NFC", target).strip()
    if not cleaned:
        return ""

    src_strip = source.strip() if source else ""
    is_source_paren = bool(src_strip.startswith("(") and src_strip.endswith(")"))
    is_source_bracket = bool(src_strip.startswith("[") and src_strip.endswith("]"))
    is_source_quoted = bool(
        len(src_strip) >= 2
        and src_strip.startswith(('"', "'", "\u201c", "\u2018"))
        and src_strip.endswith(('"', "'", "\u201d", "\u2019"))
    )

    # Multi-pass peeling to handle layered artifacts e.g. [Dyna] "Bản dịch: 'Hello'"
    for _ in range(4):
        prev = cleaned

        # 1. Strip conversational prefixes: e.g. 'Bản dịch: ...', 'Translation: ...'
        cleaned = _PREFIX_JUNK.sub("", cleaned).strip()

        # 2. Strip explicit prompt speaker metadata: e.g. '[Người nói: MC]', '[Speaker: Layla]'
        cleaned = _SPEAKER_METADATA.sub("", cleaned).strip()

        # 3. Specific speaker match if provided
        if speaker and not is_source_bracket:
            spk_norm = speaker.strip()
            if cleaned.lower().startswith(f"[{spk_norm.lower()}]"):
                cleaned = cleaned[len(f"[{spk_norm}]") :].lstrip(" :").strip()
            elif cleaned.lower().startswith(f"{spk_norm.lower()}:"):
                cleaned = cleaned[len(f"{spk_norm}:") :].strip()

        # 4. Match generic bracketed speaker label e.g. '[Dyna] "..."', '[MC]: "..."'
        if not is_source_bracket:
            cleaned = _BRACKET_SPEAKER.sub("", cleaned).strip()
            if _STANDALONE_SPEAKER.match(cleaned):
                cleaned = ""

        # 5. Match generic parenthesized speaker label e.g. '(Dyna): ...'
        if not is_source_paren:
            cleaned = _PAREN_SPEAKER.sub("", cleaned).strip()

        # 6. Match generic alphanumeric speaker name with colon e.g. 'Dyna: ...'
        cleaned = _COLON_SPEAKER.sub("", cleaned).strip()

        # 7. Strip accidental outer quotes around string (if source was not quoted)
        if not is_source_quoted and len(cleaned) >= 2 and (
            (cleaned.startswith('"') and cleaned.endswith('"'))
            or (cleaned.startswith("\u201c") and cleaned.endswith("\u201d"))
            or (cleaned.startswith("'") and cleaned.endswith("'"))
            or (cleaned.startswith("\u2018") and cleaned.endswith("\u2019"))
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
    ) -> list[QAIssue]:
        """Perform all QA validations on a translated unit."""
        issues: list[QAIssue] = []
        target = target.strip()

        # 1. Empty target check
        if not target:
            issues.append(QAIssue(uid, "empty_target", "error", "Bản dịch bị rỗng."))
            return issues

        # 1b. Leaked speaker tag or metadata check
        if _SPEAKER_METADATA_LEAK.search(target):
            issues.append(
                QAIssue(
                    uid,
                    "speaker_tag_leakage",
                    "error",
                    "Bản dịch chứa nhãn người nói hoặc metadata nhắc lệnh ([Người nói: ...]).",
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

        # 2. Ren'Py markup validity (unclosed brackets/braces)
        markup_err = find_markup_error(target)
        if markup_err is not None:
            issues.append(QAIssue(uid, "bad_markup", "error", f"Lỗi cú pháp markup: {markup_err}"))

        # 3. Placeholder checks (if source had placeholders)
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

        # 4. Verbatim check (target completely identical to source)
        clean_src = source.strip()
        if (
            clean_src.lower() == target.lower()
            and len(clean_src) > 4
            and any(c.isalpha() for c in clean_src)
        ):
            # Check if this is a known glossary term / proper noun
            is_glossary = any(g.term.lower() == clean_src.lower() for g in glossary_matches)
            if not is_glossary:
                issues.append(
                    QAIssue(
                        uid,
                        "identical_to_source",
                        "warning",
                        "Bản dịch trùng nguyên văn với văn bản gốc tiếng Anh.",
                    )
                )

        # 5. Length ratio overflow check
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

        # 6. Language leakage check (CJK characters or heavy English words)
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

        # 7. Glossary compliance & English glossary leakage
        target_norm = unicodedata.normalize("NFC", target).lower()
        for g in glossary_matches:
            if not g.translation.strip():
                continue

            # Term must actually be present in the source string!
            candidates = [g.term] + [a for a in g.aliases if a.strip()]
            flags = 0 if g.case_sensitive else re.IGNORECASE
            source_has_term = any(
                re.search(rf"(?<!\w){re.escape(cand)}(?!\w)", source, flags)
                for cand in candidates
            )
            if not source_has_term:
                continue

            g_trans = unicodedata.normalize("NFC", g.translation).lower()
            if g_trans not in target_norm:
                issues.append(
                    QAIssue(
                        uid,
                        "glossary_violation",
                        "warning",
                        f"Thuật ngữ '{g.term}' thiếu bản dịch chuẩn '{g.translation}'.",
                    )
                )

            # Check if raw English term leaked into the target text
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

        # 8. In-sentence pronoun register clash
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

        return issues


_MARKUP_OR_PLACEHOLDER = re.compile(r"(\{[^{}]*\}|\[[^\[\]]*\]|⟦[^⟦⟧]*⟧)")


def repair_glossary_leakage(
    text: str, glossary_matches: Sequence[GlossaryEntry]
) -> str:
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
