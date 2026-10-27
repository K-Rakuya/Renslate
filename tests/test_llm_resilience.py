"""Comprehensive tests for LLM parser resilience and Tag Invariant Enforcement (R3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from renslate.core.config import LLMConfig
from renslate.core.errors import LLMError
from renslate.engine.llm import LLMClient, extract_json_object
from renslate.renpy.mask import enforce_tag_invariants, reconcile_translation, validate_raw


class TestDoubledAndQuadrupleQuotes:
    """Tests for doubled, tripled, and quadruple quote normalization and parsing."""

    def test_doubled_quotes_without_commas(self) -> None:
        """Doubled quotes on keys and values without commas must not truncate values."""
        raw = '""d:1"": ""Line 1""\n""d:2"": ""Line 2""'
        result = extract_json_object(raw)
        assert result == {"d:1": "Line 1", "d:2": "Line 2"}
        assert result["d:1"] != ""
        assert result["d:2"] != ""

    def test_doubled_quotes_crlf_without_commas(self) -> None:
        """Doubled quotes with Windows CRLF newlines and no commas."""
        raw = '""s:1"": ""First dialogue""\r\n""s:2"": ""Second dialogue""'
        result = extract_json_object(raw)
        assert result == {"s:1": "First dialogue", "s:2": "Second dialogue"}

    def test_doubled_quotes_with_commas(self) -> None:
        """Doubled quotes with trailing commas."""
        raw = '""d:1"": ""Line 1"",\n""d:2"": ""Line 2"""'
        result = extract_json_object(raw)
        assert result["d:1"] == "Line 1"
        assert result["d:2"] == "Line 2"

    def test_doubled_quotes_single_line_space_separated(self) -> None:
        """Doubled quotes separated only by whitespace."""
        raw = '""d:1"": ""Alpha"" ""d:2"": ""Beta""'
        result = extract_json_object(raw)
        assert result == {"d:1": "Alpha", "d:2": "Beta"}

    def test_quadruple_quotes(self) -> None:
        """Quadruple quotes produced by models escaping nested JSON."""
        raw = '""""d:1"""": """"Line 1""""\n""""d:2"""": """"Line 2""""'
        result = extract_json_object(raw)
        assert result == {"d:1": "Line 1", "d:2": "Line 2"}

    def test_triple_quotes(self) -> None:
        """Triple quotes on keys and values."""
        raw = '"""d:1""": """Text 1"""\n"""d:2""": """Text 2"""'
        result = extract_json_object(raw)
        assert result == {"d:1": "Text 1", "d:2": "Text 2"}

    def test_asymmetric_doubled_quotes(self) -> None:
        """Keys doubled but values single, or values doubled but keys single."""
        raw = '""d:1"": "Normal val"\n"d:2": ""Doubled val""'
        result = extract_json_object(raw)
        assert result == {"d:1": "Normal val", "d:2": "Doubled val"}

    def test_doubled_single_quotes(self) -> None:
        """Doubled single quotes (''key'': ''val'')."""
        raw = "''d:1'': ''Line 1''\n''d:2'': ''Line 2''"
        result = extract_json_object(raw)
        assert result == {"d:1": "Line 1", "d:2": "Line 2"}


class TestSmartQuotesNormalization:
    """Tests for normalizing Unicode typographic quotation marks to ASCII."""

    def test_curly_double_quotes_in_json_object(self) -> None:
        """JSON object using curly double quotes as delimiters."""
        raw = "{\n  “d:1”: “Xin chào bạn!”,\n  “d:2”: “Hẹn gặp lại!”\n}"
        result = extract_json_object(raw)
        assert result == {"d:1": "Xin chào bạn!", "d:2": "Hẹn gặp lại!"}

    def test_curly_double_quotes_unbracketed(self) -> None:
        """Unbracketed key-value pairs delimited with curly quotes."""
        raw = "“d:1”: “Dòng 1”\n“d:2”: “Dòng 2”"
        result = extract_json_object(raw)
        assert result == {"d:1": "Dòng 1", "d:2": "Dòng 2"}

    def test_guillemets_quotes(self) -> None:
        """French/Russian style guillemets quotes (« and »)."""
        raw = "«d:1»: «Văn bản một»\n«d:2»: «Văn bản hai»"
        result = extract_json_object(raw)
        assert result == {"d:1": "Văn bản một", "d:2": "Văn bản hai"}

    def test_low_ninety_german_style_quotes(self) -> None:
        """German style low-9 quotes („ and “)."""
        raw = "„d:1“: „Nội dung 1“\n„d:2“: „Nội dung 2“"
        result = extract_json_object(raw)
        assert result == {"d:1": "Nội dung 1", "d:2": "Nội dung 2"}

    def test_curly_single_quotes(self) -> None:
        """Curly single quotes (\u2018 and \u2019)."""
        raw = "\u2018d:1\u2019: \u2018Xin chào\u2019\n\u2018d:2\u2019: \u2018Tạm biệt\u2019"
        result = extract_json_object(raw)
        assert result == {"d:1": "Xin chào", "d:2": "Tạm biệt"}


class TestFragmentedStreamsAndUnbracketedPairs:
    """Tests for extracting structured key-value pairs from incomplete or fragmented streams."""

    def test_unbracketed_plain_pairs(self) -> None:
        """Key-value pairs without enclosing curly braces."""
        raw = '"d:1": "Hello world"\n"d:2": "Goodbye world"'
        result = extract_json_object(raw)
        assert result == {"d:1": "Hello world", "d:2": "Goodbye world"}

    def test_stream_with_chatter_preamble_and_postamble(self) -> None:
        """LLM chatter before and after valid JSON key-value pairs."""
        raw = (
            "Sure! Here is the translation for the dialogues:\n\n"
            '"d:1": "Dòng đầu tiên",\n'
            '"d:2": "Dòng thứ hai"\n\n'
            "I hope this helps your project! Let me know if you need more."
        )
        result = extract_json_object(raw)
        assert result == {"d:1": "Dòng đầu tiên", "d:2": "Dòng thứ hai"}

    def test_truncated_json_salvage(self) -> None:
        """Truncated JSON stream where the last item is cut off mid-string."""
        raw = (
            "{\n"
            '  "d:1": "Complete line 1",\n'
            '  "d:2": "Complete line 2",\n'
            '  "d:3": "Incomplete line cut off'
        )
        result = extract_json_object(raw)
        assert "d:1" in result
        assert result["d:1"] == "Complete line 1"
        assert "d:2" in result
        assert result["d:2"] == "Complete line 2"

    def test_markdown_bullet_list_pairs(self) -> None:
        """Pairs formatted as markdown bullet list items."""
        raw = '- "d:1": "Mục danh sách 1"\n- "d:2": "Mục danh sách 2"\n- "d:3": "Mục danh sách 3"'
        result = extract_json_object(raw)
        assert result == {
            "d:1": "Mục danh sách 1",
            "d:2": "Mục danh sách 2",
            "d:3": "Mục danh sách 3",
        }

    def test_unquoted_uids_in_stream(self) -> None:
        """Unquoted dialogue UIDs followed by colon and quoted string."""
        raw = 'd:1: "Nội dung 1"\nd:2: "Nội dung 2"'
        result = extract_json_object(raw)
        assert result == {"d:1": "Nội dung 1", "d:2": "Nội dung 2"}

    def test_escaped_quotes_inside_value(self) -> None:
        """Values containing escaped quotation marks."""
        raw = '"d:1": "Cô ấy nói \\"Xin chào!\\" với tôi"\n"d:2": "Dòng 2"'
        result = extract_json_object(raw)
        assert result["d:1"] == 'Cô ấy nói "Xin chào!" với tôi'
        assert result["d:2"] == "Dòng 2"


class TestSingleItemRecoveryAndFallback:
    """Tests for single-item fallback and recovery when batch parsing fails."""

    def test_single_item_plain_text_with_fallback_key(self) -> None:
        """Plain translated text without JSON syntax using fallback_single_key."""
        raw = "Xin chào thế giới!"
        result = extract_json_object(raw, fallback_single_key="d:100")
        assert result == {"d:100": "Xin chào thế giới!"}

    def test_single_item_quoted_text_with_fallback_key(self) -> None:
        """Plain translated text wrapped in quotes without JSON syntax."""
        raw = '"Xin chào thế giới!"'
        result = extract_json_object(raw, fallback_single_key="d:100")
        assert result == {"d:100": "Xin chào thế giới!"}

    def test_single_item_unbracketed_pair(self) -> None:
        """A single key-value pair without enclosing braces."""
        raw = '"d:1": "Chỉ có một dòng duy nhất"'
        result = extract_json_object(raw)
        assert result == {"d:1": "Chỉ có một dòng duy nhất"}

    def test_single_item_recovers_via_llm_client(self) -> None:
        """Verify LLMClient.complete_json recovers plain text when fallback_single_key is given."""
        cfg = LLMConfig(base_url="http://127.0.0.1:8080/v1", model="test-model")
        client = LLMClient(cfg)

        fake_resp = {
            "choices": [{"message": {"role": "assistant", "content": "Bản dịch trực tiếp."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_resp).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res, p_tok, c_tok = client.complete_json(
                "sys", "user", fallback_single_key="single_unit_1"
            )

        assert res == {"single_unit_1": "Bản dịch trực tiếp."}
        assert p_tok == 10
        assert c_tok == 5

    def test_empty_or_chatter_only_raises_llm_error(self) -> None:
        """Conversational refusal or chatter without translation raises LLMError."""
        raw = "I cannot translate this content because of safety guidelines."
        with pytest.raises(LLMError):
            extract_json_object(raw)


class TestTagInvariantEnforcement:
    """Tests for formal Tag Invariant Enforcement: Tags(Target) ⊆ Tags(Source)."""

    def test_strip_hallucinated_size_tag(self) -> None:
        """Hallucinated {size} tags absent in source are cleanly stripped."""
        source = "Hello world!"
        target = "{size=24}Xin chào thế giới!{/size}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Xin chào thế giới!"

    def test_strip_hallucinated_font_tag(self) -> None:
        """Hallucinated {font} tags absent in source are cleanly stripped."""
        source = "Chapter 1: The Beginning"
        target = "{font=fonts/DejaVuSans.ttf}Chương 1: Khởi đầu{/font}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Chương 1: Khởi đầu"

    def test_strip_hallucinated_color_tag(self) -> None:
        """Hallucinated {color} tags absent in source are cleanly stripped."""
        source = "Warning: Danger ahead!"
        target = "{color=#ff0000}Cảnh báo: Nguy hiểm phía trước!{/color}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Cảnh báo: Nguy hiểm phía trước!"

    def test_strip_nested_hallucinated_formatting_tags(self) -> None:
        """Multiple nested hallucinated tags (font, size, color) are all stripped."""
        source = "Important announcement"
        target = (
            "{font=custom.ttf}{size=30}{color=#00ff00}{alpha=0.8}"
            "Thông báo quan trọng"
            "{/alpha}{/color}{/size}{/font}"
        )
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Thông báo quan trọng"

    def test_retain_tags_present_in_source(self) -> None:
        """Formatting tags present in source text are preserved in target."""
        source = "{b}Important{/b} and {i}italic{/i}"
        target = "{size=24}{b}Quan trọng{/b}{/size} và {i}nghiêng{/i}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "{b}Quan trọng{/b} và {i}nghiêng{/i}"

    def test_retain_source_size_tag(self) -> None:
        """If source had {size}, target {size} is retained."""
        source = "{size=30}Big Title{/size}"
        target = "{size=30}Tiêu đề lớn{/size}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "{size=30}Tiêu đề lớn{/size}"

    def test_preserve_escaped_brackets(self) -> None:
        """Escaped Ren'Py brackets ([[ and ]]) are preserved when stripping hallucinated tags."""
        source = "Press [[Enter]] to continue."
        target = "{size=20}Nhấn [[Enter]] để tiếp tục.{/size}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Nhấn [[Enter]] để tiếp tục."

    def test_preserve_escaped_braces(self) -> None:
        """Escaped Ren'Py braces ({{ and }}) are preserved."""
        source = "Template {{var}}"
        target = "{color=#ffffff}Mẫu {{var}}{/color}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Mẫu {{var}}"

    def test_strip_hallucinated_tag_with_universal_closer(self) -> None:
        """Hallucinated tag closed with universal closer {/} is stripped along with the closer."""
        source = "Simple message"
        target = "{size=25}Thông điệp đơn giản{/}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Thông điệp đơn giản"

    def test_retain_valid_tag_with_universal_closer(self) -> None:
        """Source tag closed with universal closer {/} is preserved in target."""
        source = "{i}Message{/}"
        target = "{size=20}{i}Thông điệp{/}{/size}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "{i}Thông điệp{/}"

    def test_strip_unclosed_hallucinated_tag(self) -> None:
        """Unclosed hallucinated tag at start of string is stripped."""
        source = "Hello"
        target = "{size=24}Xin chào"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Xin chào"

    def test_strip_orphaned_hallucinated_closer(self) -> None:
        """Orphaned closer for hallucinated tag is stripped."""
        source = "Hello"
        target = "Xin chào{/size}"
        cleaned = enforce_tag_invariants(source, target)
        assert cleaned == "Xin chào"

    def test_validate_raw_ignores_hallucinated_formatting_tags(self) -> None:
        """validate_raw does not return unknown_placeholder for hallucinated formatting tags."""
        source = "Hello world!"
        target = "{size=24}Xin chào thế giới!{/size}"
        issues = validate_raw(source, target)
        assert not issues

    def test_validate_raw_still_flags_unknown_variables(self) -> None:
        """validate_raw still strictly flags unknown variables (e.g. [unknown_player])."""
        source = "Hello [player]!"
        target = "Xin chào [unknown_player]!"
        issues = validate_raw(source, target)
        assert any(
            i.code == "unknown_placeholder" and "[unknown_player]" in i.detail for i in issues
        )

    def test_reconcile_translation_enforces_tag_invariants_raw(self) -> None:
        """reconcile_translation auto-heals hallucinated tags in raw text."""
        source = "Good morning!"
        target = "{font=sans.ttf}{color=#0000ff}Chào buổi sáng!{/color}{/font}"
        cleaned, issues = reconcile_translation(source, target)
        assert not issues
        assert cleaned == "Chào buổi sáng!"

    def test_reconcile_translation_enforces_tag_invariants_masked(self) -> None:
        """reconcile_translation unmasks synthetic placeholders and strips extraneous tags."""
        source = "Meet {b}[player]{/b} today!"
        # Model produced synthetic tokens but added an extra {size=20}...{/size} around them
        target = "{size=20}⟦T1⟧Gặp ⟦V1⟧⟦/T1⟧ hôm nay!{/size}"
        cleaned, issues = reconcile_translation(source, target)
        assert not issues
        assert cleaned == "{b}Gặp [player]{/b} hôm nay!"
