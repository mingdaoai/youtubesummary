#!/usr/bin/env python3
"""Tests for MCP YouTube Transcript Server"""

import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, ".")

from mcp_youtube_transcript import (
    get_youtube_transcript,
    get_video_id_from_url,
    get_available_transcript_languages,
)


class TestGetVideoIdFromUrl:
    def test_standard_url(self):
        with patch("mcp_youtube_transcript.extract_video_id") as mock_extract:
            mock_extract.return_value = "dQw4w9WgXcQ"
            result = get_video_id_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            assert result == "dQw4w9WgXcQ"

    def test_short_url(self):
        with patch("mcp_youtube_transcript.extract_video_id") as mock_extract:
            mock_extract.return_value = "dQw4w9WgXcQ"
            result = get_video_id_from_url("https://youtu.be/dQw4w9WgXcQ")
            assert result == "dQw4w9WgXcQ"

    def test_invalid_url(self):
        with patch("mcp_youtube_transcript.extract_video_id") as mock_extract:
            mock_extract.return_value = None
            result = get_video_id_from_url("https://example.com/video")
            assert "Could not extract video ID" in result


class TestGetYoutubeTranscript:
    def test_transcript_with_url(self):
        with patch("mcp_youtube_transcript.download_youtube_transcript") as mock_dl:
            mock_dl.return_value = "This is a test transcript."
            result = get_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            assert result == "This is a test transcript."

    def test_transcript_with_video_id(self):
        with patch("mcp_youtube_transcript.download_youtube_transcript") as mock_dl:
            mock_dl.return_value = "This is a test transcript."
            result = get_youtube_transcript("dQw4w9WgXcQ")
            mock_dl.assert_called_with(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language_codes=None
            )
            assert result == "This is a test transcript."

    def test_transcript_with_language_codes(self):
        with patch("mcp_youtube_transcript.download_youtube_transcript") as mock_dl:
            mock_dl.return_value = "English transcript."
            result = get_youtube_transcript(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language_codes=["en", "en-US"]
            )
            assert result == "English transcript."

    def test_transcript_not_available(self):
        with patch("mcp_youtube_transcript.download_youtube_transcript") as mock_dl:
            mock_dl.return_value = None
            result = get_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            assert "Failed to retrieve transcript" in result

    def test_transcript_exception(self):
        with patch("mcp_youtube_transcript.download_youtube_transcript") as mock_dl:
            mock_dl.side_effect = Exception("Network error")
            result = get_youtube_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            assert "Error getting transcript" in result


class TestGetAvailableTranscriptLanguages:
    def test_languages_available(self):
        with patch("mcp_youtube_transcript.get_available_languages") as mock_lang:
            mock_lang.return_value = [
                {"language": "English", "language_code": "en", "is_generated": False, "is_translatable": True},
                {"language": "Spanish", "language_code": "es", "is_generated": True, "is_translatable": True},
            ]
            result = get_available_transcript_languages("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            assert "English" in result
            assert "Spanish" in result
            assert "en" in result
            assert "es" in result

    def test_no_languages_available(self):
        with patch("mcp_youtube_transcript.get_available_languages") as mock_lang:
            mock_lang.return_value = None
            result = get_available_transcript_languages("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            assert "No transcript languages available" in result

    def test_languages_with_video_id(self):
        with patch("mcp_youtube_transcript.get_available_languages") as mock_lang:
            mock_lang.return_value = [
                {"language": "English", "language_code": "en", "is_generated": False, "is_translatable": True},
            ]
            result = get_available_transcript_languages("dQw4w9WgXcQ")
            mock_lang.assert_called_with("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


class TestMCPToolsRegistration:
    def test_mcp_instance_exists(self):
        from mcp_youtube_transcript import mcp
        assert mcp is not None
        assert mcp.name == "YouTube Transcript Server"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
