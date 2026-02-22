#!/usr/bin/env python3
"""Tests for MCP YouTube Summarizer Server"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, ".")


class TestSummarizeYoutubeVideo:
    def test_summarize_with_url(self):
        mock_transcript_data = {
            "transcript": "This is a test transcript about investing.",
            "title": "Test Video Title",
            "language": "en"
        }
        with patch("mcp_youtube_summarizer.get_or_load_transcript") as mock_load, \
             patch("mcp_youtube_summarizer.create_gemini_client") as mock_client, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
            mock_load.return_value = mock_transcript_data
            mock_extract.return_value = "test123"
            mock_model = MagicMock()
            mock_client.return_value = mock_model
            with patch("mcp_youtube_summarizer.summarize_transcript") as mock_summarize:
                mock_summarize.return_value = "This is a summary."
                from mcp_youtube_summarizer import summarize_youtube_video
                result = summarize_youtube_video("https://www.youtube.com/watch?v=test123")
                assert "Test Video Title" in result
                assert "This is a summary" in result

    def test_summarize_with_video_id(self):
        mock_transcript_data = {
            "transcript": "Test transcript.",
            "title": "Test Title",
            "language": "en"
        }
        with patch("mcp_youtube_summarizer.get_or_load_transcript") as mock_load, \
             patch("mcp_youtube_summarizer.create_gemini_client") as mock_client, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
            mock_load.return_value = mock_transcript_data
            mock_extract.return_value = "abc123"
            mock_model = MagicMock()
            mock_client.return_value = mock_model
            with patch("mcp_youtube_summarizer.summarize_transcript") as mock_summarize:
                mock_summarize.return_value = "Summary result."
                from mcp_youtube_summarizer import summarize_youtube_video
                summarize_youtube_video("abc123")
                mock_extract.assert_called()

    def test_summarize_force_refresh(self):
        with patch("mcp_youtube_summarizer.CACHE_DIR") as mock_cache_dir, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
            mock_cache_dir.__truediv__ = lambda self, x: Path("/tmp/test")
            mock_extract.return_value = "test123"
            with patch("pathlib.Path.exists") as mock_exists, \
                 patch("pathlib.Path.unlink") as mock_unlink:
                mock_exists.return_value = True
                from mcp_youtube_summarizer import summarize_youtube_video
                with patch("mcp_youtube_summarizer.get_or_load_transcript") as mock_load, \
                     patch("mcp_youtube_summarizer.create_gemini_client"), \
                     patch("mcp_youtube_summarizer.summarize_transcript"):
                    mock_load.return_value = {"transcript": "t", "title": "T", "language": "en"}
                    summarize_youtube_video("test123", force_refresh=True)

    def test_summarize_error_handling(self):
        with patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
            mock_extract.side_effect = Exception("Invalid URL")
            from mcp_youtube_summarizer import summarize_youtube_video
            result = summarize_youtube_video("invalid")
            assert "Error summarizing video" in result


class TestAskAboutVideo:
    def test_ask_question(self):
        from mcp_youtube_summarizer import _video_contexts
        _video_contexts.clear()
        _video_contexts["test123"] = {
            "transcript": "The video discusses AI.",
            "summary": "AI summary",
            "title": "AI Video",
            "language": "en",
            "chat_history": []
        }
        with patch("mcp_youtube_summarizer.create_gemini_client") as mock_client, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract, \
             patch("mcp_youtube_summarizer.answer_question_with_chunking") as mock_answer:
            mock_extract.return_value = "test123"
            mock_answer.return_value = "AI is discussed in the video."
            from mcp_youtube_summarizer import ask_about_video
            result = ask_about_video("test123", "What is this video about?")
            assert "AI is discussed" in result

    def test_ask_question_loads_transcript_if_needed(self):
        from mcp_youtube_summarizer import _video_contexts
        _video_contexts.clear()
        mock_transcript_data = {
            "transcript": "New transcript.",
            "title": "New Video",
            "language": "en"
        }
        with patch("mcp_youtube_summarizer.get_or_load_transcript") as mock_load, \
             patch("mcp_youtube_summarizer.create_gemini_client") as mock_client, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract, \
             patch("mcp_youtube_summarizer.summarize_transcript") as mock_summarize, \
             patch("mcp_youtube_summarizer.answer_question_with_chunking") as mock_answer:
            mock_extract.return_value = "new123"
            mock_load.return_value = mock_transcript_data
            mock_summarize.return_value = "Summary"
            mock_answer.return_value = "Answer"
            from mcp_youtube_summarizer import ask_about_video
            result = ask_about_video("new123", "Question?")
            mock_load.assert_called_once()


class TestGetVideoTranscript:
    def test_get_transcript(self):
        mock_transcript_data = {
            "transcript": "Full transcript content here.",
            "title": "Test Video",
            "language": "en"
        }
        with patch("mcp_youtube_summarizer.get_or_load_transcript") as mock_load, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
            mock_load.return_value = mock_transcript_data
            mock_extract.return_value = "test123"
            from mcp_youtube_summarizer import get_video_transcript
            result = get_video_transcript("https://www.youtube.com/watch?v=test123")
            assert "Test Video" in result
            assert "Full transcript content" in result
            assert "Language" in result

    def test_get_transcript_with_video_id(self):
        mock_transcript_data = {
            "transcript": "Transcript.",
            "title": "Title",
            "language": "es"
        }
        with patch("mcp_youtube_summarizer.get_or_load_transcript") as mock_load, \
             patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
            mock_load.return_value = mock_transcript_data
            mock_extract.return_value = "abc123"
            from mcp_youtube_summarizer import get_video_transcript
            result = get_video_transcript("abc123")
            assert "es" in result


class TestGetVideoInfo:
    def test_get_video_info(self):
        mock_info = {
            "title": "Test Video Title",
            "duration": 300,
            "uploader": "Test Channel",
            "description": "This is a test description for the video."
        }
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_ydl_instance.__exit__ = MagicMock(return_value=False)
            mock_ydl_instance.extract_info.return_value = mock_info
            mock_ydl.return_value = mock_ydl_instance
            with patch("mcp_youtube_summarizer.extract_video_id") as mock_extract:
                mock_extract.return_value = "test123"
                from mcp_youtube_summarizer import get_video_info
                result = get_video_info("https://www.youtube.com/watch?v=test123")
                assert "Test Video Title" in result
                assert "Test Channel" in result
                assert "5:00" in result

    def test_get_video_info_error(self):
        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_ydl_instance.__exit__ = MagicMock(return_value=False)
            mock_ydl_instance.extract_info.side_effect = Exception("Video not found")
            mock_ydl.return_value = mock_ydl_instance
            from mcp_youtube_summarizer import get_video_info
            result = get_video_info("invalid_url")
            assert "Error getting video info" in result


class TestMCPToolsRegistration:
    def test_mcp_instance_exists(self):
        from mcp_youtube_summarizer import mcp
        assert mcp is not None
        assert mcp.name == "YouTube Summarizer"


class TestHelperFunctions:
    def test_create_gemini_client(self):
        with patch("builtins.open", MagicMock(read_data="test-api-key")):
            with patch("google.generativeai.GenerativeModel"):
                with patch("google.generativeai.configure"):
                    from mcp_youtube_summarizer import create_gemini_client
                    client = create_gemini_client()
                    assert client is not None

    def test_create_gemini_client_error(self):
        with patch("builtins.open", side_effect=FileNotFoundError("No key file")):
            from mcp_youtube_summarizer import create_gemini_client
            with pytest.raises(Exception):
                create_gemini_client()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
