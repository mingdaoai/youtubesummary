#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "youtube-transcript-api>=0.6.2",
#     "requests>=2.25.0",
#     "boto3>=1.26.0",
#     "yt-dlp>=2023.1.6",
#     "openai-whisper>=20231117",
#     "mcp>=0.3.0",
#     "playwright",
#     "beautifulsoup4>=4.14.3",
#     "google-genai>=1.64.0",
# ]
# ///
"""
MCP Server for YouTube Video Summarization

Provides tools to:
- Summarize YouTube videos
- Answer questions about videos
- Get video transcripts
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Optional

from logging_utils import setup_logger, flush_logger
import logging

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp import FastMCP
    except ImportError:
        raise ImportError("mcp library not installed. Run: pip install mcp")

from google import genai
from google.genai import types
import yt_dlp

from youtubeSummarize import (
    extract_video_id,
    get_cached_transcript,
    cache_transcript,
    try_youtube_transcript,
    download_video_if_needed,
    detect_and_confirm_language,
    transcribe_with_whisper,
    PROXY_CONFIG,
    YTDL_PROXY_OPTS,
    CACHE_DIR,
    summarize_transcript,
    answer_question_with_chunking,
    GEMINI_MODEL_NAME,
)

logger = setup_logger(__name__)

mcp = FastMCP("YouTube Summarizer")

_video_contexts: dict = {}

def create_gemini_client():
    try:
        api_key_path = os.path.expanduser("~/.mingdaoai/gemini.key")
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Error creating Gemini client: {e}")
        raise

def download_video_automatic(video_id: str, video_url: str, video_path: Path) -> str:
    base_opts = {"quiet": True}
    base_opts.update(YTDL_PROXY_OPTS)
    
    if video_path.exists():
        logger.info(f"Video file {video_path} already exists. Using cached mp4.")
        try:
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info.get('title', '')
        except Exception as e:
            logger.error(f"yt-dlp failed to extract info: {e}")
            return ''
    
    logger.info(f"Downloading video automatically...")
    format_list = [
        '18',
        '22',
        'best[ext=mp4]/best',
        'best',
        'bestvideo+bestaudio/best',
    ]
    cookies_path = os.path.join(os.getcwd(), 'cookies.txt')
    last_exception = None
    
    for fmt in format_list:
        ydl_opts = {
            'outtmpl': str(video_path),
            'format': fmt,
            'merge_output_format': 'mp4',
        }
        ydl_opts.update(YTDL_PROXY_OPTS)
        
        if os.path.exists(cookies_path):
            ydl_opts['cookiefile'] = cookies_path
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                title = info.get('title', '')
            logger.info(f"Downloaded video with format: {fmt}")
            return title
        except Exception as e:
            logger.error(f"yt-dlp failed with format '{fmt}': {e}")
            last_exception = e
            continue
    
    logger.error("All format attempts failed.")
    raise last_exception

def get_transcript_auto(video_id: str) -> dict:
    result = try_youtube_transcript(video_id)
    if result:
        cache_file = CACHE_DIR / f"{video_id}.json"
        title = None
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    title = cache_data.get("title")
            except Exception:
                title = None
        if not title:
            try:
                ydl_opts = {"quiet": True}
                ydl_opts.update(YTDL_PROXY_OPTS)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                    title = info.get('title')
            except Exception:
                title = None
        if title:
            cache_transcript(video_id, result["transcript"], result["language"], title)
        return {**result, "title": title} if title else result
    
    logger.info("Downloading video for transcription...")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = CACHE_DIR / f"{video_id}.mp4"
    
    title = download_video_automatic(video_id, video_url, video_path)
    language = detect_and_confirm_language(video_id, title)
    
    mp3_path = CACHE_DIR / f"{video_id}.mp3"
    if not mp3_path.exists():
        import subprocess
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "192k", str(mp3_path)]
        logger.info(f"Extracting mp3 from video...")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    logger.info("Transcribing with Whisper...")
    transcript, whisper_language = transcribe_with_whisper(mp3_path)
    cache_transcript(video_id, transcript, whisper_language, title)
    
    try:
        if video_path.exists():
            video_path.unlink()
        if mp3_path.exists():
            mp3_path.unlink()
    except Exception:
        pass
    
    return {"transcript": transcript, "language": whisper_language, "title": title}

def get_or_load_transcript(video_id: str) -> dict:
    cached = get_cached_transcript(video_id)
    if cached and 'transcript' in cached and 'language' in cached:
        if 'title' not in cached or not cached['title']:
            try:
                ydl_opts = {"quiet": True}
                ydl_opts.update(YTDL_PROXY_OPTS)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                    title = info.get('title')
                    if title:
                        cache_transcript(video_id, cached['transcript'], cached['language'], title)
                        cached['title'] = title
            except Exception:
                pass
        logger.info("Loaded transcript from cache.")
        return cached
    logger.info("Downloading transcript...")
    return get_transcript_auto(video_id)


@mcp.tool()
def summarize_youtube_video(url: str, force_refresh: bool = False) -> str:  # type: ignore
    """
    Summarize a YouTube video from its URL.
    
    Automatically downloads and transcribes the video if no transcript is available.
    
    Args:
        url: YouTube URL (e.g., "https://www.youtube.com/watch?v=VIDEO_ID") or video ID
        force_refresh: If True, re-download and re-summarize even if cached
    
    Returns:
        Video summary as markdown text
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = f"https://www.youtube.com/watch?v={url}"
        
        video_id = extract_video_id(url)
        logger.info(f"Summarizing video: {video_id}")
        
        if force_refresh:
            cache_file = CACHE_DIR / f"{video_id}.json"
            if cache_file.exists():
                cache_file.unlink()
        
        transcript_data = get_or_load_transcript(video_id)
        transcript = transcript_data['transcript']
        title = transcript_data.get('title', 'Unknown Title')
        
        model = create_gemini_client()
        summary = summarize_transcript(model, transcript)
        
        _video_contexts[video_id] = {
            'transcript': transcript,
            'summary': summary,
            'title': title,
            'language': transcript_data.get('language', 'en')
        }
        
        result = f"# {title}\n\n## Summary\n\n{summary}"
        logger.info(f"Successfully summarized video: {video_id}")
        return result
        
    except Exception as e:
        error_msg = f"Error summarizing video: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return error_msg


@mcp.tool()
def ask_about_video(url: str, question: str) -> str:  # type: ignore
    """
    Ask a question about a YouTube video.
    
    Loads the video transcript and answers questions based on its content.
    Automatically downloads and transcribes if needed.
    
    Args:
        url: YouTube URL or video ID
        question: Question about the video content
    
    Returns:
        Answer to the question
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = f"https://www.youtube.com/watch?v={url}"
        
        video_id = extract_video_id(url)
        logger.info(f"Answering question for video: {video_id}")
        
        if video_id not in _video_contexts:
            transcript_data = get_or_load_transcript(video_id)
            transcript = transcript_data['transcript']
            title = transcript_data.get('title', 'Unknown Title')
            
            model = create_gemini_client()
            summary = summarize_transcript(model, transcript)
            
            _video_contexts[video_id] = {
                'transcript': transcript,
                'summary': summary,
                'title': title,
                'language': transcript_data.get('language', 'en'),
                'chat_history': []
            }
        
        ctx = _video_contexts[video_id]
        model = create_gemini_client()
        
        initial_context = [
            {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
            {"role": "user", "content": f"Here is a summary of the video: {ctx['summary']}\n\nThe full transcript is available if you need specific details."}
        ]
        
        chat_history = ctx.get('chat_history', [])
        
        answer = answer_question_with_chunking(
            model, 
            question, 
            ctx['transcript'], 
            ctx['summary'], 
            chat_history, 
            initial_context
        )
        
        ctx['chat_history'].append({"question": question, "answer": answer})
        
        logger.info(f"Successfully answered question for video: {video_id}")
        return answer
        
    except Exception as e:
        error_msg = f"Error answering question: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return error_msg


@mcp.tool()
def get_video_transcript(url: str) -> str:  # type: ignore
    """
    Get the full transcript of a YouTube video.
    
    Automatically downloads and transcribes if no transcript is available.
    
    Args:
        url: YouTube URL or video ID
    
    Returns:
        Full transcript text with metadata
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = f"https://www.youtube.com/watch?v={url}"
        
        video_id = extract_video_id(url)
        logger.info(f"Getting transcript for video: {video_id}")
        
        transcript_data = get_or_load_transcript(video_id)
        transcript = transcript_data['transcript']
        title = transcript_data.get('title', 'Unknown Title')
        language = transcript_data.get('language', 'en')
        
        result = f"# {title}\n\n**Language:** {language}\n\n## Transcript\n\n{transcript}"
        logger.info(f"Successfully retrieved transcript for video: {video_id}")
        return result
        
    except Exception as e:
        error_msg = f"Error getting transcript: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return error_msg


@mcp.tool()
def get_video_info(url: str) -> str:  # type: ignore
    """
    Get basic information about a YouTube video without summarizing.
    
    Args:
        url: YouTube URL or video ID
    
    Returns:
        Video title and metadata
    """
    try:
        if not url.startswith(('http://', 'https://')):
            url = f"https://www.youtube.com/watch?v={url}"
        
        video_id = extract_video_id(url)
        
        ydl_opts = {"quiet": True}
        ydl_opts.update(YTDL_PROXY_OPTS)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        
        title = info.get('title', 'Unknown')
        duration = info.get('duration', 0)
        uploader = info.get('uploader', 'Unknown')
        description = info.get('description', '')[:500]
        
        duration_str = f"{duration // 60}:{duration % 60:02d}" if duration else "Unknown"
        
        result = f"""# {title}

**Video ID:** {video_id}
**Duration:** {duration_str}
**Uploader:** {uploader}

**Description Preview:**
{description}..."""
        
        return result
        
    except Exception as e:
        error_msg = f"Error getting video info: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return error_msg


if __name__ == "__main__":
    mcp.run(transport="stdio")
