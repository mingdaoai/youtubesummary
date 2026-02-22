#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "youtube-transcript-api>=0.6.2",
#     "requests>=2.25.0",
#     "boto3>=1.26.0",
#     "yt-dlp>=2023.1.6",
#     "mcp>=0.3.0",
#     "beautifulsoup4>=4.14.3",
#     "google-genai>=0.3.0",
#     "openai-whisper>=20231117",
# ]
# ///
"""
MCP Server for YouTube Transcript Extraction

This MCP server provides tools to get YouTube video transcripts based on URL or video ID.
Uses caching for extremely fast second-time retrieval.
"""

import logging
import re
import time
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp import FastMCP
    except ImportError:
        raise ImportError("mcp library not installed. Run: pip install mcp")

from youtubeSummarize import get_transcript, extract_video_id, get_available_languages
from logging_utils import setup_logger, flush_logger

# MCP server uses stdio for protocol - disable console logging to avoid interference
# Logs will still be written to file
logger = setup_logger(__name__)
# Remove console handler (stdout/stderr) to prevent MCP protocol corruption
for handler in logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        logger.removeHandler(handler)

# FastMCP adds a RichHandler to the root logger - remove it to prevent stdout interference
import logging as _logging
for handler in _logging.root.handlers[:]:
    if hasattr(handler, '__class__') and 'RichHandler' in handler.__class__.__name__:
        _logging.root.removeHandler(handler)
# Also disable propagation to prevent logs bubbling up to root
logger.propagate = False

# Initialize the MCP server
mcp = FastMCP("YouTube Transcript Server")


@mcp.tool()
def get_youtube_transcript(
    url: str,
    language_codes: Optional[list[str]] = None,
    offset: int = 0,
    limit: Optional[int] = None
) -> str:  # type: ignore
    """
    Get the transcript of a YouTube video from its URL or video ID.
    
    Args:
        url: YouTube URL (e.g., "https://www.youtube.com/watch?v=VIDEO_ID") or just video ID
        language_codes: Optional list of preferred language codes (e.g., ["en", "en-US"])
                       If None, will use any available transcript
        offset: Starting character position for pagination (default: 0)
        limit: Maximum number of characters to return. If None, returns full transcript.
               Use for large transcripts to avoid token limits.
    
    Returns:
        JSON string with transcript and pagination metadata:
        - content: The transcript text (paginated if limit specified)
        - total_length: Total characters in full transcript
        - offset: Current offset position
        - returned_length: Characters returned in this response
        - has_more: Boolean indicating if more content available
        
    Example:
        url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        language_codes: ["en", "en-US"]
        offset: 0
        limit: 10000
    """
    import json as _json
    try:
        start_time = time.time()
        logger.info(f"Getting transcript for: {url}, offset={offset}, limit={limit}")
        
        if url.startswith(('http://', 'https://')):
            video_id = extract_video_id(url)
        else:
            video_id = url
        
        if not video_id:
            error_msg = f"Could not extract video ID from URL: {url}"
            logger.error(error_msg)
            return _json.dumps({"error": error_msg})
        
        result = get_transcript(video_id)
        
        elapsed = time.time() - start_time
        
        if result and 'transcript' in result:
            full_transcript = result['transcript']
            total_length = len(full_transcript)
            
            if limit is None:
                paginated_content = full_transcript
                returned_length = total_length
                has_more = False
                actual_offset = 0
            else:
                actual_offset = max(0, min(offset, total_length))
                paginated_content = full_transcript[actual_offset:actual_offset + limit]
                returned_length = len(paginated_content)
                has_more = (actual_offset + limit) < total_length
            
            response = {
                "content": paginated_content,
                "total_length": total_length,
                "offset": actual_offset if limit else 0,
                "returned_length": returned_length,
                "has_more": has_more,
                "video_id": video_id,
                "language": result.get('language'),
                "title": result.get('title'),
                "fetch_time_seconds": round(elapsed, 2)
            }
            
            logger.info(f"Retrieved transcript ({returned_length}/{total_length} chars) in {elapsed:.2f}s")
            return _json.dumps(response, ensure_ascii=False)
        else:
            error_msg = "Failed to retrieve transcript. The video may not have captions available."
            logger.error(error_msg)
            return _json.dumps({"error": error_msg})
            
    except Exception as e:
        error_msg = f"Error getting transcript: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return _json.dumps({"error": error_msg})


@mcp.tool()
def get_video_id_from_url(url: str) -> str:  # type: ignore
    """
    Extract the YouTube video ID from a URL.
    
    Args:
        url: YouTube URL in any format
        
    Returns:
        Video ID string
        
    Example:
        url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    """
    try:
        video_id = extract_video_id(url)
        if video_id:
            logger.info(f"Extracted video ID: {video_id} from URL: {url}")
            return video_id
        else:
            error_msg = f"Could not extract video ID from URL: {url}"
            logger.error(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error extracting video ID: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return error_msg


@mcp.tool()
def get_available_transcript_languages(url: str) -> str:  # type: ignore
    """
    Get the list of available transcript languages for a YouTube video.
    
    Args:
        url: YouTube URL or video ID
        
    Returns:
        Formatted string listing available languages
        
    Example:
        url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    """
    try:
        # If the input looks like just a video ID, add the YouTube URL prefix
        if not url.startswith(('http://', 'https://')):
            url = f"https://www.youtube.com/watch?v={url}"
        
        languages = get_available_languages(url)
        
        if languages:
            result = f"Available transcript languages for video:\n"
            for lang in languages:
                generated = "✓" if lang['is_generated'] else "✗"
                translatable = "✓" if lang['is_translatable'] else "✗"
                result += f"  - {lang['language']} ({lang['language_code']}) [Generated: {generated}, Translatable: {translatable}]\n"
            logger.info(f"Found {len(languages)} available transcript languages")
            return result.strip()
        else:
            error_msg = "No transcript languages available or video not found."
            logger.warning(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Error getting available languages: {str(e)}"
        logger.error(error_msg, exc_info=True)
        flush_logger(logger)
        return error_msg


if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run(transport="stdio")
