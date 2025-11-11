#!/usr/bin/env python3
"""
MCP Server for YouTube Transcript Extraction

This MCP server provides tools to get YouTube video transcripts based on URL or video ID.
"""

import re
import logging
from typing import Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    try:
        from mcp import FastMCP
    except ImportError:
        raise ImportError("mcp library not installed. Run: pip install mcp")

from youtubeTranscript import extract_video_id, download_youtube_transcript, get_available_languages

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP("YouTube Transcript Server")


@mcp.tool()
def get_youtube_transcript(url: str, language_codes: Optional[list[str]] = None) -> str:  # type: ignore
    """
    Get the transcript of a YouTube video from its URL or video ID.
    
    Args:
        url: YouTube URL (e.g., "https://www.youtube.com/watch?v=VIDEO_ID") or just video ID
        language_codes: Optional list of preferred language codes (e.g., ["en", "en-US"])
                       If None, will use any available transcript
    
    Returns:
        Transcript text as a string
        
    Example:
        url: "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        language_codes: ["en", "en-US"]
    """
    try:
        logger.info(f"Getting transcript for: {url}")
        
        # If the input looks like just a video ID, add the YouTube URL prefix
        if not url.startswith(('http://', 'https://')):
            url = f"https://www.youtube.com/watch?v={url}"
        
        # Download transcript
        transcript = download_youtube_transcript(url, language_codes=language_codes)
        
        if transcript:
            logger.info(f"Successfully retrieved transcript ({len(transcript)} characters)")
            return transcript
        else:
            error_msg = "Failed to retrieve transcript. The video may not have captions available."
            logger.error(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Error getting transcript: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


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
        return error_msg


if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run(transport="stdio")
