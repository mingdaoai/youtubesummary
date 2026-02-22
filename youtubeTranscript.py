#!/usr/bin/env python3

import re
import sys
import time
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter

from logging_utils import setup_logger, flush_logger

logger = setup_logger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract YouTube video ID from various YouTube URL formats.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID if found, None otherwise
    """
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def download_youtube_transcript(video_url: str, language_codes: List[str] = None, max_retries: int = 3) -> Optional[str]:
    """
    Download transcript for a YouTube video with retry mechanism.
    
    Args:
        video_url: YouTube video URL
        language_codes: List of preferred language codes (e.g., ['en', 'en-US'])
                       If None, will try to get any available transcript
        max_retries: Maximum number of retry attempts
        
    Returns:
        Transcript text if successful, None if failed
    """
    # Extract video ID from URL
    video_id = extract_video_id(video_url)
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {video_url}")
        return None
    
    logger.info(f"Extracted video ID: {video_id}")
    
    # Create API instance
    api = YouTubeTranscriptApi()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}")
            
            # Get transcript using the new API
            if language_codes:
                logger.info(f"Attempting to get transcript in languages: {language_codes}")
                try:
                    fetched_transcript = api.fetch(video_id, languages=language_codes)
                except (NoTranscriptFound, TranscriptsDisabled) as e:
                    logger.warning(f"Failed to get transcript in preferred languages: {e}")
                    logger.info("Trying to get any available transcript...")
                    fetched_transcript = api.fetch(video_id)
            else:
                logger.info("Getting any available transcript...")
                fetched_transcript = api.fetch(video_id)
            
            # Format transcript as plain text
            formatter = TextFormatter()
            transcript_text = formatter.format_transcript(fetched_transcript)
            
            logger.info(f"Successfully downloaded transcript ({len(transcript_text)} characters)")
            return transcript_text
            
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logger.error(f"Transcript not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                return None
    
    return None


def download_youtube_transcript_alternative(video_url: str) -> Optional[str]:
    """
    Alternative method to download transcript using transcript list API.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Transcript text if successful, None if failed
    """
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {video_url}")
            return None
        
        logger.info(f"Trying alternative method for video ID: {video_id}")
        
        # Create API instance
        api = YouTubeTranscriptApi()
        
        # Get transcript list
        transcript_list = api.list(video_id)
        
        # Try to find a suitable transcript
        transcript = None
        
        # First, try to find English
        try:
            transcript = transcript_list.find_transcript(['en'])
            logger.info("Found English transcript")
        except NoTranscriptFound:
            pass
        
        # If no English, try any available transcript
        if not transcript:
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                logger.info("Found manually created English transcript")
            except NoTranscriptFound:
                pass
        
        # If still no transcript, try any available
        if not transcript:
            try:
                transcript = transcript_list.find_generated_transcript(['en'])
                logger.info("Found generated English transcript")
            except NoTranscriptFound:
                pass
        
        # Last resort: get any available transcript
        if not transcript:
            available_transcripts = list(transcript_list)
            if available_transcripts:
                transcript = available_transcripts[0]
                logger.info(f"Using first available transcript: {transcript.language}")
            else:
                logger.error("No transcripts available")
                return None
        
        # Fetch the actual transcript data
        transcript_data = transcript.fetch()
        
        # Format as text
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript_data)
        
        logger.info(f"Successfully downloaded transcript using alternative method ({len(transcript_text)} characters)")
        return transcript_text
        
    except Exception as e:
        logger.error(f"Alternative method also failed: {str(e)}")
        return None


def save_transcript_to_file(transcript_text: str, video_id: str, output_dir: str = ".") -> str:
    """
    Save transcript to a text file.
    
    Args:
        transcript_text: The transcript text
        video_id: YouTube video ID
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    from pathlib import Path
    
    output_path = Path(output_dir) / f"transcript_{video_id}.txt"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        
        logger.info(f"Transcript saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error saving transcript to file: {str(e)}")
        raise


def get_available_languages(video_url: str) -> List[Dict[str, str]]:
    """
    Get list of available transcript languages for a video.
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        List of language information dictionaries
    """
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {video_url}")
            return []
        
        # Create API instance
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        languages = []
        
        for transcript in transcript_list:
            languages.append({
                'language_code': transcript.language_code,
                'language': transcript.language,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            })
        
        return languages
        
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        logger.warning(f"Transcripts not available for this video: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting available languages: {str(e)}")
        return []


def main():
    """Main function to demonstrate transcript downloading."""
    # The YouTube URL provided by the user
    video_url = "https://www.youtube.com/watch?v=4v7tJ55rzs4"
    
    logger.info(f"Starting transcript download for: {video_url}")
    
    # First, let's check what languages are available
    logger.info("Checking available transcript languages...")
    available_languages = get_available_languages(video_url)
    
    if available_languages:
        logger.info("Available transcript languages:")
        for lang in available_languages:
            logger.info(f"  - {lang['language']} ({lang['language_code']}) "
                       f"[Generated: {lang['is_generated']}, Translatable: {lang['is_translatable']}]")
    else:
        logger.warning("No transcript languages found or error occurred")
    
    # Download transcript (prefer English, fallback to any available)
    transcript_text = download_youtube_transcript(
        video_url, 
        language_codes=['en', 'en-US', 'en-GB']
    )
    
    # If the main method failed, try the alternative method
    if not transcript_text:
        logger.info("Main method failed, trying alternative method...")
        transcript_text = download_youtube_transcript_alternative(video_url)
    
    if transcript_text:
        # Extract video ID for filename
        video_id = extract_video_id(video_url)
        
        # Save to file
        if video_id:
            output_file = save_transcript_to_file(transcript_text, video_id)
            logger.info(f"Transcript successfully downloaded and saved to: {output_file}")
        else:
            logger.error("Could not extract video ID for saving file")
        
        # Print first 500 characters as preview
        logger.info("Transcript preview (first 500 characters):")
        print("-" * 50)
        print(transcript_text[:500])
        if len(transcript_text) > 500:
            print("...")
        print("-" * 50)
        
    else:
        logger.error("Failed to download transcript")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
