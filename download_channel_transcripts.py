#!/usr/bin/env python3
"""
Download YouTube transcripts for all videos in a channel.

This script:
1. Takes a YouTube channel URL or channel ID
2. Gets all videos from the channel using yt-dlp
3. Downloads transcripts for each video (reusing existing cache when available)
4. Saves transcripts in an organized directory structure

Usage:
    python download_channel_transcripts.py <channel_url_or_id>
    python download_channel_transcripts.py https://www.youtube.com/@channelname
    python download_channel_transcripts.py UCxxxxxxxxxxxxxxxxxxxxx
"""

import os
import sys
import json
import logging
import argparse
import time
import datetime
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse
import yt_dlp
import re
import boto3
import requests

# Import functions from youtubeSummarize.py
from youtubeSummarize import get_transcript, CACHE_DIR, extract_video_id, detect_and_confirm_language

# ========== CONFIGURATION ==========
OUTPUT_DIR = Path(__file__).parent / "channel_transcripts"
OUTPUT_DIR.mkdir(exist_ok=True)

# AWS Configuration
AWS_S3_BUCKET = 'mdaudiosound'
AWS_TRANSCRIBE_REGION = 'us-west-2'

# Cache Configuration
CACHE_DURATION_DAYS = 7  # Default cache duration for video/audio files (in days)
AWS_TRANSCRIBE_CACHE_DAYS = 30  # Cache duration for AWS Transcribe JSON responses (in days)

# Format preference cache file
FORMAT_CACHE_FILE = CACHE_DIR / "last_successful_format.txt"

# ========== FORMAT PREFERENCE FUNCTIONS ==========
def get_last_successful_format() -> Optional[str]:
    """Get the last successful download format from cache."""
    if FORMAT_CACHE_FILE.exists():
        try:
            with open(FORMAT_CACHE_FILE, 'r', encoding='utf-8') as f:
                format_str = f.read().strip()
                if format_str:
                    return format_str
        except Exception:
            pass
    return None

def save_successful_format(format_str: Optional[str]):
    """Save the successful download format to cache."""
    try:
        if format_str:
            with open(FORMAT_CACHE_FILE, 'w', encoding='utf-8') as f:
                f.write(format_str)
        elif FORMAT_CACHE_FILE.exists():
            # Clear cache if format_str is None
            FORMAT_CACHE_FILE.unlink()
    except Exception:
        pass

# ========== CACHE MANAGEMENT FUNCTIONS ==========
def is_file_expired(file_path: Path, cache_days: int = CACHE_DURATION_DAYS) -> bool:
    """Check if a cached file is expired based on its modification time."""
    if not file_path.exists():
        return True
    
    file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
    return file_age.days >= cache_days

def get_file_age_str(file_path: Path) -> str:
    """Get a human-readable string for file age."""
    if not file_path.exists():
        return "N/A"
    file_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
    if file_age.days > 0:
        return f"{file_age.days} day{'s' if file_age.days > 1 else ''}"
    elif file_age.seconds >= 3600:
        hours = file_age.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''}"
    elif file_age.seconds >= 60:
        minutes = file_age.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        return f"{file_age.seconds} second{'s' if file_age.seconds > 1 else ''}"

def cleanup_expired_cache_files(video_id: str, cache_days: int = CACHE_DURATION_DAYS):
    """Clean up expired video and audio cache files for a given video ID."""
    video_path = CACHE_DIR / f"{video_id}.mp4"
    mp3_path = CACHE_DIR / f"{video_id}.mp3"
    aws_cache_file = CACHE_DIR / f"{video_id}_aws_transcribe.json"
    
    cleaned = False
    if video_path.exists() and is_file_expired(video_path, cache_days):
        try:
            age_str = get_file_age_str(video_path)
            video_path.unlink()
            logger.info(f"🗑️  Deleted expired video cache ({age_str} old): {video_path.name}")
            cleaned = True
        except Exception as e:
            logger.warning(f"⚠️  Failed to delete expired video cache: {e}")
    
    if mp3_path.exists() and is_file_expired(mp3_path, cache_days):
        try:
            age_str = get_file_age_str(mp3_path)
            mp3_path.unlink()
            logger.info(f"🗑️  Deleted expired audio cache ({age_str} old): {mp3_path.name}")
            cleaned = True
        except Exception as e:
            logger.warning(f"⚠️  Failed to delete expired audio cache: {e}")
    
    # Also clean up expired AWS Transcribe JSON cache
    if aws_cache_file.exists() and is_file_expired(aws_cache_file, AWS_TRANSCRIBE_CACHE_DAYS):
        try:
            age_str = get_file_age_str(aws_cache_file)
            aws_cache_file.unlink()
            logger.info(f"🗑️  Deleted expired AWS Transcribe cache ({age_str} old): {aws_cache_file.name}")
            cleaned = True
        except Exception as e:
            logger.warning(f"⚠️  Failed to delete expired AWS Transcribe cache: {e}")
    
    return cleaned

# ========== AWS TRANSCRIBE HELPER FUNCTIONS ==========
def normalize_language_code_for_aws(language: str) -> str:
    """Normalize language code to AWS Transcribe compatible format."""
    # Map common language codes to AWS Transcribe supported codes
    language_map = {
        'zh': 'zh-CN',  # Default Chinese to Simplified Chinese
        'zh-Hans': 'zh-CN',
        'zh-Hant': 'zh-TW',
        'en': 'en-US',  # Default English to US English
        'es': 'es-ES',  # Default Spanish to Spain Spanish
        'fr': 'fr-FR',  # Default French to France French
        'de': 'de-DE',  # Default German to Germany German
        'pt': 'pt-BR',  # Default Portuguese to Brazilian Portuguese
        'ja': 'ja-JP',
        'ko': 'ko-KR',
        'ar': 'ar-SA',
        'hi': 'hi-IN',
        'ru': 'ru-RU',
    }
    
    # If already in correct format, return as-is
    if '-' in language and len(language) >= 5:
        return language
    
    # Map to AWS compatible code
    normalized = language_map.get(language.lower(), language)
    
    # If still not in correct format, try to construct it
    if normalized == language and '-' not in normalized:
        # Try common patterns
        if normalized.startswith('zh'):
            normalized = 'zh-CN'
        elif normalized.startswith('en'):
            normalized = 'en-US'
    
    return normalized

def upload_to_s3_if_needed(file_path: Path, s3, bucket: str, s3_key: str):
    """Upload file to S3 if it doesn't already exist."""
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        logger.info(f"S3 object '{bucket}/{s3_key}' already exists. Skipping upload.")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.info(f"Uploading {file_path} to S3 bucket '{bucket}' at '{s3_key}'")
            try:
                s3.upload_file(str(file_path), bucket, s3_key)
                logger.info("Upload to S3 completed.")
            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}", exc_info=True)
                raise
        else:
            logger.error(f"Error checking S3 object: {e}", exc_info=True)
            raise

def start_or_resume_transcribe_job(transcribe, job_name: str, job_uri: str, language: str):
    """Start or resume an AWS Transcribe job."""
    job_exists = False
    try:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_exists = True
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        logger.info(f"Found existing AWS Transcribe job: {job_name} with status: {job_status}")
    except transcribe.exceptions.BadRequestException:
        job_exists = False
    if not job_exists:
        logger.info(f"Starting AWS Transcribe job: {job_name} for URI: {job_uri} with language: {language}")
        try:
            transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={'MediaFileUri': job_uri},
                MediaFormat='mp3',  # Changed from mp4 to mp3 for audio files
                LanguageCode=language,
                Settings={
                    'ShowSpeakerLabels': True,
                    'MaxSpeakerLabels': 2,
                }
            )
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        except Exception as e:
            logger.error(f"Failed to start transcription job: {e}", exc_info=True)
            raise
    return status

def wait_for_transcribe_job(transcribe, job_name: str, status):
    """Wait for AWS Transcribe job to complete."""
    logger.info(f"Waiting for AWS Transcribe job '{job_name}' to complete...")
    while True:
        try:
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            logger.info(f"Transcription job status: {job_status}")
            if job_status in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        except Exception as e:
            logger.error(f"Error while checking transcription job status: {e}", exc_info=True)
            raise
    return status

def transcribe_with_aws_transcribe(mp3_path: Path, video_id: str, language: str) -> tuple[str, str]:
    """Transcribe audio file using AWS Transcribe. Returns (transcript_text, language)."""
    # Check for cached AWS Transcribe JSON response
    aws_cache_file = CACHE_DIR / f"{video_id}_aws_transcribe.json"
    transcript_data = None
    
    if aws_cache_file.exists() and not is_file_expired(aws_cache_file, AWS_TRANSCRIBE_CACHE_DAYS):
        logger.info(f"📁 Using cached AWS Transcribe JSON (age: {get_file_age_str(aws_cache_file)})")
        try:
            with open(aws_cache_file, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️  Failed to read cached AWS Transcribe JSON: {e}, will fetch fresh")
            transcript_data = None
    
    # If no valid cache, fetch from AWS
    if transcript_data is None:
        logger.info(f"🎤 Transcribing with AWS Transcribe (this may take a while)...")
        sys.stdout.flush()
        
        # Upload to S3
        s3 = boto3.client('s3')
        s3_key = f'input/{video_id}.mp3'
        upload_to_s3_if_needed(mp3_path, s3, AWS_S3_BUCKET, s3_key)
        
        # Start transcription job
        transcribe = boto3.client('transcribe', AWS_TRANSCRIBE_REGION)
        job_name = f"transcribe-job-{video_id}"
        job_uri = f"s3://{AWS_S3_BUCKET}/{s3_key}"
        status = start_or_resume_transcribe_job(transcribe, job_name, job_uri, language)
        
        # Wait for completion
        status = wait_for_transcribe_job(transcribe, job_name, status)
        
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            logger.info(f"Fetching transcript from URL: {transcript_uri}")
            try:
                response = requests.get(transcript_uri)
                response.raise_for_status()
                transcript_data = response.json()
                
                # Cache the raw JSON response
                try:
                    with open(aws_cache_file, 'w', encoding='utf-8') as f:
                        json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"💾 Cached AWS Transcribe JSON response")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to cache AWS Transcribe JSON: {e}")
            except Exception as e:
                logger.error(f"Failed to fetch transcript from URL: {e}", exc_info=True)
                raise
        else:
            logger.error("Transcription job failed.")
            raise Exception("AWS Transcribe job failed")
    
    # Extract transcript text - use the complete transcript from AWS Transcribe
    # AWS Transcribe provides a complete, punctuated transcript in results.transcripts
    if 'results' in transcript_data and 'transcripts' in transcript_data['results']:
        if len(transcript_data['results']['transcripts']) > 0:
            transcript = transcript_data['results']['transcripts'][0]['transcript']
        else:
            raise Exception("No transcript found in AWS Transcribe response")
    else:
        raise Exception("Invalid AWS Transcribe response structure")
    
    logger.info(f"✅ AWS Transcribe transcription complete")
    return transcript, language

# ========== LOGGING ==========
# Global logger - will be configured with file handler in download_channel_transcripts()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with immediate flushing
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
console_handler.flush = lambda: sys.stdout.flush()  # Force immediate flush
logger.addHandler(console_handler)

def setup_file_logging(log_file: Path):
    """
    Set up file logging handler for a specific channel download session.
    
    Args:
        log_file: Path to the log file
    """
    # Remove existing file handlers to avoid duplicates
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    
    # Create file handler with immediate flushing
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Force immediate flush after each log
    original_emit = file_handler.emit
    def emit_with_flush(record):
        original_emit(record)
        file_handler.flush()
    file_handler.emit = emit_with_flush


def extract_channel_id(url_or_id: str) -> Optional[str]:
    """
    Extract channel ID from various YouTube URL formats.
    
    Args:
        url_or_id: Channel URL or channel ID
        
    Returns:
        Channel ID if found, None otherwise
    """
    # If it's already a channel ID (starts with UC)
    if url_or_id.startswith('UC') and len(url_or_id) == 24:
        return url_or_id
    
    # Try to extract from URL
    patterns = [
        r'youtube\.com/channel/([a-zA-Z0-9_-]+)',
        r'youtube\.com/c/([a-zA-Z0-9_-]+)',
        r'youtube\.com/@([a-zA-Z0-9_-]+)',
        r'youtube\.com/user/([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    
    return None


def get_channel_info(channel_url_or_id: str, cache_dir: Optional[Path] = None) -> Dict:
    """
    Get channel information using yt-dlp, with caching support.
    
    Args:
        channel_url_or_id: Channel URL or ID
        cache_dir: Optional directory to check for cached channel info
        
    Returns:
        Dictionary with channel information
    """
    # Normalize channel URL for cache key
    cache_key = channel_url_or_id
    if not cache_key.startswith('http'):
        if cache_key.startswith('UC'):
            cache_key = f"https://www.youtube.com/channel/{cache_key}"
        elif cache_key.startswith('@'):
            cache_key = f"https://www.youtube.com/{cache_key}"
        else:
            cache_key = f"https://www.youtube.com/@{cache_key}"
    
    # Try to load from cache if cache_dir is provided
    if cache_dir:
        # Try to find existing channel_info.json in any subdirectory
        for subdir in cache_dir.iterdir():
            if subdir.is_dir():
                channel_info_file = subdir / "channel_info.json"
                if channel_info_file.exists():
                    try:
                        with open(channel_info_file, 'r', encoding='utf-8') as f:
                            cached_info = json.load(f)
                            # Check if this matches the requested channel
                            if (cached_info.get('channel_url') == cache_key or 
                                cached_info.get('channel_id') in cache_key or
                                cache_key in cached_info.get('channel_url', '')):
                                logger.info(f"📦 Found cached channel info in: {subdir.name}")
                                logger.info(f"   Channel: {cached_info.get('channel_name')}")
                                logger.info(f"   Channel ID: {cached_info.get('channel_id')}")
                                return cached_info
                    except Exception as e:
                        logger.debug(f"Failed to read cached channel info from {channel_info_file}: {e}")
    
    # If not cached, fetch from YouTube
    logger.info(f"🔍 Fetching channel information from YouTube...")
    logger.info(f"   URL: {cache_key}")
    sys.stdout.flush()
    
    try:
        logger.info("   Initializing yt-dlp...")
        sys.stdout.flush()
        
        ydl_opts = {
            'quiet': True,  # Keep quiet for channel info extraction
            'no_warnings': True,
            'extract_flat': True,  # Use flat extraction to avoid format testing
            'ignoreerrors': True,  # Ignore errors on individual videos
        }
        
        logger.info("   Creating YoutubeDL instance...")
        sys.stdout.flush()
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("   Extracting channel info (this may take a moment)...")
            sys.stdout.flush()
            
            # Try to extract channel info
            # Use extract_flat to avoid processing individual videos
            info = ydl.extract_info(cache_key, download=False)
            
            logger.info("   ✅ Successfully extracted channel info")
            logger.info(f"   Processing channel data...")
            sys.stdout.flush()
            
            # Get channel name and ID
            channel_name = info.get('channel', 'Unknown Channel')
            channel_id = info.get('channel_id', '')
            
            logger.info(f"   Channel name: {channel_name}")
            logger.info(f"   Channel ID: {channel_id}")
            sys.stdout.flush()
            
            # Clean channel name for filesystem
            safe_channel_name = re.sub(r'[^\w\s-]', '', channel_name)
            safe_channel_name = re.sub(r'[-\s]+', '_', safe_channel_name)
            
            channel_info = {
                'channel_name': channel_name,
                'channel_id': channel_id,
                'safe_channel_name': safe_channel_name,
                'channel_url': info.get('channel_url', cache_key),
                'fetched_at': datetime.datetime.now().isoformat()
            }
            
            logger.info("   ✅ Channel info processed successfully")
            sys.stdout.flush()
            
            return channel_info
            
    except Exception as e:
        logger.error(f"❌ Failed to get channel info: {e}")
        import traceback
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        sys.stdout.flush()
        raise


def get_video_upload_date(video_id: str) -> Optional[str]:
    """
    Get upload date for a single video by fetching full metadata.
    This is used when extract_flat=True doesn't provide dates.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Upload date as YYYY-MM-DD string, or None if failed
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Try different date fields
            upload_date = (
                info.get('upload_date') or
                info.get('release_date') or
                None
            )
            
            if upload_date:
                # Format: YYYYMMDD -> YYYY-MM-DD
                if len(upload_date) == 8:
                    return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
                return upload_date
            
            # Try timestamp
            timestamp = info.get('timestamp')
            if timestamp:
                from datetime import datetime
                return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                
    except Exception as e:
        logger.debug(f"Failed to fetch date for {video_id}: {e}")
        return None
    
    return None


def get_all_channel_videos(channel_url_or_id: str) -> List[Dict]:
    """
    Get all videos from a channel using yt-dlp.
    
    Args:
        channel_url_or_id: Channel URL or ID
        
    Returns:
        List of video dictionaries with video_id, title, url, etc.
    """
    videos = []
    
    # Normalize channel URL
    if not channel_url_or_id.startswith('http'):
        # If it's a channel ID, construct URL
        if channel_url_or_id.startswith('UC'):
            channel_url_or_id = f"https://www.youtube.com/channel/{channel_url_or_id}"
        elif channel_url_or_id.startswith('@'):
            channel_url_or_id = f"https://www.youtube.com/{channel_url_or_id}"
        else:
            channel_url_or_id = f"https://www.youtube.com/@{channel_url_or_id}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,  # Only get metadata, don't download
    }
    
    # Try different URL formats
    url_variants = [
        f"{channel_url_or_id}/videos",
        channel_url_or_id,
    ]
    
    for url_variant in url_variants:
        try:
            logger.info(f"Trying to extract videos from: {url_variant}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_variant, download=False)
                
                # Check if it's a channel or playlist
                if 'entries' in info:
                    entries = info.get('entries', [])
                else:
                    # Single video or channel info
                    if 'id' in info and info.get('id'):
                        # It's a single video, wrap it
                        entries = [info]
                    else:
                        # Try to get uploads playlist
                        channel_id = info.get('channel_id', '')
                        if channel_id:
                            uploads_url = f"https://www.youtube.com/channel/{channel_id}/videos"
                            logger.info(f"Trying uploads playlist: {uploads_url}")
                            try:
                                playlist_info = ydl.extract_info(uploads_url, download=False)
                                entries = playlist_info.get('entries', [])
                            except Exception:
                                entries = []
                        else:
                            entries = []
                
                # Process entries
                for entry in entries:
                    if entry and entry.get('id'):
                        video_id = entry.get('id', '')
                        title = entry.get('title', 'Unknown Title')
                        
                        # Construct URL if not present
                        if 'url' in entry and entry['url']:
                            url = entry['url']
                        elif 'webpage_url' in entry:
                            url = entry['webpage_url']
                        else:
                            url = f"https://www.youtube.com/watch?v={video_id}"
                        
                        # Extract upload date if available (when extract_flat=False)
                        upload_date = None
                        if 'upload_date' in entry:
                            upload_date_str = entry['upload_date']
                            # Format: YYYYMMDD -> YYYY-MM-DD
                            if len(upload_date_str) == 8:
                                upload_date = f"{upload_date_str[:4]}-{upload_date_str[4:6]}-{upload_date_str[6:8]}"
                            else:
                                upload_date = upload_date_str
                        elif 'timestamp' in entry:
                            # Convert timestamp to date
                            from datetime import datetime
                            upload_date = datetime.fromtimestamp(entry['timestamp']).strftime('%Y-%m-%d')
                        
                        videos.append({
                            'video_id': video_id,
                            'title': title,
                            'url': url,
                            'duration': entry.get('duration', 0),
                            'view_count': entry.get('view_count', 0),
                            'upload_date': upload_date,  # May be None if extract_flat=True
                        })
                
                if videos:
                    logger.info(f"Successfully extracted {len(videos)} videos")
                    return videos
                else:
                    logger.warning(f"No videos found with URL variant: {url_variant}")
                    
        except Exception as e:
            logger.warning(f"Failed with URL variant {url_variant}: {e}")
            continue
    
    # If all variants failed, raise error
    raise Exception(f"Failed to extract videos from channel: {channel_url_or_id}")


def save_transcript_to_file(video_data: Dict, transcript_data: Dict, channel_dir: Path) -> Path:
    """
    Save transcript to a JSON file.
    
    Args:
        video_data: Video metadata
        transcript_data: Transcript data with 'transcript', 'language', 'title'
        channel_dir: Directory to save transcripts
        
    Returns:
        Path to the saved file
    """
    video_id = video_data['video_id']
    safe_title = re.sub(r'[^\w\s-]', '', video_data['title'])
    safe_title = re.sub(r'[-\s]+', '_', safe_title)
    safe_title = safe_title[:100]  # Limit length
    
    filename = f"{video_id}_{safe_title}.json"
    file_path = channel_dir / filename
    
    # Prepare data to save
    data = {
        'video_id': video_id,
        'title': video_data['title'],
        'url': video_data['url'],
        'duration': video_data.get('duration', 0),
        'view_count': video_data.get('view_count', 0),
        'transcript': transcript_data.get('transcript', ''),
        'language': transcript_data.get('language', ''),
        'transcript_title': transcript_data.get('title', video_data['title']),
    }
    
    # Add upload_date if available
    if 'upload_date' in video_data and video_data['upload_date']:
        data['upload_date'] = video_data['upload_date']
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return file_path


def download_channel_transcripts(channel_url_or_id: str, skip_existing: bool = True, refresh_video_list: bool = False, retranscribe_whisper: bool = False, cache_days: int = CACHE_DURATION_DAYS, force_retranscribe: bool = False) -> Dict:
    """
    Download transcripts for all videos in a channel.
    
    Args:
        channel_url_or_id: Channel URL or ID
        skip_existing: If True, skip videos that already have transcripts
        refresh_video_list: If True, force refresh of video list (ignore cache)
        
    Returns:
        Dictionary with statistics
        
    Note:
        Video list is cached for 8 hours. Use refresh_video_list=True to force refresh.
    """
    stats = {
        'total_videos': 0,
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    try:
        # Get channel info
        logger.info("=" * 80)
        logger.info("Starting channel transcript download")
        logger.info("=" * 80)
        logger.info(f"Channel URL/ID: {channel_url_or_id}")
        logger.info("")
        sys.stdout.flush()
        
        # Try to get channel info (with caching)
        channel_info = get_channel_info(channel_url_or_id, cache_dir=OUTPUT_DIR)
        
        logger.info("")
        logger.info("✅ Channel Information Retrieved:")
        logger.info(f"   Channel Name: {channel_info['channel_name']}")
        logger.info(f"   Channel ID: {channel_info['channel_id']}")
        logger.info(f"   Channel URL: {channel_info['channel_url']}")
        sys.stdout.flush()
        
        # Create output directory for this channel
        channel_dir = OUTPUT_DIR / channel_info['safe_channel_name']
        logger.info("")
        logger.info(f"📁 Creating output directory: {channel_dir}")
        sys.stdout.flush()
        channel_dir.mkdir(exist_ok=True)
        logger.info(f"   ✅ Directory ready")
        sys.stdout.flush()
        
        # Set up file logging for this channel
        log_file = channel_dir / "download.log"
        logger.info(f"📝 Setting up file logging: {log_file}")
        sys.stdout.flush()
        setup_file_logging(log_file)
        logger.info(f"   ✅ File logging active (console + file)")
        sys.stdout.flush()
        
        # Save channel info (cache it for future use)
        channel_info_file = channel_dir / "channel_info.json"
        logger.info(f"💾 Saving channel info to: {channel_info_file}")
        sys.stdout.flush()
        with open(channel_info_file, 'w', encoding='utf-8') as f:
            json.dump(channel_info, f, indent=2, ensure_ascii=False)
        logger.info(f"   ✅ Channel info cached")
        sys.stdout.flush()
        
        # Get all videos (check cache first)
        logger.info("-" * 80)
        logger.info("Step 1: Fetching all videos from channel...")
        logger.info("-" * 80)
        sys.stdout.flush()
        
        videos_list_file = channel_dir / "videos_list.json"
        videos = None
        cache_age_days = None
        
        # Check if cached video list exists (unless forced refresh)
        if videos_list_file.exists() and not refresh_video_list:
            try:
                logger.info(f"📦 Checking for cached video list: {videos_list_file}")
                sys.stdout.flush()
                
                with open(videos_list_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Check if cache is for the same channel
                if cached_data.get('channel_id') == channel_info['channel_id']:
                    extracted_at_str = cached_data.get('extracted_at', '')
                    if extracted_at_str:
                        extracted_at = datetime.datetime.fromisoformat(extracted_at_str)
                        cache_age = datetime.datetime.now() - extracted_at
                        cache_age_hours = cache_age.total_seconds() / 3600
                        cache_age_days = cache_age.days
                        
                        # Use cache if it's less than 8 hours old
                        cache_duration_hours = 8
                        if cache_age_hours < cache_duration_hours:
                            videos = cached_data.get('videos', [])
                            remaining_hours = cache_duration_hours - cache_age_hours
                            if cache_age_hours < 1:
                                age_str = f"{cache_age_hours * 60:.1f} minutes"
                            else:
                                age_str = f"{cache_age_hours:.1f} hours"
                            logger.info(f"✅ Using cached video list ({len(videos)} videos, {age_str} old)")
                            logger.info(f"   Cache expires in {remaining_hours:.1f} hours")
                            sys.stdout.flush()
                        else:
                            if cache_age_hours < 24:
                                age_str = f"{cache_age_hours:.1f} hours"
                            else:
                                age_str = f"{cache_age_days} days"
                            logger.info(f"⚠️  Cached video list is {age_str} old (expired, will refresh)")
                            sys.stdout.flush()
                    else:
                        # Old format without timestamp, use it but will refresh
                        videos = cached_data.get('videos', [])
                        if videos:
                            logger.info(f"✅ Using cached video list ({len(videos)} videos, no timestamp)")
                            sys.stdout.flush()
                else:
                    logger.info(f"⚠️  Cached video list is for different channel, will refresh")
                    sys.stdout.flush()
            except Exception as e:
                logger.warning(f"⚠️  Failed to read cached video list: {e}, will fetch fresh")
                sys.stdout.flush()
        
        # Fetch fresh if no cache or cache expired
        if videos is None:
            logger.info("🔍 Fetching fresh video list from YouTube...")
            sys.stdout.flush()
            videos = get_all_channel_videos(channel_url_or_id)
            
            # Save video list to file
            logger.info(f"💾 Saving video list to: {videos_list_file}")
            sys.stdout.flush()
            with open(videos_list_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'channel_name': channel_info['channel_name'],
                    'channel_id': channel_info['channel_id'],
                    'total_videos': len(videos),
                    'extracted_at': datetime.datetime.now().isoformat(),
                    'videos': videos
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"   ✅ Video list saved ({len(videos)} videos)")
            sys.stdout.flush()
        
        stats['total_videos'] = len(videos)
        logger.info(f"✅ Total videos: {len(videos)}")
        sys.stdout.flush()
        
        logger.info("-" * 80)
        logger.info("Step 2: Starting transcript download...")
        logger.info("-" * 80)
        sys.stdout.flush()
        
        # Process each video
        start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        for idx, video in enumerate(videos, 1):
            video_id = video['video_id']
            title = video['title']
            video_start_time = time.time()
            
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"Video {idx}/{len(videos)}: {title}")
            logger.info(f"Video ID: {video_id}")
            logger.info(f"URL: {video['url']}")
            logger.info(f"Duration: {video.get('duration', 0)}s | Views: {video.get('view_count', 0):,}")
            sys.stdout.flush()
            
            # Check if transcript already exists
            if skip_existing and not force_retranscribe:
                existing_files = list(channel_dir.glob(f"{video_id}_*.json"))
                if existing_files:
                    logger.info(f"⏭️  Transcript already exists: {existing_files[0].name}")
                    logger.info(f"⏭️  Skipping (use --no-skip-existing or --force-retranscribe to re-download)")
                    stats['skipped'] += 1
                    consecutive_failures = 0  # Reset counter on skip (not a failure)
                    elapsed = time.time() - video_start_time
                    logger.info(f"⏱️  Time: {elapsed:.2f}s | Status: SKIPPED")
                    sys.stdout.flush()
                    continue
            
            # Check cache first - but check if it was transcribed with Whisper (needs retranscription)
            # Skip cache if force_retranscribe is True
            cache_file = CACHE_DIR / f"{video_id}.json"
            needs_retranscription = force_retranscribe
            if cache_file.exists() and not force_retranscribe:
                logger.info(f"📁 Checking cache: {cache_file}")
                sys.stdout.flush()
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    if cached_data.get('transcript'):
                        # Check if this was transcribed with Whisper (we want to retranscribe with AWS)
                        # We'll check if video/audio files exist - if they do, it means Whisper was used
                        video_path = CACHE_DIR / f"{video_id}.mp4"
                        mp3_path = CACHE_DIR / f"{video_id}.mp3"
                        
                        # If video or audio exists, it was likely transcribed with Whisper
                        # Only retranscribe if flag is set or if retranscribe_whisper is True
                        if (video_path.exists() or mp3_path.exists()) and retranscribe_whisper:
                            logger.info(f"⚠️  Found Whisper transcript, will retranscribe with AWS Transcribe")
                            needs_retranscription = True
                        else:
                            # Check if we can determine from metadata (if available)
                            # For now, assume if files don't exist and transcript exists, it's from AWS or YouTube API
                            logger.info(f"✅ Found cached transcript ({len(cached_data.get('transcript', ''))} chars)")
                            # Save to channel directory
                            file_path = save_transcript_to_file(video, cached_data, channel_dir)
                            logger.info(f"✅ Saved to: {file_path.name}")
                            stats['downloaded'] += 1
                            consecutive_failures = 0  # Reset counter on success
                            elapsed = time.time() - video_start_time
                            logger.info(f"⏱️  Time: {elapsed:.2f}s | Status: SUCCESS (from cache)")
                            sys.stdout.flush()
                            continue
                    else:
                        logger.warning(f"⚠️  Cache file exists but has no transcript, will download fresh")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to read cache: {e}, will download fresh")
                    sys.stdout.flush()
            
            # Fetch upload date if not already in video data
            if 'upload_date' not in video or not video.get('upload_date'):
                logger.info("📅 Fetching upload date...")
                sys.stdout.flush()
                upload_date = get_video_upload_date(video_id)
                if upload_date:
                    video['upload_date'] = upload_date
                    logger.info(f"   ✅ Upload date: {upload_date}")
                else:
                    logger.warning(f"   ⚠️  Could not fetch upload date")
                sys.stdout.flush()
            
            # Download transcript (with automatic Whisper fallback)
            try:
                logger.info(f"📥 Downloading transcript (this may take a while)...")
                sys.stdout.flush()
                
                # First try to get transcript via YouTube API
                transcript_data = None
                transcript_method = None
                try:
                    from youtubeSummarize import try_youtube_transcript, download_video_if_needed, transcribe_with_whisper, detect_and_confirm_language, cache_transcript
                    
                    result = try_youtube_transcript(video_id)
                    if result and result.get('transcript'):
                        transcript_data = result
                        transcript_method = "YouTube API"
                        logger.info(f"✅ Got transcript from YouTube API")
                except Exception as e:
                    logger.info(f"⚠️  YouTube transcript API failed: {e}")
                    logger.info(f"🔄 Will download video and transcribe with AWS Transcribe...")
                
                # If no transcript from API, download video and use AWS Transcribe
                if not transcript_data or not transcript_data.get('transcript') or needs_retranscription:
                    if needs_retranscription:
                        logger.info(f"🔄 Retranscribing with AWS Transcribe (replacing Whisper transcript)...")
                    else:
                        logger.info(f"📥 No transcript available via API, downloading video for AWS Transcribe...")
                    logger.info(f"   This will download the video and transcribe it with AWS Transcribe (may take several minutes)")
                    sys.stdout.flush()
                    
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    temp_dir = CACHE_DIR
                    temp_dir.mkdir(exist_ok=True)
                    video_path = temp_dir / f"{video_id}.mp4"
                    
                    # Clean up expired cache files before checking
                    cleanup_expired_cache_files(video_id, cache_days)
                    
                    # Auto-download video (no user prompt)
                    # Normal download logic - uses cache if available
                    if not video_path.exists() or is_file_expired(video_path, cache_days):
                        logger.info(f"📥 Downloading video: {video_id}")
                        sys.stdout.flush()
                        cookies_path = os.path.join(os.getcwd(), 'cookies.txt')
                        downloaded = False
                        last_error = None
                        
                        # First, try to list available formats to see what's actually available
                        logger.info(f"🔍 Checking available formats for video...")
                        sys.stdout.flush()
                        try:
                            ydl_opts_list = {
                                'quiet': True,
                                'listformats': True,
                            }
                            if os.path.exists(cookies_path):
                                ydl_opts_list['cookiefile'] = cookies_path
                            
                            with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
                                info = ydl.extract_info(video_url, download=False)
                                formats = info.get('formats', [])
                                if formats:
                                    logger.info(f"   Found {len(formats)} available formats")
                                    # Try to find a suitable format
                                    for fmt in formats:
                                        if fmt.get('vcodec') != 'none' and fmt.get('ext') in ['mp4', 'webm', 'mkv']:
                                            format_id = fmt.get('format_id')
                                            height = fmt.get('height', 0)
                                            logger.info(f"   Trying format {format_id} ({height}p, {fmt.get('ext')})")
                                            break
                        except Exception as e:
                            logger.debug(f"   Could not list formats: {e}")
                        
                        # Try format selectors in order of preference (lowest quality first for speed)
                        # First, check if we have a cached successful format
                        last_successful_format = get_last_successful_format()
                        default_format_list = [
                            None,  # No format specified - let yt-dlp choose automatically
                            'worst',  # Any format, lowest quality
                            'worstvideo+worstaudio/worst',  # Separate streams, lowest quality
                            'best[height<=360]',  # 360p or lower
                            'best[height<=480]',  # 480p or lower
                            'best[height<=720]',  # 720p or lower
                            'best',  # Best available
                        ]
                        
                        # If we have a cached successful format, try it first
                        if last_successful_format:
                            format_list = [last_successful_format] + [f for f in default_format_list if f != last_successful_format]
                            logger.info(f"🎯 Trying cached successful format first: {last_successful_format}")
                        else:
                            format_list = default_format_list
                        
                        for fmt in format_list:
                            try:
                                ydl_opts = {
                                    'outtmpl': str(video_path),
                                    'merge_output_format': 'mp4',
                                    'quiet': True,
                                    'no_warnings': False,
                                    # Add headers to avoid 403 errors
                                    'http_headers': {
                                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                        'Accept-Language': 'en-us,en;q=0.5',
                                        'Sec-Fetch-Mode': 'navigate',
                                    },
                                    # YouTube extractor options - use android client which works better
                                    'extractor_args': {
                                        'youtube': {
                                            'player_client': ['android', 'web'],  # Try android first, then web
                                        }
                                    },
                                }
                                if fmt is not None:
                                    ydl_opts['format'] = fmt
                                # If fmt is None, don't set format at all - let yt-dlp auto-select
                                
                                if os.path.exists(cookies_path):
                                    ydl_opts['cookiefile'] = cookies_path
                                
                                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                    info = ydl.extract_info(video_url, download=True)
                                    title = info.get('title', title)
                                
                                format_used = fmt if fmt else "auto-selected"
                                logger.info(f"✅ Downloaded video (format: {format_used})")
                                # Remember this successful format for next time (only save actual format strings, not None)
                                if fmt is not None:
                                    save_successful_format(fmt)
                                downloaded = True
                                break
                            except Exception as e:
                                error_msg = str(e)
                                last_error = error_msg
                                # Check if it's a format-specific error or a more general error
                                if 'format' not in error_msg.lower() and 'not available' not in error_msg.lower():
                                    # This might be a different error (restricted, private, etc.)
                                    logger.warning(f"   Download failed: {error_msg[:100]}")
                                    break
                                continue
                        
                        if not downloaded:
                            logger.error(f"❌ All download attempts failed")
                            logger.error(f"   Last error: {last_error}")
                            logger.error(f"   Video may be restricted, private, region-locked, or unavailable")
                            raise Exception(f"Failed to download video: {last_error}")
                    else:
                        if is_file_expired(video_path, cache_days):
                            logger.info(f"⚠️  Cached video file expired, will re-download")
                            video_path.unlink()
                            # Re-download will happen in next iteration
                            raise Exception("Cached video expired, re-downloading")
                        logger.info(f"📁 Using cached video file (age: {get_file_age_str(video_path)})")
                        try:
                            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                                info = ydl.extract_info(video_url, download=False)
                                title = info.get('title', title)
                        except Exception:
                            pass
                    
                    # Extract audio
                    logger.info(f"🎵 Extracting audio from video...")
                    sys.stdout.flush()
                    mp3_path = temp_dir / f"{video_id}.mp3"
                    if not mp3_path.exists() or is_file_expired(mp3_path, cache_days):
                        cmd = [
                            "ffmpeg", "-y", "-i", str(video_path),
                            "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "192k", str(mp3_path)
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        logger.info(f"✅ Audio extracted")
                    else:
                        if is_file_expired(mp3_path, cache_days):
                            logger.info(f"⚠️  Cached audio file expired, will re-extract")
                            mp3_path.unlink()
                            # Re-extract will happen below
                        else:
                            logger.info(f"📁 Using cached audio file (age: {get_file_age_str(mp3_path)})")
                    
                    # Detect language for AWS Transcribe
                    from youtubeSummarize import detect_and_confirm_language
                    detected_language = detect_and_confirm_language(video_id, title)
                    # Normalize language code to AWS Transcribe compatible format
                    language = normalize_language_code_for_aws(detected_language)
                    if language != detected_language:
                        logger.info(f"🌐 Normalized language code: {detected_language} -> {language}")
                    
                    # Transcribe with AWS Transcribe
                    transcript, transcribe_language = transcribe_with_aws_transcribe(mp3_path, video_id, language)
                    
                    # Cache the transcript
                    from youtubeSummarize import cache_transcript
                    cache_transcript(video_id, transcript, transcribe_language, title)
                    
                    transcript_data = {
                        'transcript': transcript,
                        'language': transcribe_language,
                        'title': title
                    }
                    transcript_method = "AWS Transcribe"
                    
                    logger.info(f"✅ AWS Transcribe transcription complete")
                    
                    # Keep video and audio files (don't delete them)
                    logger.info(f"📁 Keeping video and audio files for future use")
                
                # Save transcript
                if transcript_data and transcript_data.get('transcript'):
                    transcript_length = len(transcript_data.get('transcript', ''))
                    language = transcript_data.get('language', 'unknown')
                    method = transcript_method or "Unknown"
                    logger.info(f"✅ Transcript ready: {transcript_length} chars, language: {language} (via {method})")
                    
                    # Save transcript
                    file_path = save_transcript_to_file(video, transcript_data, channel_dir)
                    logger.info(f"✅ Saved to: {file_path.name}")
                    stats['downloaded'] += 1
                    elapsed = time.time() - video_start_time
                    logger.info(f"⏱️  Time: {elapsed:.2f}s | Status: SUCCESS")
                    
                    # Progress summary
                    progress_pct = (idx / len(videos)) * 100
                    logger.info(f"📊 Progress: {idx}/{len(videos)} ({progress_pct:.1f}%) | "
                              f"Downloaded: {stats['downloaded']} | "
                              f"Skipped: {stats['skipped']} | "
                              f"Failed: {stats['failed']}")
                    sys.stdout.flush()
                else:
                    logger.warning(f"⚠️  No transcript available after all methods")
                    stats['failed'] += 1
                    consecutive_failures += 1
                    stats['errors'].append({
                        'video_id': video_id,
                        'title': title,
                        'error': 'No transcript available after YouTube API and Whisper'
                    })
                    elapsed = time.time() - video_start_time
                    logger.info(f"⏱️  Time: {elapsed:.2f}s | Status: FAILED")
                    sys.stdout.flush()
            except Exception as e:
                logger.error(f"❌ Failed to download transcript: {e}")
                import traceback
                logger.debug(f"Error traceback:\n{traceback.format_exc()}")
                stats['failed'] += 1
                consecutive_failures += 1
                stats['errors'].append({
                    'video_id': video_id,
                    'title': title,
                    'error': str(e)
                })
                elapsed = time.time() - video_start_time
                logger.info(f"⏱️  Time: {elapsed:.2f}s | Status: FAILED")
                sys.stdout.flush()
            
            # Check for consecutive failures and exit if threshold reached
            if consecutive_failures >= max_consecutive_failures:
                logger.error("")
                logger.error("=" * 80)
                logger.error(f"❌ STOPPING: {consecutive_failures} consecutive failures detected")
                logger.error("=" * 80)
                logger.error(f"This usually indicates:")
                logger.error("  - Videos are restricted/unavailable")
                logger.error("  - Network/API issues")
                logger.error("  - Format compatibility problems")
                logger.error("")
                logger.error(f"Processed {idx}/{len(videos)} videos before stopping")
                logger.error(f"Successfully downloaded: {stats['downloaded']}")
                logger.error(f"Failed: {stats['failed']}")
                logger.error(f"Skipped: {stats['skipped']}")
                logger.error("")
                logger.error("You can:")
                logger.error("  1. Check the error messages above")
                logger.error("  2. Fix the underlying issue")
                logger.error("  3. Resume by running the script again (it will skip already processed videos)")
                logger.error("=" * 80)
                sys.stdout.flush()
                break
            
            # Reset consecutive failures counter on success
            if transcript_data and transcript_data.get('transcript'):
                consecutive_failures = 0
        
        # Save statistics
        total_elapsed = time.time() - start_time
        stats['total_time_seconds'] = total_elapsed
        stats['completed_at'] = datetime.datetime.now().isoformat()
        
        stats_file = channel_dir / "download_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Channel: {channel_info['channel_name']}")
        logger.info(f"Total videos found: {stats['total_videos']}")
        logger.info(f"✅ Downloaded: {stats['downloaded']}")
        logger.info(f"⏭️  Skipped: {stats['skipped']}")
        logger.info(f"❌ Failed: {stats['failed']}")
        logger.info(f"⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
        if stats['downloaded'] > 0:
            avg_time = total_elapsed / stats['downloaded']
            logger.info(f"⏱️  Average time per video: {avg_time:.1f}s")
        logger.info(f"📁 Output directory: {channel_dir}")
        logger.info(f"📝 Log file: {log_file}")
        logger.info(f"📊 Stats file: {stats_file}")
        
        if stats['failed'] > 0:
            logger.info("")
            logger.info(f"⚠️  {stats['failed']} videos failed. Check download_stats.json for details.")
        
        logger.info("=" * 80)
        sys.stdout.flush()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error downloading channel transcripts: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Download YouTube transcripts for all videos in a channel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_channel_transcripts.py https://www.youtube.com/@channelname
  python download_channel_transcripts.py UCxxxxxxxxxxxxxxxxxxxxx
  python download_channel_transcripts.py https://www.youtube.com/channel/UCxxxxxxxxxxxxxxxxxxxxx
  python download_channel_transcripts.py --no-skip-existing https://www.youtube.com/@channelname
        """
    )
    
    parser.add_argument(
        'channel',
        help='YouTube channel URL or channel ID'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-download transcripts even if they already exist'
    )
    
    parser.add_argument(
        '--refresh-video-list',
        action='store_true',
        help='Force refresh of video list (ignore cache)'
    )
    parser.add_argument(
        '--retranscribe-whisper',
        action='store_true',
        help='Retranscribe all videos that were previously transcribed with Whisper using AWS Transcribe'
    )
    parser.add_argument(
        '--force-retranscribe',
        action='store_true',
        help='Force retranscription of all videos (even if transcripts exist), but skip video download if video file already exists'
    )
    parser.add_argument(
        '--cache-days',
        type=int,
        default=CACHE_DURATION_DAYS,
        help=f'Number of days to keep video/audio cache files (default: {CACHE_DURATION_DAYS} days)'
    )
    
    args = parser.parse_args()
    
    # Download transcripts
    try:
        stats = download_channel_transcripts(
            args.channel,
            skip_existing=not args.no_skip_existing and not args.force_retranscribe,
            refresh_video_list=args.refresh_video_list,
            retranscribe_whisper=args.retranscribe_whisper,
            cache_days=args.cache_days,
            force_retranscribe=args.force_retranscribe
        )
        
        if stats['failed'] > 0:
            logger.warning(f"\n⚠️  {stats['failed']} videos failed to download transcripts")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

