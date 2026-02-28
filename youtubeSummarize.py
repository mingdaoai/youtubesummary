#!/usr/bin/env uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=0.8.0",
#     "yt-dlp>=2023.1.6",
#     "requests>=2.25.0",
#     "boto3>=1.26.0",
#     "openai-whisper>=20231117",
# ]
# ///
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
import re
import time
import json
import traceback
import yt_dlp
from urllib.parse import urlparse
import logging
import requests
import subprocess
import datetime
try:
    from ads.youtube_analyze.youtubeMetaUtil import download_transcript as meta_download_transcript
    META_DOWNLOAD_AVAILABLE = True
except ImportError:
    META_DOWNLOAD_AVAILABLE = False
    meta_download_transcript = None  # type: ignore
import whisper
from youtubeTranscript import download_youtube_transcript, download_youtube_transcript_alternative, get_available_languages

from logging_utils import setup_logger, flush_logger

logger = setup_logger(__name__)

# Multi-Tor support imports
try:
    import sys
    sys.path.append('/Users/haha/github/multi-tor')
    from multi_tor_downloader import MultiTorTranscriptDownloader  # type: ignore
    MULTI_TOR_AVAILABLE = True
except ImportError:
    MULTI_TOR_AVAILABLE = False
    print("Multi-Tor not available. Install PySocks and ensure multi-tor setup is complete.")

# Legacy Tor support imports (fallback)
try:
    import socks
    TOR_AVAILABLE = True
except ImportError:
    TOR_AVAILABLE = False
    if not MULTI_TOR_AVAILABLE:
        print("PySocks not installed. Install with: pip install PySocks")

# ========== PROXY CONFIGURATION ==========
def get_proxy_config():
    """
    Get proxy configuration from environment variables or config file.
    Priority: Environment variables > Config file > No proxy
    
    Environment variables:
    - HTTP_PROXY / http_proxy: HTTP proxy URL
    - HTTPS_PROXY / https_proxy: HTTPS proxy URL
    - SOCKS_PROXY / socks_proxy: SOCKS proxy URL (e.g., socks5://127.0.0.1:9050)
    - USE_TOR: Set to 'true' to use Tor proxy at 127.0.0.1:9050
    
    Returns:
        dict with 'http', 'https', 'socks' keys or None if no proxy configured
    """
    proxy_config = {}
    
    # Check for SOCKS proxy (highest priority for Tor)
    socks_proxy = os.environ.get('SOCKS_PROXY') or os.environ.get('socks_proxy')
    use_tor = os.environ.get('USE_TOR', '').lower() == 'true'
    
    if socks_proxy:
        proxy_config['socks'] = socks_proxy
    elif use_tor:
        proxy_config['socks'] = 'socks5://127.0.0.1:9050'
    
    # Check for HTTP/HTTPS proxies
    http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
    https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
    
    if http_proxy:
        proxy_config['http'] = http_proxy
    if https_proxy:
        proxy_config['https'] = https_proxy
    
    # Try to load from config file if no env vars set
    if not proxy_config:
        config_path = Path(__file__).parent / ".proxy_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    proxy_config.update(file_config)
            except Exception as e:
                logger.warning(f"Failed to load proxy config from file: {e}")
    
    return proxy_config if proxy_config else None

def get_ytdlp_proxy_opts(proxy_config):
    """
    Convert proxy config to yt-dlp options.
    
    Args:
        proxy_config: Dict with proxy settings
        
    Returns:
        Dict with yt-dlp proxy options
    """
    if not proxy_config:
        return {}
    
    opts = {}
    
    # yt-dlp supports SOCKS proxies directly
    if 'socks' in proxy_config:
        opts['proxy'] = proxy_config['socks']
    elif 'https' in proxy_config:
        opts['proxy'] = proxy_config['https']
    elif 'http' in proxy_config:
        opts['proxy'] = proxy_config['http']
    
    return opts

def get_requests_proxies(proxy_config):
    """
    Convert proxy config to requests proxies format.
    
    Args:
        proxy_config: Dict with proxy settings
        
    Returns:
        Dict suitable for requests library
    """
    if not proxy_config:
        return None
    
    proxies = {}
    
    if 'http' in proxy_config:
        proxies['http'] = proxy_config['http']
    if 'https' in proxy_config:
        proxies['https'] = proxy_config['https']
    
    # For SOCKS, requests needs requests[socks] installed
    if 'socks' in proxy_config:
        # SOCKS proxy works for both http and https
        proxies['http'] = proxy_config['socks']
        proxies['https'] = proxy_config['socks']
    
    return proxies if proxies else None

# Global proxy configuration
PROXY_CONFIG = get_proxy_config()
REQUESTS_PROXIES = get_requests_proxies(PROXY_CONFIG)
YTDL_PROXY_OPTS = get_ytdlp_proxy_opts(PROXY_CONFIG)

def save_proxy_config(proxy_config):
    """Save proxy configuration to file."""
    config_path = Path(__file__).parent / ".proxy_config.json"
    try:
        with open(config_path, 'w') as f:
            json.dump(proxy_config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save proxy config: {e}")
        return False

def configure_proxies_interactive():
    """Interactive proxy configuration."""
    print("\n=== Proxy Configuration ===")
    print("Configure proxies for YouTube downloads.")
    print("Leave blank to skip a proxy type.\n")
    
    config = {}
    
    # SOCKS proxy (for Tor)
    print("SOCKS proxy (for Tor, e.g., socks5://127.0.0.1:9050):")
    socks = input("  SOCKS proxy URL: ").strip()
    if socks:
        config['socks'] = socks
    
    # HTTP proxy
    print("\nHTTP proxy (e.g., http://proxy.example.com:8080):")
    http = input("  HTTP proxy URL: ").strip()
    if http:
        config['http'] = http
    
    # HTTPS proxy
    print("\nHTTPS proxy (e.g., https://proxy.example.com:8080):")
    https = input("  HTTPS proxy URL: ").strip()
    if https:
        config['https'] = https
    
    if config:
        if save_proxy_config(config):
            print("✅ Proxy configuration saved!")
            print(f"   Config file: {Path(__file__).parent / '.proxy_config.json'}")
            return config
        else:
            print("❌ Failed to save proxy configuration")
            return None
    else:
        print("No proxies configured.")
        return None

# ========== CONFIGURATION ==========
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
GEMINI_MODEL_NAME = "gemini-2.5-flash"  # Gemini model for text generation
HISTORY_JSON = Path(__file__).parent / ".youtubesummary_history.json"

# ========== UTILS ==========
def create_gemini_client():
    """Create Gemini client using API key from file."""
    try:
        api_key_path = os.path.expanduser("~/.mingdaoai/gemini.key")
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
        
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error creating Gemini client: {e}")
        print("Make sure Gemini API key is configured at ~/.mingdaoai/gemini.key")
        sys.exit(1)

def extract_video_id(url: str) -> str:
    if "shorts/" in url:
        return url.split("shorts/")[-1].split("?")[0]
    elif "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    else:
        raise Exception("Invalid YouTube URL format")

def get_cached_transcript(video_id: str) -> dict | None:
    cache_file = CACHE_DIR / f"{video_id}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def cache_transcript(video_id: str, transcript: str, language: str, title: str = None):
    cache_file = CACHE_DIR / f"{video_id}.json"
    data = {"transcript": transcript, "language": language}
    if title:
        data["title"] = title
    else:
        # If title already exists in cache, preserve it
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    old = json.load(f)
                    if "title" in old:
                        data["title"] = old["title"]
            except Exception:
                pass
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def try_youtube_transcript(video_id):
    """
    Try to extract transcript using multi-Tor or configured proxies.
    Returns a dict with 'transcript' and 'language' or None if not available.
    """
    try:
        # First try multi-Tor approach if available
        if MULTI_TOR_AVAILABLE:
            logger.info("Using multi-Tor for transcript download")
            logger.info(f"🎯 Target video: https://www.youtube.com/watch?v={video_id}")
            try:
                downloader = MultiTorTranscriptDownloader(logger)
                transcript_segments = downloader.get_video_transcript_with_multi_tor(video_id)
                
                if transcript_segments:
                    # Convert segments to text format
                    transcript_text = " ".join([segment['text'] for segment in transcript_segments])
                    language = 'en'  # Default to English
                    cache_transcript(video_id, transcript_text, language)
                    logger.info(f"✅ Successfully downloaded transcript using multi-Tor: {len(transcript_segments)} segments")
                    logger.info(f"📊 Multi-Tor stats: {downloader.get_proxy_stats()}")
                    return {"transcript": transcript_text, "language": language}
                else:
                    logger.error("❌ Multi-Tor failed to get transcript - trying fallback methods")
                    logger.info(f"📊 Multi-Tor final stats: {downloader.get_proxy_stats()}")
            except Exception as e:
                logger.error(f"❌ Multi-Tor failed with exception: {e} - trying fallback methods")
                import traceback
                logger.debug(f"Multi-Tor exception traceback: {traceback.format_exc()}")
        
        # Fallback: Try with configured proxies
        if PROXY_CONFIG:
            logger.info(f"Trying transcript download with proxy configuration: {PROXY_CONFIG}")
            try:
                # Use youtubeTranscript module with proxy support
                transcript_result = download_youtube_transcript(
                    video_id, 
                    proxy=PROXY_CONFIG.get('socks') or PROXY_CONFIG.get('https') or PROXY_CONFIG.get('http')
                )
                if transcript_result:
                    cache_transcript(video_id, transcript_result['transcript'], transcript_result['language'])
                    logger.info(f"✅ Successfully downloaded transcript using proxy")
                    return transcript_result
            except Exception as e:
                logger.warning(f"Proxy transcript download failed: {e}")
        
        # Last resort: Try direct connection
        logger.info("Trying direct connection for transcript download")
        try:
            transcript_result = download_youtube_transcript(video_id)
            if transcript_result:
                cache_transcript(video_id, transcript_result['transcript'], transcript_result['language'])
                logger.info(f"✅ Successfully downloaded transcript via direct connection")
                return transcript_result
        except Exception as e:
            logger.error(f"Direct transcript download failed: {e}")
        
        logger.error("❌ All transcript download methods failed")
        return None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

def transcribe_with_whisper(mp3_path):
    """Transcribe the given mp3 file using Whisper and return transcript and language."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(str(mp3_path))
        transcript = result["text"].strip()
        language = result.get("language", "en-US")
        return transcript, language
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

def download_video_if_needed(video_id, video_url, video_path):
    """Download video with proxy support."""
    # Build base yt-dlp options with proxy support
    base_opts = {"quiet": True}
    base_opts.update(YTDL_PROXY_OPTS)
    
    if video_path.exists():
        logger.info(f"Video file {video_path} already exists. Using cached mp4 for transcription.")
        try:
            with yt_dlp.YoutubeDL(base_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', '')
        except Exception as e:
            logger.error(f"yt-dlp failed to extract info for existing file: {e}", exc_info=True)
            title = ''
    else:
        response = input("Would you like to download and transcribe the video? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        format_list = [
            '18',  # mp4 360p (lowest resolution first)
            '22',  # mp4 720p
            'best[ext=mp4]/best',  # best MP4
            'best',  # best overall
            'bestvideo+bestaudio/best',  # highest quality (fallback)
        ]
        cookies_path = os.path.join(os.getcwd(), 'cookies.txt')
        last_exception = None
        for fmt in format_list:
            ydl_opts = {
                'outtmpl': str(video_path),
                'format': fmt,
                'merge_output_format': 'mp4',
            }
            # Add proxy configuration
            ydl_opts.update(YTDL_PROXY_OPTS)
            
            if os.path.exists(cookies_path):
                ydl_opts['cookiefile'] = cookies_path
                logger.info(f"Using cookies from {cookies_path}")
            
            if PROXY_CONFIG:
                logger.info(f"Using proxy for download: {YTDL_PROXY_OPTS.get('proxy', 'none')}")
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    title = info.get('title', '')
                logger.info(f"Downloaded video with format: {fmt}")
                break  # Success!
            except Exception as e:
                logger.error(f"yt-dlp failed with format '{fmt}': {e}", exc_info=True)
                last_exception = e
                if '403' in str(e):
                    logger.warning("\nHTTP 403 Forbidden error detected. This may be due to age restriction, region lock, or YouTube signature changes.\n")
                    logger.warning("Try updating yt-dlp: pip install -U yt-dlp")
                    logger.warning("If the video requires login, export your YouTube cookies as cookies.txt and place it in the current directory.")
                    if PROXY_CONFIG:
                        logger.warning(f"Current proxy configuration: {PROXY_CONFIG}")
                        logger.warning("Try disabling proxy or using a different proxy.")
                continue
        else:
            logger.error("All format attempts failed.")
            traceback.print_exc()
            raise last_exception
    return title

def detect_and_confirm_language(video_id, title):
    cache_file = CACHE_DIR / f"{video_id}.json"
    cached_language = None
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                cached_language = cached_data.get("language")
        except Exception:
            cached_language = None
    if re.search(r'[\u4e00-\u9fff]', title):
        detected_language = 'zh-CN'
    else:
        detected_language = 'en-US'
    if cached_language:
        language = cached_language
        logger.info(f"Using cached language: {language}")
    else:
        language = detected_language
        logger.info(f"Detected language: {language}")
        # Automatically use the detected language and cache it
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
            except Exception:
                cache_data = {}
        cache_data['language'] = language
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)
    return language

def upload_to_s3_if_needed(video_path, s3, bucket, s3_key):
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        logger.info(f"S3 object '{bucket}/{s3_key}' already exists. Skipping upload.")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.info(f"Uploading {video_path} to S3 bucket '{bucket}' at '{s3_key}'")
            try:
                s3.upload_file(str(video_path), bucket, s3_key)
                logger.info("Upload to S3 completed.")
            except Exception as e:
                logger.error(f"Failed to upload to S3: {e}", exc_info=True)
                raise
        else:
            logger.error(f"Error checking S3 object: {e}", exc_info=True)
            raise

def start_or_resume_transcribe_job(transcribe, job_name, job_uri, language):
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
                MediaFormat='mp4',
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

def wait_for_transcribe_job(transcribe, job_name, status):
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

def fetch_transcript_from_s3(s3, bucket, key):
    transcript_response = s3.get_object(Bucket=bucket, Key=key)
    transcript_data = json.loads(transcript_response['Body'].read().decode('utf-8'))
    return transcript_data

def retry_transcribe_if_needed(transcribe, s3, job_name, job_uri, language, bucket, key):
    transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    logger.info(f"Retrying AWS Transcribe job: {job_name} for URI: {job_uri} with language: {language}")
    try:
        transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': job_uri},
            MediaFormat='mp4',
            LanguageCode=language,
            Settings={
                'ShowSpeakerLabels': True,
                'MaxSpeakerLabels': 2,
            }
        )
        # Wait for completion
        while True:
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            logger.info(f"Transcription job status (retry): {job_status}")
            if job_status in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            parsed = urlparse(transcript_uri)
            key = parsed.path.lstrip('/')
            logger.info(f"Fetching transcript from S3 (retry): bucket='{bucket}', key='{key}'")
            transcript_data = fetch_transcript_from_s3(s3, bucket, key)
            return transcript_data
        else:
            logger.error("Transcription job failed after retry.")
            raise Exception("Transcription job failed after retry")
    except Exception as e:
        logger.error(f"Failed to retry transcription job: {e}", exc_info=True)
        raise

def download_transcript(video_id: str) -> dict:
    logger.info("Step 1: Try extracting transcript with youtubeTranscript.py")
    result = try_youtube_transcript(video_id)
    if result:
        # Try to cache title if not present
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
        # Update cache with title if needed
        if title:
            cache_transcript(video_id, result["transcript"], result["language"], title)
        return {**result, "title": title} if title else result
    logger.info("Step 2: Download video if needed")
    temp_dir = CACHE_DIR
    temp_dir.mkdir(exist_ok=True)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = temp_dir / f"{video_id}.mp4"
    title = download_video_if_needed(video_id, video_url, video_path)
    logger.info("Step 3: Detect and confirm language")
    language = detect_and_confirm_language(video_id, title)
    logger.info("Step 4: Extract mp3 from video before uploading")
    mp3_path = temp_dir / f"{video_id}.mp3"
    if not mp3_path.exists():
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "192k", str(mp3_path)
            ]
            logger.info(f"Running ffmpeg to extract mp3: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"MP3 extraction complete: {mp3_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
    else:
        logger.info(f"MP3 file {mp3_path} already exists. Using cached mp3 for upload.")

    # Step 5: Transcribe with Whisper
    logger.info("Step 5: Transcribe audio with Whisper")
    try:
        transcript, whisper_language = transcribe_with_whisper(mp3_path)
        cache_transcript(video_id, transcript, whisper_language, title)
        logger.info("Transcript successfully cached using Whisper.")
        # Optionally delete mp4 and mp3 after successful transcription
        try:
            if video_path.exists():
                video_path.unlink()
                logger.info(f"Deleted video file: {video_path}")
            if mp3_path.exists():
                mp3_path.unlink()
                logger.info(f"Deleted mp3 file: {mp3_path}")
        except Exception as e:
            traceback.print_exc()
            raise
        return {"transcript": transcript, "language": whisper_language, "title": title}
    except Exception as e:
        logger.error("Whisper transcription failed, falling back to AWS routines.", exc_info=True)

    # ========== AWS TRANSCRIBE ROUTINES (fallback, kept for future use) ==========
    logger.info("Step 5: Upload mp3 to S3")
    import boto3
    s3 = boto3.client('s3')
    bucket = 'mdaudiosound'
    s3_key = f'input/{video_id}.mp3'
    upload_to_s3_if_needed(mp3_path, s3, bucket, s3_key)
    logger.info("Step 6: Start or resume AWS Transcribe job")
    transcribe = boto3.client('transcribe', 'us-west-2')
    job_name = f"transcribe-job-{video_id}"
    job_uri = f"s3://{bucket}/{s3_key}"
    status = start_or_resume_transcribe_job(transcribe, job_name, job_uri, language)
    logger.info("Step 7: Wait for AWS Transcribe job to complete")
    status = wait_for_transcribe_job(transcribe, job_name, status)
    logger.info("Step 8: Fetch transcript from URL")
    retry_transcribe = False
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        logger.info(f"Fetching transcript from URL: {transcript_uri}")
        try:
            response = requests.get(transcript_uri)
            response.raise_for_status()
            transcript_data = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch transcript from URL: {e}", exc_info=True)
            raise
        # Extract transcript text
        transcript = "\n".join(item['alternatives'][0]['content'] 
                             for item in transcript_data['results']['items']
                             if item['type'] == 'pronunciation')
        # Cache and return
        cache_transcript(video_id, transcript, language, title)
        logger.info("Transcript successfully cached.")
        # Delete mp4 and mp3 after successful transcription
        try:
            if video_path.exists():
                video_path.unlink()
                logger.info(f"Deleted video file: {video_path}")
            if mp3_path.exists():
                mp3_path.unlink()
                logger.info(f"Deleted mp3 file: {mp3_path}")
        except Exception as e:
            traceback.print_exc()
            raise
        return {"transcript": transcript, "language": language, "title": title}
    else:
        logger.error("Transcription job failed.")
        raise Exception("Transcription job failed")

def get_transcript(video_id: str) -> dict:
    cached = get_cached_transcript(video_id)
    if cached and 'transcript' in cached and 'language' in cached:
        # Try to ensure title is present in cache
        if 'title' not in cached or not cached['title']:
            try:
                ydl_opts = {"quiet": True}
                ydl_opts.update(YTDL_PROXY_OPTS)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                    title = info.get('title')
            except Exception:
                title = None
            if title:
                cache_transcript(video_id, cached['transcript'], cached['language'], title)
                cached['title'] = title
        print("Loaded transcript from cache.")
        return cached
    print("Downloading transcript...")
    return download_transcript(video_id)

# ========== GEMINI CLIENT ==========

def invoke_gemini_model(client, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
    """Invoke Gemini model with a prompt."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        
        if not response.candidates:
            raise ValueError("No candidates returned in response")
        
        candidate = response.candidates[0]
        
        finish_reason = getattr(candidate, 'finish_reason', None)
        if finish_reason is not None:
            try:
                finish_reason_str = str(finish_reason)
                if hasattr(finish_reason, 'name'):
                    finish_reason_str = finish_reason.name
                elif isinstance(finish_reason, int):
                    finish_reason_map = {
                        0: "FINISH_REASON_UNSPECIFIED",
                        1: "STOP",
                        2: "MAX_TOKENS",
                        3: "SAFETY",
                        4: "RECITATION",
                        5: "OTHER"
                    }
                    finish_reason_str = finish_reason_map.get(finish_reason, f"UNKNOWN({finish_reason})")
            except Exception:
                finish_reason_str = str(finish_reason)
        else:
            finish_reason_str = "UNKNOWN"
        
        has_parts = False
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                has_parts = True
        
        if not has_parts:
            error_msg = f"Response blocked or filtered. Finish reason: {finish_reason_str}"
            if finish_reason:
                finish_reason_val = finish_reason
                if hasattr(finish_reason, 'value'):
                    finish_reason_val = finish_reason.value
                elif not isinstance(finish_reason, int):
                    finish_reason_val = None
                
                if finish_reason_val == 3 or (isinstance(finish_reason_str, str) and 'SAFETY' in finish_reason_str.upper()):
                    error_msg += " (Content was blocked by safety filters)"
                elif finish_reason_val == 4 or (isinstance(finish_reason_str, str) and 'RECITATION' in finish_reason_str.upper()):
                    error_msg += " (Content was blocked due to recitation detection)"
                elif finish_reason_val == 2 or (isinstance(finish_reason_str, str) and 'MAX_TOKENS' in finish_reason_str.upper()):
                    error_msg += " (Response exceeded max tokens, but no content was returned)"
            raise ValueError(error_msg)
        
        try:
            return response.text.strip()
        except (ValueError, AttributeError) as e:
            text_parts = []
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
            
            if text_parts:
                return " ".join(text_parts).strip()
            else:
                raise ValueError(f"Could not extract text from response. Finish reason: {finish_reason_str}")
        
    except ValueError as e:
        print(f"Error invoking Gemini model: {e}")
        raise
    except Exception as e:
        print(f"Error invoking Gemini model: {e}")
        raise

def summarize_transcript(model, transcript: str) -> str:
    prompt = (
        "Summarize the following YouTube video transcript in a concise paragraph. "
        "Focus on the main points and key takeaways. Create a structure with some bullet points that is easy to understand and follow.\n\nTranscript:\n" + transcript
    )
    print("\nSummarizing transcript with Gemini...")
    summary = invoke_gemini_model(model, prompt, max_tokens=8000, temperature=0.3)
    # Remove <think>...</think> section if present (multiline)
    summary = re.sub(r'<think>[\s\S]*?</think>', '', summary, flags=re.DOTALL).strip()
    print("\n===== SUMMARY =====\n" + summary + "\n===================\n")
    return summary

def chunk_text(text: str, chunk_size: int = 20000, overlap: int = 2000) -> list[str]:
    """
    Split text into chunks with overlapping windows.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # Move start position forward, accounting for overlap
        start = end - overlap
        
        # If we've reached the end, break
        if end >= len(text):
            break
    
    return chunks

def process_chunk_for_question(model, chunk: str, question: str, chunk_index: int, total_chunks: int, max_retries: int = 3) -> dict:
    """
    Process a single transcript chunk to answer a question.
    Returns JSON with answer and relevance flag.
    
    Args:
        model: Gemini model instance
        chunk: Text chunk to process
        question: The question to answer
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks
        max_retries: Maximum number of retries on error
    
    Returns:
        Dictionary with 'answer', 'is_relevant', and 'chunk_index'
    """
    # Reduce chunk size if it's too long (to avoid MAX_TOKENS errors)
    # If chunk is > 10k chars, truncate it
    if len(chunk) > 10000:
        chunk = chunk[:10000] + "... [truncated]"
    
    prompt = f"""You are analyzing a portion of a YouTube video transcript (chunk {chunk_index + 1} of {total_chunks}).

Question: {question}

Transcript chunk:
{chunk}

Please analyze this chunk and provide a JSON response with the following structure:
{{
    "is_relevant": true/false,  // Whether this chunk contains information relevant to answering the question
    "answer": "Your answer based on this chunk, or empty string if not relevant"
}}

If the chunk is relevant, provide a concise answer based on this chunk. If not relevant, set is_relevant to false and answer to empty string.
Return ONLY valid JSON, no other text."""

    response_text = None
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # Reduce max_tokens on retry to avoid MAX_TOKENS errors
            max_tokens = 3000 if attempt > 0 else 4000
            
            response_text = invoke_gemini_model(model, prompt, max_tokens=max_tokens, temperature=0.3)
            
            # Try to extract JSON from response (in case there's extra text)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
            
            result = json.loads(response_text)
            
            # Validate structure
            if 'is_relevant' not in result or 'answer' not in result:
                return {
                    'is_relevant': False,
                    'answer': '',
                    'chunk_index': chunk_index
                }
            
            result['chunk_index'] = chunk_index
            return result
            
        except ValueError as e:
            # MAX_TOKENS or other model errors
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                logger.warning(f"Error processing chunk {chunk_index + 1} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Try reducing chunk size further on retry
                if "MAX_TOKENS" in str(e) and len(chunk) > 10000:
                    chunk = chunk[:10000] + "... [truncated]"
                    prompt = f"""You are analyzing a portion of a YouTube video transcript (chunk {chunk_index + 1} of {total_chunks}).

Question: {question}

Transcript chunk:
{chunk}

Please analyze this chunk and provide a JSON response with the following structure:
{{
    "is_relevant": true/false,
    "answer": "Your answer based on this chunk, or empty string if not relevant"
}}

Return ONLY valid JSON, no other text."""
            else:
                # Last attempt or non-retryable error
                break
                
        except json.JSONDecodeError as e:
            # JSON parsing error - we have response_text but it's not valid JSON
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.warning(f"JSON parsing error for chunk {chunk_index + 1} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Last attempt - try to extract something useful
                if response_text and len(response_text.strip()) > 20:
                    # Try to infer relevance from response text
                    return {
                        'is_relevant': True,  # Assume relevant if we got substantial text
                        'answer': response_text.strip()[:500],  # Limit answer length
                        'chunk_index': chunk_index
                    }
                break
    
    # If we get here, all retries failed
    logger.error(f"Failed to process chunk {chunk_index + 1} after {max_retries} attempts: {last_exception}")
    
    # Return a safe default
    return {
        'is_relevant': False,
        'answer': '',
        'chunk_index': chunk_index
    }

def combine_chunk_answers(model, relevant_answers: list[dict], question: str, summary: str) -> str:
    """
    Combine answers from relevant chunks and generate final answer.
    
    Args:
        model: Gemini model instance
        relevant_answers: List of dicts with 'answer' and 'chunk_index' from relevant chunks
        question: The original question
        summary: The video summary for context
    
    Returns:
        Final combined answer
    """
    if not relevant_answers:
        return "I couldn't find relevant information in the transcript to answer this question."
    
    # Combine all relevant answers
    combined_answers = "\n\n".join([
        f"Answer from chunk {ans['chunk_index'] + 1}:\n{ans['answer']}"
        for ans in relevant_answers
        if ans.get('is_relevant', False) and ans.get('answer', '').strip()
    ])
    
    prompt = f"""You are answering a question about a YouTube video based on multiple relevant transcript chunks.

Question: {question}

Video Summary (for context):
{summary}

Relevant information from transcript chunks:
{combined_answers}

Please provide a comprehensive, coherent answer that synthesizes the information from all relevant chunks. 
If there are contradictions or different perspectives, mention them. 
Make sure your answer is well-structured and directly addresses the question."""

    final_answer = invoke_gemini_model(model, prompt, max_tokens=16000, temperature=0.3)
    return final_answer.strip()

def answer_question_with_chunking(model, question: str, transcript: str, summary: str, chat_history: list, initial_context: list) -> str:
    """
    Answer a question using chunking approach for long transcripts.
    
    Args:
        model: Gemini model instance
        question: The question to answer
        transcript: Full transcript text
        summary: Video summary
        chat_history: Previous Q&A pairs
        initial_context: Initial context messages
    
    Returns:
        Final answer
    """
    # Estimate transcript size (roughly 4 chars per token, but be conservative)
    # Use chunking if transcript is > 20k characters (roughly 5k tokens)
    CHUNK_THRESHOLD = 20000
    
    if len(transcript) <= CHUNK_THRESHOLD:
        # Transcript is short enough, use regular approach
        prompt_parts = []
        for msg in initial_context:
            if msg["role"] == "system":
                prompt_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                prompt_parts.append(f"User: {msg['content']}")
        
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        for turn in recent_history:
            prompt_parts.append(f"User: {turn['question']}")
            prompt_parts.append(f"Assistant: {turn['answer']}")
        
        prompt_parts.append(f"User: {question}")
        prompt_parts.append(f"User: Here is the full transcript for detailed reference:\n{transcript}")
        
        prompt = "\n\n".join(prompt_parts)
        answer = invoke_gemini_model(model, prompt, max_tokens=16000, temperature=0.3)
        answer = re.sub(r'<think>[\s\S]*?</think>', '', answer, flags=re.DOTALL).strip()
        return answer
    
    # Use chunking approach
    print(f"\nTranscript is long ({len(transcript)} chars). Using chunking approach...")
    
    # Split transcript into chunks (use smaller size to avoid MAX_TOKENS errors)
    chunks = chunk_text(transcript, chunk_size=12000, overlap=1500)
    print(f"Split transcript into {len(chunks)} chunks. Processing each chunk...")
    
    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}...", end='', flush=True)
        result = process_chunk_for_question(model, chunk, question, i, len(chunks))
        chunk_results.append(result)
        if result.get('is_relevant', False):
            print(f" ✓ (relevant)")
        else:
            print(f" - (not relevant)")
    
    # Filter to only relevant chunks
    relevant_answers = [r for r in chunk_results if r.get('is_relevant', False) and r.get('answer', '').strip()]
    
    print(f"\nFound {len(relevant_answers)} relevant chunk(s). Combining answers...")
    
    # Combine relevant answers
    final_answer = combine_chunk_answers(model, relevant_answers, question, summary)
    
    return final_answer

def answer_question(model, question: str, chat_history: list, initial_context: list, transcript: str = None) -> str:
    # Extract summary from initial_context
    summary = ""
    for msg in initial_context:
        if msg["role"] == "user" and "summary" in msg['content'].lower():
            # Extract summary text
            content = msg['content']
            if "Here is a summary of the video:" in content:
                summary = content.split("Here is a summary of the video:")[1].split("\n\nThe full transcript")[0].strip()
            break
    
    # If transcript is provided, use chunking approach for long transcripts
    if transcript:
        return answer_question_with_chunking(model, question, transcript, summary, chat_history, initial_context)
    
    # Otherwise, use regular approach without transcript
    prompt_parts = []
    
    # Add initial context (system message and summary)
    for msg in initial_context:
        if msg["role"] == "system":
            prompt_parts.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
    
    # Limit chat history to last 10 exchanges to avoid prompt bloat
    recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
    
    # Add chat history
    for turn in recent_history:
        prompt_parts.append(f"User: {turn['question']}")
        prompt_parts.append(f"Assistant: {turn['answer']}")
    
    # Add current question
    prompt_parts.append(f"User: {question}")
    
    # Join all parts
    prompt = "\n\n".join(prompt_parts)
    
    answer = invoke_gemini_model(model, prompt, max_tokens=16000, temperature=0.3)
    # Remove <think>...</think> section if present (multiline)
    answer = re.sub(r'<think>[\s\S]*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer

def load_url_history():
    if HISTORY_JSON.exists():
        try:
            with open(HISTORY_JSON, "r", encoding="utf-8") as f:
                history = json.load(f)
                # Migrate old entries that don't have date_requested
                migrated = False
                for item in history:
                    if 'date_requested' not in item:
                        item['date_requested'] = datetime.datetime.now().isoformat()
                        migrated = True
                # Save migrated history if any changes were made
                if migrated:
                    save_url_history(history)
                return history
        except Exception:
            return []
    return []

def save_url_history(history):
    try:
        with open(HISTORY_JSON, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save URL history: {e}", exc_info=True)
        traceback.print_exc()

def add_to_url_history(url, title):
    history = load_url_history()
    # Remove duplicates (by url)
    history = [item for item in history if item.get("url") != url]
    history.insert(0, {"url": url, "title": title, "date_requested": datetime.datetime.now().isoformat()})
    save_url_history(history)

# ========== MAIN CHATBOT LOGIC ==========
def main():
    print("YouTube Video Summarizer & Chatbot (Claude 3.5 Sonnet on AWS Bedrock)")
    
    # Display proxy configuration status
    if PROXY_CONFIG:
        print(f"🔌 Proxy configuration detected:")
        for proxy_type, proxy_url in PROXY_CONFIG.items():
            # Mask credentials in proxy URL for security
            masked_url = proxy_url
            if '@' in proxy_url:
                # URL has credentials, mask them
                parts = proxy_url.split('@')
                protocol = parts[0].split('://')[0]
                masked_url = f"{protocol}://***:***@{parts[1]}"
            print(f"   {proxy_type.upper()}: {masked_url}")
    else:
        print("🔌 No proxy configured - using direct connection")
    
    # Check Tor status
    if MULTI_TOR_AVAILABLE:
        print("✅ Multi-Tor setup detected - will use for transcript downloading")
        # Test if any Tor instances are running
        tor_instances_running = 0
        tor_ports = [9050, 9052, 9054, 9056, 9058]
        for port in tor_ports:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()
                if result == 0:
                    tor_instances_running += 1
            except Exception:
                pass
        
        if tor_instances_running > 0:
            print(f"✅ {tor_instances_running} Tor instances running - will use multi-Tor for transcript downloading")
        else:
            print("⚠️ No Tor instances running - start with: /Users/haha/github/multi-tor/tor-manager.sh start")
            if PROXY_CONFIG:
                print("📡 Will fall back to configured proxy for downloads")
    elif TOR_AVAILABLE:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(("127.0.0.1", 9050))
            sock.close()
            if result == 0:
                print("✅ Single Tor proxy detected - will use for transcript downloading")
            else:
                print("⚠️ Tor not running - will use direct connection (may be blocked)")
                if PROXY_CONFIG:
                    print("📡 Will use configured proxy instead")
        except Exception:
            print("⚠️ Tor not running - will use direct connection (may be blocked)")
            if PROXY_CONFIG:
                print("📡 Will use configured proxy instead")
    else:
        print("⚠️ PySocks not installed - install with: pip install PySocks")
        if PROXY_CONFIG:
            print("📡 Will use HTTP/HTTPS proxy instead of SOCKS")
    # Load history and prompt user
    url_history = load_url_history()
    url = None
    if url_history:
        print("\nPreviously used YouTube videos:")
        # Show oldest first, so reverse the list
        oldest_first = list(reversed(url_history))
        total = len(oldest_first)
        for idx, item in enumerate(oldest_first, 1):
            # Numbering: latest video (last in list) is 1, oldest is total
            number = total - idx + 1
            # Format date if available, otherwise show "Unknown date"
            date_str = "Unknown date"
            if 'date_requested' in item:
                try:
                    date_obj = datetime.datetime.fromisoformat(item['date_requested'])
                    date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    date_str = "Unknown date"
            print(f"  {number}. {item['title']}\n     {item['url']}\n     Requested: {date_str}")
        print("\nEnter a new YouTube video URL, or select a number from above:")
        user_input = input("Your choice: ").strip()
        # Map user input to the correct url
        if user_input.isdigit() and 1 <= int(user_input) <= total:
            # Since 1 is latest (last in oldest_first), map accordingly
            url = oldest_first[total - int(user_input)]["url"]
        else:
            url = user_input
    else:
        url = input("Enter YouTube video URL: ").strip()
    try:
        video_id = extract_video_id(url)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    transcript_dict = get_transcript(video_id)
    model = create_gemini_client()
    # Get title for history
    # Try to get title from cache or fallback to YouTube API
    title = None
    cache_file = CACHE_DIR / f"{video_id}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                title = cache_data.get("title")
        except Exception:
            title = None
    if not title:
        # Try yt-dlp to get title (with proxy support)
        try:
            ydl_opts = {"quiet": True}
            ydl_opts.update(YTDL_PROXY_OPTS)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', url)
        except Exception:
            title = url
    add_to_url_history(url, title)
    summary = summarize_transcript(model, transcript_dict['transcript'])
    chat_history = []
    # Prepare initial context (system and user messages)
    # Store transcript separately - don't include it in every request to avoid prompt bloat
    transcript_text = transcript_dict['transcript']
    initial_context = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
        {"role": "user", "content": f"Here is a summary of the video: {summary}\n\nThe full transcript is available if you need specific details, but try to answer based on the summary first."}
    ]
    print("\nYou can now ask questions about the video. Type 'exit' to quit, or enter a new YouTube URL to start over.")
    while True:
        question = input("\nYour question (or new YouTube URL): ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        # Check if input is a YouTube URL
        if ("youtube.com" in question or "youtu.be" in question) and len(question.split()) == 1:
            # Try to extract video ID and restart
            try:
                video_id = extract_video_id(question)
            except Exception as e:
                print(f"Error: {e}")
                continue
            transcript_dict = get_transcript(video_id)
            # Get title for history
            title = None
            cache_file = CACHE_DIR / f"{video_id}.json"
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
                        info = ydl.extract_info(question, download=False)
                        title = info.get('title', question)
                except Exception:
                    title = question
            add_to_url_history(question, title)
            summary = summarize_transcript(model, transcript_dict['transcript'])
            chat_history = []
            transcript_text = transcript_dict['transcript']
            initial_context = [
                {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
                {"role": "user", "content": f"Here is a summary of the video: {summary}\n\nThe full transcript is available if you need specific details, but try to answer based on the summary first."}
            ]
            print("\nYou can now ask questions about the new video. Type 'exit' to quit, or enter a new YouTube URL to start over.")
            continue
        answer = answer_question(model, question, chat_history, initial_context, transcript_text)
        print("\nAnswer:", answer)
        chat_history.append({"question": question, "answer": answer})

if __name__ == "__main__":
    try:
        import readline
        import atexit
        histfile = str(Path(__file__).parent / ".youtubesummary_history")
        try:
            readline.read_history_file(histfile)
        except FileNotFoundError:
            pass
        atexit.register(readline.write_history_file, histfile)
    except ImportError:
        print("Module readline not available. Command history will not be enabled.")
    main()
