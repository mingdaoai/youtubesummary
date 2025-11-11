#!/usr/bin/env python3
"""
Test script to transcribe a single YouTube video.
"""

import sys
from pathlib import Path
import yt_dlp
import subprocess
import os

# Import functions from youtubeSummarize.py
from youtubeSummarize import try_youtube_transcript, download_video_if_needed, transcribe_with_whisper, detect_and_confirm_language, cache_transcript, CACHE_DIR

def test_transcribe_video(video_id: str):
    """Test transcribing a single video."""
    print(f"\n{'='*60}")
    print(f"Testing transcription for video: {video_id}")
    print(f"{'='*60}\n")
    
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Step 1: Try YouTube transcript API
    print("Step 1: Trying YouTube transcript API...")
    transcript_data = None
    transcript_method = None
    try:
        result = try_youtube_transcript(video_id)
        if result and result.get('transcript'):
            transcript_data = result
            transcript_method = "YouTube API"
            print(f"✅ Got transcript from YouTube API")
            print(f"   Language: {result.get('language', 'unknown')}")
            print(f"   Title: {result.get('title', 'unknown')}")
            print(f"   Transcript length: {len(result.get('transcript', ''))} characters")
            return True
    except Exception as e:
        print(f"⚠️  YouTube transcript API failed: {e}")
    
    # Step 2: Download video and use Whisper
    if not transcript_data or not transcript_data.get('transcript'):
        print(f"\nStep 2: No transcript available via API, downloading video for Whisper transcription...")
        sys.stdout.flush()
        
        temp_dir = CACHE_DIR
        temp_dir.mkdir(exist_ok=True)
        video_path = temp_dir / f"{video_id}.mp4"
        mp3_path = temp_dir / f"{video_id}.mp3"
        
        # Download video
        if not video_path.exists():
            print(f"📥 Downloading video: {video_id}")
            sys.stdout.flush()
            cookies_path = os.path.join(os.getcwd(), 'cookies.txt')
            downloaded = False
            last_error = None
            
            # First, try to list available formats
            print(f"🔍 Checking available formats for video...")
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
                        print(f"   Found {len(formats)} available formats")
                        # Show first few formats
                        for i, fmt in enumerate(formats[:5]):
                            print(f"   Format {fmt.get('format_id')}: {fmt.get('height', '?')}p, {fmt.get('ext')}, {fmt.get('vcodec', '?')}")
            except Exception as e:
                print(f"   Could not list formats: {e}")
            
            # Try format selectors
            format_list = [
                None,  # Auto-select
                'worst',
                'worstvideo+worstaudio/worst',
                'best[height<=360]',
                'best[height<=480]',
                'best[height<=720]',
                'best',
            ]
            
            for fmt in format_list:
                try:
                    print(f"   Trying format: {fmt if fmt else 'auto-select'}")
                    sys.stdout.flush()
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
                        # YouTube extractor options
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['android', 'web'],  # Try android first, then web
                            }
                        },
                    }
                    if fmt is not None:
                        ydl_opts['format'] = fmt
                    
                    if os.path.exists(cookies_path):
                        ydl_opts['cookiefile'] = cookies_path
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(video_url, download=True)
                        title = info.get('title', 'Unknown')
                    
                    format_used = fmt if fmt else "auto-selected"
                    print(f"✅ Downloaded video (format: {format_used})")
                    print(f"   Title: {title}")
                    downloaded = True
                    break
                except Exception as e:
                    error_msg = str(e)
                    last_error = error_msg
                    print(f"   ❌ Format '{fmt if fmt else 'auto'}' failed: {error_msg[:100]}")
                    # Check if it's a format-specific error or a more general error
                    if 'format' not in error_msg.lower() and 'not available' not in error_msg.lower():
                        print(f"   This appears to be a non-format error (restricted, private, etc.)")
                        break
                    continue
            
            if not downloaded:
                print(f"❌ All download attempts failed")
                print(f"   Last error: {last_error}")
                return False
        else:
            print(f"📁 Using cached video file")
            try:
                with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    title = info.get('title', 'Unknown')
                    print(f"   Title: {title}")
            except Exception:
                pass
        
        # Extract audio
        print(f"\n🎵 Extracting audio from video...")
        sys.stdout.flush()
        if not mp3_path.exists():
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "192k", str(mp3_path)
            ]
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Audio extracted")
        else:
            print(f"📁 Using cached audio file")
        
        # Transcribe with Whisper
        print(f"\n🎤 Transcribing with Whisper (this may take a while)...")
        sys.stdout.flush()
        try:
            transcript, whisper_language = transcribe_with_whisper(mp3_path)
            print(f"✅ Whisper transcription complete")
            print(f"   Language detected: {whisper_language}")
            print(f"   Transcript length: {len(transcript)} characters")
            print(f"   First 200 chars: {transcript[:200]}...")
            
            # Cache the transcript
            cache_transcript(video_id, transcript, whisper_language, title)
            print(f"✅ Transcript cached")
            return True
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_video.py <video_id>")
        print("Example: python test_single_video.py lum_iVpY38M")
        sys.exit(1)
    
    video_id = sys.argv[1]
    success = test_transcribe_video(video_id)
    
    if success:
        print(f"\n{'='*60}")
        print("✅ SUCCESS: Video transcribed successfully!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("❌ FAILED: Could not transcribe video")
        print(f"{'='*60}\n")
        sys.exit(1)

