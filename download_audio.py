#!/usr/bin/env python
"""
Download audio from a YouTube video.
Usage: python download_audio.py <youtube_url>
"""
import sys
import yt_dlp
import os
from pathlib import Path

def download_audio(url: str, output_dir: str = None):
    """
    Download audio from a YouTube video.
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file (default: current directory)
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path.cwd()
    
    # Configure yt-dlp options for audio-only download
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
    }
    
    # Check for cookies file (like in the main script)
    cookies_path = os.path.join(os.getcwd(), 'cookies.txt')
    if os.path.exists(cookies_path):
        ydl_opts['cookiefile'] = cookies_path
        print(f"Using cookies from {cookies_path}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get the title
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            print(f"Downloading audio: {title}")
            
            # Download the audio
            ydl.download([url])
            
            # Find the downloaded file
            # The file will be saved as title.mp3 (after post-processing)
            audio_file = output_path / f"{title}.mp3"
            
            # yt-dlp might sanitize the filename, so check for actual file
            if not audio_file.exists():
                # List files in output directory to find the actual filename
                files = list(output_path.glob("*.mp3"))
                if files:
                    # Get the most recently modified file
                    audio_file = max(files, key=lambda p: p.stat().st_mtime)
            
            print(f"\n✅ Audio downloaded successfully: {audio_file}")
            return str(audio_file)
            
    except Exception as e:
        print(f"❌ Error downloading audio: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_audio.py <youtube_url> [output_dir]", file=sys.stderr)
        print("Example: python download_audio.py https://www.youtube.com/watch?v=dQw4w9WgXcQ", file=sys.stderr)
        sys.exit(1)
    
    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    download_audio(url, output_dir)

if __name__ == "__main__":
    main()
