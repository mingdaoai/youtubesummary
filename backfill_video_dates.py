#!/usr/bin/env python3
"""
Backfill upload dates for existing transcript files.
This script reads existing transcript JSON files and adds upload_date field
by fetching full metadata from YouTube using yt-dlp.
"""

import json
import sys
from pathlib import Path
import yt_dlp
from typing import Dict, Optional
import time

def get_video_upload_date(video_id: str) -> Optional[str]:
    """
    Get upload date for a single video using yt-dlp.
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Upload date as ISO format string, or None if failed
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
        print(f"  ⚠️  Error fetching date for {video_id}: {e}")
        return None
    
    return None


def backfill_dates_for_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Backfill upload date for a single transcript file.
    
    Args:
        file_path: Path to transcript JSON file
        dry_run: If True, don't write changes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read existing file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Skip if date already exists
        if 'upload_date' in data and data['upload_date']:
            print(f"  ✓ {file_path.name} already has date: {data['upload_date']}")
            return True
        
        video_id = data.get('video_id')
        if not video_id:
            print(f"  ⚠️  {file_path.name} has no video_id")
            return False
        
        print(f"  🔍 Fetching date for {video_id}...", end=' ', flush=True)
        upload_date = get_video_upload_date(video_id)
        
        if upload_date:
            data['upload_date'] = upload_date
            print(f"✓ {upload_date}")
            
            if not dry_run:
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                return True
            else:
                print(f"  [DRY RUN] Would set date to: {upload_date}")
                return True
        else:
            print("✗ Failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {file_path.name}: {e}")
        return False


def backfill_channel_dates(channel_dir: Path, dry_run: bool = False, limit: Optional[int] = None):
    """
    Backfill upload dates for all transcript files in a channel directory.
    
    Args:
        channel_dir: Directory containing transcript JSON files
        dry_run: If True, don't write changes
        limit: Maximum number of files to process (None for all)
    """
    transcript_files = sorted(channel_dir.glob("*.json"))
    
    # Filter out non-transcript files
    transcript_files = [f for f in transcript_files 
                       if f.name not in ['channel_info.json', 'videos_list.json', 'download_stats.json', 'download.log']]
    
    total = len(transcript_files)
    if limit:
        transcript_files = transcript_files[:limit]
        print(f"Processing {len(transcript_files)} of {total} files (limit: {limit})")
    else:
        print(f"Processing {total} transcript files...")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    print("-" * 80)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(transcript_files, 1):
        print(f"[{i}/{len(transcript_files)}] {file_path.name}")
        
        # Check if already has date
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'upload_date' in data and data['upload_date']:
                skip_count += 1
                print(f"  ✓ Already has date: {data['upload_date']}")
                continue
        except:
            pass
        
        if backfill_dates_for_file(file_path, dry_run):
            success_count += 1
        else:
            fail_count += 1
        
        # Rate limiting - be nice to YouTube
        if i < len(transcript_files):
            time.sleep(0.5)  # 0.5 second delay between requests
    
    print("-" * 80)
    print(f"Summary:")
    print(f"  ✓ Success: {success_count}")
    print(f"  ⊘ Skipped (already has date): {skip_count}")
    print(f"  ✗ Failed: {fail_count}")
    print(f"  Total: {len(transcript_files)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Backfill upload dates for existing transcript files'
    )
    parser.add_argument(
        'channel_dir',
        type=Path,
        help='Directory containing transcript JSON files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of files to process (for testing)'
    )
    
    args = parser.parse_args()
    
    if not args.channel_dir.exists():
        print(f"Error: Directory does not exist: {args.channel_dir}")
        sys.exit(1)
    
    backfill_channel_dates(args.channel_dir, args.dry_run, args.limit)


if __name__ == '__main__':
    main()

