#!/usr/bin/env python3
"""
Test script for the YouTube Transcript MCP Server

This script tests the MCP server functionality by directly calling the functions.
"""

import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Test the functions directly (not through MCP)
from youtubeTranscript import (
    extract_video_id, 
    download_youtube_transcript, 
    get_available_languages
)

def test_extract_video_id():
    """Test video ID extraction."""
    print("\n=== Testing Video ID Extraction ===")
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "dQw4w9WgXcQ"
    ]
    
    for url in test_urls:
        video_id = extract_video_id(url)
        print(f"URL: {url}")
        print(f"Video ID: {video_id}")
        print()


def test_get_available_languages():
    """Test getting available languages."""
    print("\n=== Testing Available Languages ===")
    
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        languages = get_available_languages(test_url)
        if languages:
            print(f"Found {len(languages)} available languages:")
            for lang in languages:
                print(f"  - {lang['language']} ({lang['language_code']}) "
                      f"[Generated: {lang['is_generated']}, Translatable: {lang['is_translatable']}]")
        else:
            print("No languages found")
    except Exception as e:
        print(f"Error: {e}")


def test_download_transcript():
    """Test downloading a transcript."""
    print("\n=== Testing Transcript Download ===")
    
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    print(f"Downloading transcript for: {test_url}")
    
    try:
        transcript = download_youtube_transcript(test_url)
        if transcript:
            print(f"Successfully downloaded transcript ({len(transcript)} characters)")
            print(f"\nFirst 200 characters:\n{transcript[:200]}...")
        else:
            print("Failed to download transcript")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all tests."""
    print("YouTube Transcript MCP Server Test")
    print("=" * 50)
    
    # Run tests
    test_extract_video_id()
    
    # Uncomment the following lines to test actual transcript downloads
    # Note: These will make real API calls to YouTube
    # test_get_available_languages()
    # test_download_transcript()
    
    print("\n✅ Basic tests completed!")
    print("\nNote: To test transcript downloading, uncomment the test functions in the main() function.")


if __name__ == "__main__":
    main()
