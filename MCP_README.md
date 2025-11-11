# YouTube Transcript MCP Server

This directory contains an MCP (Model Context Protocol) server that provides YouTube transcript extraction capabilities.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the MCP Server

### Standalone Mode

Run the server directly:

```bash
python mcp_youtube_transcript.py
```

### As an MCP Server

The server can be used with MCP clients. Configure it in your MCP client configuration:

```json
{
  "mcpServers": {
    "youtube-transcript": {
      "command": "python",
      "args": ["/path/to/youtubesummary/mcp_youtube_transcript.py"]
    }
  }
}
```

## Available Tools

### 1. `get_youtube_transcript`

Get the transcript of a YouTube video from its URL or video ID.

**Parameters:**
- `url` (string, required): YouTube URL or video ID
  - Example: `"https://www.youtube.com/watch?v=dQw4w9WgXcQ"`
  - Or just: `"dQw4w9WgXcQ"`
- `language_codes` (list[string], optional): Preferred language codes
  - Example: `["en", "en-US"]`
  - If None, uses any available transcript

**Returns:**
- Transcript text as a string

**Example Usage:**
```python
transcript = get_youtube_transcript(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    language_codes=["en"]
)
```

### 2. `get_video_id_from_url`

Extract the YouTube video ID from a URL.

**Parameters:**
- `url` (string, required): YouTube URL in any format

**Returns:**
- Video ID string

**Example Usage:**
```python
video_id = get_video_id_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
# Returns: "dQw4w9WgXcQ"
```

### 3. `get_available_transcript_languages`

Get the list of available transcript languages for a YouTube video.

**Parameters:**
- `url` (string, required): YouTube URL or video ID

**Returns:**
- Formatted string listing available languages with details about generation status

**Example Usage:**
```python
languages = get_available_transcript_languages("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

## Features

- **No Authentication Required**: Uses the public YouTube transcript API without requiring OAuth tokens
- **Multiple Language Support**: Supports multiple language codes and automatic fallback
- **Error Handling**: Comprehensive error handling and logging
- **Flexible Input**: Accepts full YouTube URLs or just video IDs

## Dependencies

- `youtube-transcript-api`: For fetching transcripts from YouTube
- `mcp`: Model Context Protocol server framework
- Other dependencies as listed in `requirements.txt`

## Note

This server does not require YouTube authentication. It uses the public YouTube transcript API (`youtube-transcript-api`), which can access publicly available transcripts without any tokens or keys.

However, be aware that:
- Not all videos have transcripts available
- YouTube may implement rate limiting
- Some videos may be region-restricted or age-restricted
