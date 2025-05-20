# youtubesummary

A command-line tool to summarize YouTube videos and answer questions about their content using AI. It downloads the transcript of a YouTube video, generates a concise summary with the DeepSeek model via the Groq API, and allows interactive Q&A about the video. Transcripts are cached for efficiency. Requires a Groq API key and supports both standard and Shorts YouTube URLs.

## Example AI Prompt to Generate This Code

If you are using Cursor, you can generate the code of this tool using the following prompt:

"Write a Python script that:
- Downloads the transcript of a YouTube video (supporting both standard and Shorts URLs)
- Summarizes the transcript using an AI model via an API (e.g., Groq/DeepSeek)
- Caches transcripts locally
- Allows the user to ask questions about the video content in an interactive chat mode
- Handles API keys securely and provides clear error messages."