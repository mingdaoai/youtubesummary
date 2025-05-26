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

## How it works

1. **Transcript Retrieval**: The tool first attempts to fetch the transcript using the YouTubeTranscriptApi. If unavailable, it downloads the video and uses AWS Transcribe to generate a transcript, caching results for future use.
2. **Summarization**: The transcript is summarized using the DeepSeek model via the Groq API, providing a concise overview of the video's main points.
3. **Interactive Q&A**: After summarization, users can interactively ask questions about the video. The tool uses the transcript and summary as context to answer questions using the AI model.
4. **Caching**: Transcripts and detected languages are cached locally to speed up repeated queries and avoid redundant downloads or transcriptions.
5. **Error Handling**: The tool provides clear error messages, prints stack traces for exceptions, and raises errors for unhandled cases.
6. **API Key Management**: API keys are read securely from the user's home directory (e.g., `~/.mingdaoai/groq.key`).
7. **Logging**: Logging is configured to include full file paths and line numbers for easier debugging.