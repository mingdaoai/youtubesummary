# youtubesummary

A command-line tool to summarize YouTube videos and answer questions about their content using AI. It downloads the transcript of a YouTube video, generates a concise summary with the Claude 3.5 Sonnet model via AWS Bedrock, and allows interactive Q&A about the video. Transcripts are cached for efficiency. Requires AWS credentials and supports both standard and Shorts YouTube URLs.

## Example AI Prompt to Generate This Code

If you are using Cursor, you can generate the code of this tool using the following prompt:

"Write a Python script that:
- Downloads the transcript of a YouTube video (supporting both standard and Shorts URLs)
- Summarizes the transcript using an AI model via AWS Bedrock (Claude 3.5 Sonnet)
- Caches transcripts locally
- Allows the user to ask questions about the video content in an interactive chat mode
- Handles AWS credentials securely and provides clear error messages."

## Prerequisites

1. **AWS Account**: You need an AWS account with access to Amazon Bedrock
2. **AWS Credentials**: Configure your AWS credentials using one of these methods:
   - Run `aws configure` and enter your access key, secret key, and region
   - Set environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
   - Use IAM roles if running on EC2
3. **Bedrock Model Access**: Ensure you have access to the Claude 3.5 Sonnet model in your AWS region
4. **Python Dependencies**: Install required packages with `pip install -r requirements.txt`

## How it works

1. **Transcript Retrieval**: The tool first attempts to fetch the transcript using the YouTubeTranscriptApi. If unavailable, it downloads the video and uses Whisper or AWS Transcribe to generate a transcript, caching results for future use.
2. **Summarization**: The transcript is summarized using the Claude 3.5 Sonnet model via AWS Bedrock, providing a concise overview of the video's main points.
3. **Interactive Q&A**: After summarization, users can interactively ask questions about the video. The tool uses the transcript and summary as context to answer questions using the AI model.
4. **Caching**: Transcripts and detected languages are cached locally to speed up repeated queries and avoid redundant downloads or transcriptions.
5. **Error Handling**: The tool provides clear error messages, prints stack traces for exceptions, and raises errors for unhandled cases.
6. **AWS Integration**: Uses AWS Bedrock for AI model access with proper credential management and error handling.
7. **Logging**: Logging is configured to include full file paths and line numbers for easier debugging.

## Usage

1. **Configure AWS Credentials**:
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region (e.g., us-east-1)
   ```

2. **Test the Setup**:
   ```bash
   python test_bedrock.py
   ```

3. **Run the Tool**:
   ```bash
   python youtubeSummarize.py
   ```

4. **Enter a YouTube URL** when prompted, or select from previously used videos.

## Supported Regions

Claude 3.5 Sonnet is available in the following AWS regions:
- US East (N. Virginia) - us-east-1
- US East (Ohio) - us-east-2  
- US West (Oregon) - us-west-2
- Europe (Ireland) - eu-west-1
- Asia Pacific (Tokyo) - ap-northeast-1

Make sure your AWS credentials are configured for one of these regions.