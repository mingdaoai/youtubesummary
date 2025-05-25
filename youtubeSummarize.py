#!/usr/bin/env python
import os
import sys
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
import groq
import re
import time
import json
import traceback
import yt_dlp
from urllib.parse import urlparse
import logging
import requests

# ========== CONFIGURATION ==========
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
GROQ_KEY_PATH = Path.home() / ".mingdaoai" / "groq.key"
DEEPSEEK_MODEL = "DeepSeek-R1-Distill-Llama-70b"

# ========== UTILS ==========
def load_groq_key():
    try:
        with open(GROQ_KEY_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Groq API key not found at {GROQ_KEY_PATH}")
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

def cache_transcript(video_id: str, transcript: str, language: str):
    cache_file = CACHE_DIR / f"{video_id}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"transcript": transcript, "language": language}, f, ensure_ascii=False)

def try_youtube_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'zh-Hant', 'zh-Hans'])
        transcript = "\n".join(item["text"] for item in transcript_list)
        language = 'en-US'
        cache_transcript(video_id, transcript, language)
        return {"transcript": transcript, "language": language}
    except Exception as e:
        import youtube_transcript_api
        if isinstance(e, getattr(youtube_transcript_api._errors, 'TranscriptsDisabled', type(None))):
            logging.error(f"Could not get transcript: {str(e)}")
        else:
            logging.error(f"Could not get transcript: {str(e)}", exc_info=True)
            traceback.print_exc()
        return None

def download_video_if_needed(video_id, video_url, video_path):
    if video_path.exists():
        logging.info(f"Video file {video_path} already exists. Using cached mp4 for transcription.")
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                title = info.get('title', '')
        except Exception as e:
            logging.error(f"yt-dlp failed to extract info for existing file: {e}", exc_info=True)
            title = ''
    else:
        response = input("Would you like to download and transcribe the video? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
        format_list = [
            'bestvideo+bestaudio/best',
            'best[ext=mp4]/best',
            'best',
            '18',  # mp4 360p
            '22',  # mp4 720p
        ]
        cookies_path = os.path.join(os.getcwd(), 'cookies.txt')
        last_exception = None
        for fmt in format_list:
            ydl_opts = {
                'outtmpl': str(video_path),
                'format': fmt,
                'merge_output_format': 'mp4',
            }
            if os.path.exists(cookies_path):
                ydl_opts['cookiefile'] = cookies_path
                logging.info(f"Using cookies from {cookies_path}")
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_url, download=True)
                    title = info.get('title', '')
                logging.info(f"Downloaded video with format: {fmt}")
                break  # Success!
            except Exception as e:
                logging.error(f"yt-dlp failed with format '{fmt}': {e}", exc_info=True)
                last_exception = e
                if '403' in str(e):
                    logging.warning("\nHTTP 403 Forbidden error detected. This may be due to age restriction, region lock, or YouTube signature changes.\n")
                    logging.warning("Try updating yt-dlp: pip install -U yt-dlp")
                    logging.warning("If the video requires login, export your YouTube cookies as cookies.txt and place it in the current directory.")
                continue
        else:
            logging.error("All format attempts failed.")
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
        logging.info(f"Using cached language: {language}")
    else:
        language = detected_language
        logging.info(f"Detected language: {language}")
        confirm = input("Is this correct? (y/n): ")
        if confirm.lower() != 'y':
            language = input("Enter language code (en-US/zh-CN): ")
        # Immediately cache the confirmed language
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
        logging.info(f"S3 object '{bucket}/{s3_key}' already exists. Skipping upload.")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            logging.info(f"Uploading {video_path} to S3 bucket '{bucket}' at '{s3_key}'")
            try:
                s3.upload_file(str(video_path), bucket, s3_key)
                logging.info("Upload to S3 completed.")
            except Exception as e:
                logging.error(f"Failed to upload to S3: {e}", exc_info=True)
                raise
        else:
            logging.error(f"Error checking S3 object: {e}", exc_info=True)
            raise

def start_or_resume_transcribe_job(transcribe, job_name, job_uri, language):
    job_exists = False
    try:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_exists = True
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        logging.info(f"Found existing AWS Transcribe job: {job_name} with status: {job_status}")
    except transcribe.exceptions.BadRequestException:
        job_exists = False
    if not job_exists:
        logging.info(f"Starting AWS Transcribe job: {job_name} for URI: {job_uri} with language: {language}")
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
            logging.error(f"Failed to start transcription job: {e}", exc_info=True)
            raise
    return status

def wait_for_transcribe_job(transcribe, job_name, status):
    logging.info(f"Waiting for AWS Transcribe job '{job_name}' to complete...")
    while True:
        try:
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            logging.info(f"Transcription job status: {job_status}")
            if job_status in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
            status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        except Exception as e:
            logging.error(f"Error while checking transcription job status: {e}", exc_info=True)
            raise
    return status

def fetch_transcript_from_s3(s3, bucket, key):
    transcript_response = s3.get_object(Bucket=bucket, Key=key)
    transcript_data = json.loads(transcript_response['Body'].read().decode('utf-8'))
    return transcript_data

def retry_transcribe_if_needed(transcribe, s3, job_name, job_uri, language, bucket, key):
    transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    logging.info(f"Retrying AWS Transcribe job: {job_name} for URI: {job_uri} with language: {language}")
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
            logging.info(f"Transcription job status (retry): {job_status}")
            if job_status in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            parsed = urlparse(transcript_uri)
            key = parsed.path.lstrip('/')
            logging.info(f"Fetching transcript from S3 (retry): bucket='{bucket}', key='{key}'")
            transcript_data = fetch_transcript_from_s3(s3, bucket, key)
            return transcript_data
        else:
            logging.error("Transcription job failed after retry.")
            raise Exception("Transcription job failed after retry")
    except Exception as e:
        logging.error(f"Failed to retry transcription job: {e}", exc_info=True)
        raise

def download_transcript(video_id: str) -> dict:
    # Step 1: Try YouTubeTranscriptApi
    result = try_youtube_transcript(video_id)
    if result:
        return result
    # Step 2: Download video if needed
    temp_dir = CACHE_DIR
    temp_dir.mkdir(exist_ok=True)
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = temp_dir / f"{video_id}.mp4"
    title = download_video_if_needed(video_id, video_url, video_path)
    # Step 3: Detect and confirm language
    language = detect_and_confirm_language(video_id, title)
    # Step 4: Upload to S3 if needed
    import boto3
    s3 = boto3.client('s3')
    bucket = 'mdaudiosound'
    s3_key = f'input/{video_id}.mp4'
    upload_to_s3_if_needed(video_path, s3, bucket, s3_key)
    # Step 5: Start or resume AWS Transcribe job
    transcribe = boto3.client('transcribe', 'us-west-2')
    job_name = f"transcribe-job-{video_id}"
    job_uri = f"s3://{bucket}/{s3_key}"
    status = start_or_resume_transcribe_job(transcribe, job_name, job_uri, language)
    # Step 6: Wait for job completion
    status = wait_for_transcribe_job(transcribe, job_name, status)
    # Step 7: Fetch transcript from URL, retry if needed
    retry_transcribe = False
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        logging.info(f"Fetching transcript from URL: {transcript_uri}")
        try:
            response = requests.get(transcript_uri)
            response.raise_for_status()
            transcript_data = response.json()
        except Exception as e:
            logging.error(f"Failed to fetch transcript from URL: {e}", exc_info=True)
            raise
        # Extract transcript text
        transcript = "\n".join(item['alternatives'][0]['content'] 
                             for item in transcript_data['results']['items']
                             if item['type'] == 'pronunciation')
        # Cache and return
        cache_transcript(video_id, transcript, language)
        logging.info("Transcript successfully cached.")
        return {"transcript": transcript, "language": language}
    else:
        logging.error("Transcription job failed.")
        raise Exception("Transcription job failed")

def get_transcript(video_id: str) -> dict:
    cached = get_cached_transcript(video_id)
    if cached and 'transcript' in cached and 'language' in cached:
        print("Loaded transcript from cache.")
        return cached
    print("Downloading transcript...")
    return download_transcript(video_id)

# ========== DEEPSEEK (GROQ) CLIENT ==========
def create_groq_client():
    api_key = load_groq_key()
    return groq.Client(api_key=api_key)

def summarize_transcript(client, transcript: str) -> str:
    prompt = (
        "Summarize the following YouTube video transcript in a concise paragraph. "
        "Focus on the main points and key takeaways.\n\nTranscript:\n" + transcript
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes YouTube videos."},
        {"role": "user", "content": prompt}
    ]
    print("\nSummarizing transcript with DeepSeek...")
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
        max_tokens=2000,
    )
    summary = chat_completion.choices[0].message.content.strip()
    # Remove <think>...</think> section if present (multiline)
    summary = re.sub(r'<think>[\s\S]*?</think>', '', summary, flags=re.DOTALL).strip()
    print("\n===== SUMMARY =====\n" + summary + "\n===================\n")
    return summary

def answer_question(client, question: str, chat_history: list, initial_context: list) -> str:
    # Compose the full message history: initial context + chat history + new question
    messages = initial_context.copy()
    for turn in chat_history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=DEEPSEEK_MODEL,
        temperature=0.3,
        max_tokens=5000,
    )
    answer = chat_completion.choices[0].message.content.strip()
    # Remove <think>...</think> section if present (multiline)
    answer = re.sub(r'<think>[\s\S]*?</think>', '', answer, flags=re.DOTALL).strip()
    return answer

# ========== MAIN CHATBOT LOGIC ==========
def main():
    print("YouTube Video Summarizer & Chatbot (DeepSeek)")
    url = input("Enter YouTube video URL: ").strip()
    try:
        video_id = extract_video_id(url)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    transcript_dict = get_transcript(video_id)
    client = create_groq_client()
    summary = summarize_transcript(client, transcript_dict['transcript'])
    chat_history = []
    # Prepare initial context (system and user messages)
    initial_context = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
        {"role": "user", "content": f"Here is a summary of the video: {summary}\n\nHere is the full transcript (for reference):\n{transcript_dict['transcript']}"}
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
            summary = summarize_transcript(client, transcript_dict['transcript'])
            chat_history = []
            initial_context = [
                {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
                {"role": "user", "content": f"Here is a summary of the video: {summary}\n\nHere is the full transcript (for reference):\n{transcript_dict['transcript']}"}
            ]
            print("\nYou can now ask questions about the new video. Type 'exit' to quit, or enter a new YouTube URL to start over.")
            continue
        answer = answer_question(client, question, chat_history, initial_context)
        print("\nAnswer:", answer)
        chat_history.append({"question": question, "answer": answer})

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(pathname)s:%(lineno)d %(message)s',
    )
    main()
