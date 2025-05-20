import os
import sys
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
import groq
import re

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

def get_cached_transcript(video_id: str) -> str | None:
    cache_file = CACHE_DIR / f"{video_id}.txt"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return f.read()
    return None

def cache_transcript(video_id: str, transcript: str):
    cache_file = CACHE_DIR / f"{video_id}.txt"
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(transcript)

def download_transcript(video_id: str) -> str:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = "\n".join(item["text"] for item in transcript_list)
        cache_transcript(video_id, transcript)
        return transcript
    except Exception as e:
        print(f"Could not get transcript: {str(e)}")
        sys.exit(1)

def get_transcript(video_id: str) -> str:
    cached = get_cached_transcript(video_id)
    if cached:
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
        max_tokens=800,
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
    transcript = get_transcript(video_id)
    client = create_groq_client()
    summary = summarize_transcript(client, transcript)
    chat_history = []
    # Prepare initial context (system and user messages)
    initial_context = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
        {"role": "user", "content": f"Here is a summary of the video: {summary}\n\nHere is the full transcript (for reference):\n{transcript}"}
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
            transcript = get_transcript(video_id)
            summary = summarize_transcript(client, transcript)
            chat_history = []
            initial_context = [
                {"role": "system", "content": "You are a helpful assistant that answers questions about a YouTube video based on its transcript and summary."},
                {"role": "user", "content": f"Here is a summary of the video: {summary}\n\nHere is the full transcript (for reference):\n{transcript}"}
            ]
            print("\nYou can now ask questions about the new video. Type 'exit' to quit, or enter a new YouTube URL to start over.")
            continue
        answer = answer_question(client, question, chat_history, initial_context)
        print("\nAnswer:", answer)
        chat_history.append({"question": question, "answer": answer})

if __name__ == "__main__":
    main()
