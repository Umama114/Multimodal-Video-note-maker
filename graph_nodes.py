import os
import re
import requests
import streamlit as st
import subprocess
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict, Optional
from google import genai
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class AgentState(TypedDict):
    user_input: str
    video_path: str
    audio_path: str
    transcript: str
    use_vision: bool
    visual_log: str
    final_notes: str

def input_loader_node(state: AgentState):
    user_input = state["user_input"]
    download_folder = "downloads"
    os.makedirs(download_folder, exist_ok=True)

    if os.path.exists(user_input):
        print(f"Local video detected: {user_input}")
        return {"video_path": user_input}

    if "youtube.com" in user_input or "youtu.be" in user_input:
        print("Routing request through RapidAPI Gateway...")
        
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", user_input)
        if not match:
            raise ValueError("Could not extract a valid YouTube Video ID from the URL.")
        video_id = match.group(1)
        
        url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
        querystring = {
            "videoId": video_id,
            "urlAccess": "normal",
            "videos": "auto",
            "audios": "auto"
        }
        
        headers = {
            "x-rapidapi-key": st.secrets["RAPIDAPI_KEY"],
            "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com"
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status() 
            data = response.json()
            
            items = data.get("videos", {}).get("items", [])
            if not items:
                raise ValueError("No downloadable video formats returned by the API.")
                
            download_url = items[0].get("url")
            
            raw_title = data.get("title", "downloaded_video")
            safe_title = "".join(c if c.isalnum() else "_" for c in raw_title)
            video_path = os.path.join(download_folder, f"{safe_title}.mp4")
            
            print(f"Downloading file: {safe_title}...")
            with requests.get(download_url, stream=True) as video_stream:
                video_stream.raise_for_status()
                with open(video_path, 'wb') as f:
                    for chunk in video_stream.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print("✅ Video successfully retrieved!")
            return {"video_path": video_path}
            
        except Exception as e:
            print(f"❌ RapidAPI extraction failed: {e}")
            raise e

    raise ValueError("Invalid URL input.")

def extract_audio_node(state: AgentState):
    video_path = state["video_path"]
    audio_path = os.path.splitext(video_path)[0] + ".mp3"

    print(f"Extracting audio to {audio_path}...")
    
    command = [
        "ffmpeg", "-i", video_path, 
        "-vn", "-acodec", "libmp3lame", "-y", audio_path
    ]
    
    subprocess.run(command, check=True, capture_output=True)
    return {"audio_path": audio_path}

def transcribe_node(state: AgentState):
    audio_path = state["audio_path"]
    
    print("Transcribing audio with Groq (Whisper-v3)...")
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3",
            response_format="text",
        )
    
    return {"transcript": transcription}

def vision_router(state: AgentState):
    if state.get("use_vision"):
        return "visual"
    else:
        return "note"

def visual_description_node(state: AgentState):
    video_path = state["video_path"] 
    print(f"👁️ Gemini is watching: {os.path.basename(video_path)}...")

    # 1. Upload
    video_file = gemini_client.files.upload(file=video_path)
    
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = gemini_client.files.get(name=video_file.name)

    print("\n✅ Processing complete. Attempting generation...")

    visual_log = ""
    for attempt in range(3):
        try:
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=[
                    video_file, 
                    "Provide a detailed visual log of this video. Note text on screen and key transitions."
                ]
            )
            visual_log = response.text
            break
            
        except Exception as e:
            if "429" in str(e):
                print(f"\nRate limit hit. I'll wait 60 seconds and try again (Attempt {attempt+1}/3)...")
                time.sleep(65)
            else:
                print(f"\nUnexpected error: {e}")
                visual_log = "Visual analysis failed due to technical error."
                break
    
    gemini_client.files.delete(name=video_file.name)
    
    return {"visual_log": visual_log}

def notes_generator_node(state: AgentState):
    transcript = state.get("transcript", "")
    visual_log = state.get("visual_log", "")
    
    print("Synthesizing final study notes...")

    if state.get("use_vision"):
        prompt = f"""
        You are a professional editor. Combine the following TRANSCRIPT and VISUAL LOG 
        into a high-quality Markdown study notes.

        TRANSCRIPT:
        {transcript}
    
        VISUAL LOG:
        {visual_log}
    
        Format it with a Title, Summary, Notes (contain subheadings if any and all the information from the transcription and relevant information from visual_log ) and 'Visual & Audio Highlights' section.
        """
    else:
        prompt = f"""
        You are a professional editor. Use the following TRANSCRIPT
        to make a high-quality Markdown study guide.
    
        TRANSCRIPT:
        {transcript}
    
        Format it with a Title, Summary, Notes (contain subheadings if any and all the information from the transcription ) and 'Audio Highlights' section.
        """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    return {"final_notes": response.choices[0].message.content}