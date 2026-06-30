import os
import tempfile
import streamlit as st
import subprocess
import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget
from groq import Groq
from dotenv import load_dotenv
from typing import TypedDict, Optional
from google import genai
import time

load_dotenv()

client= Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        print(f"local video detected: {user_input}")
        return {"video_path": user_input}

    if "youtube.com" in user_input or "youtu.be" in user_input:
        print("downloading YouTube video using unfragmented stream routing...")
        
        ydl_opts = {
            'restrictfilenames': True,
            # We are dropping the complex rules. Just give us the best pre-combined file.
            'format': 'best', 
            'outtmpl': f'{download_folder}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'nocache_dir': True,
            'extractor_args': {
                'youtube': {
                    # Disguising as iOS or TV bypasses data-center IP blocks much better
                    'player_client': ['ios', 'tv', 'android', 'web'],
                    'player_js_version': 'actual'
                }
            }
        }

        try:
            ydl_opts["impersonate"] = ImpersonateTarget.from_str("chrome")
        except Exception as e:
            print(f"Impersonation fallback: {e}")

        cookie_path = None

        if "YOUTUBE_COOKIES" in st.secrets:
            print("Repairing cookie alignment (converting spaces back to strict tabs)...")
            raw_cookies = st.secrets["YOUTUBE_COOKIES"]
            reconstructed_lines = []
            
            for line in raw_cookies.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    reconstructed_lines.append(line)
                    continue
                
                columns = stripped.split()
                if len(columns) >= 7:
                    domain, domain_flag, path, secure_flag, expiry, name = columns[:6]
                    value = " ".join(columns[6:])
                    reconstructed_lines.append(f"{domain}\t{domain_flag}\t{path}\t{secure_flag}\t{expiry}\t{name}\t{value}")
                else:
                    reconstructed_lines.append(line)
                    
            fixed_cookie_data = "\n".join(reconstructed_lines)
            
            temp_cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
            temp_cookie_file.write(fixed_cookie_data)
            temp_cookie_file.close()
            
            cookie_path = temp_cookie_file.name
            ydl_opts['cookiefile'] = cookie_path
        else:
            print("⚠️ WARNING: YOUTUBE_COOKIES key missing from Streamlit secrets engine!")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(user_input, download=True)
            video_path = ydl.prepare_filename(info)

        if cookie_path and os.path.exists(cookie_path):
            os.remove(cookie_path)
            
        return {"video_path": video_path}

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