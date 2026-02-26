# 🎬 Multimodal AI Study Note Generator

An intelligent pipeline built with **LangGraph**, **Gemini 2.5**, and **Llama 3.3** that "watches" and "listens" to videos to generate structured study guides.

## ✨ Features
- **Visual Analysis:** Uses Gemini to track text on screen and scene changes.
- **Audio Transcription:** High-speed transcription via Groq (Whisper-v3).
- **Agentic Workflow:** Built with LangGraph for conditional routing (Skip vision for speed!).
- **PDF Export:** Download notes directly as print-ready PDFs.
- **Streamlit UI:** A clean, interactive dashboard for easy use.

## 🛠️ Tech Stack
- **Orchestration:** LangGraph
- **LLMs:** Gemini (Vision), Llama 3.3 (Synthesis)
- **Audio:** Groq/Whisper-v3 & FFmpeg
- **Frontend:** Streamlit

## 🚀 How to Run the Program

1. **Clone the Repository**
   Open your terminal and run:
   git clone https://github.com/Umama114/Multimodal-Video-note-maker.git
   cd Multimodal-Video-note-maker

2. **Create a Virtual Environment**
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate

3. **Install Dependencies**
   pip install -r requirements.txt

4. **Environment Setup**
   ## Create a .env file in the root directory and add your keys:
   GOOGLE_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   *Note: Ensure there are no spaces around the = sign*

5. **Run the App**
   streamlit run streamlit_app.py