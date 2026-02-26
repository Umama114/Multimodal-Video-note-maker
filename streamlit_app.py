import streamlit as st
import os
import io
from markdown_pdf import MarkdownPdf, Section
from main import app

def convert_md_to_pdf(markdown_text):
    
    pdf = MarkdownPdf(toc_level=2)
    
    pdf.add_section(Section(markdown_text))
    
    pdf_buffer = io.BytesIO()
    pdf.save_bytes(pdf_buffer)
    return pdf_buffer.getvalue()

st.set_page_config(
    page_title="AI Video Note Taker",
    page_icon="🎓",
    layout="wide"
)

st.title("Multimodal Study Guide Generator")
st.markdown("Enter a YouTube URL or local video address to get AI-generated visual logs, transcripts, and structured notes.")

with st.sidebar:
    st.header("Settings")
    on = st.toggle("activate visual description")
    if on:
        st.write("visual description is on")
    st.info("This will add visual description to the notes but it will take longer as the AI has to watch the video then generate visual description")
    
input_type = st.radio("Select Input Source:", ["YouTube Link", "Local Video Upload"], horizontal=True)

url = None

if input_type == "YouTube Link":
    url = st.text_input("YouTube URL:", placeholder="https://www.youtube.com/...")
else:
    uploaded_file = st.file_uploader(
    "Upload a video file", 
    type=["mp4", "mov", "avi"], 
    max_upload_size=2048
    )
    if uploaded_file is not None:
        os.makedirs("downloads", exist_ok=True)
        url = os.path.join("downloads", uploaded_file.name)
        with open(url, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")

process_button = st.button("🚀 Generate Notes")

if process_button:
    if not url:
        st.error("Please provide a valid YouTube URL.")
    else:
        st.video(url)
        
        with st.status("AI is processing...", expanded=True) as status:
            st.write("Loading input...")
            initial_state = {"user_input": url,"use_vision": on}
            
            final_result = app.invoke(initial_state)
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        tab1, tab2, tab3 = st.tabs(["Final Notes", "Transcript", "Visual Log"])

        with tab1:
            notes_content = final_result.get("final_notes", "No notes generated.")
            st.markdown(notes_content)
    
            pdf_data = convert_md_to_pdf(notes_content)
    
            st.download_button(
             label="Download Study Guide (PDF)",
             data=pdf_data,
             file_name="study_guide.pdf",
             mime="application/pdf"
             )

        with tab2:
            with st.expander("Show Full Transcript"):
                st.write(final_result.get("transcript", "No transcript available."))

        with tab3:
            st.info("Key visual transitions and text detected by Gemini:")
            st.write(final_result.get("visual_log", "No visual log generated."))

else:
    st.info("select YouTube Link or Local Video Upload and past the same to begin")