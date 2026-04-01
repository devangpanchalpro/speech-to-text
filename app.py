import streamlit as st
import os
import json
from main import process_audio_pipeline

# Page Config
st.set_page_config(page_title="CoreInventory Audio Processor", page_icon="🎵", layout="wide")

# Title and Description
st.title("🎵 CoreInventory Audio Processor")
st.markdown("Upload a patient-doctor conversation to transcribe and identify speakers.")

# Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Sarvam AI API Key", value=os.environ.get("SARVAM_API_KEY", ""), type="password").strip()
    
    if not api_key:
        st.error("❌ Sarvam AI API key is required for transcription.")
    else:
        st.success("✅ Sarvam AI enabled for transcription!")
    
    # Pre-fill Gemini API key from environment if available
    default_gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_api_key = st.text_input("Gemini API Key (Required for English Translation)", value=default_gemini_key, type="password").strip()
    
    if not gemini_api_key:
        st.warning("⚠️ No Gemini API key. **English translation** and advanced speaker identification will be disabled.")
    else:
        st.success("✅ Gemini AI enabled for Translation & Diarization!")
    st.info("The Sarvam AI key is required for basic transcription.")
    
    st.divider()
    
    # Language Selection
    st.subheader("Transcription")
    language_options = {
        "Auto Detect (Recommended)": "unknown",
        "English (Indian)": "en-IN",
        "Hindi (Traditional)": "hi-IN",
        "Gujarati": "gu-IN",
        "Marathi": "mr-IN"
    }
    selected_lang_name = st.selectbox("Target Language", list(language_options.keys()), index=0)
    language_code = language_options[selected_lang_name]

# File Uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "ogg", "wav", "m4a", "flac"])

if uploaded_file is not None:
    # 1. Save the uploaded file to 'audio_files'
    os.makedirs("audio_files", exist_ok=True)
    file_path = os.path.join("audio_files", uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # 2. Process Button
    if st.button("🚀 Process Audio"):
        with st.status("Processing audio...", expanded=True) as status:
            st.write("🎵 Converting audio to MP3...")
            # Note: We can expand this to more st.write calls if we refactor main.py to accept a logger
            result = process_audio_pipeline(file_path, api_key=api_key, gemini_api_key=gemini_api_key, language_code=language_code)
            
            if result:
                status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                
                st.divider()
                st.header("📊 Results")
                
                # Metrics / Summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Language", result['metadata']['detected_language'])
                col2.metric("Doctor", result['identification']['doctor']['name'])
                col3.metric("Patient", result['identification']['patient']['name'])
                
                # Tabs for Transcript and JSON
                tabs = ["📝 Full Transcript", "💾 JSON Data"]
                if result['identification'].get('conversation'):
                    tabs.insert(0, "💬 Diarized Chat")

                tab_objs = st.tabs(tabs)

                # Handle dynamic tabs
                current_tab_idx = 0
                if result['identification'].get('conversation'):
                    with tab_objs[current_tab_idx]:
                        for turn in result['identification']['conversation']:
                            speaker = turn['speaker']
                            text = turn['text']
                            translated = turn.get('translated_text', '')

                            # Prefix with icon
                            icon = "👨‍⚕️" if "Doctor" in speaker else "🏥"

                            st.markdown(f"{icon} **{speaker}**")
                            st.write(f"**Original:** {text}")
                            if translated and translated != text:
                                st.info(f"**English:** {translated}")
                            st.divider()
                    current_tab_idx += 1
                
                # Full Transcript Tab
                with tab_objs[current_tab_idx]:
                    st.subheader("🌐 English Transcription")
                    st.text_area("Final English Text", result.get('transcript_english', ''), height=250)

                    st.divider()

                    st.subheader("📄 Original Transcript")
                    original_text = result.get('transcript', result.get('transcript_hindi', ''))
                    st.text_area("Source Language Text", original_text, height=150)
                current_tab_idx += 1

                # JSON Data Tab
                with tab_objs[current_tab_idx]:
                    st.json(result)
                    
                # Download link
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 Download JSON Result",
                    data=json_str,
                    file_name=f"result_{os.path.splitext(uploaded_file.name)[0]}.json",
                    mime="application/json"
                )
            else:
                status.update(label="❌ Processing Failed", state="error", expanded=True)
                st.error("❌ Processing failed. Please check the logs or your API keys.")

# Footer
st.divider()
st.markdown("Developed with ❤️ using Sarvam AI and Streamlit")
