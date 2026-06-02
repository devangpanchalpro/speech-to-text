import requests
import os
import time
import tempfile
from pydub import AudioSegment

try:
    import imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
except Exception as e:
    pass

class SarvamClient:
    """Client for Sarvam AI Speech-to-Text and Diarization."""
    
    API_URL = "https://api.sarvam.ai/speech-to-text"
    
    def __init__(self, api_key=None):
        self.api_key = (api_key or os.environ.get("SARVAM_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Sarvam API Key is required. Please provide a valid API key in settings.")
        
    def _transcribe_single_chunk(self, audio_file_path, language_code="unknown"):
        """
        Helper method to transcribe a single audio file (must be under 30 seconds).
        """
        headers = {
            "api-subscription-key": self.api_key,
        }
        
        with open(audio_file_path, "rb") as f:
            files = {"file": (os.path.basename(audio_file_path), f, "audio/mpeg")}
            data = {
                "model": "saaras:v3",
                "language_code": language_code
            }
            
            print(f"📡 Transcribing chunk {audio_file_path} (Language: {language_code})...")
            response = requests.post(self.API_URL, headers=headers, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Sarvam API Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()

    def transcribe(self, audio_file_path, language_code="unknown"):
        """
        Transcribes audio using Sarvam AI Saaras:v3 model.
        Automatically chunks audio if it exceeds 29 seconds.
        """
        try:
            audio = AudioSegment.from_file(audio_file_path)
            duration_secs = len(audio) / 1000.0
        except Exception as e:
            print(f"⚠️ Failed to inspect audio duration via pydub: {e}. Proceeding without chunking.")
            audio = None
            duration_secs = 0.0

        if audio is None or duration_secs <= 29.0:
            return self._transcribe_single_chunk(audio_file_path, language_code=language_code)

        print(f"ℹ️ Audio duration ({duration_secs:.2f}s) exceeds the 30-second limit. Chunking audio...")
        chunk_length_ms = 25 * 1000  # 25 seconds chunks
        min_last_chunk_ms = 5 * 1000  # 5 seconds minimum for the last chunk
        chunks = []
        
        total_ms = len(audio)
        start = 0
        while start < total_ms:
            end = start + chunk_length_ms
            if total_ms - end < min_last_chunk_ms:
                end = total_ms
            chunks.append(audio[start:end])
            start = end

        print(f"ℹ️ Split into {len(chunks)} chunks.")
        
        transcripts = []
        detected_languages = []
        
        for index, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_chunk_file:
                temp_chunk_path = temp_chunk_file.name
            
            try:
                # Export chunk to temporary file
                chunk.export(temp_chunk_path, format="mp3")
                
                # Keep language consistent if auto-detected
                current_lang = language_code
                if current_lang == "unknown" and detected_languages:
                    current_lang = detected_languages[0]
                
                print(f"🎙️ Processing chunk {index + 1}/{len(chunks)}...")
                chunk_res = self._transcribe_single_chunk(temp_chunk_path, language_code=current_lang)
                
                if chunk_res and 'transcript' in chunk_res:
                    transcripts.append(chunk_res['transcript'])
                    if 'language_code' in chunk_res:
                        detected_languages.append(chunk_res['language_code'])
                else:
                    print(f"⚠️ Chunk {index + 1} returned empty transcription.")
            except Exception as e:
                print(f"❌ Error transcribing chunk {index + 1}: {e}")
                raise e
            finally:
                if os.path.exists(temp_chunk_path):
                    try:
                        os.remove(temp_chunk_path)
                    except Exception as e:
                        print(f"⚠️ Could not remove temp chunk file: {e}")
                        
        if transcripts:
            combined_transcript = " ".join(t.strip() for t in transcripts if t.strip())
            final_lang = detected_languages[0] if detected_languages else language_code
            return {
                "transcript": combined_transcript,
                "language_code": final_lang
            }
        else:
            raise ValueError("All audio chunks failed to transcribe.")

    def transcribe_with_diarization(self, audio_file_path, language_code="en-IN"):
        """
        Placeholder for Batch API diarization if needed.
        Currently using the standard STT which might have segments.
        Note: Sarvam's high-accuracy diarization is usually in the Batch API.
        """
        # For now, we use the standard transcription and check if it has segments
        return self.transcribe(audio_file_path, language_code)

    def translate_text(self, text, source_language_code="hi-IN", target_language_code="en-IN"):
        """
        Translates text using Sarvam AI Translation API.
        Automatically chunks text if it exceeds 900 characters to bypass API constraints.
        """
        if not text or not text.strip():
            return text

        # Check if text length exceeds Sarvam's 1000 character limit
        if len(text) > 900:
            print(f"ℹ️ Input text length ({len(text)} chars) exceeds Sarvam's limit. Translating in chunks...")
            # Split text into chunks under 900 characters, splitting at spaces
            chunks = []
            current_chunk = []
            current_len = 0
            for word in text.split(" "):
                if current_len + len(word) + 1 > 800:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_len = len(word)
                else:
                    current_chunk.append(word)
                    current_len += len(word) + 1
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            translated_chunks = []
            for i, chunk in enumerate(chunks):
                print(f"🌎 Translating chunk {i+1}/{len(chunks)}...")
                # Recursively call translate_text for this smaller chunk
                translated_chunk = self.translate_text(chunk, source_language_code, target_language_code)
                translated_chunks.append(translated_chunk)
            
            return " ".join(translated_chunks)

        translate_url = "https://api.sarvam.ai/translate"
        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # 'unknown' is not a valid language code for Sarvam translation, use 'auto'
        if source_language_code == "unknown":
            source_language_code = "auto"
            
        data = {
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code
        }
        
        print(f"🌎 Translating with Sarvam API ({source_language_code} -> {target_language_code})...")
        try:
            response = requests.post(translate_url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return result.get("translated_text", text)
            else:
                print(f"❌ Sarvam Translate API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return text
        except Exception as e:
            print(f"⚠️ Sarvam Translate Error: {e}")
            return text

if __name__ == "__main__":
    # Quick test
    client = SarvamClient()
    # Replace with a real path for testing
    # result = client.transcribe("path/to/audio.mp3")
    # print(result)

