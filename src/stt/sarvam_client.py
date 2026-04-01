import requests
import os
import time

class SarvamClient:
    """Client for Sarvam AI Speech-to-Text and Diarization."""
    
    API_URL = "https://api.sarvam.ai/speech-to-text"
    
    def __init__(self, api_key=None):
        self.api_key = (api_key or os.environ.get("SARVAM_API_KEY", "")).strip()
        if not self.api_key:
            raise ValueError("Sarvam API Key is required. Please provide a valid API key in settings.")
        
    def transcribe(self, audio_file_path, language_code="unknown"):
        """
        Transcribes audio using Sarvam AI Saaras:v3 model.
        Returns the transcription result.
        NOTE: For automatic language detection, 'hi-IN' triggers 
        Sarvam's multilingual support for most Indian contexts.
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
            
            print(f"📡 Transcribing {audio_file_path} (Language: {language_code})...")
            response = requests.post(self.API_URL, headers=headers, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Sarvam API Error: {response.status_code}")
                print(f"Response: {response.text}")
                response.raise_for_status()

    def transcribe_with_diarization(self, audio_file_path, language_code="en-IN"):
        """
        Placeholder for Batch API diarization if needed.
        Currently using the standard STT which might have segments.
        Note: Sarvam's high-accuracy diarization is usually in the Batch API.
        """
        # For now, we use the standard transcription and check if it has segments
        return self.transcribe(audio_file_path, language_code)

if __name__ == "__main__":
    # Quick test
    client = SarvamClient()
    # Replace with a real path for testing
    # result = client.transcribe("path/to/audio.mp3")
    # print(result)
