from google import genai
import json
import re
from typing import List, Dict, Optional

class GeminiClient:
    """Handles interaction with Gemini AI using the modern google-genai SDK."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def diarize_transcript(self, transcript_text: str) -> Optional[List[Dict[str, str]]]:
        """
        Uses Gemini to split a flat transcript into a Doctor-Patient conversation.
        Tries multiple models as fallback if one fails (e.g., due to quota).
        """
        if not self.client or not transcript_text:
            return None

        prompt = f"""
        The following is a flat transcript of a conversation between a Doctor and a Patient.
        It may be in Hindi, Gujarati, or English.
        
        TASK:
        1. Split this into a logical dialogue (turns).
        2. Identify the speaker for each turn: "Doctor" or "Patient".
        3. If names are mentioned (e.g., "My name is Om"), use the format "Patient (Om)" or "Doctor (Name)".
        4. For each turn, provide the text in the ORIGINAL language (text) AND translate it to English (translated_text).
        
        TRANSCRIPT:
        {transcript_text}
        
        OUTPUT FORMAT (Strict JSON Array):
        [
          {{"speaker": "Doctor", "text": "...", "translated_text": "..."}},
          {{"speaker": "Patient (Om)", "text": "...", "translated_text": "..."}}
        ]
        """
        
        # List of models to try in order of preference
        models_to_try = [
            "gemini-2.5-flash", 
            "gemini-2.0-flash", 
        ]
        
        for model_name in models_to_try:
            try:
                print(f"🤖 Attempting diarization with {model_name}...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json'
                    }
                )
                
                if response and response.text:
                    response_text = response.text
                    # Extract JSON array if model added markdown markers
                    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                    else:
                        return json.loads(response_text)
                
            except Exception as e:
                print(f"⚠️ Gemini Error ({model_name}): {e}. Trying fallback...")
                continue
        
        return None

    def translate_text(self, text: str, source_language: str = "auto", target_language: str = "English") -> str:
        """
        Translate text from source_language to target_language using Gemini.
        Returns the translated text (plain string) or original text on failure.
        """
        if not self.client or not text:
            return text

        prompt = f"""
Translate the following text from {source_language} to {target_language}. Respond with only the translated text.

Text:
{text}
"""
        # List of models to try
        models_to_try = [
            "gemini-2.5-flash", 
            "gemini-2.0-flash",
        ]

        for model_name in models_to_try:
            try:
                print(f"🌎 Translating with {model_name}...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"⚠️ Gemini Translation Error ({model_name}): {e}. Trying fallback...")
                continue
        
        return text
