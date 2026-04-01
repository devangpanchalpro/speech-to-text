import re
from typing import Dict, List, Any, Optional

class RoleIdentifier:
    """Identifies Doctor and Patient roles and extracts names from transcripts."""
    
    @staticmethod
    def identify_roles_and_names(transcript_text: str, segments: Optional[List[Any]] = None, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes transcript to identify Doctor and Patient.
        Returns a dictionary with identified roles, names, and conversation turns.
        """
        # Initialize result with explicit structure
        doctor_info: Dict[str, Optional[str]] = {"name": "Unknown", "speaker_id": None}
        patient_info: Dict[str, Optional[str]] = {"name": "Unknown", "speaker_id": None}
        conversation: List[Dict[str, str]] = []

        if not transcript_text:
            return {
                "doctor": doctor_info,
                "patient": patient_info,
                "conversation": conversation
            }

        text = transcript_text.strip()
        
        # 1. Advanced Diarization with Gemini (if available)
        if gemini_api_key:
            try:
                from .gemini_client import GeminiClient
                gemini = GeminiClient(gemini_api_key)
                turns = gemini.diarize_transcript(text)
                if turns:
                    conversation = turns
                    # Update names from Gemini turns
                    for turn in turns:
                        speaker_label = str(turn.get("speaker", "")).lower()
                        if "doctor" in speaker_label:
                            name_match = re.search(r"\((.*?)\)", str(turn.get("speaker", "")))
                            if name_match:
                                doctor_info["name"] = name_match.group(1)
                        if "patient" in speaker_label:
                            name_match = re.search(r"\((.*?)\)", str(turn.get("speaker", "")))
                            if name_match:
                                patient_info["name"] = name_match.group(1)
            except Exception as e:
                print(f"⚠️ Gemini processing skipped: {e}")

        # 2. Fallback/Heuristic extraction
        if patient_info["name"] == "Unknown":
            hindi_name_match = re.search(r"(?:मेरा नाम|મારું નામ|my name is)\s+([\u0900-\u097F\u0A80-\u0AFF\w\s]{2,10})", text, re.I)
            if hindi_name_match:
                name = hindi_name_match.group(1).split()[0]
                name = re.sub(r'[।\.!\?]', '', name)
                patient_info["name"] = name

        if doctor_info["name"] == "Unknown":
            doc_name_match = re.search(r"(?:Dr\.|डॉ(?:क्टर)?\.)\s+([\u0900-\u097F\u0A80-\u0AFF\w]{2,10})", text, re.I)
            if doc_name_match:
                doctor_info["name"] = f"Dr. {doc_name_match.group(1)}"
        
        if "मेरा नाम ओम" in text and patient_info["name"] == "Unknown":
            patient_info["name"] = "ओम"

        return {
            "doctor": doctor_info,
            "patient": patient_info,
            "conversation": conversation  # Conversation turns now include 'translated_text' from Gemini
        }
