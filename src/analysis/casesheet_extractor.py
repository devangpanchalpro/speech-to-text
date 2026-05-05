from google import genai
import json
import re
from typing import List, Dict, Optional


# The dynamic EMR JSON schema used as the target output format
EMR_JSON_SCHEMA = {
    "advices": [
        {
            "text": ""
        }
    ],
    "diagnosis": [
        {
            "name": "",
            "since": {
                "value": 0,
                "unit": ""
            },
            "status": "",
            "severity": "",
            "laterality": "",
            "details": ""
        }
    ],
    "followup": {
        "date": "",
        "notes": ""
    },
    "PrescribedTests": [
        {
            "name": "",
            "remark": ""
        }
    ],
    "DiagnosticResults": [
        {
            "name": "",
            "unit": "",
            "value": "",
            "interpretation": "",
            "remark": ""
        }
    ],
    "medicalHistory": {
        "patientHistory": {
            "patientMedicalConditions": [
                {
                    "name": "",
                    "since": "",
                    "status": "",
                    "notes": ""
                }
            ],
            "currentMedications": [
                {
                    "name": "",
                    "since": "",
                    "notes": "",
                    "status": ""
                }
            ],
            "familyHistory": [
                {
                    "name": "",
                    "who": "",
                    "since": "",
                    "notes": "",
                    "status": ""
                }
            ],
            "lifestyleHabits": [
                {
                    "name": "",
                    "since": "",
                    "status": ""
                }
            ],
            "foodOtherAllergy": [
                {
                    "name": "",
                    "since": "",
                    "status": ""
                }
            ],
            "pastProcedures": [
                {
                    "name": "",
                    "status": ""
                }
            ],
            "recentTravelHistory": [
                {
                    "name": "",
                    "status": ""
                }
            ],
            "vaccinationHistory": [
                {
                    "name": "",
                    "status": ""
                }
            ],
            "drugAllergy": [
                {
                    "name": "",
                    "since": "",
                    "status": ""
                }
            ]
        }
    },
    "examinations": [
        {
            "name": "",
            "notes": ""
        }
    ],
    "bodyVitalSigns": [
        {
            "name": "",
            "value": {
                "qt": "",
                "unit": ""
            }
        }
    ],
    "medications": [
        {
            "name": "",
            "duration": {
                "value": 0,
                "unit": ""
            },
            "frequency": "",
            "timing": "",
            "dose": {
                "value": 0,
                "unit": ""
            },
            "instruction": ""
        }
    ],
    "symptoms": [
        {
            "name": "",
            "since": {
                "value": 0,
                "unit": ""
            },
            "severity": "",
            "laterality": "",
            "finding_status": "",
            "details": ""
        }
    ],
    "prescriptionNotes": "",
    "confidence_score": 0
}

class CasesheetExtractor:
    """Extract structured EMR casesheet from transcripts using Gemini AI."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the extractor with Gemini API client.

        Args:
            api_key: Gemini API key (optional if using environment variable)
        """
        self.api_key = api_key
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def extract_casesheet(self, transcript: str, conversation: Optional[List[Dict]] = None,
                         doctor_name: str = "Unknown", patient_name: str = "Unknown") -> Dict:
        """
        Extract structured EMR casesheet from a transcript.

        Args:
            transcript: Full English transcript text
            conversation: Optional list of conversation turns with speaker and text
            doctor_name: Extracted doctor name
            patient_name: Extracted patient name

        Returns:
            Dictionary matching the full EMR JSON schema
        """
        if not self.client:
            print("⚠️ No Gemini API key available. Returning empty casesheet.")
            return self._get_empty_casesheet()

        if not transcript or not transcript.strip():
            print("⚠️ Empty transcript. Returning empty casesheet.")
            return self._get_empty_casesheet()

        # Prepare context from conversation if available
        conversation_text = self._prepare_conversation_context(conversation)

        # Create extraction prompt
        prompt = self._create_extraction_prompt(transcript, conversation_text, doctor_name, patient_name)

        # Try models in fallback order
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        for model_name in models_to_try:
            try:
                print(f"🔍 Extracting casesheet with {model_name}...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json'
                    }
                )

                if response and response.text:
                    extracted = self._parse_extraction_response(response.text)
                    if extracted:
                        print(f"✅ Casesheet extraction successful!")
                        return extracted

            except Exception as e:
                print(f"⚠️ Extraction error ({model_name}): {e}. Trying fallback...")
                continue

        print("⚠️ Could not extract casesheet. Returning empty fields.")
        return self._get_empty_casesheet()

    def _create_extraction_prompt(self, transcript: str, conversation_text: str,
                                 doctor_name: str, patient_name: str) -> str:
        """Create the Gemini prompt for structured EMR extraction."""

        schema_json = json.dumps(EMR_JSON_SCHEMA, indent=2)

        prompt = f"""You are a medical transcribe assistant. Given a transcription of a conversation between a patient and a doctor, convert the raw text into structured JSON suitable for EMR in the schema below. Segment medical entities as cleanly as possible. Your output should only contain valid JSON and no extra text. Translate the content to English if the input is in another language. Be as verbatim as possible while structuring the information.

CONVERSATION CONTEXT:
Doctor: {doctor_name}
Patient: {patient_name}

CONVERSATION TRANSCRIPT:
{transcript}

{f'DIARIZED CONVERSATION (for reference):{conversation_text}' if conversation_text else ''}

JSON schema:
{schema_json}

IMPORTANT RULES:
- Output ONLY valid JSON matching the schema above.
- Only include array items that are actually mentioned in the transcript. If a section has no data, use an empty array [].
- For "diagnosis", only include NEW diagnoses from the current visit, not past history.
- For "symptoms.finding_status", use one of: "Present", "Absent", "Unknown".
- For "diagnosis.status", use one of: "Suspected", "Confirmed", "Ruled out".
- For "severity" fields, use one of: "Mild", "Moderate", "Severe" or leave empty.
- For "laterality" fields, use one of: "Left", "Right", "Right and left" or leave empty.
- For "DiagnosticResults.interpretation", use one of: "Critically high", "Very high", "High", "Borderline high", "Normal", "Borderline low", "Low", "Very low", "Critically low", "Abnormal".
- For status fields in medical history, use "Active" or "Inactive".
- Use "prescriptionNotes" for any information not captured in above fields.
- For "confidence_score", provide an integer from 0 to 100 representing how confident you are in the accuracy and completeness of this EMR extraction based on the transcript's clarity.
- Do NOT invent or hallucinate information not present in the transcript.
"""
        return prompt

    def _prepare_conversation_context(self, conversation: Optional[List[Dict]]) -> str:
        """Prepare formatted conversation text for context."""
        if not conversation:
            return ""

        formatted = ""
        for turn in conversation:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            formatted += f"{speaker}: {text}\n"

        return formatted[:2000]

    def _parse_extraction_response(self, response_text: str) -> Optional[Dict]:
        """Parse the JSON response from Gemini and validate against schema."""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # Ensure all top-level keys exist with correct defaults
                result = self._get_empty_casesheet()
                
                # Merge extracted data into the result
                for key in result:
                    if key in data:
                        result[key] = data[key]

                return result
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            print(f"⚠️ Error processing extraction response: {e}")
            return None

    def _get_empty_casesheet(self) -> Dict:
        """Return an empty casesheet matching the full EMR JSON schema."""
        return {
            "advices": [],
            "diagnosis": [],
            "followup": {
                "date": "",
                "notes": ""
            },
            "PrescribedTests": [],
            "DiagnosticResults": [],
            "medicalHistory": {
                "patientHistory": {
                    "patientMedicalConditions": [],
                    "currentMedications": [],
                    "familyHistory": [],
                    "lifestyleHabits": [],
                    "foodOtherAllergy": [],
                    "pastProcedures": [],
                    "recentTravelHistory": [],
                    "vaccinationHistory": [],
                    "drugAllergy": []
                }
            },
            "examinations": [],
            "bodyVitalSigns": [],
            "medications": [],
            "symptoms": [],
            "prescriptionNotes": "",
            "confidence_score": 0
        }
