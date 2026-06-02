from google import genai
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional


# ─── HMIS JSON Schema — Gemini fills values directly into this structure ───
HMIS_JSON_SCHEMA = {
    "visitId": "",
    "patientId": "",
    "facilityId": "",
    "healthProfessionalId": "",
    "appointmentId": "",
    "visitTime": "",
    "doctorPrivateNote": "",
    "isConsultationEnded": False,
    "linkWithAbha": False,
    "additionalVitals": [
        {
            "value": "",
            "vitalId": ""
        }
    ],
    "medications": [
        {
            "prescribedMedicine": "",
            "dosage": 1,
            "frequency": "1-0-0",
            "timing": "Any Time",
            "duration": 1,
            "durationPeriod": 1,
            "medicineType": 1,
            "reason": "",
            "medicationId": "",
            "isInternal": False,
            "scheduleDays": [],
            "medicationSchedule": 1,
            "unitType": 897,
            "additionalData": {
                "medicineComposition": ""
            },
            "dosePhases": [
                {
                    "order": 1,
                    "dosage": 1,
                    "duration": 1,
                    "durationPeriod": 1,
                    "timing": "Any Time",
                    "frequency": "1-0-0"
                }
            ]
        }
    ],
    "cheifComplaints": [
        {
            "id": "",
            "value": "",
            "recordedDate": "",
            "additionalData": {
                "medicalConceptId": ""
            }
        }
    ],
    "diagnosis": [
        {
            "id": "",
            "value": "",
            "additionalData": {
                "medicalConceptId": ""
            }
        }
    ],
    "diagnosticInvestigations": [
        {
            "id": "",
            "value": "",
            "instructions": "",
            "recordedDate": ""
        }
    ],
    "labInvestigations": [
        {
            "id": "",
            "value": "",
            "instructions": "",
            "recordedDate": ""
        }
    ],
    "advices": [
        {
            "value": "",
            "id": "",
            "additionalData": {}
        }
    ],
    "medicalAdvices": [
        {
            "value": "",
            "id": ""
        }
    ],
    "treatment": [],
    "extendedProps": [],
    "eyeExamination": {}
}


class CasesheetExtractor:
    """Extract structured HMIS-compatible JSON from transcripts using Gemini AI."""

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
        Extract structured HMIS JSON from a transcript.

        Args:
            transcript: Full English transcript text
            conversation: Optional list of conversation turns with speaker and text
            doctor_name: Extracted doctor name
            patient_name: Extracted patient name

        Returns:
            Dictionary matching the HMIS JSON schema — ready to pass to HMIS API
        """
        if not self.client:
            print("⚠️ No Gemini API key available. Returning empty HMIS JSON.")
            return self._get_empty_hmis()

        if not transcript or not transcript.strip():
            print("⚠️ Empty transcript. Returning empty HMIS JSON.")
            return self._get_empty_hmis()

        # Prepare context from conversation if available
        conversation_text = self._prepare_conversation_context(conversation)

        # Create extraction prompt
        prompt = self._create_extraction_prompt(transcript, conversation_text, doctor_name, patient_name)

        # Try models in fallback order (strictly free-tier models)
        models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

        for model_name in models_to_try:
            # Retry up to 2 times per model (for rate limits)
            for attempt in range(2):
                try:
                    print(f"🔍 Extracting HMIS JSON with {model_name}" + (f" (retry {attempt})..." if attempt > 0 else "..."))
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
                            print(f"✅ HMIS extraction successful!")
                            return extracted
                        else:
                            print(f"⚠️ JSON parse failed for {model_name}, trying next...")
                            break  # Don't retry same model for parse errors

                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        # Rate limit — wait and retry
                        wait_time = 30 if attempt == 0 else 60
                        print(f"⏳ Rate limited on {model_name}. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"⚠️ Extraction error ({model_name}): {e}. Trying fallback...")
                        break  # Move to next model

        print("⚠️ Could not extract HMIS JSON. Returning empty fields.")
        return self._get_empty_hmis()

    def _create_extraction_prompt(self, transcript: str, conversation_text: str,
                                 doctor_name: str, patient_name: str) -> str:
        """Create the Gemini prompt for direct HMIS JSON extraction."""

        now = datetime.now().isoformat()
        schema_json = json.dumps(HMIS_JSON_SCHEMA, indent=2)

        prompt = f"""You are a medical transcription assistant. Given a doctor-patient conversation transcript, extract all medical information and output it DIRECTLY in the HMIS JSON format below.

CONVERSATION CONTEXT:
Doctor: {doctor_name}
Patient: {patient_name}
Current DateTime: {now}

CONVERSATION TRANSCRIPT:
{transcript}

{f'DIARIZED CONVERSATION (for reference):{conversation_text}' if conversation_text else ''}

TARGET JSON SCHEMA (fill values based on transcript):
{schema_json}

CRITICAL RULES — FOLLOW EXACTLY:

1. OUTPUT ONLY valid JSON matching the schema above. Keys must stay EXACTLY the same. Only values change.

2. ALWAYS LEAVE THESE BLANK (HMIS assigns them):
   - "visitId": ""
   - "patientId": ""
   - "facilityId": ""
   - "healthProfessionalId": ""
   - "appointmentId": ""
   - All "id" fields inside arrays: ""
   - All "medicationId" fields: ""
   - All "medicalConceptId" fields: ""

3. SET THESE AUTOMATICALLY:
   - "visitTime": "{now}"
   - "recordedDate": "{now}" (for all items that have this field)
   - "isConsultationEnded": false
   - "linkWithAbha": false

4. MEDICATIONS — For each medicine mentioned:
   - "prescribedMedicine": exact medicine name from transcript
   - "dosage": number of units (default 1)
   - "frequency": MUST be in format like "1-1-1" (morning-afternoon-night)
     * "once a day" / "once daily" → "1-0-0"
     * "twice a day" → "1-0-1"
     * "three times a day" / "thrice daily" → "1-1-1"
     * "at night" / "bedtime" → "0-0-1"
     * "morning only" → "1-0-0"
   - "timing": use one of: "After Meal", "Before Meal", "Any Time"
   - "duration": number value (default 1)
   - "durationPeriod": 1=days, 2=weeks, 3=months
   - "reason": why this medicine (from transcript)
   - "medicineType": always 1
   - "unitType": always 897
   - "dosePhases": copy the same dosage/duration/frequency/timing into dosePhases array with order:1

5. CHIEF COMPLAINTS (cheifComplaints) — Map patient symptoms here:
   - "value": symptom name (e.g., "Fever", "Cough and Cold", "Stomach Pain")

6. ADDITIONAL VITALS — For any vital signs mentioned:
   - "vitalId": must be one of: body_temperature, systolic_bp, diastolic_bp, heart_rate, respiratory_rate, oxygen_saturation_spo2, body_height, body_weight, body_mass_index, randomBloodGlucose
   - "value": the numeric value as string

7. DIAGNOSIS — Medical diagnoses/conditions identified:
   - "value": diagnosis name

8. LAB INVESTIGATIONS (labInvestigations) — Blood tests, lab tests ordered:
   - "value": test name
   - "instructions": any special instructions

9. DIAGNOSTIC INVESTIGATIONS (diagnosticInvestigations) — X-rays, scans, imaging:
   - "value": investigation name
   - "instructions": any special instructions

10. ADVICES — Doctor's advice to patient:
    - "value": advice text

11. MEDICAL ADVICES (medicalAdvices) — Clinical examination findings/notes:
    - "value": finding or note text

12. TREATMENT — General treatment notes as array of strings

13. "doctorPrivateNote" — Any extra doctor notes/observations

14. If a section has NO data from the transcript, use EMPTY ARRAY [].

15. Do NOT invent or hallucinate information not in the transcript.

16. Translate non-English content to English for values.
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

    def _repair_json(self, json_str: str) -> str:
        """Fix common Gemini JSON issues: trailing commas, comments, etc."""
        # Remove single-line comments (// ...)
        json_str = re.sub(r'//.*?(?=\n|$)', '', json_str)
        # Remove multi-line comments (/* ... */)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(?<={|,)\s*(\w+)\s*:', r' "\1":', json_str)
        return json_str

    def _parse_extraction_response(self, response_text: str) -> Optional[Dict]:
        """Parse the JSON response from Gemini and ensure all HMIS keys exist."""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                print("⚠️ No JSON object found in response.")
                return None

            json_str = json_match.group(0)

            # Try parsing directly first
            data = None
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try repairing common issues
                print("🔧 Attempting JSON repair...")
                repaired = self._repair_json(json_str)
                try:
                    data = json.loads(repaired)
                    print("✅ JSON repair successful!")
                except json.JSONDecodeError as e2:
                    print(f"⚠️ JSON repair also failed: {e2}")
                    return None

            if data is None:
                return None

            # Ensure all required HMIS keys exist with defaults
            result = self._get_empty_hmis()
            
            # Merge extracted data into the result
            for key in result:
                if key in data:
                    result[key] = data[key]

            # Force context IDs to always be blank (HMIS assigns)
            result["visitId"] = ""
            result["patientId"] = ""
            result["facilityId"] = ""
            result["healthProfessionalId"] = ""
            result["appointmentId"] = ""

            return result

        except Exception as e:
            print(f"⚠️ Error processing extraction response: {e}")
            return None

    def _get_empty_hmis(self) -> Dict:
        """Return an empty HMIS JSON with all required keys."""
        return {
            "visitId": "",
            "patientId": "",
            "facilityId": "",
            "healthProfessionalId": "",
            "appointmentId": "",
            "visitTime": datetime.now().isoformat(),
            "doctorPrivateNote": "",
            "isConsultationEnded": False,
            "linkWithAbha": False,
            "additionalVitals": [],
            "medications": [],
            "cheifComplaints": [],
            "diagnosis": [],
            "diagnosticInvestigations": [],
            "labInvestigations": [],
            "advices": [],
            "medicalAdvices": [],
            "treatment": [],
            "extendedProps": [],
            "eyeExamination": {}
        }
