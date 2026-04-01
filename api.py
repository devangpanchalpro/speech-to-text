import os
import shutil
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Security, UploadFile, File, Form
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv

from main import process_audio_pipeline

# Load environment variables
load_dotenv()

# App setup
app = FastAPI(
    title="CoreInventory Audio Processor API",
    description="API for converting audio to text, diarization, and EMR casesheet extraction from doctor-patient audio conversations.",
    version="1.0.0",
)

# Authentication Setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    """Validate the API key from the request header."""
    expected_api_key = os.getenv("APP_API_KEY")
    
    if not expected_api_key:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: APP_API_KEY is not set."
        )

    if api_key_header == expected_api_key:
        return api_key_header
    
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API Key"
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to CoreInventory Audio Processor API. Use /docs to view the API documentation."}

@app.post("/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    sarvam_api_key: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    language_code: Optional[str] = Form("unknown"),
    api_key: str = Depends(get_api_key)
):
    """
    Process an audio file and return structured EMR JSON.
    
    Uploads a doctor-patient audio conversation, transcribes it, identifies speakers,
    translates to English, and extracts a full structured EMR casesheet.

    - **file**: The audio file to process (mp3, wav, mp4, ogg, m4a, flac)
    - **sarvam_api_key**: (Optional) Sarvam AI API Key. Falls back to SARVAM_API_KEY env var.
    - **gemini_api_key**: (Optional) Gemini API Key. Falls back to GEMINI_API_KEY env var.
      **Required** for EMR extraction, translation, and advanced diarization.
    - **language_code**: (Optional) Language code for transcription. Defaults to 'unknown' (auto-detect).
    
    Returns the full structured EMR JSON populated from the audio content.
    """
    
    # Validation against empty file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename uploaded")
        
    # Resolve API keys (use passed in form variables, fallback to env variables)
    resolved_sarvam_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
    resolved_gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
    
    if not resolved_sarvam_key:
        raise HTTPException(
            status_code=400, 
            detail="Sarvam API Key is required (pass via form or set SARVAM_API_KEY in environment)."
        )

    if not resolved_gemini_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini API Key is required for EMR extraction (pass via form or set GEMINI_API_KEY in environment)."
        )

    # Save the file temporarily
    input_dir = "audio_files"
    os.makedirs(input_dir, exist_ok=True)
    file_path = os.path.join(input_dir, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
        
    # Run the pipeline
    try:
        result = process_audio_pipeline(
            audio_filename=file_path,
            api_key=resolved_sarvam_key,
            gemini_api_key=resolved_gemini_key,
            language_code=language_code
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Audio processing pipeline failed. Check backend logs.")
        
        # Build the response with the EMR casesheet as the primary body
        casesheet = result.get("casesheet", {})
        
        # Ensure all EMR fields exist even if extraction missed some
        emr_response = {
            # ─── EMR Casesheet (Primary Output) ───
            "advices": casesheet.get("advices", []),
            "diagnosis": casesheet.get("diagnosis", []),
            "followup": casesheet.get("followup", {"date": "", "notes": ""}),
            "PrescribedTests": casesheet.get("PrescribedTests", []),
            "DiagnosticResults": casesheet.get("DiagnosticResults", []),
            "medicalHistory": casesheet.get("medicalHistory", {
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
            }),
            "examinations": casesheet.get("examinations", []),
            "bodyVitalSigns": casesheet.get("bodyVitalSigns", []),
            "medications": casesheet.get("medications", []),
            "symptoms": casesheet.get("symptoms", []),
            "prescriptionNotes": casesheet.get("prescriptionNotes", ""),
            
            # ─── Audio & Transcript Context ───
            "_metadata": {
                "original_file": result.get("metadata", {}).get("original_file", ""),
                "detected_language": result.get("metadata", {}).get("detected_language", ""),
                "doctor_name": result.get("identification", {}).get("doctor", {}).get("name", "Unknown"),
                "patient_name": result.get("identification", {}).get("patient", {}).get("name", "Unknown"),
            },
            "_transcript": result.get("transcript", ""),
            "_transcript_english": result.get("transcript_english", ""),
            "_conversation": result.get("identification", {}).get("conversation", []),
        }
        
        return emr_response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
