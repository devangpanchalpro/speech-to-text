import os
import shutil
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Security, UploadFile, File, Form, Query
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv

from main import process_audio_pipeline
from src.database.db_manager import DatabaseManager

# Load environment variables
load_dotenv()

# ─── Resolve API keys once at startup from .env ───
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# App setup
app = FastAPI(
    title="CoreInventory Audio Processor API",
    description="API for converting audio to text, diarization, and HMIS-compatible JSON extraction from doctor-patient audio conversations. Data is persisted to PostgreSQL.",
    version="3.0.0",
)

# ─── Database initialization on startup ───
@app.on_event("startup")
def startup_db():
    """Initialize and re-sequence the database table on application startup."""
    try:
        db = DatabaseManager()
        db.initialize_db()
        db.resequence_records()
        db.close()
    except Exception as e:
        print(f"⚠️ Database startup warning: {e}")
        print("   The API will still work, but data won't be saved to PostgreSQL.")

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
    language_code: Optional[str] = Form("unknown"),
    api_key: str = Depends(get_api_key)
):
    """
    Process an audio file and return HMIS-compatible JSON.
    
    Uploads a doctor-patient audio conversation, transcribes it, identifies speakers,
    translates to English, extracts structured EMR data, and converts to HMIS format.

    - **file**: The audio file to process (mp3, wav, mp4, ogg, m4a, flac, opus)
    - **language_code**: (Optional) Language code for transcription. Defaults to 'unknown' (auto-detect).
    
    API keys (Sarvam, Gemini) are configured server-side via .env file.
    
    Returns HMIS-compatible JSON populated from the audio content.
    """
    
    # Validation against empty file
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename uploaded")
    
    # Validate server-side keys are configured
    if not SARVAM_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Server configuration error: SARVAM_API_KEY is not set in .env"
        )

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: GEMINI_API_KEY is not set in .env"
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
        
    # Run the pipeline with .env keys
    try:
        result = process_audio_pipeline(
            audio_filename=file_path,
            api_key=SARVAM_API_KEY,
            gemini_api_key=GEMINI_API_KEY,
            language_code=language_code
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Audio processing pipeline failed. Check backend logs.")
        
        # Build the response — HMIS format directly from pipeline
        hmis_data = result.get("hmis", {})
        
        # Add metadata for reference
        hmis_response = hmis_data
        hmis_response["_metadata"] = {
            "original_file": result.get("metadata", {}).get("original_file", ""),
            "detected_language": result.get("metadata", {}).get("detected_language", ""),
            "doctor_name": result.get("identification", {}).get("doctor", {}).get("name", "Unknown"),
            "patient_name": result.get("identification", {}).get("patient", {}).get("name", "Unknown"),
            "transcript": result.get("transcript", ""),
            "transcript_english": result.get("transcript_english", ""),
        }
        
        return hmis_response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")


@app.get("/records")
def get_records(
    limit: int = Query(50, ge=1, le=200, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Records to skip"),
    api_key: str = Depends(get_api_key)
):
    """
    Retrieve stored consultation records from the database.
    
    Returns a list of all processed consultations, ordered by most recent first.
    
    - **limit**: Max number of records to return (default: 50, max: 200)
    - **offset**: Number of records to skip for pagination (default: 0)
    """
    try:
        db = DatabaseManager()
        records = db.get_all_records(limit=limit, offset=offset)
        total = db.get_record_count()
        db.close()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/records/{record_id}")
def get_record_by_id(
    record_id: int,
    api_key: str = Depends(get_api_key)
):
    """
    Retrieve a single consultation record by its ID.
    
    - **record_id**: The database ID of the consultation record.
    """
    try:
        db = DatabaseManager()
        record = db.get_record_by_id(record_id)
        db.close()
        if record is None:
            raise HTTPException(status_code=404, detail=f"Record with ID {record_id} not found.")
        return record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
