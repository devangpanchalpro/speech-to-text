import os
import json
from src.audio.audio_processor import AudioProcessor
from src.stt.sarvam_client import SarvamClient
from src.analysis.role_identifier import RoleIdentifier
from src.analysis.casesheet_extractor import CasesheetExtractor

# Directory setup
INPUT_DIR = "audio_files"
OUTPUT_DIR = "outputs"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_audio_pipeline(audio_filename, api_key=None, gemini_api_key=None, language_code="unknown"):
    """
    Full pipeline: Convert -> STT -> Role Identification -> JSON Output
    """
    try:
        # Resolve path - carefully avoid double-prepending INPUT_DIR
        if os.path.isabs(audio_filename):
            input_path = audio_filename
        elif audio_filename.startswith(INPUT_DIR + os.sep) or audio_filename.startswith(INPUT_DIR + "/"):
            input_path = audio_filename
        else:
            input_path = os.path.join(INPUT_DIR, audio_filename)
        
        # 1. Audio Processing
        print("\n" + "="*50)
        print("🎵 STAGE 1: Audio Processing")
        print("="*50)
        processor = AudioProcessor()
        # We save processed mp3 into the audio_files folder
        base_name = os.path.basename(input_path)
        audio_path_mp3 = os.path.join(INPUT_DIR, os.path.splitext(base_name)[0] + ".mp3")
        audio_path_mp3 = processor.convert_to_mp3(input_path, output_path=audio_path_mp3)
        
        # Cleanup: Delete the original if it's not the same as mp3
        if os.path.exists(input_path) and os.path.abspath(input_path) != os.path.abspath(audio_path_mp3):
            try:
                os.remove(input_path)
                print(f"🗑️ Original file removed: {input_path}")
            except Exception as e:
                print(f"⚠️ Could not remove original file: {e}")
        
        # 2. Transcription
        print("\n" + "="*50)
        print("📝 STAGE 2: Transcription (Sarvam AI)")
        print("="*50)
        client = SarvamClient(api_key=api_key)
        transcript_data = client.transcribe(audio_path_mp3, language_code=language_code)
        
        if not transcript_data or 'transcript' not in transcript_data:
            print("❌ Transcription failed.")
            return None
            
        # Ensure empty string from UI counts as None
        if not gemini_api_key:
            gemini_api_key = None

        # 3. Translation (Source to English) — Do this FIRST so English text is available for all later stages
        print("\n" + "="*50)
        print("🌎 STAGE 3: Translating to English")
        print("="*50)
        source_transcript = transcript_data['transcript']
        translated_english = source_transcript
        
        actual_lang = transcript_data.get('language_code', language_code)

        if actual_lang.startswith('en'):
            print(f"ℹ️ Audio is detected as English ({actual_lang}). Skipping translation.")
            translated_english = source_transcript
        elif api_key:
            try:
                translated_text = client.translate_text(
                    text=source_transcript, 
                    source_language_code=actual_lang, 
                    target_language_code='en-IN'
                )
                
                if translated_text:
                    translated_english = translated_text
                
                if translated_english == source_transcript and source_transcript.strip():
                    print("⚠️ Translation might have failed or returned original text.")
            except Exception as e:
                print(f"⚠️ Translation error: {e}")
                translated_english = source_transcript
        else:
            print("ℹ️ No Sarvam API key provided; English version will match original.")
            translated_english = source_transcript

        # 4. Role & Name Identification (uses both original + English transcript)
        print("\n" + "="*50)
        print("🔍 STAGE 4: Identifying Roles & Names")
        print("="*50)
        role_id = RoleIdentifier()
        identification = role_id.identify_roles_and_names(
            transcript_data['transcript'], 
            gemini_api_key=gemini_api_key,
            actual_lang=actual_lang
        )

        # 5. Casesheet Extraction (Dynamic EMR JSON)
        print("\n" + "="*50)
        print("📋 STAGE 5: Casesheet Extraction (EMR Format)")
        print("="*50)
        casesheet_data = {}
        if gemini_api_key:
            try:
                extractor = CasesheetExtractor(gemini_api_key)
                casesheet_data = extractor.extract_casesheet(
                    transcript=translated_english,
                    conversation=identification.get('conversation'),
                    doctor_name=identification['doctor']['name'],
                    patient_name=identification['patient']['name']
                )
            except Exception as e:
                print(f"⚠️ Casesheet extraction error: {e}")
                casesheet_data = CasesheetExtractor()._get_empty_casesheet()
        else:
            print("ℹ️ No Gemini API key provided; skipping casesheet extraction.")
            casesheet_data = CasesheetExtractor()._get_empty_casesheet()

        # 6. Save Final Output
        metadata = {
            "original_file": os.path.basename(input_path),
            "processed_file": os.path.basename(audio_path_mp3),
            "detected_language": transcript_data['language_code']
        }

        result = {
            "metadata": metadata,
            "transcript": source_transcript,
            "transcript_english": translated_english,
            "identification": identification,
            "casesheet": casesheet_data
        }
        
        output_filename = f"result_{os.path.splitext(os.path.basename(input_path))[0]}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
            
        print(f"\n✅ Pipeline complete! Final result saved to {output_path}")
        print("="*50)
        print(f"👨‍⚕️ Doctor: {identification['doctor']['name']}")
        print(f"🏥 Patient: {identification['patient']['name']}")
        print(f"🌐 Language: {transcript_data['language_code']}")
        print("="*50)
        
        return result

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_audio = sys.argv[1]
        process_audio_pipeline(target_audio)
    else:
        print("Usage: python main.py <audio_file_name>")
