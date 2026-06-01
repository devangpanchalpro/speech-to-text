# CoreInventory Audio Pipeline

An enhanced audio processing pipeline for transcribing multilingual conversations, identifying speaker roles (Doctor/Patient), extracting names, and generating HMIS-compatible JSON.

## Features
-   **Multi-format Support**: Process mp3, mp4, ogg, wav, m4a, opus, etc. (Automatic conversion to mp3).
-   **High-Accuracy Transcription**: Powered by Sarvam AI's Saaras:v3 model.
-   **Speaker & Role Identification**: Identifies "Doctor" and "Patient" using conversation context and name extraction.
-   **HMIS JSON Output**: Directly outputs JSON compatible with the HMIS website API format.
-   **FastAPI Server**: Production-ready REST API with API key authentication.

## Project Structure
-   `audio_files/`: Put your input audio files here (mp3, mp4, ogg, opus, etc.).
-   `outputs/`: JSON results will be saved here.
-   `src/audio/`: Audio conversion logic (requires FFmpeg).
-   `src/stt/`: Sarvam AI Client.
-   `src/analysis/`: Speaker role identification, casesheet extraction, and HMIS mapping.
-   `api.py`: FastAPI REST API server.
-   `main.py`: Main pipeline entry point.

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **FFmpeg**: Ensure FFmpeg is installed and in your system PATH.
3.  **Environment Variables**: Copy `.env.example` to `.env` and fill in your API keys:
    ```
    SARVAM_API_KEY=your_key
    GEMINI_API_KEY=your_key
    APP_API_KEY=your_api_key
    ```

## Usage

### 🚀 FastAPI Server (Recommended)
Start the API server:
```bash
uvicorn api:app --reload --port 8000
```

Then access the interactive API docs at `http://localhost:8000/docs`.

#### API Endpoint
```bash
POST /process-audio
```
- **Headers**: `X-API-Key: your_api_key`
- **Body** (form-data): `file` (audio file)
- **Returns**: HMIS-compatible JSON with extracted medical data

### 💻 Command Line
Run the pipeline directly:
```bash
python main.py audio.mp3
```
The result will be saved in the `outputs/` folder.

### 🐳 Docker
```bash
docker-compose up --build
```
API will be available at `http://localhost:8000`.

> [!NOTE]
> The Docker container automatically handles FFmpeg installation and all Python dependencies.
