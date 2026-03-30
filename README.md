# CoreInventory Audio Pipeline

An enhanced audio processing pipeline for transcribing multilingual conversations, identifying speaker roles (Doctor/Patient), and extracting names.

## Features
-   **Multi-format Support**: Process mp3, mp4, ogg, wav, m4a, etc. (Automatic conversion to mp3).
-   **High-Accuracy Transcription**: Powered by Sarvam AI's Saaras:v3 model.
-   **Speaker & Role Identification**: Identifies "Doctor" and "Patient" using conversation context and name extraction.
-   **Clean Structure**: Modular codebase for easy maintenance.

## Project Structure
-   `audio_files/`: Put your input audio files here (mp3, mp4, ogg, etc.).
-   `outputs/`: JSON results will be saved here.
-   `src/audio/`: Audio conversion logic (requires FFmpeg).
-   `src/stt/`: Sarvam AI Client.
-   `src/analysis/`: Speaker role and name identification.
-   `main.py`: Main entry point.

## Setup
1.  **Install Dependencies**:
    ```bash
    pip install pydub requests streamlit
    ```
2.  **FFmpeg**: Ensure FFmpeg is installed and in your system PATH.
3.  **API Key**: Set your Sarvam AI API key in the `SarvamClient` or as an environment variable `SARVAM_API_KEY`.

### 🐳 Docker (Experimental - New Feature)
The easiest way to run the application is using Docker. Ensure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed.

1.  **Build and Run with Docker Compose**:
    ```bash
    docker-compose up --build
    ```
2.  **Access the App**:
    Open your browser and go to `http://localhost:8501`.

3.  **Stopping the Container**:
    ```bash
    docker-compose down
    ```

> [!NOTE]
> The Docker container automatically handles FFmpeg installation and all Python dependencies.

## Usage

### 🚀 Streamlit UI (Recommended)
Launch the user-friendly interface:
```bash
python -m streamlit run app.py
```
This allows you to upload files directly and see the results instantly.

### 💻 Command Line
Run the pipeline with an audio file name:
```bash
python main.py audio.mp3
```
The result will be saved in the `outputs/` folder.
