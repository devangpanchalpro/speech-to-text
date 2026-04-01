import os
import subprocess
import imageio_ffmpeg

class AudioProcessor:
    """Handles audio format conversion and loading using imageio-ffmpeg directly."""
    
    SUPPORTED_FORMATS = ['mp3', 'mp4', 'ogg', 'wav', 'm4a', 'flac']

    @staticmethod
    def convert_to_mp3(input_path, output_path=None):
        """
        Converts any supported audio format to mp3 using ffmpeg via subprocess.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        file_extension = input_path.split('.')[-1].lower()
        
        if file_extension not in AudioProcessor.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {file_extension}. Supported: {AudioProcessor.SUPPORTED_FORMATS}")

        if not output_path:
            output_path = os.path.splitext(input_path)[0] + ".mp3"

        if file_extension == 'mp3' and os.path.abspath(input_path) == os.path.abspath(output_path):
            return input_path

        print(f"🎵 Converting {input_path} to {output_path}...")
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Build ffmpeg command
        # -y: overwrite output files without asking
        # -i: input file
        command = [ffmpeg_exe, "-y", "-i", input_path, output_path]
        
        try:
            # Run conversion
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"✅ Conversion successful: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg Error: {e.stderr}")
            raise e
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            raise e

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        try:
            AudioProcessor.convert_to_mp3(sys.argv[1])
        except Exception as e:
            print(f"Test failed: {e}")
