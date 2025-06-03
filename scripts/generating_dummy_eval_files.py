import os
import shutil
from pydub import AudioSegment
import heapq
from pathlib import Path
from tqdm import tqdm
import wave
import contextlib
import mutagen
import subprocess
import tempfile

def get_audio_duration(file_path):
    """get audio duration without loading the entire file"""
    ext = Path(file_path).suffix.lower()
    try:
        # for wav files, use wave module (very fast)
        if ext == '.wav':
            with contextlib.closing(wave.open(file_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate) * 1000  # in ms
                return duration
        # for mp3, ogg, flac and other formats, use mutagen (faster than pydub)
        else:
            audio = mutagen.File(file_path)
            if audio is not None and hasattr(audio, 'info') and hasattr(audio.info, 'length'):
                return audio.info.length * 1000  # in ms
    except Exception:
        pass

    # fallback to pydub if the above methods fail
    try:
        audio = AudioSegment.from_file(file_path)
        return len(audio)
    except Exception as e:
        print(f"Error getting duration of {file_path}: {e}")
        return 0

def export_to_ogg(audio, output_path):
    """
    export audio to ogg format using direct ffmpeg command.
    modified to use libopus instead of libvorbis.
    """
    # first export to a temporary wav file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        temp_wav_path = temp_wav.name
        audio.export(temp_wav_path, format="wav")
    
    try:
        # then convert to ogg using ffmpeg with libopus as encoder
        cmd = ['ffmpeg', '-y', '-i', temp_wav_path, '-c:a', 'libopus', '-b:a', '96k', output_path]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"error running ffmpeg: {e}")
        return False
    finally:
        # clean up temp file
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def is_ffmpeg_available():
    """check if ffmpeg is available"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def process_audio_files(input_dir, output_dir, num_files=10):
    """
    find the longest audio files in input_dir, trim them to exactly 60 seconds,
    and save them to output_dir. always save as ogg format.
    
    args:
        input_dir (str): directory containing audio files
        output_dir (str): directory to save processed files
        num_files (int): number of files to process
    """
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # check if ffmpeg is available
    if not is_ffmpeg_available():
        print("ERROR: ffmpeg is not available. please install it to use this script.")
        return []
    
    # get all audio files with their durations
    audio_files = []
    
    # list all files first to create a progress bar
    all_files = [f for f in os.listdir(input_dir) if not os.path.isdir(os.path.join(input_dir, f))]
    print(f"analyzing {len(all_files)} files in directory...")
    
    for filename in tqdm(all_files, desc="scanning audio files"):
        file_path = os.path.join(input_dir, filename)
        
        # skip non-audio files (update extensions if needed)
        ext = Path(filename).suffix.lower()
        if ext not in ['.mp3', '.wav', '.ogg', '.0gg', '.flac']:
            continue
            
        try:
            # get audio duration (fast method)
            duration_ms = get_audio_duration(file_path)
            
            if duration_ms > 0:
                # use negative duration for max-heap (to get longest files)
                heapq.heappush(audio_files, (-duration_ms, file_path))
        except Exception as e:
            print(f"error processing {file_path}: {e}")
    
    # get the longest n files
    selected_files = min(num_files, len(audio_files))
    print(f"selecting {selected_files} longest files...")
    longest_files = []
    for _ in range(selected_files):
        if audio_files:
            longest_files.append(heapq.heappop(audio_files)[1])
    
    # process each file
    print(f"processing {len(longest_files)} files...")
    processed_files = []
    
    for file_path in tqdm(longest_files, desc="trimming audio files"):
        try:
            audio = AudioSegment.from_file(file_path)
            
            # trim to exactly 60 seconds (60000 ms)
            if len(audio) > 60000:
                audio = audio[:60000]
            else:
                # if file is shorter than 60s, loop it until it reaches 60s
                # limit repetitions to avoid memory issues
                repeat_count = min(60000 // len(audio) + 1, 100)
                audio = audio * repeat_count
                audio = audio[:60000]
            
            # get output filename, ensuring .ogg extension
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}.ogg"
            output_path = os.path.join(output_dir, output_filename)
            
            # export the file as ogg using direct ffmpeg command with libopus
            if export_to_ogg(audio, output_path):
                processed_files.append(file_path)
            
        except Exception as e:
            print(f"error processing {file_path}: {e}")
    
    return processed_files 

# configuration
input_directory = '/home/george-vengrovski/Documents/projects/Bird_JEPA/temp_safe/test_wav'
output_directory = '/home/george-vengrovski/Documents/projects/Bird_JEPA/kaggle_files_for_speed_testing'

# process the longest audio files
longest_files = process_audio_files(
    input_dir=input_directory,
    output_dir=output_directory,
    num_files=800
)

print(f"processed {len(longest_files)} files")
print("original files processed:")
for file in longest_files[:10]:  # show only first 10 files to avoid excessive output
    print(f" - {file}")
if len(longest_files) > 10:
    print(f"... and {len(longest_files) - 10} more files")
