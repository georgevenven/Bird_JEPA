import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import multiprocessing
import soundfile as sf
import gc
import csv
import sys
import psutil
from tqdm import tqdm
import json
import argparse
from scipy.signal import ellip, filtfilt
import time
from multiprocessing import Pool, TimeoutError
import logging
import torch
import torchaudio
from timing_utils import Timer, timed_operation, timing_stats

class WavtoSpec:
    def __init__(
        self,
        src_dir,
        dst_dir,
        song_detection_json_path=None,
        step_size=119,
        nfft=1024,
        generate_random_files_number=None,
        single_threaded=True
    ):
        """
        Constructor for the WavtoSpec class.

        Args:
            src_dir (str): Source directory containing .wav files.
            dst_dir (str): Destination directory to save spectrograms.
            song_detection_json_path (str or None): Path to JSON with song detection data.
            step_size (int): Hop length for the STFT.
            nfft (int): FFT size for the STFT.
            generate_random_files_number (int or None): If set, processes only N random files.
            single_threaded (bool): Whether to run single-threaded (True) or multi-threaded (False).
        """
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.song_detection_json_path = None if song_detection_json_path == "None" else song_detection_json_path
        self.use_json = self.song_detection_json_path is not None
        self.step_size = step_size
        self.nfft = nfft
        self.generate_random_files_number = generate_random_files_number
        self.single_threaded = single_threaded
        
        # Initialize the shared counter
        manager = multiprocessing.Manager()
        self.skipped_files_count = manager.Value('i', 0)

        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            filename='error_log.log',
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_directory(self):
        print("Starting process_directory")
        print(f"Source directory: {self.src_dir}")

        # Gather .wav and .mp3 files
        audio_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(self.src_dir)
            for file in files if file.lower().endswith(('.wav', '.mp3', '.ogg'))
        ]

        # Handle random file selection if user requested
        if (self.generate_random_files_number is not None 
                and self.generate_random_files_number > len(audio_files)):
            print(f"Requested {self.generate_random_files_number} random files, but only {len(audio_files)} available. "
                  "Processing all files.")
            self.generate_random_files_number = None

        if self.generate_random_files_number is not None:
            print(f"Selecting {self.generate_random_files_number} random files")
            audio_files = np.random.choice(audio_files, self.generate_random_files_number, replace=False)
            print(f"{len(audio_files)} random files selected")

        manager = multiprocessing.Manager()
        failed_files = manager.list()  # shared list for failed files

        pbar = tqdm(total=len(audio_files), desc="Processing files")

        # ------------------------------------
        # SINGLE-THREADED MODE
        # ------------------------------------
        if self.single_threaded:
            print("Running in single-threaded mode.")
            for file_path in audio_files:
                try:
                    print(f"[Single-thread] Processing file: {file_path}")
                    if self.song_detection_json_path is not None:
                        # Only process if this file has valid vocalization info
                        if self.has_vocalization(file_path):
                            self.multiprocess_process_file(file_path, self.song_detection_json_path)
                        else:
                            print(f"File {file_path} skipped due to no vocalization")
                    else:
                        self.multiprocess_process_file(file_path, None)

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    self.skipped_files_count.value += 1
                    failed_files.append(file_path)

                pbar.update()

            # Retry failed files (still single-threaded)
            if failed_files:
                print(f"Retrying {len(failed_files)} failed files (single-threaded)...")
                pbar_failed = tqdm(total=len(failed_files), desc="Retrying failed files")
                for file_path in failed_files:
                    try:
                        print(f"[Single-thread retry] Processing file: {file_path}")
                        if self.song_detection_json_path is not None:
                            if self.has_vocalization(file_path):
                                self.multiprocess_process_file(file_path, self.song_detection_json_path)
                            else:
                                print(f"File {file_path} skipped due to no vocalization")
                        else:
                            self.multiprocess_process_file(file_path, None)
                    except Exception as e:
                        logging.error(f"Error re-processing {file_path}: {e}")
                        self.skipped_files_count.value += 1
                    pbar_failed.update()
                pbar_failed.close()

        # ------------------------------------
        # MULTI-THREADED MODE
        # ------------------------------------
        else:
            max_processes = multiprocessing.cpu_count()
            print(f"Running in multi-threaded mode with {max_processes} workers.")

            def update_progress_bar(_):
                pbar.update()

            def error_callback(e):
                logging.error(f"Error: {e}")
                self.skipped_files_count.value += 1
                pbar.update()

            def file_failed_callback(file_path):
                logging.error(f"File failed: {file_path}")
                failed_files.append(file_path)
                pbar.update()

            with Pool(processes=max_processes, maxtasksperchild=100) as pool:
                for file_path in audio_files:
                    # Simple memory check loop
                    while True:
                        available_memory = psutil.virtual_memory().available
                        current_memory_usage = psutil.Process(os.getpid()).memory_info().rss
                        if available_memory > current_memory_usage:
                            break
                        print("Not enough memory to spawn new process, waiting...")
                        time.sleep(1)

                    print(f"[Multi-thread] Processing file: {file_path}")
                    pool.apply_async(
                        WavtoSpec.safe_process_file,
                        args=(self, file_path),
                        callback=update_progress_bar,
                        error_callback=lambda e, fp=file_path: file_failed_callback(fp)
                    )

                pool.close()
                pool.join()

            # Retry failed files in multi-threaded mode
            if failed_files:
                print(f"Retrying {len(failed_files)} failed files in multi-threaded mode...")
                pbar_failed = tqdm(total=len(failed_files), desc="Retrying failed files")
                with Pool(processes=max_processes, maxtasksperchild=100) as pool:
                    for file_path in failed_files:
                        print(f"[Multi-thread retry] Processing file: {file_path}")
                        pool.apply_async(
                            WavtoSpec.safe_process_file,
                            args=(self, file_path),
                            callback=lambda _: pbar_failed.update(),
                            error_callback=error_callback
                        )
                    pool.close()
                    pool.join()
                pbar_failed.close()

        pbar.close()

        total_processed = len(audio_files) - len(failed_files)
        print(f"Total files processed: {total_processed}")
        print(f"Total files skipped due to errors or no vocalization data: {self.skipped_files_count.value}")

    def has_vocalization(self, file_path):
        file_name = os.path.basename(file_path)
        vocalization_data, _ = self.check_vocalization(
            file_name=file_name,
            data=None,
            samplerate=None,
            song_detection_json_path=self.song_detection_json_path
        )
        if vocalization_data is None:
            self.increment_skip_counter()
            return False
        return True

    @staticmethod
    def safe_process_file(instance, file_path):
        """
        Thin wrapper for multiprocess calls. 
        Logs errors instead of letting them bubble up.
        """
        try:
            if instance.song_detection_json_path is not None:
                if instance.has_vocalization(file_path):
                    instance.multiprocess_process_file(file_path, instance.song_detection_json_path)
                else:
                    logging.info(f"File {file_path} skipped due to no vocalization")
                    return None
            else:
                instance.multiprocess_process_file(file_path, None)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            instance.increment_skip_counter()
            return None
        return file_path

    def multiprocess_process_file(self, file_path, song_detection_json_path):
        return self.convert_to_spectrogram(
            file_path,
            song_detection_json_path=song_detection_json_path,
            save_npz=True
        )

    def increment_skip_counter(self):
        """Helper method to increment the skip counter in a thread-safe way"""
        if hasattr(self, 'skipped_files_count'):
            self.skipped_files_count.value += 1

    def convert_to_spectrogram(self, file_path, song_detection_json_path, min_length_ms=25,
                               min_timebins=25, save_npz=True):
        try:
            # Use librosa to load audio instead of soundfile (handles both WAV and MP3)
            data, samplerate = librosa.load(file_path, sr=None, mono=True)
            
            # Check duration
            length_in_ms = (len(data) / samplerate) * 1000
            if length_in_ms < min_length_ms:
                logging.info(f"File {file_path} skipped: below length threshold ({length_in_ms}ms < {min_length_ms}ms)")
                self.increment_skip_counter()
                return None, None
            
            # For vocalization data
            if song_detection_json_path is not None:
                vocalization_data, syllable_labels = self.check_vocalization(
                    file_name=os.path.basename(file_path),
                    data=data,
                    samplerate=samplerate,
                    song_detection_json_path=song_detection_json_path
                )
                if not vocalization_data:
                    logging.info(f"File {file_path} skipped: no vocalization data found")
                    self.increment_skip_counter()
                    return None, None
            else:
                # Assume entire file is a vocalization if no JSON is given
                vocalization_data = [(0, len(data)/samplerate)]
                syllable_labels = {}
            
            # Apply high-pass filter (no change needed)
            b, a = ellip(5, 0.2, 40, 500/(samplerate/2), 'high')
            data = filtfilt(b, a, data)
            
            # Compute STFT
            Sxx = librosa.stft(data.astype(float), n_fft=self.nfft, hop_length=self.step_size, window='hann')
            del data  # Remove data from memory as soon as we're done with it
            
            Sxx_log = librosa.amplitude_to_db(np.abs(Sxx), ref=np.max)
            
            # Prepare label array
            labels = np.zeros(Sxx.shape[1], dtype=int)

            # Mark labeled segments if provided
            for label, intervals in syllable_labels.items():
                for start_sec, end_sec in intervals:
                    start_bin = np.searchsorted(np.arange(Sxx.shape[1]) * self.step_size / samplerate, start_sec)
                    end_bin = np.searchsorted(np.arange(Sxx.shape[1]) * self.step_size / samplerate, end_sec)
                    labels[start_bin:end_bin] = int(label)
                
            # Process each segment individually to reduce peak memory usage
            results = []
            for i, (start_sec, end_sec) in enumerate(vocalization_data):
                start_bin = np.searchsorted(
                    np.arange(Sxx.shape[1]) * self.step_size / samplerate, start_sec)
                end_bin = np.searchsorted(
                    np.arange(Sxx.shape[1]) * self.step_size / samplerate, end_sec)

                if (end_bin - start_bin) < min_timebins:
                    print(f"Segment {i} has fewer than {min_timebins} time bins and will be skipped.")
                    continue

                # Extract segment
                segment_Sxx_log = Sxx_log[:, start_bin:end_bin].copy()  # Force copy to avoid reference issues
                segment_labels = labels[start_bin:end_bin].copy()
                segment_vocalization = np.ones(end_bin - start_bin, dtype=int)

                if save_npz:
                    spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                    segment_spec_file_path = os.path.join(
                        self.dst_dir, f"{spec_filename}_segment_{i}.npz"
                    )
                    
                    # Quantize spectrogram to uint8 to save space
                    # First normalize to 0-255 range
                    min_val = segment_Sxx_log.min()
                    max_val = segment_Sxx_log.max()
                    if max_val > min_val:  # Avoid division by zero
                        segment_Sxx_log_uint8 = np.uint8(255 * (segment_Sxx_log - min_val) / (max_val - min_val))
                    else:
                        segment_Sxx_log_uint8 = np.zeros_like(segment_Sxx_log, dtype=np.uint8)
                    
                    # Save the quantized data without compression
                    np.savez(  # Use uncompressed version
                        segment_spec_file_path,
                        s=segment_Sxx_log_uint8,
                        vocalization=segment_vocalization,
                        labels=segment_labels,
                        min_val=min_val,  # Store scaling info for reconstruction
                        max_val=max_val
                    )
                    
                    print(f"Segment {i} spectrogram, vocalization data, and labels saved to {segment_spec_file_path}")
                    
                    # Clear segment data after saving to reduce memory usage
                    del segment_Sxx_log
                    del segment_Sxx_log_uint8
                    del segment_labels
                    del segment_vocalization
                    gc.collect()  # Force garbage collection
                
                results.append((start_sec, end_sec))
            
            return Sxx_log, results, labels

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            gc.collect()  # Force garbage collection
            return None, None, None

    def check_vocalization(self, file_name, data, samplerate, song_detection_json_path):
        """
        Return only those segments specifically marked as containing song.
        If no JSON is provided, or if 'song_present' is not True, skip the file.
        """
        if not self.use_json:
            return [(0, len(data)/samplerate)], {}

        if not os.path.exists(song_detection_json_path):
            logging.error(f"JSON file {song_detection_json_path} does not exist")
            return None, None

        try:
            with open(song_detection_json_path, 'r') as json_file:
                json_data = json.load(json_file)
                for entry in json_data:
                    if entry['filename'] == file_name:
                        # If the JSON explicitly says there's no song, skip entirely
                        if not entry.get('song_present', False):
                            logging.info(f"File {file_name} skipped: no song present according to JSON")
                            return None, None
                        
                        onsets_offsets = []
                        for seg in entry.get('segments', []):
                            onsets_offsets.append(
                                (seg['onset_ms'] / 1000, seg['offset_ms'] / 1000)
                            )

                        # if there are no valid segments, skip
                        if not onsets_offsets:
                            return None, None

                        syllable_labels = entry.get('syllable_labels', {})
                        return onsets_offsets, syllable_labels
        except Exception as e:
            logging.error(f"Error reading JSON file {song_detection_json_path}: {e}")
            return None, None

        logging.info(f"File {file_name} skipped: no matching entry found in {song_detection_json_path}")
        return None, None

class SpectrogramGenerator:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128, debug=False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.debug = debug

        with Timer("spectrogram_generator_initialization", debug=debug):
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0
            )

    @timed_operation("generate_spectrogram")
    def generate_spectrogram(self, audio):
        """
        Generate mel spectrogram from audio waveform.
        Args:
            audio (torch.Tensor): Audio waveform of shape (1, T) or (T,)
        Returns:
            torch.Tensor: Mel spectrogram of shape (n_mels, T)
        """
        with Timer("audio_preprocessing", debug=self.debug):
            # Ensure audio is the right shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

        with Timer("mel_spectrogram_computation", debug=self.debug):
            # Generate mel spectrogram
            mel_spec = self.mel_transform(audio)

        with Timer("spectrogram_postprocessing", debug=self.debug):
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-9)

            # Normalize
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

        return mel_spec

    @timed_operation("generate_batch_spectrograms")
    def generate_batch_spectrograms(self, audio_batch):
        """
        Generate mel spectrograms for a batch of audio waveforms.
        Args:
            audio_batch (torch.Tensor): Batch of audio waveforms of shape (B, 1, T) or (B, T)
        Returns:
            torch.Tensor: Batch of mel spectrograms of shape (B, n_mels, T)
        """
        with Timer("batch_preprocessing", debug=self.debug):
            # Ensure audio batch is the right shape
            if audio_batch.dim() == 2:
                audio_batch = audio_batch.unsqueeze(1)

        with Timer("batch_spectrogram_generation", debug=self.debug):
            # Generate spectrograms for each audio in the batch
            spectrograms = []
            for audio in audio_batch:
                spec = self.generate_spectrogram(audio)
                spectrograms.append(spec)

        with Timer("batch_postprocessing", debug=self.debug):
            # Stack spectrograms into a batch
            batch_spec = torch.stack(spectrograms)

        return batch_spec

    @timed_operation("apply_augmentation")
    def apply_augmentation(self, spectrogram, augmentation_type='time_mask'):
        """
        Apply augmentation to a spectrogram.
        Args:
            spectrogram (torch.Tensor): Mel spectrogram of shape (n_mels, T)
            augmentation_type (str): Type of augmentation to apply
        Returns:
            torch.Tensor: Augmented mel spectrogram
        """
        with Timer("augmentation_application", debug=self.debug):
            if augmentation_type == 'time_mask':
                # Time masking
                time_masking_param = 80
                freq_masking_param = 80
                
                B, C, T = spectrogram.shape
                for i in range(B):
                    for _ in range(2):  # Apply 2 time masks
                        t = np.random.randint(0, time_masking_param)
                        t_zero = np.random.randint(0, T - t)
                        spectrogram[i, :, t_zero:t_zero + t] = 0
                        
                    for _ in range(2):  # Apply 2 frequency masks
                        f = np.random.randint(0, freq_masking_param)
                        f_zero = np.random.randint(0, C - f)
                        spectrogram[i, f_zero:f_zero + f, :] = 0

            elif augmentation_type == 'noise':
                # Add Gaussian noise
                noise = torch.randn_like(spectrogram) * 0.005
                spectrogram = spectrogram + noise

            elif augmentation_type == 'pitch_shift':
                # Simulate pitch shift by shifting frequencies
                shift = np.random.randint(-4, 4)
                spectrogram = torch.roll(spectrogram, shifts=shift, dims=1)

        return spectrogram

    @timed_operation("generate_masked_spectrogram")
    def generate_masked_spectrogram(self, spectrogram, mask_ratio=0.15):
        """
        Generate a masked version of the spectrogram for training.
        Args:
            spectrogram (torch.Tensor): Original mel spectrogram
            mask_ratio (float): Ratio of the spectrogram to mask
        Returns:
            tuple: (masked_spectrogram, mask)
        """
        with Timer("mask_generation", debug=self.debug):
            # Create random mask
            mask = torch.rand_like(spectrogram) < mask_ratio

        with Timer("mask_application", debug=self.debug):
            # Apply mask
            masked_spectrogram = spectrogram.clone()
            masked_spectrogram[mask] = 0

        return masked_spectrogram, mask

    @timed_operation("process_audio_file")
    def process_audio_file(self, audio_path):
        """
        Process an audio file into a spectrogram.
        Args:
            audio_path (str): Path to the audio file
        Returns:
            torch.Tensor: Mel spectrogram
        """
        with Timer("audio_loading", debug=self.debug):
            # Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)

        with Timer("spectrogram_generation", debug=self.debug):
            # Generate spectrogram
            spectrogram = this.generate_spectrogram(waveform)

        return spectrogram

def main():
    parser = argparse.ArgumentParser(description="Convert WAV files to spectrograms.")
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing WAV files.')
    parser.add_argument('--dst_dir', type=str, required=True, help='Destination directory to save spectrograms.')
    parser.add_argument('--song_detection_json_path', type=str, default=None,
                        help='Path to the JSON file with song detection data.')
    parser.add_argument('--step_size', type=int, default=119, help='Hop length for the spectrogram.')
    parser.add_argument('--nfft', type=int, default=1024, help='FFT size for the spectrogram.')
    parser.add_argument('--generate_random_files_number', type=int, default=None,
                        help='Number of random files to process.')
    parser.add_argument('--single_threaded', 
                        type=str,
                        default='true',
                        choices=['true', 'false', '1', '0', 'yes', 'no'],
                        help='Whether to run single-threaded (True) or multi-threaded (False). '
                             'Default is True. Example usage: --single_threaded false')

    args = parser.parse_args()
    
    # Convert the string to boolean after parsing
    single_threaded = args.single_threaded.lower() in ['true', '1', 'yes']
    
    wav_to_spec = WavtoSpec(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        song_detection_json_path=args.song_detection_json_path,
        step_size=args.step_size,
        nfft=args.nfft,
        generate_random_files_number=args.generate_random_files_number,
        single_threaded=single_threaded
    )
    wav_to_spec.process_directory()  

if __name__ == "__main__":
    main()



'''
Spectrogram Generation and Preprocessing
Raw audio recordings are filtered to remove low-frequency noise components and emphasize relevant spectral features of canary songs. A 5th-order elliptic high-pass filter (0.2 dB ripple, 40 dB stopband attenuation, 500 Hz cutoff frequency) is applied to the time-domain signal, attenuating energy below 500 Hz that is often dominated by low-frequency noise or cage sounds.

Spectrograms are generated using a Short-Time Fourier Transform (STFT) with a Hann window, a 1024-point FFT, and a 119-sample hop length. At a 44.2 kHz sampling rate, this hop length provides temporal resolution sufficient to capture fine-grained structure in canary syllables while maintaining computational efficiency. The complex spectrogram is converted to a decibel scale via amplitude-to-dB conversion (referenced to maximum amplitude).

Handling of Long and Short Recordings
Recordings shorter than 500 ms are discarded to maintain consistent quality and uniformity in the dataset. For longer recordings, the system processes them in manageable segments to prevent memory issues during STFT and embedding processes.

Vocalization-Based Filtering and JSON Metadata
To avoid processing non-vocal segments (silence or environmental noise), a "song detection" JSON file can be provided. For each WAV file, the system checks for corresponding JSON entries containing segment onsets, offsets, and syllable-level annotations. Only time intervals marked as containing vocalizations are retained. If no vocalizations are present, the file is skipped.

Segmenting Spectrograms
After applying the STFT, segments with fewer than 1000 time bins (~2.7 seconds) are discarded to ensure sufficient contextual information for downstream analyses. For segments meeting this minimum length criteria, the pipeline generates spectrogram excerpts corresponding to vocalization intervals.

Each excerpt is saved as a compressed NumPy archive (.npz) containing three arrays:
1. The log-amplitude spectrogram
2. A binary "vocalization" array indicating that the entire segment contains song
3. Syllable-level label arrays if available from JSON annotations

Labeling and Integration
When syllable-level labels are provided, the pipeline aligns these annotations with the spectrogram's time bins, converting time bins to seconds using the sampling rate and hop length. This integration facilitates subsequent evaluations of model performance and alignment with biologically meaningful vocal units.

Parallelization and Memory Management
The code can leverage multiprocessing capabilities, distributing file processing tasks across available CPU cores. Memory checks ensure that new processes are spawned only when sufficient system memory is available. Files that encounter errors are logged and the pipeline attempts to reprocess them after completing the initial batch.
'''