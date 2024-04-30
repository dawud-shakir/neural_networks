# create_and_save_specs.py


import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os


### Set options here 
                        # originally:
N_FFT = 2048            # 2048
HOP_LENGTH = 512        # 512

# Before preprocessing, the audio files have 1025 frequency bins and 1293 time frames.

normalize_by = 1.0               # normalize values (1.0, np.max)

x_axis = 'time'         # 'time'
y_axis = 'log'          # 'log'


DATA_DIR = os.getcwd() + "/data"    # input directory (audio files)

OUTPUT_DIR = os.getcwd() + "/spectrograms"      # output directory (for spectrograms)

DPI = 600   # output resolution (, 300, 600)



### Find all .au files recursively in /data directory

audio_paths = librosa.util.find_files(DATA_DIR, ext='au', recurse=True)

# Do we have 1,000 files as expected?
assert len(audio_paths) == 1000, f"Expected 1,000 .au files, found {len(audio_paths)}"




### Create and save a spectrogram
### Convert to decibel for visualization
###     dB = 10 x log_10(power / ref)

def create_and_save_spectrogram(audio_path, output_dir):
    y, sr = librosa.load(audio_path, sr=None)    # load audio

    stft_audio = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH) # short-time fourier transform  
    y_audio = np.abs(stft_audio) ** 2                           # square complex values to get magnitude (power) 
    y_audio = librosa.power_to_db(y_audio, ref=normalize_by)    # normalize values by 1.0 (default)

    # Spectrogram
    librosa.display.specshow(y_audio, sr=sr, hop_length=HOP_LENGTH, x_axis=x_axis, y_axis=y_axis)

    #librosa.display.specshow(y_audio, sr=sr, hop_length=HOP_LENGTH)

    #plt.colorbar(format='%+2.0f dB')
    #plt.title(f'Spectrogram: {os.path.basename(audio_path)}')

    ## Spectrograms keep the same relative path, e.g.,
    ## data/train/... -> spectrograms/train/...
    relative_path = os.path.relpath(audio_path, DATA_DIR)  # get relative path
    output_path = os.path.join(output_dir, relative_path.replace('.au', '.png'))  # change extension
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    # create directory 

    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=DPI)    # save with dpi resolution (, 300, 600)
    plt.close()
    



# Iterate over each audio file and create spectrogram
for audio_path in audio_paths:
    print(audio_path)
    create_and_save_spectrogram(audio_path, OUTPUT_DIR)