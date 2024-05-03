# create_and_save_spectrograms.py


import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os

import time

### Set options here 
                        
N_FFT = 2048                                            # fourier transforms
HOP_LENGTH = 512                                        # step size for audio stfts
NORMALIZE_BY = np.max                                   # normalize values (default: 1.0)

x_axis = 'time'         # 'time'
y_axis = 'log'          # 'log'

DATA_DIRECTORY = os.getcwd() + "/data"                  # input directory (audio files)
OUTPUT_DIRECTORY = os.getcwd() + "/spectrograms"        # output directory (for spectrograms)
OUTPUT_DOT_PER_INCH = 600                               # output resolution (, 300, 600)

####################################################################################

### Find all .au files recursively in /data directory

audio_paths = librosa.util.find_files(DATA_DIRECTORY, ext='au', recurse=True)

# Do we have 1,000 files as expected?
assert len(audio_paths) == 1000, f"Expected 1,000 .au files, found {len(audio_paths)}"

# Read labels from audio path
# example: "blues.00000.au" -> "blues"
labels = [os.path.basename(path).split(".")[0] for path in audio_paths]

######## generate plots for report

import torch
import torchaudio
import IPython.display as ipd


# audio path ...
au_path = audio_paths[0]
# .. class
au_class = labels[0]

#### How long to convert audio files to spectrograms?

tic_time = time.time

sound, sample_rate = torchaudio.load(audio_paths[0])
assert sound.shape == torch.Size([1, 661794]), "wrong audio file or channel"

toc_time = (time.time - tic_time)

print("converted audio files to mel in {} seconds".format(tock, "%.4"))

print("loaded sound size:", sound.shape)


ipd.Audio(data=sound,rate=sample_rate) # load a local WAV file


sound, sample_rate = librosa.load(audio_paths[0], sr=sample_rate)
    
# Show wave form of audio file
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y=sound, sr=sample_rate, color="blue")


X = librosa.stft(y=sound)
Xdb = librosa.amplitude_to_db(abs(X))

#librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
Xdb.shape

NUM_OF_MELS = 128


S = librosa.feature.melspectrogram(y=sound, sr=sample_rate, n_mels=NUM_OF_MELS)
log_S = librosa.power_to_db(S, ref=np.max)  # log scale and normalize values by max 


#NUM_OF_MFCCS = 23
#DERIVATIVE = 2 # second order 

#MFCC = librosa.feature.mfcc(S=log_S, n_mfcc=NUM_OF_MFCCS)   # mfcc
#delta2_mfcc = librosa.feature.delta(MFCC, order=DERIVATIVE) # derivative

#MFCC = librosa.feature.mfcc(y=x, sr=sample_rate,n_mfcc=23,dct_type=2)
plt.figure(figsize=(14, 10))
SAMPLE_RATE = sample_rate
plt.subplot(3,1,2)
plt.title(f"sample rate = {SAMPLE_RATE}")
librosa.display.waveshow(y=log_S, sr=SAMPLE_RATE, color="blue")





plt.subplot(3,1,2)
SAMPLE_RATE *= 2
plt.title(f"sample rate = {SAMPLE_RATE}")
librosa.display.waveshow(y=log_S, sr=SAMPLE_RATE, color="red")

plt.subplot(3,1,2)
SAMPLE_RATE *= 4
plt.title(f"sample rate = {SAMPLE_RATE}")
librosa.display.waveshow(y=log_S, sr=SAMPLE_RATE, color="green")

plt.subplot(3,1,2)
SAMPLE_RATE *= 100
plt.title(f"sample rate = {SAMPLE_RATE}")
librosa.display.waveshow(y=log_S, sr=SAMPLE_RATE, color="green")


plt.figure(figsize=(14, 5))

librosa.display.specshow(log_S)
#print(np.max(MFCC),np.min(MFCC))
#MFCC = (MFCC+200)/500
#print(np.max(MFCC),np.min(MFCC))
plt.colorbar()
plt.tight_layout()
plt.show()






### Create and save a spectrogram
### Convert to decibel for visualization
###     dB = 10 x log_10(power / ref)

def create_and_save_spectrogram(audio_path, output_dir):
    y, sr = librosa.load(audio_path, sr=None)    # load audio

    stft_audio = librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH) # short-time fourier transform  
    y_audio = np.abs(stft_audio) ** 2                           # square complex values to get magnitude (power) 
    y_audio = librosa.power_to_db(y_audio, ref=NORMALIZE_BY)    # normalize values by 1.0 (default)

    # Spectrogram
    librosa.display.specshow(y_audio, sr=sr, hop_length=HOP_LENGTH, x_axis=x_axis, y_axis=y_axis)

    #librosa.display.specshow(y_audio, sr=sr, hop_length=HOP_LENGTH)

    #plt.colorbar(format='%+2.0f dB')
    #plt.title(f'Spectrogram: {os.path.basename(audio_path)}')

    ## Spectrograms keep the same relative path, e.g.,
    ## data/train/... -> spectrograms/train/...
    relative_path = os.path.relpath(audio_path, DATA_DIRECTORY)  # get relative path
    output_path = os.path.join(output_dir, relative_path.replace('.au', '.png'))  # change extension
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    # create directory 

    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(output_path, dpi=OUTPUT_DOT_PER_INCH)    # save with dpi resolution (, 300, 600)
    plt.close()
    



# Iterate over each audio file and create spectrogram
for audio_path in audio_paths:
    print(audio_path)
    create_and_save_spectrogram(audio_path, OUTPUT_DIRECTORY)