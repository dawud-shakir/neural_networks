# Waveform_and_Spectogram_Plots
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os

# Base path to the data
lpath = os.getcwd()


# Locally saved!
SAVE_PATH = "/Users/a2024/Library/Mobile Documents/com~apple~CloudDocs/2024/CS529/Project 3/figures/"
SAVE_EXT = ".png"



# (l)oad path
github = "https://github.com/dawud-shakir/logistic_regression/raw/main"   # same as logistic regression

# Function to plot waveform
def plot_waveform(y, sr, ax, title):
    librosa.display.waveshow(y, sr=sr, ax=ax, color="blue")
    ax.set_title(f"{title} Waveform")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True)

# Function to plot spectrogram
def plot_stft_spectrogram(y, sr, hop_length, y_axis, ax, title):
    librosa.display.specshow(
        y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis, ax=ax
    )
    ax.set_title(f"{title} Spectrogram ({y_axis})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_axis.capitalize())
    #plt.colorbar()

# Function to plot spectrogram
def plot_mel_spectrogram(y, sr, hop_length, y_axis, ax, title):
    n_mels = 128
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Convert the Mel spectrogram to decibels (logarithmic scale)
    mel_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.specshow(
        mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis, ax=ax
    )
    ax.set_title(f"{title} Spectrogram ({y_axis})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_axis.capitalize())
    #plt.colorbar()

# Function to plot sample spectrum (frequency scale)
def plot_sample_spectrum(y, sr, hop_length, y_axis, ax, title):
    n_mels = 128
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Convert the Mel spectrogram to decibels (logarithmic scale)
    mel_db = librosa.power_to_db(S, ref=np.max)

    librosa.display.waveshow(
        mel_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis, ax=ax
    )
    ax.set_title(f"{title} Sample Spectrum for 1 Audio Image ({y_axis})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_axis.capitalize())
    #plt.colorbar()



# Sampling  
frameSize = 2048
hopSize = 512

# Audio files
audio_files = {
    "Blues": lpath + "/data/train/blues/blues.00000.au",
    "Classical": lpath + "/data/train/classical/classical.00000.au",
    "Country": lpath + "/data/train/country/country.00000.au",
    "Disco": lpath + "/data/train/disco/disco.00000.au",
    "Hiphop": lpath + "/data/train/hiphop/hiphop.00000.au",
    "Jazz": lpath + "/data/train/jazz/jazz.00000.au",
    "Metal": lpath + "/data/train/metal/metal.00000.au",
    "Pop": lpath + "/data/train/pop/pop.00000.au",
    "Reggae": lpath + "/data/train/reggae/reggae.00000.au",
    "Rock": lpath + "/data/train/rock/rock.00000.au",
    #"Test #1": lpath + "/data/test/test.00596.au"
}

plot_these = [2]

num_smaller_plots = len(audio_files)

# Create a subplot for waveforms
fig_waveform, axs_waveform = plt.subplots(num_smaller_plots, 1, figsize=(10, 20))

# Create subplots for spectrograms
if plot_these.count(1):
    fig_spectrogram_linear, axs_spectrogram_linear = plt.subplots(num_smaller_plots, 1, figsize=(10, 20))
if plot_these.count(2):
    fig_spectrogram_log, axs_spectrogram_log = plt.subplots(num_smaller_plots, 1, figsize=(10, 20))
if plot_these.count(3):
    fig_spectrogram_mel, axs_spectrogram_mel = plt.subplots(num_smaller_plots, 1, figsize=(10, 20))
if plot_these.count(4):
    fig_spectrogram_mel_freq, axs_spectrogram_mel_freq = plt.subplots(num_smaller_plots, 1, figsize=(10, 20))





# Process each audio file and plot waveforms and spectrograms
for idx, (genre, path) in enumerate(audio_files.items()):
    audio, sample_rate = librosa.load(path)
    stft_audio = librosa.stft(audio, n_fft=frameSize, hop_length=hopSize)
    y_audio = np.abs(stft_audio) ** 2
    y_log_audio = librosa.power_to_db(y_audio)

    ### Subplot waveform
    plot_waveform(audio, sample_rate, axs_waveform[idx], genre)

    # ### Subplot linear spectrogram
    # Are these worth showing ?
    # plot_stft_spectrogram(y_audio, sample_rate, hopSize, "linear", axs_spectrogram_linear[idx], genre)

    ### Subplot log spectrogram
    plot_stft_spectrogram(y_log_audio, sample_rate, hopSize, "log", axs_spectrogram_log[idx], genre)

    ### Subplot mel spectrogram
    plot_mel_spectrogram(audio, sample_rate, hopSize, "mel", axs_spectrogram_mel[idx], genre)

    ### Subplot frequency spectrum (new)
    plot_sample_spectrum(audio, sample_rate*3, hopSize, "log", axs_spectrogram_mel_freq[idx], genre)



# Display figures for each plot type
fig_waveform.tight_layout()
fig_waveform.show()

# fig_spectrogram_linear.tight_layout()
# fig_spectrogram_linear.show()

fig_spectrogram_log.tight_layout()
fig_spectrogram_log.show()

fig_spectrogram_mel.tight_layout()
fig_spectrogram_mel.show()

fig_spectrogram_mel_freq.tight_layout()
fig_spectrogram_mel_freq.show()



if 1:
    dpi=600 # dots-per-inch is the resolution to save in (300 and 600 are common)

    fig_waveform.savefig(SAVE_PATH + "waveforms" + SAVE_EXT, dpi=dpi)  
    #fig_waveform.close()  # Close the figure after saving to avoid displaying it

    # fig_spectrogram_linear.savefig(SAVE_PATH + "spectrogram_linear" + SAVE_EXT, dpi=dpi) 
    # #fig_spectrogram_linear.close()  # Close the figure after saving to avoid displaying it

    fig_spectrogram_log.savefig(SAVE_PATH + "spectrogram_log" + SAVE_EXT, dpi=dpi) 
    #fig_spectrogram_log.close()  # Close the figure after saving to avoid displaying it

    fig_spectrogram_mel.savefig(SAVE_PATH + "spectrogram_mel" + SAVE_EXT, dpi=dpi) 
    #fig_spectrogram_log.close()  # Close the figure after saving to avoid displaying it
