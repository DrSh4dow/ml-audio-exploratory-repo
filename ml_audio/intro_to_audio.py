import librosa
import matplotlib.pyplot as plt
import numpy as np

array, sampling_rate = librosa.load(librosa.example("trumpet"))


dft_input = array[:4096]

# calculate the DFT
window = np.hanning(len(dft_input))
windowed_input = dft_input * window
dft = np.fft.rfft(windowed_input)

# get the amplitude spectrum in decibels
amplitude = np.abs(dft)
amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)

# get the frequency bins
frequency = librosa.fft_frequencies(sr=sampling_rate, n_fft=len(dft_input))

# plt.figure().set_figwidth(12)
# plt.plot(frequency, amplitude_db)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Amplitude (dB)")
# plt.xscale("log")
#
# plt.show()

# D = librosa.stft(array)
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#
# plt.figure().set_figwidth(12)
# librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
# plt.colorbar()
#
# plt.show()

S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)

S_db = librosa.power_to_db(S, ref=np.max)

plt.figure().set_figwidth(12)

librosa.display.specshow(S_db, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()
plt.show()
