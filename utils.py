import matplotlib.pyplot as plt       # For plotting spectrograms
from scipy.io import wavfile          # For reading .wav files
from pydub import AudioSegment        # For audio processing (e.g., apply_gain)



def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

rate, data = get_wav_info("raw_data/backgrounds/1.wav")



# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)    
    nfft = 200                              # Window size for the Fast Fourier Transform (FFT).
    fs = 8000                               # Sampling frequencies
    noverlap = 120                          # Overlap between successive windows 
    nchannels = data.ndim                   # number of dimensions of the data array.
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx
    
    
    
    

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)
    
    

