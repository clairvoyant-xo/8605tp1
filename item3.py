import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.fft as fft

def calcular_fft_tramo(t0,tf,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs)

    return fft.fft(audio[n0:nf])

fs, audio = wav.read("./tp/hh15.wav")

t0 = 0.86
tpulso = 0.006

n1 = 15
n2 = 20
n3 = 25

X1 = calcular_fft_tramo(t0,t0 + n1 * tpulso,fs,audio)
X2 = calcular_fft_tramo(t0,t0 + n2 * tpulso,fs,audio)
X3 = calcular_fft_tramo(t0,t0 + n3 * tpulso,fs,audio)

wav.write("./tp/inversa_15_pulsos.wav",fs,np.real(fft.ifft(X1)).astype(audio.dtype))
wav.write("./tp/inversa_20_pulsos.wav",fs,np.real(fft.ifft(X2)).astype(audio.dtype))
wav.write("./tp/inversa_25_pulsos.wav",fs,np.real(fft.ifft(X3)).astype(audio.dtype))
