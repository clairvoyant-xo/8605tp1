import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.fft as fft

def calcular_fft_tramo(t0,tf,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    return fft.fft(audio[n0:nf])

def hacer_vocal_duracion(t,audio,fs):
    repeticiones = (int)  ((t * fs) / len(audio))
    return np.tile(audio,repeticiones)

fs, audio = wav.read("./hh15.wav")

t0 = 0.88
f0 = 180
tpulso = 1/f0

n1 = 1
n2 = 5
n3 = 10
n4 = 15

X1 = calcular_fft_tramo(t0,t0 + n1 * tpulso,fs,audio)
X2 = calcular_fft_tramo(t0,t0 + n2 * tpulso,fs,audio)
X3 = calcular_fft_tramo(t0,t0 + n3 * tpulso,fs,audio)
X4 = calcular_fft_tramo(t0,t0 + n4 * tpulso,fs,audio)

x1 = np.real(fft.ifft(X1))
x2 = np.real(fft.ifft(X2))
x3 = np.real(fft.ifft(X3))
x4 = np.real(fft.ifft(X4))

a1 = hacer_vocal_duracion(1,x1,fs).astype(audio.dtype)
a2 = hacer_vocal_duracion(1,x2,fs).astype(audio.dtype)
a3 = hacer_vocal_duracion(1,x3,fs).astype(audio.dtype)
a4 = hacer_vocal_duracion(1,x4,fs).astype(audio.dtype)

wav.write("./inversa_1_pulso.wav",fs,a1)
wav.write("./inversa_5_pulsos.wav",fs,a2)
wav.write("./inversa_10_pulsos.wav",fs,a3)
wav.write("./inversa_15_pulsos.wav",fs,a4)
