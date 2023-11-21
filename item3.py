import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.fft as fft

def calcular_fft_tramo(t0,tf,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs)

    return fft.fft(audio[n0:nf])

def hacer_vocal_duracion(t,audio,fs):
    longitud = (int) (t * fs)
    vocal = audio
    while(len(vocal) < longitud):
        vocal = np.append(vocal,audio)
    return vocal

fs, audio = wav.read("./hh15.wav")

t0 = 0.8
tpulso = 0.0071

n1 = 1
n2 = 10
n3 = 20
n4 = 30

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
wav.write("./inversa_10_pulsos.wav",fs,a2)
wav.write("./inversa_20_pulsos.wav",fs,a3)
wav.write("./inversa_30_pulsos.wav",fs,a3)
