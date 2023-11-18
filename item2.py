import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.fft as fft

def calcular_fft_tramo(t0,tf,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs)

    x = audio[n0:nf]
    while(len(x) < 4096):
        x = np.append(x,np.array([0]))
    X = fft.fft(x)

    longitud_muestra = len(x)
    f = np.arange(0,longitud_muestra,1) * fs / (longitud_muestra) - fs / 2

    return fft.fftshift(np.abs(X)),f

fs, audio = wav.read("./tp/hh15.wav")

t0 = 0.86
tpulso = 0.006

n1 = 1
n2 = 5
n3 = 20

X1, f1 = calcular_fft_tramo(t0,t0 + n1 * tpulso,fs,audio)
X2, f2 = calcular_fft_tramo(t0,t0 + n2 * tpulso,fs,audio)
X3, f3 = calcular_fft_tramo(t0,t0 + n3 * tpulso,fs,audio)

plt.figure(1)
plt.title('Amplitud de FFT de un pulso de vocal [a]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f1,X1,markerfmt=' ',basefmt="gray")
plt.grid()

plt.figure(2)
plt.title('Amplitud de FFT de 5 pulsos de vocal [a]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f2,X2,markerfmt=' ',basefmt="gray")
plt.grid()

plt.figure(3)
plt.title('Amplitud de FFT de 20 pulsos de vocal [a]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f3,X3,markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
