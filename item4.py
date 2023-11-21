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

fs, audio = wav.read("./audio_propio.wav")
muestras = len(audio)
duracion = muestras / fs

plt.figure(1)
t = np.linspace(0,duracion,muestras)
plt.stem(t,audio,markerfmt=' ',basefmt="gray")
plt.title('Señal de audio propia')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()

t0 = 1.92
tpulso = 0.008

n = 10

X, f = calcular_fft_tramo(t0,t0 + n * tpulso,fs,audio)

plt.figure(2)
plt.title('Amplitud de FFT de 10 pulsos de vocal [a]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f,X,markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
