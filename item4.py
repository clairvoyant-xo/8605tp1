import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.fft as fft

def calcular_fft_tramo(t0,tf,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]
    if(len(x) < 4096):
        x = np.append(x,np.zeros(4096-len(x)))

    return fft.fftshift(fft.fft(x)),fft.fftshift(fft.fftfreq(len(x), 1/fs))

fs, audio = wav.read("./audio_propio.wav")
muestras = len(audio)
duracion = muestras / fs

print('Frecuencia de muestreo: ' + str(fs) + ' Hz')
print('Duración: ' + str(duracion) + ' segundos')

plt.figure(1)
t = np.linspace(0,duracion,muestras)
plt.stem(t,audio,markerfmt=' ',basefmt="gray")
plt.title('Señal de audio propia')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()

t0 = 0.23
f0 = 140
tpulso = 1/f0

n = 10

X, f = calcular_fft_tramo(t0,t0 + n * tpulso,fs,audio)

plt.figure(2)
plt.title('Amplitud de FFT de 10 pulsos de vocal [a]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f,np.abs(X),markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
