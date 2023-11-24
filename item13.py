import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.io.wavfile as wav

def psola_expandir(t0,tf,audio,fs,padding,n):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]
    x = np.append(x,np.zeros(padding))

    return np.tile(x,n), np.arange(0,len(x) * n,1)

def psola_comprimir(t0,tf,audio,fs,overlap,n):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]

    for i in range(0,overlap):
        x[i] += x[-(overlap - i)]

    x = x[:-overlap]

    return np.tile(x,n), np.arange(0,len(x) * n,1)

def calcular_fft_pulsos(fs,x):
    return fft.fftshift(fft.fft(x)),fft.fftshift(fft.fftfreq(len(x), 1/fs))

fs, audio = wav.read("./hh15.wav")

t0 = 0.88
f0 = 180

padding_10 = 8
padding_20 = 16
padding_30 = 24

vocal_psola, n = psola_comprimir(t0,t0 + 1/f0,audio,fs,padding_30,200)

wav.write("./vocal_psola.wav",fs,vocal_psola.astype(audio.dtype))

plt.figure(1)
plt.title('Pulso glótico filtrado con PSOLA')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.stem(n,vocal_psola,markerfmt=' ',basefmt="gray")
plt.grid()

X, f = calcular_fft_pulsos(fs,vocal_psola)

plt.figure(2)
plt.title('Amplitud de FFT del pulso glótico filtrado con PSOLA')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f,np.abs(X),markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
