import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.io.wavfile as wav

def psola_expandir(t0,tf,audio,fs,padding,n):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]
    x = np.append(x,np.zeros(padding))

    return np.tile(x,n), np.arange(0,len(x) * n,1) / fs

def psola_comprimir(t0,tf,audio,fs,overlap,n):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]

    for i in range(0,overlap):
        x[i] += x[-(overlap - i)]

    x = x[:-overlap]

    return np.tile(x,n), np.arange(0,len(x) * n,1) / fs

def calcular_fft_pulsos(fs,x):
    return fft.fftshift(fft.fft(x)),fft.fftshift(fft.fftfreq(len(x), 1/fs))

def calcular_fft_tramo(t0,tf,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]
    if(len(x) < 4096):
        x = np.append(x,np.zeros(4096-len(x)))

    return fft.fftshift(fft.fft(x)),fft.fftshift(fft.fftfreq(len(x), 1/fs))

fs, audio = wav.read("./hh15.wav")

t0 = 0.88
f0 = 175

padding_10 = 8
padding_20 = 16
padding_30 = 24

vocal_psola_c, tc = psola_comprimir(t0,t0 + 1/f0,audio,fs,padding_10,14)
vocal_psola_e, te = psola_expandir(t0,t0 + 1/f0,audio,fs,padding_10,14)

wav.write("./vocal_psola_c.wav",fs,vocal_psola_c.astype(audio.dtype))
wav.write("./vocal_psola_e.wav",fs,vocal_psola_e.astype(audio.dtype))

X, f1 = calcular_fft_tramo(t0,t0 + 10/f0,fs,audio)
Pc, f2 = calcular_fft_pulsos(fs,vocal_psola_c)
Pe, f3 = calcular_fft_pulsos(fs,vocal_psola_e)

plt.figure(1)
plt.title('Amplitud de FFT del fonema [a] original procesado con PSOLA')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f1,np.abs(X),markerfmt=' ',basefmt="gray")
plt.stem(f2,np.abs(Pc),'r',markerfmt=' ',basefmt="gray")
plt.stem(f3,np.abs(Pe),'g',markerfmt=' ',basefmt="gray")
plt.legend(['Fonema original','Fonema expandido 10%','Fonema comprimido 10%'])
plt.grid()

plt.show()
