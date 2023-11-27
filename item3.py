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
    return np.tile(audio,repeticiones), np.arange(0,len(audio) * repeticiones / fs,1/fs)

fs, audio = wav.read("./hh15.wav")

t0 = 0.88
f0 = 175
tpulso = 1/f0

n1 = 1
n2 = 5
n3 = 10

X1 = calcular_fft_tramo(t0,t0 + n1 * tpulso,fs,audio)
X2 = calcular_fft_tramo(t0,t0 + n2 * tpulso,fs,audio)
X3 = calcular_fft_tramo(t0,t0 + n3 * tpulso,fs,audio)

x1 = np.real(fft.ifft(X1))
x2 = np.real(fft.ifft(X2))
x3 = np.real(fft.ifft(X3))

a1, t1 = hacer_vocal_duracion(1,x1,fs)
a2, t2 = hacer_vocal_duracion(1,x2,fs)
a3, t3 = hacer_vocal_duracion(1,x3,fs)

wav.write("./inversa_1_pulso.wav",fs,a1.astype(audio.dtype))
wav.write("./inversa_5_pulsos.wav",fs,a2.astype(audio.dtype))
wav.write("./inversa_10_pulsos.wav",fs,a3.astype(audio.dtype))

plt.figure(1)
plt.title('Reconstrucción a partir de FFT de un pulso de vocal [a]')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.stem(t1,a1,markerfmt=' ',basefmt="gray")
plt.grid()

plt.figure(2)
plt.title('Reconstrucción a partir de FFT de 5 pulsos de vocal [a]')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.stem(t2,a2,markerfmt=' ',basefmt="gray")
plt.grid()

plt.figure(3)
plt.title('Reconstrucción a partir de FFT de 10 pulsos de vocal [a]')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.stem(t3,a3,markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
