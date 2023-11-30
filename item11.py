import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.fft as fft
import scipy.io.wavfile as wav

def cepstrum_tramo_voz(t0,tf,audio,fs):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    c0 = (int) (1/500 * fs)
    cf = (int) (1/50 * fs) + 1

    tramo = audio[n0:nf]

    cepstrum = fft.ifft(np.log(np.abs(fft.fft(tramo))))
    q = np.arange(0,len(tramo) / fs, 1/fs)

    return cepstrum[c0:cf],q[c0:cf]

def cepstrum_tramo_bajo(t0,tf,audio,fs):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    c0 = (int) (0 * fs)
    cf = (int) (1/500 * fs) + 1

    tramo = audio[n0:nf]

    cepstrum = fft.ifft(np.log(np.abs(fft.fft(tramo))))
    q = np.arange(0,len(tramo) / fs, 1/fs)

    return cepstrum[c0:cf],q[c0:cf]    

fs, audio = wav.read("./hh15.wav")

t0 = 0.86
tf = 1

C1, q1 = cepstrum_tramo_voz(t0,tf,audio,fs)
C2, q2 = cepstrum_tramo_bajo(t0,tf,audio,fs)

X, f = fft.fftshift(fft.fft(np.append(C2,np.zeros(2000)))),fft.fftshift(fft.fftfreq(len(C2) + 2000, 1/fs))

plt.figure(1)
plt.title('Cepstrum de fonema [a] original (tramo de voz)')
plt.xlabel('Quefrencia [s]')
plt.ylabel('Re(C)')
plt.stem(q1,np.real(C1),markerfmt=' ',basefmt="gray")
plt.grid()

plt.figure(2)
plt.title('Amplitud de FFT Cepstrum de fonema [a] original (tramo bajo)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.plot(f[len(f)//2:len(f)],20 * np.log10(np.abs(X)[len(X)//2:len(X)]))
plt.grid()

plt.show()
