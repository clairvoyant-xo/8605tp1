import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.fft as fft
import scipy.io.wavfile as wav

def cepstrum_tramo(t0,tf,audio,fs):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    c0 = (int) (1/500 * fs)
    cf = (int) (1/50 * fs) + 1

    tramo = audio[n0:nf]

    cepstrum = fft.ifft(np.log(np.abs(fft.fft(tramo))))
    q = np.arange(0,len(tramo) / fs, 1/fs)

    return cepstrum[c0:cf],q[c0:cf]

fs, audio = wav.read("./hh15.wav")

t0 = 0.86
tf = 1

C, q = cepstrum_tramo(t0,tf,audio,fs)

plt.figure(1)
plt.title('Cepstrum del audio original')
plt.xlabel('Quefrencia [s]')
plt.ylabel('Re(C)')
plt.stem(q,np.real(C),markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
