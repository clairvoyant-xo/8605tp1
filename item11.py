import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.fft as fft
import scipy.io.wavfile as wav

def cepstrum_rango_vocal(x,fs):
    n0 = (int) (1/500 * fs)
    nf = (int) (1/50 * fs) + 1

    cepstrum = fft.ifft(np.log(np.abs(fft.fft(x))))
    q = np.arange(0,len(x) / fs, 1/fs)

    return cepstrum[n0:nf],q[n0:nf]

fs, audio = wav.read("./hh15.wav")

C, q = cepstrum_rango_vocal(audio,fs)

plt.figure(1)
plt.title('Cepstrum del audio original')
plt.xlabel('Quefrencia [s]')
plt.ylabel('Re(C)')
plt.stem(q,np.real(C),markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
