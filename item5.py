import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgn

def calcular_espectrograma_tramo(t0,tf,ancho,paso,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs)
    x = audio[n0:nf]
    return sgn.spectrogram(x,fs,nperseg=(int) (ancho * fs),noverlap=(int) ((ancho - paso) * fs))

fs, audio = wav.read("./tp/hh15.wav")

f, t, espectro = calcular_espectrograma_tramo(0.85,1,0.1,0.01,fs,audio)

plt.figure(1)
plt.pcolormesh(t,f,espectro,norm='log')
plt.title('Espectrograma de la señal de audio original')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.show()
