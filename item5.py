import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgn

def calcular_espectrograma_tramo(t0,tf,ancho,paso,fs,audio):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1
    x = audio[n0:nf]
    f, t, espectro = sgn.spectrogram(x,fs,nperseg=(int) (ancho * fs),noverlap=(int) ((ancho - paso) * fs))
    t = t + t0
    return f, t, espectro

fs, audio = wav.read("./hh15.wav")

t0 = 0.85
tf = 1
ancho = 0.1
paso = 0.001

f, t, espectro = calcular_espectrograma_tramo(t0,tf,ancho,paso,fs,audio)

plt.figure(1)
plt.pcolormesh(t,f,espectro,norm='log')
plt.title('Espectrograma de la señal de audio original')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')
plt.show()
