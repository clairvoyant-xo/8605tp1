import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sgn

def calcular_espectrograma_audio(fs,audio,ventana,paso):
    return sgn.spectrogram(audio,fs,nperseg=(int) (ventana * fs),noverlap=(int) ((ventana-paso) * fs))

fs, audio = wav.read("./hh15.wav")

ventana = 0.1
paso = 0.01

f, t, espectro = calcular_espectrograma_audio(fs,audio,ventana,paso)

plt.figure(1)
plt.pcolormesh(t,f,espectro,norm='log')
plt.title('Espectrograma de la señal original')
plt.ylabel('Frecuencia [Hz]')
plt.xlabel('Tiempo [s]')

plt.show()
