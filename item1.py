import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

fs, audio = wav.read("./tp/hh15.wav")
muestras = len(audio)
duracion = muestras / fs

print('Frecuencia de muestreo: ' + str(fs) + ' Hz')
print('Duración: ' + str(duracion) + ' segundos')

t = np.linspace(0,duracion,muestras)
plt.stem(t,audio,markerfmt=' ',basefmt="gray")
plt.title('Señal de audio original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.show()
