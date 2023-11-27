import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

def u(n):
    return np.heaviside(n,1)

def pulso_glotico(t,p0,tp,tn):
    return (0.5 * p0) * (1 - np.cos(t/tp * np.pi)) * (u(t) - u(t - tp)) + p0 * np.cos((t-tp)/tn * np.pi/2) * (u(t - tp) - u(t - (tp+tn)))

def muestrear_pulsos_gloticos(t0,p0,tp,tn,fs,n):
    longitud_pulso = (int) (t0 * fs)
    x = np.empty(longitud_pulso)
    for i in range(0,longitud_pulso):
        x[i] = pulso_glotico(i/fs,p0,tp,tn)
    return np.tile(x,n)

def calcular_fft_pulsos(fs,x):
    return fft.fftshift(fft.fft(x)),fft.fftshift(fft.fftfreq(len(x), 1/fs))

f0 = 200
p0 = 200
tp = 0.4 * 1/f0
tn = 0.16 * 1/f0
k = 20
t0 = 0
tf = 0.1
fs = 16e3

x = muestrear_pulsos_gloticos(1/f0,p0,tp,tn,fs,k)
t = np.arange(0,len(x),1) / fs

plt.figure(1)
plt.title('Pulsos glóticos de entrada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.stem(t,x,markerfmt=' ',basefmt="gray")
plt.grid()

X, f = calcular_fft_pulsos(fs,x)

plt.figure(2)
plt.title('Amplitud de FFT de 20 pulsos glóticos')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f,np.abs(X),markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
