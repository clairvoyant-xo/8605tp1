import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

def u(n):
    return np.heaviside(n,1)

def pulso_glotico(t,p0,tp,tn):
    return (0.5 * p0) * (1 - np.cos(t/tp * np.pi)) * (u(t) - u(t - tp)) + p0 * np.cos((t-tp)/tn * np.pi/2) * (u(t - tp) - u(t - (tp+tn)))

def tren_pulsos_gloticos(t,p0,tp,tn,n,f):
    suma = 0
    for i in range(0,n):
        suma += pulso_glotico(t - i/f,p0,tp,tn)
    return suma

def muestrear_pulsos_gloticos(p0,tp,tn,n,f,t0,tf,fs):
    muestras = (int) ((tf - t0) * fs)
    x = np.zeros(muestras)
    for i in range(0,muestras):
        x[i] = tren_pulsos_gloticos(t0 + i/fs,p0,tp,tn,n,f)
    return x,muestras

def calcular_fft_pulsos(fs,x):
    longitud_muestra = len(x)
    X = fft.fft(x)
    f = np.arange(0,longitud_muestra,1) * fs / (longitud_muestra) - fs / 2

    return fft.fftshift(np.abs(X)),f

f0 = 200
p0 = 1
tp = 0.4 * 1/f0
tn = 0.16 * 1/f0
k = 20
t0 = -0.05
tf = 0.15
fs = 16e3

t = np.linspace(t0,tf,1000)

plt.figure(1)
plt.plot(t,tren_pulsos_gloticos(t,p0,tp,tn,k,f0))
plt.title('Gráfico de pulsos glóticos')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()

x, n = muestrear_pulsos_gloticos(p0,tp,tn,k,f0,t0,tf,fs)
n = np.arange(0,n,1)

plt.figure(2)
plt.title('Muestreo de pulsos glóticos')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.stem(n,x,markerfmt=' ',basefmt="gray")
plt.grid()

X, f = calcular_fft_pulsos(fs,x)

plt.figure(3)
plt.title('Amplitud de FFT de pulsos glóticos')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f,X,markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
