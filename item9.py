import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.fft as fft
import scipy.io.wavfile as wav

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

def p(F,B,fs):
    return np.exp(-2 * np.pi * B/fs) * np.exp(2j * np.pi * F/fs)

def polos_vocal(vocal,fs):
    polos = np.empty(8,dtype=np.clongdouble)
    for i in range(0,4):
        polos[2*i] = p(vocal[0][i],vocal[1][i],fs)
        polos[2*i+1] = np.conj(polos[2*i])
    return polos

def forma_sos_filtro(polos):
    sos = np.empty([4,6])
    for i in range(0,4):
        sos[i][0] = 1
        sos[i][1] = 0
        sos[i][2] = 0
        sos[i][3] = 1
        sos[i][4] = -2 * np.real(polos[2*i])
        sos[i][5] = np.abs(polos[2*i])**2
    return sos

def generar_vocal(vocal,fs,f0,p0,tp,tn,t):
    p = polos_vocal(vocal,fs)
    sos = forma_sos_filtro(p)
    x = muestrear_pulsos_gloticos(1/f0,p0,tp,tn,fs,(int) (t * f0))

    return sgn.sosfilt(sos,x), np.arange(0,len(x),1) / fs

fs = 16e3
f0 = 200
p0 = 200
tp = 0.4 * 1/f0
tn = 0.16 * 1/f0

vocal_a = [[830,1400,2890,3930],[110,160,210,230]]
vocal_e = [[500,2000,3130,4150],[80,156,190,220]]
vocal_i = [[330,2765,3740,4366],[70,130,178,200]]
vocal_o = [[546,934,2966,3930],[97,130,185,240]]
vocal_u = [[382,740,2760,3380],[74,150,210,180]]

vocal, t = generar_vocal(vocal_a,fs,f0,p0,tp,tn,1)

a0,b0 = sgn.iirnotch(f0,30,fs=fs)
vocal_sin_f0 = sgn.lfilter(a0,b0,vocal)

fs, audio = wav.read("./hh15.wav")
wav.write("./vocal_sin_f0.wav",fs,vocal_sin_f0.astype(audio.dtype))

plt.figure(1)
plt.title('Simulación de fonema [a]')
plt.xlabel('Tiempo [t]')
plt.ylabel('Amplitud')
plt.stem(t,vocal,markerfmt=' ',basefmt="gray")
plt.stem(t,vocal_sin_f0,'r',markerfmt=' ',basefmt="gray")
plt.legend(['Fonema original','Fonema filtrado'])
plt.grid()

X, f = calcular_fft_pulsos(fs,vocal)
S, f = calcular_fft_pulsos(fs,vocal_sin_f0)

plt.figure(2)
plt.title('Amplitud de FFT de simulación de fonema [a]')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f,np.abs(X),markerfmt=' ',basefmt="gray")
plt.stem(f,np.abs(S),'r',markerfmt=' ',basefmt="gray")
plt.legend(['Fonema original','Fonema filtrado'])
plt.grid()

plt.show()
