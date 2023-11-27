import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
import scipy.fft as fft
import scipy.io.wavfile as wav

def psola_expandir(t0,tf,audio,fs,padding,n):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]
    x = np.append(x,np.zeros(padding))

    return np.tile(x,n), np.arange(0,len(x) * n,1) / fs

def psola_comprimir(t0,tf,audio,fs,overlap,n):
    n0 = (int) (t0 * fs)
    nf = (int) (tf * fs) + 1

    x = audio[n0:nf]

    for i in range(0,overlap):
        x[i] += x[-(overlap - i)]

    x = x[:-overlap]

    return np.tile(x,n), np.arange(0,len(x) * n,1) / fs

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

vocal, t1 = generar_vocal(vocal_a,fs,f0,p0,tp,tn,1)

padding_10 = 8

vocal_psola_c, tc = psola_comprimir(0,1/f0,vocal,fs,padding_10,200)
vocal_psola_e, te = psola_expandir(0,1/f0,vocal,fs,padding_10,200)

fs, audio = wav.read("./hh15.wav")
wav.write("./vocal_psola_c.wav",fs,vocal_psola_c.astype(audio.dtype))
wav.write("./vocal_psola_e.wav",fs,vocal_psola_e.astype(audio.dtype))

X, f1 = calcular_fft_pulsos(fs,vocal)
Pc, f2 = calcular_fft_pulsos(fs,vocal_psola_c)
Pe, f3 = calcular_fft_pulsos(fs,vocal_psola_e)

plt.figure(1)
plt.title('Amplitud de FFT del fonema [a] sintetizado procesado con PSOLA')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud')
plt.stem(f1,np.abs(X),markerfmt=' ',basefmt="gray")
plt.stem(f2,np.abs(Pc),'r',markerfmt=' ',basefmt="gray")
plt.stem(f3,np.abs(Pe),'g',markerfmt=' ',basefmt="gray")
plt.legend(['Fonema original','Fonema expandido 10%','Fonema comprimido 10%'])
plt.grid()

plt.show()
