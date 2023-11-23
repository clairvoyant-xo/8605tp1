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

fs = 16e3
f0 = 200
p0 = 500
tp = 0.4 * 1/f0
tn = 0.16 * 1/f0
k = 200

vocal_a = [[830,1400,2890,3930],[110,160,210,230]]
vocal_e = [[500,2000,3130,4150],[80,156,190,220]]
vocal_i = [[330,2765,3740,4366],[70,130,178,200]]
vocal_o = [[546,934,2966,3930],[97,130,185,240]]
vocal_u = [[382,740,2760,3380],[74,150,210,180]]

p = polos_vocal(vocal_a,fs)
sos = forma_sos_filtro(p)

x = muestrear_pulsos_gloticos(1/f0,p0,tp,tn,fs,k)
vocal = sgn.sosfilt(sos,x)

C, q = cepstrum_rango_vocal(vocal,fs)

plt.figure(1)
plt.title('Cepstrum de pulso glótico filtrado')
plt.xlabel('Quefrencia [s]')
plt.ylabel('Re(C)')
plt.stem(q,np.real(C),markerfmt=' ',basefmt="gray")
plt.grid()

plt.show()
