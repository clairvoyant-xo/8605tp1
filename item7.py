import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn

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

vocal_a = [[830,1400,2890,3930],[110,160,210,230]]
vocal_e = [[500,2000,3130,4150],[80,156,190,220]]
vocal_i = [[330,2765,3740,4366],[70,130,178,200]]
vocal_o = [[546,934,2966,3930],[97,130,185,240]]
vocal_u = [[382,740,2760,3380],[74,150,210,180]]

p = polos_vocal(vocal_a,fs)
sos = forma_sos_filtro(p)

w, h = sgn.sosfreqz(sos,fs=fs)

plt.figure(1)
plt.title('Respuesta en frecuencia del filtro digital correspondiente al fonema [a]')
plt.plot(w, 20 * np.log10(abs(h)))
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid()

plt.figure(2)
plt.title('Diagrama de polos y ceros del filtro digital correspondiente al fonema [a]')
plt.xlabel('Re(Z)')
plt.ylabel('Im(Z)')
plt.scatter(np.real(p),np.imag(p),s=50, marker='x')
plt.grid()

plt.show()
