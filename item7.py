import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn

def p(F,B,fs):
    return np.exp(-2 * np.pi * B/fs) * np.exp(2j * np.pi * F/fs)

def polos_vocal(vocal,fs):
    polos = np.empty(8,dtype=np.cdouble)
    for i in range(0,3):
        polos[2*i] = p(vocal[0][i],vocal[1][i],fs)
        polos[2*i+1] = np.conj(polos[2*i])
    return polos

fs = 16e3

vocal_a = [[830,1400,2890,3930],[110,160,210,230]]
vocal_e = [[500,2000,3130,4150],[80,156,190,220]]
vocal_i = [[330,2765,3740,4366],[70,130,178,200]]
vocal_o = [[546,934,2966,3930],[97,130,185,240]]
vocal_u = [[382,740,2760,3380],[74,150,210,180]]

z, p, k = [], polos_vocal(vocal_e,fs), 1

w, h = sgn.freqz_zpk(z, p, k)
w = w * 0.5 * fs/np.pi

fig = plt.figure(1)

ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title('Respuesta en frecuencia del filtro digital')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitud [dB]', color='b')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.grid(True)

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot(w, angles, 'g')
ax2.set_ylabel('Fase [rad]', color='g')
plt.axis('tight')

plt.show()
