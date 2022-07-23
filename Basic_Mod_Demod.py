import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, ord):
    return butter(ord, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, ord):
    b, a = butter_lowpass(cutoff, fs, ord)
    y = lfilter(b, a, data)
    return y


taps = 21
n_power = 0.01

data_bits = np.round(np.random.uniform(low=0,high=1,size=10))
t = np.linspace(-0.1,0.1,1900)
n_samples = t.shape[0]
sigma = 0.05
pulse_shape = np.exp(-t**2/(2*sigma**2))
mod_data = np.zeros(t.shape[0]*len(data_bits))

f_s = 100e3
f_sig = 20e3
f_cutoff = 30e3
n_pts = t.shape[0]*len(data_bits)
t1 = np.linspace(0, n_pts / f_s, n_pts)
sine = np.sin(2*np.pi*f_sig*t1)


for i in range(len(data_bits)):
    if data_bits[i] == 0:
        data_bits[i] = -1
    mod_data[t.shape[0]*i:t.shape[0]*(i+1)] = pulse_shape*data_bits[i] + np.random.normal(0, n_power, n_samples)


plt.figure(1)
plt.plot(mod_data)
plt.title('Modulated Data (with gaussian pulse)')
plt.ylabel('Amplitude')
plt.xlabel('Samples')

base_band = sine*mod_data
y = np.abs(np.fft.fft(base_band))

plt.figure(2)
plt.plot(base_band)
plt.title('Base-band Signal')
plt.ylabel('Amplitude')
plt.xlabel('Samples')

plt.figure(3)
plt.plot(np.arange(-n_pts/2,n_pts/2)*(f_s/n_pts), 10.0*np.log10(np.fft.fftshift(y/np.max(y))))
plt.title('Frequency domain (Input baseband signal)')
plt.xlabel('Frequency(in Hz)')
plt.ylabel('Energy (in dB)')

demod_frequency = np.argmax(y[0:np.int(n_pts/2)])*(f_s/n_pts)

regen_sine = np.sin(2*np.pi*demod_frequency*t1)

demod_sig = base_band*regen_sine
y1 = np.fft.fftshift(np.abs(np.fft.fft(demod_sig)))

plt.figure(4)
plt.plot(np.arange(-n_pts/2,n_pts/2)*(f_s/n_pts), 10.0*np.log10(y1/np.max(y1)))
plt.title('Frequency domain (demodulated baseband signal)')
plt.xlabel('Frequency(in Hz)')
plt.ylabel('Energy (in dB)')

filtered_signal = butter_lowpass_filter(demod_sig,f_cutoff,f_s,taps)
y_f = np.fft.fftshift(np.abs(np.fft.fft(filtered_signal)))

plt.figure(5)
plt.plot(np.arange(-n_pts/2,n_pts/2)*(f_s/n_pts), 10.0*np.log10(y_f/np.max(y_f)))
plt.title('Frequency domain (filtered demodulated baseband signal)')
plt.xlabel('Frequency(in Hz)')
plt.ylabel('Energy (in dB)')

plt.figure(6)
plt.plot(filtered_signal)
plt.title('filtered demodulated baseband signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')















