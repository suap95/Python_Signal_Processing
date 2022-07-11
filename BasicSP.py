import sys
import numpy as np
import matplotlib.pyplot as plt

f_s = 1000   # Sampling frequency
f_sig = 100  # Signal frequency
n_fft_pts = 64
t = np.linspace(0, n_fft_pts/f_s, n_fft_pts)

# Generate some signal

sine_wave = np.sin(2*np.pi*f_sig*t)

# Frequency domain transformation

fft_out = np.fft.fftshift(np.abs(np.fft.fft(sine_wave)))

# Plot the original signal and results

plt.figure(1)
plt.plot(sine_wave)
plt.title('Original Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.figure(2)
plt.plot(np.arange(-n_fft_pts/2,n_fft_pts/2)*(f_s/n_fft_pts),fft_out)
plt.xlabel('Frequency (in Hz)')
plt.ylabel('Energy')
plt.title('Frequency domain')



