import sys
import numpy as np
import matplotlib.pyplot as plt

f_s = 1000   # Sampling frequency
f_sig = 100  # Signal frequency
n_fft_pts = 64
t = np.linspace(0, n_fft_pts/f_s, n_fft_pts)
noise = np.random.normal(0,1,n_fft_pts)

# Generate some signal

sine = np.sin(2*np.pi*f_sig*t) + noise
cosine = np.cos(2*np.pi*f_sig*t) + noise

complex_signal = cosine+1j*sine

# Frequency domain transformation

fft_out = np.fft.fftshift(np.abs(np.fft.fft(complex_signal)))

# Plot the original signal and results

plt.figure(1)
plt.plot(sine)
plt.title('Original Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.figure(2)
plt.plot(np.arange(-n_fft_pts/2,n_fft_pts/2)*(f_s/n_fft_pts),fft_out)
plt.xlabel('Frequency (in Hz)')
plt.ylabel('Energy')
plt.title('Frequency domain')



