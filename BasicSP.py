import sys
import numpy as np
import matplotlib.pyplot as plt


def gen_signal(f_s, f_sig, n_fft_pts, n_power):

    '''
    function generates sine, cosine and complex signal with the
    provided parameters.

            Parameters:
                    f_s : Sampling frequency
                    f_sig : Signal frequency
                    n_fft_pts : number of fft points
                    n_power : noise power

            Returns:
                    Returns a complex signal, cosine and sine.
    '''

    t = np.linspace(0, n_fft_pts / f_s, n_fft_pts)
    noise = np.random.normal(0, n_power, n_fft_pts)
    sine = np.sin(2 * np.pi * f_sig * t) + noise
    cosine = np.cos(2 * np.pi * f_sig * t) + noise
    complex_signal = cosine + 1j*sine

    return complex_signal, cosine, sine


def fft_operation(data, f_s, n_fft_pts):

    '''
     function computes fft of given data

             Parameters:
                     data :  Inout data
                     f_s : sampling frequency
                     n_fft_pts : number of fft points
             Returns:
                     Returns a transformed signal, x_axis points
     '''

    out_fft = np.fft.fftshift(np.abs(np.fft.fft(data)))
    x_axis = np.arange(-n_fft_pts/2, n_fft_pts/2)*(f_s/n_fft_pts)
    return out_fft, x_axis


# Plot the original signal and results

samp_freq = 1000
sig_freq = 100
fft_points = 512
noise_variance = 0.05

c_signal, cosine_s, sine_s = gen_signal(samp_freq, sig_freq, fft_points, noise_variance)
fft_out, x_data = fft_operation(c_signal, samp_freq, fft_points)

plt.figure(1)
plt.plot(sine_s)
plt.title('Sine')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.figure(2)
plt.plot(cosine_s)
plt.title('Cosine')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.figure(3)
plt.plot(x_data, 10.0*np.log10(fft_out/np.max(fft_out)))
plt.xlabel('Frequency (in Hz)')
plt.ylabel('Energy (in dB)')
plt.title('Frequency domain')



