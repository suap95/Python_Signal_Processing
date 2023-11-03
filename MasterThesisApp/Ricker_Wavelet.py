import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

NT = 800
fs = 40e6
Ts = 1/fs
fc = 4.5471e6
phi = -2.6143
sigma = 0
bw_factor = 1.4549e13
bw_factor1 = 1.4549e5
n = np.arange(-NT, NT)
n1 = np.arange(-NT,NT)*Ts
n3 = np.arange(0, 2*NT)#*Ts
n2 = np.arange(2*NT)/fs
n11 = np.arange(-NT,0)*Ts
n12 = np.arange(0,NT)*Ts
s = 0.9

gabor1 = np.exp(-(bw_factor*n11**2 + bw_factor*n11**2*s))*np.cos(2*np.pi*(n11)*fc + phi)
gabor2 = np.exp(-(bw_factor*n12**2 - bw_factor*n12**2*s))*np.cos(2*np.pi*(n12)*fc + phi)
gabor = np.zeros(2*NT)
gabor[0:NT] = gabor1
gabor[NT:2*NT] = gabor2
t = np.linspace(0,1,np.int(fs))
ricker_pulse = np.zeros((10,NT*2))
gaussian_pulse = np.zeros((10,NT*2))
triangle_pulse = np.zeros((10,NT*2))
random_nums = np.zeros(80*20)
random_pulses = np.zeros(320*5)
counter = 0
for i in range(80):
    some_num = np.round((np.random.random()))
    some_num1 = 2*some_num - 1
    random_nums[counter] = some_num1
    random_nums[counter:counter + 20] = random_nums[counter]
    counter = counter + 20

gabor_pulse = np.exp(-(n1)**2*bw_factor)*np.exp(-1j*0.5*(n1))*np.cos(2*np.pi*(n1)*fc + phi)
square_wave = sig.square(2*np.pi*1e6*n1)#*random_nums
#sine_wave = np.round(np.sin(2*np.pi*n1*1e6))
plt.figure(1)
#plt.plot(np.imag(gabor_pulse)/np.max(np.imag(gabor_pulse)))
#plt.plot(np.real(gabor_pulse)/np.max(np.real(gabor_pulse)))
#plt.plot(n11,gabor1)
plt.plot(gabor)
#plt.plot(random_nums)

a_time = 0#0.05 * 10 ** -6
sample_number = a_time/Ts
print(sample_number)
y = np.sinc(bw_factor*n1**2)*np.cos(2*np.pi*(n1-a_time)*fc + phi)
tr = sig.sawtooth(2 * np.pi * 1 * n1,0.5)*np.cos(2*np.pi*(n1-a_time)*fc + phi)
t = np.arange(0, 2*NT, 1)
w = sig.chirp(n2, f0=10e6, t1=3e-5, f1=9e6, method='linear')
w1 = sig.windows.boxcar(NT*2,sym=True)
w2 = w[780:820]*w1[780:820]
w[0:2*NT] = 0
w[780:820] = w2
'''
plt.figure(4)
plt.plot(np.fft.fftshift(np.abs(np.fft.fft(w))),label='Chirp signal (10e6 to 9e6)')
plt.legend()
#plt.plot(n2,w)
plt.figure(5)
plt.plot(w,label='Chirp signal (10e6 to 9e6)')
plt.legend()
mod_signal = w*np.cos(2 * np.pi * fc * (n1 - a_time) + phi)

plt.figure(6)
plt.plot(mod_signal,label='Chirp signal (10e6 to 9e6)')
plt.legend()
'''
for i in range(9):
    sigma = sigma + 0.1
    ricker_pulse[i, :] = 2/((np.sqrt(3*sigma))*np.pi**0.05)*(1-bw_factor*((n1 - a_time)/sigma)**2)*np.exp(-bw_factor*(n1 - a_time)**2/(2*sigma**2))* np.cos(2 * np.pi * fc * (n1 - a_time) + phi)
    gaussian_pulse[i, :] = np.exp(-bw_factor * (((n1 - a_time) / (2 * sigma)) ** 2)) * np.cos(2 * np.pi * fc * (n1 - a_time) + phi)
    #triangle_pulse[i,:] =
#ricker_pulse1 = sig.ricker(1600,4)
'''
plt.figure(1)
plt.plot(sig.hilbert(np.abs(ricker_pulse[0,:])),color='red',label='sigma=0.1')
plt.plot(sig.hilbert(np.abs(ricker_pulse[1,:])),color='blue',label='sigma=0.2')
plt.plot(sig.hilbert(np.abs(ricker_pulse[2,:])),color='green',label='sigma=0.3')
plt.plot(sig.hilbert(np.abs(ricker_pulse[3,:])),color='magenta',label='sigma=0.4')
plt.plot(sig.hilbert(np.abs(ricker_pulse[4,:])),color='yellow',label='sigma=0.5')
plt.plot(sig.hilbert(np.abs(ricker_pulse[5,:])),color='black',label='sigma=0.6')
plt.legend()

plt.figure(2)
plt.plot(sig.hilbert(np.abs(gaussian_pulse[0,:])),color='red',label='sigma=0.1')
plt.plot(sig.hilbert(np.abs(gaussian_pulse[1,:])),color='blue',label='sigma=0.2')
plt.plot(sig.hilbert(np.abs(gaussian_pulse[2,:])),color='green',label='sigma=0.3')
plt.plot(sig.hilbert(np.abs(gaussian_pulse[3,:])),color='magenta',label='sigma=0.4')
plt.plot(sig.hilbert(np.abs(gaussian_pulse[4,:])),color='yellow',label='sigma=0.5')
plt.plot(sig.hilbert(np.abs(gaussian_pulse[5,:])),color='black',label='sigma=0.6')
plt.legend()
'''
t = np.linspace(0,1,1000)
test_pulse = np.exp(-bw_factor*n1**2/(2*2**2))
triangle = sig.sawtooth(2 * np.pi * 2 * n1, 0.5) + 1

fft_spectra_ricker = np.fft.fftshift(np.abs(np.fft.fft(ricker_pulse[3,:])))

plt.figure(3)
plt.plot(fft_spectra_ricker)

peak_index = np.argmax(np.abs(test_pulse))
peak_index1 = np.argmax(np.abs(triangle))

new_spectra = test_pulse*np.exp(1j*2*np.pi*n1*peak_index*fc)
new_spectra1 = np.fft.fftshift(np.abs(np.fft.fft(new_spectra)))
new_spectra2 = triangle*np.exp(1j*2*np.pi*n1*fc*peak_index1)
new_spectra3 = np.fft.fftshift(np.abs(np.fft.fft(new_spectra2)))

#plt.figure(4)
#plt.plot(new_spectra3)

#new_spectra4 = np.fft.fftshift(new_spectra3)
#new_spectra4 = new_spectra3*np.exp(-1j*2*np.pi*t*200)
new_spectra4 = np.fft.ifft(new_spectra3)

original_pulse = new_spectra1*np.exp(-1j*2*np.pi*n1*peak_index*fc)
original_pulse = np.fft.ifft(original_pulse)

#plt.figure(5)
#plt.plot(original_pulse)

#plt.figure(6)
#plt.plot(new_spectra4)
