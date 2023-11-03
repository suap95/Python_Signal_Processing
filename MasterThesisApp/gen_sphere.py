# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:03:05 2021

@author: perezmejia
"""

#from Sphere import Sphere
from Sphere_test import Sphere
from Rectangle import Rectangle
from Scenario import Scenario
#from Plane_original import Plane
from Plane import Plane
from time import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as ssig
from scipy.ndimage.interpolation import shift

'''
This script generates data by simulating what a transducer would measure. The transducer is defined by a grid on which rays are generated. The rays are cast and traced in a scenario. The scenario contains objects from which rays are reflected back and measured at the transducer. Amplitudes are calculated following a cosine law that depends on the AOA at the reflective object, as well as a 2D Gaussian directivity function.

The azimuth is measured from the x-axis and controls the transducer's aim on the x-y plane. The coelevation is measured from the z axis. As an example, coelevation = 0 yields a normal vector [0,0,1] REGARDLESS of the azimuth. This is standard for spherical coordinates.

The parameters and everything else are exemplified throughout the script.
'''

#%% SCENARIO PARAMETERS: DON'T TOUCH THESE!
NT = 800 #number of time domain samples
#c = 1481 #speed of sound in water [m/s] (taken from wikipedia :P)
c = 6000 #speed of sound for synthetic experiments [m/s]
no_of_symbols = 32 # for ofdm transmit signal
fs = 40e6 #sampling frequency [Hz]
fc = 4.5471e6 #pulse center frequency [Hz]
symbol_time = no_of_symbols/fs
delta_f = 1/symbol_time
bw_factor = 1.4549e13 #pulse bandwidth factor [Hz^2]
phi = -2.6143 #pulse phase in radians

x0 = 0 #horizontal reference for reflecting plane location
z0 = 25e-3 #vertical reference for reflecting plane location

center = np.array([x0*1e-3, 0*1e-3, z0])
radius = 5e-3
sph = Sphere(center, radius)

#put the reflecting rectangles in an array
objects = np.array([sph])

#and put everything into a scenario object
scenario = Scenario(objects, c, NT, fs, bw_factor, fc, phi,no_of_symbols,symbol_time)

#now, the transducer parameters that will remain fixed throughout the sims:
opening = 10*np.pi/180 #opening angle in radians
nv = 181 #number of vertical gridpoints for the rays
nh = 181 #number of horizontal gridpoints for the rays
vres = 3e-3/nv #ray grid vertical resolution in [m] (set to 0.2mm)
hres = 1.5e-3/nh #ray grid horizontal res in [m] (set to 0.1mm)
distance = np.sqrt(3)*(nh-1)/2*hres #distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.

#%% EXPERIMENT: HERE'S WHERE STUFF SHOULD BE CODED AND TESTED

plt.close('all')

#this variable will let you see what the transducer is looking at as an image. you can get an idea of where to aim it that way. set it to false when you no longer need it (e.g. once the code has been made automatic).
debug=True

#this is the center of the transducer plane. the coordinates are given in [m], so leave the 1e-3 term untouched.
p = np.array([15,10,0])*1e-3

#now, we aim the transducer:
azimuth = 0*np.pi/180 #azimuth in [radians]
coelevation = 5*np.pi/180 #coelevation in [radians]

#create a transducer with these parameters, and some of the previously defined quantities that shouldn't be modified:
transducer = Plane(p, distance, azimuth, coelevation, vres, hres, nv, nh, opening)
transducer.prepareImagingPlane() #this ALWAYS has to be called right after creating a transducer!
time_axis = np.arange(NT)/fs
#NOTE: if any of the parameters p, distance, azimuth, or coelevation have to be changed, a new transducer has to be created and prepareImagingPlane() has to be called again. this is because several parameters have to be computed again internally.

#now, use the transducer and the scenario to generate an A-scan:
t2 = time_axis[NT-1]
y1 = np.array([1 if np.abs(n)<=t2/2+(symbol_time/2) and np.abs(n) >=t2/2-(symbol_time/2) else 0 for n in time_axis])
start = time()

Ascan5 = transducer.insonify(scenario,4,1)  ### ofdm signal
Ascan = transducer.insonify(scenario,3,1)  ### gabor pulse
Ascan1 = transducer.insonify(scenario,1,1)  ### ricker pulse
Ascan2 = transducer.insonify(scenario,2,1)  ### gaussian pulse
Ascan4 = transducer.insonify(scenario,0,1)  ### sinc pulse

auto_corr1 = np.abs(ssig.hilbert(ssig.correlate(Ascan,Ascan,mode='same')))
auto_corr2 = np.abs(ssig.hilbert(ssig.correlate(Ascan1,Ascan1,mode='same')))
auto_corr3 = np.abs(ssig.hilbert(ssig.correlate(Ascan2,Ascan2,mode='same')))
auto_corr4 = np.abs(ssig.hilbert(ssig.correlate(Ascan4,Ascan4,mode='same')))
auto_corr5 = np.abs(ssig.hilbert(ssig.correlate(Ascan5,Ascan5,mode='same')))



'''
fft_shift_result = np.zeros((NT,t1.shape[1]))
signal = np.array([1 if n >= 0 - (symbol_time / 2) and n <= 0 + (symbol_time / 2) else 0 for n in t1[:, 0]])
fft_out = np.fft.fftshift(np.abs(np.fft.ifft(signal)))

for i in range(t1.shape[1]):
	t3 = t1[0, i]
	samples = t3*fs
	samples = NT/2-np.abs(samples)
	print(samples,t3)
	fft_shift_result[:,i] = shift(fft_out,np.round(-samples),cval=0)

mod_signal = 100*fft_shift_result*np.cos(2*np.pi*t1*fc+phi)
'''
'''
y2 = np.zeros((NT,t1.shape[1]))
for i in range(t1.shape[1]):
	t2 = np.abs(t1[0, i] - t1[NT - 1, i])
	#print(t1[0, i], t1[NT - 1, i], t2)
	#y2[:,i] = np.array([1 if n>=t2/2-np.abs(t1[0,i])-0.5e-6 and n<=t2/2-np.abs(t1[0,i])+0.5e-6 else 0 for n in t1[:,0]]) #and n+np.abs(t1[0,0])>=t2/2+0.5e-5
	signal = np.array([1 if n>=0 - (symbol_time/2) and n <= 0 + (symbol_time/2) else 0 for n in t1[:, 0]])
	#signal1 = signal*np.exp(1j*2*np.pi*(delta_f+50e3)*time_axis)
	signal2 = np.fft.fftshift(np.abs(np.fft.ifft(signal)))
	signal3 = np.array([1 if n>=0 - (symbol_time/2) and n <= 0 + (symbol_time/2) else 0 for n in t1[:, i]])
	y2[:, i] = signal2*signal3
'''
'''
t2 = np.abs(t1[0,10] - t1[NT-1,10])
print(t1[0,10],t1[NT-1,10],t2)
y3 = np.array([1 if n>=t2/2-np.abs(t1[0,10])-0.5e-5 and n<=t2/2-np.abs(t1[0,10])+0.5e-5 else 0 for n in t1[:,0]])
t2 = np.abs(t1[0,500] - t1[NT-1,500])
print(t1[0,500],t1[NT-1,500],t2)
y4 = np.array([1 if n>=t2/2-np.abs(t1[0,500])-0.5e-5 and n<=t2/2-np.abs(t1[0,500])+0.5e-5 else 0 for n in t1[:,0]])
t2 = np.abs(t1[0,1000] - t1[NT-1,1000])
print(t1[0,1000],t1[NT-1,1000],t2)
y5 = np.array([1 if n>=t2/2-np.abs(t1[0,1000])-0.5e-5 and n<=t2/2-np.abs(t1[0,1000])+0.5e-5 else 0 for n in t1[:,0]])
'''
'''
y3 = 5*y2*np.cos(2*np.pi*fc*t1+phi)*temp_image
y4 = np.sum(y3,axis=1)/(nv+nh)
'''
w = ssig.windows.blackman(NT,sym=True)
plt.figure(3)
plt.plot(w)
print(time() - start)

#let's check the plot out:

'''
plt.figure(1)
plt.plot(Ascan*w,label='sinc pulse')
plt.plot(Ascan1*w,label='ricker pulse')
plt.plot(Ascan2*w,label='gaussian pulse')
plt.legend()
'''
plt.figure(2)
plt.plot(Ascan,label='gabor pulse')
plt.plot(Ascan4,label='sinc pulse')
plt.plot(Ascan1,label='ricker pulse')
plt.plot(Ascan2,label='gaussian pulse')
plt.plot(Ascan5,label='ofdm signal')
plt.legend()

plt.figure(4)
plt.plot(Ascan5,label='ofdm signal')
plt.xlabel('Samples')
plt.ylabel('Ampltitude')
plt.title('Ascan (with ofdm signal)')
plt.legend()

plt.figure(5)
plt.plot(Ascan4,label='sinc pulse')
plt.xlabel('Samples')
plt.ylabel('Ampltitude')
plt.title('Ascan (with sinc pulse)')
plt.legend()

plt.figure(6)
plt.plot(Ascan2,label='gaussian pulse')
plt.xlabel('Samples')
plt.ylabel('Ampltitude')
plt.title('Ascan (with gaussian pulse)')
plt.legend()

plt.figure(7)
plt.plot(Ascan1,label='ricker pulse')
plt.xlabel('Samples')
plt.ylabel('Ampltitude')
plt.title('Ascan (with ricker pulse)')
plt.legend()

plt.figure(8)
plt.plot(Ascan,label='gabor pulse')
plt.xlabel('Samples')
plt.ylabel('Ampltitude')
plt.title('Ascan (with gabor pulse)')
plt.legend()

plt.figure(9)
plt.plot(auto_corr1/np.max(auto_corr1),label='gabor pulse')
plt.xlabel('Samples')
plt.ylabel('Magnitude of correlation')
plt.title('AutoCorrelation')
plt.legend()

plt.figure(10)
plt.plot(auto_corr2/np.max(auto_corr2),label='ricker pulse')
plt.xlabel('Samples')
plt.ylabel('Magnitude of correlation')
plt.title('AutoCorrelation')
plt.legend()

plt.figure(11)
plt.plot(auto_corr3/np.max(auto_corr3),label='gaussian pulse')
plt.xlabel('Samples')
plt.ylabel('Magnitude of correlation')
plt.title('AutoCorrelation')
plt.legend()

plt.figure(12)
plt.plot(auto_corr4/np.max(auto_corr4),label='sinc pulse')
plt.xlabel('Samples')
plt.ylabel('Magnitude of correlation')
plt.title('AutoCorrelation')
plt.legend()

plt.figure(13)
plt.plot(auto_corr5/np.max(auto_corr5),label='ofdm signal')
plt.xlabel('Samples')
plt.ylabel('Magnitude of correlation')
plt.title('AutoCorrelation')
plt.legend()
# plt.plot(np.fft.fftshift(np.abs(np.fft.fft(y2[:,1]))))
# plt.plot(np.fft.fftshift(np.abs(np.fft.fft(y2[:,2]))))
# plt.plot(np.fft.fftshift(np.abs(np.fft.fft(y2[:,10]))))

if debug:
	plt.figure()
	plt.imshow(transducer.image)
