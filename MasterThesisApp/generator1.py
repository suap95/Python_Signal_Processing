#from Rectangle import Rectangle
#from Plane_original import Plane
from Rectangle_test import Rectangle_test
from Scenario import Scenario
from Plane import Plane
from time import time
from timeit import default_timer as timer
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.linalg import get_blas_funcs
import tikzplotlib

'''
This script generates data by simulating what a transducer would measure. The transducer is defined by a grid on which rays are generated. The rays are cast and traced in a scenario. The scenario contains objects from which rays are reflected back and measured at the transducer. Amplitudes are calculated following a cosine law that depends on the AOA at the reflective object, as well as a 2D Gaussian directivity function.

The azimuth is measured from the x-axis and controls the transducer's aim on the x-y plane. The coelevation is measured from the z axis. As an example, coelevation = 0 yields a normal vector [0,0,1] REGARDLESS of the azimuth. This is standard for spherical coordinates.

The parameters and everything else are exemplified throughout the script.
'''

#%% SCENARIO PARAMETERS: DON'T TOUCH THESE!
NT = 800 #number of time domain samples
#c = 1481 #speed of sound in water [m/s] (taken from wikipedia :P)
c = 6000 #speed of sound for synthetic experiments [m/s] 
fs = 40e6 #sampling frequency [Hz]
fc = 4.545e6 #pulse center frequency [Hz]
bw_factor = 0.7e13 #pulse bandwidth factor [Hz^2]
phi = -2.6143 #pulse phase in radians
no_of_symbols = 32 # for ofdm transmit signal
Ts = 1 / fs  # sampling time
symbol_time = no_of_symbols/fs
delta_f = 1/symbol_time

x0 = 0 #horizontal reference for reflecting plane location
z0 = 25e-3 #vertical reference for reflecting plane location

#I calculated these by hand D:
center1 = np.array([-28.3564e-3, 0, z0 + 5e-3])
#center1 = np.array([0, 0, z0 + 5e-3])
az1 = np.pi
co1 = 170*np.pi/180
h1 = 20e-3
w1 = 57.5877e-3
rect1 = Rectangle_test(center1, az1, co1, h1, w1)
#rect1 = Rectangle(center1, az1, co1, h1, w1)
center2 = np.array([18.66025e-3, 0, z0 + 5e-3])
az2 = 0
co2 = 165*np.pi/180
h2 = 20e-3
w2 = 38.6370e-3
rect2 = Rectangle_test(center2, az2, co2, h2, w2)
#rect2 = Rectangle(center2, az2, co2, h2, w2)

#put the reflecting rectangles in an array
objects = np.array([rect1, rect2])

#and put everything into a scenario object
scenario = Scenario(objects, c, NT, fs, bw_factor, fc, phi,no_of_symbols,symbol_time)

#now, the transducer parameters that will remain fixed throughout the sims:
opening = 15*np.pi/180 #opening angle in radians
nv = 121 #number of vertical gridpoints for the rays
nh = 121 #number of horizontal gridpoints for the rays
vres = 3e-3/nv #ray grid vertical resolution in [m] (set to 0.2mm)
hres = 1.5e-3/nh #ray grid horizontal res in [m] (set to 0.1mm)
distance = np.sqrt(3)*(nh-1)/2*hres #distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.

#%% EXPERIMENT: HERE'S WHERE STUFF SHOULD BE CODED AND TESTED

plt.close('all')

#this variable will let you see what the transducer is looking at as an image. you can get an idea of where to aim it that way. set it to false when you no longer need it (e.g. once the code has been made automatic).
debug=True

#this is the center of the transducer plane. the coordinates are given in [m], so leave the 1e-3 term untouched.
p = np.array([0,0,0])*1e-3
p1 = np.array([10,0,0])*1e-3

#now, we aim the transducer:
azimuth = 0*np.pi/180 #azimuth in [radians]
coelevation = 0*np.pi/180 #coelevation in [radians]

#create a transducer with these parameters, and some of the previously defined quantities that shouldn't be modified:
transducer = Plane(p, distance, azimuth, coelevation, vres, hres, nv, nh, opening)
start = timer()
transducer.prepareImagingPlane() #this ALWAYS has to be called right after creating a transducer!

#NOTE: if any of the parameters p, distance, azimuth, or coelevation have to be changed, a new transducer has to be created and prepareImagingPlane() has to be called again. this is because several parameters have to be computed again internally.

#now, use the transducer and the scenario to generate an A-scan:

Ascan = transducer.insonify(scenario,2,0) #,trans_beam_pattern,image,through,direc,sources,f

transducer1 = Plane(p1, distance, azimuth, coelevation, vres, hres, nv, nh, opening)
start = timer()
transducer1.prepareImagingPlane() #this ALWAYS has to be called right after creating a transducer!

#NOTE: if any of the parameters p, distance, azimuth, or coelevation have to be changed, a new transducer has to be created and prepareImagingPlane() has to be called again. this is because several parameters have to be computed again internally.

#now, use the transducer and the scenario to generate an A-scan:

Ascan1 = transducer1.insonify(scenario,2,0) #,trans_beam_pattern,image,through,direc,sources,f

psd = np.fft.fftshift(np.abs(np.fft.fft(Ascan/np.max(Ascan))))
autocorr = np.abs(signal.hilbert(signal.correlate(Ascan, Ascan,mode='full')))

time_axis = np.arange(NT)/fs

plt.figure(1)
#plt.plot(time_axis, Ascan,color='r',label='(10e-3,0,22e-3)')
#plt.plot(time_axis*1e6, Ascan,color='r',label='(10e-3,0,22e-3)')
plt.plot(time_axis*1e6,Ascan1,color='b',label='(10 mm,0,22 mm)')
plt.xlabel('Time (us)')
plt.ylabel('Amplitude')
#plt.title('Amplitude Scan at (10e-3,0,22e-3) and (0,0,20e-3)')
plt.title('Reflected Pulse')
plt.legend()
tikzplotlib.save("C:/Users/user/Documents/test1.tex")

plt.figure(2)
plt.plot(np.arange(-NT/2,NT/2)*(fs/NT)*1e-6, np.fft.fftshift(np.abs(np.fft.fft(Ascan))), label='fc (in MHz) = 4.545')
plt.xlabel('Frequency (in MHz)')
plt.ylabel('Energy (linear scale)')
plt.legend()
plt.title('Power spectral density')
tikzplotlib.save("C:/Users/user/Documents/test2.tex")

plt.figure(3)
plt.plot(autocorr/np.max(autocorr))
plt.title('Auto-Correlation Function')
plt.xlabel('Samples')
plt.ylabel('Magnitude')
tikzplotlib.save("C:/Users/user/Documents/test3.tex")

'''

through1 = through.reshape(through.shape[0]*through.shape[1],3)

#r = transducer.rays[:,:]

# plt.figure(1)
# plt.scatter(through1[:,0],through1[:,1])
#
# plt.figure(2)
# plt.imshow(through[:,:,0])
# plt.title('x coords')
#
# plt.figure(3)
# plt.imshow(through[:,:,1])
# plt.title('y coords')
#
# plt.figure(4)
# plt.imshow(through[:,:,2])
# plt.title('z coords')

print(timer() - start)
#r = transducer.rays.direction
#r = direc
r = direc
#r = np.flipud(r)
s = sources
#rays_temp = sources + r*10
rays_temp = r.reshape(r.shape[0]*r.shape[1],3)
#r = r.reshape(r.shape[0]*r.shape[1],3)
r_magnitude = np.linalg.norm(r[1,1,:])
v_grid = np.linspace(0,nv,nv)
h_grid = np.linspace(0,nh,nh)
v,h = np.meshgrid(v_grid,h_grid)
z = np.zeros((nv,nh))
z1 = np.zeros((nv,nh))
z[:,:] = trans_beam_pattern[:,:]
z1[:,:] = image[:,:]
z_temp = np.sum(z1,axis=1)
image1 = np.repeat(image[:,:,np.newaxis],nv,axis=2)
image1[:,:,] = z_temp

replica_z = np.repeat(image[:, :, np.newaxis], nv, axis=2)
beam = np.sum(replica_z,axis=0)
#let's check the plot out:


plt.figure(2)
plt.imshow(trans_beam_pattern)
plt.xlabel('x grid')
plt.ylabel('y grid')

plt.figure(3)
plt.imshow(np.rot90(beam))
plt.xlabel('y grid')
plt.ylabel('z grid')

rays_temp1 = rays_temp*trans_beam_pattern.reshape(nv*nh, 1)
magnitude_rays = np.zeros((nv*nh,1))
color_magnitude = trans_beam_pattern.reshape(nv*nh,1)

s1 = np.zeros(2)
s1[0] = s[0]
s1[1] = s[2]
origin = np.repeat(s1[:,np.newaxis],nv*nh,axis=1)

plt.figure(4)
plt.quiver(*origin, rays_temp1[:, 0], rays_temp1[:, 2], color_magnitude,scale=2)
plt.axis([-20*1e-3,20*1e-3,0*1e-3,40*1e-3])
plt.title('Opening angle=40')
plt.xlabel('x (length)')
plt.xlabel('z (height)')
plt.colorbar()
plt.show()

if debug:
	plt.figure(5)
	plt.imshow(transducer.image)
	'''