import sys
import scipy.io as sio
import numpy as np
import math
from numpy import linalg as nlin
import numpy.matlib as nmat
#from Sphere import Sphere
from Sphere_test import Sphere
from matplotlib import cm
import ffmpeg
import matplotlib.pyplot as plt
import scipy.signal as ssig
import time
from matplotlib import animation,rcParams
rcParams['animation.writer'] = 'ffmpeg'
from Rectangle_test import Rectangle_test
from Scenario import Scenario
from Plane import Plane
import tikzplotlib
#from Plane_ricker import Plane_ricker
#from IPython.display import HTML, Image

# %% SCENARIO PARAMETERS: DON'T TOUCH THESE!
NT = 800  # number of time domain samples
# c = 1481 #speed of sound in water [m/s] (taken from wikipedia :P)
c = 6000  # speed of sound for synthetic experiments [m/s]
fs = 40e6  # sampling frequency [Hz]
no_of_symbols = 32 # for ofdm transmit signal
Ts = 1 / fs  # sampling time
symbol_time = no_of_symbols/fs
delta_f = 1/symbol_time
fc = 4.5471e6  # pulse center frequency [Hz]
bw_factor = 1.4549e13  # pulse bandwidth factor [Hz^2]
phi = -2.6143  # pulse phase in radians

x0 = 0  # horizontal reference for reflecting plane location
z0 = 20e-3  # vertical reference for reflecting plane location

# I calculated these by hand D:
center1 = np.array([-28.3564e-3, 0, z0 + 5e-3])
az1 = np.pi
co1 = 170 * np.pi / 180
h1 = 20e-3
w1 = 57.5877e-3
rect1 = Rectangle_test(center1, az1, co1, h1, w1)

center2 = np.array([18.66025e-3, 0, z0 + 5e-3])
az2 = 0
co2 = 165 * np.pi / 180
h2 = 20e-3
w2 = 38.6370e-3
rect2 = Rectangle_test(center2, az2, co2, h2, w2)

center = np.array([0*1e-3, 0*1e-3, z0])
radius = 5e-3
sph = Sphere(center, radius)

# put the reflecting rectangles in an array
#objects = np.array([sph])
objects = np.array([rect1,rect2])

# and put everything into a scenario object
scenario = Scenario(objects, c, NT, fs, bw_factor, fc, phi,no_of_symbols,symbol_time)

# now, the transducer parameters that will remain fixed throughout the sims:
opening = 15 * np.pi / 180  # opening angle in radians
nv = 181  # number of vertical gridpoints for the rays
nh = 181  # number of horizontal gridpoints for the rays
vres = 3e-3 / nv  # ray grid vertical resolution in [m] (set to 0.2mm)
hres = 1.5e-3 / nh  # ray grid horizontal res in [m] (set to 0.1mm)
distance = np.sqrt(3) * (nh - 1) / 2 * hres  # distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.

sigma = 0.5
n = np.arange(-NT, NT) * Ts
n1 = np.arange(-NT, NT)
a_time = 0.05 * 10 ** -6
g = np.exp(-bw_factor * ((n - a_time)/(2 * sigma)) ** 2) * np.cos(2 * np.pi * fc * (n - a_time) + phi)
ricker_pulse = 2/((np.sqrt(3*sigma))*np.pi**0.008)*(1-bw_factor *((n - a_time)/sigma)**2)*np.exp(-bw_factor*(n - a_time)**2/(2*sigma**2))
sinc_pulse = np.sinc(bw_factor*(n - a_time)**2)*np.cos(2*np.pi*(n - a_time)*fc + phi)
delta_pulse = np.array([1 if n2 >= 0 - (symbol_time / 2) and n2 <= 0 + (symbol_time / 2) else 0 for n2 in n])
ifft_sig = np.fft.fftshift(np.abs(np.fft.ifft(delta_pulse)))
ofdm_signal = 20*ifft_sig*np.cos(2*np.pi*n*fc+phi)

n11 = np.arange(-NT,0)*Ts
n12 = np.arange(0,NT)*Ts
s = 0.5

gabor1 = np.exp(-(bw_factor*n11**2 + bw_factor*n11**2*s))*np.cos(2*np.pi*(n11)*fc + phi)
gabor2 = np.exp(-(bw_factor*n12**2 - bw_factor*n12**2*s))*np.cos(2*np.pi*(n12)*fc + phi)
gabor = np.zeros(2*NT)
gabor[0:NT] = gabor1
gabor[NT:2*NT] = gabor2

#g = ssig.hilbert(g)
#g = np.abs(g)
#g = g / np.max(g)


def collect_scan_finer_grid(p2, az, coe,snr,o_ang,nv2,nh2,pulse_s):
    vres1 = 3e-3 / nv2  # ray grid vertical resolution in [m] (set to 0.2mm)
    hres1 = 1.5e-3 / nh2  # ray grid horizontal res in [m] (set to 0.1mm)
    distance1 = np.sqrt(3) * (nh2 - 1) / 2 * hres1  # distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.

    transducer = Plane(p2, distance1, az*np.pi/180, coe*np.pi/180, vres1, hres1, nv2, nh2, o_ang*np.pi/180)
    transducer.prepareImagingPlane()  # this ALWAYS
    # has to be called right after creating a transducer!
    scan1 = transducer.insonify(scenario,pulse_s,0)  #0 to select rectangle as object and 1 to select sphere as object
    scan = np.abs(ssig.hilbert(scan1))
    sig_avg_pow = np.mean(scan)
    sig_avg_pow_db = 10.0*np.log10(sig_avg_pow)
    noise_pow = sig_avg_pow_db - snr
    noise_avg_w = 10.0 ** (noise_pow / 10)
    noise_samples = np.random.normal(0, np.sqrt(noise_avg_w), len(scan))
    scan1 = scan1 + noise_samples
    #coe = coe * 180 / np.pi
    #az = az * 180 / np.pi
    #file_name = str(p2[0]) + str(p2[1]) + str(np.round(coe,2)) + str(np.round(az,2))
    #write_data(file_name,scan)
    #print(end2 - start)
    #scan = np.abs(signal.hilbert(scan))
    #scan = np.abs(scan)

    #print('Inside finer grid')
    #print(end-start,end1-end,end2-end1)
    return scan1


def matched_filter(pulse,ascan):
    y = np.abs(ssig.hilbert(ssig.correlate(ascan, pulse, mode='same')))
    return y


def diag_spsa(r0,c0,a,A,alpha,gamma,c,iters,snr,corr):
    coordr0 = np.zeros((iters))
    coordc0 = np.zeros((iters))
    prevgrad = np.ones((2, 1))
    inner_products = np.zeros(10)
    coordr0[0] = r0
    coordc0[0] = c0
    Energy = []
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    counter = 0
    epsilon = 1
    '''
    plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    plt.plot(coordr01, coordc01, 'b*', label='SPSA')
    plt.legend(loc="upper right", prop={"size": 10})
    plt.xlim(-4, 4)
    plt.ylim(-3, 6)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    '''
    # plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    # plt.plot(coordr01, coordc01, 'r*')
    # plt.xlim(-4, 4)
    # plt.xlabel('Azimuth')
    # plt.ylabel('Elevation')
    # plt.ylim(-3, 6)
    # plt.colorbar()
    # plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
    # plt.plot(coordr0[0],coordc0[0],'r*')
    # plt.colorbar()
    parameters = np.array([r0,c0])
    parameters = parameters.reshape(2,1)
    point = np.array([-5, -5, 0]) * 1e-3
    inner_product_mean = 1e-6
    prev_energy = 0
    total_var = 0
    total_var1 = []
    prev_deltagrad_mat = np.zeros((2,2))

    for n in range(iters):
        step = a / ((n + A + 1) ** alpha)
        ck = c / ((n + 1) ** gamma)
        deltar = np.random.binomial(1, 0.5)
        deltar = 2 * deltar - 1

        deltac = np.random.binomial(1, 0.5)
        deltac = 2 * deltac - 1

        scan = collect_scan_finer_grid(point, r0 + ck * deltar, c0 + ck * deltac, snr)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))

        index = np.argmax(scan)
        fplus = np.sum(scan[index - 5:index + 5])

        scan = collect_scan_finer_grid(point, r0 - ck * deltar, c0 - ck * deltac, snr)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fminus = np.sum(scan[index - 5:index + 5])

        gradr = (fplus - fminus) / (2 * ck * deltar)
        gradc = (fplus - fminus) / (2 * ck * deltac)

        #print(fplus, fminus,gradr,gradc)

        #gradv = np.array([gradr, gradc])

        #gradr3 = np.asscalar(gradv[0])
        #gradc3 = np.asscalar(gradv[1])

        deltar1 = np.random.binomial(1, 0.5)
        deltar1 = 2 * deltar1 - 1

        deltac1 = np.random.binomial(1, 0.5)
        deltac1 = 2 * deltac1 - 1

        scan = collect_scan_finer_grid(point, r0 + ck * deltar + ck * deltar1, c0 + ck * deltac + ck * deltac1, snr)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fplus1 = np.sum(scan[index - 5:index + 5])

        scan = collect_scan_finer_grid(point, r0 - ck * deltar + ck * deltar1, c0 - ck * deltac + ck * deltac1, snr)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fminus1 = np.sum(scan[index - 5:index + 5])

        deltarg = (fplus1 - fplus) / (2 * ck * deltar1)
        deltacg = (fminus1 - fminus) / (2 * ck * deltac1)
        temp_arr = np.array([deltarg,deltacg])
        deltagrad_mat = np.abs(np.diag(np.diag(temp_arr))) + (-1)*np.identity(2)
        temp_arr = temp_arr.reshape(2, 1)
        deltagrad_mat_avg = n/(n+1)*prev_deltagrad_mat + 1/(n+1)*deltagrad_mat
        parameters = parameters - step * np.linalg.inv(deltagrad_mat_avg).dot(temp_arr)
        prev_deltagrad_mat = deltagrad_mat_avg

        r0 = np.asscalar(parameters[0])
        c0 = np.asscalar(parameters[1])

        scan = collect_scan_finer_grid(point, r0, c0, snr)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        energy = np.sum(scan[index - 5:index + 5])
        Energy.append(energy)
        coordr01.append(r0)
        coordc01.append(c0)
        print(energy,r0,c0)

    return Energy,coordr01,coordc01


def spsa(point,r0,c0,a,A,alpha,gamma,c,iters,snr,corr,o_angle,samples,nv1,nh1,pulse,pulse_sel):
    coordr0 = np.zeros((iters))
    coordc0 = np.zeros((iters))
    prevgrad = np.ones((2,1))
    inner_products = np.zeros(10)
    coordr0[0] = r0
    coordc0[0] = c0
    Energy = []
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    counter = 0
    epsilon = 1
    '''
    plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    plt.plot(coordr01, coordc01, 'b*', label='SPSA')
    plt.legend(loc="upper right", prop={"size": 10})
    plt.xlim(-4, 4)
    plt.ylim(-3, 6)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    '''
    # plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    # plt.plot(coordr01, coordc01, 'r*')
    # plt.xlim(-4, 4)
    # plt.xlabel('Azimuth')
    # plt.ylabel('Elevation')
    # plt.ylim(-3, 6)
    # plt.colorbar()
    # plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
    # plt.plot(coordr0[0],coordc0[0],'r*')
    # plt.colorbar()
    #point = np.array([6,0,0])*1e-3
    inner_product_mean = 1e-6
    prev_energy = 0
    total_var = 0
    total_var1 = []
    max_index = 0
    max_index_list = []
    #n=0
    for n in range(iters):
        step = a / ((n + A + 1) ** alpha)
        ck = c / ((n + 1) ** gamma)
        deltar = np.random.binomial(1, 0.5)
        deltar = 2 * deltar - 1

        deltac = np.random.binomial(1, 0.5)
        deltac = 2 * deltac - 1

        scan = collect_scan_finer_grid(point, r0 + ck * deltar, c0 + ck * deltac,snr,o_angle,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse,scan)
        else:
            scan = np.abs(ssig.hilbert(scan))

        index = np.argmax(scan)
        fplus = np.sum(scan[index-samples:index+samples])
        # scan = collect_scan_finer_grid(point, r0 + ck * deltar, c0 + ck * deltac, snr,o_angle,nv1,nh1)
        # if corr == 1:
        #     scan = matched_filter(g, scan)
        # else:
        #     scan = np.abs(ssig.hilbert(scan))
        # index = np.argmax(scan)
        # fplus1 = np.sum(scan[index - samples:index + samples])
        #print(fplus,fplus1)

        scan = collect_scan_finer_grid(point, r0 - ck * deltar, c0 - ck * deltac,snr,o_angle,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fminus = np.sum(scan[index - samples:index + samples])

        # scan = collect_scan_finer_grid(point, r0 - ck * deltar, c0 - ck * deltac, snr,o_angle,nv1,nh1)
        # if corr == 1:
        #     scan = matched_filter(g, scan)
        # else:
        #     scan = np.abs(ssig.hilbert(scan))
        # index = np.argmax(scan)
        # fminus1 = np.sum(scan[index - samples:index + samples])
        #print(fminus, fminus1)
        #print(fminus)

        #fplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + ck * deltar, c0 + ck * deltac, dr, dc)[1500:])
        #fminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - ck * deltar, c0 - ck * deltac, dr, dc)[1500:])

        gradr = (fplus - fminus) / (2 * deltar)

        #gradr1 = (fplus1 - fminus1) / (2 * deltar)

        gradc = (fplus - fminus) / (2 * deltac)

        #gradc1 = (fplus1 - fminus1) / (2 * deltac)

        #gradv = np.array([gradr,gradr1])

        #gradcv = np.array([gradc, gradc1])
        #val1 = np.where(epsilon > np.linalg.norm(gradr1),epsilon,np.linalg.norm(gradr1))
        #val2 = np.where(epsilon > np.linalg.norm(gradr),epsilon,np.linalg.norm(gradr))
        #val3 = np.where(epsilon > np.linalg.norm(gradc1),epsilon,np.linalg.norm(gradc1))
        #val4 = np.where(epsilon > np.linalg.norm(gradc),epsilon,np.linalg.norm(gradc))
        #print(val1,val2)

        #gradr2 = gradr/val1 + gradr1/val2

        #gradc2 = gradc/val3 + gradc1/val4

        gradv_final = np.array([gradr,gradc])

        gradv_final = gradv_final.reshape(2,1)

        #inner_products[counter] = gradv.transpose().dot(prevgrad)
        #counter += 1
        '''
        if counter == 10:
            inner_product_mean = np.mean(inner_products)
            print(inner_product_mean)
            counter = 0
        '''
        #gradv = gradv/np.linalg.norm(gradv)

        gradr3 = np.asscalar(gradv_final[0])
        gradc3 = np.asscalar(gradv_final[1])

        r0 = r0 + step * gradr3
        c0 = c0 + step * gradc3

        scan = collect_scan_finer_grid(point, r0, c0,snr,o_angle,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        energy = np.sum(scan[index - samples:index + samples])
        total_var = np.abs(energy - prev_energy)

        prev_energy = energy
        print('Final energy : ' + str(energy) + ' ' + str(total_var) + ' ' + str(r0) + ' ' + str(c0))
        #energy = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])

        #coordr0[n] = r0
        #coordc0[n] = c0
        total_var1.append(total_var)

        coordr01.append(r0)
        coordc01.append(c0)

        #n += 1
        #prevgrad = gradv
        '''
        coordr01.append(r0)
        coordc01.append(c0)
        plt.title('K=' + str(n))
        plt.plot(coordr01, coordc01, 'bo-', markersize='1')
        plt.savefig('Image_SPSA' + str(n) + '.png')
        '''
        #plt.title('K=' + str(n))
        #plt.plot(coordr01, coordc01, 'ro-', markersize='1')
        #plt.savefig('Image' + str(n) + '.png')
        max_index = index
        max_index_list.append(max_index)
        Energy.append(energy)

    return Energy,coordr01,coordc01#,r0,c0,max_index,max_index_list


def rdsa(r0,c0,dr,dc,noofiterations,epsilon,gamma,alpha,A,a,c,snr,corr,o_angl,samples,nv1,nh1,pulse,pulse_sel):
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    #cvals = np.zeros((iters + 1))
    #rvals = np.zeros((iters + 1))
    inner_products = np.zeros(10)
    #cvals[0] = c0
    #rvals[0] = r0
    CurrentVal = 1  # Dummy Initialization
    d1 = -1
    noise = 0.1
    prevgrad = np.ones((2, 1))
    probLessThan0 = (1 + epsilon) / (2 + epsilon)
    #coordr0 = np.zeros((noofiterations + 1))
    #coordc0 = np.zeros((noofiterations + 1))
    Energy = []
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    counter = 0
    point = np.array([6,0,0])*1e-3
    inner_product_mean = 1e-6
    epsilon1 = 1
    '''
    plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    plt.plot(coordr01, coordc01, 'y*', label='RDSA')
    plt.legend(loc="upper right", prop={"size": 10})
    plt.xlim(-4, 4)
    plt.ylim(-3, 6)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    '''

    for n in range(noofiterations):
        # print('Energy Difference:' + str(abs(CurrentVal - PreviousVal)))
        step = a / ((n + A + 1) ** alpha)
        delta = c / ((n + 1) ** gamma)
        d2 = bool(np.random.binomial(1, probLessThan0))
        if d2:
            d1 = -1
        else:
            d1 = 1 + epsilon

        d3 = bool(np.random.binomial(1, probLessThan0))
        if d3:
            d4 = -1
        else:
            d4 = 1 + epsilon

        scan = collect_scan_finer_grid(point, r0 + delta * d1, c0 + delta * d4, snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])

        # scan = collect_scan_finer_grid(point, r0 + delta * d1, c0 + delta * d4, snr,o_angl,nv1,nh1)
        # if corr == 1:
        #     scan = matched_filter(g, scan)
        # else:
        #     scan = np.abs(ssig.hilbert(scan))
        # index = np.argmax(scan)
        # frplus1 = np.sum(scan[index - samples:index + samples])

        #frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + delta * d1, c0 + delta * d4, dr, dc)[1500:]) + noise
        scan = collect_scan_finer_grid(point, r0 - delta * d1, c0 - delta * d4, snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])

        # scan = collect_scan_finer_grid(point, r0 - delta * d1, c0 - delta * d4, snr,o_angl,nv1,nh1)
        # if corr == 1:
        #     scan = matched_filter(g, scan)
        # else:
        #     scan = np.abs(ssig.hilbert(scan))
        # index = np.argmax(scan)
        # frminus1 = np.sum(scan[index - samples:index + samples])

        #frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - delta * d1, c0 - delta * d4, dr, dc)[1500:]) + noise

        gradr = (frplus - frminus) / (2 * delta*d1) * (1 / (1 + epsilon)) #* d1

        #gradr1 = (frplus1 - frminus1) / (2 * delta*d1) * (1 / (1 + epsilon)) #* d1

        gradc = (frplus - frminus) / (2 * delta*d4) * (1 / (1 + epsilon)) #* d4

        #gradc1 = (frplus1 - frminus1) / (2 * delta*d4) * (1 / (1 + epsilon)) #* d4

        #m = np.linalg.norm(np.array([gradr, gradc]))

        #val1 = np.where(epsilon1 > np.linalg.norm(gradr1), epsilon1, np.linalg.norm(gradr1))
        #val2 = np.where(epsilon1 > np.linalg.norm(gradr), epsilon1, np.linalg.norm(gradr))
        #val3 = np.where(epsilon1 > np.linalg.norm(gradc1), epsilon1, np.linalg.norm(gradc1))
        #val4 = np.where(epsilon1 > np.linalg.norm(gradc), epsilon1, np.linalg.norm(gradc))

        #gradr2 = gradr / val1 + gradr1 / val2

        #gradc2 = gradc / val3 + gradc1 / val4

        gradv_final = np.array([gradr, gradc])

        gradv_final = gradv_final.reshape(2, 1)

        gradr3 = np.asscalar(gradv_final[0])
        gradc3 = np.asscalar(gradv_final[1])

        #gradv = np.array([gradr, gradc])
        #gradv = gradv.reshape(2, 1)

        #inner_product = gradv.transpose().dot(prevgrad)

        #inner_products[counter] = gradv.transpose().dot(prevgrad)
        #counter += 1

        r0 = r0 + step * gradr3
        c0 = c0 + step * gradc3

        coordr01.append(r0)
        coordc01.append(c0)
        #n += 1

        #coordr0[i + 1] = r0
        #coordc0[i + 1] = c0
        '''
        coordr01.append(r0)
        coordc01.append(c0)

        plt.title('K=' + str(i))
        plt.plot(coordr01, coordc01, 'yo-', markersize='1')
        plt.savefig('Image_RDSA' + str(i) + '.png')
        '''
        scan = collect_scan_finer_grid(point, r0, c0,snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        CurrentVal = np.sum(scan[index - samples:index + samples])
        print(CurrentVal,r0,c0)
        #CurrentVal = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(CurrentVal)
    return Energy,coordr01,coordc01


def line_search(r0,c0,c,roh,noofiterations,dr,dc):
    PreviousVal = 0
    CurrentVal = 1  # Dummy Initialization
    coordr0 = np.zeros((noofiterations + 1))
    coordc0 = np.zeros((noofiterations + 1))
    deltar = dr / 2
    deltac = dc / 2
    Energy = []

    # while abs(CurrentVal - PreviousVal) > 10e-5:  # Energy Difference should be greater than 10e-5
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    Energy = []
    '''
    plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    plt.plot(coordr01, coordc01, 'k*',label='Line Search')
    plt.legend(loc="upper right", prop={"size": 10})
    plt.xlim(-4, 4)
    plt.ylim(-3, 6)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    '''
    for i in range(noofiterations):
        step = 0.05
        b = 1
        #print('Energy Difference:' + str(abs(CurrentVal - PreviousVal)))
        PreviousVal = CurrentVal
        # print('iteration number:' + str(iters))
        frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])

        gradr = (frplus - frminus) / (2 * deltar)

        fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltac, dr, dc)[1500:])
        fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltac, dr, dc)[1500:])

        gradc = (fcplus - fcminus) / (2 * deltac)

        m = np.linalg.norm(np.array([gradr, gradc]))

        # Loop until f(x + step*gradr) becomes greater than f(x) + c*step*m

        while b == 1:
            Temp1 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + step * gradr, c0 + step * gradc, dr, dc)[1500:])
            Temp2 = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
            Temp2 = Temp2 + c * step * m

            if Temp1 < Temp2:
                step = roh * step
            else:
                b = 0

        r0 = r0 + step * gradr
        c0 = c0 + step * gradc

        CurrentVal = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(CurrentVal)
        # coordr0.append(r0)
        # coordc0.append(c0)
        coordr0[i + 1] = r0
        coordc0[i + 1] = c0
        '''
        coordr01.append(r0)
        coordc01.append(c0)

        plt.title('K=' + str(i))
        plt.plot(coordr01, coordc01, 'ko-', markersize='1')
        plt.savefig('Image_LS' + str(i) + '.png')
        '''

    return Energy,coordr0,coordc0


def gradient_ascent(r0,c0,step,dr,dc,iters,snr,corr,o_angl,samples,nv1,nh1,pulse,pulse_sel):
    # how far to look in order to find the gradient, along with stepsize
    deltar = dr
    deltac = dc
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    prevgrad = np.ones((2,1))
    #cvals = np.zeros((iters))
    #rvals = np.zeros((iters))
    #cvals[0] = r0
    #rvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    angles = []
    Energy = []
    print('Inside gradient ascent')
    point = np.array([10,0,0])*1e-3
    '''
    plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    plt.plot(coordr01, coordc01, 'g*',label='Gradient Ascent')
    plt.legend(loc="upper right", prop={"size": 10})
    plt.xlim(-4, 4)
    plt.ylim(-3, 6)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    '''
    # use a simple gradient ascent with fixed stepsize
    for i in range(iters):
        scan = collect_scan_finer_grid(point, r0 + deltar, c0,snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])
        #frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        #frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0,snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])

        gradr = (frplus - frminus) / (2 * deltar)
        # print(gradr)
        scan = collect_scan_finer_grid(point, r0, c0 + deltar,snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])
        #fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        #fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0, c0 - deltar,snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        gradc = (fcplus - fcminus) / (2 * deltac)
        # print(gradc)
        '''
        scan = collect_scan_finer_grid(point, r0, c0, snr)
        index = np.argmax(scan)
        fval = np.sum(scan[index - 50:index + 50])

        # fval = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        f11 = (frplus - 2 * fval + frminus) / (deltar ** 2)

        scan = collect_scan_finer_grid(point, r0 + deltar, c0 + deltac, snr)
        index = np.argmax(scan)
        f1 = np.sum(scan[index - 50:index + 50])
        # f1 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0 + deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 + deltar, c0 - deltac, snr)
        index = np.argmax(scan)
        f2 = np.sum(scan[index - 50:index + 50])
        # f2 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0 - deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0 + deltac, snr)
        index = np.argmax(scan)
        f3 = np.sum(scan[index - 50:index + 50])
        # f3 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0 + deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0 - deltac, snr)
        index = np.argmax(scan)
        f4 = np.sum(scan[index - 50:index + 50])
        # f4 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0 - deltac, dr, dc)[1500:])

        f21 = (f1 - f2 - f3 + f4) / (4 * deltar * deltac)

        f12 = f21

        f22 = (fcplus - 2 * fval + fcminus) / (deltac ** 2)

        h = np.array([f11, f12, f21, f22])
        H = np.reshape(h, (2, 2))
        H1 = np.linalg.inv(H)
        '''
        r0 = r0 + step * gradr
        c0 = c0 + step * gradc

        #gradv = np.array([gradr,gradc])
        #gradv = gradv.reshape(2,1)

        #angle = np.arccos(gradv.transpose().dot(prevgrad)/(np.linalg.norm(gradv)*np.linalg.norm(prevgrad)))*180/np.pi
        #angles.append(angle)

        #rvals[i] = r0
        #cvals[i] = c0

        coordr01.append(r0)
        coordc01.append(c0)

        scan = collect_scan_finer_grid(point, r0, c0,snr,o_angl,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        peak_index_list[i] = index
        temp = np.sum(scan[index - samples:index + samples])
        #temp = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(temp)
        print('current echo pseudo-energy: ' + str(temp) + '  ' + str(r0) + '  ' + str(c0))
    return Energy,coordr01,coordc01


def quasi_newton_method(r0, c0, step, dr, dc, iters, scaling_factor,snr):
    deltar = dr
    deltac = dc
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    #energies = np.zeros(iters)
    prev_G = np.zeros((2,1))
    Energy = []
    cvals = []
    rvals = []
    rvals.append(r0)
    cvals.append(c0)
    point = np.array([10, 0, 0]) * 1e-3
    #hess_init = np.random.randn(2)
    #hess_init = hess_init.dot(hess_init.T)
    #hess_init = hess_init + np.identity(2)
    hess_init = (scaling_factor)*np.identity(2)
    #hess_init = hess_init1
    hess_update_inv = np.linalg.inv(hess_init)
    hess_estimate_error = []
    #############################################################

    for i in range(iters):
        #peak_energy = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 + deltar, c0, snr,15,181,181,2)
        scan = matched_filter(g, scan)
        index = np.argmax(scan)
        peak_energy = np.sum(scan[index - 15:index + 15])
        #peak_energy1 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0, snr,15,181,181,2)
        scan = matched_filter(g, scan)
        index = np.argmax(scan)
        peak_energy1 = np.sum(scan[index - 15:index + 15])

        grad_r = (peak_energy - peak_energy1) / (2 * deltar)

        #peak_energy2 = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0, c0 + deltac, snr,15,181,181,2)
        scan = matched_filter(g, scan)
        index = np.argmax(scan)
        peak_energy2 = np.sum(scan[index - 15:index + 15])

        scan = collect_scan_finer_grid(point, r0, c0 - deltac, snr,15,181,181,2)
        scan = matched_filter(g, scan)
        index = np.argmax(scan)
        peak_energy3 = np.sum(scan[index - 15:index + 15])

        grad_c = (peak_energy2 - peak_energy3) / (2 * deltac)
        # peak_energy3 = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltac, dr, dc)[1500:])

        grad_v = np.array([grad_r, grad_c])
        G1 = np.reshape(grad_v, (2, 1))
        diff = G1 - prev_G

        #v_init = v1
        v1 = v1 + step * (hess_update_inv.dot(G1))
        #v1 = v1 + step * (G1)

        #diff1 = v1 - v_init
        diff1 = step * (hess_update_inv.dot(G1))
        diff_t = diff1.transpose()
        diff1_t = diff.transpose()
        b = hess_init.dot(diff1)
        b1 = b.dot(diff_t)
        b2 = b1.dot(hess_init)

        c3 = diff_t.dot(hess_init)
        c3 = c3.dot(diff1)
        c3 = np.asscalar(c3[0])
        b3 = (-1)*(b2 * (1/c3))
        t = (diff.dot(diff1_t))/(diff1_t.dot(diff1))
        hess_update = hess_init + b3 + t #+ 0.0000001) ### Update hessian matrix #
        for_norm = np.linalg.norm((hess_update - hess_init), 'fro')
        hess_estimate_error.append(for_norm)
        hess_init = hess_update
        #hess_update_inv = np.linalg.inv(hess_update)
        #print(for_norm)
        '''
        if for_norm < 20:
            hess_update_inv = np.linalg.inv(hess_update)
            hess_init = hess_update
        else:
            hess_update_inv = hess_init
        '''
        prev_G = G1  ### keep previous gradient information

        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        rvals.append(r0)
        cvals.append(c0)

        scan = collect_scan_finer_grid(point, r0, c0, snr,15,181,181,2)
        scan = matched_filter(g, scan)
        index = np.argmax(scan)
        max_energy = np.sum(scan[index - 15:index + 15])
        print(max_energy,for_norm,r0,c0)
        Energy.append(max_energy)

    return Energy,rvals,cvals


def compute_hessian_init(r0,c0,step,dr,dc,iters):
    deltar = 2 * dr
    deltac = 2 * dc
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))

    cvals = np.zeros((iters))
    rvals = np.zeros((iters))
    rvals[0] = r0
    cvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    Energy = []

    frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
    frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
    # print(frplus)
    # print(frminus)
    gradr = (frplus - frminus) / (2 * deltar)

    fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltac, dr, dc)[1500:])
    fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltac, dr, dc)[1500:])
    gradc = (fcplus - fcminus) / (2 * deltac)

    fval = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
    f11 = (frplus - 2 * fval + frminus) / (deltar ** 2)

    f1 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0 + deltac, dr, dc)[1500:])
    f2 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0 - deltac, dr, dc)[1500:])
    f3 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0 + deltac, dr, dc)[1500:])
    f4 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0 - deltac, dr, dc)[1500:])

    f21 = (f1 - f2 - f3 + f4) / (4 * deltar * deltac)

    f12 = f21

    f22 = (fcplus - 2 * fval + fcminus) / (deltac ** 2)

    h = np.array([f11, f12, f21, f22])
    H = np.reshape(h, (2, 2))
    return H


def conjugate_gradient(r0,c0,step,dr,dc,iters):
    deltar = 4*dr
    deltac = 4*dc
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    cvals = np.zeros((iters))
    rvals = np.zeros((iters))
    rvals[0] = r0
    cvals[0] = c0
    coordr01 = []
    coordc01 = []
    angles = []
    coordr01.append(r0)
    coordc01.append(c0)
    Energy = []
    prev_grad = np.ones((2,1))
    prev_s = np.zeros((2,1))

    for i in range(1, iters):
        frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)

        fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltac, dr, dc)[1500:])
        fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltac, dr, dc)[1500:])
        gradc = (fcplus - fcminus) / (2 * deltac)

        grad = np.array([gradr, gradc])
        grad_v = np.reshape(grad, (2, 1))

        angle = np.arccos(grad_v.transpose().dot(prev_grad) / (np.linalg.norm(grad_v) * np.linalg.norm(prev_grad))) * 180 / np.pi
        angle2 = np.asscalar(angle[0])

        if angle2 > 30:
            prev_grad = prev_grad*0.5
            print('Inside')

        Beta = (grad_v.transpose().dot((grad_v - prev_grad)))/(prev_grad.transpose().dot(prev_grad))

        s = grad_v + Beta*prev_s
        prev_grad = grad_v
        v1 = v1 + step * s
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])
        rvals[i] = r0
        cvals[i] = c0
        prev_s = s
        angles.append(angle)
        energy = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(energy)
        print(energy,angle)

    return Energy, rvals, cvals,angles


def newton_method(r0,c0,step,dr,dc,iters,snr,corr,o_angl,samples,nv1,nh1,pulse_s):
    deltar = dr/2
    deltac = dc/2
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    v = np.array([r0,c0])
    v1 = np.reshape(v,(2,1))
    step_para = np.array([step,step])
    step_para = step_para.reshape(2,1)
    prev_grad = np.ones((2,1))*1e-5
    #cvals = np.zeros((iters))
    #rvals = np.zeros((iters))
    #rvals[0] = r0
    #cvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    Energy = []
    point = np.array([10, 0, 0]) * 1e-3
    prev = 0

    for i in range(iters):
        #step = step/(i+1)
        step_para[0] = step/(4*(i+1))
        #step_para[1] = step/(i+1)
        scan = collect_scan_finer_grid(point, r0 + deltar,c0, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])

        scan = collect_scan_finer_grid(point, r0 - deltar, c0, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])

        #frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        #frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)

        scan = collect_scan_finer_grid(point, r0, c0 + deltac, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])

        scan = collect_scan_finer_grid(point, r0, c0 - deltac, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        #fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltac, dr, dc)[1500:])
        #fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltac, dr, dc)[1500:])
        gradc = (fcplus - fcminus) / (2 * deltac)

        scan = collect_scan_finer_grid(point, r0, c0, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fval = np.sum(scan[index - samples:index + samples])

        #fval = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        f11 = (frplus - 2*fval + frminus)/(deltar**2)

        scan = collect_scan_finer_grid(point, r0 + deltar, c0 + deltac, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        f1 = np.sum(scan[index - samples:index + samples])
        #f1 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0 + deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 + deltar, c0 - deltac, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        f2 = np.sum(scan[index - samples:index + samples])
        #f2 = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0 - deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0 + deltac, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        f3 = np.sum(scan[index - samples:index + samples])
        #f3 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0 + deltac, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0 - deltac, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        f4 = np.sum(scan[index - samples:index + samples])
        #f4 = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0 - deltac, dr, dc)[1500:])

        f21 = (f1 - f2 - f3 + f4)/(4*deltar*deltac)

        f12 = f21

        f22 = (fcplus - 2*fval + fcminus)/(deltac**2)

        h = np.array([f11, f12, f21, f22])

        H = np.reshape(h, (2,2))
        H1 = np.linalg.inv(H)
        g1 = np.array([gradr, gradc])
        G = np.reshape(g1,(2,1))
        #print(G,G.transpose().dot(prev_grad)/(np.linalg.norm(G)*np.linalg.norm(prev_grad)))
        v1 = v1 - step_para*(H1.dot(G))
        #rvals[i] = v1[0]
        #cvals[i] = v1[1]
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        coordr01.append(r0)
        coordc01.append(c0)
        scan = collect_scan_finer_grid(point, r0, c0, snr,o_angl,nv1,nh1,pulse_s)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        temp = np.sum(scan[index - samples:index + samples])
        print(temp,r0,c0)
        prev = temp
        prev_grad = G
        #temp = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(temp)

    return Energy, coordr01, coordc01


def newton_method1(r0,c0,step,dr,dc,iters,flag):
    deltar = dr
    deltac = dc
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid

    cvals = np.zeros((iters + 1))
    rvals = np.zeros((iters + 1))
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    Energy = []

    for i in range(iters):
        frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)

        fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        gradc = (fcplus - fcminus) / (2 * deltac)

        fval = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])

        if flag == 0:
            r0 = r0 - step * (fval / gradr)
            c0 = c0 - step * (fval / gradc)
        else:
            r0 = r0 + step * (fval / gradr)
            c0 = c0 + step * (fval / gradc)

        rvals[i + 1] = r0
        cvals[i + 1] = c0

        temp = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(temp)

    return Energy, rvals, cvals


def gradient_adagrad(r0,c0,step,momentum,dr,dc,iters,snr,corr,o_angl,samples,nv1,nh1):
    deltar = dr / 2
    deltac = dc / 2
    # k = 3
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    prevgrad = np.ones((2, 1)) * 1e-6
    prevgrad_avg = np.ones((2, 1)) * 1e-6
    # counter = 0
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    # cvals = np.zeros((iters))
    # rvals = np.zeros((iters))
    # inner_products = np.zeros(5)
    # cvals[0] = r0
    # rvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    # angles = []
    Energy = []
    print('Inside gradient ascent')
    point = np.array([6, 0, 0]) * 1e-3
    avg_r = 0
    avg_c = 0
    counter = 0

    for n in range(iters):
        #step1 = (1 - 3/(5 + n))
        #step = step /(n+1)
        scan = collect_scan_finer_grid(point, r0 + deltar,c0,snr,o_angl,nv1,nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))

        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])
        # frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        # frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)
        # print(gradr)
        scan = collect_scan_finer_grid(point, r0, c0 + deltac, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])
        # fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        # fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0, c0 - deltac, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        gradc = (fcplus - fcminus) / (2 * deltac)
        grad_vec = np.array([gradr, gradc])

        #avg_r = (1 - momentum) * gradr ** 2 + momentum * np.asscalar(prevgrad[0])
        #avg_c = (1 - momentum) * gradc ** 2 + momentum * np.asscalar(prevgrad[1])

        avg_r += gradr ** 2 + np.square(np.asscalar(prevgrad[0]))
        avg_c += gradc ** 2 + np.square(np.asscalar(prevgrad[1]))
        #counter += 1
        grad_vec = grad_vec.reshape(2, 1)
        prevgrad = grad_vec
        #prevgrad[0] = avg_r
        #prevgrad[1] = avg_c
        grad_squrs = np.array([avg_r, 0, 0, avg_c])
        G = grad_squrs.reshape(2, 2)
        #G1 = np.linalg.inv(G).dot(grad_vec)
        v1 = v1 + step * np.linalg.inv(np.sqrt(G)).dot(grad_vec)
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        scan = collect_scan_finer_grid(point, r0, c0, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        temp = np.sum(scan[index - samples:index + samples])

        Energy.append(temp)
        coordr01.append(r0)
        coordc01.append(c0)
        #print(G,grad_vec,G1)
        print('current echo pseudo-energy: ' + str(temp) + '  ' + str(r0) + '  ' + str(c0) + '\n')
        #avg_r = 0
        #avg_c = 0

    return Energy, coordr01, coordc01


def gradient_adadelta(r0,c0,step,momentum,dr,dc,iters,snr,corr,o_angl,samples,nv1,nh1):
    deltar = dr / 2
    deltac = dc / 2
    # k = 3
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    prevgrad = np.ones((2, 1)) * 1e-6
    prevgrad_avg = np.ones((2, 1)) * 1e-6
    # counter = 0
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    # cvals = np.zeros((iters))
    # rvals = np.zeros((iters))
    # inner_products = np.zeros(5)
    # cvals[0] = r0
    # rvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    # angles = []
    Energy = []
    print('Inside gradient ascent')
    point = np.array([6, 0, 0]) * 1e-3
    avg_r = 0
    avg_c = 0
    counter = 0

    for n in range(iters):
        #step1 = (1 - 3/(5 + n))
        #step = step /(n+1)
        scan = collect_scan_finer_grid(point, r0 + deltar,c0,snr,o_angl,nv1,nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))

        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])
        # frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        # frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)
        # print(gradr)
        scan = collect_scan_finer_grid(point, r0, c0 + deltac, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])
        # fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        # fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0, c0 - deltac, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        gradc = (fcplus - fcminus) / (2 * deltac)
        grad_vec = np.array([gradr, gradc])

        #avg_r = (1 - momentum) * gradr ** 2 + momentum * np.asscalar(prevgrad[0])
        #avg_c = (1 - momentum) * gradc ** 2 + momentum * np.asscalar(prevgrad[1])

        avg_r += (1 - momentum)*gradr ** 2 + momentum*np.square(np.asscalar(prevgrad[0]))
        avg_c += (1 - momentum)*gradc ** 2 + momentum*np.square(np.asscalar(prevgrad[1]))
        print(avg_r,avg_c)
        counter += 1
        grad_vec = grad_vec.reshape(2, 1)
        prevgrad = grad_vec
        #prevgrad[0] = avg_r
        #prevgrad[1] = avg_c
        grad_squrs = np.array([avg_r, 0, 0, avg_c])
        G = grad_squrs.reshape(2, 2)

        if counter == 5:
            counter = 0
            avg_r = 0
            avg_c = 0

        #G1 = np.linalg.inv(G).dot(grad_vec)
        v1 = v1 + step * np.linalg.inv(np.sqrt(G)).dot(grad_vec)
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        scan = collect_scan_finer_grid(point, r0, c0, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        temp = np.sum(scan[index - samples:index + samples])

        Energy.append(temp)
        coordr01.append(r0)
        coordc01.append(c0)
        #print(G,grad_vec,G1)
        print('current echo pseudo-energy: ' + str(temp) + '  ' + str(r0) + '  ' + str(c0) + '\n')
        #avg_r = 0
        #avg_c = 0

    return Energy, coordr01, coordc01


def gradient_adaptive_moment(r0,c0,step,momentum,dr,dc,iters,snr,corr,o_angl,samples,nv1,nh1):
    deltar = dr / 2
    deltac = dc / 2
    # k = 3
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    prevgrad = np.ones((2, 1)) * 1e-6
    moment = np.ones((2, 1)) * 1e-6
    sq_moment = np.ones((2, 1)) * 1e-6
    #prevgrad_avg = np.ones((2, 1)) * 1e-6
    # counter = 0
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    # cvals = np.zeros((iters))
    # rvals = np.zeros((iters))
    # inner_products = np.zeros(5)
    # cvals[0] = r0
    # rvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    # angles = []
    Energy = []
    print('Inside gradient ascent')
    point = np.array([6, 0, 0]) * 1e-3
    avg_r = 0
    avg_c = 0
    counter = 0

    for n in range(iters):
        #step1 = (1 - 3/(5 + n))
        #step = step /(n+1)
        scan = collect_scan_finer_grid(point, r0 + deltar,c0,snr,o_angl,nv1,nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))

        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])
        # frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        # frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)
        # print(gradr)
        scan = collect_scan_finer_grid(point, r0, c0 + deltac, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])
        # fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        # fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0, c0 - deltac, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        gradc = (fcplus - fcminus) / (2 * deltac)
        grad_vec = np.array([gradr, gradc])
        grad_vec = grad_vec.reshape(2, 1)

        moment = momentum * moment + (1 - momentum) * grad_vec
        sq_moment = momentum * sq_moment + (1 - momentum) * np.square(grad_vec)
        moment_est = moment / (1 - momentum ** (n+1))
        sq_moment_est = sq_moment / (1 - momentum ** (n+1))
        #avg_r = (1 - momentum) * gradr ** 2 + momentum * np.asscalar(prevgrad[0])
        #avg_c = (1 - momentum) * gradc ** 2 + momentum * np.asscalar(prevgrad[1])

        #avg_r += gradr ** 2 + np.square(np.asscalar(prevgrad[0]))
        #avg_c += gradc ** 2 + np.square(np.asscalar(prevgrad[1]))
        #counter += 1

        #prevgrad = grad_vec
        #prevgrad[0] = avg_r
        #prevgrad[1] = avg_c
        #grad_squrs = np.array([avg_r, 0, 0, avg_c])
        #G = grad_squrs.reshape(2, 2)
        #G1 = np.linalg.inv(G).dot(grad_vec)
        #G = np.diagflat(sq_moment_est)
        v1 = v1 + step * np.linalg.inv(np.sqrt(np.diagflat(sq_moment_est))).dot(moment_est)
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        scan = collect_scan_finer_grid(point, r0, c0, snr, o_angl, nv1, nh1)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        temp = np.sum(scan[index - samples:index + samples])

        Energy.append(temp)
        coordr01.append(r0)
        coordc01.append(c0)
        #print(G,grad_vec,G1)
        print('current echo pseudo-energy: ' + str(temp) + '  ' + str(r0) + '  ' + str(c0) + '\n')
        #avg_r = 0
        #avg_c = 0

    return Energy, coordr01, coordc01


def gradient_accelerated(r0,c0,step,momentum,dr,dc,iters,snr,corr,o_angl,samples,nv1,nh1,pulse,pulse_sel):
    deltar = 12*dr
    deltac = 12*dc
    #k = 3
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    prevgrad = np.ones((2, 1))*1e-6
    #counter = 0
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    #cvals = np.zeros((iters))
    #rvals = np.zeros((iters))
    #inner_products = np.zeros(5)
    #cvals[0] = r0
    #rvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    #angles = []
    Energy = []
    print('Inside gradient ascent')
    point = np.array([10,0,0])*1e-3
    #inner_mean = 1e-6
    #momentum_vec = np.zeros(k)
    #a = 0.1

    for n in range(iters):
        step1 = (1 - 3/(5 + n))
        #step = step /(n+1)
        scan = collect_scan_finer_grid(point, r0 + step*np.asscalar(prevgrad[0]) + deltar,c0 + step*np.asscalar(prevgrad[1]),snr,o_angl,nv1,nh1,2)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])
        #frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        #frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 + step*np.asscalar(prevgrad[0]) - deltar,c0 + step*np.asscalar(prevgrad[1]),snr,o_angl,nv1,nh1,2)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])
        # print(frplus)
        # print(frminus)
        gradr = (frplus - frminus) / (2 * deltar)
        # print(gradr)
        scan = collect_scan_finer_grid(point, r0 + step*np.asscalar(prevgrad[0]), c0 + step*np.asscalar(prevgrad[1]) + deltac,snr,o_angl,nv1,nh1,2)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])
        #fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        #fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 + step1*np.asscalar(prevgrad[0]), c0 + step1*np.asscalar(prevgrad[1]) - deltac,snr,o_angl,nv1,nh1,2)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        gradc = (fcplus - fcminus) / (2 * deltac)
        # print(gradc)

        gradv = np.array([gradr,gradc])
        gradv = gradv.reshape(2,1)

        #gradv_sum = gradv.reshape(2, 1)

        #for j in range(k):
        #    gradv_sum = gradv + momentum_vec[j] * prevgrad #+ step * gradv + momentum_vec[1] * prevgrad + step * gradv + momentum_vec[2] * prevgrad
        gradv = momentum*prevgrad + step * gradv
        #gradv = step * gradv + momentum * prevgrad
        v1 = v1 + gradv
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        #r0 = r0 + step * gradr
        #c0 = c0 + step * gradc
        #angle = np.arccos(gradv.transpose().dot(prevgrad)/(np.linalg.norm(gradv)*np.linalg.norm(prevgrad)))*180/np.pi
        #angles.append(angle)

        coordr01.append(r0)
        coordc01.append(c0)

        prevgrad = gradv


        #gradv = gradv_sum*(1/k)
        #inner_products[counter] = gradv.transpose().dot(prevgrad)
        #counter += 1
        '''
        if counter == 5:
            counter = 0
            inner_mean = np.mean(inner_products)
            print(inner_mean)
        '''
        '''
        coordr01.append(r0)
        coordc01.append(c0)

        plt.title('K=' + str(i))
        plt.plot(coordr01, coordc01, 'go-', markersize='1')
        plt.savefig('Image_GA' + str(i) + '.png')
        '''

        scan = collect_scan_finer_grid(point, r0, c0,snr,o_angl,nv1,nh1,2)
        if corr == 1:
            scan = matched_filter(g, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        temp = np.sum(scan[index - samples:index + samples])
        #temp = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        Energy.append(temp)
        print('current echo pseudo-energy: ' + str(temp)+ '  ' + str(r0) + '  ' + str(c0) + ' ' + str(step1) + '\n')
    return Energy,coordr01,coordc01


def gradient_ascent_momentum(r0,c0,step,momentum,dr,dc,iters,snr,corr,o_ang,samples,nv1,nh1,pulse,pulse_sel):
    deltar = 4*dr
    deltac = 4*dc
    k = 3
    perturbation = 0.000001  # this is just in case an angle DOES fall on the grid
    prevgrad = np.ones((2, 1))*1e-6
    v = np.array([r0, c0])
    v1 = np.reshape(v, (2, 1))
    #cvals[0] = r0
    #rvals[0] = c0
    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    Energy = []
    print('Inside gradient ascent')
    point = np.array([10,0,0])*1e-3

    #momentum_vec = np.zeros(k)
    #a = 0.1

    #for l in range(k):
        #momentum_vec[l] = 1 - a ** ((l+1) - 1)

    for k in range(iters):
        scan = collect_scan_finer_grid(point, r0 + deltar, c0, snr,o_ang,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frplus = np.sum(scan[index - samples:index + samples])
        #frplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + deltar, c0, dr, dc)[1500:])
        #frminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - deltar, c0, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0 - deltar, c0, snr,o_ang,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        frminus = np.sum(scan[index - samples:index + samples])

        gradr = (frplus - frminus) / (2 * deltar)

        scan = collect_scan_finer_grid(point, r0, c0 + deltar, snr,o_ang,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcplus = np.sum(scan[index - samples:index + samples])
        #fcplus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 + deltar, dr, dc)[1500:])
        #fcminus = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0 - deltar, dr, dc)[1500:])
        scan = collect_scan_finer_grid(point, r0, c0 - deltar, snr,o_ang,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        fcminus = np.sum(scan[index - samples:index + samples])

        gradc = (fcplus - fcminus) / (2 * deltac)

        gradv = np.array([gradr,gradc])
        gradv = gradv.reshape(2,1)

        gradv = step * gradv + momentum * prevgrad
        v1 = v1 + gradv
        r0 = np.asscalar(v1[0])
        c0 = np.asscalar(v1[1])

        #r0 = r0 + step * gradr
        #c0 = c0 + step * gradc

        coordr01.append(r0)
        coordc01.append(c0)

        prevgrad = gradv
        scan = collect_scan_finer_grid(point, r0, c0, snr,o_ang,nv1,nh1,pulse_sel)
        if corr == 1:
            scan = matched_filter(pulse, scan)
        else:
            scan = np.abs(ssig.hilbert(scan))
        index = np.argmax(scan)
        temp = np.sum(scan[index - samples:index + samples])
        Energy.append(temp)
        print('current echo pseudo-energy: ' + str(temp) + '  ' + str(r0) + '  ' + str(c0) + '\n')

    return Energy,coordr01,coordc01


def spsa_adaptive(a,A,gamma,c,alpha,iters,r0,c0):
    coordr0 = np.zeros((iters + 1))
    coordc0 = np.zeros((iters + 1))
    Energy = np.zeros((iters))

    Prevr0 = r0
    Prevc0 = c0

    deltar = np.random.binomial(1, 0.5)
    deltar = 2 * deltar - 1

    deltac = np.random.binomial(1, 0.5)
    deltac = 2 * deltac - 1

    step = 0
    ck = c / ((0 + 1) ** gamma)

    fplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + ck * deltar, c0 + ck * deltac, dr, dc)[1500:])
    fminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - ck * deltar, c0 - ck * deltac, dr, dc)[1500:])

    PrevEnergy = max(fplus, fminus)

    coordr01 = []
    coordc01 = []
    coordr01.append(r0)
    coordc01.append(c0)
    '''
    plt.imshow(np.fliplr(interpolated), extent=[r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]], aspect='auto')
    plt.plot(coordr01, coordc01, 'r*',label='SPSA Adaptive')
    plt.legend(loc="upper right", prop={"size": 10})
    plt.xlim(-4, 4)
    plt.ylim(-3, 6)
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.colorbar()
    '''
    for n in range(iters):
        # PrevEnergy = MaxVal

        step = a / ((n + A + 1) ** alpha)
        ck = c / ((n + 1) ** gamma)

        deltar = np.random.binomial(1, 0.5)
        deltar = 2 * deltar - 1

        deltac = np.random.binomial(1, 0.5)
        deltac = 2 * deltac - 1

        fplus = np.sum(interpolate(measurements, coordinates - perturbation, r0 + ck * deltar, c0 + ck * deltac, dr, dc)[1500:])
        fminus = np.sum(interpolate(measurements, coordinates - perturbation, r0 - ck * deltar, c0 - ck * deltac, dr, dc)[1500:])

        gradr = (fplus - fminus) / (2 * deltar)

        gradc = (fplus - fminus) / (2 * deltac)

        # minVal = max(fplus,fminus)
        # print(minVal,PrevEnergy)

        MaxVal = max(fplus, fminus)

        if abs(MaxVal - PrevEnergy) < 0:
            #print('B')
            r0 = Prevr0
            c0 = Prevc0
            step *= 0.5
        else:
            #print('A')
            r0 = r0 + step * gradr
            c0 = c0 + step * gradc
            Prevr0 = r0
            Prevc0 = c0
            PrevEnergy = MaxVal

        energy = np.sum(interpolate(measurements, coordinates - perturbation, r0, c0, dr, dc)[1500:])
        coordr0[n + 1] = r0
        coordc0[n + 1] = c0
        '''
        coordr01.append(r0)
        coordc01.append(c0)
        plt.title('K=' + str(n))
        plt.plot(coordr01, coordc01, 'ro-', markersize='1')
        plt.savefig('Image_SPSA_AS' + str(n) + '.png')
        '''
        Energy[n] = energy

    return Energy,coordr0,coordc0

# %%interpolator
def interpolate(function, coordinates, r, c, dr, dc):
    '''
    This function takes as input samples of a 2D function and interpolates a
    point (r,c) that doesn't fall on the grid via convex combination of 3
    points.

    Parameters
    ----------
    function : matrix that contains samples of a function taken on a regular
        grid with spacing dr, dc. These samples will be used to interpolate
        the points in between the grid.
    coordinates : matrix with the same size as 'function' that contains the
        values of the independent variables on which the function depends.
        This grid is necessary in order to find the closest 3 points to be
        used for convex combination.
    r : value of the first variable at which we would like to interpolate.
    c : value of the second variable at which we would like to interpolate.
    dr : grid spacing for the first variable in the 'coordinates' matrix.
    dc : grid spacing for the second variable in the 'coordinates' matrix.

    Returns
    -------
    interp : value of the function at point (r,c) as given by a convex
    combination of the 3 nearest points, according to the coordinates matrix.
    '''
    #global iters

    # offset the variables so that they match the grid origin at [0,0]
    r_orig = coordinates[0, 0, 0]
    c_orig = coordinates[1, 0, 0]
    r = r - r_orig
    c = c - c_orig
    # print(r,c,r_orig,c_orig)
    # print(coordinates[0,0,0])
    # print(coordinates[1,0,0])
    # need to convert the variables into indices on the coordinate grid
    rn = r / dr  # first, normalize them
    cn = c / dc
    # print(rn, cn)
    # the variables where we want to evaluate should now be indices in the
    # provided coord system

    # find the 4 points on the grid closest to (r,c)
    A = np.floor(np.array([rn, cn])).astype(int)
    B = np.array([np.ceil(rn), np.floor(cn)]).astype(int)
    C = np.ceil(np.array([rn, cn])).astype(int)
    D = np.array([np.floor(rn), np.ceil(cn)]).astype(int)

    # print(A,B,C,D)
    points = np.zeros((2, 4), dtype=int)  # place the points as cols of a mat
    points[:, 0] = A
    points[:, 1] = B
    points[:, 2] = C
    points[:, 3] = D
    # print(points)
    # find the l1 distance from the desired point to the 4 grid neighbors
    manhattan = nlin.norm(points - np.array([rn, cn]).reshape(2, 1), 1, axis=0)
    # print(points - np.array([rn,cn]).reshape(2,1))
    # print(manhattan)
    # print(np.array([rn,cn]).reshape(2,1))
    # then, discard the point furthest away from the desired point
    discard = np.argmax(manhattan)
    # print(discard)

    points = np.delete(points, discard, axis=1)
    # print(points)
    # interpolate as a convex combination of the points in barycentric-coords
    # first, extend the coordinate system
    matrix = np.concatenate((points, np.ones((1, 3))), axis=0)
    # print(matrix)
    # find the weights by left-multiplying the extended target point by the inv
    w = nlin.inv(matrix).dot(np.array([rn, cn, 1]))
    # print(np.array([rn,cn,1]))
    # print(w)
    # and finally, find the interpolated value
    #print(function.shape[0])
    #print(points)
    gridded = np.zeros((function.shape[0], 3))
    gridded[:, 0] = function[:, points[0, 0], points[1, 0]]
    #print(function[:, points[0, 0], points[1, 0]])
    gridded[:, 1] = function[:, points[0, 1], points[1, 1]]
    gridded[:, 2] = function[:, points[0, 2], points[1, 2]]
    # print(len(gridded.dot(w)))

    return gridded.dot(w)


# %%main
# read the data, reshape it nicely, and keep its envelope

measurements = sio.loadmat('angular_measurements.mat')
measurements = np.transpose(measurements['data'], (2, 0, 1))

# flip every other slice, since the data was collected in zig-zag
for i in np.arange(0, 41, 2):
    measurements[:, i, :] = np.fliplr(np.squeeze(measurements[:, i, :]))
#print(len(measurements))
Test = measurements[:, 21, 24]

# plt.plot(Test,color='red')
# plt.show()
measurements = np.power(np.abs(ssig.hilbert(measurements, axis=0)), 2)
measurements1 = measurements[:, 19, 20]
EnergyOfTheSignal = np.sum(measurements1[1700:2048])
#print(EnergyOfTheSignal)
#plt.plot(measurements1, color='green')
#plt.show()
#lowpass filter the envelopes to make the back wall echo look nicer
smoothness = 60
for i in range(measurements.shape[1]):
    for j in range(measurements.shape[2]):
        measurements[:, i, j] = np.convolve(measurements[:, i, j], np.ones(smoothness) / smoothness, mode='same')
# %% cost function plots
# make plots of the cost function
# original coords
coords_r = np.arange(-15, 16)
coords_c = np.arange(-15, 16)
coordinates = np.zeros((2, 31, 31))
coordinates[0, :, :] = coords_r
coordinates[1, :, :] = coords_c
print(coordinates[0,15,0])
print(coordinates[1,0,15])
original = np.sum(measurements[1500:], axis=0)  # original cost func
coordinates2 = np.reshape(coordinates,(31,31,2))
print(coordinates2[15,15,:])
dr = 1
dc = 1

coords_r1 = np.arange(-15, 15, dr)
coords_c1 = np.arange(-15, 15, dc)
coords_c2 = np.arange(-15, 12, 3)

interp_factor = 1
interpolated = np.zeros((original.shape[0] * interp_factor, original.shape[1] * interp_factor))
#interpolated2 = np.zeros((NT,coords_r1.shape[0],coords_c1.shape[0]))
#interpolated3 = np.zeros((coords_r1.shape[0],coords_c1.shape[0]))

#interpolated2 = sio.loadmat('C:/Users/user/Documents/ARP_data_new/Simulator_data.mat')
#interpolated2 = interpolated2['Simulator']

#interpolated2 = list(interpolated2.items())
#interpolated2 = np.array(interpolated2)
#interpolated2 = interpolated2['data']

perturbation = 0.000001
# interpolated coords
coords_r = np.arange(-15, 15, 1 / interp_factor) * dr
coords_c = np.arange(-15, 15, 1 / interp_factor) * dc

r_plot = coords_r  # for use in plotting later
c_plot = coords_c  # for use in plotting later
'''
for i in range(len(coords_r)):
    for j in range(len(coords_c)):
        r = coords_r[i]
        c = coords_c[j]
        interpolated[j,i] = np.sum(interpolate(measurements, coordinates-perturbation, r, c, dr, dc)[1500:])
'''
'''
for i in range(len(coords_r1)):
    for j in range(len(coords_c1)):
        index = np.argmax(interpolated2[:, j, i])
        #print(index)
        interpolated3[j, i] = np.sum(interpolated2[index - 15:index + 15,j, i])

X = np.arange(-15,15,1)
Y = np.arange(-15,15,1)
X, Y = np.meshgrid(X, Y)
'''
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(X,Y,interpolated3,cmap='viridis', linewidth=0, antialiased=False)
#ax.set_zlim(0, 30)
#plt.show()

#x_range = np.arange(-15, 16, 1)
y_range = np.arange(0, 10, 1)
interp_r = np.arange(-15, 15, 1)
interp_c = np.arange(-15, 15, 1)
all_measurements = np.zeros((NT,coords_r1.shape[0],coords_c1.shape[0]))
interpolated1 = np.zeros((interp_r.shape[0],interp_c.shape[0]))
coordinates1 = np.zeros((2, coords_r1.shape[0],coords_c1.shape[0]))
coordinates1[0, :, :] = coords_r1
coordinates1[1, :, :] = coords_c1

'''
for k in range(len(x_range)):
    for l in range(len(y_range)):
        for i in range(len(coords_r1)):
            for j in range(len(coords_c1)):
                o_point = np.array([x_range[k], y_range[l], 0]) * 1e-3
                r = coords_r1[i]
                c = coords_c1[j]
                scan = collect_scan_finer_grid(o_point, r, c, 10, 15, 181, 181, 2)
                all_measurements[:, i, j] = scan
                scan = matched_filter(g, scan)
                index = np.argmax(scan)
                fplus = np.sum(scan[index - 15:index + 15])
                interpolated1[j, i] = fplus
                print(fplus)
                #some_matrix = np.array(all_measurements[:,k,l,:,:])
                #interpolated1[j, i] = fplus
                #print(r,c,x_range[k], y_range[l])
        
        for i in range(len(interp_r)):
            for j in range(len(interp_c)):
                r = interp_r[i]
                c = interp_c[j]
                scan = interpolate(all_measurements, coordinates1 - perturbation, r, c, dr, dc)
                scan = matched_filter(g, scan)
                index = np.argmax(scan)
                fplus = np.sum(scan[index - 15:index + 15])
                interpolated1[j, i] = fplus
                #print(fplus)

        
        adict = {"Simulator": interpolated1, "label": "experiment"}
        adict['interpolated1'] = interpolated1
        name_str = 'C:/Users/user/Documents/Master_Thesis_CostFunction2/' + str(x_range[k]) + '_' + str(y_range[l]) + '.mat'
        print(name_str)
        sio.savemat(name_str, adict)
'''
'''
for i in range(len(coords_r1)):
    for j in range(len(coords_c1)):
        o_point = np.array([10, 0, 0]) * 1e-3
        r = coords_r1[i]
        c = coords_c1[j]
        scan = collect_scan_finer_grid(o_point, r, c, 20, 15, 181, 181, 2)
        all_measurements[:, i, j] = scan
        scan = matched_filter(g, scan)
        index = np.argmax(scan)
        fplus = np.sum(scan[index - 15:index + 15])
        interpolated1[j, i] = fplus
        print(fplus)
        #some_matrix = np.array(all_measurements[:,k,l,:,:])
        #interpolated1[j, i] = fplus
        #print(r,c,x_range[k], y_range[l])
        # for i in range(len(interp_r)):
        #     for j in range(len(interp_c)):
        #         r = interp_r[i]
        #         c = interp_c[j]
        #         scan = interpolate(all_measurements, coordinates1 - perturbation, r, c, dr, dc)
        #         scan = matched_filter(g, scan)
        #         index = np.argmax(scan)
        #         fplus = np.sum(scan[index - 15:index + 15])
        #         interpolated1[j, i] = fplus
        #         #print(fplus)


x_range1 = np.arange(10,20,10)
adict = {"Simulator": interpolated1, "label": "experiment"}
adict['interpolated1'] = interpolated1
name_str = 'C:/Users/user/Documents/Master_Thesis_CostFunction2/' + str(x_range1[0]) + '_' + str(y_range[0])+'_20_dB' + '.mat'
print(name_str)
sio.savemat(name_str, adict)
'''


#plt.figure(6)
#plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
#plt.imshow(interpolated,extent = [-20, 20, -20, 20])
x_range = np.arange(10,20,10)
name_str = 'C:/Users/user/Documents/Master_Thesis_CostFunction2/' + str(x_range[0]) + '_' + str(y_range[0]) + '_20_dB' + '.mat'
interpolated1 = sio.loadmat(name_str)
interpolated1 = interpolated1['Simulator']
x_range = x_range*1e-3
#coords_r = np.arange(-20, 21).reshape(41, 1) * dr + np.zeros((1, 41))
#coords_c = np.arange(-20, 21).reshape(1, 41) * dc + np.zeros((41, 1))
#coordinates = np.zeros((2, 41, 41))
#coordinates[0, :, :] = coords_r
#coordinates[1, :, :] = coords_c

# initial guess on the angles
r0 = 0
c0 = 0
r0_1 = r0
c0_1 = c0
iters = 30

iters1 = 20
grad_iters = 100
outeriters = 2

energy_range = np.arange(0,iters,1)

index1,index2 = np.where(interpolated1 == np.max(interpolated1))
print(index1,index2)
#Energy4_Fi = np.zeros(iters)
#coordr0 = np.zeros(iters)
#coordc0 = np.zeros(iters)
#coordr01 = np.zeros(iters)
#coordc01 = np.zeros(iters)
#coordr02 = np.zeros(iters)
#coordc02 = np.zeros(iters)
#coordr03 = np.zeros(iters)
#coordc03 = np.zeros(iters)
#coordr04 = np.zeros(iters)
#coordc04 = np.zeros(iters)
snr_range = np.arange(10,60,10)
x_axis_indices = np.arange(iters)
Energy_Fi = np.zeros((snr_range.shape[0],iters))
Energy1_Fi = np.zeros((snr_range.shape[0],iters))
Energy2_Fi = np.zeros((snr_range.shape[0],iters))
Energy3_Fi = np.zeros((snr_range.shape[0],iters))
Energy4_Fi = np.zeros((snr_range.shape[0],iters))
Energy5_Fi = np.zeros((snr_range.shape[0],iters))
coordr04_t = np.zeros((snr_range.shape[0],iters+1))
coordc04_t = np.zeros((snr_range.shape[0],iters+1))
coordr05_t = np.zeros((snr_range.shape[0],iters+1))
coordc05_t = np.zeros((snr_range.shape[0],iters+1))
coordr06_t = np.zeros((snr_range.shape[0],iters+1))
coordc06_t = np.zeros((snr_range.shape[0],iters+1))
coordr03 = np.zeros((snr_range.shape[0],iters+1))
coordc03 = np.zeros((snr_range.shape[0],iters+1))
coordr01_t = np.zeros((x_range.shape[0],iters+1))
coordc01_t = np.zeros((x_range.shape[0],iters+1))
coordr02_t = np.zeros((snr_range.shape[0],iters+1))
coordc02_t = np.zeros((snr_range.shape[0],iters+1))
coordr03_t = np.zeros((snr_range.shape[0],iters+1))
coordc03_t = np.zeros((snr_range.shape[0],iters+1))
Final_scans = np.zeros((snr_range.shape[0], 5, NT))
point_x = np.zeros(x_range.shape[0])
point_y = np.zeros(x_range.shape[0])
point_z = np.zeros(x_range.shape[0])
peak_index_list = np.zeros(iters)

point_x1 = np.zeros(iters+1)
point_y1 = np.zeros(iters+1)
point_z1 = np.zeros(iters+1)
direction_vec = np.zeros((3,x_range.shape[0]))
distances = np.zeros(x_range.shape[0])
#energy_c,rvalc,cvalc,angles3 = conjugate_gradient(r0,c0,0.05,dr,dc,50)
#energy_n, rval, cval = newton_method(r0, c0, 0.1, dr, dc, iters)
#hess = compute_hessian_init(r0, c0, 0.1, dr, dc, iters)
#energy_n1, rval1, cval1 = quasi_newton_method(r0,c0,0.01,dr,dc,100,1,hess)
'''
Energy3,coordr02,coordc02 = gradient_accelerated(0, 0, 0.5, 0.8, dr, dc, iters, snr_range[2])
Energy2,coordr01,coordc01 = gradient_ascent_momentum(0, 0, 0.5, 0.8, dr, dc, iters, snr_range[2])
Energy1,coordr0,coordc0 = gradient_ascent(0, 0, 0.5, dr, dc, iters,snr_range[2])
'''
'''
plt.figure(1)
plt.plot(Energy2,label='GA(momentum)')
plt.plot(Energy3,label='GA(Nesterov)')
plt.plot(Energy1,label='GA')
plt.legend()

plt.figure(2)
#plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
#plt.plot(coordr01[0],coordc01[0],'r*')
plt.plot(coordr01,coordc01,'ro-',color='r',label="GA(momentum)")
plt.plot(coordr02,coordc02,'mo-',color='m',label="GA(Nesterov)")
plt.plot(coordr0,coordc0,'yo-',color='y',label="GA")
#plt.plot(coordr01,coordc01,'mo-',color='m',label="Gradient Ascent(momentum)")
#plt.plot(coordr0,coordc0,'ro-',color='r',label="SPSA_Adaptive")
#plt.plot(rval,cval,'mo-',color='m',label="Newton's Method")
#plt.plot(coordr02,coordc02,'go-',color='g',label="Line Search")
#plt.plot(coordr03,coordc03,'yo-',color='y',label="RDSA")
#plt.plot(rval1,cval1,'mo-',color='m',label='Quasi newton')
#plt.plot(coordr04_t,coordc04_t,'ko-',color='k',label="SPSA")
#plt.plot(rvalc,cvalc,'ro-',color='r',label='Nonlinear CG')

plt.legend(loc="upper right",prop={"size": 8})
plt.colorbar()
plt.xlabel('Azimuth',weight='bold',fontsize=10)
plt.ylabel('Elevation',weight='bold',fontsize=10)
#Energy3, coordr03, coordc03 = rdsa(r0, c0, dr, dc, iters, 0.5, 0.101, 0.9, 0.9, 0.1, 0.35)
'''

array_length = len(coordr04_t[0, :])
#pdf = np.zeros((coe_range.shape[0],snr_range.shape[0]))
grad_vec = np.zeros((grad_iters,2))
grad_cov_mat = np.zeros((snr_range.shape[0],2))
param_cov_matrix = np.zeros((iters1,2))
param_cov_matrix1 = np.zeros((snr_range.shape[0],2))
#Energy2,coordr01,coordc01,products = gradient_ascent_momentum(0, 0, 0.5, 0.8, dr, dc, iters, snr_range[0])
prev_element = 0
prev_element1 = 0
# point = np.array([10, 10, 0]) * 1e-3
# scan = collect_scan_finer_grid(point, r0, c0,0)
# scan = matched_filter(g, scan)
# plt.figure(15)
# plt.plot(scan)'

o_a = 10
peak_index = 0
point = np.array([10,0,0])*1e-3

for i in range(x_range.shape[0]):
    Energy1_Fi[i, :], coordr01_t[i, :], coordc01_t[i, :] = spsa(point, r0, c0, 0.4, 0.2, 0.5, 0.101, 1, iters,snr_range[0], 1, 15, 15, 181, 181, g, 2)
    Energy2_Fi[i, :], coordr02_t[i, :], coordc02_t[i, :] = spsa(point, r0, c0, 0.4, 0.2, 0.5, 0.101, 1, iters,snr_range[1], 1, 15, 15, 181, 181, g, 2)   #newton_method(0, -2, 0.05, 4 * dr, 4 * dc, iters, snr_range[2],1,15,10,181,181,2)
    Energy3_Fi[i, :], coordr03_t[i, :], coordc03_t[i, :] = spsa(point, r0, c0, 0.4, 0.2, 0.5, 0.101, 1, iters,snr_range[2], 1, 15, 15, 181, 181, g, 2)   #gradient_accelerated(r0, c0, 0.05, 0.7, dr, dc, iters, snr_range[2], 1, 15,15,181,181,g,2)
    Energy4_Fi[i, :], coordr04_t[i, :], coordc04_t[i, :] = spsa(point, r0, c0, 0.4, 0.2, 0.5, 0.101, 1, iters,snr_range[3], 1, 15, 15, 181, 181, g, 2)
    Energy5_Fi[i, :], coordr05_t[i, :], coordc05_t[i, :] = spsa(point, r0, c0, 0.4, 0.2, 0.5, 0.101, 1, iters,snr_range[4], 1, 15, 15, 181, 181, g, 2)
    #Energy1_Fi[i, :], coordr06_t[i, :], coordc06_t[i, :] = gradient_ascent(r0, r0, 0.05, dr, dc, iters,snr_range[1],1,15,15,181,181,g,2)
    #Energy2_Fi[i, :], coordr02_t[i, :], coordc02_t[i, :] = gradient_ascent_momentum(r0, c0, 0.05, 0.85, dr, dc, iters,snr_range[1], 1, 15, 15, 181, 181,g,2)# newton_method(0, -2, 0.05, 4 * dr, 4 * dc, iters, snr_range[2],1,15,10,181,181,2)
    #Energy3_Fi[i, :], coordr03_t[i, :], coordc03_t[i, :] = gradient_accelerated(r0, c0, 0.05, 0.85, dr, dc, iters,snr_range[1], 1, 15, 15, 181, 181, g,2)# gradient_accelerated(r0, c0, 0.05, 0.7, dr, dc, iters, snr_range[2], 1, 15,15,181,181,g,2)
    #Energy4_Fi[i, :], coordr04_t[i, :], coordc04_t[i, :] = spsa(point, r0, c0, 0.4, 0.2, 0.5, 0.101, 1, iters,snr_range[1], 1, 15, 15, 181, 181, g, 2)
    #Energy5_Fi[i, :], coordr05_t[i, :], coordc05_t[i, :] = newton_method(r0, c0, 0.05, 4 * dr, 4 * dc, iters,snr_range[1], 1, 15, 15, 181, 181, 2)
    #Energy_Fi[i, :], coordr06_t[i, :], coordc06_t[i, :] = quasi_newton_method(r0, c0, 0.05, 4 * dr, 4 * dc, iters, 0.6,snr_range[1])


### added only for animation purpose
'''
for i in range(iters):
    tof = peak_index_list[i] * Ts
    d = tof * c / 2
    point_x1[i] = x_range[0] + d * np.sin((180 - coordc01_t[0,i]) * np.pi / 180) * np.cos(coordr01_t[0,i] * np.pi / 180)
    point_y1[i] = 0 + d * np.sin((180 - coordc01_t[0,i]) * np.pi / 180) * np.sin(coordr01_t[0,i] * np.pi / 180)
    point_z1[i] = d * np.cos((180 - coordc01_t[0,i]) * np.pi / 180)
    print(point_x1[i],point_y1[i],point_z1[i])
'''
'''
o_point3 = np.array([x_range[0],0,0])

for i in range(coords_c2.shape[0]):
    ascan = collect_scan_finer_grid(o_point3,0,coords_c2[i],20,15,181,181,2)
    y = matched_filter(g,ascan)
    index = np.argmax(y)
    fmax = np.sum(y[index - 15:index + 15])
    Energy3_Fi[0,i] = fmax
    tof = index * Ts
    d = tof * c / 2
    point_x1[i] = x_range[0] + d * np.sin((180 - coords_c2[i]) * np.pi / 180) * np.cos(0 * np.pi / 180)
    point_y1[i] = 0 + d * np.sin((180 - coords_c2[i]) * np.pi / 180) * np.sin(0 * np.pi / 180)
    point_z1[i] = d * np.cos((180 - coords_c2[i]) * np.pi / 180)
    print(fmax,coords_c2[i])
'''
###################################
# for i in range(coordr01_t[0,:].shape[0]-1):
#     Energy3_Fi[0,i] = interpolated3[np.int(coordr01_t[0,i]),np.int(coordc01_t[0,i])]
#
# #fig, ax = plt.subplots(subplot_kw={"projection": "3d"
'''
plt.figure(1)
plt.title('Accelerated Gradient Ascent')
plt.plot(Energy1_Fi[0,:],label='Momentum = 0.65',color='blue')
plt.plot(Energy2_Fi[0,:],label='Momentum = 0.7',color='magenta')
plt.plot(Energy3_Fi[0,:],label='Momentum = 0.8',color='green')
plt.plot(Energy4_Fi[0,:],label='Momentum = 0.85',color='red')
plt.xlabel('Iterations')
plt.ylabel('Pseudo Energy')
plt.legend(loc='lower right',fontsize=10)
tikzplotlib.save("C:/Users/user/Documents/test.tex")

plt.figure(2)
plt.imshow(np.flipud(interpolated1)/np.max(interpolated1),extent=[-20, 20, -20, 20])
plt.colorbar()
plt.plot(coordr01_t[0,0],coordc01_t[0,0],marker="*", markersize=25,markerfacecolor='black',label='Initial Guess')
plt.plot(np.int(coordinates[1,0,15]),np.int(coordinates[0,15,0]), marker="D", markersize=15,markerfacecolor='green',label='True Solution')
plt.plot(coordr04_t[0,:],coordc04_t[0,:],'ro-',label='Momentum = 0.85')
plt.plot(coordr03_t[0,:],coordc03_t[0,:],'go-',label='Momentum = 0.8')
plt.plot(coordr02_t[0,:],coordc02_t[0,:],'mo-',label='Momentum = 0.7')
plt.plot(coordr01_t[0,:],coordc01_t[0,:],'bo-',label='Momentum = 0.65')
plt.title('Path Traced')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper left',fontsize=6)
'''
'''
plt.figure(1)
plt.title('Accelerated gradient ascent')
plt.plot(Energy1_Fi[0,:]/np.max(Energy4_Fi[0,:]),label='Momentum = 0.65 ',color='yellow')
plt.plot(Energy2_Fi[0,:]/np.max(Energy4_Fi[0,:]),label='Momentum = 0.7',color='red')
plt.plot(Energy3_Fi[0,:]/np.max(Energy4_Fi[0,:]),label='Momentum = 0.8',color='magenta')
plt.plot(Energy4_Fi[0,:]/np.max(Energy4_Fi[0,:]),label='Momentum = 0.85',color='blue')
plt.xlabel('Iterations')
plt.ylabel('Pseudo Energy')
plt.legend(loc='upper right')
tikzplotlib.save("C:/Users/user/Documents/test.tex")

plt.figure(2)
plt.imshow(np.flipud(interpolated1)/np.max(interpolated1),extent=[-20, 20, -20, 20])
plt.colorbar()
plt.plot(coordr01_t[0,0],coordc01_t[0,0],marker="*", markersize=25,markerfacecolor='black',label='Initial Guess')
plt.plot(np.int(coordinates[1,0,15]),np.int(coordinates[0,15,0]), marker="D", markersize=15,markerfacecolor='green',label='True Solution')
plt.plot(coordr04_t[0,:],coordc04_t[0,:],'bo-',label='scaling factor = 0.85')
plt.plot(coordr03_t[0,:],coordc03_t[0,:],'mo-',label='scaling factor = 0.8')
plt.plot(coordr02_t[0,:],coordc02_t[0,:],'ro-',label='scaling factor = 0.7')
plt.plot(coordr01_t[0,:],coordc01_t[0,:],'yo-',label='scaling factor = 0.65')
plt.title('Path Traced')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper left')
'''
#tikzplotlib.save("C:/Users/user/Documents/test.tex")

m1 = -5/28.3564
m2 = 5/18.66025

x1 = np.linspace(-28.3564, 0, num=20, endpoint=True)
x2 = np.linspace(0, 18.66025, num=20, endpoint=True)
x = np.concatenate((x1,x2), axis=0)

z1 = (m1*(x1 + 28.3564) + 25)
z2 = (m2*(x2 - 18.66025) + 25)
z = np.concatenate((z1,z2), axis=0)

y = np.linspace(-10, 10, 10)

X,Y = np.meshgrid(x,y)
Z = nmat.repmat(z,len(y),1)

fig = plt.figure(figsize=(12,7))

#ax = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

T1, T2 = np.meshgrid(np.linspace(-18,18,30),np.linspace(-18,18,30))
#fig1, ax1 = plt.subplots(nrows=1, ncols=3,figsize=(12, 7))
#ax.set_title('Measurement Setup')
#fig2, ax2 = plt.subplots(figsize=(7, 7))
cont = ax1.contourf(T1, T2, interpolated1/np.max(interpolated1), 30, cmap='jet')
ax1.set_xlabel('Azimuth(degrees)')
ax1.set_ylabel('elevation(degrees)')
ax1.set_title('Algorithm Comparison')
cbar = fig.colorbar(cont,ticks=[0,0.2,0.4,0.6,0.8,1],ax=ax1)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Pseudo Energy')
ax2.set_title('Algorithm Comparison')
# ax.plot_surface(X,Y,Z)
# ax.set_zlim3d(15,25)
# ax.invert_zaxis()
# ax.plot([point[0]*1e3], [point[1]], [15], color='yellow', marker='o', markersize=10, alpha=0.8,label='Transducer')
# quiver = ax.quiver([point[0]*1e3], [point[1]*1e3], [15],[0], [0], [4], linewidths = (5,), edgecolor="red",label='Pulse direction')

'''
fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax5 = fig.add_subplot(2, 2, 2)
ax6 = fig.add_subplot(2, 2, 4)

line12, = ax5.plot([], [],'r-',label='Energy')
point12, = ax5.plot([], [], 'o', color='red', markersize=4)
line13, = ax6.plot([], [],'ro-',label='Path Traced')
point13, = ax6.plot([], [], '*', color='red', markersize=4)

ax6.contour(T1, T2, interpolated3, 30, cmap='jet')
ax6.set_xlabel('Azimuth(degrees)')
ax6.set_ylabel('Co elevation(degrees)')
ax5.set_xlim(0,iters)
ax5.set_ylim(0,200)

o_point1 = np.array([x_range[0],0,0])
u, v = np.mgrid[0:2*np.pi:0.01, 0:np.pi:0.01]
x = radius*np.cos(u)*np.sin(v)
y = radius*np.sin(u)*np.sin(v)
z = radius*np.cos(v)
ax.plot_surface(x-center[0], y-center[1], z-center[2], color="g", alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.plot([o_point1[0]], [o_point1[1]], [o_point1[2]], color='yellow', marker='o', markersize=10, alpha=0.8,label='Transducer')
quiver = ax.quiver([o_point1[0]], [o_point1[1]], [o_point1[2]],[-point_x[0]], [point_y[0]], [point_z[0]], linewidths = (5,), edgecolor="red",label='Normal')
'''
# fig2 = plt.figure()
# ax2 = fig2.gca(projection='3d')
'''
o_point2 = np.array([x_range[0],0,0])

fig2 = plt.figure(figsize=(10,5))
ax2 = fig2.add_subplot(1, 2, 1, projection='3d')
ax2.set_title('Measurement Setup')
ax3 = fig2.add_subplot(1, 2, 2)
#ax4 = fig2.add_subplot(2, 2, 4)
#ax4.contour(T1, T2, interpolated3, 30, cmap='jet')
#ax4.set_xlabel('Azimuth(degrees)')
#ax4.set_ylabel('Co elevation(degrees)')
ax3.set_xlim(0,iters)
ax3.set_ylim(0,200)
ax3.set_title('Cost Function')

line10, = ax3.plot([], [],'r-',label='Energy')
point10, = ax3.plot([], [], 'o', color='red', markersize=4)
#line11, = ax4.plot([], [],'ro-',label='Path Traced')
#point11, = ax4.plot([], [], '*', color='red', markersize=4)

lista, listb = [], []

m1 = -5/28.3564
m2 = 5/18.66025

x1 = np.linspace(-28.3564e-3, 0, num=20, endpoint=True)
x2 = np.linspace(0, 18.66025e-3, num=20, endpoint=True)
x = np.concatenate((x1,x2), axis=0)

z1 = (m1*(x1 + 28.3564e-3) + 25e-3)
z2 = (m2*(x2 - 18.66025e-3) + 25e-3)
z = np.concatenate((z1,z2), axis=0)

y = np.linspace(-10, 10, 10)

X,Y = np.meshgrid(x,y)
Z = nmat.repmat(z,len(y),1)

ax2.plot_surface(X, Y, Z)

ax2.set_xlabel('x-axis [m]')
ax2.set_ylabel('y-axis [m]')
ax2.set_zlabel('depth [m]')
ax2.invert_zaxis()

ax2.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o', markersize=10, alpha=0.8,label='Transducer')
quiver2 = ax2.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [-point_x[0]], [point_y[0]], [-point_z[0]],linewidths=(5,), edgecolor="red", label='Direction')
value_display10 = ax2.text(-2*1e-3, -2*1e-3, 0, '')

ax3.set_xlabel('Iterations')
ax3.set_ylabel('Energy')


def init_3():
     line10.set_data([], [])
     point10.set_data([], [])
     #line11.set_data([],[])
     #point11.set_data([],[])
     return line10,point10#,line11,point11


def animate_3(counter):
    global quiver2
    quiver2.remove()
    #ax3.plot([energy_range[counter+1]],[Energy3_Fi[0,counter+1]])
    ax2.view_init(elev=8, azim=-90)
    o_point2 = np.array([x_range[0], 0, 0])
    ax2.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o', markersize=10, alpha=0.8,label='Transducer')
    quiver2 = ax2.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [-point_x1[counter]], [point_y1[counter]], [-point_z1[counter]],linewidths=(5,), edgecolor="red", label='Direction')
    #ax3.plot([energy_range[:counter]],[Energy3_Fi[0,:counter]])
    #ax3.clear()
    ax3.set_ylim(np.min(Energy3_Fi[0,:])-10,np.max(Energy3_Fi[0,:]+10))
    point10.set_data([energy_range[counter]], [Energy3_Fi[0, counter]])
    line10.set_data([energy_range[:counter+1]], [Energy3_Fi[0,:counter+1]])
    #point11.set_data([coordr01_t[0, counter]], [coordc01_t[0, counter]])
    #line11.set_data([coordr01_t[0,:counter + 1]], [coordc01_t[0, :counter + 1]])
    #point11.set_data([coordr01_t[0, counter]], [coords_c1[counter]])s
    #line11.set_data([coordr01_t[0, :counter + 1]], [coords_c1[:counter + 1]])
    #lista.append(energy_range[counter])
    #listb.append(Energy3_Fi[0,counter])
    #line10.set_data(lista, listb)
    #point10.set_data(lista[counter], listb[counter])
    #ax3.clear()
    #ax3.plot([energy_range[:counter]], [Energy3_Fi[0,:counter]])
    #ax3.canvas.draw()
    print(Energy3_Fi[0,counter])
    return line10,point10#,line11,point11
    #value_display10.set_text('distance = ' + str(distances[i] * 1e3) + ' mm')


ax2.legend()
ax3.legend()
#ax4.legend()
'''
'''
def init_2():
    line12.set_data([], [])
    point12.set_data([], [])
    line13.set_data([], [])
    point13.set_data([], [])
    return line12,point12,line13,point13


def animate_2(counter):
    global quiver
    quiver.remove()
    ax.view_init(elev=13, azim=-93)
    o_point1 = np.array([x_range[0], 0, 0])
    ax.plot([o_point1[0]], [o_point1[1]], [o_point1[2]], color='yellow', marker='o', markersize=10, alpha=0.8,label='Transducer')
    quiver = ax.quiver([o_point1[0]], [o_point1[1]], [o_point1[2]], [-point_x1[counter]], [point_y1[counter]], [point_z1[counter]],linewidths=(5,), edgecolor="red", label='Normal')
    ax5.set_ylim(np.min(Energy3_Fi[0, :]) - 10, np.max(Energy3_Fi[0, :] + 10))
    point12.set_data([energy_range[counter]], [Energy3_Fi[0, counter]])
    line12.set_data([energy_range[:counter + 1]], [Energy3_Fi[0, :counter + 1]])
    point13.set_data([coordr01_t[0, counter]], [coordc01_t[0, counter]])
    line13.set_data([coordr01_t[0, :counter + 1]], [coordc01_t[0, :counter + 1]])
    return line12,point12,line13,point13


ax.legend()
ax5.legend()
ax6.legend()
'''

# Create animation
#line, = ax1.plot([], [], 'y', label='SNR = 10 dB', lw=1.5)
#point0, = ax1 .plot([], [], '*', color='yellow', markersize=10)
line1, = ax1.plot([], [], 'b', label='SNR = 20 dB', lw=1.5)
point1, = ax1.plot([], [], '*', color='blue', markersize=10)
'''
line2, = ax1.plot([], [], 'g', label='Accelerated gradient ascent', lw=1.5)
point2, = ax1.plot([], [], '*', color='green', markersize=10)
line3, = ax1.plot([], [], 'm', label='SPSA', lw=1.5)
point3, = ax1.plot([], [], '*', color='magenta', markersize=10)
line4, = ax1.plot([], [], 'k', label='Newtons Method', lw=1.5)
point4, = ax1.plot([], [], '*', color='black', markersize=10)
line11, = ax1.plot([], [], 'c', label='Quasi Newtons Method', lw=1.5)
point11, = ax1.plot([], [], '*', color='cyan', markersize=10)
'''
point5, = ax1.plot(np.int(coordinates[1,0,15]),np.int(coordinates[0,15,0]), 'D', color='cyan', markersize=15,label='True Solution')
point13, = ax1.plot([0],[0], '*', color='white', markersize=15,label='Initial guess')

line6, = ax2.plot(x_axis_indices, Energy1_Fi[0,:], 'y', label='SNR = 10 dB', lw=1.5)
point6, = ax2.plot(x_axis_indices[0], Energy1_Fi[0,0], '*', color='yellow', markersize=10)
line7, = ax2.plot(x_axis_indices, Energy2_Fi[0,:], 'b', label='SNR = 20 dB', lw=1.5)
point7, = ax2.plot(x_axis_indices[0], Energy2_Fi[0,0], '*', color='blue', markersize=10)
line8, = ax2.plot(x_axis_indices, Energy3_Fi[0,:], 'g', label='SNR = 30 dB', lw=1.5)
point8, = ax2.plot(x_axis_indices[0], Energy3_Fi[0,0], '*', color='green', markersize=10)
line9, = ax2.plot(x_axis_indices, Energy4_Fi[0,:], 'm', label='SNR = 40 dB', lw=1.5)
point9, = ax2.plot(x_axis_indices[0], Energy4_Fi[0,0], '*', color='magenta', markersize=10)
line10, = ax2.plot(x_axis_indices, Energy5_Fi[0,:], 'k', label='SNR = 50 dB', lw=1.5)
point10, = ax2.plot(x_axis_indices[0], Energy5_Fi[0,0], '*', color='black', markersize=10)
#line12, = ax2.plot(x_axis_indices, Energy_Fi[0,:], 'c', label='SNR = 10 dB', lw=1.5)
#point12, = ax2.plot(x_axis_indices[0],Energy_Fi[0,0] , '*', color='cyan', markersize=10)

'''
def init():
    point6.set_data(x_axis_indices[0], Energy1_Fi[0,0])
    line6.set_data(x_axis_indices, Energy1_Fi[0,:])
    point7.set_data(x_axis_indices[0], Energy2_Fi[0, 0])
    line7.set_data(x_axis_indices, Energy2_Fi[0, :])
    point8.set_data(x_axis_indices[0], Energy3_Fi[0, 0])
    line8.set_data(x_axis_indices, Energy3_Fi[0, :])
    point9.set_data(x_axis_indices[0], Energy4_Fi[0, 0])
    line9.set_data(x_axis_indices, Energy4_Fi[0, :])
    point10.set_data(x_axis_indices[0], Energy5_Fi[0, 0])
    line10.set_data(x_axis_indices, Energy5_Fi[0, :])
    return line6, point6,line7,point7,line8,point8,line9,point9,line10,point10


def animate_2(i):
    print(x_axis_indices[i-1], Energy1_Fi[0, i-1])
    point6.set_data(x_axis_indices[i-1], Energy1_Fi[0, i-1])
    line6.set_data(x_axis_indices[:i], Energy1_Fi[0, :i])
    point7.set_data(x_axis_indices[i-1], Energy2_Fi[0, i-1])
    line7.set_data(x_axis_indices[:i], Energy2_Fi[0, :i])
    point8.set_data(x_axis_indices[i-1], Energy3_Fi[0, i-1])
    line8.set_data(x_axis_indices[:i], Energy3_Fi[0, :i])
    point9.set_data(x_axis_indices[i-1], Energy4_Fi[0, i-1])
    line9.set_data(x_axis_indices[:i], Energy4_Fi[0, :i])
    point10.set_data(x_axis_indices[i-1], Energy5_Fi[0, i-1])
    line10.set_data(x_axis_indices[:i], Energy5_Fi[0, :i])
    return line6, point6,line7,point7,line8,point8,line9,point9,line10,point10
'''


def init_1():
    #line.set_data([], [])
    #point0.set_data([], [])
    line1.set_data([], [])
    point1.set_data([], [])
    #
    # line2.set_data([], [])
    # point2.set_data([], [])
    #
    # line3.set_data([], [])
    # point3.set_data([], [])
    #
    # line4.set_data([], [])
    # point4.set_data([], [])
    #
    # line11.set_data([], [])
    # point11.set_data([],[])

    point6.set_data(x_axis_indices[0], Energy1_Fi[0, 0])
    line6.set_data(x_axis_indices, Energy1_Fi[0, :])
    point7.set_data(x_axis_indices[0], Energy2_Fi[0, 0])
    line7.set_data(x_axis_indices, Energy2_Fi[0, :])
    point8.set_data(x_axis_indices[0], Energy3_Fi[0, 0])
    line8.set_data(x_axis_indices, Energy3_Fi[0, :])
    point9.set_data(x_axis_indices[0], Energy4_Fi[0, 0])
    line9.set_data(x_axis_indices, Energy4_Fi[0, :])
    point10.set_data(x_axis_indices[0], Energy5_Fi[0, 0])
    line10.set_data(x_axis_indices, Energy5_Fi[0, :])
    #point12.set_data(x_axis_indices[0], Energy_Fi[0, 0])
    #line12.set_data(x_axis_indices, Energy_Fi[0, :])

    return line1, point1, line6, point6, line7, point7, line8, point8, line9, point9, line10, point10
    #return line, point0,  line1, point1,  line2, point2, line3, point3,  line4, point4,line6, point6,line7,point7,line8,point8,line9,point9,line10,point10#line11,point11,line12,point12


def animate_1(i):
    # Animate line
    #global quiver
    #quiver.remove()
    #ax.view_init(elev=7, azim=-88)
    #ax.plot([point[0]*1e3], [point[1]], [15], color='yellow', marker='o', markersize=10, alpha=0.8, label='Transducer')
    #quiver = ax.quiver([point[0]*1e3], [point[1]], [15], [-point_x1[i]*1e3], [point_y1[i]*1e3],[4], linewidths=(5,), edgecolor="red", label='Pulse direction')

    #point0.set_data(coordr01_t[0, i-1], coordc01_t[0, i-1])

    point1.set_data(coordr02_t[0, i-1], coordc02_t[0, i-1])

    #point2.set_data(coordr03_t[0, i-1], coordc03_t[0, i-1])

    #point3.set_data(coordr04_t[0, i-1], coordc04_t[0, i-1])

    #point4.set_data(coordr05_t[0, i-1], coordc05_t[0, i-1])

    #point11.set_data(coordr06_t[0, i - 1], coordc06_t[0, i - 1])

    point6.set_data(x_axis_indices[i - 1], Energy1_Fi[0, i - 1])

    point7.set_data(x_axis_indices[i - 1], Energy2_Fi[0, i - 1])

    point8.set_data(x_axis_indices[i - 1], Energy3_Fi[0, i - 1])

    point9.set_data(x_axis_indices[i - 1], Energy4_Fi[0, i - 1])

    point10.set_data(x_axis_indices[i - 1], Energy5_Fi[0, i - 1])

    #point12.set_data(x_axis_indices[i - 1], Energy_Fi[0, i - 1])

    #line.set_data(coordr01_t[0, :i], coordc01_t[0, :i])

    line1.set_data(coordr02_t[0, :i], coordc02_t[0, :i])

    #line2.set_data(coordr03_t[0, :i], coordc03_t[0, :i])

    #line3.set_data(coordr04_t[0, :i], coordc04_t[0, :i])

    #line4.set_data(coordr05_t[0, :i], coordc05_t[0, :i])

    #line11.set_data(coordr06_t[0, :i], coordc06_t[0, :i])

    line6.set_data(x_axis_indices[:i], Energy1_Fi[0, :i])

    line7.set_data(x_axis_indices[:i], Energy2_Fi[0, :i])

    line8.set_data(x_axis_indices[:i], Energy3_Fi[0, :i])

    line9.set_data(x_axis_indices[:i], Energy4_Fi[0, :i])

    line10.set_data(x_axis_indices[:i], Energy5_Fi[0, :i])

    #line12.set_data(x_axis_indices[:i], Energy_Fi[0, :i])

    # Animate points

    # Animate value display
    #value_display.set_text('Pseudo Energy (for 0.01) = ' + str(np.round(Energy1_Fi[0, i],2)))
    #value_display1.set_text('Pseudo Energy (for 0.02) = ' + str(np.round(Energy2_Fi[0, i],2)))
    #value_display2.set_text('Pseudo Energy (for 0.05)= ' + str(np.round(Energy3_Fi[0, i],2)))
    #value_display3.set_text('Pseudo Energy (for 0.1)= ' + str(np.round(Energy4_Fi[0, i],2)))
    #value_display4.set_text('Pseudo Energy (for 0.2)= ' + str(np.round(Energy5_Fi[0, i],2)))
    return line1, point1, line6, point6, line7, point7, line8, point8, line9, point9, line10, point10
    #return line, point0, line1, point1, line2, point2, line3, point3, line4, point4, line6, point6, line7, point7, line8, point8,line9, point9, line10, point10#,line11, point11,line12,point12


ax2.legend(loc='lower right',fontsize=3,prop=dict(weight='bold'))
ax1.legend(loc=1,fontsize=3,prop=dict(weight='bold'))
#ax.legend()

anim1 = animation.FuncAnimation(fig, animate_1, init_func=init_1,frames=iters, interval=300)
#anim2 = animation.FuncAnimation(fig2, animate_2, init_func=init,frames=x_axis_indices.shape[0]-1, interval=300,repeat_delay=60, blit=True)
#anim1 = animation.FuncAnimation(fig2, animate_3,init_func=init_3,frames=coords_c2.shape[0]-1, interval=10)
#anim1 = animation.FuncAnimation(fig2, animate_3,frames=x_range.shape[0], interval=1000)
#anim1 = animation.FuncAnimation(fig2, animate_3, init_func=init_3,frames=iters, interval=100)

#plt.show()
#Writer = animation.FFMpegWriter(fps=30, codec='ffmpeg')
#html = anim1.to_html5_video()
#HTML(html)
anim1.save('C:/Users/user/Documents/Master_Thesis_final_animations/spsa_20dB.gif', writer='Pillow', fps=5, dpi=100, metadata={'title':'test'})
#anim2.save('C:/Users/user/Documents/Master_Thesis_final_animations/Stochastic_gradient_ascent_energy.gif', writer='Pillow', fps=5, dpi=100, metadata={'title':'test'})

'''
plt.figure(3)
plt.plot(Energy1_Fi[0,:]/np.max(Energy1_Fi[0,:]),label='spsa (ofdm)')
plt.plot(Energy3_Fi[0,:]/np.max(Energy3_Fi[0,:]),label='spsa(gaussian)')
plt.plot(Energy2_Fi[0,:]/np.max(Energy2_Fi[0,:]),label='spsa (sinc)')
plt.plot(Energy4_Fi[0,:]/np.max(Energy4_Fi[0,:]),label='spsa(ricker)')
plt.plot(Energy5_Fi[0,:]/np.max(Energy5_Fi[0,:]),label='spsa(gabor)')
plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.title('snr=20')
plt.legend()
'''
'''
plt.figure(3)
plt.plot(Energy1_Fi[1,:]/np.max(Energy1_Fi[1,:]),label='spsa')
plt.plot(Energy2_Fi[0,:]/np.max(Energy2_Fi[0,:]),label='gradient_ascent_momentum')
plt.plot(Energy4_Fi[0,:]/np.max(Energy4_Fi[0,:]),label='gradient_accelerated')
plt.plot(Energy3_Fi[1,:]/np.max(Energy3_Fi[1,:]),label='rdsa')
plt.xlabel('Iterations')
plt.ylabel('Energy')
plt.title('snr=25')
plt.legend()
'''
'''
plt.figure(12)
plt.plot(Energy2_Fi[0,:]/np.max(Energy2_Fi[0,:]),'ro-',label='snr=10(rdsa)')
plt.plot(Energy2_Fi[1,:]/np.max(Energy2_Fi[1,:]),'mo-',label='snr=20(rdsa)')
plt.plot(Energy2_Fi[2,:]/np.max(Energy2_Fi[2,:]),'bo-',label='snr=30(rdsa)')
plt.title('coe=0,az=0(rdsa)')
plt.xlabel('Iterations')
plt.ylabel('energy')

plt.figure(14)
plt.plot(Energy3_Fi[0,:]/np.max(Energy3_Fi[0,:]),'ro-',label='snr=10')
plt.plot(Energy3_Fi[1,:]/np.max(Energy3_Fi[1,:]),'mo-',label='snr=20')
plt.plot(Energy3_Fi[2,:]/np.max(Energy3_Fi[2,:]),'bo-',label='snr=30')
plt.title('coe=0,az=0(ga with momentum)')
plt.xlabel('Iterations')
plt.ylabel('energy')
plt.legend()

plt.figure(15)
plt.plot(Energy4_Fi[0,:]/np.max(Energy4_Fi[0,:]),'ro-',label='snr=10')
plt.plot(Energy4_Fi[1,:]/np.max(Energy4_Fi[1,:]),'mo-',label='snr=20')
plt.plot(Energy4_Fi[2,:]/np.max(Energy4_Fi[2,:]),'bo-',label='snr=30')
plt.title('coe=0,az=0(Newtons method)')
plt.xlabel('Iterations')
plt.ylabel('energy')
plt.legend()

plt.figure(17)
plt.title('With opening=5')
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
plt.plot(coordr02_t[0,0],coordc02_t[0,0],'b*')
plt.plot(coordr02_t[0,:],coordc02_t[0,:],'ro-',label='snr=10')
plt.plot(coordr02_t[1,:],coordc02_t[1,:],'mo-',label='snr=20')
plt.plot(coordr02_t[2,:],coordc02_t[2,:],'bo-',label='snr=30')
plt.title('coe=0,az=0(Newtons Method)')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper right')

plt.figure(13)
plt.title('With opening=5')
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
plt.plot(coordr03_t[0,0],coordc03_t[0,0],'b*')
plt.plot(coordr03_t[0,:],coordc03_t[0,:],'ro-',label='snr=10')
plt.plot(coordr03_t[1,:],coordc03_t[1,:],'mo-',label='snr=20')
plt.plot(coordr03_t[2,:],coordc03_t[2,:],'bo-',label='snr=30')
plt.title('coe=0,az=0(rdsa)')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper right')

plt.figure(16)
plt.title('With opening=5')
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
plt.plot(coordr01_t[0,0],coordc01_t[0,0],'b*')
plt.plot(coordr01_t[0,:],coordc01_t[0,:],'ro-',label='snr=10')
plt.plot(coordr01_t[1,:],coordc01_t[1,:],'mo-',label='snr=20')
plt.plot(coordr01_t[2,:],coordc01_t[2,:],'bo-',label='snr=30')
plt.title('coe=0,az=0(ga with momentum)')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper right')

plt.figure(1)
plt.title('With opening=5')
plt.plot(Energy_Fi[0,:]/np.max(Energy_Fi[0,:]),'r',label='Without Matched Filtering(snr=0,SPSA)')
plt.plot(Energy1_Fi[0,:]/np.max(Energy1_Fi[0,:]),'b',label='With Matched Filtering(snr=0,SPSA)')
plt.xlabel('Iterations')
plt.ylabel('energy')
plt.legend(loc='upper right')

plt.figure(4)
plt.title('With opening=5')
plt.plot(Energy_Fi[1,:]/np.max(Energy_Fi[1,:]),'r',label='Without Matched Filtering(snr=10,SPSA)')
plt.plot(Energy1_Fi[1,:]/np.max(Energy1_Fi[1,:]),'b',label='With Matched Filtering(snr=10,SPSA)')
plt.xlabel('Iterations')
plt.ylabel('energy')
plt.legend(loc='upper right')

plt.figure(8)
plt.title('With opening=5')
plt.plot(Energy_Fi[2,:]/np.max(Energy_Fi[2,:]),'r',label='Without Matched Filtering(snr=20,SPSA)')
plt.plot(Energy1_Fi[2,:]/np.max(Energy1_Fi[2,:]),'b',label='With Matched Filtering(snr=20,SPSA)')
plt.xlabel('Iterations')
plt.ylabel('energy')
plt.legend(loc='upper right')

# plt.figure(2)
# plt.plot(Final_scans[0,0,:]/np.max(Final_scans[0,0,:]),'r',label='Without Matched Filtering(snr=10,RDSA)')
# plt.plot(Final_scans[0,1,:]/np.max(Final_scans[0,1,:]),'b',label='With Matched Filtering(snr=10,RDSA)')
# plt.legend(loc='upper right')
#
# plt.figure(3)
# plt.plot(Final_scans[1,0,:]/np.max(Final_scans[1,0,:]),'r',label='Without Matched Filtering(snr=20,RDSA)')
# plt.plot(Final_scans[1,1,:]/np.max(Final_scans[1,1,:]),'b',label='With Matched Filtering(snr=20,RDSA)')
# plt.legend(loc='upper right')

plt.figure(5)
#plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
plt.title('With opening=5')
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
plt.plot(coordr04_t[0,0],coordc04_t[0,0],'b*')
plt.plot(coordr04_t[0,:],coordc04_t[0,:],'ro-',label='SPSA(wo ma filter,snr=0)')
plt.plot(coordr03[0,:],coordc03[0,:],'mo-',label='SPSA(with ma filter,snr=0)')
plt.title('coe=-10,az=0')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper right')

plt.figure(7)
#plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
plt.title('With opening=5')
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
plt.plot(coordr04_t[1,0],coordc04_t[1,0],'b*')
plt.plot(coordr04_t[1,:],coordc04_t[1,:],'ro-',label='SPSA(wo ma filter,snr=10)')
plt.plot(coordr03[1,:],coordc03[1,:],'mo-',label='SPSA(with ma filter,snr=10)')
plt.title('coe=-10,az=0')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper right')

plt.figure(7)
#plt.imshow(np.fliplr(interpolated),extent = [r_plot[0], r_plot[-1], c_plot[0], c_plot[-1]])
plt.title('With opening=5')
plt.imshow(np.flipud(interpolated3),extent=[-15, 15, -15, 15])
plt.plot(coordr04_t[2,0],coordc04_t[2,0],'b*')
plt.plot(coordr04_t[2,:],coordc04_t[2,:],'ro-',label='SPSA(wo ma filter,snr=20)')
plt.plot(coordr03[2,:],coordc03[2,:],'mo-',label='SPSA(with ma filter,snr=20)')
plt.title('coe=-10,az=0')
plt.xlabel('Azimuth(degrees)')
plt.ylabel('Co elevation(degrees)')
plt.legend(loc='upper right')

energie1 = Energy_Fi[0,:]/np.max(Energy_Fi[0,:])
energie2 = Energy1_Fi[0,:]/np.max(Energy1_Fi[0,:])

total_var = 0
total_var1 = 0

for i in range(energie1.shape[0]-1):
    total_var += np.abs(energie1[i+1] - energie1[i])
    total_var1 += np.abs(energie2[i+1] - energie2[i])
'''