import sys
from time import sleep
import scipy.io as sio
import numpy as np
import random
from Rectangle_test import Rectangle_test
from Scenario import Scenario
from Plane import Plane
from scipy import signal
from Sphere_test import Sphere
import itertools
from matplotlib import cm
from colour import Color
import imp

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTransform,QDoubleValidator,QIntValidator,QFont
from PyQt5.QtCore import QObject, QThread, pyqtSignal,QTimer,QRunnable
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,QTabWidget,QInputDialog,QLineEdit,QFormLayout,QGroupBox,QHBoxLayout,QGridLayout,QFileDialog,QComboBox,QTableWidget,QTableWidgetItem,QDialogButtonBox,QRadioButton
)

import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot, colormap
#from pyqtgraph_extended import ColorBarItem,get_colormap_lut
from pgcolorbar.colorlegend import ColorLegendItem
import pyqtgraph_extensions as pgx
import tikzplotlib

# %% SCENARIO PARAMETERS: DON'T TOUCH THESE!
NT = 400  # number of time domain samples
# c = 1481 #speed of sound in water [m/s] (taken from wikipedia :P)
c = 6000  # speed of sound for synthetic experiments [m/s]
fs = 20e6  # sampling frequency [Hz]
fc = 3e6  # pulse center frequency [Hz]
bw_factor = 0.7e13  # pulse bandwidth factor [Hz^2]
phi = -2.6143  # pulse phase in radians
no_of_symbols = 32  # for ofdm transmit signal
Ts = 1 / fs  # sampling time
symbol_time = no_of_symbols / fs

target_snr_db = np.float(35)

x0 = 0  # horizontal reference for reflecting plane location
z0 = 20e-3  # vertical reference for reflecting plane location

# I calculated these by hand D:
center1 = np.array([-28.3564e-3, 0, z0 + 5e-3])
# center1 = np.array([0, 0, z0 + 5e-3])
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
rect2 = Rectangle_test(center2, az2,co2, h2, w2)

center = np.array([0*1e-3, 0*1e-3, z0])
radius = 5e-3
sph = Sphere(center, radius)

# put the reflecting rectangles in an array
objects = np.array([rect1, rect2])
#objects = np.array([sph])

# and put everything into a scenario object
scenario = Scenario(objects, c, NT, fs, bw_factor, fc,phi,no_of_symbols, symbol_time)

# now, the transducer parameters that will remain fixed throughout the sims:
# self.opening = 45 * np.pi / 180  # opening angle in radians
opening = np.int(15) * np.pi / 180  # opening angle in radians
opening_test = 15
nv = 121  # number of vertical gridpoints for the rays
nh = 121  # number of horizontal gridpoints for the rays
vres = 3e-3 / nv  # ray grid vertical resolution in [m] (set to 0.2mm)
hres = 1.5e-3 / nh  # ray grid horizontal res in [m] (set to 0.1mm)
distance = np.sqrt(3) * (nh - 1) / 2 * hres  # distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.
time_axis = np.arange(NT) / fs
epsilon = 1
sigma = 0.5
n = np.arange(-NT, NT) * Ts
g = np.zeros(n.shape[0])
a_time = 0.05 * 10 ** -6
n11 = np.arange(-NT, 0) * Ts
n12 = np.arange(0, NT) * Ts
s = 0.5
pulse_shapes = np.zeros((2*NT,4))

gaussian = np.exp(-bw_factor * (n - a_time) ** 2) * np.cos(2 * np.pi * fc * (n - a_time) +phi)
#self.gaussian = np.exp(-bw_factor * (n - a_time) ** 2) * np.cos(2 * np.pi * fc * (n - a_time) + phi)
#ricker_pulse = 2 / ((np.sqrt(3 * sigma)) * np.pi ** 0.008) * (1 - bw_factor * ((n - a_time) / sigma) ** 2) * np.exp(-bw_factor * (n - a_time) ** 2 / (2 * sigma ** 2))
ricker_pulse = (1-0.5*(2*np.pi*fc)**2*(n - a_time)**2)*np.exp(-1/4*(2*np.pi*fc)**2*(n - a_time)**2)
sinc_pulse = np.sinc(bw_factor * (n - a_time) ** 2) * np.cos(2 * np.pi * (n - a_time) * fc + phi)


gabor1 = np.exp(-(bw_factor * n11 ** 2 + bw_factor * n11 ** 2 * s)) * np.cos(2 * np.pi * (n11) * fc + phi)
gabor2 = np.exp(-(bw_factor * n12 ** 2 - bw_factor * n12 ** 2 * s)) * np.cos(2 * np.pi * (n12) * fc + phi)
gabor = np.zeros(2 * NT)
gabor[0:NT] = gabor1
gabor[NT:2 * NT] = gabor2

pulse_shapes[:,0] = sinc_pulse
pulse_shapes[:,1] = ricker_pulse
pulse_shapes[:,2] = gaussian
pulse_shapes[:,3] = gabor

algo_list = ["SPSA","RDSA","GA","GA with momentum","Accelerated GA","Newtons","Q Newtons"]
pulse_list = ["Sinc","ricker","gaussian","gabor"]
flag_sph_rect1 = 0


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    '''
    def run(self):
        """Long-running task."""
        for i in range(5):
            sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()
    '''
    def run(self):
        """Long-running task."""
        for i in range(5):
            sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()


class data_generate(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(['QString'])

    def __init__(self,x_start,x_end,x_step,y_start,y_end,y_step,th_step,ph_step,opening_ang,snr,pulse_index):
        super().__init__()
        self.x_start = x_start
        self.x_end = x_end
        self.x_step = x_step
        self.y_start = y_start
        self.y_end = y_end
        self.y_step = y_step
        self.th_step = th_step
        self.ph_step = ph_step
        self.point = np.array([self.x_start, self.y_start, 0]) * 1e-3
        self.th = np.arange(-15, 15, th_step)
        self.ph = np.arange(-15, 15, ph_step)
        self.x_range = np.arange(x_start, x_end, x_step)
        self.y_range = np.arange(y_start, y_end, y_step)
        self.opening_ang = opening_ang
        self.snr = snr
        self.pulse_index = pulse_index

    def run(self):
        for l in range(self.x_range.shape[0]):
            for k in range(self.y_range.shape[0]):
                self.point = np.array([self.x_range[l],self.y_range[k],0])*1e-3
                for i in range(self.ph.shape[0]):
                    for j in range(self.th.shape[0]):
                        print(self.point, self.ph[i], self.th[j])
                        data = ascan(self.point,self.ph[i],self.th[j],self.opening_ang,self.snr,self.pulse_index)
                        filename = 'Ascan_'+str(self.point)+'_'+str(self.ph[i])+'_'+str(self.th[j])
                        print(filename)
                        self.progress.emit(str(filename))
                        np.savez('C:/Users/user/Documents/Ray_tracer_data/'+filename+'.npz',data=data)
                        #sleep(1)

        self.finished.emit()


class auto_calibrate(QObject):
    finished = pyqtSignal()
    finished1 = pyqtSignal()
    progress = pyqtSignal(float, float, float, int, int)
    final_sig = pyqtSignal(float, float, float, int, float, float)
    #progress1 = pyqtSignal(int, int)

    def __init__(self, a, c1, A, gamma, alpha, scale_step, scalefactor, epsilon, point_x, point_y, point_z,ph_init,th_init,step,delta,iterations,opening_ang,snr,window,pulse_index,runfornewton,runnewtoniters,momentum_v):
        super().__init__()
        self.a = a
        self.c1 = c1
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.scale_step = scale_step
        self.scalefactor = scalefactor
        self.point_x = point_x
        self.point_y = point_y
        self.point_z = point_z
        self.ph_init = ph_init
        self.th_init = th_init
        self.epsilon = epsilon
        self.p3 = np.array([self.point_x, self.point_y, self.point_z]) * 1e-3
        self.ph_temp = self.ph_init
        self.th_temp = self.th_init
        self.step = step
        self.delta = delta
        self.iterations = iterations
        print(iterations,delta)
        self.opening_ang = opening_ang
        self.snr = snr
        self.window = window
        self.pulse_index = pulse_index
        self.runfornewton = runfornewton
        self.spsaNewton_ph = 0
        self.spsaNewton_th = 0
        self.rdsaNewton_ph = 0
        self.rdsaNewton_th = 0
        self.gaNewton_ph = 0
        self.gaNewton_th = 0
        self.runnewtoniters = runnewtoniters
        self.momentum_v =  momentum_v

        print('Pulse Index : '+ str(pulse_index))

    def spsa(self):
        for i in range(self.iterations):
            step = self.a / ((i + self.A + 1) ** self.alpha) * self.scale_step
            ck = self.c1 / ((i + 1) ** self.gamma) * self.scale_step

            deltar = np.random.binomial(1, 0.5)
            deltar = 2 * deltar - 1

            deltac = np.random.binomial(1, 0.5)
            deltac = 2 * deltac - 1

            a_scan = ascan(self.p3, self.ph_temp + ck * deltar, self.th_temp + ck * deltac, self.opening_ang,self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:,self.pulse_index],self.window)
            a_scan1 = ascan(self.p3, self.ph_temp - ck * deltar, self.th_temp - ck * deltac, self.opening_ang,self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_th = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltac)
            grad_ph = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltar)

            self.ph_temp = self.ph_temp + grad_ph * step
            self.th_temp = self.th_temp + grad_th * step

            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, pulse_shapes[:,self.pulse_index],self.window)
            max_energy = peak_energy4
            if i == self.runnewtoniters:
                self.spsaNewton_ph = self.ph_temp
                self.spsaNewton_th = self.th_temp
            # energies[i] = max_energy
            # prev_gradient = current_gradient
            # peak = peak_i4
            # print(max_energy)
            # sleep(0.01)
            #print(max_energy,grad_th,self.pulse_index)

            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 0)

        print('SPSA return : ' + str(self.ph_temp) + '  ' + str(self.th_temp))
        self.final_sig.emit(self.ph_temp,self.th_temp,max_energy,0,0.0,0.0)

    def rdsa(self):
        for i in range(self.iterations):
            step = self.a / ((i + self.A + 1) ** self.alpha) * self.scale_step
            ck = self.c1 / ((i + 1) ** self.gamma) * self.scale_step

            prob = (1 + self.epsilon) / (2 + self.epsilon)

            deltar = bool(np.random.binomial(1, prob))
            if deltar:
                deltar = -1
            else:
                deltar = 1 + self.epsilon

            deltac = bool(np.random.binomial(1, prob))
            if deltac:
                deltac = -1
            else:
                deltac = 1 + self.epsilon

            a_scan = ascan(self.p3, self.ph_temp + ck * deltar, self.th_temp + ck * deltac, self.opening_ang,self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:,self.pulse_index],self.window)
            a_scan1 = ascan(self.p3, self.ph_temp - ck * deltar, self.th_temp - ck * deltac, self.opening_ang,self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_th = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltac)
            grad_ph = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltar)

            self.ph_temp = self.ph_temp + grad_ph * step
            self.th_temp = self.th_temp + grad_th * step

            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, pulse_shapes[:,self.pulse_index],self.window)
            max_energy = peak_energy4
            # energies[i] = max_energy
            # prev_gradient = current_gradient
            # peak = peak_i4
            # print(max_energy)
            # sleep(0.01)
            if i == self.runnewtoniters:
                self.rdsaNewton_ph = self.ph_temp
                self.rdsaNewton_th = self.th_temp

            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 1)

        self.final_sig.emit(self.ph_temp, self.th_temp, max_energy, 1,0.0,0.0)

    def gradient_ascent(self):
        for i in range(self.iterations):
            a_scan = ascan(self.p3, self.ph_temp, (self.th_temp + self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:,self.pulse_index],self.window)
            a_scan1 = ascan(self.p3, self.ph_init, (self.th_temp - self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_th = (peak_energy - peak_energy1) / (2 * self.delta)

            a_scan2 = ascan(self.p3, (self.ph_temp + self.delta), self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy2, output_array2, peak_i2 = find_peaks(a_scan2, pulse_shapes[:,self.pulse_index],self.window)
            a_scan3 = ascan(self.p3, (self.ph_temp - self.delta), self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy3, output_array3, peak_i3 = find_peaks(a_scan3, pulse_shapes[:,self.pulse_index],self.window)

            grad_ph = (peak_energy2 - peak_energy3) / (2 * self.delta)

            self.ph_temp = self.ph_temp + grad_ph * self.step
            self.th_temp = self.th_temp + grad_th * self.step

            # grad_vec = np.array([grad_th, grad_ph])
            # grad_vec = np.reshape(grad_vec, (2, 1))
            # inner_product_grad = grad_vec.transpose().dot(self.prev_grad)/(np.linalg.norm(grad_vec)*np.linalg.norm(self.prev_grad))
            # grad_vec_norm = np.linalg.norm(grad_vec)
            # self.prev_grad = grad_vec
            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, pulse_shapes[:,self.pulse_index],self.window)
            max_energy = peak_energy4

            print(max_energy)

            if i == self.runnewtoniters:
                self.gaNewton_ph = self.ph_temp
                self.gaNewton_th = self.th_temp

            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 2)

        self.final_sig.emit(self.ph_temp, self.th_temp, max_energy, 2,0.0,0.0)

    def gradient_ascent_momentum(self):
        v1 = np.zeros((2, 1))
        v1[0] = self.ph_temp
        v1[1] = self.th_temp
        grad_vec = np.zeros((2, 1))
        current_update = np.zeros((2, 1))
        previous_update = np.zeros((2, 1))

        for i in range(self.iterations):
            a_scan = ascan(self.p3, self.ph_temp, (self.th_temp + self.delta), self.opening_ang, self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:, self.pulse_index], self.window)
            a_scan1 = ascan(self.p3, self.ph_init, (self.th_temp - self.delta), self.opening_ang, self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:, self.pulse_index], self.window)

            grad_th = (peak_energy - peak_energy1) / (2 * self.delta)

            a_scan2 = ascan(self.p3, (self.ph_temp + self.delta), self.th_temp, self.opening_ang, self.snr,self.pulse_index)
            peak_energy2, output_array2, peak_i2 = find_peaks(a_scan2, pulse_shapes[:, self.pulse_index], self.window)
            a_scan3 = ascan(self.p3, (self.ph_temp - self.delta), self.th_temp, self.opening_ang, self.snr,self.pulse_index)
            peak_energy3, output_array3, peak_i3 = find_peaks(a_scan3, pulse_shapes[:, self.pulse_index], self.window)

            grad_ph = (peak_energy2 - peak_energy3) / (2 * self.delta)

            grad_vec[0] = grad_ph
            grad_vec[1] = grad_th

            current_update = grad_vec*self.step + self.momentum_v*previous_update

            v1 = v1 + current_update

            previous_update = current_update
            self.th_temp = np.asscalar(v1[1])
            self.ph_temp = np.asscalar(v1[0])

            a_scan9 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang, self.snr, self.pulse_index)
            peak_energy9, output_array9, peak_i9 = find_peaks(a_scan9, pulse_shapes[:, self.pulse_index], self.window)
            max_energy = peak_energy9
            print(max_energy, self.th_temp, self.ph_temp, v1)

            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 3)

        self.final_sig.emit(self.ph_temp, self.th_temp, max_energy, 3, 0.0, 0.0)

    def accelerated_gradient(self):
        v1 = np.zeros((2, 1))
        v1[0] = self.ph_temp
        v1[1] = self.th_temp
        grad_vec = np.zeros((2, 1))
        current_update = np.zeros((2, 1))
        previous_update = np.zeros((2, 1))

        for i in range(self.iterations):
            a_scan = ascan(self.p3, self.ph_temp, (self.th_temp + self.step*np.asscalar(previous_update[1]) + self.delta), self.opening_ang, self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:, self.pulse_index], self.window)
            a_scan1 = ascan(self.p3, self.ph_init, (self.th_temp + self.step*np.asscalar(previous_update[1]) - self.delta), self.opening_ang, self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:, self.pulse_index], self.window)

            grad_th = (peak_energy - peak_energy1) / (2 * self.delta)

            a_scan2 = ascan(self.p3, (self.ph_temp + self.step*np.asscalar(previous_update[0]) + self.delta), self.th_temp, self.opening_ang, self.snr,self.pulse_index)
            peak_energy2, output_array2, peak_i2 = find_peaks(a_scan2, pulse_shapes[:, self.pulse_index], self.window)
            a_scan3 = ascan(self.p3, (self.ph_temp + self.step*np.asscalar(previous_update[0]) - self.delta), self.th_temp, self.opening_ang, self.snr,self.pulse_index)
            peak_energy3, output_array3, peak_i3 = find_peaks(a_scan3, pulse_shapes[:, self.pulse_index], self.window)

            grad_ph = (peak_energy2 - peak_energy3) / (2 * self.delta)

            grad_vec[0] = grad_ph
            grad_vec[1] = grad_th

            current_update = grad_vec*self.step + self.momentum_v*previous_update

            v1 = v1 + current_update

            previous_update = current_update
            self.th_temp = np.asscalar(v1[1])
            self.ph_temp = np.asscalar(v1[0])

            a_scan9 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang, self.snr, self.pulse_index)
            peak_energy9, output_array9, peak_i9 = find_peaks(a_scan9, pulse_shapes[:, self.pulse_index], self.window)
            max_energy = peak_energy9
            print(max_energy, self.th_temp, self.ph_temp, v1)

            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 4)

        self.final_sig.emit(self.ph_temp, self.th_temp, max_energy, 4, 0.0, 0.0)

    def newton_method(self):
        v1 = np.zeros((2,1))
        v1[0] = self.ph_temp
        v1[1] = self.th_temp
        init_ph = self.ph_temp
        init_th = self.th_temp

        for i in range(self.iterations):
            a_scan = ascan(self.p3, (self.ph_temp + self.delta), self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:,self.pulse_index],self.window)
            a_scan1 = ascan(self.p3, (self.ph_temp - self.delta), self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_ph = (peak_energy - peak_energy1) / (2 * self.delta * self.scalefactor)

            a_scan2 = ascan(self.p3,self.ph_temp,(self.th_temp + self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy2, output_array2, peak_i2 = find_peaks(a_scan2, pulse_shapes[:,self.pulse_index],self.window)
            a_scan3 = ascan(self.p3, self.ph_temp,(self.th_temp - self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy3, output_array3, peak_i3 = find_peaks(a_scan3, pulse_shapes[:,self.pulse_index],self.window)

            grad_th = (peak_energy2 - peak_energy3) / (2 * self.delta * self.scalefactor)
            # gradtemp = grad_th

            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, pulse_shapes[:,self.pulse_index],self.window)

            f11 = (peak_energy - 2 * peak_energy4 + peak_energy1) / (self.delta ** 2)
            # f11 = (peak_energy - peak_energy1) / (deltac ** 2)
            # f11temp = f11

            a_scan5 = ascan(self.p3, (self.ph_temp + self.delta), (self.th_temp + self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy5, output_array5, peak_i5 = find_peaks(a_scan5, pulse_shapes[:,self.pulse_index],self.window)

            a_scan6 = ascan(self.p3, (self.ph_temp + self.delta), (self.th_temp - self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy6, output_array6, peak_i6 = find_peaks(a_scan6, pulse_shapes[:,self.pulse_index],self.window)

            a_scan7 = ascan(self.p3, (self.ph_temp - self.delta), (self.th_temp + self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy7, output_array7, peak_i7 = find_peaks(a_scan7, pulse_shapes[:,self.pulse_index],self.window)

            a_scan8 = ascan(self.p3, (self.ph_temp - self.delta), (self.th_temp - self.delta), self.opening_ang,self.snr,self.pulse_index)
            peak_energy8, output_array8, peak_i8 = find_peaks(a_scan8, pulse_shapes[:,self.pulse_index],self.window)

            f1 = peak_energy5
            f2 = peak_energy6
            f3 = peak_energy7
            f4 = peak_energy8

            f21 = (f1 - f2 - f3 + f4) / (4 * self.delta * self.delta)

            f12 = f21

            f22 = (peak_energy2 - 2 * peak_energy4 + peak_energy3) / (self.delta ** 2)

            h = np.array([f11, f12, f21, f22])
            H = np.reshape(h, (2, 2))
            H1 = np.linalg.inv(H)
            grad_v = np.array([grad_ph, grad_th])
            G = np.reshape(grad_v, (2, 1))
            t = H1.dot(G)
            v1 = v1 - self.step * t
            # v1 = v1 + step * G
            # v1[0] = v1[0] + (peak_energy4/grad_th)*(1/(100))
            # v1[1] = v1[1] + (peak_energy4/grad_ph)*(1/(100))

            self.th_temp = np.asscalar(v1[1])
            self.ph_temp = np.asscalar(v1[0])

            a_scan9 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy9, output_array9, peak_i9 = find_peaks(a_scan9, pulse_shapes[:,self.pulse_index],self.window)
            max_energy = peak_energy9
            print(max_energy,self.th_temp,self.ph_temp,v1)
            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 5)

        self.final_sig.emit(self.ph_temp, self.th_temp, max_energy, 5, init_ph, init_th)

    def quasi_newton_method(self):
        v = np.array([self.ph_temp, self.th_temp])
        v1 = np.reshape(v, (2, 1))
        prev_G = np.zeros((2, 1))
        hess_init = (0.5) * np.identity(2)
        hess_update_inv = np.linalg.inv(hess_init)
        hess_estimate_error = []
        #############################################################

        for i in range(self.iterations):
            ascan1 = ascan(self.p3, (self.ph_temp + self.delta), self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(ascan1,pulse_shapes[:,self.pulse_index],self.window)

            ascan1 = ascan(self.p3, (self.ph_temp - self.delta), self.th_temp, self.opening_ang, self.snr,self.pulse_index)
            peak_energy1, output_array, peak_i = find_peaks(ascan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_r = (peak_energy - peak_energy1) / (2 * self.delta)

            ascan1 = ascan(self.p3, self.ph_temp, (self.th_temp + self.delta), self.opening_ang, self.snr,self.pulse_index)
            peak_energy2, output_array, peak_i = find_peaks(ascan1, pulse_shapes[:,self.pulse_index],self.window)

            ascan1 = ascan(self.p3, self.ph_temp, (self.th_temp - self.delta), self.opening_ang, self.snr,self.pulse_index)
            peak_energy3, output_array, peak_i = find_peaks(ascan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_c = (peak_energy2 - peak_energy3) / (2 * self.delta)

            grad_v = np.array([grad_r, grad_c])
            G1 = np.reshape(grad_v, (2, 1))
            diff = G1 - prev_G

            v1 = v1 + self.step * (hess_update_inv.dot(G1))

            diff1 = self.step * (hess_update_inv.dot(G1))
            diff_t = diff1.transpose()
            diff1_t = diff.transpose()
            b = hess_init.dot(diff1)
            b1 = b.dot(diff_t)
            b2 = b1.dot(hess_init)

            c3 = diff_t.dot(hess_init)
            c3 = c3.dot(diff1)
            c3 = np.asscalar(c3[0])
            b3 = (-1) * (b2 * (1 / c3))
            t = (diff.dot(diff1_t)) / (diff1_t.dot(diff1))
            hess_update = hess_init + b3 + t   ### Update hessian matrix #

            hess_init = hess_update

            prev_G = G1  ### keep previous gradient information

            self.ph_temp = np.asscalar(v1[0])
            self.th_temp = np.asscalar(v1[1])

            ascan1 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang, self.snr,self.pulse_index)
            peak_energy5, output_array, peak_i = find_peaks(ascan1, pulse_shapes[:,self.pulse_index],self.window)
            max_energy = peak_energy5

            print(max_energy, self.ph_temp, self.th_temp)
            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 6)

        self.final_sig.emit(self.ph_temp, self.th_temp, max_energy, 6,0.0,0.0)

    def run(self):
        """Long-running task."""
        #print('Init')
        #print(self.iterations)
        self.spsa()
        '''
        for i in range(self.iterations):
            step = self.a / ((i + self.A + 1) ** self.alpha) * self.scale_step
            ck = self.c1 / ((i + 1) ** self.gamma) * self.scale_step

            deltar = np.random.binomial(1, 0.5)
            deltar = 2 * deltar - 1

            deltac = np.random.binomial(1, 0.5)
            deltac = 2 * deltac - 1

            a_scan = ascan(self.p3, self.ph_temp + ck * deltar, self.th_temp + ck * deltac, self.opening_ang)
            peak_energy, output_array, peak_i = find_peaks(a_scan, gaussian)
            a_scan1 = ascan(self.p3, self.ph_temp - ck * deltar, self.th_temp - ck * deltac, self.opening_ang)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, gaussian)

            grad_th = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltac)
            grad_ph = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltar)

            self.ph_temp = self.ph_temp + grad_ph * step
            self.th_temp = self.th_temp + grad_th * step

            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp, self.opening_ang)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, gaussian)
            max_energy = peak_energy4
            # energies[i] = max_energy
            # prev_gradient = current_gradient
            # peak = peak_i4
            #print(max_energy)
            #sleep(0.01)
            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 0)
        '''

        self.ph_temp = self.ph_init
        self.th_temp = self.th_init
        self.rdsa()
        '''

        for i in range(self.iterations):
            step = self.a / ((i + self.A + 1) ** self.alpha) * self.scale_step
            ck = self.c1 / ((i + 1) ** self.gamma) * self.scale_step

            prob = (1 + self.epsilon) / (2 + self.epsilon)

            deltar = bool(np.random.binomial(1, prob))
            if deltar:
                deltar = -1
            else:
                deltar = 1 + self.epsilon

            deltac = bool(np.random.binomial(1, prob))
            if deltac:
                deltac = -1
            else:
                deltac = 1 + self.epsilon

            a_scan = ascan(self.p3, self.ph_temp + ck * deltar, self.th_temp + ck * deltac,self.opening_ang)
            peak_energy, output_array, peak_i = find_peaks(a_scan, gaussian)
            a_scan1 = ascan(self.p3, self.ph_temp - ck * deltar, self.th_temp - ck * deltac,self.opening_ang)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, gaussian)

            grad_th = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltac)
            grad_ph = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltar)

            self.ph_temp = self.ph_temp + grad_ph * step
            self.th_temp = self.th_temp + grad_th * step

            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp,self.opening_ang)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, gaussian)
            max_energy = peak_energy4
            # energies[i] = max_energy
            # prev_gradient = current_gradient
            # peak = peak_i4
            #print(max_energy)
            #sleep(0.01)
            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 1)
        '''

        self.ph_temp = self.ph_init
        self.th_temp = self.th_init

        self.gradient_ascent()

        #self.accelerated_gradient()

        self.ph_temp = self.ph_init
        self.th_temp = self.th_init

        self.gradient_ascent_momentum()

        self.ph_temp = self.ph_init
        self.th_temp = self.th_init

        self.accelerated_gradient()

        self.ph_temp = self.ph_init
        self.th_temp = self.th_init

        self.quasi_newton_method()

        if self.runfornewton == 0:
            self.ph_temp = self.spsaNewton_ph
            self.th_temp = self.spsaNewton_th
        elif self.runfornewton == 1:
            self.ph_temp = self.rdsaNewton_ph
            self.th_temp = self.rdsaNewton_th
        elif self.runfornewton == 2:
            self.ph_temp = self.gaNewton_ph
            self.th_temp = self.gaNewton_th

        print('Initial guess ' + str(self.ph_temp)+'  '+str(self.th_temp))
        self.newton_method()

        '''

        for i in range(self.iterations):
            a_scan = ascan(self.p3, self.ph_temp, (self.th_temp + self.delta),self.opening_ang)
            peak_energy, output_array, peak_i = find_peaks(a_scan, gaussian)
            a_scan1 = ascan(self.p3, self.ph_init, (self.th_temp - self.delta),self.opening_ang)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, gaussian)

            grad_th = (peak_energy - peak_energy1) / (2 * self.delta)

            a_scan2 = ascan(self.p3, (self.ph_temp + self.delta), self.th_temp,self.opening_ang)
            peak_energy2, output_array2, peak_i2 = find_peaks(a_scan2, gaussian)
            a_scan3 = ascan(self.p3, (self.ph_temp - self.delta), self.th_temp,self.opening_ang)
            peak_energy3, output_array3, peak_i3 = find_peaks(a_scan3, gaussian)

            grad_ph = (peak_energy2 - peak_energy3) / (2 * self.delta)

            self.ph_temp = self.ph_temp + grad_ph * self.step
            self.th_temp = self.th_temp + grad_th * self.step

            #grad_vec = np.array([grad_th, grad_ph])
            #grad_vec = np.reshape(grad_vec, (2, 1))
            # inner_product_grad = grad_vec.transpose().dot(self.prev_grad)/(np.linalg.norm(grad_vec)*np.linalg.norm(self.prev_grad))
            #grad_vec_norm = np.linalg.norm(grad_vec)
            #self.prev_grad = grad_vec
            a_scan4 = ascan(self.p3, self.ph_temp, self.th_temp,self.opening_ang)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, gaussian)
            max_energy = peak_energy4

            print(max_energy)

            self.progress.emit(max_energy, self.ph_temp, self.th_temp, i, 2)
        '''

        #self.progress.emit(max_energy, self.ph_temp, self.th_temp, 1, 3)
        self.finished.emit()


class surface_reco(QObject):
    finished = pyqtSignal()
    finished1 = pyqtSignal()
    progress = pyqtSignal(float, float, float, float, float, float, float)
    # progress1 = pyqtSignal(int, int)

    def __init__(self, a, c1, A, gamma, alpha, scale_step, scalefactor, epsilon, start_x, end_x, start_y, end_y, ph_init,th_init, x_step, y_step, step, delta, iterations, opening_ang, snr,window,pulse_index):
        super().__init__()
        self.a = a
        self.c1 = c1
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.scale_step = scale_step
        self.scalefactor = scalefactor
        self.epsilon = epsilon
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.ph_init = ph_init
        self.th_init = th_init
        self.ph_temp = self.ph_init
        self.th_temp = self.th_init
        self.x_step = x_step
        self.y_step = y_step
        self.step = step
        self.delta = delta
        self.iterations = iterations
        self.opening_ang = opening_ang
        self.snr = snr
        self.window = window
        self.pulse_index = pulse_index
        print(self.start_x,self.end_x,self.start_y,self.end_y)
        print(self.x_step,self.y_step)
        self.x_range = np.arange(self.start_x,self.end_x,self.x_step)
        self.y_range = np.arange(self.start_y, self.end_y, self.y_step)

    def spsa(self,point):
        peak = 0
        max_energy = 0
        for i in range(self.iterations):
            step = self.a / ((i + self.A + 1) ** self.alpha) * self.scale_step
            ck = self.c1 / ((i + 1) ** self.gamma) * self.scale_step

            deltar = np.random.binomial(1, 0.5)
            deltar = 2 * deltar - 1

            deltac = np.random.binomial(1, 0.5)
            deltac = 2 * deltac - 1

            a_scan = ascan(point, self.ph_temp + ck * deltar, self.th_temp + ck * deltac, self.opening_ang,self.snr,self.pulse_index)
            peak_energy, output_array, peak_i = find_peaks(a_scan, pulse_shapes[:,self.pulse_index],self.window)
            a_scan1 = ascan(point, self.ph_temp - ck * deltar, self.th_temp - ck * deltac, self.opening_ang,self.snr,self.pulse_index)
            peak_energy1, output_array1, peak_i1 = find_peaks(a_scan1, pulse_shapes[:,self.pulse_index],self.window)

            grad_th = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltac)
            grad_ph = (peak_energy - peak_energy1) / (2 * self.scalefactor * deltar)

            self.ph_temp = self.ph_temp + grad_ph * step
            self.th_temp = self.th_temp + grad_th * step

            a_scan4 = ascan(point, self.ph_temp, self.th_temp, self.opening_ang,self.snr,self.pulse_index)
            peak_energy4, output_array4, peak_i4 = find_peaks(a_scan4, pulse_shapes[:,self.pulse_index],self.window)
            max_energy = peak_energy4
            peak = peak_i4
            print(max_energy,self.ph_temp,self.th_temp)
            # energies[i] = max_energy
            # prev_gradient = current_gradient
            # peak = peak_i4
            # print(max_energy)
            # sleep(0.01)

        di1 = calcualate_tof(peak, c, Ts)
        x = point[0] + di1 * np.sin(self.th_temp * np.pi / 180) * np.cos(self.ph_temp * np.pi / 180)
        y = point[1] + di1 * np.sin(self.th_temp * np.pi / 180) * np.sin(self.ph_temp * np.pi / 180)
        z = di1 * np.cos(self.th_temp * np.pi / 180)

        self.progress.emit(max_energy, self.ph_temp, self.th_temp, x, y, z,di1)

        return x,y,z

    def run(self):
        for i in range(self.x_range.shape[0]):
            for j in range(self.y_range.shape[0]):
                p = np.array([self.x_range[i],self.y_range[j],0])*1e-3
                print(p)
                x,y,z = self.spsa(p)
                self.ph_temp = self.ph_init
                self.th_temp = self.th_init
                print(x,y,z,p)

        print('Surface reconstruction')
        self.finished.emit()


class auto_calibrate_api(QObject):
    def __init__(self,a,b,c,func):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.func = func

    def run(self):
        result = self.func(self.a,self.b,self.c)
        print(result)
        print('testing something')


def ascan(point, azi, coele,opening_angle,snr,p_index):
    transducer = Plane(point, distance, azi * np.pi / 180, coele * np.pi / 180, vres, hres, nv, nh, opening_angle * np.pi / 180)
    transducer.prepareImagingPlane()  # this ALWAYS has to be called right after creating a transducer!
    Ascan = transducer.insonify(scenario, p_index, flag_sph_rect1)
    Ascan1 = np.abs(signal.hilbert(Ascan))
    signal_power = np.mean(Ascan1)
    sig_avg_db = 10.0 * np.log10(signal_power)
    noise_avg_db = sig_avg_db - snr
    noise_avg_w = 10.0 ** (noise_avg_db / 10)
    noise_samples = np.random.normal(0, np.sqrt(noise_avg_w), len(Ascan1))
    Ascan = Ascan + noise_samples
    return Ascan


def find_peaks(ascan, g_pulse,window):
    y = np.abs(signal.hilbert(signal.correlate(ascan, g_pulse, mode='same')))
    peak_index = np.argmax(y)
    #print('Peak Index : '+str(peak_index))
    end_index = peak_index + window
    start_index = peak_index - window
    peak_energy = np.sum(y[start_index:end_index])
    return peak_energy, ascan, peak_index


def calcualate_tof(peaks, speed, ts):
    tof = peaks * ts
    d = tof * speed / 2
    return d


class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Ultrasound")
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.resize(2000, 1000)
        self.tabs = QTabWidget(self)
        self.centralWidget = QWidget()
        self.centralWidget1 = QWidget()
        self.centralWidget2 = QWidget()
        self.tableW = QTableWidget(self.centralWidget)
        self.tableW.move(1200, 470)
        self.tableW.setRowCount(100)
        self.tableW.setColumnCount(11)
        my_font2 = QFont("Times New Roman", 10)
        my_font2.setBold(True)
        self.tableW.resize(600, 500)
        self.tableW.setFont(my_font2)
        self.tableW.verticalHeader().setVisible(False)
        self.tableW.setItem(0, 0, QTableWidgetItem("Algorithm"))
        self.tableW.setItem(0, 1, QTableWidgetItem("Azimuth (Init in deg)"))
        self.tableW.setItem(0, 2, QTableWidgetItem("Co-elev (Init in deg)"))
        self.tableW.setItem(0, 3, QTableWidgetItem("Azimuth(deg)"))
        self.tableW.setItem(0, 4, QTableWidgetItem("Co-elev (deg)"))
        self.tableW.setItem(0, 5, QTableWidgetItem("Energy(l2 norm)"))
        self.tableW.setItem(0, 6, QTableWidgetItem("SNR (dB)"))
        self.tableW.setItem(0, 7, QTableWidgetItem("Iterations"))
        self.tableW.setItem(0, 8, QTableWidgetItem("Pulse shape"))
        self.tableW.setItem(0, 9, QTableWidgetItem("Momentum"))
        self.tableW.setItem(0, 10, QTableWidgetItem("Object"))

        #self.setCentralWidget(self.centralWidget)
        # Create and connect widgets
        # self.tab1 = QWidget()
        # self.tab2 = QWidget()
        self.tabs.resize(3000, 1000)
        # Add tabs
        self.tabs.addTab(self.centralWidget, "Stochastic Approximation")
        self.tabs.addTab(self.centralWidget2, "Pulse Selection")
        self.tabs.addTab(self.centralWidget1, "Surface Reconstruction")
        #self.tabs.addTab(self.tab2, "Tab 2")
        self.stepLabel = QLabel("Energy: 0", self.centralWidget)
        self.stepLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.stepLabel.resize(200, 20)
        self.stepLabel.move(100, 450)

        my_font = QFont("Times New Roman", 10)
        my_font.setBold(True)
        my_font1 = QFont("Times New Roman", 12)
        my_font1.setBold(True)

        '''
        self.clicksLabel = QLabel("Counting: 0 clicks", self)
        self.clicksLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.clicksLabel.move(50,20)
        self.stepLabel = QLabel("Long-Running Step: 0",self)
        self.stepLabel.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.stepLabel.resize(200,20)
        self.stepLabel.move(100,300)
        
        self.countBtn = QPushButton("Click me!", self)
        self.countBtn.clicked.connect(self.countClicks)
        self.countBtn.resize(50, 50)
        self.countBtn.move(500, 500)
        self.longRunningBtn = QPushButton("Long-Running Task!", self)
        self.longRunningBtn.clicked.connect(self.runLongTask)
        self.longRunningBtn.resize(200,50)
        self.longRunningBtn.move(100,100)
        self.plot_button = QPushButton("Plot data!", self)
        self.plot_button.clicked.connect(self.plot_click)
        self.plot_button.resize(200, 50)
        self.plot_button.move(100, 150)
        self.stop_button = QPushButton("Stop!", self)
        self.stop_button.clicked.connect(self.stop_timer)
        self.stop_button.resize(200, 50)
        self.stop_button.move(100, 200)
        '''

        self.data_arrays()

        groupbox = QGroupBox("Pulse Selection", self.centralWidget2)
        groupbox.resize(400, 200)
        groupbox.move(10, 20)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        # layout.addWidget(groupbox)
        vbox4 = QVBoxLayout(self)
        vbox5 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)

        groupbox.setLayout(vbox5)
        vbox4.addWidget(groupbox)

        self.c_freq_label = QLabel(self)
        self.c_freq_label.setText("Center frequency (MHz)")
        self.c_freq_label.setFont(my_font)

        vbox5.addWidget(self.c_freq_label, 0, 0)

        self.center_freq = QLineEdit(self)
        self.center_freq.setValidator(QIntValidator())
        self.center_freq.textChanged.connect(self.center_freqchanged)
        self.center_freq.setText("3")

        vbox5.addWidget(self.center_freq, 0, 1)

        self.sampling_freq_label = QLabel(self)
        self.sampling_freq_label.setText("Sampling frequency (MHz)")
        self.sampling_freq_label.setFont(my_font)

        vbox5.addWidget(self.sampling_freq_label, 1, 0)

        self.sampling_freq = QLineEdit(self)
        self.sampling_freq.setValidator(QIntValidator())
        self.sampling_freq.textChanged.connect(self.sampling_freqchanged)
        self.sampling_freq.setText("20")

        vbox5.addWidget(self.sampling_freq, 1, 1)

        self.bw_factor_label = QLabel(self)
        self.bw_factor_label.setText("Bandwidth Factor (MHz^2)")
        self.bw_factor_label.setFont(my_font)

        vbox5.addWidget(self.bw_factor_label, 2, 0)

        self.bw_factor = QLineEdit(self)
        self.bw_factor.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.bw_factor.textChanged.connect(self.bw_factorchanged)
        self.bw_factor.setText("0.7")

        vbox5.addWidget(self.bw_factor, 2, 1)

        self.windowl_label = QLabel(self)
        self.windowl_label.setText("Window")
        self.windowl_label.move(10, 150)
        self.windowl_label.resize(100, 20)
        self.windowl_label.setFont(my_font)

        vbox5.addWidget(self.windowl_label, 3, 0)

        self.windowl = QLineEdit(self)
        self.windowl.setValidator(QIntValidator())
        self.windowl.resize(100, 20)
        self.windowl.move(150, 150)
        self.windowl.textChanged.connect(self.windowlchanged)
        self.windowl.setText("15")

        vbox5.addWidget(self.windowl, 3, 1)

        self.snr_p_label = QLabel(self.centralWidget2)
        self.snr_p_label.setText("SNR")
        self.snr_p_label.setFont(my_font)
        self.snr_p_label.move(350,20)

        vbox5.addWidget(self.snr_p_label, 4, 0)

        self.snr_p = QLabel(self.centralWidget2)
        self.snr_p.setText(str(30) + " dB")
        self.snr_p.setFont(my_font)
        self.snr_p.move(390, 20)

        vbox5.addWidget(self.snr_p, 4, 1)

        self.pulse_select = QComboBox(self.centralWidget)
        self.pulse_select.addItem("sinc")
        self.pulse_select.addItem("ricker")
        self.pulse_select.addItem("gaussian")
        self.pulse_select.addItem("gabor")
        self.pulse_select.setFont(my_font1)

        vbox5.addWidget(self.pulse_select, 5, 0)

        self.pulse_select.activated.connect(self.pulseselectionchange)

        groupbox = QGroupBox("Power Spectral Density(Scale)", self.centralWidget2)
        groupbox.resize(400, 200)
        groupbox.move(10, 300)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        # layout.addWidget(groupbox)
        vbox4 = QVBoxLayout(self)
        vbox5 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)

        groupbox.setLayout(vbox5)
        vbox4.addWidget(groupbox)

        self.linearScale = QRadioButton('Linear ')
        vbox5.addWidget(self.linearScale, 0, 0)
        self.linearScale.setFont(my_font)
        self.dBScale = QRadioButton('dB')
        vbox5.addWidget(self.dBScale, 1, 0)
        self.dBScale.setFont(my_font)
        self.linearScale.setChecked(True)

        self.linearScale.toggled.connect(self.onClickedScaleChange)
        self.dBScale.toggled.connect(self.onClickedScaleChange)

        self.filename_label = QLabel(self.centralWidget)
        self.filename_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.filename_label.resize(200, 20)
        self.filename_label.move(900, 320)

        self.generate_data_click = QPushButton("Data generate", self.centralWidget)
        self.generate_data_click.clicked.connect(self.generate_data)
        self.generate_data_click.resize(200, 50)
        self.generate_data_click.move(900, 350)
        self.generate_data_click.setFont(my_font1)

        '''
        self.load_click = QPushButton("Load Parameters", self.centralWidget)
        self.load_click.clicked.connect(self.load_param)
        self.load_click.resize(200, 50)
        self.load_click.move(1200, 350)
        self.load_click.setFont(my_font1)

        self.load_function = QPushButton("Load Function", self.centralWidget)
        self.load_function.clicked.connect(self.load_myFunction)
        self.load_function.resize(200, 50)
        self.load_function.move(1400, 350)
        self.load_function.setFont(my_font1)

        self.calibrate_my_func = QPushButton("User Function Calibrate", self.centralWidget)
        self.calibrate_my_func.clicked.connect(self.calibrate_myfuncion)
        self.calibrate_my_func.resize(200, 50)
        self.calibrate_my_func.move(1600, 350)
        self.calibrate_my_func.setFont(my_font1)
        '''

        self.calibrate_click = QPushButton("Calibrate", self.centralWidget)
        self.calibrate_click.clicked.connect(self.start_calibrate)
        self.calibrate_click.resize(200, 50)
        self.calibrate_click.move(100, 370)
        self.calibrate_click.setFont(my_font1)

        self.update_pulseplot = QPushButton("Refresh", self.centralWidget2)
        self.update_pulseplot.clicked.connect(self.refresh)
        self.update_pulseplot.setFont(my_font1)
        self.update_pulseplot.move(20,600)

        groupbox = QGroupBox("Data Generation", self.centralWidget)
        groupbox.resize(500, 250)
        groupbox.move(800, 10)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        # layout.addWidget(groupbox)
        vbox4 = QVBoxLayout(self)
        vbox5 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)

        groupbox.setLayout(vbox5)
        vbox4.addWidget(groupbox)

        self.xstart_label = QLabel(self)
        self.xstart_label.setText("X start")
        self.xstart_label.setFont(my_font)

        vbox5.addWidget(self.xstart_label,0,0)

        self.x_start = QLineEdit(self)
        self.x_start.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.x_start.setText("0")

        vbox5.addWidget(self.x_start, 0, 1)

        self.xend_label = QLabel(self)
        self.xend_label.setText("X End")
        self.xend_label.setFont(my_font)

        vbox5.addWidget(self.xend_label, 1, 0)

        self.x_end = QLineEdit(self)
        self.x_end.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.x_end.setText("10")

        vbox5.addWidget(self.x_end, 1, 1)

        self.ystart_label = QLabel(self)
        self.ystart_label.setText("Y start")
        self.ystart_label.setFont(my_font)

        vbox5.addWidget(self.ystart_label, 0, 2)

        self.y_start = QLineEdit(self)
        self.y_start.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.y_start.setText("0")

        vbox5.addWidget(self.y_start, 0, 3)

        self.yend_label = QLabel(self)
        self.yend_label.setText("Y End")
        self.yend_label.setFont(my_font)

        vbox5.addWidget(self.yend_label, 1, 2)

        self.y_end = QLineEdit(self)
        self.y_end.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.y_end.setText("10")

        vbox5.addWidget(self.y_end, 1, 3)

        self.x_step_d_label = QLabel(self)
        self.x_step_d_label.setText("X step")
        self.x_step_d_label.setFont(my_font)

        vbox5.addWidget(self.x_step_d_label, 2, 0)

        self.x_step_d = QLineEdit(self)
        self.x_step_d.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.x_step_d.setText("5")

        vbox5.addWidget(self.x_step_d, 2, 1)

        self.y_step_d_label = QLabel(self)
        self.y_step_d_label.setText("Y step")
        self.y_step_d_label.setFont(my_font)

        vbox5.addWidget(self.y_step_d_label, 2, 2)

        self.y_step_d = QLineEdit(self)
        self.y_step_d.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.y_step_d.setText("5")

        vbox5.addWidget(self.y_step_d, 2, 3)

        self.az_step_d_label = QLabel(self)
        self.az_step_d_label.setText("Azimuth step")
        self.az_step_d_label.setFont(my_font)

        vbox5.addWidget(self.az_step_d_label, 3, 0)

        self.az_step_d = QLineEdit(self)
        self.az_step_d.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.az_step_d.setText("1")

        vbox5.addWidget(self.az_step_d, 3, 1)

        self.th_step_d_label = QLabel(self)
        self.th_step_d_label.setText("Coelev step")
        self.th_step_d_label.setFont(my_font)

        vbox5.addWidget(self.th_step_d_label, 3, 2)

        self.th_step_d = QLineEdit(self)
        self.th_step_d.setValidator(QDoubleValidator(0.99, 99.99, 2))
        self.th_step_d.setText("1")

        vbox5.addWidget(self.th_step_d, 3, 3)

        self.step = QLineEdit(self)
        self.step.setValidator(QDoubleValidator(0.99, 99.99, 4))
        self.step.resize(100, 20)
        self.step.move(150,100)
        self.step.textChanged.connect(self.stepchanged)
        self.step.setText("0.01")

        self.step_label = QLabel(self)
        self.step_label.setText("Step size")
        self.step_label.setFont(my_font)
        self.step_label.move(10,100)
        self.step_label.resize(100,20)
        #self.a = np.float(self.step.text())

        self.iterations = QLineEdit(self)
        self.iterations.setValidator(QIntValidator())
        self.iterations.resize(100, 20)
        self.iterations.move(150, 150)
        self.iterations.textChanged.connect(self.iterationschanged)
        self.iterations.setText("50")

        self.iterations_label = QLabel(self)
        self.iterations_label.setText("Iterations")
        self.iterations_label.move(10,150)
        self.iterations_label.resize(100,20)
        self.iterations_label.setFont(my_font)

        self.opening = QLineEdit(self)
        self.opening.setValidator(QIntValidator())
        self.opening.resize(10, 20)
        self.opening.move(150, 200)
        self.opening.textChanged.connect(self.openingchanged)
        self.opening.setText("15")

        self.opening_label = QLabel(self)
        self.opening_label.setText("Opening Angle")
        self.opening_label.resize(100, 20)
        self.opening_label.move(10, 200)
        self.opening_label.setFont(my_font)

        self.snr = QLineEdit(self)
        self.snr.setValidator(QIntValidator())
        self.snr.resize(100, 20)
        self.snr.move(150, 150)
        self.snr.textChanged.connect(self.snrchanged)
        self.snr.setText("30")

        self.snr_label = QLabel(self)
        self.snr_label.setText("SNR (in dB)")
        self.snr_label.move(10, 150)
        self.snr_label.resize(100, 20)
        self.snr_label.setFont(my_font)

        self.momentum = QLineEdit(self)
        self.momentum.setValidator(QIntValidator())
        self.momentum.resize(100, 20)
        self.momentum.move(150, 150)
        self.momentum.textChanged.connect(self.momentumchanged)
        self.momentum.setText("0.6")

        self.momentum_label = QLabel(self)
        self.momentum_label.setText("Momentum")
        #self.momentum_label.move(10, 150)
        #self.momentum_label.resize(100, 20)
        self.momentum_label.setFont(my_font)

        groupbox = QGroupBox("Configuration Parameters",self.centralWidget)
        groupbox.resize(400, 250)
        groupbox.move(10, 10)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")

        vbox3 = QVBoxLayout(self)
        vbox2 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)

        groupbox.setLayout(vbox2)
        vbox3.addWidget(groupbox)

        vbox2.addWidget(self.step_label,0,0)
        vbox2.addWidget(self.step,0,1)
        vbox2.addWidget(self.opening_label,1,0)
        vbox2.addWidget(self.opening,1,1)
        vbox2.addWidget(self.iterations_label,2,0)
        vbox2.addWidget(self.iterations,2,1)
        vbox2.addWidget(self.snr_label,3,0)
        vbox2.addWidget(self.snr,3,1)
        vbox2.addWidget(self.momentum_label, 4, 0)
        vbox2.addWidget(self.momentum, 4, 1)
        #vbox2.addWidget(self.windowl_label, 4, 0)
        #vbox2.addWidget(self.windowl, 4, 1)

        #layout.addWidget(groupbox)
        '''
        vbox = QVBoxLayout(self)
        groupbox.setLayout(vbox)

        vbox.addWidget(self.step_label)
        vbox.addWidget(self.step)
        vbox.addWidget(self.opening_label)
        vbox.addWidget(self.opening)
        vbox.addWidget(self.iterations_label)
        vbox.addWidget(self.iterations)
        vbox.addWidget(self.snr_label)
        vbox.addWidget(self.snr)
        '''
        groupbox = QGroupBox("Positioner Control", self.centralWidget)
        groupbox.resize(250, 250)
        groupbox.move(500, 10)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        #layout.addWidget(groupbox)
        vbox3 = QVBoxLayout(self)
        vbox2 = QGridLayout(self)
        #vbox = QHBoxLayout(self)
        #vbox1 = QHBoxLayout(self)

        groupbox.setLayout(vbox2)
        vbox3.addWidget(groupbox)

        #vbox.addLayout(vbox2)
        #vbox1.addLayout(vbox2)

        self.x_label = QLabel(self)
        self.x_label.setText("X")

        vbox2.addWidget(self.x_label,0,0)
        #vbox.addStretch()

        self.x = QLabel(self)
        self.x.setText("0")

        vbox2.addWidget(self.x,0,1)
        #vbox.addStretch()

        self.y_label = QLabel(self)
        self.y_label.setText("Y")

        vbox2.addWidget(self.y_label,1,0)
        #vbox1.addStretch()

        self.y = QLabel(self)
        self.y.setText("0")

        vbox2.addWidget(self.y,1,1)

        self.x_step_label = QLabel(self)
        self.x_step_label.setText("X step")

        vbox2.addWidget(self.x_step_label, 2, 0)

        self.x_step = QLineEdit(self)
        self.x_step.setValidator(QIntValidator())
        #self.x_step.resize(10, 20)
        #self.x_step.move(150, 200)
        self.x_step.textChanged.connect(self.xstep)
        self.x_step.setText("1")

        vbox2.addWidget(self.x_step,2,1)

        self.y_step_label = QLabel(self)
        self.y_step_label.setText("Y step")

        vbox2.addWidget(self.y_step_label, 3, 0)

        self.y_step = QLineEdit(self)
        self.y_step.setValidator(QIntValidator())
        #self.y_step.resize(10, 20)
        #self.y_step.move(150, 200)
        self.y_step.textChanged.connect(self.ystep)
        self.y_step.setText("1")

        vbox2.addWidget(self.y_step, 3, 1)

        self.azinit_label = QLabel(self)
        self.azinit_label.setText("Azimuth Initial")

        vbox2.addWidget(self.azinit_label, 4, 0)

        self.azinit_t = QLineEdit(self)
        self.azinit_t.setValidator(QIntValidator())
        # self.y_step.resize(10, 20)
        # self.y_step.move(150, 200)
        self.azinit_t.textChanged.connect(self.azinit)
        self.azinit_t.setText("0")

        vbox2.addWidget(self.azinit_t, 4, 1)

        self.thinit_label = QLabel(self)
        self.thinit_label.setText("Coelevation Initial")

        vbox2.addWidget(self.thinit_label, 5, 0)

        self.thinit_t = QLineEdit(self)
        self.thinit_t.setValidator(QIntValidator())
        # self.y_step.resize(10, 20)
        # self.y_step.move(150, 200)
        self.thinit_t.textChanged.connect(self.thinit)
        self.thinit_t.setText("-5")

        vbox2.addWidget(self.thinit_t, 5, 1)
        #vbox1.addStretch()
        #vbox2.addLayout(vbox1)
        #groupbox.setLayout(vbox)

        #groupbox.setLayout(vbox)
        #groupbox.setLayout(vbox1)

        self.Upx_click = QPushButton("Up x", self.centralWidget)
        self.Upx_click.clicked.connect(self.upx)
        self.Upx_click.resize(100, 30)
        self.Upx_click.move(500, 300)
        self.Upx_click.setFont(my_font1)

        self.Upy_click = QPushButton("Up y", self.centralWidget)
        self.Upy_click.clicked.connect(self.upy)
        self.Upy_click.resize(100, 30)
        self.Upy_click.move(650, 300)
        self.Upy_click.setFont(my_font1)

        self.Downx_click = QPushButton("Down x", self.centralWidget)
        self.Downx_click.clicked.connect(self.downx)
        self.Downx_click.resize(100, 30)
        self.Downx_click.move(500, 350)
        self.Downx_click.setFont(my_font1)

        self.Downy_click = QPushButton("Down y", self.centralWidget)
        self.Downy_click.clicked.connect(self.downy)
        self.Downy_click.resize(100, 30)
        self.Downy_click.move(650, 350)
        self.Downy_click.setFont(my_font1)

        self.update_cost_click = QPushButton("Update Cost Function", self.centralWidget)
        self.update_cost_click.clicked.connect(self.update_cost_function)
        self.update_cost_click.resize(200, 30)
        self.update_cost_click.move(530, 400)
        self.update_cost_click.setFont(my_font1)

        self.close_click = QPushButton("Close", self.centralWidget)
        self.close_click.clicked.connect(self.close_window)
        self.close_click.resize(100, 30)
        self.close_click.move(1800, 30)
        self.close_click.setFont(my_font1)

        groupbox = QGroupBox("Object Selection", self.centralWidget)
        groupbox.resize(350, 150)
        groupbox.move(1400, 300)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        # layout.addWidget(groupbox)
        vbox3 = QVBoxLayout(self)
        vbox2 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)
        groupbox.setLayout(vbox2)
        vbox3.addWidget(groupbox)

        self.rbtn1 = QRadioButton('Rectangle')
        vbox2.addWidget(self.rbtn1, 0, 0)
        self.rbtn1.setFont(my_font)
        self.rbtn2 = QRadioButton('Sphere')
        vbox2.addWidget(self.rbtn2, 1, 0)
        self.rbtn2.setFont(my_font)
        self.rbtn1.setChecked(True)

        self.rbtn1.toggled.connect(self.onClicked)
        self.rbtn2.toggled.connect(self.onClicked)

        groupbox = QGroupBox("Newtons Method", self.centralWidget)
        groupbox.resize(350, 250)
        groupbox.move(1400, 10)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        # layout.addWidget(groupbox)
        vbox3 = QVBoxLayout(self)
        vbox2 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)
        groupbox.setLayout(vbox2)
        vbox3.addWidget(groupbox)

        self.spsabtn1 = QRadioButton('SPSA ')
        vbox2.addWidget(self.spsabtn1, 0, 0)
        self.spsabtn1.setFont(my_font)
        self.rdsabtn2 = QRadioButton('RDSA')
        vbox2.addWidget(self.rdsabtn2, 1, 0)
        self.rdsabtn2.setFont(my_font)
        self.gabtn3 = QRadioButton('Gradient Ascent')
        vbox2.addWidget(self.gabtn3, 2, 0)
        self.gabtn3.setFont(my_font)
        self.gabtn3.setChecked(True)

        self.newton_iters = QLineEdit(self)
        self.newton_iters.setValidator(QIntValidator())
        self.newton_iters.textChanged.connect(self.newton_iterschanged)
        self.newton_iters.setText("10")
        vbox2.addWidget(self.newton_iters, 3, 1)
        #self.newton_iters.setFont(my_font)

        self.newton_iters_label = QLabel(self)
        self.newton_iters_label.setText("Iterations")
        vbox2.addWidget(self.newton_iters_label, 3, 0)
        self.newton_iters_label.setFont(my_font)

        self.spsabtn1.toggled.connect(self.onAlgoSelect)
        self.rdsabtn2.toggled.connect(self.onAlgoSelect)
        self.gabtn3.toggled.connect(self.onAlgoSelect)

        self.surface_click = QPushButton("Surface Reconstruction", self.centralWidget1)
        self.surface_click.clicked.connect(self.surfacereco)
        self.surface_click.resize(300, 30)
        self.surface_click.move(10, 300)
        self.surface_click.setFont(my_font1)

        self.updatedpoint_label = QLabel(self.centralWidget1)
        self.updatedpoint_label.move(10, 350)
        self.updatedpoint_label.setText("Updated Point")
        self.updatedpoint_label.setFont(my_font1)

        self.updatedpoint = QLabel(self.centralWidget1)
        self.updatedpoint.move(200,350)
        self.updatedpoint.resize(300,30)
        self.updatedpoint.setFont(my_font1)

        groupbox = QGroupBox("Surface Parameters", self.centralWidget1)
        groupbox.resize(500, 250)
        groupbox.move(10, 10)
        groupbox.setStyleSheet("QGroupBox { font-weight: bold; font-size: 14px} ")
        # layout.addWidget(groupbox)
        vbox3 = QVBoxLayout(self)
        vbox2 = QGridLayout(self)
        # vbox = QHBoxLayout(self)
        # vbox1 = QHBoxLayout(self)

        groupbox.setLayout(vbox2)
        vbox3.addWidget(groupbox)

        self.x_scan_s_label = QLabel(self)
        self.x_scan_s_label.setText("X start")

        vbox2.addWidget(self.x_scan_s_label,0,0)

        self.x_scan_s = QLineEdit(self)
        self.x_scan_s.setValidator(QIntValidator())
        self.x_scan_s.textChanged.connect(self.xscanschanged)
        self.x_scan_s.setText("-15")

        vbox2.addWidget(self.x_scan_s, 0, 1)

        self.x_scan_e_label = QLabel(self)
        self.x_scan_e_label.setText("X end")

        vbox2.addWidget(self.x_scan_e_label, 1, 0)

        self.x_scan_e = QLineEdit(self)
        self.x_scan_e.setValidator(QIntValidator())
        self.x_scan_e.textChanged.connect(self.xscanechanged)
        self.x_scan_e.setText("15")

        vbox2.addWidget(self.x_scan_e, 1, 1)

        self.x_scan_step_label = QLabel(self)
        self.x_scan_step_label.setText("X step")

        vbox2.addWidget(self.x_scan_step_label, 2, 0)

        self.x_scan_step = QLineEdit(self)
        self.x_scan_step.setValidator(QIntValidator())
        self.x_scan_step.textChanged.connect(self.xscanstepchanged)
        self.x_scan_step.setText("1")

        vbox2.addWidget(self.x_scan_step, 2, 1)

        self.y_scan_s_label = QLabel(self)
        self.y_scan_s_label.setText("Y start")

        vbox2.addWidget(self.y_scan_s_label, 0, 2)

        self.y_scan_s = QLineEdit(self)
        self.y_scan_s.setValidator(QIntValidator())
        self.y_scan_s.textChanged.connect(self.yscanschanged)
        self.y_scan_s.setText("-10")

        vbox2.addWidget(self.y_scan_s, 0, 3)

        self.y_scan_e_label = QLabel(self)
        self.y_scan_e_label.setText("Y end")

        vbox2.addWidget(self.y_scan_e_label, 1, 2)

        self.y_scan_e = QLineEdit(self)
        self.y_scan_e.setValidator(QIntValidator())
        self.y_scan_e.textChanged.connect(self.yscanechanged)
        self.y_scan_e.setText("10")

        vbox2.addWidget(self.y_scan_e, 1, 3)

        self.y_scan_step_label = QLabel(self)
        self.y_scan_step_label.setText("Y step")

        vbox2.addWidget(self.y_scan_step_label, 2, 2)

        self.y_scan_step = QLineEdit(self)
        self.y_scan_step.setValidator(QIntValidator())
        self.y_scan_step.textChanged.connect(self.yscanstepchanged)
        self.y_scan_step.setText("1")

        vbox2.addWidget(self.y_scan_step, 2, 3)

        #self.step.move(100,220)

        #flo = QFormLayout()
        #flo.addRow("Step size",self.step)
        #flo.move(100,100)
        #num, ok = QInputDialog.getInt(self, "integer input dualog", "Step size")
        '''
        self.timer = QTimer(self)
        # adding action to timer
        self.timer.timeout.connect(self.update_plot)
        # update the timer every tenth second
        '''
        fn = QFont()
        fn.setPointSize(10)
        fn.setBold(True)
        #self.legend = pg.LegendItem((100,100),offset=(100,50))
        self.graphWidget = pg.PlotWidget(self.centralWidget)
        self.graphWidget.addLegend()
        #self.legend.setParentItem(self.graphWidget.getPlotItem())
        self.graphWidget1 = pg.PlotWidget(self.centralWidget)
        self.graphWidget1.addLegend()
        #self.graphWidget.addItem(self.legend)
        #self.graphWidget1.setBackground('k')
        self.graphWidget1.setXRange(-14, 14)
        self.graphWidget1.setYRange(-14, 14)
        self.graphWidget1.getAxis("bottom").setTickFont(fn)
        self.graphWidget1.getAxis("left").setTickFont(fn)
        self.graphWidget1.move(550,470)
        self.graphWidget1.resize(500,500)
        self.graphWidget1.setLabel('left', 'Co elevation')
        self.graphWidget1.setLabel('bottom', 'Azimuth')
        self.graphWidget1.setTitle("Path Traced")

        interpolated2 = sio.loadmat('C:/Users/user/Documents/Master_thesis_CostFunction1/0_0.mat')
        interpolated2 = interpolated2['Simulator']
        #interpolated2 = interpolated2.T

        self.graphWidget6 = pg.PlotWidget(self.centralWidget)
        #self.graphWidget6.setBackground('k')
        self.graphWidget6.move(1050,470)
        self.graphWidget6.resize(100,480)
        self.graphWidget6.getPlotItem().hideAxis('bottom')
        self.graphWidget6.getAxis("left").setTickFont(fn)
        '''

        coords_r1 = np.arange(-15, 15, 1)
        coords_c1 = np.arange(-15, 15, 1)
        interpolated3 = np.ones((coords_r1.shape[0], coords_c1.shape[0]))

        for i in range(len(coords_r1)):
            for j in range(len(coords_c1)):
                index = np.argmax(interpolated2[:, j, i])
                # print(index)
                interpolated3[j, i] = np.sum(interpolated2[index - 15:index + 15, j, i])
        '''
        # Sample array
        # data = np.random.normal(size=(200, 200))
        # data[40:80, 40:120] += 4
        # data = pg.gaussianFilter(data, (15, 15))
        # data += np.random.normal(size=(200, 200)) * 1

        blue, red = Color('blue'), Color('red')
        colors = blue.range_to(red, 256)
        colors_array = np.array([np.array(color.get_rgb()) * 255 for color in colors])
        look_up_table = colors_array.astype(np.uint8)

        #cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)

        tr = QTransform()
        tr.translate(-15, -15)

        self.image = pgx.ImageItem()
        self.image.setOpts(axisOrder='row-major')  # 2021/01/19 Add
        self.image.setLookupTable(look_up_table)
        self.image.setTransform(tr)
        self.image.setImage(interpolated2/np.max(interpolated2))

        cb = pgx.ColorBarItem() #image=self.image
        cb.setManual(look_up_table,[1,0])
        #cb.setImage(self.image)
        #cb.pixelHeight()
        cb.setFixedWidth(0)
        cb.setFixedHeight(1)
        cb.axis_to_levels()
        #cb.shape()
        #cm = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256),color=look_up_table)
        #ColorBarItem()
        #bar.setImage(self.image)
        #image.setLevels([0,1])
        #color_bar = ColorLegendItem(imageItem=self.image, showHistogram=True, label='sample')  # 2021/01/20 add label
        #color_bar.setLevels([np.min(interpolated3),np.max(interpolated3)])
        #color_bar.resetColorLevels()
        #color_bar = ColorLegendItem(imageItem=image, showHistogram=True, label='sample')  # 2021/01/20 add label
        #color_bar.resetColorLevels()

        #cb.axis_to_levels()
        #cb.vb.enableAutoRange(1)

        #self.graphWidget6.addItem(cb)
        self.graphWidget1.addItem(self.image)
        self.graphWidget6.addItem(cb)

        #cb.vb.enableAutoRange(1)
        #self.graphWidget1.addItem(cb)

        #self.graphWidget6.addItem(imv)

        #self.graphWidget1.addItem(color_bar)
        #self.graphWidget1.plot(coords_r1, coords_c1)
        #self.graphWidget1.addItem(color_bar)
        #pg.image(np.ones((20,20)))
        #self.path_traced.addLegend()
        #self.path_traced.setXRange(-15, 15)
        #self.path_traced.setYRange(-15, 15)
        #self.path_traced.move(500, 500)
        #self.path_traced.resize(500, 500)
        #self.path_traced.setTitle('Path Traced')

        #self.x = np.arange(50)
        #self.y = np.array(random.sample(range(1, 100), 50))
        self.plot_data = []
        self.plot_data1 = []
        self.plot_data2 = []
        self.plot_data3 = []
        self.plot_data4 = []
        self.plot_data5 = []
        self.legends = ['SPSA','RDSA']
        self.pens = ['r','g']

        lobj = self.graphWidget.plot([], [], pen=pg.mkPen('r', width=2), name='SPSA')
        self.plot_data.append(lobj)
        lobj1 = self.graphWidget.plot([], [], pen=pg.mkPen('g', width=2), name='RDSA')
        self.plot_data.append(lobj1)
        lobj2 = self.graphWidget.plot([], [], pen=pg.mkPen('b', width=2), name='Gradient Ascent')
        self.plot_data.append(lobj2)
        lobj3 = self.graphWidget.plot([], [], pen=pg.mkPen('y', width=2), name='Gradient Ascent with momentum')
        self.plot_data.append(lobj3)
        lobj3 = self.graphWidget.plot([], [], pen=pg.mkPen('c', width=2), name='Accelerated Gradient Ascent')
        self.plot_data.append(lobj3)
        lobj4 = self.graphWidget.plot([], [], pen=pg.mkPen('k', width=2), name='Newtons Method')
        self.plot_data.append(lobj4)
        lobj5 = self.graphWidget.plot([], [], pen=pg.mkPen('m', width=2), name='Quasi-Newtons Method')
        self.plot_data.append(lobj5)

        lobj5 = self.graphWidget1.plot([], [], symbolPen='r',symbol='o', symbolSize=8, name='SPSA')
        self.plot_data1.append(lobj5)
        lobj6 = self.graphWidget1.plot([], [], symbolPen='g',symbol='o', symbolSize=8, name='RDSA')
        self.plot_data1.append(lobj6)
        lobj7 = self.graphWidget1.plot([], [], symbolPen='b', symbol='o', symbolSize=8, name='Gradient Ascent')
        self.plot_data1.append(lobj7)
        lobj8 = self.graphWidget1.plot([], [], symbolPen='y', symbol='o', symbolSize=8, name='Gradient Ascent with momentum')
        self.plot_data1.append(lobj8)
        lobj9 = self.graphWidget1.plot([], [], symbolPen='c', symbol='o', symbolSize=8, name='Accelerated Gradient Ascent')
        self.plot_data1.append(lobj9)
        lobj10 = self.graphWidget1.plot([], [], symbolPen='k', symbol='o', symbolSize=8, name='Newtons Method')
        self.plot_data1.append(lobj10)
        lobj11 = self.graphWidget1.plot([], [], symbolPen='m', symbol='o', symbolSize=8, name='Quasi-Newtons Method')
        self.plot_data1.append(lobj11)

        self.graphWidget2 = pg.PlotWidget(self.centralWidget2)
        self.graphWidget2.addLegend()
        lobj = self.graphWidget2.plot([], [], pen='r')
        self.plot_data2.append(lobj)
        #self.graphWidget2.setBackground('k')
        self.graphWidget2.setLabel('left', 'Amplitude')
        self.graphWidget2.setLabel('bottom', 'Time (in us)')
        self.graphWidget2.getAxis("bottom").setTickFont(fn)
        self.graphWidget2.getAxis("left").setTickFont(fn)
        #self.graphWidget2.setXRange(0, NT)
        #self.graphWidget2.setYRange(-1, 1)
        self.graphWidget2.resize(500, 500)
        self.graphWidget2.move(500, 0)
        self.graphWidget2.setTitle("Ascan")

        self.graphWidget3 = pg.PlotWidget(self.centralWidget2)
        self.graphWidget3.addLegend()
        lobj = self.graphWidget3.plot([], [], pen='b',name='Auto-Correlation')
        self.plot_data3.append(lobj)
        lobj1 = self.graphWidget3.plot([], [], pen='r',name='Window')
        self.plot_data3.append(lobj1)
        #self.graphWidget3.setBackground('k')
        self.graphWidget3.setLabel('left', 'Magnitude')
        self.graphWidget3.setLabel('bottom', 'samples')
        self.graphWidget3.getAxis("bottom").setTickFont(fn)
        self.graphWidget3.getAxis("left").setTickFont(fn)
        #self.graphWidget3.setXRange(-NT/2, NT/2)
        # self.graphWidget2.setYRange(-1, 1)
        self.graphWidget3.resize(500, 500)
        self.graphWidget3.move(500, 500)
        self.graphWidget3.setTitle("Auto-correlation Function(Ascan)")

        self.graphWidget4 = pg.PlotWidget(self.centralWidget2)
        self.graphWidget4.addLegend()
        lobj = self.graphWidget4.plot([], [], pen='b')
        self.plot_data4.append(lobj)
        #self.graphWidget4.setBackground('k')
        self.graphWidget4.setLabel('left', 'Energy')
        self.graphWidget4.setLabel('bottom', 'frequencies')
        self.graphWidget4.getAxis("bottom").setTickFont(fn)
        self.graphWidget4.getAxis("left").setTickFont(fn)
        # self.graphWidget3.setXRange(-NT/2, NT/2)
        # self.graphWidget2.setYRange(-1, 1)
        self.graphWidget4.resize(500, 500)
        self.graphWidget4.move(1000, 500)
        self.graphWidget4.setTitle("Power spectral density")

        self.graphWidget5 = pg.PlotWidget(self.centralWidget2)
        self.graphWidget5.addLegend()
        lobj = self.graphWidget5.plot([], [], pen='b')
        self.plot_data5.append(lobj)
        #lobj1 = self.graphWidget5.plot([], [], pen='r')
        #self.plot_data5.append(lobj1)
        #self.graphWidget5.setBackground('k')
        self.graphWidget5.setLabel('left', 'Amplitude')
        self.graphWidget5.setLabel('bottom', 'Time (in us)')
        self.graphWidget5.getAxis("bottom").setTickFont(fn)
        self.graphWidget5.getAxis("left").setTickFont(fn)
        # self.graphWidget3.setXRange(-NT/2, NT/2)
        # self.graphWidget2.setYRange(-1, 1)
        self.graphWidget5.resize(500, 500)
        self.graphWidget5.move(1000, 0)
        self.graphWidget5.setTitle("Original Pulse")

        self.pulse_s = self.pulse_select.currentIndex()
        init_point = np.array([self.pos_x, self.pos_y, 0]) * 1e-3
        a_scan = ascan(init_point, self.azinit_val, self.thinit_val, self.opening_angle, self.snr_value, 0)
        samples = np.arange(NT)/fs*1e6
        # self.graphWidget2.plot(samples, a_scan)
        self.plot_data2[0].setData(samples, a_scan)
        tikzplotlib.save("C:/Users/user/Documents/test1.tex")
        #psd = np.fft.fftshift(np.abs(np.fft.fft(a_scan)))
        corr_temp = signal.correlate(a_scan, pulse_shapes[:, self.pulse_s], mode='same')
        acf = np.abs(signal.hilbert(corr_temp))
        psd = np.fft.fftshift(np.abs(np.fft.fft(corr_temp)))
        peak_index = np.argmax(acf)
        #peak_value = np.max(acf)
        rectangle = np.zeros(acf.shape[0])
        rectangle[peak_index-self.window_capture:peak_index+self.window_capture] = 1
        #rect_freq = np.fft.fftshift(np.abs(np.fft.fft(rectangle)))
        #psd = np.fft.fftshift(np.abs(np.fft.fft(corr_temp*rectangle)))
        #convolution = np.convolve(rect_freq,psd, mode='same')

        #self.plot_data5[0].setData(rect_freq)
        samples1 = np.arange(0, 2*NT)*(1/fs)*1e6
        self.plot_data5[0].setData(samples1,pulse_shapes[:, self.pulse_s])

        frequencies = np.arange(-NT/2,NT/2)*fs/NT*1e-6
        print(frequencies.shape)
        self.plot_data4[0].setData(frequencies, psd)
        self.plot_data3[0].setData(acf)
        self.plot_data3[1].setData(rectangle)

        #self.plot_data[0].setData(self.x,self.y)
        #self.plot_data[1].setData(self.x, self.y+10)
        #styles = {'color': 'r', 'font-size': '8px', 'font-weight': 'bold'}
        styles = {'font-size': '20px'}
        fn = QFont()
        # fn.setPointSize(20)
        fn.setBold(True)
        #self.graphWidget.setBackground('k')
        self.graphWidget.setLabel('left','Pseudo Energy')
        self.graphWidget.setLabel('bottom','Iterations')
        #self.graphWidget.getAxis("bottom").setTickFont(fn)
        #self.graphWidget.getAxis("left").setTickFont(fn)
        self.graphWidget.setXRange(0, 101)
        #self.graphWidget.setYRange(0, 150)
        self.graphWidget.resize(500, 500)
        self.graphWidget.move(0, 470)
        self.graphWidget.setTitle("Cost Function")

        #self.data_arrays()
        #self.graphWidget.setGeometry(300, 300, 1250, 850)
        # Set the layout
        #layout = QVBoxLayout()
        # layout.addWidget(self.clicksLabel)
        # layout.addWidget(self.countBtn)
        # layout.addStretch()
        #layout.addWidget(self.stepLabel)
        #layout.addWidget(self.longRunningBtn)
        #layout.addWidget(self.graphWidget)

        #self.setCentralWidget(self.centralWidget)

        # plot data: x, y values
        #self.graphWidget.plot(hour, temperature)
        #self.centralWidget.setLayout(layout)

    def data_arrays(self):
        #self.hour = np.arange(100)
        #self.temperature = random.sample(range(0, 100), 100)
        self.azinit_val = 0
        self.thinit_val = 0
        self.counter = 0
        self.index = 0
        self.ph_init = 0
        self.th_init = 0
        self.energy = []
        self.no_iters = 0
        self.opening_angle = 0
        self.window_capture = 15
        self.iters = []
        self.ph_val = []
        self.th_val = []
        self.pos_x = 0
        self.pos_y = 0
        self.x_step_s = 1
        self.y_step_s = 1
        self.step_size = 0
        self.snr_value = 30
        self.momentum_value = 0.8
        self.func_copy = 0
        self.pulse_s = 0
        self.xscanstart = 0
        self.yscanstart = 0
        self.xscanend = 0
        self.yscanend = 0
        self.xscanstep = 0
        self.yscanstep = 0
        self.centerfrequency = 3
        self.samplingfrequency = 20
        self.bw_factorvalue = 0.7
        self.param = []
        self.table_index = 0
        self.flag_sph_rect = 0
        self.runAlgoForNewtonsmethod = 2
        self.scalechangeflag = 0
        #self.ph_val.append(0)
        #self.th_val.append(0)
        self.energy.append(0)
        self.iters.append(0)
    '''
    def countClicks(self):
        self.clicksCount += 1
        self.clicksLabel.setText(f"Counting: {self.clicksCount} clicks")

    def reportProgress(self, n, index):
        self.stepLabel.setText(f"Long-Running Step: {n}")
    '''
    def onClicked(self):
        global objects,flag_sph_rect1
        if self.rbtn1.isChecked():
            self.flag_sph_rect = 0
            flag_sph_rect1 = 0
            objects = np.array([rect1,rect2])
            print(flag_sph_rect1)
            print("Rectangle is selected")
        elif self.rbtn2.isChecked():
            self.flag_sph_rect = 1
            flag_sph_rect1 = 1
            objects = np.array([sph])
            print(flag_sph_rect1)
            print("Sphere is selected")
        self.pos_x = 0
        self.pos_y = 0
        self.x.setText(str(self.pos_x))
        self.y.setText(str(self.pos_y))
        self.azinit_val = 0
        self.thinit_val = 0
        self.newton_iterations = 10
        self.azinit_t.setText(str(self.azinit_val))
        self.thinit_t.setText(str(self.thinit_val))
        self.update_cost_function()
        self.update_pulseshapes(self.centerfrequency, self.samplingfrequency, self.bw_factorvalue)

    def onAlgoSelect(self):
        if self.spsabtn1.isChecked():
            self.runAlgoForNewtonsmethod = 0
        elif self.rdsabtn2.isChecked():
            self.runAlgoForNewtonsmethod = 1
        elif self.gabtn3.isChecked():
            self.runAlgoForNewtonsmethod = 2

    def onClickedScaleChange(self):
        if self.linearScale.isChecked():
            self.scalechangeflag = 0
            print(self.scalechangeflag)
        elif self.dBScale.isChecked():
            self.scalechangeflag = 1
            print(self.scalechangeflag)
        else:
            self.scalechangeflag = 0
            print('default value')

    def newton_iterschanged(self):
        try:
            self.newton_iterations = np.int(self.newton_iters.text())
            if self.newton_iterations > self.no_iters:
                self.newton_iterations = np.int(self.no_iters/2)
            print(self.newton_iterations)
        except:
            print('error')

    def close_window(self):
        try:
            self.thread1.quit()
            self.thread2.quit()
            self.thread3.quit()
            QApplication.exit()
        except:
            QApplication.exit()

    def update_cost_function(self):

        self.plot_data1.clear()

        self.graphWidget1.removeItem(self.image)

        #x_corr = self.pos_x / 5
        #y_corr = self.pos_y / 5

        #x_corr = np.round(x_corr)
        #y_corr = np.round(y_corr)

        #x_corr = np.int(x_corr) * 5
        #y_corr = np.int(y_corr) * 5

        x_corr = self.pos_x
        y_corr = self.pos_y

        print(x_corr,y_corr)

        if self.flag_sph_rect == 0:
            path_string = 'C:/Users/user/Documents/Master_thesis_CostFunction1/' + str(x_corr) + '_' + str(y_corr) + '.mat'
            print(path_string)
        else:
            path_string = 'C:/Users/user/Documents/Cost_function_sphere/' + str(x_corr) + '_' + str(y_corr) + '.mat'
            print(path_string)

        interpolated2 = sio.loadmat(path_string)
        interpolated2 = interpolated2['Simulator']

        blue, red = Color('blue'), Color('red')
        colors = blue.range_to(red, 256)
        colors_array = np.array([np.array(color.get_rgb()) * 255 for color in colors])
        look_up_table = colors_array.astype(np.uint8)

        tr = QTransform()
        tr.translate(-15, -15)

        #self.image = pg.ImageItem()
        #self.image.setOpts(axisOrder='row-major')  # 2021/01/19 Add
        self.image.setLookupTable(look_up_table)
        self.image.setTransform(tr)
        self.image.setImage(interpolated2 / np.max(interpolated2))

        self.graphWidget1.addItem(self.image)

        lobj5 = self.graphWidget1.plot([], [], symbolPen='r', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj5)
        lobj6 = self.graphWidget1.plot([], [], symbolPen='g', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj6)
        lobj7 = self.graphWidget1.plot([], [], symbolPen='b', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj7)
        lobj8 = self.graphWidget1.plot([], [], symbolPen='y', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj8)
        lobj9 = self.graphWidget1.plot([], [], symbolPen='c', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj9)
        lobj10 = self.graphWidget1.plot([], [], symbolPen='w', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj10)
        lobj11 = self.graphWidget1.plot([], [], symbolPen='m', symbol='o', symbolSize=8)
        self.plot_data1.append(lobj11)
        #self.graphWidget1.removeItem(self.image)

    def load_param(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Single File', 'C:/Users/user/Documents/','*.npz')
        print(fileName)
        try:
            with np.load(str(fileName)) as data:
                self.param = data['x']
            print(self.param)
        except:
            print('FileName is not valid')

    def load_myFunction(self):
        from function_file import find_normal
        self.func_copy = find_normal

    def calibrate_myfuncion(self):
        self.thread1 = QThread()
        #a = self.param[0]
        self.worker1 = auto_calibrate_api(self.param[0],self.param[1],self.param[2],self.func_copy)
        self.worker1.moveToThread(self.thread1)
        self.thread1.started.connect(self.worker1.run)
        self.thread1.start()

    def xstep(self):
        self.x_step_s = np.int(self.x_step.text())

    def ystep(self):
        self.y_step_s = np.int(self.y_step.text())

    def upx(self):
        if self.flag_sph_rect == 0:
            if self.pos_x < 18:
                self.pos_x += self.x_step_s
            else:
                self.pos_x = 18
            self.x.setText(str(self.pos_x))
        else:
            if self.pos_x < 4:
                self.pos_x += self.x_step_s
            else:
                self.pos_x = 4
            self.x.setText(str(self.pos_x))

    def upy(self):
        if self.flag_sph_rect == 0:
            if self.pos_y < 10:
                self.pos_y += self.y_step_s
            else:
                self.pos_y = 10
            self.y.setText(str(self.pos_y))
        else:
            if self.pos_y < 4:
                self.pos_y += self.y_step_s
            else:
                self.pos_y = 4
            self.y.setText(str(self.pos_y))

    def downx(self):
        if self.flag_sph_rect == 0:
            if self.pos_x > -18:
                self.pos_x -= self.x_step_s
            else:
                self.pos_x = -18
            self.x.setText(str(self.pos_x))
        else:
            if self.pos_x > -5:
                self.pos_x -= self.x_step_s
            else:
                self.pos_x = -5
            self.x.setText(str(self.pos_x))

    def downy(self):
        if self.flag_sph_rect == 0:
            if self.pos_y > -10:
                self.pos_y -= self.y_step_s
            else:
                self.pos_y = -10
            self.y.setText(str(self.pos_y))
        else:
            if self.pos_y > -5:
                self.pos_y -= self.y_step_s
            else:
                self.pos_y = -5
            self.y.setText(str(self.pos_y))

    def refresh(self):
        self.update_pulseshapes(self.centerfrequency, self.samplingfrequency,self.bw_factorvalue)
        self.plot_pulse_shapes()

    def xscanstepchanged(self):
        self.xscanstep = np.int(self.x_scan_step.text())
        print(self.xscanstep)

    def yscanstepchanged(self):
        self.yscanstep = np.int(self.y_scan_step.text())
        print(self.yscanstep)

    def xscanschanged(self):
        self.xscanstart = np.int(self.x_scan_s.text())

    def yscanschanged(self):
        self.yscanstart = np.int(self.y_scan_s.text())

    def xscanechanged(self):
        self.xscanend = np.int(self.x_scan_e.text())

    def yscanechanged(self):
        self.yscanend = np.int(self.y_scan_e.text())

    def snrchanged(self):
        try:
            self.snr_value = np.float(self.snr.text())
            self.snr_p.setText(str(self.snr_value)+" dB")
            print(self.snr_value)
        except:
            self.snr_value = 30
            print("error")
    def momentumchanged(self):
        try:
            self.momentum_value = np.float(self.momentum.text())
            print(self.momentum_value)
        except:
            self.momentum_value = 0.8
            print("error")

    def stepchanged(self):
        try:
            self.step_size = np.float(self.step.text())
            print(self.step_size)
        except:
            print("error")

    def iterationschanged(self):
        try:
            self.no_iters = np.int(self.iterations.text())
            self.graphWidget.setXRange(0, self.no_iters)
            print(self.no_iters)
        except:
            print("error")

    def openingchanged(self):
        try:
            self.opening_angle = np.int(self.opening.text())
            print(self.opening_angle)
        except:
            print("error")

    def windowlchanged(self):
        try:
            self.window_capture = np.int(self.windowl.text())
        except:
            print('error')

    def pulseselectionchange(self):
        self.pulse_s = self.pulse_select.currentIndex()
        self.update_pulseshapes(self.centerfrequency, self.samplingfrequency,self.bw_factorvalue)
        self.plot_pulse_shapes()
        '''
        init_point = np.array([self.pos_x,self.pos_y,0])*1e-3
        a_scan = ascan(init_point, self.azinit_val, self.thinit_val, self.opening_angle,self.snr_value,self.pulse_s)
        samples = np.arange(0,NT)
        #self.graphWidget2.plot(samples, a_scan)
        self.plot_data2[0].setData(samples, a_scan)
        psd = np.fft.fftshift(np.abs(np.fft.fft(a_scan)))
        frequencies = np.arange(-NT / 2, NT / 2) * fs / NT * 1e-6
        print(frequencies.shape)
        self.plot_data3[0].setData(frequencies, psd)
        '''
        print(self.pulse_s)

    def center_freqchanged(self):
        try:
            self.centerfrequency = np.int(self.center_freq.text())
            print(self.centerfrequency)
        except:
            print('error')

    def sampling_freqchanged(self):
        try:
            self.samplingfrequency = np.int(self.sampling_freq.text())
            print(self.samplingfrequency)
        except:
            print('error')

    def bw_factorchanged(self):
        try:
            self.bw_factorvalue = np.float(self.bw_factor.text())
            print(self.bw_factorvalue)
        except:
            print('error')

    def azinit(self):
        try:
            self.azinit_val = np.int(self.azinit_t.text())
            print(self.azinit_val)
        except:
            print('error')

    def thinit(self):
        try:
            self.thinit_val = np.int(self.thinit_t.text())
            print(self.thinit_val)
        except:
            print('error')

    def plot_pulse_shapes(self):
        init_point = np.array([self.pos_x, self.pos_y, 0]) * 1e-3
        a_scan = ascan(init_point, self.azinit_val, self.thinit_val, self.opening_angle, self.snr_value, self.pulse_s)
        samples = np.arange(0, NT)*(1/self.samplingfrequency)
        samples1 = np.arange(0,2*NT)*(1/self.samplingfrequency)
        # self.graphWidget2.plot(samples, a_scan)
        self.plot_data2[0].setData(samples, a_scan)
        #tikzplotlib.save("C:/Users/user/Documents/test1.tex")
        corr_temp = signal.correlate(a_scan, pulse_shapes[:, self.pulse_s], mode='same')
        acf = np.abs(signal.hilbert(corr_temp))
        psd = np.fft.fftshift(np.abs(np.fft.fft(corr_temp)))
        peak_ind = np.argmax(acf)
        frequencies = np.arange(-NT / 2, NT / 2) * (self.samplingfrequency/NT)
        rectangle = np.zeros(acf.shape[0])
        rectangle[peak_ind - self.window_capture: peak_ind + self.window_capture] = 1
        #print(frequencies.shape)
        if self.scalechangeflag == 0:
            self.plot_data4[0].setData(frequencies, psd)
        elif self.scalechangeflag == 1:
            self.plot_data4[0].setData(frequencies, 10.0*np.log10(psd))

        self.plot_data3[0].setData(acf)
        self.plot_data3[1].setData(rectangle)
        self.plot_data5[0].setData(samples1,pulse_shapes[:, self.pulse_s])

    def update_pulseshapes(self,cen_f,samp_f,bw_f):
        global scenario
        cen_f = cen_f*1e6
        samp_f = samp_f*1e6
        bw_f = bw_f*1e13
        #print(cen_f, samp_f)

        scenario = Scenario(objects, c, NT, samp_f, bw_f, cen_f, phi, no_of_symbols, symbol_time)
        n = np.arange(-NT, NT) *(1/samp_f)
        n11 = np.arange(-NT, 0) * (1/samp_f)
        n12 = np.arange(0, NT) * (1/samp_f)
        gaussian = np.exp(-bw_f * (n - a_time) ** 2) * np.cos(2 * np.pi * cen_f * (n - a_time) + phi)

        #ricker_pulse = 2 / ((np.sqrt(3 * sigma)) * np.pi ** 0.008) * (1 - bw_f * ((n - a_time) / sigma) ** 2) * np.exp( -bw_f * (n - a_time) ** 2 / (2 * sigma ** 2))
        ricker_pulse = (1-0.5*(2*np.pi*cen_f)**2*(n - a_time)**2)*np.exp(-1/4*(2*np.pi*cen_f)**2*(n - a_time)**2)
        sinc_pulse = np.sinc(-bw_f * (n - a_time) ** 2) * np.cos(2 * np.pi * (n - a_time) * cen_f + phi)
        gabor1 = np.exp(-(bw_f * n11 ** 2 + bw_f * n11 ** 2 * s)) * np.cos(2 * np.pi * (n11) * cen_f + phi)
        gabor2 = np.exp(-(bw_f * n12 ** 2 - bw_f * n12 ** 2 * s)) * np.cos(2 * np.pi * (n12) * cen_f + phi)
        gabor = np.zeros(2 * NT)
        gabor[0:NT] = gabor1
        gabor[NT:2 * NT] = gabor2

        pulse_shapes[:, 0] = sinc_pulse
        pulse_shapes[:, 1] = ricker_pulse
        pulse_shapes[:, 2] = gaussian
        pulse_shapes[:, 3] = gabor
        print("Update pulse shapes")

    def update_table(self,az,ele,energy,ind,ph_n,th_n):
        self.tableW.setItem(self.table_index+1, 0, QTableWidgetItem(algo_list[ind]))
        if ind == 5:
            self.tableW.setItem(self.table_index + 1, 1, QTableWidgetItem(str(ph_n)))
            self.tableW.setItem(self.table_index + 1, 2, QTableWidgetItem(str(th_n)))
        else:
            self.tableW.setItem(self.table_index + 1, 1, QTableWidgetItem(str(self.azinit_val)))
            self.tableW.setItem(self.table_index + 1, 2, QTableWidgetItem(str(self.thinit_val)))

        self.tableW.setItem(self.table_index+1, 3, QTableWidgetItem(str(az)))
        self.tableW.setItem(self.table_index+1, 4, QTableWidgetItem(str(ele)))
        self.tableW.setItem(self.table_index+1, 5, QTableWidgetItem(str(energy)))
        self.tableW.setItem(self.table_index+1, 6, QTableWidgetItem(str(self.snr_value)))
        self.tableW.setItem(self.table_index+1, 7, QTableWidgetItem(str(self.no_iters)))
        self.tableW.setItem(self.table_index+1, 8, QTableWidgetItem(str(pulse_list[self.pulse_s])))
        self.tableW.setItem(self.table_index+1, 9, QTableWidgetItem(str(self.momentum_value)))
        if self.flag_sph_rect == 0:
            self.tableW.setItem(self.table_index + 1, 10, QTableWidgetItem('Rectangle'))
        else:
            self.tableW.setItem(self.table_index + 1, 10, QTableWidgetItem('Sphere'))
        self.table_index += 1
        self.table_index = self.table_index % 100

    def plotSPSA(self, n, ph, th, index, l):
        self.stepLabel.setText(f"Energy: {n}")
        #print(ph, th, l)
        self.energy.append(n)
        self.iters.append(index + 1)
        #print(self.ph_val,self.th_val)
        self.ph_val.append(ph)
        self.th_val.append(th)
        #print(self.ph_val,self.th_val)
        self.plot_data[l].setData(self.iters[:index+1], self.energy[:index+1])
        #print(self.ph_val[:index],self.th_val[:index])
        self.plot_data1[l].setData(self.ph_val[:index], self.th_val[:index])

        if index == self.no_iters-1:
            print('clear data')
            self.energy.clear()
            self.iters.clear()
            self.ph_val.clear()
            self.th_val.clear()
            self.energy.append(0)
            self.iters.append(0)
            self.ph_val.append(self.azinit_val)
            self.th_val.append(self.thinit_val)
            #print(self.ph_val,self.th_val)
            #self.plot_data1[l].setData(self.ph_val, self.th_val)

    def feedback_surface_reco(self,energy,azimuth,elevation,x,y,z,distance):
        self.updatedpoint.setText(str(x) + '  '+str(y)+'  '+str(z))

    def surfacereco(self):
        self.thread3 = QThread()
        self.worker3 = surface_reco(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, 0.8, self.xscanstart, self.xscanend,self.yscanstart,self.yscanend, self.azinit_val, self.thinit_val,self.xscanstep,self.yscanstep, self.step_size, 2, self.no_iters, self.opening_angle,self.snr_value,self.window_capture,self.pulse_s)
        self.worker3.moveToThread(self.thread3)
        self.thread3.started.connect(self.worker3.run)
        self.worker3.finished.connect(self.worker3.deleteLater)
        self.thread3.finished.connect(self.thread3.deleteLater)
        self.worker3.progress.connect(self.feedback_surface_reco)
        self.thread3.start()
        self.surface_click.setEnabled(False)
        self.thread3.finished.connect(
            lambda: self.surface_click.setEnabled(True)
        )

    def start_calibrate(self):

        self.ph_val.clear()
        self.th_val.clear()

        self.ph_val.append(self.azinit_val)
        self.th_val.append(self.thinit_val)

        self.plot_data[0].setData(self.iters, self.energy)
        self.plot_data[1].setData(self.iters, self.energy)
        self.plot_data[2].setData(self.iters, self.energy)
        self.plot_data[3].setData(self.iters, self.energy)
        self.plot_data[4].setData(self.iters, self.energy)
        self.plot_data[5].setData(self.iters, self.energy)
        self.plot_data[6].setData(self.iters, self.energy)

        self.plot_data1[0].setData(self.ph_val, self.th_val)
        self.plot_data1[1].setData(self.ph_val, self.th_val)
        self.plot_data1[2].setData(self.ph_val, self.th_val)
        self.plot_data1[3].setData(self.ph_val, self.th_val)
        self.plot_data1[4].setData(self.ph_val, self.th_val)
        self.plot_data1[5].setData(self.ph_val, self.th_val)
        self.plot_data1[6].setData(self.ph_val, self.th_val)

        #print(self.ph_val,self.th_val)

        self.thread1 = QThread()
        self.worker1 = auto_calibrate(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, 0.8, self.pos_x, self.pos_y, 0, self.azinit_val, self.thinit_val, self.step_size, 2, self.no_iters, self.opening_angle,self.snr_value,self.window_capture,self.pulse_s,self.runAlgoForNewtonsmethod,self.newton_iterations,self.momentum_value)
        self.worker1.moveToThread(self.thread1)
        self.thread1.started.connect(self.worker1.run)
        self.worker1.finished.connect(self.thread1.quit)
        self.worker1.finished.connect(self.worker1.deleteLater)
        self.thread1.finished.connect(self.thread1.deleteLater)
        self.worker1.progress.connect(self.plotSPSA)
        self.worker1.final_sig.connect(self.update_table)
        #self.worker1.progress1.connect(self.plotRDSA)
        self.thread1.start()
        # Final resets
        self.calibrate_click.setEnabled(False)
        '''
        self.thread1.finished.connect(
            lambda: self.energy.clear()
        )
        self.thread1.finished.connect(
            lambda: self.iters.clear()
        )
        '''
        self.thread1.finished.connect(
            lambda: self.calibrate_click.setEnabled(True)
        )
        self.thread1.finished.connect(
            lambda: self.stepLabel.setText("Energy: 0")
        )
        #self.timer1.start(500)
        #print("calibrate")

    def feedback_data_gen(self,filen):
        self.filename_label.setText(str(filen))

    def generate_data(self):
        self.thread2 = QThread()
        self.worker2 = data_generate(np.int(self.x_start.text()),np.int(self.x_end.text()),np.int(self.x_step_d.text()),np.int(self.y_start.text()),
                                     np.int(self.y_end.text()),np.int(self.y_step_d.text()),np.int(self.th_step_d.text()),
                                     np.int(self.az_step_d.text()),self.opening_angle,self.snr_value,self.pulse_s)
        self.worker2.moveToThread(self.thread2)
        self.thread2.started.connect(self.worker2.run)
        self.worker2.finished.connect(self.thread2.quit)
        self.worker2.finished.connect(self.worker2.deleteLater)
        self.thread2.finished.connect(self.thread2.deleteLater)
        self.worker2.progress.connect(self.feedback_data_gen)
        self.thread2.start()
        self.generate_data_click.setEnabled(False)
        self.thread2.finished.connect(
            lambda: self.generate_data_click.setEnabled(True)
        )


app = QApplication(sys.argv)
win = Window()
win.show()
sys.exit(app.exec())