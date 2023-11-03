
'''
x = np.ones((3,3))
root = tk.Tk()
root.title("Ultrasound")
root.geometry("500x500")

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("A Scan Data",fontsize=10)
ax.set_xlabel("Samples",fontsize=8)
ax.set_ylabel("Amplitude",fontsize=8)

fig1 = plt.figure()
ax = fig1.add_subplot(212)
ax.set_title("Transducer",fontsize=10)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().place(x=100,y=100,width=200,height=200)
canvas.draw()

canvas = FigureCanvasTkAgg(fig1, master=root)
canvas.get_tk_widget().place(x=100,y=300,width=200,height=200)
canvas.draw()

root.mainloop()
'''

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from tkinter import ttk
from Rectangle_test import Rectangle_test
from Scenario import Scenario
from Plane import Plane
from timeit import default_timer as timer
from scipy import signal
import matplotlib.animation as animation
import time

colors= ['red','green','brown','purple','cyan'] #'gold','green','maroon''brown','purple','cyan'
legends = ['SPSA', 'gradient ascent','non linear cg','gradient ascent (momentum)','RDSA']
# take the data
lst = [('Azimuth (deg)', 'Co-elevation (deg)', 'Energy',''),
       ('', '', '','SPSA'),
       ('', '', '','Gradient Ascent'),
       ('', '', '','Non Linear CG'),
       ('', '', '','GA(with momentum)'),
       ('', '', '','RDSA')]

cells = {}
total_rows = 6
total_columns = 4


class positioner:
    def __init__(self, window):
        self.window = window

        self.configure = LabelFrame(self.window, text="Configuration Parameters", height=200, width=400,font=('Helvetica', 12, 'bold'))
        self.configure.place(x=1250, y=540)
        self.pos_control = LabelFrame(self.window, text="Positioner Control", height=500, width=400,font=('Helvetica', 12, 'bold'))
        self.pos_control.place(x=1250, y=10)
        self.data_generate = LabelFrame(self.window, text="Data Generation", height=200, width=400,font=('Helvetica', 12, 'bold'))
        self.data_generate.place(x=1250, y=770)

        self.x = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        self.y = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        self.azimuth = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        self.coelevation = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        #self.calib_az = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        #self.calib_coe = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.x_label = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        self.y_label = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        self.az_label = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        self.coe_label = Label(self.pos_control,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        #self.calib_az_label = Label(window,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        #self.calib_coe_label = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        '''
        self.noofiters = Entry(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.noofiters_label = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.noofiters.insert(END,str(50))
        self.step_size = Entry(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.step_size.insert(END,str(0.1))
        self.step_size_label = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.target_snr = Entry(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.target_snr.insert(END, str(35))
        self.target_snr_label = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        '''

        self.noofiters = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.noofiters_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.noofiters.insert(END, str(50))
        self.step_size = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.step_size.insert(END, str(0.1))
        self.step_size_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.target_snr = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.target_snr.insert(END, str(35))
        self.target_snr_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.opening_angle = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.opening_angle.insert(END, str(40))
        self.opening_angle_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.x_pos = 0
        self.y_pos = 0
        self.az_pos = 0
        self.coe_pos = 0
        self.ph_init = 0
        self.th_init = 0
        self.noofiter = 0
        self.step = 0
        self.run = 0
        self.prev_grad = np.ones((2, 1))*1e-5
        self.prev_s = np.zeros((2, 1))

        self.x.config(text=str(self.x_pos))
        self.y.config(text=str(self.y_pos))
        self.azimuth.config(text=str(self.az_pos))
        self.coelevation.config(text=str(self.coe_pos))
        #self.calib_az.config(text=str(self.ph_init))
        #self.calib_coe.config(text=str(self.th_init))

        self.az_label.config(text='Azimuth')
        self.coe_label.config(text='Coelevation')
        self.x_label.config(text='x')
        self.y_label.config(text='y')
        #self.calib_az_label.config(text='Azimuth(calibrated)')
        #self.calib_coe_label.config(text='Co-elevation(calibrated)')
        self.noofiters_label.config(text='Iterations')
        self.step_size_label.config(text='Step size')
        self.target_snr_label.config(text='SNR in dB')
        self.opening_angle_label.config(text='Opening Angle')

        self.x.place(x=10, y=110)
        self.y.place(x=10, y=210)
        self.azimuth.place(x=10, y=310)
        self.coelevation.place(x=10, y=410)

        self.x_label.place(x=110, y=110)
        self.y_label.place(x=110, y=210)
        self.az_label.place(x=110, y=310)
        self.coe_label.place(x=110, y=410)
        self.h_b = 1
        self.w_b = 10

        self.button = Button(self.pos_control, text="Update Transducer", command=self.plot, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b + 10)
        self.button.place(x=210, y=10)

        self.upx = Button(self.pos_control, text="Up x", command=self.up_x, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.upx.place(x=210, y=110)
        self.downx = Button(self.pos_control, text="Down x", command=self.down_x, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.downx.place(x=210, y=140)

        self.upy = Button(self.pos_control, text="Up y", command=self.up_y, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.upy.place(x=210, y=210)
        self.downy = Button(self.pos_control, text="Down y", command=self.down_y, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.downy.place(x=210, y=240)

        self.upaz = Button(self.pos_control, text="Up az", command=self.up_az, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.upaz.place(x=210, y=310)
        self.downaz = Button(self.pos_control, text="Down az", command=self.down_az, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.downaz.place(x=210, y=340)

        self.upcoe = Button(self.pos_control, text="Up coe", command=self.up_coe, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.upcoe.place(x=210, y=410)
        self.downcoe = Button(self.pos_control, text="Down coe", command=self.down_coe, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)
        self.downcoe.place(x=210, y=440)

        '''
        self.x.place(x=1350, y=200)
        self.y.place(x=1350, y=300)
        self.azimuth.place(x=1350,y=400)
        self.coelevation.place(x=1350, y=500)

        self.x_label.place(x=1250, y=200)
        self.y_label.place(x=1250, y=300)
        self.az_label.place(x=1250, y=400)
        self.coe_label.place(x=1250, y=500)
        self.h_b = 1
        self.w_b = 10

        self.button = Button (window, text="Update Transducer", command=self.plot,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b+10)
        self.button.place(x=1350,y=100)

        self.upx = Button(window, text="Up x", command=self.up_x,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.upx.place(x=1400, y=200)
        self.downx = Button(window, text="Down x", command=self.down_x,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.downx.place(x=1400,y=230)

        self.upy = Button(window, text="Up y", command=self.up_y,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.upy.place(x=1400, y=300)
        self.downy = Button(window, text="Down y", command=self.down_y,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.downy.place(x=1400, y=330)

        self.upaz = Button(window, text="Up az", command=self.up_az,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.upaz.place(x=1400, y=400)
        self.downaz = Button(window, text="Down az", command=self.down_az,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.downaz.place(x=1400, y=430)

        self.upcoe = Button(window, text="Up coe", command=self.up_coe,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.upcoe.place(x=1400, y=500)
        self.downcoe = Button(window, text="Down coe", command=self.down_coe,borderwidth=2,relief='solid',font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.downcoe.place(x=1400, y=530)
        '''

        self.calibrate = Button(self.configure, text="Calibrate", command=self.calibrate, borderwidth=2, relief='solid', font=('Helvetica', 10, 'bold'),height=self.h_b,width=self.w_b)
        self.calibrate.place(x=120, y=140)
        #self.calib_az.place(x=1400, y=700)
        #self.calib_coe.place(x=1400, y=730)
        #self.calib_az_label.place(x=1200, y=700)
        #self.calib_coe_label.place(x=1200, y=730)
        '''
        self.noofiters.place(x=1350,y=800)
        self.noofiters_label.place(x=1200,y=800)
        self.step_size.place(x=1350,y=830)
        self.step_size_label.place(x=1200, y=830)
        self.target_snr.place(x=1350,y=860)
        self.target_snr_label.place(x=1200,y=860)
        '''
        self.noofiters.place(x=120, y=10)
        self.noofiters_label.place(x=10, y=10)
        self.step_size.place(x=120, y=40)
        self.step_size_label.place(x=10, y=40)
        self.target_snr.place(x=120, y=70)
        self.target_snr_label.place(x=10, y=70)
        self.opening_angle.place(x=120, y=110)
        self.opening_angle_label.place(x=10, y=110)

        # %% SCENARIO PARAMETERS: DON'T TOUCH THESE!
        self.NT = 400  # number of time domain samples
        # c = 1481 #speed of sound in water [m/s] (taken from wikipedia :P)
        self.c = 6000  # speed of sound for synthetic experiments [m/s]
        self.fs = 20e6  # sampling frequency [Hz]
        self.fc = 3e6  # pulse center frequency [Hz]
        self.bw_factor = 0.7e13  # pulse bandwidth factor [Hz^2]
        self.phi = -2.6143  # pulse phase in radians
        self.no_of_symbols = 32  # for ofdm transmit signal
        self.Ts = 1 / self.fs  # sampling time
        self.symbol_time = self.no_of_symbols / self.fs

        self.target_snr_db = np.float(self.target_snr.get())

        self.x0 = 0  # horizontal reference for reflecting plane location
        self.z0 = 20e-3  # vertical reference for reflecting plane location

        # I calculated these by hand D:
        self.center1 = np.array([-28.3564e-3, 0, self.z0 + 5e-3])
        # center1 = np.array([0, 0, z0 + 5e-3])
        self.az1 = np.pi
        self.co1 = 170 * np.pi / 180
        self.h1 = 20e-3
        self.w1 = 57.5877e-3
        self.rect1 = Rectangle_test(self.center1, self.az1, self.co1, self.h1,self. w1)

        self.center2 = np.array([18.66025e-3, 0, self.z0 + 5e-3])
        self.az2 = 0
        self.co2 = 165 * np.pi / 180
        self.h2 = 20e-3
        self.w2 = 38.6370e-3
        self.rect2 = Rectangle_test(self.center2, self.az2, self.co2, self.h2, self.w2)

        # put the reflecting rectangles in an array
        self.objects = np.array([self.rect1, self.rect2])

        # and put everything into a scenario object
        self.scenario = Scenario(self.objects, self.c, self.NT, self.fs, self.bw_factor, self.fc, self.phi,self.no_of_symbols,self.symbol_time)

        # now, the transducer parameters that will remain fixed throughout the sims:
        #self.opening = 45 * np.pi / 180  # opening angle in radians
        self.opening = np.int(self.opening_angle.get())* np.pi / 180  # opening angle in radians
        self.nv = 181  # number of vertical gridpoints for the rays
        self.nh = 181  # number of horizontal gridpoints for the rays
        self.vres = 3e-3 / self.nv  # ray grid vertical resolution in [m] (set to 0.2mm)
        self.hres = 1.5e-3 / self.nh  # ray grid horizontal res in [m] (set to 0.1mm)
        self.distance = np.sqrt(3) * (self.nh - 1) / 2 * self.hres # distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.
        self.time_axis = np.arange(self.NT) / self.fs
        self.epsilon = 1
        self.sigma = 0.5
        self.n = np.arange(-self.NT, self.NT) * self.Ts
        self.g = np.zeros(self.n.shape[0])
        self.a_time = 0.05 * 10 ** -6
        self.gaussian = np.exp(-self.bw_factor * (self.n - self.a_time) ** 2) * np.cos(2 * np.pi * self.fc * (self.n - self.a_time) + self.phi)
        self.ricker_pulse = 2 / ((np.sqrt(3 * self.sigma)) * np.pi ** 0.008) * (1 - self.bw_factor * ((self.n - self.a_time) / self.sigma) ** 2) * np.exp(-self.bw_factor * (self.n - self.a_time) ** 2 / (2 * self.sigma ** 2))
        self.sinc_pulse = np.sinc(self.bw_factor * (self.n - self.a_time) ** 2) * np.cos(2 * np.pi * (self.n - self.a_time) * self.fc + self.phi)
        self.n11 = np.arange(-self.NT, 0) * self.Ts
        self.n12 = np.arange(0, self.NT) * self.Ts
        self.s = 0.5

        self.gabor1 = np.exp(-(self.bw_factor * self.n11 ** 2 + self.bw_factor * self.n11 ** 2 * self.s)) * np.cos(2 * np.pi * (self.n11) * self.fc + self.phi)
        self.gabor2 = np.exp(-(self.bw_factor * self.n12 ** 2 - self.bw_factor * self.n12 ** 2 * self.s)) * np.cos(2 * np.pi * (self.n12) * self.fc + self.phi)
        self.gabor = np.zeros(2 * self.NT)
        self.gabor[0:self.NT] = self.gabor1
        self.gabor[self.NT:2 * self.NT] = self.gabor2
        #self.g = signal.hilbert(self.g)
        #self.g = np.abs(self.g)
        #self.g = self.g / np.max(self.g)

        self.index = 0
        self.energies = []
        self.grad_en = []
        self.spsa_en = []
        self.iters1 = []
        self.test_grads = []
        self.pulse_select = 0
        self.momentum = 0.9

        self.OPTIONS = [
            "Gaussian Pulse",
            "Ricker Wavelet",
            "Sinc Pulse",
            "Gabor Pulse",
        ]  # etc

        self.variable = StringVar(self.window)
        self.variable.set(self.OPTIONS[0])  # default value
        #self.variable.trace("w", self.callback)

        self.w = OptionMenu(self.window, self.variable, *self.OPTIONS,command=self.callback)
        self.w.place(x=1600, y=0)

        point = np.array([self.x_pos, self.y_pos, 0])
        self.fig = Figure(figsize=(6, 5))
        self.a = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().place(x=0, y=0)

        self.pulse_selection()
        ascan1, ascan, trans = self.init_transducer(point, self.az_pos, self.coe_pos)
        self.line, = self.a.plot(self.time_axis, ascan)

        self.a.set_title('Ascan')
        self.a.set_xlabel('time (us)')
        self.a.set_ylabel('Amplitude')
        self.canvas.draw()

        self.fig1 = Figure(figsize=(6, 5))
        self.a1 = self.fig1.add_subplot(111)
        self.a1.imshow(trans.image)
        self.a1.set_title('Transducer')

        self.canvas2 = FigureCanvasTkAgg(self.fig1, master=self.window)
        self.canvas2.get_tk_widget().place(x=0, y=500)
        self.canvas2.draw()

        self.fig2 = Figure(figsize=(6, 5))
        self.a2 = self.fig2.add_subplot(111)
        self.line1, = self.a2.plot(self.iters1, self.energies,'ro-')
        self.a2.set_title('Cost function')
        self.a2.set_xlabel('Iterations')
        self.a2.set_ylabel('Energy')

        self.canvas1 = FigureCanvasTkAgg(self.fig2, master=self.window)
        self.canvas1.get_tk_widget().place(x=600, y=0)
        self.canvas1.draw()

        self.lines = []
        self.lines3 = []

        for index in range(5):
            lobj = self.a2.plot([], [], lw=1, color=colors[index], label=legends[index])[0]
            lobj1 = self.a.plot([], [], lw=1, color=colors[index], label=legends[index])[0]
            self.lines.append(lobj)
            self.lines3.append(lobj1)
            self.a2.legend()
            self.a.legend()

        for i in range(total_rows):
            for j in range(total_columns):
                self.e = Label(self.window, width=20, fg='blue',font=('Helvetica', 10, 'bold'))
                self.e.grid(row=i, column=j)
                cells[(i,j)] = self.e
                self.e.place(x=600 + j * 140, y=600 + i * 50)
                #print(lst[i][j])
                self.e.config(text=str(lst[i][j]))

    def callback(self,*kwargs):
        self.pulse_selection()
        self.plot()

    def pulse_selection(self):
        if self.variable.get() == self.OPTIONS[0]:
            self.g = self.gaussian
            self.pulse_select = 2
        elif self.variable.get() == self.OPTIONS[1]:
            self.g = self.ricker_pulse
            self.pulse_select = 1
        elif self.variable.get() == self.OPTIONS[2]:
            self.g = self.sinc_pulse
            self.pulse_select = 0
        elif self.variable.get() == self.OPTIONS[3]:
            self.g = self.gabor
            self.pulse_select = 3

    def plot(self):

        p1 = self.x_pos
        p2 = self.y_pos
        a1 = self.az_pos
        c1 = self.coe_pos

        point = np.array([p1, p2, 0])*1e-3
        self.pulse_selection()
        ascan1,ascan,trans = self.init_transducer(point,a1,c1)
        '''
        fig = Figure(figsize=(6, 5))
        a = fig.add_subplot(111)
        a.plot(self.time_axis, ascan)
        a.set_title('Ascan')
        a.set_xlabel('time (us)')
        a.set_ylabel('Amplitude')
        canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        canvas.get_tk_widget().place(x=0, y=0)
        canvas.draw()
        self.fig1 = Figure(figsize=(6, 5))
        self.a1 = self.fig1.add_subplot(111)
        '''
        #self.a.clear()
        #self.a.plot(self.time_axis, ascan)#(np.min(ascan) - 0.05, np.max(ascan) + 0.05)
        self.a.set_ylim(np.min(ascan) - 0.1, np.max(ascan) + 0.1)
        self.line.set_ydata(ascan)
        self.line.set_xdata(self.time_axis)
        self.a.set_title('Ascan')
        self.a.set_xlabel('time (us)')
        self.a.set_ylabel('Amplitude')
        self.canvas.draw()

        self.a1.imshow(trans.image)
        self.a1.set_title('Transducer')
        self.canvas2.draw()
        #canvas = FigureCanvasTkAgg(self.fig1, master=self.window)
        #canvas.get_tk_widget().place(x=0,y=500)

    def init_transducer(self, point, azi, coele):
        self.target_snr_db = np.float(self.target_snr.get())
        transducer = Plane(point, self.distance, azi*np.pi/180, coele*np.pi/180, self.vres, self.hres, self.nv, self.nh,np.int(self.opening_angle.get())* np.pi / 180)
        transducer.prepareImagingPlane()  # this ALWAYS has to be called right after creating a transducer!
        #print(point * 1e-3)
        start = timer()
        Ascan1 = transducer.insonify(self.scenario,self.pulse_select,0)
        Ascan2 = np.abs(signal.hilbert(Ascan1))
        signal_power = np.mean(Ascan2)
        sig_avg_db = 10.0*np.log10(signal_power)
        noise_avg_db = sig_avg_db - self.target_snr_db
        noise_avg_w = 10.0 ** (noise_avg_db / 10)
        noise_samples = np.random.normal(0,np.sqrt(noise_avg_w),len(Ascan2))
        Ascan1 = Ascan1 + noise_samples
        Ascan = np.abs(signal.hilbert(Ascan1))
        #Ascan1 = np.fft.fftshift(np.abs(np.fft.fft(Ascan1)))
        #print('signal_power: ' + str(signal_power), ' signal power wo abs: '+str(np.mean(Ascan1)))
        #self.a.set_ylim(np.min(Ascan1)-0.05,np.max(Ascan1)+0.05)
        #self.line.set_ydata(Ascan1)
        #self.line.set_xdata(self.time_axis)
        #self.a.clear()
        #self.a.plot(self.time_axis,Ascan1)
        #self.canvas.draw()
        #print(timer() - start)
        return Ascan, Ascan1, transducer

    def collect_scan_finer_grid(self,p2, az, coe):
        transducer = Plane(p2, self.distance, az * np.pi / 180, coe * np.pi / 180, self.vres, self.hres, self.nv, self.nh, np.int(self.opening_angle.get())* np.pi / 180)
        transducer.prepareImagingPlane()  # this ALWAYS
        # has to be called right after creating a transducer!
        scan = np.abs(signal.hilbert(transducer.insonify(self.scenario,2,0)))

        # coe = coe * 180 / np.pi
        # az = az * 180 / np.pi
        # file_name = str(p2[0]) + str(p2[1]) + str(np.round(coe,2)) + str(np.round(az,2))
        # write_data(file_name,scan)
        # print(end2 - start)
        # scan = np.abs(signal.hilbert(scan))
        # scan = np.abs(scan)

        # print('Inside finer grid')
        # print(end-start,end1-end,end2-end1)
        return scan

    def up_x(self):
        if self.x_pos <= 20:
            self.x_pos += 1
            self.x.config(text=str(self.x_pos))
        else:
            self.x_pos = 20
            self.x.config(text=str(self.x_pos))

    def down_x(self):
        if self.x_pos >= -20:
            self.x_pos -= 1
            self.x.config(text=str(self.x_pos))
        else:
            self.x_pos = -20
            self.x.config(text=str(self.x_pos))

    def up_y(self):
        if self.y_pos <= 20:
            self.y_pos += 1
            self.y.config(text=str(self.y_pos))
        else:
            self.x_pos = 20
            self.y.config(text=str(self.y_pos))

    def down_y(self):
        if self.y_pos >= -20:
            self.y_pos -= 1
            self.y.config(text=str(self.y_pos))
        else:
            self.y_pos = -20
            self.y.config(text=str(self.y_pos))

    def up_coe(self):
        if self.coe_pos <= 30:
            self.coe_pos += 1
            self.coelevation.config(text=str(self.coe_pos))
        else:
            self.coe_pos = 30
            self.coelevation.config(text=str(self.coe_pos))

    def down_coe(self):
        if self.coe_pos >= -30:
            self.coe_pos -= 1
            self.coelevation.config(text=str(self.coe_pos))
        else:
            self.coe_pos = -30
            self.coelevation.config(text=str(self.coe_pos))

    def up_az(self):
        if self.az_pos <= 30:
            self.az_pos += 1
            self.azimuth.config(text=str(self.az_pos))
        else:
            self.az_pos = 30
            self.azimuth.config(text=str(self.az_pos))

    def down_az(self):
        if self.az_pos >= -30:
            self.az_pos -= 1
            self.azimuth.config(text=str(self.az_pos))
        else:
            self.az_pos = -30
            self.azimuth.config(text=str(self.az_pos))

    def find_peaks(self, ascan, g_pulse):
        y = np.abs(signal.hilbert(signal.correlate(ascan, g_pulse, mode='same')))
        # y = np.abs(signal.hilbert(ascan))
        # y = np.abs(y)
        peak_index = np.argmax(y)
        #print(peak_index)
        end_index = peak_index + 5
        start_index = peak_index - 5
        #peak_energy = np.sum(np.abs(signal.hilbert(y[start_index:end_index])))
        peak_energy = np.sum(y[start_index:end_index])
        return peak_energy, ascan, peak_index

    def calcualate_tof(self,peaks, speed, ts):
        tof = peaks * ts
        d = tof * speed / 2
        return d
    '''
    def gradient_ascent(self,step_size, step, scaling_factor, noofiteration, point_x, point_y, point_z):
        ph_init = 0
        th_init = 0
        peak = 0
        max_energy = 0
        some_l = []
        iters = []
        #energies = np.zeros(noofiteration)
        p3 = np.array([point_x, point_y, point_z])
        print('Inside gradient ascent fine')

        fig1 = Figure(figsize=(5, 5))
        a1 = fig1.add_subplot(111)

        line1,= a1.plot(iters, some_l)
        a1.set_title('Cost function')
        a1.set_xlim(0,noofiteration)
        # #a1.set_ylim(0,200)
        canvas = FigureCanvasTkAgg(fig1, master=self.window)
        canvas.get_tk_widget().place(x=650, y=0)
        canvas.draw()
        #a1.plot(energies

        for i in range(noofiteration):
            start = timer()
            a_scan,ascan_t,trans = self.init_transducer(p3, ph_init, (th_init + step_size))
            peak_energy, output_array, peak_i = self.find_peaks(a_scan, self.g)
            a_scan1,ascan_t,trans = self.init_transducer(p3, ph_init, (th_init - step_size))
            peak_energy1, output_array1, peak_i1 = self.find_peaks(a_scan1, self.g)

            grad_th = (peak_energy - peak_energy1) / (2 * step_size * scaling_factor)

            a_scan2,ascan_t,trans = self.init_transducer(p3, (ph_init + step_size), th_init)
            peak_energy2, output_array2, peak_i2 = self.find_peaks(a_scan2, self.g)
            a_scan3,ascan_t,trans = self.init_transducer(p3, (ph_init - step_size), th_init)
            peak_energy3, output_array3, peak_i3 = self.find_peaks(a_scan3, self.g)

            grad_ph = (peak_energy2 - peak_energy3) / (2 * step_size * scaling_factor)

            ph_init = ph_init + grad_ph * step
            th_init = th_init + grad_th * step

            a_scan4,ascan_t,trans = self.init_transducer(p3, ph_init, th_init)
            peak_energy4, output_array4, peak_i4 = self.find_peaks(a_scan4, self.g)
            max_energy = peak_energy4
            #energies[i] = max_energy
            iters.append(i)
            some_l.append(max_energy)
            #print(np.min(some_l), np.max(some_l))
            a1.set_ylim(np.min(some_l), np.max(some_l)+5)
            line1.set_ydata(some_l)
            line1.set_xdata(iters)
            canvas.draw()
            end = timer()
            print(max_energy,end-start)
            peak = peak_i4

        di1 = self.calcualate_tof(peak, self.c, self.Ts)
        x = point_x + di1 * np.sin(th_init * np.pi / 180) * np.cos(ph_init * np.pi / 180)
        y = point_y + di1 * np.sin(th_init * np.pi / 180) * np.sin(ph_init * np.pi / 180)
        z = di1 * np.cos(th_init * np.pi / 180)

        return max_energy, x, y, z, th_init, ph_init
        '''
    def gradient_ascent(self, delta, point_x, point_y, point_z):

        #energies = np.zeros(noofiteration)
        p3 = np.array([point_x, point_y, point_z])*1e-3
        print('Inside gradient ascent fine')

        #a1.plot(energies

        start = timer()
        a_scan,ascan_t,trans = self.init_transducer(p3, self.ph_init, (self.th_init + delta))
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)
        a_scan1,ascan_t,trans = self.init_transducer(p3, self.ph_init, (self.th_init - delta))
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * delta)

        a_scan2,ascan_t,trans = self.init_transducer(p3, (self.ph_init + delta), self.th_init)
        peak_energy2, output_array2, peak_i2 = self.find_peaks(ascan_t, self.g)
        a_scan3,ascan_t,trans = self.init_transducer(p3, (self.ph_init - delta), self.th_init)
        peak_energy3, output_array3, peak_i3 = self.find_peaks(ascan_t, self.g)

        grad_ph = (peak_energy2 - peak_energy3) / (2 * delta)

        self.ph_init = self.ph_init + grad_ph * self.step
        self.th_init = self.th_init + grad_th * self.step

        grad_vec = np.array([grad_th, grad_ph])
        grad_vec = np.reshape(grad_vec, (2, 1))
        #inner_product_grad = grad_vec.transpose().dot(self.prev_grad)/(np.linalg.norm(grad_vec)*np.linalg.norm(self.prev_grad))
        grad_vec_norm = np.linalg.norm(grad_vec)
        self.prev_grad = grad_vec
        a_scan4,ascan_t,trans = self.init_transducer(p3, self.ph_init, self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)
        max_energy = peak_energy4
        end = timer()
        print(max_energy,self.ph_init,peak_energy2,peak_energy3)

        self.index += 1
        '''
        if self.index == noofiteration:
            di1 = self.calcualate_tof(peak, self.c, self.Ts)
            x = point_x + di1 * np.sin(self.th_init * np.pi / 180) * np.cos(self.ph_init * np.pi / 180)
            y = point_y + di1 * np.sin(self.th_init * np.pi / 180) * np.sin(self.ph_init * np.pi / 180)
            z = di1 * np.cos(self.th_init * np.pi / 180)
            
            return max_energy, x, y, z, self.th_init, self.ph_init
        '''
        return self.index,max_energy

    def gradient_ascent_momentum(self, delta, point_x, point_y, point_z):
        p3 = np.array([point_x, point_y, point_z])*1e-3
        v = np.array([self.th_init, self.ph_init])
        v1 = np.reshape(v, (2, 1))

        print('Inside gradient ascent momentum')

        start = timer()

        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init + delta)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)
        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init - delta)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * delta)

        a_scan2, ascan_t, trans = self.init_transducer(p3, self.ph_init + delta, self.th_init)
        peak_energy2, output_array2, peak_i2 = self.find_peaks(ascan_t, self.g)
        a_scan3, ascan_t, trans = self.init_transducer(p3, self.ph_init - delta, self.th_init)
        peak_energy3, output_array3, peak_i3 = self.find_peaks(ascan_t, self.g)

        grad_ph = (peak_energy2 - peak_energy3) / (2 * delta)

        grad_vec = np.array([grad_th, grad_ph])
        grad_vec = np.reshape(grad_vec, (2, 1))

        grad_vec = self.step * grad_vec + self.momentum * self.prev_grad

        angle = grad_vec.transpose().dot(self.prev_grad)
        angle2 = np.asscalar(angle[0])
        print(angle2)
        #coef_pg = (-1) * (-2 * angle2 - 1)
        #coef_cg = 1 - coef_pg

        v1 = v1 + grad_vec

        self.th_init = np.asscalar(v1[0])
        self.ph_init = np.asscalar(v1[1])

        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)
        max_energy = peak_energy4
        end2 = timer()
        self.prev_grad = grad_vec
        print(max_energy, self.ph_init)

        self.index += 1

        return self.index, max_energy

    def spsa(self, a, c1, A, gamma, alpha, scale_step, scalefactor, point_x, point_y, point_z):

        p3 = np.array([point_x, point_y, point_z])*1e-3

        start2 = timer()
        step = a / ((self.index + A + 1) ** alpha) * scale_step
        ck = c1 / ((self.index + 1) ** gamma)
        deltar = np.random.binomial(1, 0.5)
        deltar = 2 * deltar - 1

        deltac = np.random.binomial(1, 0.5)
        deltac = 2 * deltac - 1

        a_scan,ascan_t,trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        #a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        #a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        #peak_energy11, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        a_scan1,ascan_t,trans = self.init_transducer(p3, self.ph_init - ck * deltar, self.th_init - ck * deltac)
        #a_scan = self.collect_scan_finer_grid(p3,self.ph_init - ck * deltar, self.th_init - ck * deltac)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        #a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        #peak_energy22, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * deltac * ck)

        #grad_th1 = (peak_energy11 - peak_energy22) / (2 * deltac * ck)

        grad_ph = (peak_energy - peak_energy1) / (2 * deltar * ck)

        #grad_ph1 = (peak_energy11 - peak_energy22) / (2 * deltar * ck)

        #val1 = np.where(self.epsilon > np.linalg.norm(grad_th1), self.epsilon, np.linalg.norm(grad_th1))
        #val2 = np.where(self.epsilon > np.linalg.norm(grad_th), self.epsilon, np.linalg.norm(grad_th))
        #val3 = np.where(self.epsilon > np.linalg.norm(grad_ph1), self.epsilon, np.linalg.norm(grad_ph1))
        #val4 = np.where(self.epsilon > np.linalg.norm(grad_ph), self.epsilon, np.linalg.norm(grad_ph))
        #print(val1, val2)

        #gradth2 = grad_th / val1 + grad_th1 / val2

        #gradph2 = grad_ph / val3 + grad_ph1 / val4

        gradv_final = np.array([grad_th, grad_ph])

        gradv_final = gradv_final.reshape(2, 1)

        grad_ph = np.asscalar(gradv_final[1])
        grad_th = np.asscalar(gradv_final[0])

        self.ph_init = self.ph_init + grad_ph * step
        self.th_init = self.th_init + grad_th * step

        #grad_vec = np.array([grad_th, grad_ph])
        #grad_vec = np.reshape(grad_vec, (2, 1))
        inner_product_grad = gradv_final.transpose().dot(self.prev_grad)
        grad_vec_norm = np.linalg.norm(gradv_final)
        #self.prev_grad = grad_vec

        a_scan4,ascan_t,trans= self.init_transducer(p3, self.ph_init, self.th_init)
        #a_scan = self.collect_scan_finer_grid(p3,self.ph_init,self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)

        max_energy = peak_energy4
        end2 = timer()
        print(max_energy)
        self.prev_grad = gradv_final
        self.index += 1

        return self.index,max_energy,inner_product_grad

    '''
    di1 = self.calcualate_tof(peak, c, Ts)
    x = point_x + di1 * np.sin(th_init * np.pi / 180) * np.cos(ph_init * np.pi / 180)
    y = point_y + di1 * np.sin(th_init * np.pi / 180) * np.sin(ph_init * np.pi / 180)
    z = di1 * np.cos(th_init * np.pi / 180)
    return max_energy, x, y, z, th_init, ph_init
    '''

    def rdsa(self, a, c1, A, gamma, alpha, scale_step, scalefactor, epsilon1, point_x, point_y, point_z):

        p3 = np.array([point_x, point_y, point_z]) * 1e-3

        step = a / ((self.index + A + 1) ** alpha) * scale_step
        ck = c1 / ((self.index + 1) ** gamma)
        prob = (1 + epsilon1)/(2 + epsilon1)

        deltar = bool(np.random.binomial(1, prob))
        if deltar:
            deltar = -1
        else:
            deltar = 1 + epsilon1

        deltac = bool(np.random.binomial(1, prob))
        if deltac:
            deltac = -1
        else:
            deltac = 1 + epsilon1

        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        #a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        #peak_energy11, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init - ck * deltar, self.th_init - ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init - ck * deltar, self.th_init - ck * deltac)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        #a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        #peak_energy22, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * deltac*ck)

        #grad_th1 = (peak_energy11 - peak_energy22) / (2 * deltac*ck)

        grad_ph = (peak_energy - peak_energy1) / (2 * deltar*ck)

        #grad_ph1 = (peak_energy11 - peak_energy22) / (2 * deltar*ck)

        #val1 = np.where(self.epsilon > np.linalg.norm(grad_th1), self.epsilon, np.linalg.norm(grad_th1))
        #val2 = np.where(self.epsilon > np.linalg.norm(grad_th), self.epsilon, np.linalg.norm(grad_th))
        #val3 = np.where(self.epsilon > np.linalg.norm(grad_ph1), self.epsilon, np.linalg.norm(grad_ph1))
        #val4 = np.where(self.epsilon > np.linalg.norm(grad_ph), self.epsilon, np.linalg.norm(grad_ph))
        #print(val1, val2)

        #gradth2 = grad_th / val1 + grad_th1 / val2

        #gradph2 = grad_ph / val3 + grad_ph1 / val4

        gradv_final = np.array([grad_th, grad_ph])

        gradv_final = gradv_final.reshape(2, 1)

        grad_ph = np.asscalar(gradv_final[1])
        grad_th = np.asscalar(gradv_final[0])

        self.ph_init = self.ph_init + grad_ph * step
        self.th_init = self.th_init + grad_th * step

        # grad_vec = np.array([grad_th, grad_ph])
        # grad_vec = np.reshape(grad_vec, (2, 1))
        inner_product_grad = gradv_final.transpose().dot(self.prev_grad)
        grad_vec_norm = np.linalg.norm(gradv_final)
        # self.prev_grad = grad_vec

        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init,self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)

        max_energy = peak_energy4
        end2 = timer()
        print(max_energy)
        self.prev_grad = gradv_final
        self.index += 1

        return self.index, max_energy, inner_product_grad

    def non_linear_conjugate_gradient_finer_grid(self, dr, dc, point_x, point_y, point_z):
        v = np.array([self.th_init,self.ph_init])
        v1 = np.reshape(v, (2, 1))

        #energies = np.zeros(iters)
        p3 = np.array([point_x, point_y, point_z])*1e-3
        print('Inside non_linear_conjugate_gradient_finer_grid')

        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init + dr)
        # a_scan = collect_scan_finer_grid(p3, ph_init, th_init + dr)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)
        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init - dr)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * dr)

        a_scan2, ascan_t, trans = self.init_transducer(p3, self.ph_init + dc, self.th_init)
        peak_energy2, output_array2, peak_i2 = self.find_peaks(ascan_t, self.g)
        a_scan3, ascan_t, trans = self.init_transducer(p3, self.ph_init - dc, self.th_init)
        peak_energy3, output_array3, peak_i3 = self.find_peaks(ascan_t, self.g)

        grad_ph = (peak_energy2 - peak_energy3) / (2 * dc)

        grad = np.array([grad_th, grad_ph])
        #grad_norm = np.linalg.norm(grad)

        grad_v = np.reshape(grad, (2, 1))
        #angle = np.arccos(grad_v.transpose().dot(self.prev_grad) / (np.linalg.norm(grad_v) * np.linalg.norm(self.prev_grad))) * 180 / np.pi
        #angle2 = np.asscalar(angle[0])/180

        inner_product_grad = grad_v.transpose().dot(self.prev_grad)
        #beta = (grad_v.transpose().dot((grad_v - self.prev_grad))) / (self.prev_grad.transpose().dot(self.prev_grad)) # Polak–Ribière
        beta = (grad_v.transpose().dot(grad_v)) / (self.prev_grad.transpose().dot(self.prev_grad)) # Fletcher and reeves
        s = grad_v + beta * self.prev_s
        self.prev_grad = grad_v
        #v1 = v1 + self.step * s
        v1 = v1 + self.step/(self.index + 1) * s
        self.th_init = np.asscalar(v1[0])
        self.ph_init = np.asscalar(v1[1])
        self.prev_s = s
        print(self.th_init,self.ph_init)
        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)
        max_energy = peak_energy4
        peak = peak_i4

        self.index += 1
        print(max_energy)

        return self.index, max_energy, inner_product_grad

    def calibrate(self):
        self.run = 0
        self.noofiter = int(self.noofiters.get())
        self.step = float(self.step_size.get())
        self.a2.set_xlim(0, self.noofiter)
        self.ph_init = self.az_pos
        self.th_init = self.coe_pos
        self.pulse_selection()
        #self.a.clear()
        #self.a.plot()
        # let's say I have 5 algorithms
        for i in range(5):
            self.lines[i].set_xdata(self.iters1)
            self.lines[i].set_ydata(self.energies)
            self.lines3[i].set_xdata(self.time_axis)
            self.lines3[i].set_ydata(np.zeros(self.NT))

        self.run_algo()

    def run_algo(self):
        if self.run == 0:
            some_val, s_energy,test_gradient = self.spsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, self.x_pos, self.y_pos, 0)
            cells[(self.run+1, 2)].config(text=str(np.round(s_energy,2)))
            cells[(self.run+1, 0)].config(text=str(np.round(self.ph_init,2)))
            cells[(self.run+1, 1)].config(text=str(np.round(self.th_init,2)))
            self.test_grads.append(test_gradient)
            self.energies.append(s_energy)
            self.iters1.append(some_val)
            self.a2.set_ylim(np.min(self.energies)-2, np.max(self.energies) + 2)
            #self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            #self.a2.plot(self.iters1,self.energies,label='Blue')
            self.lines[self.run].set_ydata(self.energies)
            #self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
        elif self.run == 1:
            some_val, s_energy = self.gradient_ascent(1, self.x_pos, self.y_pos, 0)
            cells[(self.run+1, 2)].config(text=str(np.round(s_energy,2)))
            cells[(self.run+1, 0)].config(text=str(np.round(self.ph_init,2)))
            cells[(self.run+1, 1)].config(text=str(np.round(self.th_init,2)))
            #self.test_grads.append(test_gradient)
            self.energies.append(s_energy)
            self.iters1.append(some_val)
            self.a2.set_ylim(np.min(self.energies)-2, np.max(self.energies) + 2)
            #self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            #self.a2.plot(self.iters1, self.energies,label='Red')
            #self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_ydata(self.energies)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
        elif self.run == 2:
            some_val, s_energy,test_gradient = self.non_linear_conjugate_gradient_finer_grid(2, 2,self.x_pos, self.y_pos, 0)
            cells[(self.run+1, 2)].config(text=str(np.round(s_energy,2)))
            cells[(self.run+1, 0)].config(text=str(np.round(self.ph_init,2)))
            cells[(self.run+1, 1)].config(text=str(np.round(self.th_init,2)))
            self.test_grads.append(test_gradient)
            self.energies.append(s_energy)
            self.iters1.append(some_val)
            self.a2.set_ylim(np.min(self.energies)-2, np.max(self.energies) + 2)
            #self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            self.lines[self.run].set_ydata(self.energies)
            #self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
        elif self.run == 3:
            some_val, s_energy = self.gradient_ascent_momentum(1, self.x_pos, self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy,2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init,2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init,2)))
            #self.test_grads.append(test_gradient)
            self.energies.append(s_energy)
            self.iters1.append(some_val)
            self.a2.set_ylim(np.min(self.energies)-2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            self.lines[self.run].set_ydata(self.energies)
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
        elif self.run == 4:
            some_val, s_energy, test_gradient = self.rdsa(0.4, 0.40, 0.2, 0.101, 0.5, 10, 1, 0.1, self.x_pos, self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy, 2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init, 2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init, 2)))
            self.test_grads.append(test_gradient)
            self.energies.append(s_energy)
            self.iters1.append(some_val)
            self.a2.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            # self.a2.plot(self.iters1,self.energies,label='Blue')
            self.lines[self.run].set_ydata(self.energies)
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
        else:
            self.run = np.inf

        #self.calib_az.config(text=str(np.round(self.ph_init,3)))
        #self.calib_coe.config(text=str(np.round(self.th_init,3)))

        if self.index <= self.noofiter:
            self.window.after(100,self.run_algo)
        else:
            self.index = 0
            #p = np.array([self.x_pos, self.y_pos, 0])*1e-3
            #a_scan1, ascan, trans = self.init_transducer(p, self.ph_init, self.th_init)
            #self.a.set_ylim(np.min(a_scan1) - 0.1, np.max(a_scan1) + 0.1)
            #self.lines3[self.run].set_ydata(a_scan1)
            #self.lines3[self.run].set_xdata(self.time_axis)
            #self.canvas.draw()

            self.iters1.clear()
            self.energies.clear()
            self.test_grads.clear()
            self.prev_grad = np.ones((2, 1))*1e-5
            self.prev_s = np.zeros((2, 1))
            self.ph_init = self.az_pos
            self.th_init = self.coe_pos
            self.run += 1
            self.window.after(500, self.run_algo)
            #return



