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
from threading import Thread
matplotlib.use('TkAgg')
import numpy as np
import scipy.io as sio
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import FormatStrFormatter
import numpy.matlib as nmat
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
from datetime import datetime

colors = ['red', 'green', 'brown', 'purple', 'cyan']  # 'gold','green','maroon''brown','purple','cyan'
legends = ['SPSA', 'gradient ascent', 'non linear cg', 'gradient ascent (momentum)', 'RDSA']
legends1 = ['1', '2', '3', '4']

# take the data
lst = [('Azimuth (deg)', 'Co-elevation (deg)', 'Energy', ''),
       ('', '', '', 'SPSA'),
       ('', '', '', 'Gradient Ascent'),
       ('', '', '', 'Non Linear CG'),
       ('', '', '', 'GA(with momentum)'),
       ('', '', '', 'RDSA')]

cells = {}
total_rows = 6
total_columns = 4


class positioner:
    def __init__(self, window):
        self.window = window

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

        self.target_snr_db = np.float(35)

        self.x0 = 0  # horizontal reference for reflecting plane location
        self.z0 = 20e-3  # vertical reference for reflecting plane location

        # I calculated these by hand D:
        self.center1 = np.array([-28.3564e-3, 0, self.z0 + 5e-3])
        # center1 = np.array([0, 0, z0 + 5e-3])
        self.az1 = np.pi
        self.co1 = 170 * np.pi / 180
        self.h1 = 20e-3
        self.w1 = 57.5877e-3
        self.rect1 = Rectangle_test(self.center1, self.az1, self.co1, self.h1, self.w1)

        self.center2 = np.array([18.66025e-3, 0, self.z0 + 5e-3])
        self.az2 = 0
        self.co2 = 165 * np.pi / 180
        self.h2 = 20e-3
        self.w2 = 38.6370e-3
        self.rect2 = Rectangle_test(self.center2, self.az2, self.co2, self.h2, self.w2)

        # put the reflecting rectangles in an array
        self.objects = np.array([self.rect1, self.rect2])

        # and put everything into a scenario object
        self.scenario = Scenario(self.objects, self.c, self.NT, self.fs, self.bw_factor, self.fc, self.phi,
                                 self.no_of_symbols, self.symbol_time)

        # now, the transducer parameters that will remain fixed throughout the sims:
        # self.opening = 45 * np.pi / 180  # opening angle in radians
        self.opening = np.int(15) * np.pi / 180  # opening angle in radians
        self.opening_test = 15
        self.nv = 121  # number of vertical gridpoints for the rays
        self.nh = 121  # number of horizontal gridpoints for the rays
        self.vres = 3e-3 / self.nv  # ray grid vertical resolution in [m] (set to 0.2mm)
        self.hres = 1.5e-3 / self.nh  # ray grid horizontal res in [m] (set to 0.1mm)
        self.distance = np.sqrt(3) * (self.nh - 1) / 2 * self.hres  # distance from center of transducer imaging plane to focal point [m]. this quantity guarantees that the opening spans 60° along the horizontal axis.
        self.time_axis = np.arange(self.NT) / self.fs
        self.epsilon = 1
        self.sigma = 0.5
        self.n = np.arange(-self.NT, self.NT) * self.Ts
        self.g = np.zeros(self.n.shape[0])
        self.a_time = 0.05 * 10 ** -6

        self.tabControl = ttk.Notebook(window)

        self.tab1 = ttk.Frame(self.tabControl,width=1800,height=1000)
        self.tab2 = ttk.Frame(self.tabControl,width=1800,height=1000)
        self.tab3 = ttk.Frame(self.tabControl,width=1800,height=1000)
        #self.tab4 = ttk.Frame(self.tabControl, width=1200, height=1000)
        self.tab5 = ttk.Frame(self.tabControl, width=1200, height=1000)

        self.tabControl.add(self.tab1, text='Positioner Control')
        self.tabControl.add(self.tab2, text='Stochastic Approximation')
        self.tabControl.add(self.tab3, text='Pulse Shaping')
        #self.tabControl.add(self.tab4, text='Stochastic approximation (demo tab)')
        self.tabControl.add(self.tab5, text='Test Scenarios')
        self.tabControl.place(x=0, y=0)

        self.h_b = 1
        self.w_b = 10

        #s = ttk.Style()
        #s.configure('Red.TLabelframe', font=("TkDefaultFont", 9, "bold"))
        self.label = ttk.Label(self.tab2,text="Configuration Parameters")
        self.label.config(font=('Helvetica bold', 20))
        self.configure = ttk.Labelframe(self.tab2,  height=200, width=450,labelwidget=self.label)#.grid(column=540,row=1250)
        self.configure.place(x=750, y=540)
        self.label.place(x=750, y=510)
        self.label1 = ttk.Label(self.tab1, text="Positioner Control")
        self.label1.config(font=('Helvetica bold', 20))
        self.pos_control = ttk.LabelFrame(self.tab1,  height=900, width=480,labelwidget=self.label1)#.grid(column=10,row=1250)
        self.pos_control.place(x=1350, y=10)
        self.label1.place(x=1350, y=1)
        self.data_generate = ttk.LabelFrame(self.tab2, text="Data Generation", height=200, width=400)#.grid(column=770,row=1250)
        self.data_generate.place(x=750, y=770)
        self.pulse_paramters = ttk.LabelFrame(self.tab3, text="Pulse Parameters", height=600,width=400)  # .grid(column=770,row=1250)
        self.pulse_paramters.place(x=1250, y=10)
        #self.Demo_parameters = ttk.LabelFrame(self.tab4, text="Demo parameters", height=600,width=400)  # .grid(column=770,row=1250)
        #self.Demo_parameters.place(x=750, y=10)
        self.label2 = ttk.Label(self.tab5, text="Algorithm Selection")
        self.label2.config(font=('Helvetica bold', 20))
        self.label2.place(x=10,y=10)
        self.label3 = ttk.Label(self.tab5, text="Step size")
        self.label3.config(font=('Helvetica bold', 20))
        self.label3.place(x=10, y=250)
        self.label4 = ttk.Label(self.tab5, text="SNR")
        self.label4.config(font=('Helvetica bold', 20))
        self.label4.place(x=500, y=250)
        self.label5 = ttk.Label(self.tab5, text="Opening Angle")
        self.label5.config(font=('Helvetica bold', 20))
        self.label5.place(x=1000, y=250)
        self.label6 = ttk.Label(self.tab5, text="Parameter Selection")
        self.label6.config(font=('Helvetica bold', 20))
        self.label6.place(x=1000, y=300)
        self.label7 = ttk.Label(self.tab5, text="Default Parameters")
        self.label7.config(font=('Helvetica bold', 20))
        self.label7.place(x=1000, y=10)
        self.Algorithm = ttk.LabelFrame(self.tab5,  height=200,width=400,labelwidget=self.label2)  # .grid(column=770,row=1250)
        self.Algorithm.place(x=10, y=10)
        self.Step_size = ttk.LabelFrame(self.tab5,  height=200,width=400,labelwidget=self.label3)  # .grid(column=770,row=1250)
        self.Step_size.place(x=10, y=250)
        self.SNR = ttk.LabelFrame(self.tab5,  height=200,width=400,labelwidget=self.label4)  # .grid(column=770,row=1250)
        self.SNR.place(x=500, y=250)
        self.Opening = ttk.LabelFrame(self.tab5,  height=200,width=400,labelwidget=self.label5)  # .grid(column=770,row=1250)
        self.Opening.place(x=1000, y=250)
        self.Parameter_select = ttk.LabelFrame(self.tab5,  height=200,width=400,labelwidget=self.label6)  # .grid(column=770,row=1250)
        self.Parameter_select.place(x=500, y=10)
        self.Default_parameters = ttk.LabelFrame(self.tab5, height=200, width=400,labelwidget=self.label7)  # .grid(column=770,row=1250)
        self.Default_parameters.place(x=1000, y=10)

        self.compare_calibrate = Button(self.tab5, text="Test Calibration", command=self.compare_calibration, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b + 10)
        self.compare_calibrate.place(x=1400,y=500)
        self.step_size1 = Entry(self.Step_size, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.step_size2 = Entry(self.Step_size, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.step_size3 = Entry(self.Step_size, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.step_size4 = Entry(self.Step_size, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.step_size_id1 = ttk.Label(self.Step_size, text="1", font=('Helvetica', 10, 'bold'))
        self.step_size_id2 = ttk.Label(self.Step_size, text="2", font=('Helvetica', 10, 'bold'))
        self.step_size_id3 = ttk.Label(self.Step_size, text="3", font=('Helvetica', 10, 'bold'))
        self.step_size_id4 = ttk.Label(self.Step_size, text="4", font=('Helvetica', 10, 'bold'))

        self.step_size1.insert(END, str(0.01))
        self.step_size2.insert(END, str(0.01))
        self.step_size3.insert(END, str(0.01))
        self.step_size4.insert(END, str(0.01))

        self.step_size1.place(x=10,y=30)
        self.step_size2.place(x=10, y=100)
        self.step_size3.place(x=200, y=30)
        self.step_size4.place(x=200, y=100)

        self.step_size_id1.place(x=50,y=10)
        self.step_size_id2.place(x=50, y=80)
        self.step_size_id3.place(x=230, y=10)
        self.step_size_id4.place(x=230, y=80)

        self.SNR1 = Entry(self.SNR, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.SNR2 = Entry(self.SNR, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.SNR3 = Entry(self.SNR, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.SNR4 = Entry(self.SNR, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.snr_id1 = ttk.Label(self.SNR, text="1", font=('Helvetica', 10, 'bold'))
        self.snr_id2 = ttk.Label(self.SNR, text="2", font=('Helvetica', 10, 'bold'))
        self.snr_id3 = ttk.Label(self.SNR, text="3", font=('Helvetica', 10, 'bold'))
        self.snr_id4 = ttk.Label(self.SNR, text="4", font=('Helvetica', 10, 'bold'))

        self.snr_id1.place(x=50,y=10)
        self.snr_id2.place(x=50, y=80)
        self.snr_id3.place(x=230, y=10)
        self.snr_id4.place(x=230, y=80)

        self.SNR1.insert(END, str(30))
        self.SNR2.insert(END, str(30))
        self.SNR3.insert(END, str(30))
        self.SNR4.insert(END, str(30))

        self.SNR1.place(x=10,y=30)
        self.SNR2.place(x=10, y=100)
        self.SNR3.place(x=200, y=30)
        self.SNR4.place(x=200, y=100)

        self.Opening1 = Entry(self.Opening, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Opening2 = Entry(self.Opening, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Opening3 = Entry(self.Opening, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Opening4 = Entry(self.Opening, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.opening_id1 = ttk.Label(self.Opening, text="1", font=('Helvetica', 10, 'bold'))
        self.opening_id2 = ttk.Label(self.Opening, text="2", font=('Helvetica', 10, 'bold'))
        self.opening_id3 = ttk.Label(self.Opening, text="3", font=('Helvetica', 10, 'bold'))
        self.opening_id4 = ttk.Label(self.Opening, text="4", font=('Helvetica', 10, 'bold'))

        self.Opening1.insert(END, str(15))
        self.Opening2.insert(END, str(15))
        self.Opening3.insert(END, str(15))
        self.Opening4.insert(END, str(15))

        self.Opening1.place(x=10,y=30)
        self.Opening2.place(x=10, y=100)
        self.Opening3.place(x=200, y=30)
        self.Opening4.place(x=200, y=100)

        self.opening_id1.place(x=50, y=10)
        self.opening_id2.place(x=50, y=80)
        self.opening_id3.place(x=230, y=10)
        self.opening_id4.place(x=230, y=80)

        self.radiobutton_variable1 = IntVar()
        r11 = Radiobutton(self.Parameter_select, text="SNR", variable=self.radiobutton_variable1, value=1,command=self.enable_disable_para)
        r11.select()
        r21 = Radiobutton(self.Parameter_select, text="Step Size", variable=self.radiobutton_variable1, value=2,command=self.enable_disable_para)
        r31 = Radiobutton(self.Parameter_select, text="Opening Angle", variable=self.radiobutton_variable1, value=3,command=self.enable_disable_para)

        r11.place(x=10, y=10)
        r21.place(x=150, y=10)
        r31.place(x=10, y=50)

        self.radiobutton_variable = IntVar()
        self.r1 = Radiobutton(self.Algorithm, text="SPSA", variable=self.radiobutton_variable, value=1)
        self.r1.select()
        self.r2 = Radiobutton(self.Algorithm, text="RDSA", variable=self.radiobutton_variable, value=2)
        self.r3 = Radiobutton(self.Algorithm, text="Gradient Ascent", variable=self.radiobutton_variable, value=3)
        self.r4 = Radiobutton(self.Algorithm, text="Gradient Ascent(momentum)", variable=self.radiobutton_variable, value=4)

        self.r1.place(x=10, y=10)
        self.r2.place(x=150, y=10)
        self.r3.place(x=10, y=50)
        self.r4.place(x=150, y=50)

        self.enable_disable_para()

        self.center_frequency = Label(self.pulse_paramters, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.sampling_frequency = Label(self.pulse_paramters, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.bandwidth_factor = Label(self.pulse_paramters, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        #self.std_dev = Label(self.pulse_paramters, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Pulse_selection = Label(self.pulse_paramters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))

        self.center_frequency.place(x=10,y=50)
        self.sampling_frequency.place(x=10, y=90)
        self.bandwidth_factor.place(x=10, y=130)
        #self.std_dev.place(x=10, y=170)
        self.Pulse_selection.place(x=10,y=10)

        self.center_frequency.config(text="Center Frequency")
        self.sampling_frequency.config(text="Sampling Frequency")
        self.bandwidth_factor.config(text="Bandwidth Factor")
        #self.std_dev.config(text="Variance")
        self.Pulse_selection.config(text="Pulse Selection")
        '''
        self.demo_x = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_y = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_coe = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_az = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_opening_angle = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_snr = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))

        self.demo_button = Button(self.Demo_parameters, text="Start", command=self.start_demo, borderwidth=2,relief='solid', font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b + 10)

        self.demo_x_label = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_y_label = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_coe_label = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_az_label = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_opening_angle_label = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.demo_snr_label = Label(self.Demo_parameters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))

        self.demo_x.place(x=10, y=10)
        self.demo_y.place(x=10, y=60)
        self.demo_coe.place(x=10, y=110)
        self.demo_az.place(x=10, y=160)
        self.demo_opening_angle.place(x=10, y=210)
        self.demo_snr.place(x=10, y=260)
        self.demo_button.place(x=10, y=360)

        self.demo_x_label.place(x=100, y=10)
        self.demo_y_label.place(x=100, y=60)
        self.demo_coe_label.place(x=100, y=110)
        self.demo_az_label.place(x=100, y=160)
        self.demo_opening_angle_label.place(x=100, y=210)
        self.demo_snr_label.place(x=100, y=260)

        self.demo_x.config(text=str(10))
        self.demo_y.config(text=str(0))
        self.demo_coe.config(text=str(0))
        self.demo_az.config(text=str(0))
        self.demo_opening_angle.config(text=str(15))
        self.demo_snr.config(text=str(35))
        self.demo_x_label.config(text="X position (in mm)")
        self.demo_y_label.config(text="Y position (in mm)")
        self.demo_coe_label.config(text="Coelevation (degrees)")
        self.demo_az_label.config(text="Azimuth (degrees)")
        self.demo_opening_angle_label.config(text="Opening Angle (degrees)")
        self.demo_snr_label.config(text="SNR (in dB)")
        '''
        self.x = Label(self.pos_control, borderwidth=2, relief='ridge')
        self.y = Label(self.pos_control, borderwidth=2, relief='ridge')
        self.azimuth = Label(self.pos_control, borderwidth=2, relief='ridge')
        self.coelevation = Label(self.pos_control, borderwidth=2, relief='ridge')
        # self.calib_az = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        # self.calib_coe = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.x_label = Label(self.pos_control, borderwidth=2, relief='ridge')
        self.y_label = Label(self.pos_control, borderwidth=2, relief='ridge')
        self.az_label = Label(self.pos_control, borderwidth=2, relief='ridge')
        self.coe_label = Label(self.pos_control, borderwidth=2, relief='ridge')

        self.x.place(x=10, y=110)
        self.y.place(x=10, y=210)
        self.azimuth.place(x=10, y=310)
        self.coelevation.place(x=10, y=410)

        self.x_label.place(x=110, y=110)
        self.y_label.place(x=110, y=210)
        self.az_label.place(x=110, y=310)
        self.coe_label.place(x=110, y=410)

        # self.calib_az_label = Label(window,borderwidth=2,relief='ridge',font=('Helvetica', 10, 'bold'))
        # self.calib_coe_label = Label(window, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))

        self.noofiters = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.noofiters_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.noofiters.insert(END,str(50))
        self.window_length = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.window_length_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.window_length.insert(END, str(15))
        self.step_size = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.step_size.insert(END,str(0.01))
        self.step_size_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.target_snr = Entry(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.target_snr.insert(END, str(35))
        self.target_snr_label = Label(self.configure, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.opening_angle = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.opening_angle.insert(END, str(15))
        self.opening_angle_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.X_pos_entry = Entry(self.data_generate, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.X_pos_entry.insert(END, str(0))
        self.X_pos_entry_label = Label(self.data_generate, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.Y_pos_entry = Entry(self.data_generate, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Y_pos_entry.insert(END, str(0))
        self.Y_pos_entry_label = Label(self.data_generate, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.az_grid_entry = Entry(self.data_generate, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.az_grid_entry.insert(END, str(1))
        self.az_grid_entry_label = Label(self.data_generate, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.coe_grid_entry = Entry(self.data_generate, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.coe_grid_entry.insert(END, str(1))
        self.coe_grid_entry_label = Label(self.data_generate, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))

        self.generate = Button(self.data_generate, text="Generate Data", command=self.generate_data, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b,width=self.w_b)  # .grid(column=210,row=440)
        self.generate.place(x=100, y=150)

        self.noofiters.place(x=120, y=10)
        self.noofiters_label.place(x=10, y=10)
        self.step_size.place(x=120, y=40)
        self.step_size_label.place(x=10, y=40)
        self.target_snr.place(x=120, y=70)
        self.target_snr_label.place(x=10, y=70)
        self.opening_angle.place(x=10, y=620)
        self.opening_angle_label.place(x=170, y=620)
        self.window_length.place(x=120, y=100)
        self.window_length_label.place(x=10, y=100)
        self.X_pos_entry.place(x=140,y=10)
        self.X_pos_entry_label.place(x=10,y=10)
        self.Y_pos_entry.place(x=140,y=40)
        self.Y_pos_entry_label.place(x=10, y=40)
        self.az_grid_entry.place(x=140,y=70)
        self.az_grid_entry_label.place(x=10,y=70)
        self.coe_grid_entry.place(x=140,y=100)
        self.coe_grid_entry_label.place(x=10,y=100)

        self.coe_grid_entry_label.config(text=str("Coelevation grid"))
        self.az_grid_entry_label.config(text=str("Azimuth grid"))
        self.X_pos_entry_label.config(text=str("X Pos"))
        self.Y_pos_entry_label.config(text=str("Y Pos"))

        '''
        self.noofiters = Entry(self.configure)#.grid(column=120,row=10)
        self.noofiters.grid(column=120,row=10)
        self.noofiters_label = Label(self.configure).grid(column=10,row=10)
        self.noofiters.insert(END, str(50))
        self.step_size = Entry(self.configure).grid(column=120,row=40)
        
        self.step_size.insert(END, str(0.1))
        self.step_size_label = Label(self.configure).grid(column=10,row=40)
        self.target_snr = Entry(self.configure).grid(column=120,row=70)
        self.target_snr.insert(END, str(35))
        self.target_snr_label = Label(self.configure).grid(column=10,row=70)
        self.opening_angle = Entry(self.configure).grid(column=120,row=110)
        self.opening_angle.insert(END, str(40))
        self.opening_angle_label = Label(self.configure).grid(column=10,row=110)
        '''
        self.test_flag = 0
        self.x_pos = 0
        self.y_pos = 0
        self.x_pos_demo = 10
        self.y_pos_demo = 0
        self.az_pos = 0
        self.coe_pos = 0
        self.az_back_up = 0
        self.coe_back_up = 0
        self.ph_init = 0
        self.th_init = 0
        self.noofiter = 0
        self.step = 0
        self.run = 0
        self.prev_grad = np.ones((2, 1)) * 1e-5
        self.prev_s = np.zeros((2, 1))

        self.x.config(text=str(self.x_pos))
        self.y.config(text=str(self.y_pos))
        self.azimuth.config(text=str(self.az_pos))
        self.coelevation.config(text=str(self.coe_pos))
        # self.calib_az.config(text=str(self.ph_init))
        # self.calib_coe.config(text=str(self.th_init))

        self.az_label.config(text='Azimuth')
        self.coe_label.config(text='Coelevation')
        self.x_label.config(text='x')
        self.y_label.config(text='y')
        # self.calib_az_label.config(text='Azimuth(calibrated)')
        # self.calib_coe_label.config(text='Co-elevation(calibrated)')
        self.noofiters_label.config(text='Iterations')
        self.step_size_label.config(text='Step size')
        self.target_snr_label.config(text='SNR in dB')
        self.opening_angle_label.config(text='Opening Angle')
        self.window_length_label.config(text='Window Length')

        self.button = Button(self.pos_control, text="Update Transducer", command=self.plot, borderwidth=2,relief='solid', font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b + 10)#.grid(column=210, row=10)
        self.button.place(x=210, y=10)

        self.upx = Button(self.pos_control, text="Up x", command=self.up_x, borderwidth=2, relief='solid',
                          font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210, row=110)
        self.upx.place(x=210, y=110)
        self.downx = Button(self.pos_control, text="Down x", command=self.down_x, borderwidth=2, relief='solid',
                            font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210, row=140)
        self.downx.place(x=210, y=140)

        self.upy = Button(self.pos_control, text="Up y", command=self.up_y, borderwidth=2, relief='solid',
                          font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=210)
        self.upy.place(x=210, y=210)
        self.downy = Button(self.pos_control, text="Down y", command=self.down_y, borderwidth=2, relief='solid',
                            font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=240)
        self.downy.place(x=210, y=240)

        self.upaz = Button(self.pos_control, text="Up az", command=self.up_az, borderwidth=2, relief='solid',
                           font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=310)
        self.upaz.place(x=210, y=310)
        self.downaz = Button(self.pos_control, text="Down az", command=self.down_az, borderwidth=2, relief='solid',
                             font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=340)
        self.downaz.place(x=210, y=340)

        self.upcoe = Button(self.pos_control, text="Up coe", command=self.up_coe, borderwidth=2, relief='solid',
                            font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=410)
        self.upcoe.place(x=210, y=410)
        self.downcoe = Button(self.pos_control, text="Down coe", command=self.down_coe, borderwidth=2, relief='solid',
                              font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=440)
        self.downcoe.place(x=210, y=440)

        self.Entry_step_x = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_step_x.place(x=10, y=500)
        self.Entry_step_x.insert(END,str(1))

        self.Entry_step_y = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_step_y.place(x=10, y=530)
        self.Entry_step_y.insert(END, str(1))

        self.Entry_step_x_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_step_x_label.place(x=170, y=500)
        self.Entry_step_x_label.config(text='x step')

        self.Entry_step_y_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_step_y_label.place(x=170, y=530)
        self.Entry_step_y_label.config(text='y step')

        self.Entry_az_step = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_az_step.place(x=10, y=560)
        self.Entry_az_step.insert(END, str(1))

        self.Entry_coe_step = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_coe_step.place(x=10, y=590)
        self.Entry_coe_step.insert(END, str(1))

        self.Entry_az_step_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_az_step_label.place(x=170, y=560)
        self.Entry_az_step_label.config(text='Azimuth step')

        self.Entry_coe_step_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_coe_step_label.place(x=170, y=590)
        self.Entry_coe_step_label.config(text='Coelevation step')

        '''
        self.Entry_x = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_x.place(x=10,y=500)

        self.Entry_y = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_y.place(x=10, y=530)

        self.Entry_az = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_az.place(x=10, y=560)

        self.Entry_coe = Entry(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_coe.place(x=10, y=590)

        self.Entry_x_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_x_label.place(x=170,y=500)
        self.Entry_x_label.config(text='x pos')

        self.Entry_y_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_y_label.place(x=170, y=530)
        self.Entry_y_label.config(text='y pos')

        self.Entry_az_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_az_label.place(x=170, y=560)
        self.Entry_az_label.config(text='Azimuth')

        self.Entry_coe_label = Label(self.pos_control, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.Entry_coe_label.place(x=170, y=590)
        self.Entry_coe_label.config(text='Coelevation')
        '''
        #self.Entry_x.insert(END, str(50))

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

        self.calibrate = Button(self.configure, text="Calibrate", command=self.calibrate, borderwidth=2, relief='solid',font=('Helvetica', 10, 'bold'), height=self.h_b, width=self.w_b)#.grid(column=210,row=440)
        self.calibrate.place(x=50, y=150)

        #self.b1 = Button(self.configure, text="Start", command=self.thread_start)
        #self.b1.place(x=120, y=150)
        # self.calib_az.place(x=1400, y=700)
        # self.calib_coe.place(x=1400, y=730)
        # self.calib_az_label.place(x=1200, y=700)
        # self.calib_coe_label.place(x=1200, y=730)
        '''
        self.noofiters.place(x=1350,y=800)
        self.noofiters_label.place(x=1200,y=800)
        self.step_size.place(x=1350,y=830)
        self.step_size_label.place(x=1200, y=830)
        self.target_snr.place(x=1350,y=860)
        self.target_snr_label.place(x=1200,y=860)
        '''

        self.entry_center_frequency = Entry(self.pulse_paramters, borderwidth=2, relief='ridge', font=('Helvetica', 10, 'bold'))
        self.entry_center_frequency.place(x=200, y=50)
        self.entry_center_frequency.insert(END, str(self.fc))

        self.entry_sample_frequency = Entry(self.pulse_paramters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.entry_sample_frequency.place(x=200, y=90)
        self.entry_sample_frequency.insert(END, str(self.fs))

        self.entry_bw_factor = Entry(self.pulse_paramters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        self.entry_bw_factor.place(x=200, y=130)
        self.entry_bw_factor.insert(END, str(self.bw_factor))

        #self.entry_variance = Entry(self.pulse_paramters, borderwidth=2, relief='ridge',font=('Helvetica', 10, 'bold'))
        #self.entry_variance.place(x=200, y=170)
        #self.entry_variance.insert(END, str(self.sigma))

        #self.pulse_data = np.zeros((2*self.NT,4))
        self.acf = np.zeros(2*self.NT)
        self.psd = np.zeros(2*self.NT)

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

        self.cal_acf_psd(self.sinc_pulse)

        # self.g = signal.hilbert(self.g)
        # self.g = np.abs(self.g)
        # self.g = self.g / np.max(self.g)

        self.index = 0
        self.energies = []
        self.energies.append(0)
        self.grad_en = []
        self.spsa_en = []
        self.iters1 = []
        self.iters1.append(0)
        self.test_grads = []
        self.azimuth_list = []
        self.elevation_list = []

        self.pulse_select = 0
        self.momentum = 0.9

        self.OPTIONS = [
            "Sinc Pulse",
            "Ricker Wavelet",
            "Gaussian Pulse",
            "Gabor Pulse",
        ]  # etc

        self.variable = StringVar(self.tab3)
        self.variable.set(self.OPTIONS[0])  # default value
        # self.variable.trace("w", self.callback)

        self.w = OptionMenu(self.pulse_paramters, self.variable, *self.OPTIONS, command=self.callback)
        self.w.place(x=200, y=5)

        point = np.array([self.x_pos, self.y_pos, 0])
        self.fig = Figure(figsize=(6, 5))
        self.a = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab2)
        self.canvas.get_tk_widget().place(x=0, y=0)

        self.pulse_selection()
        coords_r1 = np.arange(-15, 15, 1)
        coords_c1 = np.arange(-15, 15, 1)
        '''
        interpolated2 = sio.loadmat('C:/Users/user/Documents/ARP_data_new/Simulator_data_rect_x=10_opening_15.mat')
        interpolated2 = interpolated2['Simulator']

        coords_r1 = np.arange(-15, 15, 1)
        coords_c1 = np.arange(-15, 15, 1)
        interpolated3 = np.zeros((coords_r1.shape[0], coords_c1.shape[0]))

        for i in range(len(coords_r1)):
            for j in range(len(coords_c1)):
                index = np.argmax(interpolated2[:, j, i])
                # print(index)
                interpolated3[j, i] = np.sum(interpolated2[index - 15:index + 15, j, i])
        '''
        interpolated4 = np.ones((coords_r1.shape[0], coords_c1.shape[0]))

        ascan1, ascan, trans = self.init_transducer(point, self.az_pos, self.coe_pos,self.test_flag)

        directions = trans.rays.direction
        sources = trans.rays.source
        directivity_pattern = trans.directivity.reshape(self.nv*self.nh,1)
        directions = directions.reshape(directions.shape[0] * directions.shape[1], 3)
        directions = directions*directivity_pattern

        s1 = np.zeros(2)
        s1[0] = sources[0]
        s1[1] = 0 #sources[2]
        origin = np.repeat(s1[:, np.newaxis], self.nv * self.nh, axis=1)
        color_magnitude = directivity_pattern.reshape(self.nv * self.nh, 1)

        #self.line, = self.a.plot(self.time_axis, ascan)
        im = self.a.imshow(interpolated4/np.max(interpolated4), extent=[-20, 20, 20, -20])
        #self.a.plot([0],[0],'r*')
        #self.fig.colorbar(im)
        self.a.set_title('Path Traced')
        self.a.set_xlabel('Azimuth (degrees)')
        self.a.set_ylabel('elevation (degrees)')
        #self.line, = self.a.plot(self.az_pos,self.coe_pos,'ro-')
        #self.a.set_title('Ascan')
        #self.a.set_xlabel('time (us)')
        #self.a.set_ylabel('Amplitude')
        self.canvas.draw()

        self.fig1 = Figure(figsize=(6, 5))
        self.a1 = self.fig1.add_subplot(111)
        #self.a1.imshow(trans.image)
        self.quiver5 = self.a1.quiver(*origin, directions[:, 0], directions[:, 2], color_magnitude,scale=2)
        self.a1.axis([-20 * 1e-3, 20 * 1e-3, 0 * 1e-3, 40 * 1e-3])
        self.a1.set_xlabel('X axis')
        self.a1.set_ylabel('Z axis (height)')
        self.a1.set_title('Transducer Beam Pattern')

        self.canvas2 = FigureCanvasTkAgg(self.fig1, master=self.tab1)
        self.canvas2.get_tk_widget().place(x=0, y=500)
        self.canvas2.draw()

        self.fig2 = Figure(figsize=(6, 5))
        self.a2 = self.fig2.add_subplot(111)
        self.line1, = self.a2.plot([],[],'ro-')
        self.a2.set_title('Cost function')
        self.a2.set_xlabel('Iterations')
        self.a2.set_ylabel('Energy')

        self.canvas1 = FigureCanvasTkAgg(self.fig2, master=self.tab2)
        self.canvas1.get_tk_widget().place(x=600, y=0)
        self.canvas1.draw()

        m1 = 5 / 28.3564
        m2 = -5 / 18.66025

        x1 = np.linspace(-28.3564e-3, 0, num=20, endpoint=True)
        x2 = np.linspace(0, 18.66025e-3, num=20, endpoint=True)
        x = np.concatenate((x1, x2), axis=0)

        z1 = (m1 * (x1 + 28.3564e-3) + 0e-3)
        z2 = (m2 * (x2 - 18.66025e-3) + 0e-3)
        z = np.concatenate((z1, z2), axis=0)

        y = np.linspace(-13, 13, 13) * 1e-3

        X, Y = np.meshgrid(x, y)
        Z = nmat.repmat(z, len(y), 1)

        self.fig3 = Figure(figsize=(6, 5))
        self.a3 = self.fig3.add_subplot(1, 1, 1, projection='3d')
        self.a3.set_title('Measurement Setup(Front View)')

        ax = self.a3.plot_surface(X, Y, Z,label='ground truth')
        ax._facecolors2d = ax._facecolor3d
        ax._edgecolors2d = ax._edgecolor3d

        self.a3.set_yticks([])
        self.a3.set_xticks(np.round(x, 2))
        #self.a3.set_zticks(np.round(z, 2))
        #self.a3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.a3.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        self.a3.set_xlabel('x-axis [m]',labelpad=10)
        self.a3.set_ylabel('y-axis [m]',labelpad=10)
        self.a3.set_zlabel('Height [m]',labelpad=10)

        self.a3.xaxis.label.set_size(10)
        self.a3.yaxis.label.set_size(10)
        self.a3.zaxis.label.set_size(10)
        # ax2.invert_zaxis()

        o_point2 = np.array([point[0],point[1],25e-3])

        self.source_point, = self.a3.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o', markersize=10, alpha=0.8,label='Transducer')
        #z_point = o_point2[2] + 1e-3 * np.cos(self.coe_pos)
        self.quiver2 = self.a3.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]],[-o_point2[0] + point[0] + 15e-3 * np.sin(self.coe_pos) * np.cos(self.az_pos)],[-o_point2[1] + point[1] + 15e-3 * np.sin(self.az_pos) * np.sin(self.coe_pos)], [-15e-3 * np.cos(self.coe_pos)], linewidths=(5,),edgecolor="red", label='Direction')
        self.a3.view_init(elev=0,azim=-90)
        #self.a3.view_init(elev=0, azim=-90)
        # self.a3.set_xmargin(-0.4)
        # self.a3.set_ymargin(-0.45)
        # self.a3.set_zmargin(0.5)
        self.a3.legend()

        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab1)
        self.canvas3.get_tk_widget().place(x=-30, y=0)
        self.canvas3.draw()

        self.fig4 = Figure(figsize=(6, 5))
        self.a4 = self.fig4.add_subplot(1, 1, 1, projection='3d')
        self.a4.set_title('Measurement Setup(Top View)')
        ax1 = self.a4.plot_surface(X, Y, Z,label='ground truth')
        ax1._facecolors2d = ax._facecolor3d
        ax1._edgecolors2d = ax1._edgecolor3d
        self.a4.set_yticks(np.round(y,2))
        self.a4.set_xticks(np.round(x,2))
        self.a4.set_zticks([])

        self.a4.set_xlabel('x-axis [m]',labelpad=10)
        self.a4.set_ylabel('y-axis [m]',labelpad=10)
        self.a4.set_zlabel('Height [m]',labelpad=10)

        self.a4.xaxis.label.set_size(10)
        self.a4.yaxis.label.set_size(10)
        self.a4.zaxis.label.set_size(10)

        print(o_point2)

        self.source_point1, = self.a4.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o',
                                         markersize=10, alpha=0.8, label='Transducer')
        # z_point = o_point2[2] + 1e-3 * np.cos(self.coe_pos)
        self.quiver3 = self.a4.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]],
                                      [-o_point2[0] + point[0] + 15e-3 * np.sin(self.coe_pos) * np.cos(self.az_pos)],
                                      [-o_point2[1] + point[1] + 15e-3 * np.sin(self.az_pos) * np.sin(self.coe_pos)],
                                      [-15e-3 * np.cos(self.coe_pos)], linewidths=(5,), edgecolor="red",
                                      label='Direction')
        self.a4.view_init(elev=90, azim=-90)
        #self.a4.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #self.a4.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # self.a4.set_xmargin(-0.4)
        # self.a4.set_ymargin(-0.45)
        # self.a4.set_zmargin(0.5)
        self.a4.legend(loc='best')

        self.canvas4 = FigureCanvasTkAgg(self.fig4, master=self.tab1)
        self.canvas4.get_tk_widget().place(x=570, y=0)
        self.canvas4.draw()

        self.fig5 = Figure(figsize=(6, 5))
        self.a5 = self.fig5.add_subplot(1, 1, 1, projection='3d')
        self.a5.set_title('Measurement Setup(Side View)')
        self.a5.set_yticks(np.round(y, 2))
        #self.a5.set_zticks(np.round(z, 2))
        self.a5.set_xticks([])
        #self.a5.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #self.a5.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2 = self.a5.plot_surface(X, Y, Z,label='ground truth')
        ax2._facecolors2d = ax2._facecolor3d
        ax2._edgecolors2d = ax2._edgecolor3d

        self.a5.set_xlabel('x-axis [m]',labelpad=10)
        self.a5.set_ylabel('y-axis [m]',labelpad=10)
        self.a5.set_zlabel('Height [m]',labelpad=10)

        self.a5.xaxis.label.set_size(10)
        self.a5.yaxis.label.set_size(10)
        self.a5.zaxis.label.set_size(10)

        self.a5.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        self.source_point2, = self.a5.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o',
                                          markersize=10, alpha=0.8, label='Transducer')
        # z_point = o_point2[2] + 1e-3 * np.cos(self.coe_pos)
        self.quiver4 = self.a5.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]],
                                      [-o_point2[0] + point[0] + 15e-3 * np.sin(self.coe_pos) * np.cos(self.az_pos)],
                                      [-o_point2[1] + point[1] + 15e-3 * np.sin(self.az_pos) * np.sin(self.coe_pos)],
                                      [-15e-3 * np.cos(self.coe_pos)], linewidths=(5,), edgecolor="red",
                                      label='Direction')
        self.a5.view_init(elev=3, azim=1)
        #self.a5.set_xmargin(-0.4)
        #self.a5.set_ymargin(-0.45)
        #self.a5.set_zmargin(0.5)
        self.a5.legend()

        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=self.tab1)
        self.canvas5.get_tk_widget().place(x=570, y=500)
        self.canvas5.draw()

        self.fig6 = Figure(figsize=(6, 5))
        self.a6 = self.fig6.add_subplot(111)
        self.canvas6 = FigureCanvasTkAgg(self.fig6, master=self.tab3)
        self.canvas6.get_tk_widget().place(x=0, y=0)
        self.a6.set_title('Original Pulse')
        self.a6.set_xlabel('Samples')
        self.a6.set_ylabel('Amplitude')
        self.a6.plot(self.gaussian)

        self.fig7 = Figure(figsize=(6, 5))
        self.a7 = self.fig7.add_subplot(111)
        self.canvas7 = FigureCanvasTkAgg(self.fig7, master=self.tab3)
        self.canvas7.get_tk_widget().place(x=550, y=0)
        self.a7.set_title('Ascan')
        self.a7.set_xlabel('Samples')
        self.a7.set_ylabel('Amplitude')
        self.a7.plot(ascan)
        self.a7.set_ylim(np.min(ascan) - 0.1, np.max(ascan) + 0.1)

        '''
        self.fig8 = Figure(figsize=(6, 5))
        self.a8 = self.fig8.add_subplot(111)
        #self.a8.plot([0],[0],'r*')
        self.canvas8 = FigureCanvasTkAgg(self.fig8, master=self.tab4)
        self.canvas8.get_tk_widget().place(x=0, y=0)
        
        im = self.a8.imshow(interpolated3 / np.max(interpolated3), extent=[-20, 20, 20, -20])
        self.fig8.colorbar(im)
        self.canvas8.draw()
        
        self.fig9 = Figure(figsize=(6, 5))
        self.a9 = self.fig9.add_subplot(111)
        self.a9.plot([],[],'ro-')
        self.a9.set_xlabel('Iterations')
        self.a9.set_ylabel('Energy')
        self.a9.set_title('Cost Function')
        self.canvas9 = FigureCanvasTkAgg(self.fig9, master=self.tab4)
        self.canvas9.get_tk_widget().place(x=0, y=500)
        '''
        self.fig10 = Figure(figsize=(6, 5))
        self.a10 = self.fig10.add_subplot(111)
        self.a10.plot(np.arange(4*self.NT-1),self.acf/np.max(self.acf))
        self.a10.set_xlabel('Samples')
        self.a10.set_ylabel('Magnitude')
        self.a10.set_title('ACF')
        self.canvas10 = FigureCanvasTkAgg(self.fig10, master=self.tab3)
        self.canvas10.get_tk_widget().place(x=0, y=500)

        self.fig11 = Figure(figsize=(6, 5))
        self.a11 = self.fig11.add_subplot(111)
        self.a11.plot(np.arange(-self.NT,self.NT)*(self.fs/(2*self.NT))*1e-6,self.psd)
        self.a11.set_xlabel('Frequency (in MHz)')
        self.a11.set_ylabel('Magnitude')
        self.a11.set_title('PSD')
        self.canvas11 = FigureCanvasTkAgg(self.fig11, master=self.tab3)
        self.canvas11.get_tk_widget().place(x=550, y=500)

        self.fig12 = Figure(figsize=(6, 5))
        self.a12 = self.fig12.add_subplot(111)
        self.a12.plot([],[],'ro-')
        self.a12.set_xlabel('Iterations')
        self.a12.set_ylabel('Energy')
        self.a12.set_title('Cost Function')
        self.canvas12 = FigureCanvasTkAgg(self.fig12, master=self.tab5)
        self.canvas12.get_tk_widget().place(x=0, y=500)
        self.canvas12.draw()

        self.fig13 = Figure(figsize=(6, 5))
        self.a13 = self.fig13.add_subplot(111)
        im = self.a13.imshow(interpolated4 / np.max(interpolated4), extent=[-20, 20, 20, -20])
        self.a13.set_title('Path Traced')
        self.a13.set_xlabel('Azimuth (degrees)')
        self.a13.set_ylabel('elevation (degrees)')
        self.canvas13 = FigureCanvasTkAgg(self.fig13, master=self.tab5)
        self.canvas13.get_tk_widget().place(x=600, y=500)
        self.canvas13.draw()

        self.lines = []
        self.lines3 = []
        self.lines4 = []
        self.lines5 = []
        self.lines6 = []

        for index in range(5):
            lobj = self.a2.plot(self.iters1, self.energies, lw=1, color=colors[index], label=legends[index])[0]
            #lobj1 = self.a9.plot(self.az_pos,self.coe_pos, lw=3, color=colors[index], label=legends[index])[0]
            #lobj2 = self.a8.plot(self.azimuth_list, self.elevation_list, lw=3, color=colors[index], label=legends[index])[0]
            lobj3 = self.a.plot(self.azimuth_list, self.elevation_list, lw=3, color=colors[index], label=legends[index])[0]

            self.lines3.append(lobj3)
            #self.lines5.append(lobj2)
            self.lines.append(lobj)
            #self.lines4.append(lobj1)

            #self.lines3.append(lobj1)
            self.a.legend()
            self.a2.legend()
            #self.a9.legend()
            #self.a8.legend()

        for index in range(4):
            lobj4 = self.a12.plot(self.iters1, self.energies, lw=1, color=colors[index], label=legends1[index])[0]
            self.lines6.append(lobj4)
            self.a12.legend()

        #self.azimuth_list.append(self.az_pos)
        #self.elevation_list.append(self.coe_pos)

        self.para_snr = []
        self.para_step = []
        self.para_opening = []
        self.algorithm_select = 0
        self.para_select = 0
        self.para_select_index = 0

        for i in range(total_rows):
            for j in range(total_columns):
                self.e = Label(self.tab2, width=20, fg='blue', font=('Helvetica', 10, 'bold'))
                self.e.grid(row=i, column=j)
                cells[(i, j)] = self.e
                self.e.place(x=0 + j * 140, y=600 + i * 50)
                # print(lst[i][j])
                self.e.config(text=str(lst[i][j]))

    def enable_disable_para(self):
        val1 = self.radiobutton_variable1.get()
        if val1 == 1:
            for child in self.Step_size.winfo_children():
                child.configure(state='disable')
            for child in self.Opening.winfo_children():
                child.configure(state='disable')
            for child in self.SNR.winfo_children():
                child.configure(state='normal')
            self.r1.select()
            self.r1.configure(state=NORMAL)
            self.r2.configure(state=NORMAL)
        elif val1 == 2:
            for child in self.SNR.winfo_children():
                child.configure(state='disable')
            for child in self.Opening.winfo_children():
                child.configure(state='disable')
            for child in self.Step_size.winfo_children():
                child.configure(state='normal')
            self.r1.deselect()
            self.r1.configure(state=DISABLED)
            self.r2.configure(state=DISABLED)
            self.r3.select()
        else:
            for child in self.SNR.winfo_children():
                child.configure(state='disable')
            for child in self.Step_size.winfo_children():
                child.configure(state='disable')
            for child in self.Opening.winfo_children():
                child.configure(state='normal')
            self.r1.select()
            self.r1.configure(state=NORMAL)
            self.r2.configure(state=NORMAL)

    def callback(self, *kwargs):
        self.generate_pulse()
        self.pulse_selection()
        print(self.variable.get())
        self.cal_acf_psd(self.g)
        self.scenario.bw_factor = np.float(self.entry_bw_factor.get())
        self.scenario.fc = np.float(self.entry_center_frequency.get())
        self.scenario.fs = np.float(self.entry_sample_frequency.get())
        self.display_pulse_shape()

    def cal_acf_psd(self,input_data):
        '''
        for i in range(input_data.shape[1]):
            self.acf[:, i] = signal.correlate(input_data[:, i], input_data[:, i], mode="full")
            self.psd[:, i] = np.fft.fftshift(np.abs(np.fft.fft(input_data[:, i])))
        '''
        self.acf = np.abs(signal.correlate(input_data,input_data,mode="full"))
        self.psd = np.fft.fftshift(np.abs(np.fft.fft(input_data)))
        print(np.argmax(self.psd))

    def generate_pulse(self):
        self.gaussian = np.exp(-np.float(self.entry_bw_factor.get()) * (self.n - self.a_time) ** 2) * np.cos(2 * np.pi * np.float(self.entry_center_frequency.get()) * (self.n - self.a_time) + self.phi)
        self.ricker_pulse = 2 / ((np.sqrt(3 * self.sigma)) * np.pi ** 0.25) * (1 - np.float(self.entry_bw_factor.get()) * ((self.n - self.a_time) / self.sigma) ** 2) * np.exp(-np.float(self.entry_bw_factor.get()) * (self.n - self.a_time) ** 2 / (2 * np.float(self.sigma) ** 2))*np.cos(2 * np.pi * (self.n - self.a_time) * np.float(self.entry_center_frequency.get()) + self.phi)
        self.sinc_pulse = np.sinc(np.float(self.entry_bw_factor.get()) * (self.n - self.a_time) ** 2) * np.cos(2 * np.pi * (self.n - self.a_time) * np.float(self.entry_center_frequency.get()) + self.phi)
        self.n11 = np.arange(-self.NT, 0) * self.Ts
        self.n12 = np.arange(0, self.NT) * self.Ts
        self.s = 0.5

        self.gabor1 = np.exp(-(np.float(self.entry_bw_factor.get()) * self.n11 ** 2 + np.float(self.entry_bw_factor.get()) * self.n11 ** 2 * self.s)) * np.cos(2 * np.pi * (self.n11) * np.float(self.entry_center_frequency.get()) + self.phi)
        self.gabor2 = np.exp(-(np.float(self.entry_bw_factor.get()) * self.n12 ** 2 - np.float(self.entry_bw_factor.get()) * self.n12 ** 2 * self.s)) * np.cos(2 * np.pi * (self.n12) * np.float(self.entry_center_frequency.get()) + self.phi)

        self.gabor = np.zeros(2 * self.NT)
        self.gabor[0:self.NT] = self.gabor1
        self.gabor[self.NT:2 * self.NT] = self.gabor2

    def pulse_selection(self):
        if self.variable.get() == self.OPTIONS[2]:
            self.g = self.gaussian
            self.pulse_select = 2
            print('gaussian pulse')
        elif self.variable.get() == self.OPTIONS[1]:
            self.g = self.ricker_pulse
            self.pulse_select = 1
            print('ricker wavelet')
        elif self.variable.get() == self.OPTIONS[0]:
            self.g = self.sinc_pulse
            self.pulse_select = 0
            print('sinc pulse')
        elif self.variable.get() == self.OPTIONS[3]:
            self.g = self.gabor
            self.pulse_select = 3
            print('gabor pulse')

        self.cal_acf_psd(self.g)

    def display_pulse_shape(self):

        p1 = self.x_pos
        p2 = self.y_pos
        a1 = self.az_pos
        c1 = self.coe_pos

        point = np.array([p1, p2, 0]) * 1e-3

        self.a6.clear()
        self.a6.plot(self.g)
        self.a6.set_xlabel('Samples')
        self.a6.set_ylabel('Amplitude')
        self.a6.set_title('Original pulse')
        self.canvas6.draw()

        ascan1, ascan, trans = self.init_transducer(point, a1, c1,self.test_flag)
        self.a7.clear()
        self.a7.plot(ascan)
        self.a7.set_ylim(np.min(ascan) - 0.1, np.max(ascan) + 0.1)
        self.a7.set_xlabel('Samples')
        self.a7.set_ylabel('Amplitude')
        self.a7.set_title('Ascan')
        self.canvas7.draw()

        self.a10.clear()
        self.a10.plot(np.arange(4 * self.NT-1),self.acf/np.max(self.acf))
        self.a10.set_xlabel('Samples')
        self.a10.set_ylabel('Magnitude')
        self.a10.set_title('ACF')
        self.canvas10.draw()

        self.a11.clear()
        self.a11.plot(np.arange(-self.NT,self.NT)*(self.fs/(2*self.NT))*1e-6,self.psd)
        self.a11.set_xlabel('Frequency (in MHz)')
        self.a11.set_ylabel('Magnitude')
        self.a11.set_title('PSD')
        self.canvas11.draw()

    def plot(self):

        p1 = self.x_pos
        p2 = self.y_pos
        a1 = self.az_pos
        c1 = self.coe_pos

        #print(a1,c1)
        #point = np.array([p1, p2, 0]) * 1e-3

        point = np.array([p1, p2, 0]) * 1e-3
        self.pulse_selection()
        ascan1, ascan, trans = self.init_transducer(point, a1, c1,self.test_flag)

        # self.a.set_ylim(np.min(ascan) - 0.1, np.max(ascan) + 0.1)
        # self.line.set_ydata(ascan)
        # self.line.set_xdata(self.time_axis)
        # self.a.set_title('Ascan')
        # self.a.set_xlabel('time (us)')
        # self.a.set_ylabel('Amplitude')
        # self.canvas.draw()

        directions = trans.rays.direction
        sources = trans.rays.source
        directivity_pattern = trans.directivity.reshape(self.nv * self.nh, 1)
        directions = directions.reshape(directions.shape[0] * directions.shape[1], 3)
        directions = directions * directivity_pattern

        s1 = np.zeros(2)
        s1[0] = sources[0]
        s1[1] = 0 #sources[2]
        origin = np.repeat(s1[:, np.newaxis], self.nv * self.nh, axis=1)
        color_magnitude = directivity_pattern.reshape(self.nv * self.nh, 1)
        print(s1)
        self.quiver5.remove()
        self.quiver5 = self.a1.quiver(*origin, directions[:, 0], directions[:, 2], color_magnitude, scale=2)
        self.a1.axis([-20*1e-3, 20*1e-3, 0*1e-3, 40*1e-3])
        #self.a1.imshow(trans.image)
        self.a1.set_title('Transducer Beam Pattern')
        self.a1.set_xlabel('X axis')
        self.a1.set_ylabel('Z axis (height)')
        self.canvas2.draw()

        o_point2 = np.array([point[0], point[1], 25e-3])
        print(o_point2)
        #self.a3.clear()
        self.source_point.remove()
        self.source_point, = self.a3.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o', markersize=10, alpha=0.8,
                     label='Transducer')

        # z_point = o_point2[2] + 1e-3 * np.cos(self.coe_pos)

        self.quiver2.remove()
        self.quiver2 = self.a3.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]],[-o_point2[0] + point[0] + 15e-3 * np.sin(self.coe_pos * np.pi / 180) * np.cos(self.az_pos * np.pi / 180)],
                                 [-o_point2[1] + point[1] + 15e-3 * np.sin(self.coe_pos * np.pi / 180) * np.sin(self.az_pos * np.pi / 180)],
                                 [-15e-3*np.cos(self.coe_pos * np.pi / 180)], linewidths=(5,), edgecolor="red", label='Direction')

        self.a3.view_init(elev=0,azim=-90)
        # self.a3.set_xmargin(-0.4)
        # self.a3.set_ymargin(-0.45)
        # self.a3.set_zmargin(0.5)
        self.canvas3.draw()

        self.source_point1.remove()
        self.source_point1, = self.a4.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o',
                                         markersize=10, alpha=0.8,
                                         label='Transducer')
        # z_point = o_point2[2] + 1e-3 * np.cos(self.coe_pos)
        self.quiver3.remove()
        self.quiver3 = self.a4.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [
            -o_point2[0] + point[0] + 15e-3 * np.sin(self.coe_pos * np.pi / 180) * np.cos(self.az_pos * np.pi / 180)],
                                      [-o_point2[1] + point[1] + 15e-3 * np.sin(self.coe_pos * np.pi / 180) * np.sin(
                                          self.az_pos * np.pi / 180)],
                                      [-15e-3 * np.cos(self.coe_pos * np.pi / 180)], linewidths=(5,), edgecolor="red",
                                      label='Direction')
        self.a4.view_init(elev=90, azim=-90)
        # self.a4.set_xmargin(-0.4)
        # self.a4.set_ymargin(-0.45)
        # self.a4.set_zmargin(0.5)
        self.canvas4.draw()

        self.source_point2.remove()
        self.source_point2, = self.a5.plot([o_point2[0]], [o_point2[1]], [o_point2[2]], color='yellow', marker='o',
                                         markersize=10, alpha=0.8,
                                         label='Transducer')
        self.quiver4.remove()
        self.quiver4 = self.a5.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [
            -o_point2[0] + point[0] + 15e-3 * np.sin(self.coe_pos * np.pi / 180) * np.cos(self.az_pos * np.pi / 180)],
                                      [-o_point2[1] + point[1] + 15e-3 * np.sin(self.coe_pos * np.pi / 180) * np.sin(
                                          self.az_pos * np.pi / 180)],
                                      [-15e-3 * np.cos(self.coe_pos * np.pi / 180)], linewidths=(5,), edgecolor="red",
                                      label='Direction')
        self.a5.view_init(elev=3, azim=0)
        # self.a5.set_xmargin(-0.4)
        # self.a5.set_ymargin(-0.45)
        # self.a5.set_zmargin(0.5)
        self.canvas5.draw()

    def init_transducer(self, point, azi, coele,test_flag):
        if test_flag == 0:
            self.target_snr_db = np.float(self.target_snr.get())
            self.opening_test = np.int(self.opening_angle.get())
        else:
            if self.para_select == 1:
                self.target_snr_db = self.para_snr[self.para_select_index]
                print(self.target_snr_db)
            elif self.para_select == 3:
                self.opening_test = self.para_opening[self.para_select_index]
                print(self.opening_test)
        start = timer()
        transducer = Plane(point, self.distance, azi * np.pi / 180, coele * np.pi / 180, self.vres, self.hres, self.nv,
                           self.nh,  self.opening_test * np.pi/180)
        transducer.prepareImagingPlane()  # this ALWAYS has to be called right after creating a transducer!
        # print(point * 1e-3)
        #start1 = timer()
        #print('Time required prepareimagingPlane : '+str(start1-start))
        Ascan1 = transducer.insonify(self.scenario, self.pulse_select, 0)
        #start2 = timer()
        #print('Time required Insonify : ' + str(start2 - start))
        Ascan2 = np.abs(signal.hilbert(Ascan1))
        signal_power = np.mean(Ascan2)
        sig_avg_db = 10.0 * np.log10(signal_power)
        noise_avg_db = sig_avg_db - self.target_snr_db
        noise_avg_w = 10.0 ** (noise_avg_db / 10)
        noise_samples = np.random.normal(0, np.sqrt(noise_avg_w), len(Ascan2))
        Ascan1 = Ascan1 + noise_samples
        Ascan = np.abs(signal.hilbert(Ascan1))
        # Ascan1 = np.fft.fftshift(np.abs(np.fft.fft(Ascan1)))
        # print('signal_power: ' + str(signal_power), ' signal power wo abs: '+str(np.mean(Ascan1)))
        # self.a.set_ylim(np.min(Ascan1)-0.05,np.max(Ascan1)+0.05)
        # self.line.set_ydata(Ascan1)
        # self.line.set_xdata(self.time_axis)
        # self.a.clear()
        # self.a.plot(self.time_axis,Ascan1)
        # self.canvas.draw()
        # print(timer() - start)
        return Ascan, Ascan1, transducer

    def generate_data(self):
        print('Data generation')

    def up_x(self):
        step_x = int(self.Entry_step_x.get())
        self.x_pos += step_x
        if self.x_pos < 18:
            print(self.x_pos)
            self.x.config(text=str(self.x_pos))
        else:
            self.x_pos = 18
            self.x.config(text=str(self.x_pos))

    def down_x(self):
        step_x = int(self.Entry_step_x.get())
        self.x_pos -= step_x
        if self.x_pos > -28:
            self.x.config(text=str(self.x_pos))
        else:
            self.x_pos = -28
            self.x.config(text=str(self.x_pos))

    def up_y(self):
        step_y = int(self.Entry_step_y.get())
        self.y_pos += step_y
        if self.y_pos < 10:
            self.y.config(text=str(self.y_pos))
        else:
            self.y_pos = 10
            self.y.config(text=str(self.y_pos))

    def down_y(self):
        step_y = int(self.Entry_step_y.get())
        self.y_pos -= step_y
        if self.y_pos > -10:
            self.y.config(text=str(self.y_pos))
        else:
            self.y_pos = -10
            self.y.config(text=str(self.y_pos))

    def up_coe(self):
        step_coe = int(self.Entry_coe_step.get())
        self.coe_pos += step_coe
        if self.coe_pos <= 30:
            self.coe_back_up = self.coe_pos
            self.coelevation.config(text=str(self.coe_pos))
        else:
            self.coe_pos = 30
            self.coe_back_up = self.coe_pos
            self.coelevation.config(text=str(self.coe_pos))

    def down_coe(self):
        step_coe = int(self.Entry_coe_step.get())
        self.coe_pos -= step_coe
        if self.coe_pos >= -30:
            self.coe_back_up = self.coe_pos
            self.coelevation.config(text=str(self.coe_pos))
        else:
            self.coe_pos = -30
            self.coe_back_up = self.coe_pos
            self.coelevation.config(text=str(self.coe_pos))

    def up_az(self):
        step_az = int(self.Entry_az_step.get())
        if self.az_pos <= 30:
            self.az_pos += step_az
            self.az_back_up = self.az_pos
            self.azimuth.config(text=str(self.az_pos))
        else:
            self.az_pos = 30
            self.az_back_up = self.az_pos
            self.azimuth.config(text=str(self.az_pos))

    def down_az(self):
        step_az = int(self.Entry_az_step.get())
        if self.az_pos >= -30:
            self.az_pos -= step_az
            self.az_back_up = self.az_pos
            self.azimuth.config(text=str(self.az_pos))
        else:
            self.az_pos = -30
            self.az_back_up = self.az_pos
            self.azimuth.config(text=str(self.az_pos))

    def find_peaks(self, ascan, g_pulse):
        y = np.abs(signal.hilbert(signal.correlate(ascan, g_pulse, mode='same')))
        # y = np.abs(signal.hilbert(ascan))
        # y = np.abs(y)
        peak_index = np.argmax(y)
        # print(peak_index)
        try:
            window_l = int(self.window_length.get())
        except ValueError:
            window_l = 15
        end_index = peak_index + window_l
        start_index = peak_index - window_l
        # peak_energy = np.sum(np.abs(signal.hilbert(y[start_index:end_index])))
        peak_energy = np.sum(y[start_index:end_index])
        return peak_energy, ascan, peak_index

    def calcualate_tof(self, peaks, speed, ts):
        tof = peaks * ts
        d = tof * speed / 2
        return tof

    def compare_calibration(self):
        self.test_flag = 1
        self.algorithm_select = self.radiobutton_variable.get()
        print(self.algorithm_select)
        self.para_select = self.radiobutton_variable1.get()
        print(self.para_select)

        if self.para_select == 1:
            print('SNR selected')
            self.para_snr.append(np.float(self.SNR1.get()))
            self.para_snr.append(np.float(self.SNR2.get()))
            self.para_snr.append(np.float(self.SNR3.get()))
            self.para_snr.append(np.float(self.SNR4.get()))
            print(self.para_snr)
        elif self.para_select == 2:
            self.para_step.append(np.float(self.step_size1.get()))
            self.para_step.append(np.float(self.step_size2.get()))
            self.para_step.append(np.float(self.step_size3.get()))
            self.para_step.append(np.float(self.step_size4.get()))
            print(self.para_step)
        else:
            self.para_opening.append(np.float(self.Opening1.get()))
            self.para_opening.append(np.float(self.Opening2.get()))
            self.para_opening.append(np.float(self.Opening3.get()))
            self.para_opening.append(np.float(self.Opening4.get()))
            print(self.para_opening)
        print('perform compare calibration')
        self.noofiter = int(self.noofiters.get())
        self.step = float(self.step_size1.get())
        self.a12.set_xlim(0, self.noofiter)
        self.ph_init = self.az_pos
        self.th_init = self.coe_pos
        self.pulse_selection()
        self.azimuth_list.append(self.az_pos)
        self.elevation_list.append(self.coe_pos)

        for i in range(4):
            self.lines6[i].set_xdata(self.iters1)
            self.lines6[i].set_ydata(self.energies)
            # lobj1 = self.a.plot(self.azimuth_list, self.elevation_list, lw=3, color=colors[i], label=legends[i])[0]
            # self.lines3.append(lobj1)
            # self.a.legend()
            #self.lines6[i].set_xdata(self.time_axis)
            #self.lines6[i].set_ydata(np.zeros(self.NT))

        self.run_compare_calibrate()

    def run_compare_calibrate(self):
        if self.algorithm_select == 1:
            some_val, s_energy, test_gradient = self.spsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, self.x_pos, self.y_pos, 0)
            self.a12.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)

            self.lines6[self.para_select_index].set_ydata(self.energies)
            self.lines6[self.para_select_index].set_xdata(self.iters1)

            self.canvas12.draw()

            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
        elif self.algorithm_select == 2:
            some_val, s_energy, test_gradient = self.rdsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, 0.1, self.x_pos,self.y_pos, 0)
            self.a12.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)

            self.lines6[self.para_select_index].set_ydata(self.energies)
            self.lines6[self.para_select_index].set_xdata(self.iters1)

            self.canvas12.draw()

            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
        elif self.algorithm_select == 3:
            some_val, s_energy = self.gradient_ascent(1, self.x_pos, self.y_pos, 0)
            self.a12.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)

            self.lines6[self.para_select_index].set_ydata(self.energies)
            self.lines6[self.para_select_index].set_xdata(self.iters1)

            self.canvas12.draw()

            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
        elif self.algorithm_select == 4:
            some_val, s_energy = self.gradient_ascent_momentum(1, self.x_pos, self.y_pos, 0)
            self.a12.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)

            self.lines6[self.para_select_index].set_ydata(self.energies)
            self.lines6[self.para_select_index].set_xdata(self.iters1)

            self.canvas12.draw()

            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init

        if self.index <= self.noofiter:
            self.window.after(10, self.run_compare_calibrate)
        else:
            if self.para_select_index < 3:
                print(self.para_select_index)
                self.para_select_index += 1
                if self.para_select == 2:
                    self.step = float(self.para_step[self.para_select_index])
                print('Step size : '+str(self.step))
            else:
                self.para_select_index = 0
                self.algorithm_select = np.inf
            self.index = 0
            self.iters1.clear()
            self.energies.clear()
            self.energies.append(0)
            self.iters1.append(0)
            self.test_grads.clear()
            self.azimuth_list.clear()
            self.elevation_list.clear()
            self.ph_init = self.az_back_up
            self.th_init = self.coe_back_up
            self.az_pos = self.az_back_up
            self.coe_pos = self.coe_back_up
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.window.after(10, self.run_compare_calibrate)

    def gradient_ascent(self, delta, point_x, point_y, point_z):
        # energies = np.zeros(noofiteration)
        p3 = np.array([point_x, point_y, point_z]) * 1e-3
        print('Inside gradient ascent fine')

        # a1.plot(energies

        start = timer()
        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init, (self.th_init + delta),self.test_flag)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)
        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init, (self.th_init - delta),self.test_flag)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * delta)

        a_scan2, ascan_t, trans = self.init_transducer(p3, (self.ph_init + delta), self.th_init,self.test_flag)
        peak_energy2, output_array2, peak_i2 = self.find_peaks(ascan_t, self.g)
        a_scan3, ascan_t, trans = self.init_transducer(p3, (self.ph_init - delta), self.th_init,self.test_flag)
        peak_energy3, output_array3, peak_i3 = self.find_peaks(ascan_t, self.g)

        grad_ph = (peak_energy2 - peak_energy3) / (2 * delta)

        self.ph_init = self.ph_init + grad_ph * self.step
        self.th_init = self.th_init + grad_th * self.step

        grad_vec = np.array([grad_th, grad_ph])
        grad_vec = np.reshape(grad_vec, (2, 1))
        # inner_product_grad = grad_vec.transpose().dot(self.prev_grad)/(np.linalg.norm(grad_vec)*np.linalg.norm(self.prev_grad))
        grad_vec_norm = np.linalg.norm(grad_vec)
        self.prev_grad = grad_vec
        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init,self.test_flag)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)
        max_energy = peak_energy4
        end = timer()
        print(max_energy, self.ph_init, peak_energy2, peak_energy3)
        di1 = self.calcualate_tof(peak_i4, self.c, self.Ts)
        print(di1)
        self.index += 1
        '''
        if self.index == noofiteration:
            di1 = self.calcualate_tof(peak, self.c, self.Ts)
            x = point_x + di1 * np.sin(self.th_init * np.pi / 180) * np.cos(self.ph_init * np.pi / 180)
            y = point_y + di1 * np.sin(self.th_init * np.pi / 180) * np.sin(self.ph_init * np.pi / 180)
            z = di1 * np.cos(self.th_init * np.pi / 180)

            return max_energy, x, y, z, self.th_init, self.ph_init
        '''
        return self.index, max_energy

    def gradient_ascent_momentum(self, delta, point_x, point_y, point_z):
        p3 = np.array([point_x, point_y, point_z]) * 1e-3
        v = np.array([self.th_init, self.ph_init])
        v1 = np.reshape(v, (2, 1))

        print('Inside gradient ascent momentum')

        start = timer()

        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init + delta,self.test_flag)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)
        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init - delta,self.test_flag)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * delta)

        a_scan2, ascan_t, trans = self.init_transducer(p3, self.ph_init + delta, self.th_init,self.test_flag)
        peak_energy2, output_array2, peak_i2 = self.find_peaks(ascan_t, self.g)
        a_scan3, ascan_t, trans = self.init_transducer(p3, self.ph_init - delta, self.th_init,self.test_flag)
        peak_energy3, output_array3, peak_i3 = self.find_peaks(ascan_t, self.g)

        grad_ph = (peak_energy2 - peak_energy3) / (2 * delta)

        grad_vec = np.array([grad_th, grad_ph])
        grad_vec = np.reshape(grad_vec, (2, 1))

        grad_vec = self.step * grad_vec + self.momentum * self.prev_grad

        angle = grad_vec.transpose().dot(self.prev_grad)
        angle2 = np.asscalar(angle[0])
        #print(angle2)
        # coef_pg = (-1) * (-2 * angle2 - 1)
        # coef_cg = 1 - coef_pg

        v1 = v1 + grad_vec

        self.th_init = np.asscalar(v1[0])
        self.ph_init = np.asscalar(v1[1])

        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init,self.test_flag)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)
        max_energy = peak_energy4
        end2 = timer()
        self.prev_grad = grad_vec
        print(max_energy, self.th_init)

        self.index += 1

        return self.index, max_energy

    def spsa(self, a, c1, A, gamma, alpha, scale_step, scalefactor, point_x, point_y, point_z):

        p3 = np.array([point_x, point_y, point_z]) * 1e-3

        start2 = timer()
        step = a / ((self.index + A + 1) ** alpha) * scale_step
        ck = c1 / ((self.index + 1) ** gamma)
        deltar = np.random.binomial(1, 0.5)
        deltar = 2 * deltar - 1

        deltac = np.random.binomial(1, 0.5)
        deltac = 2 * deltac - 1
        t1 = timer()
        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac,self.test_flag)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        print(timer()-t1)

        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        # a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # peak_energy11, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init - ck * deltar, self.th_init - ck * deltac,self.test_flag)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init - ck * deltar, self.th_init - ck * deltac)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        # a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # peak_energy22, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * deltac * ck)

        # grad_th1 = (peak_energy11 - peak_energy22) / (2 * deltac * ck)

        grad_ph = (peak_energy - peak_energy1) / (2 * deltar * ck)

        # grad_ph1 = (peak_energy11 - peak_energy22) / (2 * deltar * ck)

        # val1 = np.where(self.epsilon > np.linalg.norm(grad_th1), self.epsilon, np.linalg.norm(grad_th1))
        # val2 = np.where(self.epsilon > np.linalg.norm(grad_th), self.epsilon, np.linalg.norm(grad_th))
        # val3 = np.where(self.epsilon > np.linalg.norm(grad_ph1), self.epsilon, np.linalg.norm(grad_ph1))
        # val4 = np.where(self.epsilon > np.linalg.norm(grad_ph), self.epsilon, np.linalg.norm(grad_ph))
        # print(val1, val2)

        # gradth2 = grad_th / val1 + grad_th1 / val2

        # gradph2 = grad_ph / val3 + grad_ph1 / val4

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

        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init,self.test_flag)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init,self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)

        max_energy = peak_energy4
        end2 = timer()
        print(max_energy)
        self.prev_grad = gradv_final
        self.index += 1

        return self.index, max_energy, inner_product_grad

    '''
    di1 = self.calcualate_tof(peak, c, Ts)
    x = point_x + di1 * np.sin(th_init * np.pi / 180) * np.cos(ph_init * np.pi / 180)
    y = point_y + di1 * np.sin(th_init * np.pi / 180) * np.sin(ph_init * np.pi / 180)
    z = di1 * np.cos(th_init * np.pi / 180)
    return max_energy, x, y, z, th_init, ph_init
    '''

    def rdsa(self, a, c1, A, gamma, alpha, scale_step, scalefactor, epsilon1, point_x, point_y, point_z):

        p3 = np.array([point_x, point_y, point_z]) * 1e-3
        print("Inside rdsa")
        print(self.ph_init,self.th_init)

        step = a / ((self.index + A + 1) ** alpha) * scale_step
        ck = c1 / ((self.index + 1) ** gamma)
        prob = (1 + epsilon1) / (2 + epsilon1)

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

        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac,self.test_flag)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        # a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # peak_energy11, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init - ck * deltar, self.th_init - ck * deltac,self.test_flag)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init - ck * deltar, self.th_init - ck * deltac)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        # a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init + ck * deltar, self.th_init + ck * deltac)
        # peak_energy22, output_array, peak_i = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * deltac * ck)

        # grad_th1 = (peak_energy11 - peak_energy22) / (2 * deltac*ck)

        grad_ph = (peak_energy - peak_energy1) / (2 * deltar * ck)

        # grad_ph1 = (peak_energy11 - peak_energy22) / (2 * deltar*ck)

        # val1 = np.where(self.epsilon > np.linalg.norm(grad_th1), self.epsilon, np.linalg.norm(grad_th1))
        # val2 = np.where(self.epsilon > np.linalg.norm(grad_th), self.epsilon, np.linalg.norm(grad_th))
        # val3 = np.where(self.epsilon > np.linalg.norm(grad_ph1), self.epsilon, np.linalg.norm(grad_ph1))
        # val4 = np.where(self.epsilon > np.linalg.norm(grad_ph), self.epsilon, np.linalg.norm(grad_ph))
        # print(val1, val2)

        # gradth2 = grad_th / val1 + grad_th1 / val2

        # gradph2 = grad_ph / val3 + grad_ph1 / val4

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

        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init,self.test_flag)
        # a_scan = self.collect_scan_finer_grid(p3,self.ph_init,self.th_init)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)

        max_energy = peak_energy4
        end2 = timer()
        print(max_energy,self.ph_init,self.th_init)
        self.prev_grad = gradv_final
        self.index += 1

        return self.index, max_energy, inner_product_grad

    def non_linear_conjugate_gradient_finer_grid(self, dr, dc, point_x, point_y, point_z):
        v = np.array([self.th_init, self.ph_init])
        v1 = np.reshape(v, (2, 1))

        # energies = np.zeros(iters)
        p3 = np.array([point_x, point_y, point_z]) * 1e-3
        print('Inside non_linear_conjugate_gradient_finer_grid')

        a_scan, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init + dr,self.test_flag)
        # a_scan = collect_scan_finer_grid(p3, ph_init, th_init + dr)
        peak_energy, output_array, peak_i = self.find_peaks(ascan_t, self.g)
        a_scan1, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init - dr,self.test_flag)
        peak_energy1, output_array1, peak_i1 = self.find_peaks(ascan_t, self.g)

        grad_th = (peak_energy - peak_energy1) / (2 * dr)

        a_scan2, ascan_t, trans = self.init_transducer(p3, self.ph_init + dc, self.th_init,self.test_flag)
        peak_energy2, output_array2, peak_i2 = self.find_peaks(ascan_t, self.g)
        a_scan3, ascan_t, trans = self.init_transducer(p3, self.ph_init - dc, self.th_init,self.test_flag)
        peak_energy3, output_array3, peak_i3 = self.find_peaks(ascan_t, self.g)

        grad_ph = (peak_energy2 - peak_energy3) / (2 * dc)

        grad = np.array([grad_th, grad_ph])
        # grad_norm = np.linalg.norm(grad)

        grad_v = np.reshape(grad, (2, 1))
        # angle = np.arccos(grad_v.transpose().dot(self.prev_grad) / (np.linalg.norm(grad_v) * np.linalg.norm(self.prev_grad))) * 180 / np.pi
        # angle2 = np.asscalar(angle[0])/180

        inner_product_grad = grad_v.transpose().dot(self.prev_grad)
        # beta = (grad_v.transpose().dot((grad_v - self.prev_grad))) / (self.prev_grad.transpose().dot(self.prev_grad)) # Polak–Ribière
        beta = (grad_v.transpose().dot(grad_v)) / (
            self.prev_grad.transpose().dot(self.prev_grad))  # Fletcher and reeves
        s = grad_v + beta * self.prev_s
        self.prev_grad = grad_v
        # v1 = v1 + self.step * s
        v1 = v1 + self.step / (self.index + 1) * s
        self.th_init = np.asscalar(v1[0])
        self.ph_init = np.asscalar(v1[1])
        self.prev_s = s
        print(self.th_init, self.ph_init)
        a_scan4, ascan_t, trans = self.init_transducer(p3, self.ph_init, self.th_init,self.test_flag)
        peak_energy4, output_array4, peak_i4 = self.find_peaks(ascan_t, self.g)
        max_energy = peak_energy4
        peak = peak_i4

        self.index += 1
        print(max_energy)

        return self.index, max_energy, inner_product_grad
    '''
    def start_demo(self):
        self.run = 0
        self.noofiter = int(self.noofiters.get())
        self.step = float(self.step_size.get())
        self.a9.set_xlim(0, self.noofiter)
        self.ph_init = 0
        self.th_init = 0
        #self.pulse_selection()
        # self.a.clear()
        # self.a.plot()
        # let's say I have 5 algorithms
        self.azimuth_list.append(self.ph_init)
        self.elevation_list.append(self.th_init)
        # if self.a8.get_legend():
        #     leg = self.a8.get_legend().set_visible(False)
        #     print("here")
        self.a8.plot(self.azimuth_list[0], self.elevation_list[0], 'r*')
        for i in range(5):
            self.lines4[i].set_xdata(self.iters1)
            self.lines4[i].set_ydata(self.energies)
            self.lines5[i].set_xdata(self.azimuth_list)
            self.lines5[i].set_ydata(self.elevation_list)
            #self.a8.legend()

        self.run_demo()
    
    def run_demo(self):
        if self.run == 0:
            some_val, s_energy, test_gradient = self.spsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, self.x_pos_demo, self.y_pos_demo, 0)

            self.a9.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            self.lines4[self.run].set_ydata(self.energies)
            self.lines4[self.run].set_xdata(self.iters1)

            self.canvas9.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines5[self.run].set_xdata(self.azimuth_list)
            self.lines5[self.run].set_ydata(self.elevation_list)
            self.canvas8.draw()
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            #self.az_pos = self.ph_init
            #self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 1:
            some_val, s_energy = self.gradient_ascent(1, self.x_pos_demo, self.y_pos_demo, 0)
            self.a9.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            self.lines4[self.run].set_ydata(self.energies)
            self.lines4[self.run].set_xdata(self.iters1)

            self.canvas9.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines5[self.run].set_xdata(self.azimuth_list)
            self.lines5[self.run].set_ydata(self.elevation_list)
            self.canvas8.draw()
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            #self.az_pos = self.ph_init
            #self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 2:
            some_val, s_energy, test_gradient = self.non_linear_conjugate_gradient_finer_grid(2, 2, self.x_pos_demo, self.y_pos_demo, 0)
            self.a9.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            self.lines4[self.run].set_ydata(self.energies)
            self.lines4[self.run].set_xdata(self.iters1)

            self.canvas9.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines5[self.run].set_xdata(self.azimuth_list)
            self.lines5[self.run].set_ydata(self.elevation_list)
            self.canvas8.draw()

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            #self.az_pos = self.ph_init
            #self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 3:
            some_val, s_energy = self.gradient_ascent_momentum(1, self.x_pos_demo, self.y_pos_demo, 0)
            self.a9.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            self.lines4[self.run].set_ydata(self.energies)
            self.lines4[self.run].set_xdata(self.iters1)

            self.canvas9.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines5[self.run].set_xdata(self.azimuth_list)
            self.lines5[self.run].set_ydata(self.elevation_list)
            self.canvas8.draw()

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            #self.az_pos = self.ph_init
            #self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 4:
            some_val, s_energy, test_gradient = self.rdsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, 0.1, self.x_pos_demo, self.y_pos_demo, 0)
            self.a9.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            self.lines4[self.run].set_ydata(self.energies)
            self.lines4[self.run].set_xdata(self.iters1)

            self.canvas9.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines5[self.run].set_xdata(self.azimuth_list)
            self.lines5[self.run].set_ydata(self.elevation_list)
            self.canvas8.draw()

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            #self.az_pos = self.ph_init
            #self.coe_pos = self.th_init
            #self.plot()
        else:
            self.run = np.inf

        if self.index <= self.noofiter:
            self.window.after(100, self.run_demo)
        else:
            print(self.run)
            self.index = 0
            self.iters1.clear()
            self.energies.clear()
            self.energies.append(0)
            self.iters1.append(0)
            self.test_grads.clear()
            self.azimuth_list.clear()
            self.elevation_list.clear()
            self.prev_grad = np.ones((2, 1)) * 1e-5
            self.prev_s = np.zeros((2, 1))
            self.ph_init = 0#self.az_back_up
            self.th_init = 0#self.coe_back_up
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            #self.az_pos = self.az_back_up
            #self.coe_pos = self.coe_back_up
            self.run += 1

            self.window.after(500, self.run_demo)
            # return
    '''
    def calibrate(self):
        self.run = 0
        self.noofiter = int(self.noofiters.get())
        self.step = float(self.step_size.get())
        self.a2.set_xlim(0, self.noofiter)
        self.ph_init = self.az_pos
        self.th_init = self.coe_pos
        self.pulse_selection()
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        fptr = open("C:/Users/user/Documents/ConfigurationFiles/Config_"+dt_string+".txt","w")
        fptr.write("Configuration Parameters\n")
        fptr.write("X : " + str(self.x_pos) + "  Y : "+str(self.y_pos)+" \n")
        fptr.write("Azimuth : " + str(self.ph_init) + "  Coelevation : " + str( self.th_init) + " \n")
        fptr.write("SNR : " + str(np.int(self.target_snr.get())) + "  Opening Angle : " + str(np.int(self.opening_angle.get()))+"  Iterations : " + str(self.noofiter) + " \n")
        fptr.write("Pulse Shape : " + self.OPTIONS[self.pulse_select] + " \n")
        fptr.close()
        # self.a.clear()
        # self.a.plot()
        # let's say I have 5 algorithms
        self.azimuth_list.append(self.az_pos)
        self.elevation_list.append(self.coe_pos)
        self.a.plot(self.azimuth_list[0], self.elevation_list[0], 'r*')
        for i in range(5):
            self.lines[i].set_xdata(self.iters1)
            self.lines[i].set_ydata(self.energies)
            # lobj1 = self.a.plot(self.azimuth_list, self.elevation_list, lw=3, color=colors[i], label=legends[i])[0]
            # self.lines3.append(lobj1)
            # self.a.legend()
            self.lines3[i].set_xdata(self.time_axis)
            self.lines3[i].set_ydata(np.zeros(self.NT))

        #thethread = Thread(target=self.spsa_thread(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, self.x_pos, self.y_pos, 0), name='firstthread')
        #thethread.daemon = True
        #thethread.start()
        self.run_algo()

    def run_algo(self):
        if self.run == 0:
            some_val, s_energy, test_gradient = self.spsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, self.x_pos, self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy, 2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init, 2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init, 2)))
            self.test_grads.append(test_gradient)

            self.a2.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            # self.a2.plot(self.iters1,self.energies,label='Blue')

            self.lines[self.run].set_ydata(self.energies)
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)

            self.canvas1.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines3[self.run].set_xdata(self.azimuth_list)
            self.lines3[self.run].set_ydata(self.elevation_list)
            self.canvas.draw()
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 1:
            some_val, s_energy = self.gradient_ascent(1, self.x_pos, self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy, 2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init, 2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init, 2)))
            # self.test_grads.append(test_gradient)

            self.a2.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            # self.a2.plot(self.iters1, self.energies,label='Red')
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_ydata(self.energies)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines3[self.run].set_xdata(self.azimuth_list)
            self.lines3[self.run].set_ydata(self.elevation_list)
            self.canvas.draw()
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 2:
            some_val, s_energy, test_gradient = self.non_linear_conjugate_gradient_finer_grid(2, 2, self.x_pos,
                                                                                              self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy, 2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init, 2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init, 2)))
            self.test_grads.append(test_gradient)

            self.a2.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            self.lines[self.run].set_ydata(self.energies)
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines3[self.run].set_xdata(self.azimuth_list)
            self.lines3[self.run].set_ydata(self.elevation_list)
            self.canvas.draw()

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 3:
            some_val, s_energy = self.gradient_ascent_momentum(1, self.x_pos, self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy, 2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init, 2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init, 2)))
            # self.test_grads.append(test_gradient)

            self.a2.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            self.lines[self.run].set_ydata(self.energies)
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines3[self.run].set_xdata(self.azimuth_list)
            self.lines3[self.run].set_ydata(self.elevation_list)
            self.canvas.draw()

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
            #self.plot()
        elif self.run == 4:
            some_val, s_energy, test_gradient = self.rdsa(0.4, 0.40, 0.2, 0.101, 0.5, 1, 1, 0.1, self.x_pos,
                                                          self.y_pos, 0)
            cells[(self.run + 1, 2)].config(text=str(np.round(s_energy, 2)))
            cells[(self.run + 1, 0)].config(text=str(np.round(self.ph_init, 2)))
            cells[(self.run + 1, 1)].config(text=str(np.round(self.th_init, 2)))
            self.test_grads.append(test_gradient)

            self.a2.set_ylim(np.min(self.energies) - 2, np.max(self.energies) + 2)
            # self.a2.set_ylim(np.min(self.test_grads) - 10, np.max(self.test_grads) + 10)
            # self.a2.plot(self.iters1,self.energies,label='Blue')
            self.lines[self.run].set_ydata(self.energies)
            # self.lines[self.run].set_ydata(self.test_grads)
            self.lines[self.run].set_xdata(self.iters1)
            self.canvas1.draw()
            self.energies.append(s_energy)
            self.iters1.append(some_val)

            self.lines3[self.run].set_xdata(self.azimuth_list)
            self.lines3[self.run].set_ydata(self.elevation_list)
            self.canvas.draw()

            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.az_pos = self.ph_init
            self.coe_pos = self.th_init
            #self.plot()

            '''

            o_point2 = np.array([self.x_pos * 1e-3, self.y_pos * 1e-3, 40e-3])

            self.quiver2.remove()
            self.quiver2 = self.a3.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [
                -o_point2[0] + self.x_pos*1e-3 + 8e-3 * np.sin(self.coe_pos * np.pi / 180) * np.cos(
                    self.az_pos * np.pi / 180)],
                                          [-o_point2[1] + self.y_pos*1e-3 + 8e-3 * np.sin(self.coe_pos * np.pi / 180) * np.sin(
                                              self.az_pos * np.pi / 180)],
                                          [-8e-3 * np.cos(self.coe_pos * np.pi / 180)], linewidths=(5,),
                                          edgecolor="red", label='Direction')

            self.a3.view_init(elev=0, azim=-90)

            self.canvas3.draw()

            self.quiver3.remove()
            self.quiver3 = self.a4.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [
                -o_point2[0] + self.x_pos*1e-3 + 8e-3 * np.sin(self.coe_pos * np.pi / 180) * np.cos(
                    self.az_pos * np.pi / 180)],
                                          [-o_point2[1] + self.y_pos*1e-3 + 8e-3 * np.sin(self.coe_pos * np.pi / 180) * np.sin(
                                              self.az_pos * np.pi / 180)],
                                          [-8e-3 * np.cos(self.coe_pos * np.pi / 180)], linewidths=(5,),
                                          edgecolor="red",
                                          label='Direction')
            self.a4.view_init(elev=90, azim=-90)
            self.canvas4.draw()

            self.quiver4.remove()
            self.quiver4 = self.a5.quiver([o_point2[0]], [o_point2[1]], [o_point2[2]], [
                -o_point2[0] + self.x_pos*1e-3 + 8e-3 * np.sin(self.coe_pos * np.pi / 180) * np.cos(
                    self.az_pos * np.pi / 180)],
                                          [-o_point2[1] + self.y_pos*1e-3 + 8e-3 * np.sin(self.coe_pos * np.pi / 180) * np.sin(
                                              self.az_pos * np.pi / 180)],
                                          [-8e-3 * np.cos(self.coe_pos * np.pi / 180)], linewidths=(5,),
                                          edgecolor="red",
                                          label='Direction')
            self.a5.view_init(elev=3, azim=0)
            self.canvas5.draw()
            '''

        else:
            self.run = np.inf
        # self.calib_az.config(text=str(np.round(self.ph_init,3)))
        # self.calib_coe.config(text=str(np.round(self.th_init,3)))

        if self.index <= self.noofiter:
            self.window.after(10, self.run_algo)
        else:
            self.index = 0
            # p = np.array([self.x_pos, self.y_pos, 0])*1e-3
            # a_scan1, ascan, trans = self.init_transducer(p, self.ph_init, self.th_init)
            # self.a.set_ylim(np.min(a_scan1) - 0.1, np.max(a_scan1) + 0.1)
            # self.lines3[self.run].set_ydata(a_scan1)
            # self.lines3[self.run].set_xdata(self.time_axis)
            # self.canvas.draw()

            self.iters1.clear()
            self.energies.clear()
            self.energies.append(0)
            self.iters1.append(0)
            self.test_grads.clear()
            self.azimuth_list.clear()
            self.elevation_list.clear()
            self.prev_grad = np.ones((2, 1)) * 1e-5
            self.prev_s = np.zeros((2, 1))
            self.ph_init = self.az_back_up
            self.th_init = self.coe_back_up
            self.az_pos = self.az_back_up
            self.coe_pos = self.coe_back_up
            self.azimuth_list.append(self.ph_init)
            self.elevation_list.append(self.th_init)
            self.run += 1
            self.window.after(10, self.run_algo)
            # return



