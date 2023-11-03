import numpy as np
from positioner import positioner
from tkinter import *

window = Tk()
window.geometry("1800x1000")
p = np.array([0,0,0])*1e-3
start = positioner(window)
window.mainloop()


