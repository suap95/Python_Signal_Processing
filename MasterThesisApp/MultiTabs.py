import tkinter as tk
from tkinter import ttk
from positioner1 import positioner

window = tk.Tk()
window.title("Master Thesis")
window.geometry("1800x1000")
'''
ttk.Label(tab1,
          text="Welcome to \
          GeeksForGeeks").grid(column=0,
                               row=0,
                               padx=30,
                               pady=30)
ttk.Label(tab2,
          text="Lets dive into the\
          world of computers").grid(column=0,
                                    row=0,
                                    padx=30,
                                    pady=30)
'''
start = positioner(window)
window.mainloop()