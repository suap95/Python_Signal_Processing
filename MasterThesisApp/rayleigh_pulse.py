import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import scipy.signal as ssig

t = np.linspace(-1,1,1000)
t1 = np.linspace(-1,1,1000)
alpha = 0.1
p = (4*np.pi*t/alpha**2)*np.exp(-2*np.pi*t**2/alpha**2)
y = np.abs(ssig.hilbert(ssig.correlate(p,p,mode='same')))
data = np.sinc(t1)
pi = np.pi
data1 = ne.evaluate('sin(t1)')
data2 = np.pi*t1
data1 = data1/data2

plt.figure(3)
plt.plot(data1)

plt.figure(4)
plt.plot(data)

plt.figure(1)
plt.plot(p)

plt.figure(2)
plt.plot(y)