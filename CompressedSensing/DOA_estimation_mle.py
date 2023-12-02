import numpy as np

N = 101
d = 0.5
ant_ele = np.arange(0,N)

sig = 10*np.pi/180
amp = 1
meas = np.exp(1j*2*np.pi*d*ant_ele*np.cos(sig))

iters = 10000
phi = 11
delta = 2
alpha = 0.001

for i in range(iters):
    steer_vec = np.exp(-1j*2*np.pi*d*ant_ele*np.sin((phi + delta) * np.pi/180))
    a = steer_vec.T.dot(meas)
    b = steer_vec.T.dot(steer_vec)
    c1 = np.abs(a)**2 / np.abs(b)

    steer_vec1 = np.exp(-1j * 2 * np.pi * d * ant_ele * np.sin((phi - delta) * np.pi / 180))
    a1 = steer_vec1.T.dot(meas)
    b1 = steer_vec1.T.dot(steer_vec1)
    c2 = np.abs(a1)**2 / np.abs(b1)

    grad = (c1 - c2)/(2*delta)

    phi = phi + alpha*grad

    steer_vec = np.exp(-1j * 2 * np.pi * d * ant_ele * np.cos(phi * np.pi / 180))
    a = steer_vec.T.dot(meas)
    b = steer_vec.T.dot(steer_vec)
    c3 = np.abs(a)**2 / np.abs(b)
    print('Phi : ' + str(phi) + ' Cost Function : ' + str(c3) + 'grad : '+ str(grad))