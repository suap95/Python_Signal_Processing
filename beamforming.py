import numpy as np
import matplotlib.pyplot as plt

theta = 40*np.pi/180
d = 0.5
N = 21
antenna_ele = d*np.arange(0,N-1,1)
signal = np.cos(2*np.pi*antenna_ele*np.cos(theta)) #+ np.cos(2*np.pi*d*antenna_ele*np.cos(60*np.pi/180))

theta_grid = np.linspace(-np.pi/2,np.pi/2,10000)
array_st_matrix = np.zeros((np.int((N-1)),theta_grid.shape[0]),dtype=complex)

for i in range(1,theta_grid.shape[0]):
    array_st_matrix[:,i] = np.exp(1j*2*np.pi*antenna_ele*np.cos(theta_grid[i]))

output = np.abs(array_st_matrix.T.dot(signal))

plt.figure(1)
plt.plot(theta_grid*180/np.pi,output)
plt.title('Theta = 30 degrees')
plt.xlabel('Theta')
plt.ylabel('Power')
