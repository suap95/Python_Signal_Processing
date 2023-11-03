import numpy as np
import matplotlib.pyplot as plt

mean = 10
sigma = 0.5
num_measr = 1000

measurement_data = np.random.normal(mean,sigma,num_measr)

plt.figure(1)
plt.hist(measurement_data)
plt.xlim([0,10])

mean_est = 5
sig_est = 0.5
delta = 0.05
alpha = 0.001
iters = 1000
f_cost = np.arange(iters)

for i in range(iters):
    f_1 = np.linalg.norm((measurement_data - (mean_est + delta)))
    f_2 = np.linalg.norm((measurement_data - (mean_est - delta)))

    der1 = (f_1 - f_2)/(2 * delta)

    mean_est = mean_est - der1*alpha

    f_cost[i] = np.linalg.norm((measurement_data - mean_est))

    print('Mean : '+ str(mean_est))

sigma = np.sqrt((1/num_measr)*np.sum((measurement_data-mean_est)**2))
plt.figure(2)
plt.plot(f_cost)

t = np.linspace(-10,10,1000)
m = 1
s = 0.5

gaus_dist = np.exp(-0.5*(t-m)**2/(s)**2)

plt.figure(3)
plt.plot(t,gaus_dist)