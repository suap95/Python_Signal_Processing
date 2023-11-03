import numpy as np
import scipy.fft as sp
import matplotlib.pyplot as plt

N = 128
M = np.int(N / 2)
A = sp.dct(np.eye(N))
A = A / np.linalg.norm(A,axis=1,keepdims=True)
sparse_vec = np.zeros((N,1))

sparse_vec[15] = 1
sparse_vec[20] = 1
sparse_vec[36] = 1

# sparse_vec[55] = 1
# sparse_vec[88] = 1
# sparse_vec[63] = 1

s = A.dot(sparse_vec)

plt.figure(3)
plt.plot(s,label='Original sampled Signal')

phi = np.random.randn(M,N)
phi = phi / np.linalg.norm(phi,axis=1,keepdims = True)
y_cs = phi.dot(s)

plt.figure(2)
plt.plot(y_cs,label='Sub sampled Signal')

r = y_cs
x = np.zeros((N,1))

for i in range(500):
    B = phi.dot(A)
    c = np.conj(B.T).dot(r)
    ind = np.argmax(np.abs(c))
    x[ind] = x[ind] + c[ind]
    r = r - np.reshape(c[ind].item()*B[:,ind],[M,1])


plt.figure(1)
plt.plot(sparse_vec,label='original')
plt.plot(x,label='reconstructed')
plt.legend()

plt.figure(4)
plt.plot(s,label='original')
plt.plot(A.dot(x),label='reconstructed')
plt.legend()

