from pylops.signalprocessing import DTCWT
import numpy as np
import matplotlib.pyplot as plt


x = np.outer(np.arange(10), np.arange(10))
x = np.arange(10) + 1

DOp = DTCWT(dims=x.shape, nlevels=3) 

y = DOp @ x




i = DOp.H @ y

print("final")
print(x)
print(y)
print(i)

# from matplotlib.pylab import *
import dtcwt

# # Generate a 300x2 array of a random walk
# vecs = np.cumsum(np.random.rand(300,1) - 0.5, 0)

# # Show input
# figure()
# plot(vecs)
# title('Input')

# # 1D transform, 5 levels
# transform = dtcwt.Transform1d()
# vecs_t = transform.forward(x, nlevels=5)

# # # Show level 2 highpass coefficient magnitudes
# # figure()
# plt.plot(np.abs(vecs_t.highpasses[0]), color="purple")
# plt.plot(np.abs(vecs_t.highpasses[1]), color="blue")
# plt.plot(np.abs(vecs_t.highpasses[2]), color="green")

# title('Level 2 wavelet coefficient magnitudes')
# plt.show()
# # Show last level lowpass image
# figure()
# plot(vecs_t.lowpass)
# title('Lowpass signals')

# # Inverse
# vecs_recon = transform.inverse(vecs_t)

# # Show output
# figure()
# plot(vecs_recon)
# title('Output')

# # Show error
# figure()
# plot(vecs_recon - vecs)
# title('Reconstruction error')

# print('Maximum reconstruction error: {0}'.format(np.max(np.abs(vecs - vecs_recon))))

# show()

# import numpy as np

# n = 10
# x1 = np.arange(n) + 1

# print(x1.ndim)

# x2 = np.outer(np.arange(n) + 1, np.arange(n) + 1)

# print(x2.ndim)

# x3 = np.zeros((2,3,2))

# print(x3.ndim)