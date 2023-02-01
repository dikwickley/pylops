# import pylops.transforms.discreetcosine as ds
# import numpy as np
# from scipy import fft
# import matplotlib.pyplot as plt

# nt, nx = 10, 10
# x = np.outer(np.arange(nt), np.arange(nx))


# dct = ds.DiscreetCosine(dims=(nt,nx))
# y = dct * x

# print(y)
# plt.matshow(x, cmap="rainbow")
# plt.matshow(y, cmap="rainbow")

# cycles = 2 # how many sine cycles
# resolution = 25 # how many datapoints to generate

# length = np.pi * 2 * cycles
# r = np.sin(np.arange(0, length, length / resolution))
# dct = ds.DiscreetCosine(dims=r.shape)
# y = dct * r

# plt.plot(r, color="orange", label="input data")
# plt.plot(y, color="blue", label="dct of input")
# plt.legend(loc="upper left")


# plt.show()


# Example with scipy.fftpack:
import numpy as np
import pylops.transforms.discreetcosine as ds
import matplotlib.pyplot as plt

im = np.load("testdata/python.npy")[::5, ::5, 0]

nt, nx = im.shape

dct = ds.DiscreetCosine(dims=(nt,nx))
y = dct * im

dct2 = ds.DiscreetCosine(dims=(nt, nx), axes=1)
y2 = dct2 * y



yi = dct.H*y

fig, axs = plt.subplots(1, 4, figsize=(12, 4))
axs[0].imshow(im, cmap="viridis", vmin=0, vmax=250)
axs[0].axis("tight")
axs[0].set_title("Original")
axs[1].imshow(y, cmap="viridis", vmin=0, vmax=250)
axs[1].axis("tight")
axs[1].set_title("DCT")
axs[2].imshow(yi, cmap="viridis", vmin=0, vmax=250)
axs[2].axis("tight")
axs[2].set_title("Inverse DCT")
axs[3].imshow(y2, cmap="viridis", vmin=0, vmax=250)
axs[3].axis("tight")
axs[3].set_title("DCT along 1 axis")


plt.plot()

plt.show()
