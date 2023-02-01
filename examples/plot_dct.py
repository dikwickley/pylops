r"""
Discreet Cosine Transform

This example shows how to use the :py:class:`pylops.transforms.discreetcosine.DiscreetCosine` operator
to perform *Discreet Cosine Transform* on a given multi dimensional array.

"""

import matplotlib.pyplot as plt
import numpy as np
import pylops

###################################################################################
# A simple example:
# nt, nx = 2, 3
# x = np.outer(np.arange(nt) + 1, np.arange(nx) + 1)
# dct = pylops.transforms.DiscreetCosine(dims=(nt,nx))
# y = dct * x
# print(f"transformed x {y}")

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