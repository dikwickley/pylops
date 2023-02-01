import numpy as np
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from pylops.transforms.discreetcosine import DiscreetCosine
import matplotlib.pyplot as plt

im = np.load("testdata/python.npy")[::5, ::5, 0]


nt, nx = im.shape

#orignal image
# f = plt.figure()
# plt.imshow(im,cmap='viridis')
# plt.title("Orignal Image")

def dct2(a):
    d = DiscreetCosine(dims=a.shape, axes=1)
    return d*a
def idct2(a):
    d = DiscreetCosine(dims=a.shape, axes=1)
    return d.H*a

imsize = im.shape
dct = np.zeros(imsize)


# Do 8x8 DCT on image (in-place)
for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )

# # Extract a block from image
# plt.figure()
# plt.imshow(dct,cmap='viridis',vmax = np.max(dct)*0.01,vmin = 0)
# plt.title( "8x8 DCTs of the image")

# Threshold
thresh = 0.012
dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))


# plt.figure()
# plt.imshow(dct_thresh,cmap='viridis',vmax = np.max(dct)*0.01,vmin = 0)
# plt.title( "Thresholded 8x8 DCTs of the image")

percent_nonzeros = np.sum( dct_thresh != 0.0 ) / (imsize[0]*imsize[1]*1.0)


im_dct = np.zeros(imsize)

for i in r_[:imsize[0]:8]:
    for j in r_[:imsize[1]:8]:
        im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )


# plt.figure()
# plt.imshow( np.hstack( (im, im_dct) ) ,cmap='viridis')
# plt.title("Comparison between original and DCT compressed images" )

fig, axs = plt.subplots(2,2, figsize=(12, 4))
axs[0,0].imshow(im, cmap="viridis", vmin=0, vmax=250)
axs[0,0].axis("tight")
axs[0,0].set_title("Original")
axs[0,1].imshow(dct,cmap='viridis',vmax = np.max(dct)*0.01,vmin = 0)
axs[0,1].axis("tight")
axs[0,1].set_title("8x8 DCTs of the image")
axs[1,0].imshow(dct_thresh,cmap='viridis',vmax = np.max(dct)*0.01,vmin = 0)
axs[1,0].axis("tight")
axs[1,0].set_title( "Thresholded 8x8 DCTs")
axs[1,1].imshow( im_dct ,cmap='viridis')
axs[1,1].axis("tight")
axs[1,1].set_title("DCT compressed images")


plt.plot()



plt.show()
