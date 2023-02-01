import pylops.transforms.discreetcosine as ds
import numpy as np
from scipy import fft, fftpack

# nt, nx = 2, 3
# x = np.outer(np.arange(nt) + 1, np.arange(nx) + 1)

# dct = ds.DiscreetCosine(dims=(nt,nx))
# y2 = dct * x
# print(x)
# print(y2)
# print(dct.H*y2)

# x = np.arange(10)

# dct = ds.DiscreetCosine(dims=(10,))
# y = dct*x
# print(x)
# print(y)
# print(dct.H*y)

# n = 4

# x = np.arange(n)
# y = fftpack.dct(x)
# print(x)
# print(y)
# print(fftpack.idct(y))

# import numpy as np
# from scipy.fft import dctn, idctn


# y2 = idctn(dctn(y, norm="ortho"), norm="ortho")
# print(np.allclose(y, y2))


from pylops.utils import dottest
n = 11
rng = np.random.default_rng()
y = rng.standard_normal((n,))
dct = ds.DiscreetCosine(dims=y.shape)
print(y)
print(dct.H*(dct*y))
# print(dct)
# _ = dottest(dct, 121, 121, rtol=1e-6, complexflag=0, verb=True)