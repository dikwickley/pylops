__all__ = ["DTCWT"]

from typing import List, Optional, Union

import numpy as np
import dtcwt
import math

from pylops import LinearOperator
from pylops.utils._internal import _value_or_sized_to_tuple
from pylops.utils.decorators import reshaped
from pylops.utils.typing import DTypeLike, InputDimsLike, NDArray


class DTCWT(LinearOperator):
    r"""
    Perform Dual-Tree Complex Wavelet Transform on a given array.
    """

    def __init__(
        self,
        dims: Union[int, InputDimsLike],
        biort: str = "near_sym_a",
        qshift: str = "qshift_a",
        nlevels: int = 3, 
        include_scale: bool = False,
        ext_mode: int = 4, #only for 3d. 4 or 8
        dtype: DTypeLike = "float64",
        name: str = "C",
    ) -> None:
        self.dims = _value_or_sized_to_tuple(dims)
        self.ndim = len(self.dims)
        self.nlevels = nlevels
        self.include_scale = include_scale

        if self.ndim == 1:
            self._transform = dtcwt.Transform1d(biort=biort, qshift=qshift)
        elif self.ndim == 2:
            self._transform = dtcwt.Transform2d(biort=biort, qshift=qshift)
        elif self.ndim == 3:
            self._transform = dtcwt.Transform3d(biort=biort, qshift=qshift, ext_mode=ext_mode)
        else:
            raise ValueError("DTCWT only supports 1D, 2D and 3D")
        
        pyr = self._transform.forward(np.ones(self.dims), nlevels=self.nlevels, include_scale=True)
        self.scales = pyr.scales

        super().__init__(dtype=np.dtype(dtype), dims=self.dims, dimsd=self.dims, name=name)

    
    def _matvec(self, x: NDArray):
        pyr = self._transform.forward(x, nlevels=self.nlevels, include_scale=False)
        coeffs = pyr.highpasses
        return np.concatenate([c.ravel() for c in coeffs])
    

    def _rmatvec(self, X: NDArray) -> NDArray:
        return []
        # nlevels = len(self.scales) - 1
        # pyr = dtcwt.Pyramid(self.scales[0], [], scales=self.scales[1:])
        # for i in range(nlevels):
        #     start = i * 6
        #     end = (i + 1) * 6
        #     coeffs = X[start:end]
        #     print(f"Level {i} coeffs: {coeffs.shape}, expected shape: {self.scales[i+1].shape}")
        #     pyr.highpasses += (X[start:end].reshape(self.scales[i+1].shape))
        
        # return self._transform.inverse(pyr)
