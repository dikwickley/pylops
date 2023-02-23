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
            raise NotImplementedError("DTCWT is not implmented for 2D")
        elif self.ndim == 3:
            raise NotImplementedError("DTCWT is not implmented for 3D")
        else:
            raise ValueError("DTCWT only supports 1D, 2D and 3D")
        
        pyr = self._transform.forward(np.ones(self.dims), nlevels=self.nlevels, include_scale=True)
        self.coeff_array_size = 0
        self.lowpass_size = len(pyr.lowpass)
        self.slices = []
        for _h in pyr.highpasses:
            self.slices.append(len(_h))
            self.coeff_array_size += len(_h)
        self.coeff_array_size += self.lowpass_size

        super().__init__(dtype=np.dtype(dtype), dims=self.dims, dimsd=(self.coeff_array_size,), name=name)

    def _coeff_to_array(self, pyr: dtcwt.Pyramid) -> NDArray:
        coeffs = pyr.highpasses
        flat_coeffs = np.concatenate([c.ravel() for c in coeffs])
        flat_coeffs = np.concatenate((flat_coeffs, pyr.lowpass.ravel()))        
        return flat_coeffs
    
    def _array_to_coeff(self, X: NDArray) -> dtcwt.Pyramid:
        lowpass = np.array([x.real for x in X[-1*self.lowpass_size:]]).reshape((-1, 1))
        _ptr = 0
        highpasses = ()
        for _sl in self.slices:
            _h = X[_ptr:_ptr+_sl]
            _ptr += _sl 
            _h = _h.reshape((-1, 1))
            highpasses += (_h, )
        return dtcwt.Pyramid(lowpass,highpasses)
    

    def _matvec(self, x: NDArray):
        return self._coeff_to_array(self._transform.forward(x, nlevels=self.nlevels, include_scale=False))    

    def _rmatvec(self, X: NDArray) -> NDArray:
        return self._transform.inverse(self._array_to_coeff(X))

