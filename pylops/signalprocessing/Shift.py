import warnings

import numpy as np

from pylops.basicoperators import Diagonal
from pylops.signalprocessing import FFT


def Shift(
    dims,
    shift,
    axis=-1,
    dir=None,
    nfft=None,
    sampling=1.0,
    real=False,
    engine="numpy",
    dtype="complex128",
    **kwargs_fftw
):
    r"""Shift operator

    Apply fractional shift in the frequency domain along a specific direction
    ``dir`` of a multi-dimensional array of size ``dim``.

    Parameters
    ----------
    dims : :obj:`tuple`
        Number of samples for each dimension
    shift : :obj:`float`
        Fractional shift to apply in the same unit as ``sampling``.
    axis : :obj:`int`, optional
        .. versionadded:: 2.0.0

        Axis along which shift is applied
    dir : :obj:`int`, optional

        .. deprecated:: 2.0.0
            Use ``axis`` instead. Note that the default for ``axis`` is -1
            instead of 0 which was the default for ``dir``.
    nfft : :obj:`int`, optional
        Number of samples in Fourier Transform (same as input if ``nfft=None``)
    sampling : :obj:`float`, optional
        Sampling step :math:`\Delta t`.
    real : :obj:`bool`, optional
        Model to which fft is applied has real numbers (``True``) or not
        (``False``). Used to enforce that the output of adjoint of a real
        model is real.
    engine : :obj:`str`, optional
        Engine used for fft computation (``numpy``, ``scipy``, or ``fftw``). Choose
        ``numpy`` when working with CuPy arrays.
    dtype : :obj:`str`, optional
        Type of elements in input array.
    **kwargs_fftw
            Arbitrary keyword arguments
            for :py:class:`pyfftw.FTTW`

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved explicitly
        (``True``) or not (``False``)

    Raises
    ------
    ValueError
        If ``dims`` is provided and ``axis`` is bigger than ``len(dims)``
    NotImplementedError
        If ``engine`` is neither ``numpy``, ``scipy``, nor ``fftw``

    Notes
    -----
    The Shift operator applies the forward Fourier transform, an element-wise
    complex scaling, and inverse fourier transform

    .. math::
         \mathbf{y}= \mathbf{F}^{-1} \mathbf{S} \mathbf{F} \mathbf{x}

    Here :math:`\mathbf{S}` is a diagonal operator that scales the Fourier
    transformed input by :math:`e^{-j2\pi f t_S}`, where :math:`t_S` is the
    chosen ``shift``.

    """
    if dir is not None:
        warnings.warn(
            "dir is deprecated in version 2.0.0, use axis instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        axis = dir
    else:
        axis = axis

    Fop = FFT(
        dims,
        axis=axis,
        nfft=nfft,
        sampling=sampling,
        real=real,
        engine=engine,
        dtype=dtype,
        **kwargs_fftw
    )
    if isinstance(dims, int):
        dimsdiag = None
    else:
        dimsdiag = list(dims)
        dimsdiag[axis] = len(Fop.f)
    shift = np.exp(-1j * 2 * np.pi * Fop.f * shift)
    Sop = Diagonal(shift, dims=dimsdiag, axis=axis, dtype=Fop.cdtype)
    Op = Fop.H * Sop * Fop
    # force dtype to that of input (FFT always upcasts it to complex)
    Op.dtype = dtype
    return Op
