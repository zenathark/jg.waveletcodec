'''
File: tools.py
Author: jcgalanh@gmail.com
Description: A set of tools for signal processing based on numpy.ndarray
'''
from __future__ import division
import scipy as np

class CircularStack(object):
    '''
    classdocs
    '''
    data = []
    size = 0
    index = 0
    tail = 0

    def __init__(self, size=0):
        '''
        Constructor
        '''
        if size > 0:
            self.data = [0] * size
            self.size = size
        self.index = 0
        self.tail = 0

    def push(self, data):
        next_tail = self._get_next_tail()
        if next_tail == self.index:
            raise OverflowError
        else:
            self.data[self.tail] = data
            self.tail = next_tail

    def pop(self):
        if self.index == self.tail:
            raise IndexError
        else:
            result = self.data[self.index]
            self.index = (self.index + 1) % self.size
            return result

    def _get_next_tail(self):
        return (self.tail + 1) % self.size


def quant(wavelet, delta):
    #iw = zeros((len(wavelet.data),len(wavelet.data[0])),int64)
    iw = np.int64(wavelet.data / delta)
    wavelet.data = iw
    return wavelet


def quantize(signal, delta, **kw):
    """Quantizate a given signal using a hard threshold. The given data is
    multiplied times the given threshold. Notice that this quantization
    uses multiplication instead of division as most signal processing books
    explain quantization. This is in order to avoid more floating point
    complexity (To be checked)

    Args:
        signal The signal to be quantized
        delta  The hard threshold used for quantization

    Kwargs:
        dtype The type for the returned array. By default numpy.float64, if
        an integer type is given, the data will be truncated after
        quantization

    Raises:
        TypeError If the signal is not an instance of numpy.ndarray
    """
    iw = signal * float(delta)
    if 'dtype' in kw and kw['dtype'] is not np.float64:
        iw = iw.astype(kw['dtype'])
    return iw


def zero_padding(signal, squared=True):
    """Creates a new ndarray that """
    check_dim(signal, 2)
    rows, cols = signal.shape
    pow_rows = int(np.ceil(np.log2(rows)))
    pow_cols = int(np.ceil(np.log2(cols)))
    if squared:
        if pow_cols > pow_rows:
            pow_rows = pow_cols
        else:
            pow_cols = pow_rows
    padded_signal = np.zeros((2 ** pow_rows, 2 ** pow_cols),
                             dtype=signal.dtype)
    y_0 = np.trunc((2 ** pow_rows - rows) / 2)
    y_t = y_0 + signal.shape[0]
    x_0 = np.trunc((2 ** pow_cols - cols) / 2)
    x_t = x_0 + signal.shape[1]
    padded_signal[y_0:y_t, x_0:x_t] = signal
    return padded_signal

def unpadding(signal, dim):
    y, x = dim
    sy, sx = signal.shape
    y0 = np.trunc((sy-y) / 2)
    x0 = np.trunc((sx-x) / 2)
    yt = y0 + y
    xt = x0 + x
    return signal[y0:yt,x0:xt]


def normalize(data, **kw):
    """Calculates the normalization of the given array. The normalizated
    array is returned as a different array.

    Args:
        data The data to be normalized

    Kwargs:
        upper_bound The upper bound of the normalization. It has the value
        of 1 by default.
        lower_bound The lower bound to be used for normalization. It has the
        value of 0 by default
        dtype The type of the returned ndarray. If the dtype given is an
        integer type the returned array values will be truncated after
        normalized.

    Returns:
        An instance of np.array with normalizated values
    """
    upper_bound = 1
    lower_bound = 0
    dtype = np.float64
    if 'upper_bound' in kw:
        upper_bound = kw['upper_bound']
    if 'lower_bound' in kw:
        lower_bound = kw['lower_bound']
    if 'dtype' in kw:
        dtype = kw['dtype']
    check_ndarray(data)
    newdata = data - data.min()
    newdata = newdata / newdata.max()
    newdata = newdata * (upper_bound - lower_bound)
    newdata += lower_bound
    return newdata.astype(dtype)


def check_ndarray(data):
    """Check if the given data is an ndarray, if not raises an exception.

    Args:
        data The variable to be data checked

    Raises:
        A TypeError exception

    """
    if isinstance(data.__class__, np.ndarray):
        raise TypeError("Argument of type numpy.ndarray expected")


def check_dim(data, dim):
    """Check if the data match the given dimensions.

    Args:
        data The instance to be type checked
        dim The dimensions wanted for the data to be

    Raises:
        TypeEror if the data is not an instance of numpy.ndarray or if the
        data dimension does not match the desired amount

    """
    _check_ndarray(data)
    if data.ndim is not dim:
        msg = "Argument dimension mistmatch, expected " + dim + " dimensions"
        raise TypeError(msg)


def psnr(original, signal):
    """Calculate the psnr between the given signals.

    This method calculates the psnr of the given signal. Such signals are
    expected as a n-dimentional numpy.ndarray instance. The equation used for
    the psnr calculation is:
    .. math::
        PSNR = 20\log_10(MAX) - 10\log_{10}(MSE)

    Args:
        original (numpy.ndarray): The original signal to be compared against
        signal (numpy.ndarray): The signal to be compared

    Kwargs:
        MAX (int): The maximum value that a signal value can take. If MAX is
        not given, MAX takes the value of the maximum value of the array
        element np.type can take. Ex. if a signal has a dtype of numpy.uint8,
        max will be 255.

    Returns:
        A float that contains the PSNR

    """
    return 10


def mse(original, signal):
    """Calculate the MSE between the given signals.

    This method calculates the MSE between the given signals using the next
    equation
    .. math::
        MSE = \frac{1}{n}\sum_{i=0}{n-1}\[O(n) - I(n)\]^2

    where n is the size of the signal. If the given signal are n-dimentional,
    n can be a vector containing the size of each dimention

    Returns:
        A float that contains the MSE between the given signals

    """
    diff = (original - signal) ** 2
    MSE = diff.sum() / np.array(original.shape).prod()
    return MSE
