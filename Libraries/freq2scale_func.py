
import numpy as np
from numpy.fft import fft
#from __future__ import division, print_function, absolute_import
import pywt
import matplotlib.pyplot as plt


def central_frequency(wavelet, precision=8):
    """
    Computes the central frequency of the `psi` wavelet function.
    Parameters
    ----------
    wavelet : Wavelet instance, str or tuple
        Wavelet to integrate.  If a string, should be the name of a wavelet.
    precision : int, optional
        Precision that will be used for wavelet function
        approximation computed with the wavefun(level=precision)
        Wavelet's method (default: 8).
    Returns
    -------
    scalar
    """



    functions_approximations = wavelet.wavefun(precision)

    if len(functions_approximations) == 2:
        psi, x = functions_approximations
    else:
        # (psi, x)   for (phi, psi, x)
        # (psi_d, x) for (phi_d, psi_d, phi_r, psi_r, x)
        psi, x = functions_approximations[1], functions_approximations[-1]

    domain = float(x[-1] - x[0])
    assert domain > 0

    index = np.argmax(abs(fft(psi)[1:])) + 2
    if index > len(psi) / 2:
        index = len(psi) - index + 2

    return 1.0 / (domain / (index - 1))


def frequency2scale(wavelet, freq, precision=8):
    """Convert from to normalized frequency to CWT "scale".
    Parameters
    ----------
    wavelet : Wavelet instance or str
        Wavelet to integrate.  If a string, should be the name of a wavelet.
    freq : scalar
        Frequency, normalized so that the sampling frequency corresponds to a
        value of 1.0.
    precision : int, optional
        Precision that will be used for wavelet function approximation computed
        with ``wavelet.wavefun(level=precision)``.  Default is 8.
    Returns
    -------
    scale : scalar
    """
    return central_frequency(wavelet, precision=precision) / freq


















# fs  = 256
# wavelet = pywt.ContinuousWavelet('morl')

# frequencies = np.array([100, 90, 80, 70,60, 50 ,40, 30 ,20,10, 9, 8, 7,6, 5 ,4, 3 ,2, 1]) / fs # normalize
# scales = frequency2scale(wavelet,frequencies)
# print(scales)

# wavelet = pywt.ContinuousWavelet('cmor1.5-1.0')

# dt = 0.01  # 100 Hz sampling
# fs = 1 / dt
# frequencies = np.array([100, 50, 33.33333333, 25]) / fs # normalize
# scale = frequency2scale(wavelet, frequencies)
# print(scale)

# width = wavelet.upper_bound - wavelet.lower_bound
# # print(width)

# max_len = int(np.max(scales)*width + 1)

# max_len = int(np.max(scales)*width + 1)
# t = np.arange(max_len)
# fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))


# for n, scale in enumerate(scales):

#     # The following code is adapted from the internals of cwt
#     int_psi, x = pywt.integrate_wavelet(wavelet, precision=10)
#     step = x[1] - x[0]
#     j = np.floor(
#         np.arange(scale * width + 1) / (scale * step))
#     if np.max(j) >= np.size(int_psi):
#         j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
#     j = j.astype(np.int_)

#     # normalize int_psi for easier plotting
#     int_psi /= np.abs(int_psi).max()

#     # discrete samples of the integrated wavelet
#     filt = int_psi[j][::-1]

#     # The CWT consists of convolution of filt with the signal at this scale
#     # Here we plot this discrete convolution kernel at each scale.

#     nt = len(filt)
#     t = np.linspace(-nt//2, nt//2, nt)
#     axes[n, 0].plot(t, filt.real, t, filt.imag)
#     axes[n, 0].set_xlim([-max_len//2, max_len//2])
#     axes[n, 0].set_ylim([-1, 1])
#     axes[n, 0].text(50, 0.35, 'scale = {}'.format(scale))

#     f = np.linspace(-np.pi, np.pi, max_len)
#     filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
#     filt_fft /= np.abs(filt_fft).max()
#     axes[n, 1].plot(f, np.abs(filt_fft)**2)
#     axes[n, 1].set_xlim([-np.pi, np.pi])
#     axes[n, 1].set_ylim([0, 1])
#     axes[n, 1].set_xticks([-np.pi, 0, np.pi])
#     axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
#     axes[n, 1].grid(True, axis='x')
#     axes[n, 1].text(np.pi/2, 0.5, 'scale = {}'.format(scale))

# axes[n, 0].set_xlabel('time (samples)')
# axes[n, 1].set_xlabel('frequency (radians)')
# axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
# axes[0, 1].legend(['Power'], loc='upper left')
# axes[0, 0].set_title('filter')
# axes[0, 1].set_title(r'|FFT(filter)|$^2$')





# plt.show()