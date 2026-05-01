"""
Helper functions to be used across zeus21

Authors: Yonatan Sklansky, Emilie Thelie
UT Austin - February 2025

"""

import numpy as np
import powerbox as pbox
from pyfftw import empty_aligned as empty
import time
import gc

from . import constants
from scipy.stats import lognorm


try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        return lambda func: func
    njit = lambda func: func


def powerboxCtoR(pbobject,mapkin = None):
    'Function to convert a complex field to real 3D (eg density, T21...) on the powerbox notation'
    'Takes a powerbox object pbobject, and a map in k space (mapkin), or otherwise assumes its pbobject.delta_k() (tho in that case it should be delta_x() so...'

    realmap = empty((pbobject.N,) * pbobject.dim, dtype='complex128')
    if (mapkin is None):
        realmap[...] = pbobject.delta_k()
    else:
        realmap[...] = mapkin
    realmap[...] = pbobject.V * pbox.dft.ifft(realmap, L=pbobject.boxlength, a=pbobject.fourier_a, b=pbobject.fourier_b)[0]
    realmap = np.real(realmap)

    return realmap

def tophat_smooth(rr, ks, dk):
    x = ks * rr + 1e-5
    win_k = 3/(x**3) * (np.sin(x) - x*np.cos(x))
    deltakfilt = dk * win_k
    return np.real(np.fft.ifftn(deltakfilt))




def _WinTH(k,R):
    x = k * R
    return 3.0/x**2 * (np.sin(x)/x - np.cos(x))

def _WinTH1D(k,R):
    x = k * R
    return  np.sin(x)/x

def _WinG(k,R):
    x = k * R * constants.RGauss_factor
    return np.exp(-x**2/2.0)

def Window(k, R, WINDOWTYPE="TOPHAT"):
    if WINDOWTYPE == 'TOPHAT':
        return _WinTH(k, R)
    elif WINDOWTYPE == 'GAUSS':
        return _WinG(k, R)
    elif WINDOWTYPE == 'TOPHAT1D':
        return _WinTH1D(k, R)
    else:
        print('ERROR in Window. Wrong type')





def find_nearest_idx(array, values):
    array = np.atleast_1d(array)
    values = np.atleast_1d(values)
    idx = []
    for i in range(len(values)):
        idx.append((np.abs(array - values[i])).argmin())
    return np.unique(idx)

def print_timer(start_time, text_before="", text_after=""):
    elapsed_time = time.time() - start_time
    mins = int(elapsed_time//60)
    secs = int(elapsed_time - mins*60)
    print(f"{text_before}{mins}min {secs}s{text_after}")

def v2r(v):
    return (3/4/np.pi * v)**(1/3)
    
def r2v(r):
    return 4/3 * np.pi * r**3

def delete_class_attributes(class_instance): # delete all attributes of the class instance
    for attr in list(class_instance.__dict__):    
        delattr(class_instance, attr)
    gc.collect()



# SarahLibanore
# PDFs for the SFH

@njit
def pdf_log_transform(y_values, pdf_y_values):
    """
    Get PDF of X = ln(Y) given Y values and their PDF values
    
    Uses the transformation rule: f_X(x) = f_Y(y) * |dy/dx|
    where x = ln(y), so dy/dx = y. So: f_X(x) = f_Y(y) * y
    """
    y_values = np.asarray(y_values)
    pdf_y_values = np.asarray(pdf_y_values)
    
    # Remove any y <= 0 values (can't take log)
    y_clean = np.fmax(1e-9, y_values)  # Avoid log(0) or log(negative)
    pdf_y_clean = np.fmax(1e-50, pdf_y_values)  # Avoid zero PDF values
    
    # Transform: x = ln(y)
    x_values = np.log(y_clean)
    
    # Apply transformation rule: f_X(x) = f_Y(y) * y
    pdf_x_values = pdf_y_clean * y_clean
    return x_values, pdf_x_values

@njit
def lognormal_pdf(y, mu, sigma):
    """
    Vectorized lognormal PDF(y)
    """
    y = np.asarray(y)
    result = np.zeros_like(y)
    y_pos = np.fmax(1e-9, y)
    result = (1 / (y_pos * sigma * np.sqrt(2 * np.pi))) * \
                  np.exp(-0.5 * ((np.log(y_pos) - mu) / sigma)**2)

    return result

@njit
def normal_pdf(y, mu, sigma,dimy=None):
    """
    Vectorized normal PDF(y). dimy is the number of dimensions for mu and sigma.
    """
    y = np.asarray(y)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    if dimy is not None:
        for i in range(dimy):
            mu = mu[:, None]
            sigma = sigma[:, None]

    return np.exp(-0.5 * ((y - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


def pdf_fft_convolution(mu1, sigma1, mu2, sigma2, highp=0.99):
    """
    FFT convolution method to compute the PDF of the sum of two lognormal distributions
    Uses the convolution theorem: convolution in real space = multiplication in Fourier space
    mu1, sigma1: parameters of the first lognormal distribution
    mu2, sigma2: parameters of the second lognormal distribution
    returns:
        y_array: the range of y values for which the PDF is computed
        pdf_values: the PDF values at those y values
    """
    
    q_low = 1.0-highp  # 0.1% quantile
    q_high = highp # 99.9% quantile
    
    y1_low = lognorm.ppf(q_low, s=sigma1, scale=np.exp(mu1))
    y2_low = lognorm.ppf(q_low, s=sigma2, scale=np.exp(mu2))
    y1_high = lognorm.ppf(q_high, s=sigma1, scale=np.exp(mu1))
    y2_high = lognorm.ppf(q_high, s=sigma2, scale=np.exp(mu2))
    
    y_min = max(1., y1_low + y2_low)
    y_max = 2.*(y1_high + y2_high)
    n_points = int(y_max/y_min) #to ensure we capture the resolution
    n_points = max(n_points, 64)  # Ensure at least 64 points
    
    # Increase points for small sigmas (to capture narrow peaks) and wide ones (for sampling)
    min_sigma = min(sigma1, sigma2)
    if min_sigma < 0.5:
        n_points = int(n_points * (2 / (min_sigma+0.4)))  # Scale inversely with sigma
    # Ensure n_points is power of 2 for efficient FFT
    n_points = int(2 ** np.ceil(np.log2(n_points)))

    if (n_points < 4097):
        # Create uniform grid for FFT
        y_uniform = np.linspace(0.0, y_max, n_points).flatten()
        dy = y_uniform[1] - y_uniform[0]
        
        # Compute PDFs on uniform grid (vectorized)
        pdf1_grid = lognormal_pdf(y_uniform, mu1, sigma1)
        pdf2_grid = lognormal_pdf(y_uniform, mu2, sigma2)

        norm1 = np.trapezoid(pdf1_grid, y_uniform)
        norm2 = np.trapezoid(pdf2_grid, y_uniform)
        pdf1_grid /= norm1
        pdf2_grid /= norm2
        
        # FFT convolution - this is where the magic happens! Convolution in real space = multiplication in Fourier space
        fft1 = np.fft.fft(pdf1_grid)
        fft2 = np.fft.fft(pdf2_grid)
        fft_conv = fft1 * fft2  # Element-wise multiplication

        # Inverse FFT to get back to real space
        pdf_conv = np.real(np.fft.ifft(fft_conv)) * dy

        y_uniform, pdf_conv = y_uniform[1:], pdf_conv[1:] #to remove y=0 which is annoying for log transform
    else:
        n_points = 10 #direct integration, few points are enough to get the mean and rms, but can crank up if wanted
        y_uniform = np.geomspace(y_min, y_max, n_points).flatten()
        n_points_integral = 333
        yintegralgrid = np.geomspace(y_min, y_max, n_points_integral).flatten()
        pdf1_grid = lognormal_pdf(yintegralgrid, mu1, sigma1)
        pdf_conv = np.zeros_like(y_uniform)
        YminusYintegralgrid = y_uniform[:, np.newaxis] - yintegralgrid[np.newaxis, :]
        pdf2_grid = lognormal_pdf(YminusYintegralgrid, mu2, sigma2)
        pdf_conv = np.trapezoid(pdf1_grid[None,:] * pdf2_grid * np.heaviside(YminusYintegralgrid, 0.5), x=yintegralgrid)

    return y_uniform, np.maximum(pdf_conv, 0)


def sigma_log10(sigmaquantity, meanquantity):
    "Returns the sigma(log10) for a given quantity with mean and sigma in linear units"
    return np.sqrt(np.log((sigmaquantity/meanquantity)**2+1.))/np.log(10)

def mean_log10(sigmaquantity, meanquantity):
    "Returns the mean(log10) for a given quantity with mean and sigma in linear units"
    return np.log10(meanquantity)- 1/2 * np.log10(1 + sigmaquantity**2/meanquantity**2)

