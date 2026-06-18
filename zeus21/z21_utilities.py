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
import mcfit


try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        return lambda func: func
    njit = lambda func: func


def powerboxCtoR(pbobject,mapkin = None):
    """
    Converts a complex field to real 3D (eg density, T21...) on the powerbox notation.

    Parameters
    ----------
    pbobject: powerbox.PowerBox
        PowerBox object 
    mapkin: np.ndarray
        Map of the field in k space. Default is None. 
        Otherwise assumes its pbobject.delta_k() (although in that case it should be directly pbobject.delta_x()).

    Returns
    ----------
    realmap: np.ndarray
        Real 3D field
    """

    realmap = empty((pbobject.N,) * pbobject.dim, dtype='complex128')
    if (mapkin is None):
        realmap[...] = pbobject.delta_k()
    else:
        realmap[...] = mapkin
    realmap[...] = pbobject.V * pbox.dft.ifft(realmap, L=pbobject.boxlength, a=pbobject.fourier_a, b=pbobject.fourier_b)[0]
    realmap = np.real(realmap)

    return realmap

def tophat_smooth(rr, ks, dk):
    """
    Top-hat smoothing.

    Parameters
    ----------
    rr: np.ndarray
        Array of radii.
    ks: np.ndarray
        Array of wave numbers.
    dk: np.ndarray
        Field to be smoothed in Fourier space.

    Returns
    ----------
    np.ndarray
        Smoothed field in real space.
    """
    x = ks * rr + 1e-5
    win_k = 3/(x**3) * (np.sin(x) - x*np.cos(x))
    deltakfilt = dk * win_k
    return np.real(np.fft.ifftn(deltakfilt))




def _WinTH(k,R):
    """
    3D top-hat window function.

    Parameters
    ----------
    k: np.ndarray
        Array of wave numbers.
    R: np.ndarray
        Array of radii.

    Returns
    ----------
    np.ndarray
        3D window function.
    """
    x = k * R
    return 3.0/x**2 * (np.sin(x)/x - np.cos(x))

def _WinTH1D(k,R):
    """
    1D top-hat window function.

    Parameters
    ----------
    k: np.ndarray
        Array of wave numbers.
    R: np.ndarray
        Array of radii.

    Returns
    ----------
    np.ndarray
        1D window function.
    """
    x = k * R
    return  np.sin(x)/x

def _WinG(k,R):
    """
    Gaussian window function.

    Parameters
    ----------
    k: np.ndarray
        Array of wave numbers.
    R: np.ndarray
        Array of radii.

    Returns
    ----------
    np.ndarray
        Window function.
    """
    x = k * R * constants.RGauss_factor
    return np.exp(-x**2/2.0)

def Window(k, R, WINDOWTYPE="TOPHAT"):
    """
    Window function.

    Parameters
    ----------
    k: np.ndarray
        Array of wave numbers.
    R: np.ndarray
        Array of radii.
    WINDOWTYPE: str
        Which window function to return. Default is TOPHAT. Can also be GAUSS or TOPHAT1D.

    Returns
    ----------
    np.ndarray
        Window function.
    """
    if WINDOWTYPE == 'TOPHAT':
        return _WinTH(k, R)
    elif WINDOWTYPE == 'GAUSS':
        return _WinG(k, R)
    elif WINDOWTYPE == 'TOPHAT1D':
        return _WinTH1D(k, R)
    else:
        print('ERROR in Window. Wrong type')





def find_nearest_idx(array, values):
    """
    Finds the nearest indices for some values inside another array. 

    Parameters
    ----------
    array: np.ndarray
        Array from which to find the indices.
    values: np.ndarray
        Values for which we are searching the indices in array.

    Returns
    ----------
    np.ndarray
        Array of indices.
    """
    array = np.atleast_1d(array)
    values = np.atleast_1d(values)
    idx = []
    for i in range(len(values)):
        idx.append((np.abs(array - values[i])).argmin())
    return np.unique(idx)

def print_timer(start_time, text_before="", text_after=""):
    """
    Prints the duration since an initial time.

    Parameters
    ----------
    start_time: time.time()
        Initial time.
    text_before: str
        Text to print in front of the timer. Default is "".
    text_after: str
        Text to print after the timer. Default is "".
    """
    elapsed_time = time.time() - start_time
    mins = int(elapsed_time//60)
    secs = int(elapsed_time - mins*60)
    print(f"{text_before}{mins}min {secs}s{text_after}")

def v2r(v):
    """
    Computes the radius from a volume assuming a sphericity.

    Parameters
    ----------
    v: float | np.ndarray
        Volume of the object.

    Returns
    ----------
    float | np.ndarray
       Radius of the object.
    """
    return (3/4/np.pi * v)**(1/3)
    
def r2v(r):
    """
    Computes the volume of a sphere of radius r.

    Parameters
    ----------
    r: float | np.ndarray
        Radius.

    Returns
    ----------
    float | np.ndarray
       Volume.
    """
    return 4/3 * np.pi * r**3

def delete_class_attributes(class_instance): # delete all attributes of the class instance
    """
    Properly deallocates all the attributes of a class instance. Calls the garbage collector.
    Useful when we want to deallocate an instance 
    (doing del cls will not deallocate the attributes instantly as long as the garbage collector hasn't run).

    Parameters
    ----------
    class_instance: cls instance
        Class instance.
    """
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
    """
    Returns the sigma(log10) for a given quantity with mean and sigma in linear units
    """
    return np.sqrt(np.log((sigmaquantity/meanquantity)**2+1.))/np.log(10)

def mean_log10(sigmaquantity, meanquantity):
    """
    Returns the mean(log10) for a given quantity with mean and sigma in linear units
    """
    return np.log10(meanquantity)- 1/2 * np.log10(1 + sigmaquantity**2/meanquantity**2)


def get_Pk_from_xi(rsinput, xiinput):
    """
    Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF

    Parameters
    ----------
    rsinput : array
        Array of Rs used to evaluate xiinput
    xiinput: matrix
        Matrix of values you are Fourier Transforming. Dimension (z, R)

    Returns
    ----------
    kPf: list
        List of wavenumbers
    Pf: matrix
        Resultant Fourier Transform of xiinput. Dimension (z, k)

    """
    
    kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

    return kPf, Pf


def get_list_PS(CosmoParams, xi_list, zlisttoconvert):
    """
    Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input

    Parameters
    ----------
    xi_list : matrix
        list of correlation functions
    zlisttoconvert: array
        which redshifts xi_list is evaluated at

    Returns
    ----------
    _Pk_list: matrix
        Matrix of power spectra. Dimension (z, K)

    """
    _Pk_list = []

    for izp,zp in enumerate(zlisttoconvert):

        _kzp, _Pkzp = get_Pk_from_xi(CosmoParams.rlist_CF,xi_list[izp])
        _Pk_list.append(_Pkzp)

    return np.array(_Pk_list)


def smooth_box(box, Resolution, input_boxlength, ncells):
    """
    Smooth box 
    
    Parameters
    ----------
    box : matrix
        Box 
    Resolution : float
        Resolution over which you want to smooth 
    input_boxlength : int
        Box size
    ncells : int
        Number of cells per side

    Returns
    ----------
    box_smooth : matrix
        Smoothed box
    """

    box_fft = np.fft.fftn(box)

    klistfftx = np.fft.fftfreq(box.shape[0],input_boxlength/ncells)*2*np.pi

    klist3Dfft = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))

    box_smooth = np.array(tophat_smooth(Resolution, klist3Dfft, box_fft))
    
    return box_smooth