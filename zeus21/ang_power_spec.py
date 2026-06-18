"""
#Flat sky routines for angular power spectra calculations and Gaussian realisations.
#Author: Srini Raghunathan
#Date: 7 Dec, 2024
"""
import numpy as np, sys, os, scipy as sc

################################################################################################################
#flat-sky routines
################################################################################################################

def make_gaussian_realisation(flatskymapparams, el, cl, cl22 = None, cl12 = None, bl = None, qu_or_eb = 'eb'):

    """
    Create Gaussian realisation of an underlying power spectrum.
    Can either generate one map based on a single power spectrum or 2 maps based on the auto and the cross power spectra.

    Parameters
    ----------
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.
    el: array
        Multipoles at which the power spectrum have been evaluated. 
    cl: array
        1d power spectrum of the first observable.
    cl2: array
        1d power spectrum of the second observable.
    cl12: array
        1d cross-power spectrum of the first and the second observable.
    bl: array
        beam smoothing. same size as cl.
    qu_or_eb = str
        'eb' --> returns E/B maps if TT, EE, TE are supplied.
        'qu' --> returns Q/U maps if TT, EE, TE are supplied.

    Returns
    -------
    sim: array
        single map is only cl is supplied.
        Two maps is only cl, cl22, cl12 is supplied.
    cl: array
        auto/cross power spectra.
        Either 1d or 2D depending on return_2d.
    """
    nx, ny, dx, dy = flatskymapparams
    arcmins2radians = np.radians(1/60.)

    dx *= arcmins2radians
    dy *= arcmins2radians

    ################################################
    #map stuff
    norm = np.sqrt(1./ (dx * dy))
    ################################################

    #1d to 2d now
    cltwod = cl_to_cl2d(el, cl, flatskymapparams)
    
    ################################################
    if cl22 is not None: #for TE, etc. where two fields are correlated.
        assert cl12 is not None
        cltwod12 = cl_to_cl2d(el, cl12, flatskymapparams)
        cltwod22 = cl_to_cl2d(el, cl22, flatskymapparams)

    ################################################
    if cl22 is None:

        cltwod = cltwod**0.5 * norm
        cltwod[np.isnan(cltwod)] = 0.

        gauss_reals = np.random.standard_normal([ny,nx])
        sim = np.fft.ifft2( np.copy( cltwod ) * np.fft.fft2( gauss_reals ) ).real

    else: #for TE, etc. where two fields are correlated.

        cltwod12[np.isnan(cltwod12)] = 0.
        cltwod22[np.isnan(cltwod22)] = 0.

        gauss_reals_1 = np.random.standard_normal([ny,nx])
        gauss_reals_2 = np.random.standard_normal([ny,nx])

        gauss_reals_1_fft = np.fft.fft2( gauss_reals_1 )
        gauss_reals_2_fft = np.fft.fft2( gauss_reals_2 )

        #field_1
        cltwod_tmp = np.copy( cltwod )**0.5 * norm
        sim_1 = np.fft.ifft2( cltwod_tmp *  gauss_reals_1_fft ).real

        #field 2 - has correlation with field_1
        t1 = np.copy( gauss_reals_1_fft ) * cltwod12 / np.copy(cltwod)**0.5
        t2 = np.copy( gauss_reals_2_fft ) * ( cltwod22 - (cltwod12**2. /np.copy(cltwod)) )**0.5
        sim_2_fft = (t1 + t2) * norm
        sim_2_fft[np.isnan(sim_2_fft)] = 0.
        sim_2 = np.fft.ifft2( sim_2_fft ).real

        sim = np.asarray( [sim_1, sim_2] )


    if bl is not None:
        if np.ndim(bl) != 2:
            bl = cl_to_cl2d(el, bl, flatskymapparams)
        sim = np.fft.ifft2( np.fft.fft2(sim) * bl).real

    sim = sim - np.mean(sim)

    return sim

################################################################################################################

def map2cl(flatskymapparams, flatskymap1, flatskymap2 = None, binsize = None, minbin = 1, maxbin = 10000, mask = None, filter_2d = None, return_2d = False):

    """
    map2cl module - get the power spectra of map/maps

    Parameters
    ----------
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.
    flatskymap1: array
        map1 with dimensions (ny, nx).
    flatskymap2: array, optional.
        map2 with dimensions (ny, nx).
        if map2 is None, then we calcualte the auto powerspectrum of map1.
        else, we calcaulte the cross power spectrum between map1 and map2.
    binsize: array, optional
        el bins. computed automatically based on the grid size if None.
    minbin: int, optional
        minimum multipole for power spectrum calculation. default is 1.
    maxbin: int, optional
        maximum multipole for power spectrum calculation. default is 10000.
    mask: array, optional.
        either point source or apodisation mask to be used for power spectrum calculation.
        Adding point source mask can lead to mode-coupling for red spectra. So be careful.
    filter_2d: array, optional.
        estimate of the filter modes.
    return_2d: bool, optional.
        Return 1d or 2D power spectrum.

    Returns
    -------
    el: array
        Multipoles at which the power spectrum have been evaluated. 
        Either 1d or 2D depending on return_2d.
    cl: array
        auto/cross power spectra.
        Either 1d or 2D depending on return_2d.
    """

    nx, ny, angres_am = flatskymapparams
    angres_rad = np.radians(angres_am/60.)

    lx, ly = get_lxly(flatskymapparams)
    el_grid = np.sqrt( lx**2. + ly**2. )
    
    if binsize == None:
        binsize = lx.ravel()[1] -lx.ravel()[0]

    if flatskymap2 is None:
        flatskymap_psd = abs( np.fft.fft2(flatskymap1) * angres_rad)** 2 / (nx * ny)
    else: #cross spectra now
        assert flatskymap1.shape == flatskymap2.shape
        flatskymap_psd = np.fft.fft2(flatskymap1) * angres_rad * np.conj( np.fft.fft2(flatskymap2) ) * angres_rad / (nx * ny)

    if filter_2d is not None:
        flatskymap_psd = flatskymap_psd / filter_2d
        flatskymap_psd[np.isinf(flatskymap_psd) | np.isnan(flatskymap_psd)] = 0.
        flatskymap_psd[abs(flatskymap_psd)>1e300] = 0.

    if return_2d:
        el = el_grid
        cl = flatskymap_psd
    else:
        el, cl = radial_profile(flatskymap_psd, binsize, maxbin, minbin=minbin, xy=(lx,ly), return_errors=False)

    if mask is not None:
        fsky = np.mean(mask**2.)
        cl /= fsky

    return el, cl

################################################################################################################

def cl_to_cl2d(el, cl, flatskymapparams):

    """
    converts 1d_cl to 2d_cl

    Parameters
    ----------
    el: array
        el values over which cl is defined
    cl: array
        power spectra cl that must be interpolated on the 2D grid.
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.

    Returns
    -------
    cl2d: array
        power spectra interpolated on the 2D grid.
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)

    cl2d = np.interp(ell.flatten(), el, cl).reshape(ell.shape) 

    return cl2d

################################################################################################################

def get_lxly(flatskymapparams):

    """
    returns lx, ly based on the flatskymap parameters
    
    Parameters
    ----------
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.

    Returns
    -------
    lx: array
        Multipoles lx in 2D
    ly: array
        Multipoles ly in 2D
    """

    nx, ny, angres_am = flatskymapparams
    angres_rad = np.radians(angres_am/60.)

    lx, ly = np.meshgrid( np.fft.fftfreq( nx, angres_rad ), np.fft.fftfreq( ny, angres_rad ) )
    lx *= 2* np.pi
    ly *= 2* np.pi

    return lx, ly

################################################################################################################

def get_lxly_az_angle(lx,ly):

    """
    returns the azimuthal angle from lx, ly

    Parameters
    ----------
    lx: array
        Multipoles lx in 2D
    ly: array
        Multipoles ly in 2D

    Returns
    -------
    psi: array
        azimuthal angle
    """

    psi = 2*np.arctan2(lx, -ly)
    
    return psi

################################################################################################################
def convert_eb_qu(map1, map2, flatskymapparams, eb_to_qu = True):

    """
    converts CMB polarisation maps from E/B-mode maps to Stokes Q/U or vice versa.

    Parameters
    ----------
    map1: array
        polarisation map1. E/B or Q/U.
    map2: array
        polarisation map2. E/B or Q/U.
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.
    eb_to_qu: bool
        True --> convert E/B to Q/U.
        False --> convert Q/U to E/B.

    Returns
    -------
    map1_mod: array
        rotated polarisation map1. E/B or Q/U.
    map2_mod: array
        rotated polarisation map2. E/B or Q/U.
    """

    lx, ly = get_lxly(flatskymapparams)
    angle = get_lxly_az_angle(lx,ly)

    map1_fft, map2_fft = np.fft.fft2(map1),np.fft.fft2(map2)
    if eb_to_qu:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft - np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real
    else:
        map1_mod = np.fft.ifft2( np.cos(angle) * map1_fft + np.sin(angle) * map2_fft ).real
        map2_mod = np.fft.ifft2( -np.sin(angle) * map1_fft + np.cos(angle) * map2_fft ).real

    return map1_mod, map2_mod

################################################################################################################
def get_lpf_hpf(flatskymapparams, lmin_lmax, filter_type = 0):

    """
    returns 2D Fourier space low- or high-pass filters.

    Parameters
    ----------
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.
    lmin_lmax: int
        minimum (maximum) mumtipole for HPF (LPF).
    filter_type: int
        0 - low pass filter
        filter_type = 1 - high pass filter
        filter_type = 2 - band pass

    Returns
    -------
    fft_filter: array
        2D LPF or HPF
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_filter = np.ones(ell.shape)
    if filter_type == 0:
        fft_filter[ell>lmin_lmax] = 0.
    elif filter_type == 1:
        fft_filter[ell<lmin_lmax] = 0.
    elif filter_type == 2:
        lmin, lmax = lmin_lmax
        fft_filter[ell<lmin] = 0.
        fft_filter[ell>lmax] = 0

    return fft_filter

################################################################################################################

def get_bpf(flatskymapparams, lmin, lmax):

    """
    returns 2D Fourier band-pass filter.

    Parameters
    ----------
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.
    lmin: int
        minimum multipole for the band pass.
    lmax: int
        maximum multipole for the band pass.

    Returns
    -------
    fft_bpf: array
        2D band-pass filter.
    """

    lx, ly = get_lxly(flatskymapparams)
    ell = np.sqrt(lx**2. + ly**2.)
    fft_bpf = np.ones(ell.shape)
    fft_bpf[ell<lmin] = 0.
    fft_bpf[ell>lmax] = 0

    return fft_bpf

################################################################################################################

def wiener_filter(flatskymapparams, cl_signal, cl_noise, el = None):

    """
    returns 2D Wiener filter.

    Parameters
    ----------
    flatskymapparams: list
        [nx, ny, angres_am] where ny, nx = flatskymap.shape; and angres_am are the pixel resolution in arcminutes.
        for example: [100, 100, 0.5] is a 50' x 50' flatskymap that has dimensions 100 x 100 with angres_am = 0.5 arcminutes.
    cl_signal: array
        1d signal power spectrum.
    cl_noise: array
        1d noise power spectrum.
    el: array, optional
        Multipoles at which the signal and noise power spectra have been evaluated.
        If not passed, then we will assume delta_el = 1 using el=np.arange(len(cl_signal)).

    Returns
    -------
    wiener_filter: array
        2D Wiener filter.
    """

    if el is None:
        el = np.arange(len(cl_signal))

    #get 2D cl
    cl_signal2d = cl_to_cl2d(el, cl_signal, flatskymapparams) 
    cl_noise2d = cl_to_cl2d(el, cl_noise, flatskymapparams) 

    wiener_filter = cl_signal2d / (cl_signal2d + cl_noise2d)

    return wiener_filter


def radial_profile(image, binsize, maxbin, minbin=0.0, xy=None, return_errors=False):
    """
    Get the radial profile of an image (both real and fourier space)

    Parameters
    ----------
    image : array
        Image/array that must be radially averaged.
    binsize : float
        Size of radial bins.  In real space, this is
        radians/arcminutes/degrees/pixels.  In Fourier space, this is
        \Delta\ell.
    maxbin : float
        Maximum bin value for radial bins.
    minbin : float
        Minimum bin value for radial bins.
    xy : 2D array
        x and y grid points.  Default is None in which case the code will simply
        use pixels indices as grid points.
    return_errors : bool
        If True, return standard error.

    Returns
    -------
    bins : array
        Radial bin positions.
    vals : array
        Radially binned values.
    errors : array
        Standard error on the radially binned values if ``return_errors`` is
        True.
    """

    image = np.asarray(image)
    if xy is None:
        y, x = np.indices(image.shape)
    else:
        y, x = xy

    radius = np.hypot(y, x)
    radial_bins = np.arange(minbin, maxbin, binsize)

    hits = np.zeros(len(radial_bins), dtype=float)
    vals = np.zeros_like(hits)
    errors = np.zeros_like(hits)

    for ib, b in enumerate(radial_bins):
        inds = np.where((radius >= b) & (radius < b + binsize))
        imrad = image[inds]
        total = np.sum(imrad != 0.0)
        hits[ib] = total

        if total > 0:
            ###print(ib, b, total, np.sum(imrad), imrad)
            # mean value in each radial bin
            vals[ib] = np.sum(imrad) / total
            errors[ib] = np.std(imrad)

    bins = radial_bins + binsize / 2.0

    std_mean = np.sum(errors * hits) / np.sum(hits)
    
    if return_errors:
        errors = std_mean / hits ** 0.5
        return bins, vals, errors
    else:
        return bins, vals

################################################################################################################

def gauss_beam(fwhm, lmax=512):
    """
    Gaussian beam window function from healpy

    Computes the spherical transform of an axisimmetric gaussian beam

    For a sky of underlying power spectrum C(l) observed with beam of
    given FWHM, the measured power spectrum will be
    C(l)_meas = C(l) B(l)^2
    where B(l) is given by gaussbeam(Fwhm,Lmax).
    The polarization beam is also provided (when pol = True ) assuming
    a perfectly co-polarized beam
    (e.g., Challinor et al 2000, astro-ph/0008228)

    Parameters
    ----------
    fwhm : float
        full width half max in radians
    lmax : integer
        ell max

    Returns
    -------
    gau_beam : array
        beam window function [0, lmax].
    """

    sigma = fwhm / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(lmax + 1)
    sigma2 = sigma ** 2
    gau_beam = np.exp(-0.5 * ell * (ell + 1) * sigma2)

    return gau_beam