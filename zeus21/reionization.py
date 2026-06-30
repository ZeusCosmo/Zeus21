"""
Models reionization using an analogy of a halo mass function to ionized bubbles.

Authors: zeus21 v2 collaboration - June 2026
    Emily Bregou
    Hector Afonso G. Cruz
    Sarah Libanore
    Julian B. Muñoz
    Yonny Sklansky
    Emilie Thélie
    Alessandra Venditti
arXiv:2302.08506, arXiv:2306.09403, arXiv:2407.18294, Sklansky et al. (in prep)
"""

from . import z21_utilities
from . import cosmology
from . import constants
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.special import erfc
from tqdm import trange


class reionization_global:
    """
    Computes the global reionization history and bubble mass function (BMF).

    Parameters
    ----------
    CosmoParams : CosmoParams class
    AstroParams : AstroParams class
    HMFintclass : HMFinterp class
    z_Init : Z_init class
    SFRD_Init : SFRD_class class
    PRINT_SUCCESS : bool, optional
        Whether to print convergence status for BMF iteration.
        Default is True.

    Attributes
    ----------
    PRINT_SUCCESS : bool
        Whether to print convergence status messages.
    zlist : array
        Redshift array.
    Rs : array
        Smoothing radii array matching the rest of the code, in cMpc.
    Rs_BMF : array
        Bubble radii only used for the BMF, in cMpc.
    ds_array : array
        Sample overdensity values used to compute the ionization barrier.
    gamma : array
        Linear coefficient multiplying delta in niondot fit.
    gamma2 : array
        Quadratic coefficient multiplying delta in niondot fit.
    sigma : array
        Matter fluctuation on (zlist, Rs) grid.
    gamma_int : RegularGridInterpolator
        Interpolator for gamma as a function of z and log R.
    gamma2_int : RegularGridInterpolator
        Interpolator for gamma2 as a function of z and log R.
    sigma_BMF : array
        Matter fluctuation evaluated on (zlist, Rs_BMF) grid.
    sigma_int : RegularGridInterpolator
        Interpolator for sigma as a function of z and log R.
    trec0 : float
        Recombination time normalization at z=0.
    trec : array
        Recombination time evaluated on zlist, in years.
    trec_int : interp1d
        Interpolator for recombination time as a function of z.
    niondot_avg : array
        Average ionizing photon production rate as a function of z.
    niondot_avg_int : interp1d
        Interpolator for the average ionizing photon production rate.
    ion_frac : array
        Global ionized fraction as a function of z.
    ion_frac_initial : array
        Initial guess of the global ionized fraction before BMF convergence, from Madau equation.
    nion_norm : array
        Normalization factor for niondot fit to match the average niondot, evaluated on (zlist, Rs) grid.
    nion_norm_int : RegularGridInterpolator
        Interpolator for nion_norm as a function of z and log R.
    prebarrier_xHII : array
        Ionized fraction evaluated on delta, z, and R before computing the barrier.
    barrier : array
        Density barrier threshold for ionization as a function of z and R.
    barrier_initial : array
        Initial barrier before BMF convergence.
    barrier_int : RegularGridInterpolator
        Interpolator for the barrier as a function of z and log R.
    prebarrier_xHII_int : RegularGridInterpolator
        Interpolator for prebarrier_xHII as a function of delta, z, and log R.
    R_linear_sigma_fit_idx : int
        Index of the smoothing radius closest to AstroParams.R_linear_sigma_fit_input.
    R_linear_sigma_fit : float
        Radius used as the initial guess of the peak scale for the linear barrier fit, in cMpc.
    BMF : array
        BMF evaluated on (zlist, Rs_BMF) grid.
    BMF_initial : array
        Initial BMF based on first guess values before convergence.
    peakRofz : array
        Peak bubble radius as a function of z, in cMpc.
    peakRofz_int : interp1d
        Interpolator for peak bubble radius as a function of z.

    compute_prebarrier_xHII : method
        Computes the ionized fraction before solving for the barrier.
    compute_barrier : method
        Computes the density barrier threshold for ionization.
    nion_normalization : method
        Computes the normalization factor for the niondot fit.
    nrec : method
        Computes the cumulative number of recombinations over delta and z.
    niondot_delta_r : method
        Computes the delta and R-dependent ionizing photon production rate.
    nion_delta_r_int : method
        Computes the cumulative number of ionizing photons produced since the maximum redshift.
    Madau_Q : method
        Computes the global ionized fraction by solving the Madau equation.
    B_1 : method
        Computes the slope term of the linear ionization barrier.
    B_0 : method
        Computes the y-intercept term of the linear ionization barrier.
    B : method
        Computes the linear ionization barrier as a function of z and R.
    dlogsigma_dlogR : method
        Computes dlogsigma/dlogR.
    VRdn_dR : method
        Computes the volume-weighted BMF. Integrating this quantity gives the global xHII.
    Rdn_dR : method
        Computes the number-weighted BMF.
    BMF_peak_R : method
        Finds the radius at which the BMF peaks.
    monotonic_after_peak : method
        Enforces monotonic growth of the bubble size peak with time.
    analytic_Q : method
        Analytically integrate the BMF to compute global xHII.
    converge_BMF : method
        Iteratively updates the ionization barrier, BMF, and xHII until convergence.
    interpR : method
        Evaluates any interpolator at fixed z and an array of R.
    interpz : method
        Evaluates any interpolator at an array of z and fixed R.
    sigmaR_int : method
        Interpolates sigma at fixed z and an array of R.
    sigmaz_int : method
        Interpolates sigma at an array of z and fixed R.
    barrierR_int : method
        Interpolates the ionization barrier at fixed z and an array of R.
    barrierz_int : method
        Interpolates the ionization barrier at an array of z and fixed R.
    gammaR_int : method
        Interpolates gamma at at fixed z and an array of R.
    gammaz_int : method
        Interpolates gamma at an array of z and fixed R.
    gamma2R_int : method
        Interpolates gamma2 at at fixed z and an array of R.
    gamma2z_int : method
        Interpolates gamma2 at an array of z and fixed R.
    nion_normR_int : method
        Interpolates nion_norm at at fixed z and an array of R.
    nion_normz_int : method
        Interpolates nion_norm at an array of z and fixed R.
    interp_zR : method
        Evaluates any RegularGridInterpolator on z and R arrays.
    sigma_zR_int : method
        Interpolates sigma on z and R arrays.
    barrier_zR_int : method
        Interpolates the barrier on z and R arrays.
    gamma_zR_int : method
        Interpolates gamma on z and R arrays.
    gamma2_zR_int : method
        Interpolates gamma2 on z and R arrays.
    nion_norm_zR_int : method
        Interpolates nion_norm on z and R arrays.
    prebarrier_xHII_int_grid : method
        Evaluates prebarrier xHII on a delta field at fixed z and R.
    """
    def __init__(self, CosmoParams, AstroParams, HMFintclass, z_Init, SFRD_Init, PRINT_SUCCESS=True):

        #initializing values and interpolators that will be used in the reionization calculations. The ones that depend on the BMF (ion_frac, barrier, peakRofz) are initialized but will be updated if AstroParams.FLAG_BMF_converge is True.
        self.PRINT_SUCCESS = PRINT_SUCCESS
        self.zlist = z_Init.zintegral
        self.Rs = CosmoParams._Rtabsmoo
        #initialize separate R array for the BMF focused on the relevant range of bubble sizes.
        self.Rs_BMF = np.logspace(np.log10(AstroParams.Rbub_min), np.log10(self.Rs[-1]), 100)
        self.ds_array = np.linspace(-1, 5, 101)

        self._r_array, self._z_array = np.meshgrid(self.Rs, self.zlist, sparse=True, indexing='ij')
        self._rb_array, self._z_array = np.meshgrid(self.Rs_BMF, self.zlist, sparse=True, indexing='ij')
        
        self.gamma = SFRD_Init.gamma_niondot_II_index2D ### TODO maybe not store them as attributes?
        self.gamma2 = SFRD_Init.gamma2_niondot_II_index2D ### TODO maybe not store them as attributes?
        self.sigma =  HMFintclass.sigmaRintlog((np.log(self._r_array), self._z_array)).T
        
        self.zr = [self.zlist, np.log(self.Rs)]
        self.gamma_int = RegularGridInterpolator(self.zr, self.gamma, bounds_error = False, fill_value = None)
        self.gamma2_int = RegularGridInterpolator(self.zr, self.gamma2, bounds_error = False, fill_value = None)

        self.sigma_BMF = HMFintclass.sigmaRintlog((np.log(self._rb_array), self._z_array)).T #might need to make a new interpolator for different R range
        self.zr_BMF = [self.zlist, np.log(self.Rs_BMF)]
        self.sigma_int = RegularGridInterpolator(self.zr_BMF, self.sigma_BMF, bounds_error = False, fill_value = None)
        
        self.trec0 = 1/(constants.alphaB * cosmology.n_H(CosmoParams,0) * AstroParams.clumping) #seconds
        self.trec = self.trec0/(1+self.zlist)**3/constants.yrTos #years
        self.trec_int = interp1d(self.zlist, self.trec, bounds_error = False, fill_value = None)
        
        self.niondot_avg = SFRD_Init.niondot_avg_II ### TODO maybe not store them as attributes?
        self.niondot_avg_int = interp1d(self.zlist, self.niondot_avg, bounds_error = False, fill_value = None)

        #solving Madau equation to get an initial xHII
        self.ion_frac = np.fmin(1, self.Madau_Q(CosmoParams, self.zlist))
        self.ion_frac_initial = np.copy(self.ion_frac)

        zr_mesh = np.meshgrid(np.arange(len(self.Rs)), np.arange(len(self.zlist)))
        self.nion_norm = self.nion_normalization(zr_mesh[1], zr_mesh[0])
        self.nion_norm_int = RegularGridInterpolator(self.zr, self.nion_norm, bounds_error = False, fill_value = None)

        #using initial xHII to compute initial barrier
        self.prebarrier_xHII = np.empty((len(self.ds_array), len(self.zlist), len(self.Rs)))
        self.barrier = self.compute_barrier(CosmoParams, AstroParams, self.ion_frac, self.zlist, self.Rs)
        self.barrier_initial = np.copy(self.barrier)
        self.barrier_int = RegularGridInterpolator(self.zr, self.barrier, bounds_error = False, fill_value = None)

        self.dzr = [self.ds_array, self.zlist, np.log(self.Rs)]
        self.prebarrier_xHII_int = RegularGridInterpolator(self.dzr, self.prebarrier_xHII, bounds_error = False, fill_value = None) #allow extrapolation

        self.R_linear_sigma_fit_idx = z21_utilities.find_nearest_idx(self.Rs_BMF, AstroParams.R_linear_sigma_fit_input)[0]
        self.R_linear_sigma_fit = self.Rs_BMF[self.R_linear_sigma_fit_idx]

        #fake bubble mass function to impose peak around R_linear_sigma_fit for the initial linear barriers
        #looks something like [0, 0, ..., 1, ..., 0, 0]*(number of redshifts)
        self.BMF = np.repeat([np.eye(len(self.Rs_BMF))[self.R_linear_sigma_fit_idx]], len(self.zlist), axis=0)

        self.peakRofz = np.array([self.BMF_peak_R(z) for z in self.zlist])
        #ensuring that the peak bubble size is monotonically increasing with time.
        self.peakRofz = self.monotonic_after_peak(self.peakRofz)
        self.peakRofz_int = interp1d(self.zlist, self.peakRofz, bounds_error = False, fill_value = None)

        #first computation of BMF using the initial guesses
        self.BMF = self.VRdn_dR(self.zlist, self.Rs_BMF)
        self.BMF_initial = np.copy(self.BMF)

        #xHII from analytic integral of BMF
        self.ion_frac = np.nan_to_num(self.analytic_Q(CosmoParams, self.zlist)) 
        
        #ensure that if the barrier is negative at the largest smoothing scale (i.e. the whole box is ionized), then xHII is 1.
        self.ion_frac[self.barrier[:, -1]<=0] = 1
        
        #converge the BMF interatively
        if AstroParams.FLAG_BMF_converge:
            self.converge_BMF(CosmoParams, AstroParams, self.ion_frac)
        

    def compute_prebarrier_xHII(self, CosmoParams, ion_frac, z, R):
        """
        Computes the ionized fraction before solving for the barrier.

        Parameters
        ----------
        CosmoParams : CosmoParams class
        ion_frac : array
            Global xHII at z.
        z : array
            Redshifts.
        R : array
            Smoothing radii in cMpc.

        Returns
        -------
        prebarrier_xHII : array
            Ionized fraction evaluated over sample delta, z, and R before solving for the barrier.
        """
        nion_values = self.nion_delta_r_int(CosmoParams, z, R)  #Shape (nd, nz, nR)
        nrec_values = self.nrec(CosmoParams, ion_frac, z)[:, :, None]       #Shape (nd, nz) * (1, 1, nR)
        
        prebarrier_xHII = nion_values / (1 + nrec_values)

        return prebarrier_xHII
    
    def compute_barrier(self, CosmoParams, AstroParams, ion_frac, z, R):
        """
        Computes the density barrier threshold for ionization by finding when nion surpasses nH+nrec.

        Parameters
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        ion_frac : array
            Global xHII at z.
        z : array
            Redshifts.
        R : array
            Smoothing radii in cMpc.

        Returns
        -------
        barrier : array
            Density threshold for ionization as a function of z and R.
        """
        zarg = np.argsort(z)
        z = z[zarg]
        ion_frac = ion_frac[zarg]
    
        self.prebarrier_xHII = self.compute_prebarrier_xHII(
            CosmoParams, ion_frac, z, R
        )
    
        total_values = np.log10(self.prebarrier_xHII + 1e-10)
        # Expected shape: (len(self.ds_array), len(self.zlist), len(R))
    
        crosses = np.diff(np.sign(total_values), axis=0) != 0
        # Shape: (len(self.ds_array) - 1, len(self.zlist), len(R))
    
        #first time nion surpasses nH+nrec along the delta axis
        has_crossing = crosses.any(axis=0)
        first_idx = np.argmax(crosses, axis=0)
        # Shape: (len(self.zlist), len(R))
    
        y0 = np.take_along_axis(
            total_values[:-1, :, :],
            first_idx[None, :, :],
            axis=0,
        )[0]
    
        y1 = np.take_along_axis(
            total_values[1:, :, :],
            first_idx[None, :, :],
            axis=0,
        )[0]
    
        x0 = self.ds_array[first_idx]
        x1 = self.ds_array[first_idx + 1]
    
        #interpolate between grid values to find more precise barrier.
        with np.errstate(divide="ignore", invalid="ignore"):
            barrier = x0 - y0 * (x1 - x0) / (y1 - y0)
    
        barrier = np.where(has_crossing, barrier, np.nan)
    
        growth = (
            CosmoParams.growthint(self.zlist)
            / CosmoParams.growthint(self.zlist[0])
        )
    
        #the barrier is defined in terms of the overdensity linearly extrapolated to zlist[0], so we need to multiply by the growth factor to get the correct barrier at each redshift.
        barrier = barrier * growth[:, None]
    
        #the barrier is an unreachable number above zmax, ensuring that there are no bubbles.
        barrier[self.zlist > AstroParams.ZMAX_REION] = 100
    
        return barrier

    def nion_normalization(self, z, R):
        """
        Computes the normalization factor for the niondot fit.

        Parameters
        ----------
        z : int or array
            Redshift-grid index or indices.
        R : int or array
            Radius-grid index or indices.

        Returns
        -------
        nion_norm : float or array
            Normalization for niondot.
        """
        return 1/np.sqrt(1-2*self.gamma2[z, R]*self.sigma[z, R]**2)*np.exp(self.gamma[z, R]**2 * self.sigma[z, R]**2 / (2-4*self.gamma2[z, R]*self.sigma[z, R]**2))

    def nrec(self, CosmoParams, ion_frac, z, d_array=None):
        """
        Vectorized computation of nrec over an array of overdensities d_array.

        Parameters
        ----------
        CosmoParams: CosmoParams class
            Stores cosmology.
        ion_frac: 1D np.array
            The ionized fraction over all redshifts.
        z: array
            Redshifts
        d_array : array, optional 
            Sample delta values. 
            Default is None, in which case self.ds_array is used.

        Returns
        ----------
        nrecs: 2D np.array
            The total number of recombinations at each overdensity for a certain ionized fraction history at each redshift. The first dimension is densities, the second dimension is redshifts.
        """
        zarg = np.argsort(z) #sort just in case
        z = z[zarg]
        ion_frac = ion_frac[zarg]

        if d_array is None:
            d_array = self.ds_array
        
        #reverse the inputs to make the integral easier to compute
        z_rev = z[::-1]
        Hz_rev = cosmology.Hubinvyr(CosmoParams, z_rev)
        trec_rev = self.trec_int(z_rev)
        ion_frac_rev = ion_frac[::-1]
    
        denom = -1 / (1 + z_rev) / Hz_rev / trec_rev
        integrand_base = denom * ion_frac_rev 
        Dg = CosmoParams.growthint(z_rev) #growth factor

        nrecs = cumulative_trapezoid(integrand_base*(1+d_array[:, np.newaxis]*Dg/Dg[-1]), x=z_rev, initial=0) #(1+delta) rather than (1+delta)^2 because nrec and nion are per hydrogen atom 
        
        #TODO: nonlinear recombinations/higher order

        nrecs = nrecs[:, ::-1]  #reverse back to increasing z order
        return nrecs
    
    def niondot_delta_r(self, CosmoParams, z, R, d_array=None):
        """
        Compute niondot over an array of overdensities d_array for a given R.

        Parameters
        ----------
        CosmoParams: CosmoParams class
            Stores cosmology.
        z: array
            Redshifts
        R: float
            Radius value (cMpc)
        d_array : array, optional 
            Sample delta values. 
            Default is None, in which case self.ds_array is used.

        Output
        ----------
        niondot: 2D np.array
            The rates of ionizing photon production. The first dimension is densities, the second dimension is redshifts.
        """

        z1d = np.copy(z)
        R1d = np.copy(R)
        
        z = z[None, :, None]
        R = R[None, None, :]

        if d_array is None:
            d_array = self.ds_array[:, None, None]

        d_array = d_array * CosmoParams.growthint(z) / CosmoParams.growthint(z1d[0])
    
        gamma = self.gamma_zR_int(z1d[:, None], R1d[None, :])[None, :, :]
        gamma2 = self.gamma2_zR_int(z1d[:, None], R1d[None, :])[None, :, :]
        nion_norm = self.nion_norm_zR_int(z1d[:, None], R1d[None, :])[None, :, :]
    
        #niondot fit with gammas and normalization
        exp_term = np.exp(gamma * d_array + gamma2 * d_array**2)
        niondot = (self.niondot_avg_int(z) / nion_norm) * exp_term
        
        return niondot
    
    def nion_delta_r_int(self, CosmoParams, z, R, d_array=None):
        """
        Vectorized computation of nion over an array of overdensities d_array for a given R.

        Parameters
        ----------
        CosmoParams: CosmoParams class
            Stores cosmology.
        z: array
            Redshifts
        R: float
            Radius value (cMpc)
        d_array : array, optional 
            Sample delta values. 
            Default is None, in which case self.ds_array is used.

        Output
        ----------
        nion: 2D np.array
            The total number of ionizing photons produced since z=zmax. The first dimension is densities, the second dimension is redshifts.
        """

        z.sort() #sort if not sorted

        if d_array is None:
            d_array = self.ds_array[:, None, None]
        
        #reverse the inputs to make the integral easier to compute
        z_rev = z[::-1]
        Hz_rev = cosmology.Hubinvyr(CosmoParams, z_rev)
    
        niondot_values = self.niondot_delta_r(CosmoParams, z, R, d_array)
    
        #cumulatively integrate over all time
        integrand = -1 / (1 + z_rev[None, :, None]) / Hz_rev[None, :, None] * niondot_values[:, ::-1]
        nion = cumulative_trapezoid(integrand, x=z_rev, initial=0, axis=1)[:, ::-1] #reverse back to increasing z order
        
        return nion

    def Madau_Q(self, CosmoParams, z):
        """ 
        Computes the global ionized fraction by solving the Madau equation. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        z : float or array 
            Redshifts 
            
        Returns 
        ------- 
        Q : float or array 
            Global xHII evaluated at z. 
        """

        z = np.atleast_1d(z) #accepts scalar or array
        z_arr = np.geomspace(z, self.zlist[-1], len(self.zlist))
        dtdz = 1/cosmology.Hubinvyr(CosmoParams, z_arr)/(1 + z_arr)
        tau0 = self.trec0 * np.sqrt(CosmoParams.OmegaM) * cosmology.Hubinvyr(CosmoParams, 0) / constants.yrTos
        exp = np.exp(2/3/tau0 * (np.power(1 + z, 3/2) - np.power(1 + z_arr, 3/2))) #switched order around to be correct (typo in paper)

        niondot_avgs = self.niondot_avg_int(z_arr)
        integrand = dtdz * niondot_avgs * exp
    
        return np.trapezoid(integrand, x = z_arr, axis = 0)

    def B_1(self, z):
        """ 
        Computes the slope term of the linear ionization barrier. 
        
        Parameters 
        ---------- 
        z : float or array 
            Redshifts 
            
        Returns 
        ------- 
        B1 : array 
            Slope term of the linear barrier as a function of z. 
        """
        z = np.atleast_1d(z)
        #compute slope near the peak of the BMF
        R_pivot = self.peakRofz_int(z)
        sigmax = np.diagonal(self.sigma_zR_int(z[:, None], (R_pivot*1.1)[None, :]))
        sigmin = np.diagonal(self.sigma_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        barriermax = np.diagonal(self.barrier_zR_int(z[:, None], (R_pivot*1.1)[None, :]))
        barriermin = np.diagonal(self.barrier_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        return (barriermax - barriermin)/(sigmax**2 - sigmin**2)
        
    def B_0(self, z):
        """ 
        Computes the y-intercept term of the linear ionization barrier. 
        
        Parameters 
        ---------- 
        z : float or array 
            Redshifts 
            
        Returns 
        ------- 
        B0 : array 
            Intercept term of the linear barrier as a function of z. 
        """
        z = np.atleast_1d(z)
        R_pivot = self.peakRofz_int(z)
        sigmin = np.diagonal(self.sigma_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        barriermin = np.diagonal(self.barrier_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        return barriermin - sigmin**2 * self.B_1(z)
    
    def B(self, z, R, sig=None):
        """ 
        Computes the linear ionization barrier as a function of z and R. 
        
        Parameters 
        ---------- 
        z : float or array 
            Redshifts 
        R : float or array
            Radii in cMpc
        sig : array, optional 
            Matter fluctuation sigma(z, R). 
            Default is None, in which case sigma is interpolated internally.
            
        Returns 
        ------- 
        B : array 
            Linear barrier as a function of z and R. 
        """
        z = np.atleast_1d(z)
        if sig is None:
            R = np.atleast_1d(R)
            sig = self.sigma_zR_int(z[:, None], R[None, :])
        B0 = self.B_0(z)
        B1 = self.B_1(z)
        return B0[:, None] + B1[:, None]*sig**2
    
    def dlogsigma_dlogR(self, z, R, sig):
        """ 
        Computes dlogsigma/dlogR. 
        
        Parameters 
        ---------- 
        z : float or array 
            Redshifts 
        R : array 
            Bubble radii in cMpc. 
        sig : array 
            Matter fluctuation sigma(z, R). 
            
        Returns
        ------- 
        dlogsigma_dlogR : array 
            Logarithmic derivative dlog(sigma)/dlog(R). 
            """
        return np.gradient(np.log(sig), np.log(R), axis=1)
    
    def VRdn_dR(self, z, R):
        """ 
        Computes the volume-weighted BMF. Integrating this quantity gives the global xHII. 
        
        Parameters 
        ---------- 
        z : float or array 
            Redshifts 
        R : array 
            Bubble radii in cMpc. 
            
        Returns 
        ------- 
        VRdn_dR : array 
            Volume-weighted BMF evaluated over redshift and radius. 
        """
        z = np.atleast_1d(z)
        sig = self.sigma_zR_int(z[:, None], R[None, :])
        B0 = self.B_0(z)
        B1 = self.B_1(z)
        return np.sqrt(2/np.pi) * np.abs(self.dlogsigma_dlogR(z, R, sig)) * np.abs(B0[:, None])/sig * np.exp(-(B0[:, None]+B1[:, None]*sig**2)**2/2/sig**2)
    
    def Rdn_dR(self, z, R):
        """ 
        Computes the number-weighted BMF. 
        
        Parameters 
        ---------- 
        z : float or array 
            Redshifts 
        R : array 
            Bubble radii in cMpc. 
            
        Returns 
        ------- 
        Rdn_dR : array 
            Number-weighted bubble mass function evaluated over redshift and radius. 
        """
        return self.VRdn_dR(z, R)*3/(4*np.pi*R[None, :]**3)

    def BMF_peak_R(self, z, fit_window=5, max_bubble=100, min_bubble=0.5):
        """ 
        Finds the radius at which the BMF peaks. 
        
        Parameters 
        ---------- 
        z : float 
            Redshift 
        fit_window : int, optional 
            Number of points on each side of the coarse peak to use for the spline fit. Default is 5. 
        max_bubble : float, optional 
            Maximum allowed bubble radius in cMpc. Default is 100. 
        min_bubble : float, optional 
            Minimum allowed bubble radius in cMpc. Default is 0.5. 
            
        Returns 
        ------- 
        peak_R : float 
            Peak bubble radius in cMpc. 
        """
        min_bubble = np.max([self.Rs[1], min_bubble])

        iz = z21_utilities.find_nearest_idx(self.zlist, z)[0]

        R = self.Rs_BMF
        y = self.BMF[iz]

        #keep only finite positive values
        good = np.isfinite(y) & (y > 0) & np.isfinite(R)
        if not np.any(good):
            return min_bubble

        R_good = R[good]
        y_good = y[good]

        #coarse peak index
        ir_peak = np.argmax(y_good)

        #fit spline around the peak to get more precise.
        #find where derivative = 0. If too close to edge, return bounds.
        i_lo = max(0, ir_peak - fit_window)
        i_hi = min(len(R_good), ir_peak + fit_window + 1)

        R_window = R_good[i_lo:i_hi]
        y_window = y_good[i_lo:i_hi]

        if len(R_window) < 5:
            return np.clip(R_good[ir_peak], min_bubble, max_bubble)

        x = np.log(R_window)
        ly = np.log(y_window)

        #get more exact fit between the points using a spline.
        spline_fit = UnivariateSpline(x, ly, k=4, s=0)
        roots = spline_fit.derivative().roots()

        d2 = spline_fit.derivative(n=2)

        valid_roots = [
            root for root in roots
            if x[0] <= root <= x[-1] and d2(root) < 0
        ]

        if len(valid_roots) == 0:
            peak_R = R_good[ir_peak]
        else:
            x_peak_guess = np.log(R_good[ir_peak])
            x_peak = valid_roots[np.argmin(np.abs(np.array(valid_roots) - x_peak_guess))]
            peak_R = np.exp(x_peak)

        #return peak within allowed bounds
        return np.clip(peak_R, min_bubble, max_bubble)

    def monotonic_after_peak(self, x):
        """ 
        Enforces monotonic growth of the bubble size peak with time. 
        
        Parameters 
        ---------- 
        x : array 
            Input array. 
            
        Returns 
        ------- 
        x : array 
            Array with values left of the peak forced to be non-decreasing.
        """
        x = np.asarray(x).copy()
        i_peak = np.nanargmax(x)

        x[i_peak:] = np.minimum.accumulate(x[i_peak:])

        return x

    def analytic_Q(self, CosmoParams, z): 
        """ 
        Analytically integrate the BMF to compute global xHII. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        z : float or array 
            Redshifts 
            
        Returns 
        ------- 
        Q : array 
            Ionized fraction obtained from the analytic integral of the BMF. 
        """
        z = np.atleast_1d(z)
        Rmin = self.Rs_BMF[0] #fixing analytic to fit numeric (old version: 1e-10, arbitrarily small)
        B0 = self.B_0(z)
        B1 = self.B_1(z)
        sigmin = CosmoParams.ClassCosmo.sigma(Rmin, z[0])*CosmoParams.growthint(z)/CosmoParams.growthint(z[0]) ### Faster to multiply sigma by the growth but there is a 0.2% error on the xHII_avg
        ### TODO maybe add a flag to call ClassCosmo.sigma for every z
        s2 = sigmin**2
        return 0.5*np.exp(-2*B0*B1)*erfc((B0-B1*s2)/np.sqrt(2*s2)) + 0.5*erfc((B0+B1*s2)/np.sqrt(2*s2))

    def converge_BMF(self, CosmoParams, AstroParams, ion_frac_input):
        """ 
        Iteratively updates the ionization barrier, BMF, and xHII until convergence. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        AstroParams : AstroParams class 
        ion_frac_input : array 
            Initial ionized fraction used to begin the BMF convergence loop. 
            
        Returns 
        ------- 
        None 
        """
        self.ion_frac = ion_frac_input
        iterator = trange(AstroParams.max_iter) if self.PRINT_SUCCESS else range(AstroParams.max_iter)
        for j in iterator:
            ion_frac_prev = np.copy(self.ion_frac)
            #need xHII for recombination calculation, which affects the barrier.
            self.barrier = self.compute_barrier(CosmoParams, AstroParams, self.ion_frac, self.zlist, self.Rs)
            self.barrier_int = RegularGridInterpolator(self.zr, self.barrier, bounds_error = False, fill_value = None)
            
            #update BMF and peaks.
            self.BMF = self.VRdn_dR(self.zlist, self.Rs_BMF)
            self.peakRofz = np.array([self.BMF_peak_R(z) for z in self.zlist])
            self.peakRofz = self.monotonic_after_peak(self.peakRofz)
            self.peakRofz_int = interp1d(self.zlist, self.peakRofz, bounds_error = False, fill_value = None)

            #update xHII
            self.ion_frac = np.nan_to_num(self.analytic_Q(CosmoParams, self.zlist))
            self.ion_frac[self.barrier[:, -1]<=0] = 1

            #stop the loop if xHII hasn't changed much between iterations.
            if np.allclose(ion_frac_prev, self.ion_frac, rtol=1e-1, atol=1e-2):
                if self.PRINT_SUCCESS:
                    print(f"SUCCESS: BMF converged after {j+1} iteration{'s' if j > 0 else ''}.")
                return 
            
        print(f"WARNING: BMF didn't converge within {AstroParams.max_iter} iterations.")


    #interpolators in z and R used in reionization.py
    def interpR(self, z, R, func):
        "Interpolator to find func(z, R), designed to take a single z but an array of R in cMpc"
        _logR = np.log(R)
        logRvec = np.asarray([_logR]) if np.isscalar(_logR) else np.asarray(_logR)
        inarray = np.array([[z, LR] for LR in logRvec])
        return func(inarray)

    def interpz(self, z, R, func):
        "Interpolator to find func(z, R), designed to take a single R in cMpc but an array of z"
        zvec = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
        inarray = np.array([[zz, np.log(R)] for zz in zvec])
        return func(inarray)

    #all instances of different (z, R) interpolators, named explicitly for clarity in the code
    def sigmaR_int(self, z, R):
        return self.interpR(z, R, self.sigma_int)
    def sigmaz_int(self, z, R):
        return self.interpz(z, R, self.sigma_int)

    def barrierR_int(self, z, R):
        return self.interpR(z, R, self.barrier_int)
    def barrierz_int(self, z, R):
        return self.interpz(z, R, self.barrier_int)

    def gammaR_int(self, z, R):
        return self.interpR(z, R, self.gamma_int)
    def gammaz_int(self, z, R):
        return self.interpz(z, R, self.gamma_int)

    def gamma2R_int(self, z, R):
        return self.interpR(z, R, self.gamma2_int)
    def gamma2z_int(self, z, R):
        return self.interpz(z, R, self.gamma2_int)

    def nion_normR_int(self, z, R):
        return self.interpR(z, R, self.nion_norm_int)
    def nion_normz_int(self, z, R):
        return self.interpz(z, R, self.nion_norm_int)

    def interp_zR(self, z, R, func):
        """
        Evaluate a RegularGridInterpolator defined on (z, logR).
    
        Accepts scalar, 1D, 2D, or ND z and R.
        z and R are broadcast against each other.
    
        Examples
        --------
        scalar z, vector R:
            out.shape == R.shape
    
        vector z, scalar R:
            out.shape == z.shape
    
        z[:, None], R[None, :]:
            out.shape == (nz, nR)
        """
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)
    
        z_b, R_b = np.broadcast_arrays(z, R)
    
        points = np.column_stack([
            z_b.ravel(),
            np.log(R_b).ravel()
        ])
    
        out = func(points)
        return out.reshape(z_b.shape)

    def sigma_zR_int(self, z, R):
        return self.interp_zR(z, R, self.sigma_int)
    
    def barrier_zR_int(self, z, R):
        return self.interp_zR(z, R, self.barrier_int)
    
    def gamma_zR_int(self, z, R):
        return self.interp_zR(z, R, self.gamma_int)
    
    def gamma2_zR_int(self, z, R):
        return self.interp_zR(z, R, self.gamma2_int)
    
    def nion_norm_zR_int(self, z, R):
        return self.interp_zR(z, R, self.nion_norm_int)

    def prebarrier_xHII_int_grid(self, d, z, R):
        """
        Evaluate prebarrier xHII on a density field d(x),
        at fixed redshift z and smoothing radius R.

        Parameters
        ----------
        d: np.ndarray
            Density/overdensity field. Can be any shape (...).
        z: float
            Redshift.
        R: float
            Smoothing radius (cMpc).

        Output
        ----------
        values: np.ndarray
            xHII field with the same shape as d.
        """
        
        d = np.asarray(d, dtype=float)

        z_arr   = np.full_like(d, float(z), dtype=float)
        logr_arr = np.full_like(d, np.log(float(R)), dtype=float)

        #stack into points (..., 3) where last axis is (delta, z, logR)
        points = np.stack([d, z_arr, logr_arr], axis=-1)

        values = self.prebarrier_xHII_int(points)

        return values