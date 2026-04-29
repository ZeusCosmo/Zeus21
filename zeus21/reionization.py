"""

Models reionization using an analogy of a halo mass function to ionized bubbles 
See Sklansky et al. (in prep)

Authors: Yonatan Sklansky, Emilie Thelie
UT Austin - October 2025

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

import time


class BMF:
    """
    Computes the bubble mass function (BMF). 

    
    """
    def __init__(self, CoeffStructure, HMFintclass, CosmoParams, AstroParams, ClassyCosmo, R_linear_sigma_fit_input=10, FLAG_converge=True, max_iter=10, ZMAX_REION = 30, Rmin=0.05, PRINT_SUCCESS=True):

        self.PRINT_SUCCESS = PRINT_SUCCESS
        self.ZMAX_REION = ZMAX_REION #max redshift up to which we calculate reionization observables
        self.zlist = CoeffStructure.zintegral
        self.Rs = CoeffStructure.Rtabsmoo
        self.Rs_BMF = np.logspace(np.log10(Rmin), np.log10(self.Rs[-1]), 100)
        self.ds_array = np.linspace(-1, 5, 101)

        self._r_array, self._z_array = np.meshgrid(self.Rs, self.zlist, sparse=True, indexing='ij')
        self._rb_array, self._z_array = np.meshgrid(self.Rs_BMF, self.zlist, sparse=True, indexing='ij')
        
        self.gamma = CoeffStructure.gamma_niondot_II_index2D
        self.gamma2 = CoeffStructure.gamma2_niondot_II_index2D
        #self.sigma = np.array([[ClassyCosmo.sigma(r, z) for z in self.zlist] for r in self.Rs]).T#CoeffStructure.sigmaofRtab
        self.sigma =  HMFintclass.sigmaRintlog((np.log(self._r_array), self._z_array)).T
        
        self.zr = [self.zlist, np.log(self.Rs)]
        self.gamma_int = RegularGridInterpolator(self.zr, self.gamma, bounds_error = False, fill_value = None)
        self.gamma2_int = RegularGridInterpolator(self.zr, self.gamma2, bounds_error = False, fill_value = None)

        self.sigma_BMF = HMFintclass.sigmaRintlog((np.log(self._rb_array), self._z_array)).T #might need to make a new interpolator for different R range
        self.zr_BMF = [self.zlist, np.log(self.Rs_BMF)]
        self.sigma_int = RegularGridInterpolator(self.zr_BMF, self.sigma_BMF, bounds_error = False, fill_value = None)
        
        self.Hz = cosmology.Hubinvyr(CosmoParams, self.zlist)
        self.trec0 = 1/(constants.alphaB * cosmology.n_H(CosmoParams,0) * AstroParams._clumping) #seconds
        self.trec = self.trec0/(1+self.zlist)**3/constants.yrTos #years
        self.trec_int = interp1d(self.zlist, self.trec, bounds_error = False, fill_value = None)
        
        self.niondot_avg = CoeffStructure.niondot_avg_II
        self.niondot_avg_int = interp1d(self.zlist, self.niondot_avg, bounds_error = False, fill_value = None)

        self.ion_frac = np.fmin(1, self.Madau_Q(CosmoParams, self.zlist))
        self.ion_frac_initial = np.copy(self.ion_frac)

        zr_mesh = np.meshgrid(np.arange(len(self.Rs)), np.arange(len(self.zlist)))
        self.nion_norm = self.nion_normalization(zr_mesh[1], zr_mesh[0])
        self.nion_norm_int = RegularGridInterpolator(self.zr, self.nion_norm, bounds_error = False, fill_value = None)

        self.prebarrier_xHII = np.empty((len(self.ds_array), len(self.zlist), len(self.Rs)))
        self.barrier = self.compute_barrier(CosmoParams, self.ion_frac, self.zlist, self.Rs)
        self.barrier_initial = np.copy(self.barrier)
        self.barrier_int = RegularGridInterpolator(self.zr, self.barrier, bounds_error = False, fill_value = None)

        self.dzr = [self.ds_array, self.zlist, np.log(self.Rs)]
        self.prebarrier_xHII_int = RegularGridInterpolator(self.dzr, self.prebarrier_xHII, bounds_error = False, fill_value = None) #allow extrapolation

        self.R_linear_sigma_fit_idx = z21_utilities.find_nearest_idx(self.Rs, R_linear_sigma_fit_input)[0]
        self.R_linear_sigma_fit = self.Rs[self.R_linear_sigma_fit_idx]

        #fake bubble mass function to impose peak around R_linear_sigma_fit for the initial linear barriers
        #looks something like [0, 0, ..., 1, ..., 0, 0]*(number of redshifts)
        self.BMF = np.repeat([np.eye(len(self.Rs_BMF))[self.R_linear_sigma_fit_idx]], len(self.zlist), axis=0)

        self.peakRofz = np.array([self.BMF_peak_R(z) for z in self.zlist])
        self.peakRofz_int = interp1d(self.zlist, self.peakRofz, bounds_error = False, fill_value = None)

        #second computation of BMF using the initial guess peaks
        self.BMF = self.VRdn_dR(self.zlist, self.Rs_BMF)
        self.BMF_initial = np.copy(self.BMF)
        
        #self.ion_frac = np.nan_to_num([np.trapezoid(self.BMF[i], np.log(self.Rs_BMF)) for i in range(len(self.zlist))]) #ion_frac by numerically integrating the BMF
        self.ion_frac = np.nan_to_num(self.analytic_Q(CosmoParams, ClassyCosmo, self.zlist)) #ion_frac by analytic integral of BMF
        
        self.ion_frac[self.barrier[:, -1]<=0] = 1
        
        if FLAG_converge:
            self.converge_BMF(CosmoParams, ClassyCosmo, self.ion_frac, max_iter=max_iter)
        #two functions: compute BMF and iterate
        

    def compute_prebarrier_xHII(self, CosmoParams, ion_frac, z, R):
        """
        
        """
        nion_values = self.nion_delta_r_int(CosmoParams, z, R)  #Shape (nd, nz, nR)
        nrec_values = self.nrec(CosmoParams, ion_frac, z)[:, :, None]       #Shape (nd, nz) * (1, 1, nR)
        
        prebarrier_xHII = nion_values / (1 + nrec_values)

        return prebarrier_xHII

    def compute_barrier(self, CosmoParams, ion_frac, z, R):
        """
        Computes the density barrier threshold for ionization.
        
        Using the analytic model from Sklansky et al. (in prep), if the total number of ionized photons produced in an overdensity exceeds the sum of the number of hydrogens present and total number of recombinations occurred, then the overdensity is ionized. The density required to ionized is recorded.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        ion_frac: 1D np.array
            The ionized fractions to be used to compute the number of recombinations. 

        Output
        ----------
        barrier: 2D np.array
            The resultant density threshold array. First dimension is each redshift, second dimension is each radius scale.
        """
        barrier = np.zeros((len(z), len(R)))

        zarg = np.argsort(z) #sort just in case
        z = z[zarg]
        ion_frac = ion_frac[zarg]

        #Compute nion_values and nrec_values based on (re)computed ion_frac
        self.prebarrier_xHII =  self.compute_prebarrier_xHII(CosmoParams, ion_frac, z, R)
        total_values = np.log10(self.prebarrier_xHII + 1e-10)
        
        for ir in range(len(R)):
            #Loop over redshift indices
            for iz in range(len(self.zlist)):
                y_values = total_values[:, iz, ir]  #Shape (nd,)
        
                #Find zero crossings
                sign_change = np.diff(np.sign(y_values))
                idx = np.where(sign_change)[0]
                if idx.size > 0:
                    #Linear interpolation to find zero crossings
                    x0 = self.ds_array[idx]
                    x1 = self.ds_array[idx + 1]
                    y0 = y_values[idx]
                    y1 = y_values[idx + 1]
                    x_intersect = x0 - y0 * (x1 - x0) / (y1 - y0)
                    barrier[iz, ir] = x_intersect[0]  #Assuming we take the first crossing
                else:
                    barrier[iz, ir] = np.nan #Never crosses
        barrier = barrier * (CosmoParams.growthint(self.zlist)/CosmoParams.growthint(self.zlist[0]))[:, None] #scale barrier with growth factor
        barrier[self.zlist > self.ZMAX_REION] = 100 #sets density to an unreachable barrier, as if reionization isn't happening
        return barrier

    #normalizing the nion/sfrd model
    def nion_normalization(self, z, R):
        return 1/np.sqrt(1-2*self.gamma2[z, R]*self.sigma[z, R]**2)*np.exp(self.gamma[z, R]**2 * self.sigma[z, R]**2 / (2-4*self.gamma2[z, R]*self.sigma[z, R]**2))

    def nrec(self, CosmoParams, ion_frac, z, d_array=None):
        """
        Vectorized computation of nrec over an array of overdensities d_array.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        d_array: 1D np.array
            A list of sample overdensity values to evaluate nrec over.
        ion_frac: 1D np.array
            The ionized fraction over all redshifts.

        Output
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
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        d_array: 1D np.array
            A list of sample overdensity values to evaluate niondot over.
        R: float
            Radius value (cMpc)

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
    
        exp_term = np.exp(gamma * d_array + gamma2 * d_array**2)
        niondot = (self.niondot_avg_int(z) / nion_norm) * exp_term
        
        return niondot
    
    def nion_delta_r_int(self, CosmoParams, z, R, d_array=None):
        """
        Vectorized computation of nion over an array of overdensities d_array for a given R.

        Parameters
        ----------
        CosmoParams: zeus21.Cosmo_Parameters class
            Stores cosmology.
        d_array: 1D np.array
            A list of sample overdensity values to evaluate niondot over.
        R: float
            Radius value (cMpc)

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
    
        integrand = -1 / (1 + z_rev[None, :, None]) / Hz_rev[None, :, None] * niondot_values[:, ::-1]
        nion = cumulative_trapezoid(integrand, x=z_rev, initial=0, axis=1)[:, ::-1] #reverse back to increasing z order
        
        return nion

    #calculating naive ionized fraction
    def Madau_Q(self, CosmoParams, z):
        z = np.atleast_1d(z) #accepts scalar or array
        z_arr = np.geomspace(z, self.zlist[-1], len(self.zlist))
        dtdz = 1/cosmology.Hubinvyr(CosmoParams, z_arr)/(1 + z_arr)
        tau0 = self.trec0 * np.sqrt(CosmoParams.OmegaM) * cosmology.Hubinvyr(CosmoParams, 0) / constants.yrTos
        exp = np.exp(2/3/tau0 * (np.power(1 + z, 3/2) - np.power(1 + z_arr, 3/2))) #switched order around to be correct (typo in paper)

        niondot_avgs = self.niondot_avg_int(z_arr)
        integrand = dtdz * niondot_avgs * exp
    
        return np.trapezoid(integrand, x = z_arr, axis = 0)

    #computing linear barrier
    def B_1(self, z):
        R_pivot = self.peakRofz_int(z)
        sigmax = np.diagonal(self.sigma_zR_int(z[:, None], (R_pivot*1.1)[None, :]))
        sigmin = np.diagonal(self.sigma_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        barriermax = np.diagonal(self.barrier_zR_int(z[:, None], (R_pivot*1.1)[None, :]))
        barriermin = np.diagonal(self.barrier_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        return (barriermax - barriermin)/(sigmax**2 - sigmin**2)
        
    def B_0(self, z):
        R_pivot = self.peakRofz_int(z)
        sigmin = np.diagonal(self.sigma_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        barriermin = np.diagonal(self.barrier_zR_int(z[:, None], (R_pivot*0.9)[None, :]))
        return barriermin - sigmin**2 * self.B_1(z)
    
    def B(self, z, R, sig):
        B0 = self.B_0(z)
        B1 = self.B_1(z)
        return B0[:, None] + B1[:, None]*sig**2
    
    #computing other terms in the BMF
    def dlogsigma_dlogR(self, z, R, sig):
        return np.gradient(np.log(sig), np.log(R), axis=1)
    
    def VRdn_dR(self, z, R):
        z = np.atleast_1d(z)
        sig = self.sigma_zR_int(z[:, None], R[None, :])
        B0 = self.B_0(z)
        B1 = self.B_1(z)
        return np.sqrt(2/np.pi) * np.abs(self.dlogsigma_dlogR(z, R, sig)) * np.abs(B0[:, None])/sig * np.exp(-(B0[:, None]+B1[:, None]*sig**2)**2/2/sig**2)
    
    def Rdn_dR(self, z, R):
        return self.VRdn_dR(z, R)*3/(4*np.pi*R[None, :]**3)
#
#    def BMF_peak_R(self, z):
#        iz = z21_utilities.find_nearest_idx(self.zlist, z)
#        ir = np.argmax(self.BMF[iz])
#        return self.Rs_BMF[ir]

    def BMF_peak_R(self, z, fit_window=5, max_bubble=100, min_bubble = 0.2): #written and commented by Claude
        iz = z21_utilities.find_nearest_idx(self.zlist, z)[0]

        # Find the coarse peak index
        ir_peak = np.argmax(self.BMF[iz])

        # Slice a window around the peak
        i_lo = max(0, ir_peak - fit_window)
        i_hi = min(len(self.Rs_BMF), ir_peak + fit_window + 1)
        
        R_window = self.Rs_BMF[i_lo:i_hi]
        BMF_row = self.BMF[iz, :]
        BMF_window = BMF_row[i_lo:i_hi]
        
        # If the peak is within fit_window of either edge, the true peak may
        # be at the boundary — skip the spline and return the coarse peak
        peak_at_left_edge  = (ir_peak - fit_window <= 0)
        peak_at_right_edge = (ir_peak + fit_window >= len(self.Rs_BMF) - 1)
        
        if peak_at_left_edge or peak_at_right_edge:
            return np.clip(self.Rs_BMF[ir_peak], min_bubble, max_bubble)
        
        # Also guard against a window that's too small to fit a degree-4 spline
        # (need at least k+1 = 5 points)
        if len(R_window) < 5:
            return np.clip(self.Rs_BMF[ir_peak], min_bubble, max_bubble)
        
        # Fit a spline and find its maximum
        spline = UnivariateSpline(R_window, BMF_window, k=4, s=0)
        roots = spline.derivative().roots()
        
        # Keep only roots that are local maxima (second derivative < 0)
        # and lie within the window bounds
        d2 = spline.derivative(n=2)
        valid_roots = [
            r for r in roots
            if d2(r) < 0 and R_window[0] <= r <= R_window[-1]
        ]
        
        # Return the valid root closest to the coarse peak, or fall back
        if len(valid_roots) == 0:
            return np.clip(self.Rs_BMF[ir_peak], min_bubble, max_bubble)
        
        ir_peak_R = self.Rs_BMF[ir_peak]

        peak_R = valid_roots[np.argmin(np.abs(np.array(valid_roots) - ir_peak_R))]

        return np.clip(peak_R, min_bubble, max_bubble) #peak can't be outside the allowed bounds

    def analytic_Q(self, CosmoParams, ClassyCosmo, z): #analytically integrating the BMF to get Q
        z = np.atleast_1d(z)
        Rmin = 1e-10 #arbitrarily small
        B0 = self.B_0(z)
        B1 = self.B_1(z)
        sigmin = ClassyCosmo.sigma(Rmin, z[0])*CosmoParams.growthint(z)/CosmoParams.growthint(z[0])
        s2 = sigmin**2
        return 0.5*np.exp(-2*B0*B1)*erfc((B0-B1*s2)/np.sqrt(2*s2)) + 0.5*erfc((B0+B1*s2)/np.sqrt(2*s2))

    def converge_BMF(self, CosmoParams, ClassyCosmo, ion_frac_input, max_iter):
        self.ion_frac = ion_frac_input
        iterator = trange(max_iter) if self.PRINT_SUCCESS else range(max_iter)
        for j in iterator:
            ion_frac_prev = np.copy(self.ion_frac)

            self.barrier = self.compute_barrier(CosmoParams, self.ion_frac, self.zlist, self.Rs)
            self.barrier_int = RegularGridInterpolator(self.zr, self.barrier, bounds_error = False, fill_value = None)
            
            self.BMF = self.VRdn_dR(self.zlist, self.Rs_BMF)
            self.peakRofz = np.array([self.BMF_peak_R(z) for z in self.zlist])
            self.peakRofz_int = interp1d(self.zlist, self.peakRofz, bounds_error = False, fill_value = None)

            self.ion_frac = np.nan_to_num(self.analytic_Q(CosmoParams, ClassyCosmo, self.zlist))
            self.ion_frac[self.barrier[:, -1]<=0] = 1

            if np.allclose(ion_frac_prev, self.ion_frac, rtol=1e-1, atol=1e-2):
                if self.PRINT_SUCCESS:
                    print(f"SUCCESS: BMF converged after {j+1} iteration{'s' if j > 0 else ''}.")
                return 
            
        print(f"WARNING: BMF didn't converge within {max_iter} iterations.")


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