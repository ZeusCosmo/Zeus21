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
from tqdm import trange


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
        
        self.gamma = CoeffStructure.gamma_niondot_II_index2D
        self.gamma2 = CoeffStructure.gamma2_niondot_II_index2D
        self.sigma = CoeffStructure.sigmaofRtab
        
        self.zr = [self.zlist, np.log(self.Rs)]
        self.gamma_int = RegularGridInterpolator(self.zr, self.gamma, bounds_error = False, fill_value = np.nan)
        self.gamma2_int = RegularGridInterpolator(self.zr, self.gamma2, bounds_error = False, fill_value = np.nan)

        self.sigma_BMF = np.array([[ClassyCosmo.sigma(r, z) for z in self.zlist] for r in self.Rs_BMF]).T
        self.zr_BMF = [self.zlist, np.log(self.Rs_BMF)]
        self.sigma_int = RegularGridInterpolator(self.zr_BMF, self.sigma_BMF, bounds_error = False, fill_value = np.nan)
        
        self.Hz = cosmology.Hubinvyr(CosmoParams, self.zlist)
        self.trec0 = 1/(constants.alphaB * cosmology.n_H(CosmoParams,0) * AstroParams._clumping) #seconds
        self.trec = self.trec0/(1+self.zlist)**3/constants.yrTos #years
        self.trec_int = interp1d(self.zlist, self.trec, bounds_error = False, fill_value = np.nan)
        
        self.niondot_avg = CoeffStructure.niondot_avg_II
        self.niondot_avg_int = interp1d(self.zlist, self.niondot_avg, bounds_error = False, fill_value = np.nan)

        self.ion_frac = np.fmin(1, [self.calc_Q(CosmoParams, z) for z in self.zlist])
        self.ion_frac_initial = np.copy(self.ion_frac)

        zr_mesh = np.meshgrid(np.arange(len(self.Rs)), np.arange(len(self.zlist)))
        self.nion_norm = self.nion_normalization(zr_mesh[1], zr_mesh[0])
        self.nion_norm_int = RegularGridInterpolator(self.zr, self.nion_norm, bounds_error = False, fill_value = np.nan)

        self.prebarrier_xHII = np.empty((len(self.ds_array), len(self.zlist), len(self.Rs)))
        self.barrier = self.compute_barrier(CosmoParams, self.ion_frac, self.zlist, self.Rs)
        self.barrier_initial = np.copy(self.barrier)
        self.barrier_int = RegularGridInterpolator(self.zr, self.barrier, bounds_error = False, fill_value = np.nan)

        self.dzr = [self.ds_array, self.zlist, np.log(self.Rs)]
        self.prebarrier_xHII_int = RegularGridInterpolator(self.dzr, self.prebarrier_xHII, bounds_error = False, fill_value = None) #allow extrapolation

        self.R_linear_sigma_fit_idx = z21_utilities.find_nearest_idx(self.Rs, R_linear_sigma_fit_input)
        self.R_linear_sigma_fit = self.Rs[self.R_linear_sigma_fit_idx]

        self.BMF = np.array([self.VRdn_dR(z, self.Rs_BMF) for z in self.zlist]) #initial bubble mass function
        self.BMF_initial = np.copy(self.BMF)
        self.ion_frac = np.nan_to_num([np.trapezoid(self.BMF[i], np.log(self.Rs_BMF)) for i in range(len(self.zlist))]) #ion_frac by integrating the BMF
        self.ion_frac[self.barrier[:, -1]<=0] = 1
        
        if FLAG_converge:
            self.converge_BMF(CosmoParams, self.ion_frac, max_iter=max_iter)
        #two functions: compute BMF and iterate
        

    def compute_prebarrier_xHII(self, CosmoParams, ion_frac, z, R):
        """
        
        """
        nion_values = self.nion_delta_r_int(CosmoParams, z, R)  #Shape (nd, nz)
        nrec_values = self.nrec(CosmoParams, ion_frac, z)       #Shape (nd, nz)
        
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
        
        for ir in range(len(R)):
            #Compute nion_values and nrec_values for this 'ir'
            self.prebarrier_xHII[:, :, ir] =  self.compute_prebarrier_xHII(CosmoParams, ion_frac, z, R[ir])
            total_values = np.log10(self.prebarrier_xHII[:, :, ir] + 1e-10)
        
            #Loop over redshift indices
            for iz in range(len(self.zlist)):
                y_values = total_values[:, iz]  #Shape (nd,)
        
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
        barrier = barrier * (CosmoParams.growthint(self.zlist)/CosmoParams.growthint(self.zlist[0]))[:, np.newaxis] #scale barrier with growth factor
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

        zarg = np.argsort(z) #sort just in case
        z = z[zarg]

        if d_array is None:
            d_array = self.ds_array
        
        d_array = d_array[:, np.newaxis] * CosmoParams.growthint(z)[np.newaxis, :] / CosmoParams.growthint(z[0])
    
        gamma_R = self.gammaz_int(z, R)   
        gamma2_R = self.gamma2z_int(z, R)  
        nion_norm_R = self.nion_normz_int(z, R)
    
        exp_term = np.exp(gamma_R[np.newaxis, :] * d_array + gamma2_R[np.newaxis, :] * d_array**2)
        niondot = (self.niondot_avg_int(z)[np.newaxis, :] / nion_norm_R[np.newaxis, :]) * exp_term
        
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
            d_array = self.ds_array
        
        #reverse the inputs to make the integral easier to compute
        z_rev = z[::-1]
        Hz_rev = cosmology.Hubinvyr(CosmoParams, z_rev)
    
        niondot_values = self.niondot_delta_r(CosmoParams, z, R, d_array)
    
        integrand = -1 / (1 + z_rev) / Hz_rev * niondot_values[:, ::-1]
        nion = cumulative_trapezoid(integrand, x=z_rev, initial=0)[:, ::-1] #reverse back to increasing z order
        
        return nion

    #calculating ionized fraction
    def calc_Q(self, CosmoParams, z):
        z_arr = np.logspace(np.log10(z), np.log10(self.zlist[-1]), 50)
        dtdz = 1/cosmology.Hubinvyr(CosmoParams, z_arr)/(1 + z_arr)
        tau0 = self.trec0 * np.sqrt(CosmoParams.OmegaM) * cosmology.Hubinvyr(CosmoParams, 0) / constants.yrTos
        exp = np.exp(2/3/tau0 * (np.power(1 + z, 3/2) - np.power(1 + z_arr, 3/2))) #switched order around to be correct (typo in paper)

        niondot_avgs = self.niondot_avg_int(z_arr)
        integrand = dtdz * niondot_avgs * exp
    
        return np.trapezoid(integrand, x = z_arr)

    #computing linear barrier
    def B_1(self, z):
        sigmax = self.sigmaR_int(z, self.Rs[self.R_linear_sigma_fit_idx+1])
        sigmin = self.sigmaR_int(z, self.Rs[self.R_linear_sigma_fit_idx-1])
        barriermax = self.barrierR_int(z, self.Rs[self.R_linear_sigma_fit_idx+1])
        barriermin = self.barrierR_int(z, self.Rs[self.R_linear_sigma_fit_idx-1])
        return (barriermax - barriermin)/(sigmax**2 - sigmin**2)
        
    def B_0(self, z):
        sigmin = self.sigmaR_int(z, self.Rs[self.R_linear_sigma_fit_idx-1])
        barriermin = self.barrierR_int(z, self.Rs[self.R_linear_sigma_fit_idx-1])
        return barriermin - sigmin**2 * self.B_1(z)
    
    def B(self, z, R): 
        sig = self.sigmaR_int(z, R)
        return self.B_0(z) + self.B_1(z)*sig**2
    
    #computing other terms in the BMF
    def dsigma_dR(self, z, R):
        sigma = self.sigmaR_int(z, R)
        return sigma/R*np.gradient(np.log(sigma), np.log(R))
    
    def dlogsigma_dlogR(self, z, R):
        sigma = self.sigmaR_int(z, R)
        return self.dsigma_dR(z, R) * R/sigma
    
    def VRdn_dR(self, z, R):
        sig = self.sigmaR_int(z, R)
        return np.sqrt(2/np.pi) * np.abs(self.dlogsigma_dlogR(z, R)) * np.abs(self.B_0(z))/sig * np.exp(-self.B(z, R)**2/2/sig**2)
    
    def Rdn_dR(self, z, R):
        return self.VRdn_dR(z, R)*3/(4*np.pi*R**3)

    def converge_BMF(self, CosmoParams, ion_frac_input, max_iter):
        self.ion_frac = ion_frac_input
        iterator = trange(max_iter) if self.PRINT_SUCCESS else range(max_iter)
        for j in iterator:
            ion_frac_prev = np.copy(self.ion_frac)
            
            self.barrier = self.compute_barrier(CosmoParams, self.ion_frac, self.zlist, self.Rs)
            self.barrier_int = RegularGridInterpolator(self.zr, self.barrier, bounds_error = False, fill_value = np.nan)

            def barrierR_int(self, z, R):
                return self.interpR(z, R, self.barrier_int)
            def barrierz_int(self, z, R):
                return self.interpz(z, R, self.barrier_int)
            
            self.BMF = np.array([self.VRdn_dR(z, self.Rs_BMF) for z in self.zlist])
            self.ion_frac = np.nan_to_num([np.trapezoid(self.BMF[i], np.log(self.Rs_BMF)) for i in range(len(self.zlist))])
            self.ion_frac[self.barrier[:, -1]<=0] = 1

            if np.allclose(ion_frac_prev, self.ion_frac):
                if self.PRINT_SUCCESS:
                    print(f'SUCCESS: BMF converged in {j} iterations.')
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

    def barrierR_int(self, z, R): #unused
        return self.interpR(z, R, self.barrier_int)
    def barrierz_int(self, z, R): #unused
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