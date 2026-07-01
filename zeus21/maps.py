"""
Make maps! For fun and science.

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

from . import cosmology
from . import z21_utilities
from . import inputs
from . import T21coefficients
from . import correlations

import numpy as np
import powerbox as pbox
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from tqdm import trange
import time
from dataclasses import dataclass, field as _field, InitVar
import copy 


@dataclass(kw_only=True)
class ReioMapsConfig:
    """
    All arguments of reionization_maps that have default values
    """
    input_boxlength: float = 300.
    ncells: int = 300
    seed: int = 1234
    r_precision: float = 1.
    Rs: list | np.ndarray | None = None
    barrier: np.ndarray = None
    PRINT_TIMER: bool = True
    LOGNORMAL_DENSITY: bool = False
    COMPUTE_DENSITY_AT_ALLZ: bool = False
    COMPUTE_MASSWEIGHTED: bool = False
    lowres_massweighting: int = 1
    COMPUTE_PARTIAL_IONIZATIONS: bool = False
    COMPUTE_ZREION: bool = False



class reionization_maps:
    """
    Generates 3D maps of the reionization fields.
    
    Uses a density threshold barrier determined from a converged bubble mass function. With default parameters, the code takes about 20 minutes on laptop to run.

    Parameters
    ----------
    CosmoParams: zeus21.Cosmo_Parameters class
        Stores cosmology.
    CoeffStructure: zeus21.get_T21_coefficients class
        Stores sfrd and 21cm coefficients.
    input_z: 1D np.array
        The redshifts at which to compute output maps. Narrowed down later to select available redshifts from CoeffStructure.zintegral.
    input_boxlength: float
        Comoving physical side length of the box. Default is 300 cMpc.
    ncells: int
        Number of cells on a side. Default is 300 cells.
    seed: int
        Sets the predetermined generation of maps. Default is 1234.
    r_precision: float
        Allows to change the steps of the radii for faster computation. Default (and max) is 1, lower values make the computation faster at the cost of accuracy.
    barrier: function
        Input density barrier to be used as the threshold for map generation. Takes z value as input and returns np.array of shape. Default is None.
    PRINT_TIMER: bool
        Whether to print the time elapsed along the process. Default is True.
    LOGNORMAL_DENSITY: bool
        Whether to use lognormal (True) or Gaussian (False) density fields. Default is False.
    COMPUTE_DENSITY_AT_ALLZ: bool
        Whether to output the density field at all redshifts. If False, only the density at the lower input redshift is computed. If True, the computation time and memory usage dramatically increases. Default is False.
    COMPUTE_MASSWEIGHTED: bool
        Whether to compute the mass weighted ionized field and fraction. If True, COMPUTE_DENSITY_AT_ALLZ will be forced to True, thus increasing computation time dramatically. Default is False.
    lowres_massweighting: int
        Compute the mass-weighted ionized field and fraction more efficiently by using lower resolution density and ionized fields. Has to be >=1 and an integer. Default is 1.
    COMPUTE_PARTIAL_IONIZATIONS: bool
        Whether to compute the subpixel ionizations in the field and the ionized fractions.

    Attributes
    ----------
    dx: float
        Cell resolution of a side of the boxes.
    z: 1D np.array
        Redshifts at which the output maps are computed. Selected to be the closest to the input redshifts from the available ones in zeus21.
    r: 1D np.array
        Radii at which the density field is smoothed. Selected using r_precision from the available ones in zeus21.
    z_of_density: float
        Redshift at which the density is computed.
    density: 3D np.array
        Overdensity field at the lowest redshift asked by the user.
    density_allz: 4D np.array
        Overdensity field at all the redshifts asked by the user. First dimension correponds to redshifts. Only computed if COMPUTE_DENSITY_AT_ALLZ is True.
    ion_field_allz: 4D np.array
        Ionized fraction field at all the redshifts asked by the user. First dimension correponds to redshifts.
    ion_frac: 1D np.array
        Volume weighted ionized fraction at all the redshifts asked by the user.
    ion_frac_massweighted: 1D np.array
        Mass weighted ionized fraction at all the redshifts asked by the user. Only computed if COMPUTE_MASSWEIGHTED is True.
    """
    
    def __init__(self, CosmoParams, CoeffStructure, input_z, 
                 input_boxlength=300., ncells=300, seed=1234, r_precision=1., Rs=None, barrier=None, 
                 PRINT_TIMER=True, 
                 LOGNORMAL_DENSITY=False, COMPUTE_DENSITY_AT_ALLZ=False,
                 COMPUTE_MASSWEIGHTED=False, lowres_massweighting=1, COMPUTE_PARTIAL_IONIZATIONS=False, 
                 COMPUTE_ZREION=False
                ):
        #Measure time elapsed from start
        self._start_time = time.time()
        
        ### boxes parameters
        self.input_z = input_z
        self.ncells = ncells
        self.boxlength = input_boxlength
        self.dx = self.boxlength/self.ncells

        # radii
        if Rs is None:
            default_len = len(CosmoParams._Rtabsmoo)
            self.r_precision = r_precision
            self.r = np.logspace(np.log10(self.dx * (3/4/np.pi)**(1/3)), np.log10(self.boxlength), int(default_len*self.r_precision))
            self._r_idx = np.arange(int(default_len*self.r_precision))
        else:
            self.r_precision = r_precision
            self.r = Rs
            if self.r_precision > 1:
                raise ValueError('r_precision cannot be greater than 1 if you input your own radii.')
            self._r_idx = np.floor(np.arange(len(self.r), step=self.r_precision)).astype(int)
            smallest_r = self.dx * (3/4/np.pi)**(1/3)
            if self.r[0] < smallest_r:
                print(f'WARNING: Your input radii are too small for the pixel size. The code will still run now.\nIn the future, for best performance and physical accuracy on this boxlength and ncells, the smallest smoothing radius should be no less than R=L/N * (4pi/3)^(-1/3), or approximately {smallest_r:.2f} cMpc.')
        
        self.seed = seed

        ### FLAGS
        self.PRINT_TIMER = PRINT_TIMER
        self.LOGNORMAL_DENSITY = LOGNORMAL_DENSITY
        self.COMPUTE_DENSITY_AT_ALLZ = COMPUTE_DENSITY_AT_ALLZ
        self._has_density = COMPUTE_DENSITY_AT_ALLZ
        self.COMPUTE_MASSWEIGHTED = COMPUTE_MASSWEIGHTED
        self.COMPUTE_PARTIAL_IONIZATIONS = COMPUTE_PARTIAL_IONIZATIONS
        self.COMPUTE_PARTIAL_AND_MASSWEIGHTED = False
        if self.COMPUTE_MASSWEIGHTED and self.COMPUTE_PARTIAL_IONIZATIONS:
            self.COMPUTE_PARTIAL_AND_MASSWEIGHTED = True
        self.COMPUTE_ZREION = COMPUTE_ZREION
        if self.COMPUTE_MASSWEIGHTED or self.COMPUTE_PARTIAL_IONIZATIONS or self.COMPUTE_PARTIAL_AND_MASSWEIGHTED:
            self.COMPUTE_DENSITY_AT_ALLZ = True

        ### selecting redshifts and radii from available redshifts
        # redshifts
        self._z_idx = np.arange(len(np.atleast_1d(input_z))) #z21_utilities.find_nearest_idx(CoeffStructure.zintegral, self.input_z)
        self.z = np.atleast_1d(input_z) #CoeffStructure.zintegral[self._z_idx]

        ### generating the density field at the closest redshift to the lower one inputed
        self.z_of_density = self.z[0]
        self.density = self.generate_density(CosmoParams)
        self.sig_corr = self.sigma_correction(CosmoParams)
        self.density /= self.sig_corr #non-ergodicity correction

        ### smoothing the density field
        self._k = self.compute_k()
        self.density_smoothed_allr = self.smooth_density()

        ### evolving density
        self.density_allz = np.empty((len(self.z), self.ncells, self.ncells, self.ncells), dtype=np.float32)
        if self.COMPUTE_DENSITY_AT_ALLZ:
            self.generate_density_allz(CosmoParams)

        ### generating the ionized field, and computing the ionized fraction
        self.barrier = barrier
        if self.barrier is None:
            self.barrier = CoeffStructure.B(self.z, self.r) #BMF linear barrier
        self.ion_field_allz, self.ion_frac = self.generate_xHII(CosmoParams)

        ### computing the mass weighted ionized fraction

        self._has_mw = False
        self.lowres_massweighting = lowres_massweighting
        if self.COMPUTE_MASSWEIGHTED:
            self.compute_massweighted(CosmoParams, self.lowres_massweighting)

        self._has_p = False
        if self.COMPUTE_PARTIAL_IONIZATIONS:
            self.compute_partial(CosmoParams, CoeffStructure)

        self._has_mwp = False
        if self.COMPUTE_PARTIAL_AND_MASSWEIGHTED:
            self.compute_partial_massweighted(CosmoParams, CoeffStructure)

        if self.COMPUTE_ZREION:
            self.zreion = self.compute_zreion_frombinaryxHII()
            self.treion = self.compute_treion(CosmoParams)
            
        
        if self.PRINT_TIMER:
            z21_utilities.print_timer(self._start_time, text_before="Total computation time: ")
        

    def generate_density(self, CosmoParams):
        """ 
        Generates the initial density field at the lowest redshift. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        
        Returns 
        ------- 
        density_field : array 
            Three-dimensional delta field evaluated at self.z_of_density. 
        """
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Generating density field...")
        #generating matter power spectrum at the lowest redshift
        klist = CosmoParams._klistCF
        pk_matter = np.zeros_like(klist)
        for i, k in enumerate(klist):
            pk_matter[i] = CosmoParams.ClassCosmo.pk(k, self.z_of_density)
        pk_spl = spline(np.log(klist), np.log(pk_matter))
    
        #generating density map
        if self.LOGNORMAL_DENSITY:
            pb = pbox.LogNormalPowerBox(N=self.ncells, dim=3, pk=(lambda k: np.exp(pk_spl(np.log(k)))), boxlength=self.boxlength, seed=self.seed)
        else:
            pb = pbox.PowerBox(N=self.ncells, dim=3, pk=(lambda k: np.exp(pk_spl(np.log(k)))), boxlength=self.boxlength, seed=self.seed)
        density_field = pb.delta_x().astype(np.float32, copy=False)
        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")
        return density_field

    def generate_density_allz(self, CosmoParams):
        """ 
        Evolves the density field to all redshifts using the linear growth factor. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        
        Returns 
        ------- 
        density_allz : array 
            Four-dimensional delta field. The first dimension is z. 
        """
        if self.PRINT_TIMER:
            start_time = time.time()
            print('Evolving density field...')
        Dg = CosmoParams.growthint(self.z)
        growthfactor_ratio = (Dg/Dg[0])[:, np.newaxis, np.newaxis, np.newaxis]
        density_lastz = np.copy(self.density)
        self.density_allz = density_lastz[np.newaxis]*growthfactor_ratio
        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")

        self._has_density = True
            
        return self.density_allz

    def compute_k(self):
        """ 
        Computes the Fourier-space wavenumber grid for the box. 
        
        Returns 
        ------- 
        k : array 
            Three-dimensional array of wavenumber magnitudes. 
        """
        klistfftx = np.fft.fftfreq(self.ncells,self.dx)*2*np.pi
        k = np.sqrt(np.sum(np.meshgrid(klistfftx**2, klistfftx**2, klistfftx**2, indexing='ij'), axis=0))
        return k

    def smooth_density(self):
        """ 
        Smooths the density field over all smoothing radii. 
        
        Returns 
        ------- 
        density_smoothed_allr : array 
            Density field smoothed at each radius in self.r. 
        """
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Smoothing density field...")

        #smooth by FFT convolution with a tophat
        density_fft = np.fft.fftn(self.density)
        density_smoothed_allr = np.array([z21_utilities.tophat_smooth(rr, self._k, density_fft) for rr in self.r])
        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")
        return density_smoothed_allr

    def sigma_correction(self, CosmoParams):
        """ 
        Computes the non-ergodicity correction to the generated density variance. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        
        Returns 
        ------- 
        sigma_ratio : float 
            Ratio between the measured and theoretical sigma. 
        """
        sigma_ratio = np.std(self.density)/CosmoParams.ClassCosmo.sigma(self.r[0], self.z_of_density)
        return sigma_ratio

    def generate_xHII(self, CosmoParams):
        """ 
        Generates ionized fraction fields and volume-weighted ionized fractions. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        
        Returns 
        ------- 
        ion_field_allz : array 
            Ionized fraction field at each redshift. 
        ion_frac : array 
            Volume-weighted ionized fraction at each redshift. 
        """
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Generating ionized field...")
        ion_field_allz = np.zeros((len(self.z),self.ncells,self.ncells,self.ncells))
        ion_frac = np.zeros(len(self.z))

        #choose iterator based on if the user wants to print progress or not.
        iterator = trange(len(self.z)) if self.PRINT_TIMER else range(len(self.z))
        
        for i in iterator:
            curr_z_idx = self._z_idx[i]
            ion_field = self.ionize(CosmoParams, curr_z_idx)
            ion_field_allz[i] = ion_field
            ion_frac[i] = np.sum(ion_field)/(self.ncells**3)
        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")
        return ion_field_allz, ion_frac

    def ionize(self,CosmoParams, curr_z_idx):
        """ 
        Computes the binary ionized field at a single redshift. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        curr_z_idx : int 
            Index of the redshift at which to compute the ionized field. 
            
        Returns 
        ------- 
        ion_field : array 
            Binary ionized field at curr_z. 
        """
        Dg0 = CosmoParams.growthint(self.z[0])
        Dg = CosmoParams.growthint(self.z[curr_z_idx])
        Dg0_Dg = Dg0/Dg
        ion_field = np.any(self.density_smoothed_allr > (Dg0_Dg)*self.barrier[curr_z_idx, self._r_idx][:, None, None, None], axis=0)

        #Earlier versions of this code contained a spherize method in addition to this central pixel flagging, where spheres are ionized instead of just the central pixel. 
        #We found that central pixel flagging is generally more consistent with the bubble mass function than spherizing, so future versions will not include this.
        
        return ion_field
    
    def compute_massweighted(self, CosmoParams, lowres_massweighting=1):
        """ 
        Computes the mass-weighted ionized field and ionized fraction. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class
        lowres_massweighting : int, optional 
            Factor by which to downsample the density and ionized fields for mass weighting. Default is 1. 
            
        Returns 
        ------- 
        ion_frac_massweighted : array 
            Mass-weighted ionized fraction at each redshift. 
        ion_field_massweighted_allz : array 
            Mass-weighted ionized field at each redshift. 
        """
        if not self._has_mw:
            self.ion_frac_massweighted = np.empty(len(self.z))
            self.ion_field_massweighted_allz = np.empty_like(self.ion_field_allz)
        if not self._has_density:
            self.generate_density_allz(CosmoParams)
        self.lowres_massweighting = lowres_massweighting
        if self.lowres_massweighting < 1:
            raise Exception('lowres_massweighting should be >=1.')
        if not isinstance(self.lowres_massweighting, (int, np.int32, np.int64)):
            raise Exception('lowres_massweighting should be an integer.')
        d_allz = self.density_allz[:, ::self.lowres_massweighting, ::self.lowres_massweighting, ::self.lowres_massweighting]
        ion_allz = self.ion_field_allz[:, ::self.lowres_massweighting, ::self.lowres_massweighting, ::self.lowres_massweighting]
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Computing mass-weighted field...")
        #where the magic happens
        self.ion_field_massweighted_allz = (1+d_allz) * ion_allz
        if self.PRINT_TIMER:
            print("Computing mass-weighted ionized fraction...")
        self.ion_frac_massweighted = np.average(self.ion_field_massweighted_allz, axis=(1, 2, 3))
        
        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")

        self._has_mw = True
        
        return self.ion_frac_massweighted, self.ion_field_massweighted_allz

    def compute_partial(self, CosmoParams, CoeffStructure, r=None):
        """ 
        Computes the partially ionized field and volume-weighted partially ionized fraction. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        CoeffStructure : get_T21_coefficients class
        r : float, optional 
            Smoothing radius in cMpc used to evaluate the prebarrier ionized fraction. Default is None, in which case self.r[0] is used. 
            
        Returns 
        ------- 
        ion_frac_partial : array 
            Volume-weighted partially ionized fraction at each redshift. 
        ion_field_partial_allz : array 
            Partially ionized field at each redshift. 
        """
        if r is None:
            r = self.r[0]
        if not self._has_p:
            self.ion_frac_partial = np.empty(len(self.z))
            self.ion_field_partial_allz = np.empty_like(self.ion_field_allz)
        if not self._has_density:
            self.generate_density_allz(CosmoParams)
        sample_d = np.linspace(-5, 5, 51)

        if self.PRINT_TIMER:
            start_time = time.time()
            print("Computing partially ionized field...")

        #loop over each z.
        out_shape = self.density.shape
        iterator = trange(len(self.z)) if self.PRINT_TIMER else range(len(self.z))
        for i in iterator:
            #evaluate sample grid, and then input the actual density field into an interpolator.
            tempgrid = CoeffStructure.prebarrier_xHII_int_grid(sample_d, self.z[i], r)
            
            partialfield = np.interp(self.density.ravel(), sample_d, tempgrid).reshape(out_shape)
            
            np.abs(partialfield, out=partialfield)#abs just in case, beacuse negative numbers here are unphysical
            #add partials to binaries and then clip to 1.
            np.add(self.ion_field_allz[i], partialfield, out=self.ion_field_partial_allz[i])
            np.clip(self.ion_field_partial_allz[i], 0, 1, out=self.ion_field_partial_allz[i])

        if self.PRINT_TIMER:
            print("Computing partial ionized fraction...")

        self.ion_frac_partial = np.average(self.ion_field_partial_allz, axis=(1, 2, 3))

        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")

        self._has_p = True
            
        return self.ion_frac_partial, self.ion_field_partial_allz

    def compute_partial_massweighted(self, CosmoParams, CoeffStructure, r=None):
        """ 
        Computes the mass-weighted partially ionized field and fraction. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        CoeffStructure : get_T21_coefficients class 
        r : float, optional 
            Smoothing radius in cMpc used to evaluate the prebarrier ionized fraction. Default is None, in which case self.r[0] is used. 
            
        Returns 
        ------- 
        ion_frac_partial_massweighted : array 
            Mass-weighted partially ionized fraction at each redshift. 
        ion_field_partial_massweighted_allz : array 
            Mass-weighted partially ionized field at each redshift. 
        """
        if not self._has_p:
            self.compute_partial(CosmoParams, CoeffStructure, r)

        if not self._has_mwp:
            self.ion_frac_partial_massweighted = np.empty(len(self.z))
            self.ion_field_partial_massweighted_allz = np.empty_like(self.ion_field_allz)

        if self.PRINT_TIMER:
            start_time = time.time()
            print("Computing mass-weighted partially ionized field...")

        #where the magic happens
        iterator = trange(len(self.z)) if self.PRINT_TIMER else range(len(self.z))
        for i in iterator:
            self.ion_field_partial_massweighted_allz[i] = (1+self.density_allz[i]) * self.ion_field_partial_allz[i]
        
        if self.PRINT_TIMER:
            print("Computing mass-weighted partial ionized fraction...")

        iterator = trange(len(self.z)) if self.PRINT_TIMER else range(len(self.z))
        for i in iterator:
            self.ion_frac_partial_massweighted[i] = np.average(self.ion_field_partial_massweighted_allz[i])
        
        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")

        self._has_mwp = True
        
        return self.ion_frac_partial_massweighted, self.ion_field_partial_massweighted_allz

    def compute_zreion_frombinaryxHII(self):
        """ 
        Computes the redshift-of-reionization map from the binary ionized fraction field. 
        
        Returns 
        ------- 
        zreion : array 
            Three-dimensional map of the z at which each cell is first ionized. 
        """
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Computing zreion map...")

        vectorized_zlist = np.vectorize(lambda iz: self.z[iz])
        zreion = vectorized_zlist(np.argmin(self.ion_field_allz,axis=0)-1).reshape((self.ncells,self.ncells,self.ncells))

        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")
        return zreion
    
    def compute_treion(self,CosmoParams):
        """ 
        Computes the time-of-reionization map from the redshift-of-reionization map. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        
        Returns 
        ------- 
        treion : array 
            Three-dimensional map of the time at which each cell becomes ionized. 
        """
        if self.PRINT_TIMER:
            start_time = time.time()
            print("Computing treion map...")

        treion = cosmology.time_at_redshift(CosmoParams.ClassCosmo,self.zreion)

        if self.PRINT_TIMER:
            z21_utilities.print_timer(start_time, text_before="    done in ")
        return treion
    
    def _compute_ionfrac_from_zreion(self):
        """
        Way to compute the volume ionized fraction from zreion. Currently not used but there if needed.
        """
        zvalues = np.unique(self.zreion)
        neutfrac = np.zeros(len(zvalues))
        for i in range(len(zvalues)):
            neutfrac[i] = np.sum(self.zreion<zvalues[i]) / self.ncells**3
        return 1-neutfrac, zvalues

    def _compute_ionfrac_from_treion(self):
        """
        Way to compute the volume ionized fraction from treion. Currently not used but there if needed.
        """
        tvalues = np.unique(self.treion)
        neutfrac = np.zeros(len(tvalues))
        for i in range(len(tvalues)):
            neutfrac[i] = np.sum(self.treion>tvalues[i]) / self.ncells**3
        return 1-neutfrac, tvalues
        

@dataclass()
class T21_maps:
    # arguments to pass
    CosmoParams: InitVar[inputs.Cosmo_Parameters]
    CoeffStructure: InitVar[T21coefficients.get_T21_coefficients]
    PowerSpectra: InitVar[correlations.Power_Spectra]
    input_z: np.ndarray

    # reionization
    ReioMaps_config: ReioMapsConfig = _field(default_factory=ReioMapsConfig)
    ReioMaps: reionization_maps = _field(init=False)

    # flag
    USE_xHII_MAPS: bool = _field(default=True)

    # box params
    input_boxlength: float = _field(default=300.)
    ncells: int = _field(default=300)
    seed: int = _field(default=1234)
    input_Resolution: float = _field(default=0.5)

    # boxes
    density: np.ndarray = _field(init=False)
    xHI: np.ndarray = _field(init=False)
    T21_lin: np.ndarray = _field(init=False)
    T21_NL: np.ndarray = _field(init=False)
    T21: np.ndarray = _field(init=False)
    smooth_box: bool = _field(default=False)
    xHI_smooth: np.ndarray = _field(init=False)
    T21_smooth: np.ndarray = _field(init=False)

    # other attributes
    _klist: np.ndarray = _field(init=False)
    _k3over2pi2: np.ndarray = _field(init=False)
    T21avg: np.ndarray = _field(init=False)
    _Dsq_T21_lin: np.ndarray = _field(init=False)
    _Dsq_T21: np.ndarray = _field(init=False)
    _PdT21: np.ndarray = _field(init=False)
    _Pd: np.ndarray = _field(init=False)


    def __post_init__(self, CosmoParams, CoeffStructure, PowerSpectra):
        """ 
        Generates 21cm maps. 
        
        Parameters 
        ---------- 
        CosmoParams : CosmoParams class 
        CoeffStructure : get_T21_coefficients class 
        PowerSpectra : Power_Spectra class
        
        Returns 
        ------- 
        None 
        """

        ### z and k
        _iz = z21_utilities.find_nearest_idx(CoeffStructure.zlist, self.input_z)
        self._klist = PowerSpectra.klist_PS
        self._k3over2pi2 = self._klist**3/(2*np.pi**2)

        ### get T21 avg
        if self.USE_xHII_MAPS:
            # in this case, we will use the simulated xHI with reionization_maps
            # so, we need to remove the xHI contribution from T21avg
            self.T21avg = (CoeffStructure.T21avg / (CoeffStructure.xHI_avg + 1e-15))[_iz]
        else:
            self.T21avg = CoeffStructure.T21avg[_iz]
        
        ### get power spectra
        self._Dsq_T21_lin = ((PowerSpectra.Deltasq_T21_lin[_iz].T / CoeffStructure.T21avg[_iz]**2) * self.T21avg**2).T
        self._Dsq_T21 = ((PowerSpectra.Deltasq_T21[_iz].T / CoeffStructure.T21avg[_iz]**2) * self.T21avg**2).T
        self._PdT21 = (PowerSpectra.Deltasq_dT21[_iz]/CoeffStructure.T21avg[_iz])* self.T21avg/self._k3over2pi2
        self._Pd = (PowerSpectra.Deltasq_d_lin[_iz,:])/self._k3over2pi2

        ### generate densities
        self.density, pbs = self.generate_density_pb()


        ### map of the linear T21 fluctuation, better to use the cross to keep sign, at linear level same
        self.T21_lin = self.generate_T21_lin(pbs)

        ### map of the nonlinear correction
        # built as \sum_R [e^(gR dR) - gR dR]. Uncorrelatd with all dR so just a separate field!
        # NOTE: its not guaranteed to work, excess power can be negative in some cases! Not for each component xa, Tk, but yes for T21
        self.T21_NL = self.generate_T21_NL()

        ### add T21 lin and nonlin correction together
        self.T21 = self.T21_lin + self.T21_NL

        if self.USE_xHII_MAPS:
            ### generate xHII
            self.ReioMaps_config.input_boxlength = self.input_boxlength
            self.ReioMaps_config.ncells = self.ncells
            self.ReioMaps_config.seed = self.seed
            self.ReioMaps = reionization_maps(CosmoParams, CoeffStructure, self.input_z, **vars(self.ReioMaps_config))

            ### include ionization
            if self.ReioMaps_config.COMPUTE_PARTIAL_AND_MASSWEIGHTED:
                   self.xHI = (1. - self.ReioMaps.ion_field_massweighted_allz)
            else:
                if self.ReioMaps_config.COMPUTE_PARTIAL_IONIZATIONS:
                    self.xHI = (1. - self.ReioMaps.ion_field_partial_allz)
                else:
                    self.xHI = (1. - self.ReioMaps.ion_field_allz)
        
            self.T21 *= self.xHI

        self.T21[np.isnan(self.T21)] = 0.

        if self.smooth_box:
            Resolution = max(self.input_Resolution, self.input_boxlength/self.ncells)
            if self.USE_xHII_MAPS:
                self.xHI_smooth = z21_utilities.smooth_box(self.xHI, Resolution, self.input_boxlength, self.ncells)
            self.T21_smooth = z21_utilities.smooth_box(self.T21[0], Resolution, self.input_boxlength, self.ncells)
    

    def generate_density_pb(self):
        """ 
        Generates density fields using PowerBox. 
        
        Returns 
        ------- 
        density : array 
            Density field at each redshift. 
        pbs : list 
            PowerBox objects that generate the density fields. 
        """
        density = np.zeros((len(self.input_z),self.ncells,self.ncells,self.ncells))
        pbs = []
        for iz, z in enumerate(self.input_z):
            Pd_spl = spline(np.log(self._klist), np.log(self._Pd[iz])) # density at min z
            pb = pbox.PowerBox(
                N=self.ncells,
                dim=3,
                pk = lambda k: np.exp(Pd_spl(np.log(k))),
                boxlength = self.input_boxlength,
                seed = self.seed
            )
            density[iz] = pb.delta_x()
            pbs.append(pb)
        return density, pbs

    def generate_T21_lin(self, pbs):
        """ 
        Generates the linear 21cm temperature field. 
        
        Parameters 
        ---------- 
        pbs : list 
            PowerBox objects. 
            
        Returns 
        ------- 
        T21_lin : array 
            Linear 21cm brightness temperature field at each redshift. 
        """
        T21_lin = np.zeros((len(self.input_z),self.ncells,self.ncells,self.ncells))
        for iz, z in enumerate(self.input_z):
            pb = pbs[iz]
            powerratio_spl = spline(self._klist, self._PdT21[iz]/self._Pd[iz]) #cross can be negative, so can't interpolate over log values
            powerratio = powerratio_spl(pb.k())
            T21lin_k = powerratio * pb.delta_k()
            T21_lin[iz] = self.T21avg[iz] + z21_utilities.powerboxCtoR(pb, mapkin = T21lin_k)
            pbs.append(pb)
        
        return T21_lin

    def generate_T21_NL(self):
        """ 
        Generates the nonlinear correction to the 21cm temperature field. 
        
        Returns 
        ------- 
        T21_NL : array 
            Nonlinear 21cm brightness temperature correction field at each redshift. 
        """
        T21_NL = np.zeros((len(self.input_z),self.ncells,self.ncells,self.ncells))
        for iz, z in enumerate(self.input_z):
            excesspower21 = (self._Dsq_T21[iz]-self._Dsq_T21_lin[iz])/self._k3over2pi2
            lognormpower = interp1d(self._klist, excesspower21/self.T21avg[iz]**2, fill_value=0.0, bounds_error=False)
            pbe = pbox.LogNormalPowerBox(         #G or logG? TODO revisit
                N=self.ncells,
                dim=3,
                pk = lambda k: lognormpower(k),
                boxlength = self.input_boxlength,
                seed = self.seed+1                # uncorrelated
            )
            T21_NL[iz] = self.T21avg[iz] * pbe.delta_x()
        return T21_NL
