"""
Takes inputs and stores them in useful classes.

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

from . import constants
from . import z21_utilities

from dataclasses import dataclass, field as _field, InitVar
from typing import Any
import numpy as np
from classy import Class
from scipy.interpolate import interp1d
import mcfit
from scipy.integrate import cumulative_trapezoid



@dataclass(kw_only=True)
class User_Parameters:
    """
    User parameters for Zeus21.

    Calling the class without specifying any parameter will set them to their default values, but they can also be directly set when creating the class:

    >>> zeus21.User_Parameters(precisionboost=0.5)

    Parameters can also be changed afterwards:

    >>> UserParams = zeus21.User_Parameters()
    
    >>> UserParams.precisionboost = 0.5

    Parameters
    ----------
    precisionboost: float
        Make integrals take more points for boost in precision. Default is 1.0.
    dlogzint_target:
        Target number of redshift bins for the redsfhit arrays in log space. Default is 0.02.
    FLAG_FORCE_LINEAR_CF: bool
        False to do standard calculation, True to force linearization of correlation function. Default is False.
    MIN_R_NONLINEAR: float
        Minimum radius R/cMpc in which we start doing the nonlinear calculation. Default is 2.0.
        Below ~1 it will blow up because sigma > 1 eventually, and our exp(delta) approximation breaks.
        Check if you play with it and if you change Window().
    MAX_R_NONLINEAR: float
        Maximum radius R/cMpc in which we start doing the nonlinear calculation (above this it is very linear). Default is 100.0.
    FLAG_DO_DENS_NL: bool
        Whether to do the nonlinear (ie lognormal) calculation for the density field itself and its cross correlations. Default is False.
        Small (<3%) correction in dd, but non trivial (~10%) in d-xa and d-Tx
    FLAG_WF_ITERATIVE: bool
        Whether to iteratively do the WF correction as in Hirata2006. Default is True.
    zmin: float
        Minimum redshift to which we compute the T21 signals. Default is 5.0.
    DO_ONLY_GLOBAL: bool
        Whether zeus21 only runs the global T21 signal (and not fluctuations). Default is False.
    USE_BARYON_FLAG: bool
        Whether zeus21 computes 21-cm power spectra with (1+delta_b) prefactor instead of (1+delta)
        This means LSS terms use P_baryon(k) and P_baryon_x_cdm(k). Default is True.

    Attributes
    ----------
    C2_RENORMALIZATION_FLAG: bool
        Whether to renormalize the C2 oefficients (appendix in 2302.08506). Default is True.
    """

    precisionboost: float = 1.0
    dlogzint_target: float = 0.02
    FLAG_FORCE_LINEAR_CF: bool = False
    MIN_R_NONLINEAR: float = 2.0
    MAX_R_NONLINEAR: float = 100.0
    FLAG_DO_DENS_NL: bool = False
    FLAG_WF_ITERATIVE: bool = True
    zmin: float = 5.
    DO_ONLY_GLOBAL: bool = False
    USE_BARYON_FLAG: bool = True

    C2_RENORMALIZATION_FLAG: int = _field(init=False)

    def __post_init__(self):
        schema = {
            "FLAG_FORCE_LINEAR_CF": (bool, None),
            "FLAG_DO_DENS_NL": (bool, None),
            "FLAG_WF_ITERATIVE": (bool, None),
            "DO_ONLY_GLOBAL": (bool, None),
        }
        validate_fields(self, schema)

        self.C2_RENORMALIZATION_FLAG = not self.FLAG_FORCE_LINEAR_CF


@dataclass(kw_only=True)
class Cosmo_Parameters:
    """
    Cosmological parameters for zeus21.
    This class also runs and saves an instance of CLASS.
    
    Parameters
    ----------
        UserParams: User_Parameters
            zeus21 class for the user parameters. Default is the default instance of the User_Parameters class.
        omegab: float
            Baryon density * h^2. Default is 0.0223828.
        omegac: float
            CDM density * h^2. Default is 0.1201075.
        h_fid: float
            Hubble constant / 100. Default is 0.67810.
        As: float
            Amplitude of initial fluctuations. Default is 2.100549e-09.
        ns: float
            Spectral index. Default is 0.9660499.
        tau_fid: float
            Optical depth to reionization. Default is 0.05430842.
        kmax_CLASS: float
            Maximum wavenumber to be passed to CLASS. Default is 500.0.
        zmax_CLASS: float
            Maximum redshift to be passed to CLASS. Default is 50.0.
        zmin_CLASS: float
            Minimum redshift to be passed to CLASS. Default is 5.0.
        Rs_min: float
            Minimum radius to be passed to CLASS. Default is 0.05.
            Set to 0.929 when Flag_emulate_21cmfast is True.
        Rs_max: float
            Maximum radius to be passed to CLASS. Default is 2000.0.
            Set to 500 when Flag_emulate_21cmfast is True.
        Flag_emulate_21cmfast: bool
            Whether zeus21 emulates 21cmFAST cosmology (used in HMF, LyA, and X-ray opacity calculations). Default is False.
            When False, sets the Star Formation Rate model to GALLUMI-like, and when True to 21cmfast-like (ignores Mc and beta and has a t* later in SFR()).
        USE_RELATIVE_VELOCITIES: bool
            Whether to use v_cb. Default is False.
        HMF_CHOICE: str
            Which HMF to use. Default is "ST".
            "ST" for the classic  Sheth-Tormen (f(nu)), "Yung" for the Tinker08 (f(sigma)) calibrated to Yung+23.
        _Mhmin: float
            Minimum halo mass in Msun to compute the HMF. Default is 1e5.
        _Mhmax: float
            Maximum halo mass in Msun to compute the HMF. Default is 1e14.
        
    Attributes
    ----------
    ClassCosmo: Class
        CLASS instance to compute cosmology.
        It is set with the 6 LCDM parameters set in Cosmo_Parameters.
    omegam: float
        Matter density * h^2. Default is 0.1424903.
    OmegaM: float
        Matter density. Default is 0.3098830430481206.
    rhocrit: float
        Critical density. Default is 127339073085.43648.
    OmegaR: float
        Radiation density. Default is 9.096145657179167e-05.
    OmegaL: float
        Dark energy density. Default is 0.6900259954953076.
    OmegaB: float
        Baryon density. Default is 0.048677349798108865.
    rho_M0: float
        Actual matter density. Default is 39460219466.64208.
    z_rec: float
        Recombination reshift. Default is 1088.7722850526861.
    sigma_vcb: float
        Square root of the variance of the relative velocity field. Default is 1.
    vcb_avg: float
        Average of the relative velocity field. Default is 0.0.
    Y_He: float
        Helium mass fraction. Default is 0.24527956117097657.
    x_He:
        Helium-to-hydrogen number density ratio. Default is 0.08124848240215174.
    f_H: float
        Hydrogen number density ratio relative to baryons. Default is 0.924856789420276.
    f_He: float
        Helium number density ratio relative to baryons. Default is 0.07514321057972385.
    mu_baryon: float
        Mean baryonic weight. Default is 1.149786421719843.
    mu_baryon_Msun: float
        Mean baryonic weight relative to the solar mass. Default is 1.0305080308672013e-57.
    constRM: float
        Radius-to-mass conversions for HMF. Used for CLASS input so assumes tophat. Default is 165290580780.5916.
    zfofRint: interp1d
        Interpolation for the redshift as a function of the comoving distance.
    chiofzint: interp1d
        Interpolation for the comoving distance as a function of the redshift.
    Tadiabaticint:
        Interpolation for the adiabatic temperature as a function of redshift.
    xetanhint: interp1d
        Interpolation for the electron fraction as a function of redshift.
    growthint: interp1d
        Interpolation for the growth faction as a function of redshift.
    NRs: np.ndarray
        Number of radii. Default is 45.
    indexminNL: np.ndarray
        Index of the minimum radius R/cMpc in which we start doing the nonlinear calculation.
    indexmaxNL: np.ndarray
        Index of the maximum radius R/cMpc in which we start doing the nonlinear calculation.
    a_ST: float
        Rescaling of the HMF barrier. Default is 0.707.
        Set to 0.73 when Flag_emulate_21cmfast is True.
    p_ST: float
        Correction factor for the abundance of small mass objects. Default is 0.3.
        Set to 0.175 when Flag_emulate_21cmfast is True.
    Amp_ST: float
        Normalization factor for the halo mass function. Default is 0.3222.
        Set to 0.353 when Flag_emulate_21cmfast is True.
    delta_crit_ST: float
        Barrier for halo to collapse in Sheth-Tormen formalism. Default is 1.686.
        Set to 1.68 when Flag_emulate_21cmfast is True.
    a_corr_EPS: float
        Correction to the EPS relation between nu and nu' when doing extended PS. Follows hi-z simulation results from Schneider+21. Default is 0.707.
        Set to 1.0 when Flag_emulate_21cmfast is True.
    """
    ### Non-default parameters
    UserParams: InitVar[User_Parameters]

    ### Default parameters
    # 6 LCDM parameters
    omegab: float = 0.0223828
    omegac: float = 0.1201075
    h_fid: float = 0.67810
    As: float = 2.100549e-09
    ns: float = 0.9660499
    tau_fid: float = 0.05430842
    
    # Other params for CLASS
    kmax_CLASS: float = 500.
    zmax_CLASS: float = 50.
    zmin_CLASS: float = 5.

    # Shells that we integrate over at each z.
    Rs_min: float = 0.5 #TODO: Set to 0.5 if not doing reionization, 0.05 if doing reionization. Otherwise BMF doesn't converge
    Rs_max: float = 2000.

    # Flags
    Flag_emulate_21cmfast: bool = False
    USE_RELATIVE_VELOCITIES: bool = False
    HMF_CHOICE: str = "ST"

    # HMF mass integral
    _Mhmin: float = 1e5 # minimum halo mass in Msun
    _Mhmax: float = 1e14 # maximum halo mass in Msun

    ### Additional parameters and attributes set in the following
    # LCDM parameters
    ClassCosmo: Class = _field(init=False)
    omegam: float = _field(init=False)
    OmegaM: float = _field(init=False)
    rhocrit: float = _field(init=False)
    OmegaR: float = _field(init=False)
    OmegaL: float = _field(init=False)
    OmegaB: float = _field(init=False)
    rho_M0: float = _field(init=False)
    z_rec: float = _field(init=False)

    # v_cb parameters
    sigma_vcb: float = _field(init=False)
    vcb_avg: float = _field(init=False)

    # Number densities and mass fractions
    Y_He: float = _field(init=False)
    x_He: float = _field(init=False)
    f_H: float = _field(init=False)
    f_He: float = _field(init=False)
    mu_baryon: float = _field(init=False)
    mu_baryon_Msun: float = _field(init=False)

    # R->M conversions for HMF
    constRM: float = _field(init=False)

    # Redshifts and comoving distances
    _ztabinchi: np.ndarray = _field(init=False)
    _chitab: Any = _field(init=False)
    _Hztab: Any = _field(init=False)
    zfofRint: interp1d = _field(init=False)
    chiofzint: interp1d = _field(init=False)

    # Thermodynamics
    Tadiabaticint: interp1d = _field(init=False)
    xetanhint: interp1d = _field(init=False)

    # Growth
    growthint: interp1d = _field(init=False)

    # Radii
    NRs: np.ndarray = _field(init=False)
    _Rtabsmoo: np.ndarray = _field(init=False)
    _dlogRR: np.ndarray = _field(init=False)
    indexminNL: np.ndarray = _field(init=False)
    indexmaxNL: np.ndarray = _field(init=False)

    # HMF-related constants
    a_ST: float = _field(init=False)
    p_ST: float = _field(init=False)
    Amp_ST: float = _field(init=False)
    delta_crit_ST: float = _field(init=False)
    a_corr_EPS: float = _field(init=False)

    tageofzMyr: interp1d = _field(init=False)
    zfoftageMyr: interp1d = _field(init=False)

    def __post_init__(self, UserParams):
     
        schema = {
            "Flag_emulate_21cmfast": (bool, None),
            "USE_RELATIVE_VELOCITIES": (bool, None),
            "HMF_CHOICE": (str, {'ST','Yung'}),
        }
        validate_fields(self, schema)

       # run CLASS
        self.ClassCosmo = self.runclass()

        # derived params
        self.omegam = self.omegab + self.omegac
        self.OmegaM = self.ClassCosmo.Omega_m()
        self.rhocrit = 3 * 100**2 / (8 * np.pi* constants.MsunToKm * constants.c_kms**2 * constants.KmToMpc) * self.h_fid**2 # Msun/Mpc^3
        self.OmegaR = self.ClassCosmo.Omega_r()
        self.OmegaL = self.ClassCosmo.Omega_Lambda()
        self.OmegaB = self.ClassCosmo.Omega_b()
        self.rho_M0 = self.OmegaM * self.rhocrit
        
        _zlistforage = np.logspace(5,-3,10000)
        _zlistforage[-1]=0.0
        _Hztab = self.ClassCosmo.z_of_r(_zlistforage)[1] #chi and dchi/dz
        
        ### TODO: check if this is the same as cosmic time in cosmology
        tagetabyr = -cumulative_trapezoid(constants.Mpctoyr/_Hztab/(1+_zlistforage),_zlistforage)
        tagetabyr = np.insert(tagetabyr,0,0)

        self.tageofzMyr = interp1d(_zlistforage,tagetabyr/1e6) #interpolators for age in Myr as a function of z
        self.zfoftageMyr = interp1d(tagetabyr/1e6,_zlistforage) #and it's inverse, z for age t in Myr


        self.z_rec = self.ClassCosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        
        ### v_cb flag
        self.sigma_vcb = self.ClassCosmo.pars['sigma_vcb']
        self.vcb_avg = self.ClassCosmo.pars['v_avg']
               
        ### number densities and mass fractions
        self.Y_He = self.ClassCosmo.get_current_derived_parameters(['YHe'])['YHe']
        self.x_He = self.Y_He/4.0/(1.0 - self.Y_He) #=nHe/nH
        self.f_H = (1.0 - self.Y_He)/(1.0 - 3.0/4.0 * self.Y_He) #=nH/nb
        self.f_He = self.Y_He/4.0/(1.0 - 3.0/4.0 * self.Y_He) #=nHe/nb
        
        self.mu_baryon = (1 + self.x_He * 4.)/(1 + self.x_He) * constants.mH_GeV #mproton ~ 0.94 GeV
        self.mu_baryon_Msun = self.mu_baryon / constants.MsuntoGeV
        
        # for R->M conversions for HMF. Used for CLASS input so assumes tophat.
        self.constRM = self.OmegaM*self.rhocrit * 4.0 * np.pi/3.0

        # redshifts and comoving distances
        self._ztabinchi = np.linspace(0.0, 1100. , 10000) #cheap so do a lot
        self._chitab, self._Hztab = self.ClassCosmo.z_of_r(self._ztabinchi) #chi and dchi/dz
        self.zfofRint = interp1d(self._chitab, self._ztabinchi)
        self.chiofzint = interp1d(self._ztabinchi,self._chitab)

        # thermodynamics
        _thermo = self.ClassCosmo.get_thermodynamics()
        self.Tadiabaticint = interp1d(_thermo['z'], _thermo['Tb [K]'])
        self.xetanhint = interp1d(_thermo['z'], _thermo['x_e'])

        # growth
        _ztabingrowth = np.linspace(0., 100. , 2000)
        _growthtabint = np.array([self.ClassCosmo.scale_independent_growth_factor(zz) for zz in _ztabingrowth])
        self.growthint = interp1d(_ztabingrowth,_growthtabint)

        # shells that we integrate over at each z.
        if self.Flag_emulate_21cmfast:
            self.Rs_min = 0.62*1.5 #same as minmum R in 21cmFAST for their standard 1.5 Mpc cell resolution. 0.62 is their 'L_FACTOR'
            self.Rs_max = 500. #same as R_XLy_MAX in 21cmFAST. Too low?
            
        # radii
        self.NRs = np.floor(45*UserParams.precisionboost).astype(int)
        self._Rtabsmoo = np.logspace(np.log10(self.Rs_min), np.log10(self.Rs_max), self.NRs) # Smoothing Radii in Mpc com
        self._dlogRR = np.log(self.Rs_max/self.Rs_min)/(self.NRs-1.0)

        self.indexminNL = (np.log(UserParams.MIN_R_NONLINEAR/self.Rs_min)/self._dlogRR).astype(int)
        self.indexmaxNL = (np.log(UserParams.MAX_R_NONLINEAR/self.Rs_min)/self._dlogRR).astype(int) + 1 #to ensure it captures MAX_R

        # HMF-related constants
        if not self.Flag_emulate_21cmfast: # standard, best fit ST from Schneider+21
            self.a_ST = 0.707 # OG ST fit, or 0.85 to fit 1805.00021
            self.p_ST = 0.3
            self.Amp_ST = 0.3222
            self.delta_crit_ST = 1.686
            self.a_corr_EPS = self.a_ST
        else: # emulate 21cmFAST, including HMF from Jenkins 2001
            self.HMF_CHOICE = 'ST' # forced to match their functional form
            print('Since Flag_emulate_21cmfast==True, the code set HMF_CHOICE==ST')
            self.a_ST = 0.73
            self.p_ST = 0.175
            self.Amp_ST = 0.353
            self.delta_crit_ST = 1.68
            self.a_corr_EPS = 1.0

        # Run matter and relative velocities correlations
        self.run_correlations()

    def runclass(self):
        "Set up CLASS cosmology. Takes CosmologyIn class input and returns CLASS Cosmology object"
        ClassCosmo = Class()
        ClassCosmo.set({'omega_b': self.omegab,'omega_cdm': self.omegac,
                        'h': self.h_fid,'A_s': self.As,'n_s': self.ns,'tau_reio': self.tau_fid})
        ClassCosmo.set({'output':'mPk,mTk','lensing':'no','P_k_max_1/Mpc':self.kmax_CLASS, 'z_max_pk': self.zmax_CLASS})
        ClassCosmo.set({'gauge':'synchronous'})
        #hfid = ClassCosmo.h() # get reduced Hubble for conversions to 1/Mpc

        # and run it (see warmup for their doc)
        ClassCosmo.compute()
        
        ClassCosmo.pars['Flag_emulate_21cmfast'] = self.Flag_emulate_21cmfast
        
        ###HAC: Adding VCB feedback via a second run of CLASS:
        if self.USE_RELATIVE_VELOCITIES:
            
            kMAX_VCB = 100.0
            ###HAC: getting z_rec from first CLASS run
            z_rec = ClassCosmo.get_current_derived_parameters(['z_rec'])['z_rec']
            z_drag = ClassCosmo.get_current_derived_parameters(['z_d'])['z_d']

            ###HAC: Running CLASS a second time just to get velocity transfer functions at recombination
            ClassCosmoVCB = Class()
            ClassCosmoVCB.set({'omega_b': self.omegab,'omega_cdm': self.omegac,
                            'h': self.h_fid,'A_s': self.As,'n_s': self.ns,'tau_reio': self.tau_fid})
            ClassCosmoVCB.set({'output':'vTk'})
            ClassCosmoVCB.set({'P_k_max_1/Mpc':kMAX_VCB, 'z_max_pk':12000})
            ClassCosmoVCB.set({'gauge':'newtonian'})
            ClassCosmoVCB.compute()
            velTransFunc = ClassCosmoVCB.get_transfer(50)

            kVel = velTransFunc['k (h/Mpc)'] * self.h_fid
            theta_b = velTransFunc['t_b']
            theta_c = velTransFunc['t_cdm']

            sigma_vcb = np.sqrt(np.trapezoid(self.As * (kVel/0.05)**(self.ns-1) /kVel * (theta_b - theta_c)**2/kVel**2, kVel)) * constants.c_kms
            ClassCosmo.pars['sigma_vcb'] = sigma_vcb
            
            ###HAC: now computing average velocity assuming a Maxwell-Boltzmann distribution of velocities
            velArr = np.geomspace(0.01, constants.c_kms, 1000) #in km/s
            vavgIntegrand = (3 / (2 * np.pi * sigma_vcb**2))**(3/2) * 4 * np.pi * velArr**2 * np.exp(-3 * velArr**2 / (2 * sigma_vcb**2))
            ClassCosmo.pars['v_avg'] = np.trapezoid(vavgIntegrand * velArr, velArr)
            
            ###HAC: Computing Vcb Power Spectrum
            ClassCosmo.pars['k_vcb'] = kVel
            ClassCosmo.pars['theta_b'] = theta_b
            ClassCosmo.pars['theta_c'] = theta_c
            P_vcb = self.As * (kVel/0.05)**(self.ns-1) * (theta_b - theta_c)**2/kVel**2 * 2 * np.pi**2 / kVel**3
            
            p_vcb_intp = interp1d(np.log(kVel), P_vcb)
            ClassCosmo.pars['P_vcb'] = P_vcb
            
            ###HAC: Computing Vcb^2 (eta) Power Spectra
            kVelIntp = np.geomspace(1e-4, kMAX_VCB, 512)
            rVelIntp = 2 * np.pi / kVelIntp
            
            j0bessel = lambda x: np.sin(x)/x
            j2bessel = lambda x: (3 / x**2 - 1) * np.sin(x)/x - 3*np.cos(x)/x**2
            
            psi0 = 1 / 3 / (sigma_vcb/constants.c_kms)**2 * np.trapezoid(kVelIntp**2 / 2 / np.pi**2 * p_vcb_intp(np.log(kVelIntp)) * j0bessel(kVelIntp * np.transpose([rVelIntp])), kVelIntp, axis = 1)
            psi2 = -2 / 3 / (sigma_vcb/constants.c_kms)**2 * np.trapezoid(kVelIntp**2 / 2 / np.pi**2 * p_vcb_intp(np.log(kVelIntp)) * j2bessel(kVelIntp * np.transpose([rVelIntp])), kVelIntp, axis = 1)
            
            k_eta, P_eta = mcfit.xi2P(rVelIntp, l=0, lowring = True)((6 * psi0**2 + 3 * psi2**2), extrap = False)
            
            ClassCosmo.pars['k_eta'] = k_eta[P_eta > 0]
            ClassCosmo.pars['P_eta'] = P_eta[P_eta > 0]
            
    #        print("HAC: Finished running CLASS a second time to get velocity transfer functions")
            
        else:
            ClassCosmo.pars['v_avg'] = 0.0
            ClassCosmo.pars['sigma_vcb'] = 1.0 #Avoids excess computation, but doesn't matter what value we set it to because the flag in inputs.py sets all feedback parameters to zero
        
        return ClassCosmo
    
    def run_correlations(self):
        #we choose the k to match exactly the log FFT of input Rtabsmoo.

        self._klistCF, _dummy_ = mcfit.xi2P(self._Rtabsmoo, l=0, lowring=True)(0*self._Rtabsmoo, extrap=False)
        self.NkCF = len(self._klistCF)

        self._PklinCF = np.zeros(self.NkCF) # P(k) in 1/Mpc^3
        for ik, kk in enumerate(self._klistCF):
            self._PklinCF[ik] = self.ClassCosmo.pk(kk, 0.0) # function .pk(k,z)



        self._xif = mcfit.P2xi(self._klistCF, l=0, lowring=True)


        self.xi_RR_CF = self.get_xi_R1R2(field = 'delta')
        self.ClassCosmo.pars['xi_RR_CF'] = np.copy(self.xi_RR_CF) #store correlation function for gamma_III correction in SFRD

        ###HAC: Interpolated object for eta power spectrum
        if self.USE_RELATIVE_VELOCITIES == True:
            P_eta_interp = interp1d(self.ClassCosmo.pars['k_eta'], self.ClassCosmo.pars['P_eta'], bounds_error = False, fill_value = 0)
            self._PkEtaCF = P_eta_interp(self._klistCF)
            self.xiEta_RR_CF = self.get_xi_R1R2(field = 'vcb')
        else:
            self._PkEtaCF = np.zeros_like(self._PklinCF)
            self.xiEta_RR_CF = np.zeros_like(self.xi_RR_CF)


    def get_xi_R1R2 (self, field = None):
        "Get correlation function of density, linearly extrapolated to z=0, smoothed over two different radii with Window(k,R)"
        
        lengthRarray = self.NRs
        windowR1 = z21_utilities.Window(self._klistCF.reshape(lengthRarray, 1, 1), self._Rtabsmoo.reshape(1, 1, lengthRarray))
        windowR2 = z21_utilities.Window(self._klistCF.reshape(1, lengthRarray,1), self._Rtabsmoo.reshape(1, 1, lengthRarray))
        
        if field == 'delta':
            _PkRR = np.array([[self._PklinCF]]) * windowR1 * windowR2
        elif field == 'vcb':
            _PkRR = np.array([[self._PkEtaCF]]) * windowR1 * windowR2
        else:
            raise ValueError('field has to be either delta or vcb in get_xi_R1R2')
        
        self.rlist_CF, xi_RR_CF = self._xif(_PkRR, extrap = False)

        return xi_RR_CF



@dataclass(kw_only=True)
class Astro_Parameters:
    """
    Astrophysical parameters for zeus21.

    Parameters
    ----------
        CosmoParams: Cosmo_Parameters
            zeus21 class for the cosmological parameters. Needs to be inputed.
        accretion_model: str
            Accretion model. "exp" for exponential, "EPS" for EPS. "RP16" for the dynamically averaged fitting function in Rodríguez-Puebla+16. Default is "exp".
        USE_POPIII: bool
            Whether to use Pop III. Default is False.
        USE_LW_FEEDBACK: bool
            Whether to use the Lyman-Werner feedback. Default is True.
        quadratic_SFRD_lognormal: bool
            Whether to use the second order correction to the SFRD approximation. Default is True.
        epsstar: float
            Amplitude of the star formation efficiency (at M_pivot). Default is 0.1.
        dlog10epsstardz: float
            Derivative of epsstar with respect to z. Default is 0.0.
        alphastar: float
            Power law index of the star formation efficiency at low masses. Default 0.5.
        betastar: float
            Power law index of the star formation efficiency at high masses. Not used if Flag_emulate_21cmfast = True. Default -0.5.
        Mc: float
            Mass at which the star formation efficiency cuts. Not used if Flag_emulate_21cmfast=True. Default 3e11.
        epsstar_III: float
            Amplitude of the star formation efficiency (at M_pivot) for Pop III. Default is 10**(-2.5).
        dlog10epsstardz_III: float
            Derivative of epsstar with respect to z for Pop III. Default is 0.0.
        alphastar_III: float
            Power law index of the Pop III star formation efficiency at low masses. Default 0.0.
        betastar_III: float
            Power law index of the Pop III star formation efficiency at high masses. Default 0.0.
        Mc_III: float
            Mass at which the Pop III star formation efficiency cuts. Default 1e7.
        Mup_III: str, float, None
            High-mass cutoff for Pop III star formation efficiency. Default "Matom" for Pop III star formation confined to molecular-cooling minihalos; also accepts custom cutoff value or None for no cutoff
        DETACH_III_ACH: bool
            Whether to have a separate set of parameters for star formation efficiency for the (ACH) component for Pop III. Default is False.
        epsstar_III_ACH: float
            Amplitude of the star formation efficiency (at M_pivot) for the (ACH) component for Pop III. Default is 0.0.
        dlog10epsstardz_III_ACH: float
            Derivative of epsstar with respect to z for the (ACH) component for Pop III. Default is 0.0.
        alphastar_III_ACH: float
            Power law index of the (ACH) component of Pop III star formation efficiency at low masses. Default 0.0.
        betastar_III_ACH: float
            Power law index of the (ACH) component of Pop III star formation efficiency at high masses. Default 0.0.
        Mc_III_ACH: float
            Mass at which the (ACH) component of Pop III star formation efficiency cuts. Default 1e7.
        N_alpha_perbaryon_II: float
            Number of photons between LyA and Ly Continuum per baryon (from LB05). Default is 9690.
        N_alpha_perbaryon_III: float
            Number of photons between LyA and Ly Continuum per baryon (from LB05) for Pop III. Default is 17900.
        L40_xray: float
            Soft-band (E<2 keV) lum/SFR in Xrays in units of 10^40 erg/s/(Msun/yr). Default is 3.0.
        E0_xray: float
            Minimum energy in eV. Default is 500.
        alpha_xray: float
            Xray SED power-law index. Default is -1.0.
        L40_xray_III: float
            Soft-band (E<2 keV) lum/SFR in Xrays in units of 10^40 erg/s/(Msun/yr) for Pop III. Default is 3.0.
        alpha_xray_III: float
            Xray SED power-law index. Default is -1.0.
        Emax_xray_norm: float
            Max energy in eV to normalize SED. Default at 2000.0 eV.
        fesc10: float
            Amplitude of the escape fraction. Default is 0.1.
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        alphaesc: float
            Index for the escape fraction. Default is 0.0.
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        fesc7_III: float
            Amplitude of the Pop III escape fraction. Default is 10**(-1.35).
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        alphaesc_III: float
            Index for the Pop III escape fraction. Default is -0.3.
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        clumping: float
            Clumping factor, which is z-independent and fixed for now. Default is 3.0, changed to 2.0 when Flag_emulate_21cmfast=True.
        R_linear_sigma_fit_input: float
            Initial guess radius at which the linear fit of the barrier is computed. Default is 10.0.
        FLAG_BMF_converge: bool
            Whether zeus21 allow the BMF to try and make the average ionized fraction converge. Default is True.
        max_iter: int
            Maximum iteration allowed for the convergence of the BMF. Default is 10.
        ZMAX_REION: float
            Maximum redshift to which the reionization quantities are computed. Default is 30.0.
        Rbub_min: float
            Minimum bubble radius. Default is 0.05.
        A_LW: float
            Parameters controlling the LW feedback factor (see Munoz+22, eq 13). Default is 2.0.
        beta_LW: float
            Parameters controlling the LW feedback factor (see Munoz+22, eq 13). Default is 0.6.
        A_vcb: float
            Normalization for the relative velocity feedback parameter. Default is 1.0.
        beta_vcb: float
            Spectral index for the relative velocity feedback parameter. Default 1.8.
        Mturn_fixed: float | None
            Turn-over halo mass at which the star formation rate cuts. Default is None.
        FLAG_MTURN_SHARP: bool
            Whether to do sharp cut at Mturn_fixed or regular exponential cutoff. Only active if FLAG_MTURN_FIXED and turned on by hand. Default is False.
        FLAG_USE_PSD: bool
            Whether to derive MUV and sigmaUV from integrating SFH. Default is False.
        FLAG_COMPARE_BAGPIPES: bool
            Whethher to compare with bagpipes. Default is False.
        SEDMODEL: str = "BPASS"
            Which SED model to use for the Greens functions. Default is "BPASS".
            Can be set to "bagpipes", "BPASS_binaries", and "BPASS".
        normLHa_ZIMF: float
            Floating normalization of the LHa luminosity compared to the baseline SEDMODEL to account for HMF or metallicity changes. Default is 1.0
        alphanormLHa_ZIMF:
            Power-law index of normLHa_ZIMF against halo mass. Default is 0.0
        sigmaPSD: float
            Amplitude of fluctuations in SFR arising from the power spectral density (PSD) model of SFR variability. Default is 0.5.
            This is the baseline scatter in ln(SFR) at a reference halo mass of 10^10 Msun.
        dsigmaPSDdlog10Mh: float
            Slope of the scatter with respect to halo mass. Default is 0.0.
        tauPSD: float
            Characteristic timescale (in Myr) that enters the power spectral density (PSD) of ln(SFR). Default is 10.0.
        dlog10tauPSDdlog10Mh: float
            Slope of the timescale with respect to halo mass. Default is 0.0.
        _tcut_LUV_short: float
            Sets where the LUV short and long are separated in Myr. Default is 30.0.
        FLAG_RENORMALIZE_AVG_SFH: bool
            Whether to normalize the SFR in Msun/yr at each Mh. Default is True.
        
    Attributes
    ----------
        fstarmax: float
            Peak amplitude for the star formation efficiency. Set by zeus21 to 1.
        Emax_xray_integral: float
            Max energy in eV that zeus21 integrate up to. Higher than Emax_xray_norm since photons can redshift from higher z. Set by zeus21 to 10000.
        Nen_xray: int
            Number of energies to do the xray integrals. Set by zeus21 to 30.
        Energylist: np.ndarray
            Energies, in eV.
        dlogEnergy: float
            Used to get dlog instead of dlog10.
        N_ion_perbaryon_II: int
            Number of ionizing photons per baryon. Fixed for PopII-type (Salpeter) by zeus21 to 5000.
        N_ion_perbaryon_III: int
            Number of ionizing photons per baryon for Pop III. Fixed for PopIII-type to 44000 (or 52480 when Flag_emulate_21cmfast=True), from Klessen & Glover 2023 Table A2 (2303.12500).
        N_LW_II: float
            Number of LW photons per baryon.
            Assuming BL05 stellar spectrum, equal to N_alpha_perbaryon_II * fraction of photons that fall in the LW band.
        N_LW_III: float
            Number of LW photons per baryon.
            Assuming Intermediate IMF from 2202.02099, equal to 4.86e-22 / (11.9 * u.eV).to(u.erg).value * 5.8e14.
        FLAG_MTURN_FIXED: bool
            Whether to fix Mturn or use Matom(z) at each z. Set by zeus21 depending on Mturn_fixed.
    
    """
    ### Non-default parameters
    CosmoParams: InitVar[Cosmo_Parameters]

    ### Default and init=False parameters
    # Flags
    accretion_model: str = "exp"
    USE_POPIII: bool = False
    USE_LW_FEEDBACK: bool = True
    quadratic_SFRD_lognormal: bool = True

    # SFR(Mh) parameters - popII
    epsstar: float = 0.1
    dlog10epsstardz: float = 0.0
    alphastar: float = 0.5
    betastar: float = -0.5
    Mc: float = 3e11
    _zpivot: float = _field(init=False) # Redshift at which the eps and dlogeps/dz are evaluated. Set by zeus21 to 8.0.
    fstarmax: float = _field(init=False)

    # SFR(Mh) parameters - popIII main component --> by default, this only includes the molecular-cooling minihalo component extended up to the atomic-cooling limit, but it can be extended to a custom high-mass cutoff value by changing the value of Mup_III
    epsstar_III: float = 10**(-2.5)
    dlog10epsstardz_III: float = 0.0
    alphastar_III: float = 0.
    betastar_III: float = 0.
    Mc_III: float = 1e7
    Mup_III: str | float | None = "Matom"
    Mup3TEMP: float = 10**8.387007493446207#HAC TEMPORARY: Delete!!! Only for BAO/VAO comparison
    _zpivot_III: float = _field(init=False) # Redshift at which the eps and dlogeps/dz are evaluated for Pop III. Set by zeus21 to 8.0.

    # SFR(Mh) parameters - popIII deatched atomic-cooling component
    DETACH_III_ACH: bool = False
    epsstar_III_ACH: float = 0.
    dlog10epsstardz_III_ACH: float = 0.0
    alphastar_III_ACH: float = 0.
    betastar_III_ACH: float = 0.
    Mc_III_ACH: float = 1e7
    _zpivot_III_ACH: float = _field(init=False) # Redshift at which the eps and dlogeps/dz are evaluated for the (ACH) component for Pop III. Set by zeus21 to 8.0.

    # Lyman-alpha parameters
    N_alpha_perbaryon_II: float = 9690
    N_alpha_perbaryon_III: float = 17900

    # Xray parameters, assumed power-law for now
    L40_xray: float = 3.0
    E0_xray: float = 500.
    alpha_xray: float = -1.0
    L40_xray_III: float = 3.0
    alpha_xray_III: float = -1.0
    Emax_xray_norm: float = 2000
    Emax_xray_integral: float = _field(init=False) # Max energy in eV that we integrate up to. Higher than Emax_xray_norm since photons can redshift from higher z
    
    # table with how many energies we integrate over
    Nen_xray: int = _field(init=False)
    _log10EMIN_INTEGRATE: float = _field(init=False) # Minimum energy zeus21 integrates to, to account for photons coming from higher z that redshift.
    _log10EMAX_INTEGRATE: float = _field(init=False) # Maximum energy zeus21 integrates to, to account for photons coming from higher z that redshift.
    Energylist: np.ndarray = _field(init=False) # in eV
    dlogEnergy: float = _field(init=False) # to get dlog instead of dlog10
    
    # Reionization parameters
    fesc10: float = 0.1
    alphaesc: float = 0.0
    fesc7_III: float = 10**(-1.35)
    alphaesc_III: float = -0.3
    clumping: float = 3.
    N_ion_perbaryon_II: int = _field(init=False) # fixed for PopII-type (Salpeter)
    N_ion_perbaryon_III: int = _field(init=False) # fixed for PopIII-type, from Klessen & Glover 2023 Table A2 (2303.12500)
    R_linear_sigma_fit_input: float = 10.
    FLAG_BMF_converge: bool = True
    max_iter: int = 10
    ZMAX_REION: float = 30
    Rbub_min: float = 0.05

    # Lyman-Werner feedback paramters
    A_LW: float = 2.0
    beta_LW: float = 0.6
    N_LW_II: float = _field(init=False) # number of LW photons per baryon #assuming BL05 stellar spectrum, equal to N_alpha_perbaryon_II * fraction of photons that fall in the LW band
    N_LW_III: float = _field(init=False) # number of LW photons per baryon #assuming Intermediate IMF from 2202.02099, equal to 4.86e-22 / (11.9 * u.eV).to(u.erg).value * 5.8e14

    # relative velocity
    A_vcb: float = 1.0
    beta_vcb: float = 1.8

    # 21cmFAST emulation: SFE parameters
    Mturn_fixed: float | None = None
    FLAG_MTURN_SHARP: bool = False
    FLAG_MTURN_FIXED: bool = _field(init=False) # whether to fix Mturn or use Matom(z) at each z
    
    # BURSTINESS
    FLAG_USE_PSD: bool = False
    FLAG_COMPARE_BAGPIPES: bool = False
    SEDMODEL: str = "BPASS"
    normLHa_ZIMF: float = 1.0
    alphanormLHa_ZIMF: float = 0.0
    sigmaPSD: float = 0.5,
    dsigmaPSDdlog10Mh: float = 0.0,
    tauPSD: float = 10.0,
    dlog10tauPSDdlog10Mh: float = 0.0,
    _tcut_LUV_short: float = 30.0
    FLAG_RENORMALIZE_AVG_SFH: bool = True
    _minsigmaPSD: float = _field(init=False)
    _maxsigmaPSD: float = _field(init=False)
    _mintauPSD: float = _field(init=False)
    _maxtauPSD: float = _field(init=False)
    _tagesMyr: float = _field(init=False)
    _dt_FFT: float = _field(init=False)
    _N_FFT: float = _field(init=False)
    _omegamin: float = _field(init=False)
    _omegamax: float = _field(init=False)

    def __post_init__(self, CosmoParams):

        schema = {
            "accretion_model": (str, {"EPS", "exp"}),
            "USE_POPIII": (bool, None),
            "DETACH_III_ACH": (bool, None),
            "USE_LW_FEEDBACK": (bool, None),
            "quadratic_SFRD_lognormal": (bool, None),
            "FLAG_MTURN_SHARP": (bool, None),
            "FLAG_USE_PSD": (bool, None),
            "FLAG_COMPARE_BAGPIPES": (bool, None),
            "FLAG_RENORMALIZE_AVG_SFH": (bool, None),
            "SEDMODEL": (str, {"bagpipes", "BPASS_binaries", "BPASS"}),
        }
        validate_fields(self, schema)

        ### which SFR model we use. 0=Gallumi-like, 1=21cmfast-like
        if not CosmoParams.Flag_emulate_21cmfast: # GALLUMI-like
            self.accretion_model = self.accretion_model # choose the accretion model: 0 = exponential, 1= EPS. Default = EPS.
        else: # 21cmfast-like, ignores Mc and beta and has a t* later in SFR()
            self.tstar = 0.5
            self.fstar10 = self.epsstar

        # SFR(Mh) parameters
        self._zpivot = 8.0 # fixed, at which z we evaluate eps and dlogeps/dz
        self._zpivot_III = 8.0  # fixed, at which z we evaluate eps and dlogeps/dz
        self._zpivot_III_ACH = 8.0  # fixed, at which z we evaluate eps and dlogeps/dz
        self.fstarmax = 1.0 # where we cap it
        
        # Xray parameters
        self.Emax_xray_integral = 10000. # Max energy in eV that we integrate up to. Higher than Emax_xray_norm since photons can redshift from higher z
        if(self.E0_xray < constants.EN_ION_HI):
            print("What the heck? How can E0_XRAY < EN_ION_HI?")

        # table with how many energies we integrate over
        self.Nen_xray = 30
        self._log10EMIN_INTEGRATE = np.log10(self.E0_xray/2.0) # to account for photons coming from higher z that redshift
        self._log10EMAX_INTEGRATE = np.log10(self.Emax_xray_integral)
        self.Energylist = np.logspace(self._log10EMIN_INTEGRATE,self._log10EMAX_INTEGRATE,self.Nen_xray) # in eV
        self.dlogEnergy = (self._log10EMAX_INTEGRATE - self._log10EMIN_INTEGRATE)/(self.Nen_xray-1.0)*np.log(10.) # to get dlog instead of dlog10
        
        # Reionization parameters
        if CosmoParams.Flag_emulate_21cmfast:
            self.clumping = 2.0 # this is the 21cmFAST value
        # number of ionizing photons per baryon
        self.N_ion_perbaryon_II = 5000 # fixed for PopII-type (Salpeter)
        if CosmoParams.Flag_emulate_21cmfast:
            self.N_ion_perbaryon_III = 44000 # fixed for PopIII-type, from Klessen & Glover 2023 Table A2 (2303.12500)
        else:
            self.N_ion_perbaryon_III = 52480

        ### HAC: LW feedback parameters
        if not self.USE_LW_FEEDBACK:
            self.A_LW = 0.0
            self.beta_LW = 0.0
        # number of LW photons per baryon
        if not CosmoParams.Flag_emulate_21cmfast:
            self.N_LW_II = 6200.0 #assuming BL05 stellar spectrum, equal to N_alpha_perbaryon_II * fraction of photons that fall in the LW band
            self.N_LW_III = 12900.0 #assuming Intermediate IMF from 2202.02099, equal to 4.86e-22 / (11.9 * u.eV).to(u.erg).value * 5.8e14
        else:
            popIIcorrection = 0.6415670418531249/2.5 #scaling used by 21cmfast to get correct number of Pop II LW photons per baryon
            self.N_LW_II = popIIcorrection * self.N_alpha_perbaryon_II
            popIIIcorrection = 0.7184627927009317/6.5 #scaling used by 21cmfast to get correct number of Pop III LW photons per baryon
            self.N_LW_III = popIIIcorrection * self.N_alpha_perbaryon_III
        
        ### HAC: Relative Velocities parameters
        if not CosmoParams.USE_RELATIVE_VELOCITIES:
            self.A_vcb = 0.0
            self.beta_vcb = 0.0

        ### 21cmFAST emulation: SFE parameters
        if(self.Mturn_fixed == None): #The FIXED/SHARP routine below only applies to Pop II, not to Pop III
            self.FLAG_MTURN_FIXED = False # whether to fix Mturn or use Matom(z) at each z
        else:
            self.FLAG_MTURN_FIXED = True # whether to fix Mturn or use Matom(z) at each z


        self._minsigmaPSD = 0.1 # Minimum sigma for the PSD, to avoid numerical issues in the FFT
        self._maxsigmaPSD = 4.0 # Maximum sigma for the PSD, there'll never be enough samples if sigma>~6-10
        self._mintauPSD = 1.0 # in Myr. Minimum tau for the PSD, to avoid numerical issues in the FFT
        self._maxtauPSD = 300.0

        self._tagesMyr = np.logspace(-2, 3, 79) #times (ages) we integrate over at each z, Mh, in Myr (TODO: add precisionboost)


        self._dt_FFT = 0.3 # FFT timescale resolution, Myr, high to resolve the PS_SFR and window functions well (TODO: add UserParams precisionboost here)
        self._N_FFT = int(512/(self._dt_FFT/0.3))  # Recommend to use power of 2 for efficient FFT, resolve up to ~0.5Gyr at least


        self._omegamin = 2*np.pi/1e3
        self._omegamax  = np.pi/1.0



@dataclass(kw_only=True)
class LF_Parameters:
    """
    Luminosity functions parameters for zeus21.

    Parameters
    ----------
        zcenter: float
            Redshift bin center at which to compute the luminosity functions. Default is 6.0.
        zwidth:  float
            Redshift bin width at which to compute the luminosity functions. Default is 0.5
        RETURNLF: bool
            Whether to compute LFs. Default is True.
        RETURNBIAS: bool
            Whether to compute bias. Default is False.
        SKIP_POPII : bool
            If True, skip the Pop II component. Default is False.
        SKIP_POPIII : bool
            If True, skip the Pop III component. Default is True.
        SKIP_TOT : bool
            If True, skip the summed component. Default is False.
        FLAG_COMPUTE_UVLF: bool
            Whether to compute the UV LF/bias. Default is True.
        MUVcenters: np.ndarray | float
            M_UV bin centers at which to compute the luminosity functions. Default is np.linspace(-23,-14,100).
        MUVwidths: np.ndarray | float
            M_UV bin width at which to compute the luminosity functions. Default is 0.5.
        FLAG_RENORMALIZE_LUV 
            Whether to renormalize the lognormal LUV with sigmaUV to recover <LUV> or otherwise <MUV>. Default is False (recommended). 
        _kappaUV : float = 1.15e-28.
            SFR-to-UV conversion factor in Msun/yr per erg/s/Hz.
        sigmaUV: float
            Stochasticity (gaussian rms) in the halo-galaxy connection P(MUV | Mh). Default is 0.5.
        UV_boost_III : float
            Pop III main-component UV conversion factor relative to Pop II. Default is 1.
        _kappaUV_III : float = kappaUV/UV_boost_III.
            SFR-to-UV conversion factor in Msun/yr per erg/s/Hz for Pop III main component.
        sigmaUV_III : float
            MUV scatter for the Pop III main component. Default is 0.5.
        DUST_FLAG_III: bool
            Whether to include dust attenuation to the LF calculations for Pop III main component. Default is False.
        UV_boost_III_ACH : float
            Pop III detached atomic-cooling-halo UV conversion factor relative to Pop II. Default is 1.
        _kappaUV_III_ACH : float = kappaUV/UV_boost_III_ACH.
            SFR-to-UV conversion factor in Msun/yr per erg/s/Hz for Pop III ACH component.
        sigmaUV_III_ACH : float
            MUV scatter for the detached Pop III ACH component. Default is 0.5.
        DUST_FLAG_III_ACH: bool
            Whether to include dust attenuation to the LF calculations for Pop III ACH component. Default is False.
        FLAG_COMPUTE_HaLF: bool
            Whether to compute the Ha LF/bias. Default is False.
        log10LHacenters: np.ndarray | float
            Ha bin centers at which to compute the luminosity functions, given in log10. Default is np.linspace(38,45,10).
        log10LHawidths: np.ndarray | float
            Ha bin width at which to compute the luminosity functions, given in log10. Default is 0.5.
        DUST_FLAG: bool
            Whether to include dust attenuation to the LF calculations. Default is True.
        DUST_model: str
            Which dust model to use. Default is "Bouwens13". Can also be "Zhao24" (https://arxiv.org/pdf/2401.07893.pdf, table 1).
        HIGH_Z_DUST: bool
            Whether to do dust at higher z than 0 or set to 0. Fix at beta(z=_zmaxdata) result if so. Default is True.  
        _zmaxdata : float
            Maximum calibration redshift for the dust model. Default is 8.0.
        C0dust: float
            Calibration parameter for the dust correction for UVLF. Default is 4.43 (following Meurer+99). Input 4.54 for Overzier+01.
        C1dust: float
            Calibration parameter for the dust correction for UVLF. Default 1.99 for Meurer99. Input 2.07 for Overzier+01.
        sigma_times_AUV_dust: float
            If not 0, normalization factor to the sigma UV of dust. Default is 0.0.
    """

    zcenter: float = 6.
    zwidth:  float = 0.5

    ### Flags for computing LFs and bias for available populations
    RETURNLF: bool = True
    RETURNBIAS: bool = False
    SKIP_POPII: bool = False
    SKIP_POPIII: bool = True
    SKIP_TOT: bool = False

    ### General UVLF parameters
    FLAG_COMPUTE_UVLF: bool = True
    MUVcenters: np.ndarray | float = _field(default_factory=lambda: np.linspace(-23,-14,100))
    MUVwidths: np.ndarray | float = 0.5
    FLAG_RENORMALIZE_LUV: bool = False  # Whether to renormalize the lognormal LUV with sigmaUV to recover <LUV> or otherwise <MUV>. Recommend False.
    _kappaUV: float = _field(init=False)  # in SFR/LUV. Set by zeus21 to the value from Madau+Dickinson14, fully degenerate with epsilon
    sigmaUV: float = 0.5 

    ### PopIII UVLF parameters (main component)
    UV_boost_III: float = 1.
    _kappaUV_III: float = _field(init=False) # in SFR/LUV for PopIII.  Set by zeus21 to be a factor UV_boost_III times more efficient than PopII.  
    sigmaUV_III: float = 0.5
    DUST_FLAG_III: bool = False

    ### PopIII UVLF parameters (ACH component)
    UV_boost_III_ACH: float = 1.
    _kappaUV_III_ACH: float = _field(init=False)  # in SFR/LUV for PopIII.  Set by zeus21 to be a factor UV_boost_III_ACH times more efficient than PopII.  
    sigmaUV_III_ACH: float = 0.5
    DUST_FLAG_III_ACH: bool = False

    ### Halpha LF parameters
    log10LHacenters: np.ndarray | float = _field(default_factory=lambda: np.linspace(38,45,10))
    log10LHawidths: np.ndarray | float = 0.5
    FLAG_COMPUTE_HaLF: bool = False

    ### Dust parameters for UVLFs
    DUST_FLAG: bool = True
    DUST_model: str = 'Bouwens13'
    HIGH_Z_DUST: bool = True
    _zmaxdata: float = 8.0
    C0dust: float = 4.43
    C1dust: float = 1.99 #4.43, 1.99 is Meurer99; 4.54, 2.07 is Overzier01
    sigma_times_AUV_dust: float = 0.

    def __post_init__(self):
        schema = {
            "RETURNLF": (bool, None),
            "RETURNBIAS": (bool, None),
            "FLAG_COMPUTE_UVLF": (bool, None),
            "FLAG_COMPUTE_HaLF": (bool, None),
            "FLAG_RENORMALIZE_LUV": (bool, None),
            "DUST_FLAG": (bool, None),
            "HIGH_Z_DUST": (bool, None),
            "DUST_model": (str, {"Bouwens13", "Zhao24"}),
        }
        validate_fields(self, schema)


        # --- normalize MUV ---
        if np.isscalar(self.zcenter):
            self.MUVcenters = np.array(self.MUVcenters, dtype=float)
        else:
            self.MUVcenters = np.atleast_1d(self.MUVcenters).astype(float)

        # --- normalize MUVwidth ---
        if np.isscalar(self.MUVwidths):
            # broadcast scalar to same length as zcenter
            self.MUVwidths = np.full_like(self.MUVcenters, self.MUVwidths, dtype=float)
        else:
            self.MUVwidths = np.atleast_1d(self.MUVwidths).astype(float)

        # --- consistency check ---
        if self.MUVwidths.shape != self.MUVcenters.shape:
            raise ValueError(
                f"MUVwidth shape {self.MUVwidths.shape} does not match MUVcenter shape {self.MUVcenters.shape}"
            )

        # --- normalize logLHa ---
        if np.isscalar(self.log10LHacenters):
            self.log10LHacenters = np.array([self.log10LHacenters], dtype=float)
        else:
            self.log10LHacenters = np.atleast_1d(self.log10LHacenters).astype(float)

        # --- normalize logHazwidth ---
        if np.isscalar(self.log10LHawidths):
            # broadcast scalar to same length as zcenter
            self.log10LHawidths = np.full_like(self.log10LHacenters, self.log10LHawidths, dtype=float)
        else:
            self.log10LHawidths = np.atleast_1d(self.log10LHawidths).astype(float)

        # --- consistency check ---
        if self.log10LHawidths.shape != self.log10LHacenters.shape:
            raise ValueError(
                f"log10Hawidth shape {self.log10LHawidths.shape} does not match log10Hacenter shape {self.log10LHacenters.shape}"
            )

        ### Parameters for SFR-to-LUV conversion
        self._kappaUV = 1.15e-28  # SFR/LUV, value from Madau+Dickinson14, fully degenerate with epsilon
        self._kappaUV_III = self._kappaUV / self.UV_boost_III  # SFR/LUV for PopIII main component
        self._kappaUV_III_ACH = self._kappaUV / self.UV_boost_III_ACH  # SFR/LUV for PopIII ACH component


def validate_fields(obj, schema: dict):
    """ Helper function to check whether the input parameters are set with the proper values. """
    for field, (expected_type, allowed_values) in schema.items():
        value = getattr(obj, field)

        if not isinstance(value, expected_type):
            raise TypeError(
                f"{field} must be of type {expected_type.__name__}, got {type(value).__name__}"
            )

        if allowed_values is not None and value not in allowed_values:
            raise ValueError(
                f"{field} must be one of {allowed_values}, got '{value}'"
            )
