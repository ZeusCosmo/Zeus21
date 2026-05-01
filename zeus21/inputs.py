"""

Takes inputs and stores them in useful classes

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Sarah Libanore, Emilie Thelie
BGU, UT Austin - April 2026 
"""

from . import constants
from . import z21_utilities

from dataclasses import dataclass, field as _field, InitVar
from typing import Any
import numpy as np
from classy import Class
from scipy.interpolate import interp1d
import mcfit


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
        Make integrals take more points for boost in precision, the baseline being 1.0.
    dlogzint_target:
        Target number of redshift bins for the redsfhit arrays in log space.
    FLAG_FORCE_LINEAR_CF: int (False or True)
        False to do standard calculation, True to force linearization of correlation function.
    MIN_R_NONLINEAR: float
        Minimum radius R/cMpc in which we start doing the nonlinear calculation. 
        Below ~1 it will blow up because sigma > 1 eventually, and our exp(delta) approximation breaks. 
        Check if you play with it and if you change Window().
    MAX_R_NONLINEAR: float
        Maximum radius R/cMpc in which we start doing the nonlinear calculation (above this it is very linear)
    FLAG_DO_DENS_NL: bool
        Whether to do the nonlinear (ie lognormal) calculation for the density field itself and its cross correlations. 
        Small (<3%) correction in dd, but non trivial (~10%) in d-xa and d-Tx
    FLAG_WF_ITERATIVE: bool
        Whether to iteratively do the WF correction as in Hirata2006.
    zmin_T21: float
        Minimum redshift to which we compute the T21 signals.
    DO_ONLY_GLOBAL: bool
        Whether zeus21 only runs the global T21 signal (and not fluctuations).

    Attributes
    ----------
    C2_RENORMALIZATION_FLAG: int (False or True)
        Whether to renormalize the C2 oefficients (appendix in 2302.08506).
    """

    precisionboost: float = 1.0
    dlogzint_target: float = 0.02
    FLAG_FORCE_LINEAR_CF: bool = False
    MIN_R_NONLINEAR: float = 2.0
    MAX_R_NONLINEAR: float = 100.0
    FLAG_DO_DENS_NL: bool = False
    FLAG_WF_ITERATIVE: bool = True
    zmin_T21: float = 5.
    DO_ONLY_GLOBAL: bool = False

    C2_RENORMALIZATION_FLAG: bool = _field(init=False)


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
    Cosmological parameters (including the 6 LCDM + other parameters) for zeus21 and running of CLASS.
    
    Parameters
    ----------
        UserParams: User_Parameters
            zeus21 class for the user parameters.
        omegab: float
            Baryon density * h^2.
        omegac: float
            CDM density * h^2.
        h_fid: float
            Hubble constant / 100.
        As: float
            Amplitude of initial fluctuations.
        ns: float
            Spectral index.
        tau_fid: float
            Optical depth to reionization.
        kmax_CLASS: float
            Maximum wavenumber to be passed to CLASS.
        zmax_CLASS: float
            Maximum redshift to be passed to CLASS.
        zmin_CLASS: float
            Minimum redshift to be passed to CLASS.
        Rs_min: float
            Minimum radius to be passed to CLASS.
        Rs_max: float
            Maximum radius to be passed to CLASS.
        Flag_emulate_21cmfast: bool
            Whether zeus21 emulates 21cmFAST cosmology (used in HMF, LyA, and X-ray opacity calculations). Default is False.
            When False, sets the Star Formation Rate model to GALLUMI-like, and when True to 21cmfast-like (ignores Mc and beta and has a t* later in SFR()).
        USE_RELATIVE_VELOCITIES: bool
            Whether to use v_cb.
        HMF_CHOICE: str
            Which HMF to use.
            "ST" for the classic  Sheth-Tormen (f(nu)), "Yung" for the Tinker08 (f(sigma)) calibrated to Yung+23.
        
    Attributes
    ----------
    ClassCosmo: Class
        CLASS instance to compute cosmology.
    omegam: float
        Matter density * h^2.
    OmegaM: float 
        Matter density.
    rhocrit: float
        Critical density.
    OmegaR: float 
        Radiation density.
    OmegaL: float 
        Dark energy density. 
    OmegaB: float 
        Baryon density.
    rho_M0: float 
        Actual matter density.
    z_rec: float
        Recombination reshift.
    sigma_vcb: float 
        Square root of the variance of the relative velocity field.
    vcb_avg: float 
        Average of the relative velocity field.
    Y_He: float 
        Helium mass fraction.
    x_He: 
        Helium-to-hydrogen number density ratio.
    f_H: float
        Hydrogen number density ratio relative to baryons.
    f_He: float 
        Helium number density ratio relative to baryons.
    mu_baryon: float 
        Mean baryonic weight.
    mu_baryon_Msun: float
        Mean baryonic weight relative to the solar mass.
    constRM: float
        Radius-to-mass conversions for HMF. Used for CLASS input so assumes tophat.
    zfofRint: interp1d 
        Interpolation for the redshift as a function of the comoving distance.
    chiofzint: interp1d 
        Interpolation for the comoving distance as a function of the redshift.
    Hofzint: interp1d 
        Interpolation for the Hubble rate as a function of the redshift.
    Tadiabaticint: 
        Interpolation for the adiabatic temperature as a function of redshift.
    xetanhint: interp1d 
        Interpolation for the electron fraction as a function of redshift.
    growthint: interp1d
        Interpolation for the growth faction as a function of redshift.
    NRs: np.ndarray
        Number of radii.
    indexminNL: np.ndarray
        Index of the minimum radius R/cMpc in which we start doing the nonlinear calculation. 
    indexmaxNL: np.ndarray
        Index of the maximum radius R/cMpc in which we start doing the nonlinear calculation. 
    a_ST: float
        Rescaling of the HMF barrier.
    p_ST: float
        Correction factor for the abundance of small mass objects.
    Amp_ST: float
        Normalization factor for the halo mass function.
    delta_crit_ST: float
        Barrier for halo to collapse in Sheth-Tormen formalism.
    a_corr_EPS: float
        Correction to the EPS relation between nu and nu' when doing extended PS. Follows hi-z simulation results from Schneider+21.
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
    Rs_min: float = 0.05  ### ASK JULIAN for changing the name
    Rs_max: float = 2000. ### ASK JULIAN for changing the name

    # Flags
    Flag_emulate_21cmfast: bool = False
    USE_RELATIVE_VELOCITIES: bool = False
    HMF_CHOICE: str = "ST"


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
    Hofzint: interp1d = _field(init=False)

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
        #self.rhocrit = 2.78e11*self.h_fid**2 #Msun/Mpc^3 ### TODO
        self.OmegaR = self.ClassCosmo.Omega_r()
        self.OmegaL = self.ClassCosmo.Omega_Lambda()
        self.OmegaB = self.ClassCosmo.Omega_b()
        self.rho_M0 = self.OmegaM * self.rhocrit
        
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
        self.Hofzint = interp1d(self._ztabinchi,self._Hztab)

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
        ClassCosmo.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':self.kmax_CLASS, 'z_max_pk': self.zmax_CLASS}) ###HAC: add vTK to outputs
        ClassCosmo.set({'gauge':'synchronous'})
        #hfid = ClassCosmo.h() # get reduced Hubble for conversions to 1/Mpc

        # and run it (see warmup for their doc)
        ClassCosmo.compute()
        
        ClassCosmo.pars['Flag_emulate_21cmfast'] = self.Flag_emulate_21cmfast
        
        ###HAC: Adding VCB feedback via a second run of CLASS:
        if self.USE_RELATIVE_VELOCITIES:
            
            kMAX_VCB = 50.0
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
            velTransFunc = ClassCosmoVCB.get_transfer(z_drag)

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
        "same as get_xi_z0_lin but smoothed over two different radii with Window(k,R) \
        same separations rs as get_xi_z0_lin so it does not output them."
        
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
        Cosmo_Parameters: Cosmo_Parameters
            zeus21 class for the cosmological parameters. Needs to be inputed.
        accretion_model: str
            Accretion model. "exp" for exponential, "EPS" for EPS. Default is "EPS".
        USE_POPIII: bool
            Whether to use Pop III. Default is False.
        USE_LW_FEEDBACK: bool
            Whether to use the Lyman-Werner feedback. Default is True.
        quadratic_SFRD_lognormal: bool
            Whether to use the second order correction to the SFRD approximation. Default is True.
        epsstar: float
            Amplitude of the star formation efficiency (at M_pivot). Default is 0.1.
        dlog10epsstardz: float
            Derivative of epsstar with respect to z. Default is 0.
        alphastar: float
            Power law index of the star formation efficiency at low masses. Default 0.5.
        betastar: float
            Power law index of the star formation efficiency at high masses. Only used when astromodel=0. Default -0.5.
        Mc: float
            Mass at which the star formation efficiency cuts. Only used when astromodel=0. Default 3e11.
        sigmaUV: float
            Stochasticity (gaussian rms) in the halo-galaxy connection P(MUV | Mh). Default is 0.5.
        alphastar_III: float
            Power law index of the Pop III star formation efficiency at low masses. Default 0.
        betastar_III: float
            Power law index of the Pop III star formation efficiency at high masses. Default 0.
        fstar_III: float
            Peak amplitude of the Pop III star formation efficiency. Default 10**(-2.5).
        Mc_III: float
            Mass at which the Pop III star formation efficiency cuts. Default 1e7.
        dlog10epsstardz_III: float
            Derivative of epsstar with respect to z for Pop III. Default is 0.
        N_alpha_perbaryon_II: float
            Number of photons between LyA and Ly Continuum per baryon (from LB05). Default is 9690.
        N_alpha_perbaryon_III: float
            Number of photons between LyA and Ly Continuum per baryon (from LB05) for Pop III. Default is 17900.
        L40_xray: float
            Soft-band (E<2 keV) lum/SFR in Xrays in units of 10^40 erg/s/(Msun/yr). Default is 3.0.
        E0_xray: float
            Minimum energy in eV. Default is 500.
        alpha_xray: float
            Xray SED power-law index. Default is -1.
        L40_xray_III: float
            Soft-band (E<2 keV) lum/SFR in Xrays in units of 10^40 erg/s/(Msun/yr) for Pop III. Default is 3.0.
        alpha_xray_III: float
            Xray SED power-law index. Default is -1.
        Emax_xray_norm: float
            Max energy in eV to normalize SED. Default at 2000 eV. 
        fesc10: float
            Amplitude of the escape fraction. Default is 0.1. 
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        alphaesc: float
            Index for the escape fraction. Default is 0.
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        fesc7_III: float
            Amplitude of the Pop III escape fraction. Default is 10**(-1.35). 
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        alphaesc_III: float
            Index for the Pop III escape fraction. Default is -0.3.
            Escape fraction assumed to be a power law normalized (fesc10) at M=1e10 Msun with index alphaesc.
        clumping: float = 3. 
            Clumping factor, which is z-independent and fixed for now. Default is 3, changed to 2 when Flag_emulate_21cmfast=True.
        R_linear_sigma_fit_input: float
            Initial guess radius at which the linear fit of the barrier is computed. Default is 3.
        FLAG_BMF_converge: bool
            Whether zeus21 allow the BMF to try and make the average ionized fraction converge. Default is True.
        max_iter: int
            Maximum iteration allowed for the convergence of the BMF. Default is 10.
        ZMAX_REION: float
            Maximum redshift to which the reionization quantities are computed. Default is 30.
        Rbub_min: float
            Minimum bubble radius. Default is 0.05.
        A_LW: float
            Parameters controlling the LW feedback factor (see Munoz+22, eq 13). Default is 2.0.
        beta_LW: float
            Parameters controlling the LW feedback factor (see Munoz+22, eq 13). Default is 0.6.
        A_vcb: float
            Normalization for the relative velocity feedback parameter. Default is 1.0.
        beta_vcb: float
            Spectral index for the relative velocity feedback parameter. Default 1.8
        Mturn_fixed: float | None 
            Turn-over halo mass at which the star formation rate cuts. Default is None.
        FLAG_MTURN_SHARP: bool
            Whether to do sharp cut at Mturn_fixed or regular exponential cutoff. Only active if FLAG_MTURN_FIXED and turned on by hand. Default is False.
        C0dust: float
            Calibration parameter for the dust correction for UVLF. Default is 4.43 (following Meurer+99). Input 4.54 for Overzier+01.
        C1dust: float
            Calibration parameter for the dust correction for UVLF. Default 1.99 for Meurer99. Input 2.07 for Overzier+01.
        
    Attributes
    ----------
        _zpivot: float
            Redshift at which the eps and dlogeps/dz are evaluated. Set by zeus21 to 8.
        fstarmax: float
            Peak amplitude for the star formation efficiency. Set by zeus21 to 1.
        _zpivot_III: float
            Redshift at which the eps and dlogeps/dz are evaluated for Pop III. Set by zeus21 to 8.
        Emax_xray_integral: float
            Max energy in eV that zeus21 integrate up to. Higher than Emax_xray_norm since photons can redshift from higher z. Set by zeus21 to 10000.
        Nen_xray: int
            Number of energies to do the xray integrals. Set by zeus21 to 30.
        _log10EMIN_INTEGRATE: float
            Minimum energy zeus21 integrates to, to account for photons coming from higher z that redshift. 
        _log10EMAX_INTEGRATE: float
            Maximum energy zeus21 integrates to, to account for photons coming from higher z that redshift. 
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
        _kappaUV: float 
            SFR/LUV. Set by zeus21 to the value from Madau+Dickinson14.
            Fully degenerate with epsilon.
        _kappaUV_III: float 
            SFR/LUV for PopIII.  Set by zeus21 to the value from Madau+Dickinson14.
            Assume X more efficient than PopII.
    
    Methods
    ----------
        SED_XRAY
            SED of our Xray sources. Takes energy En in eV.
            Normalized to integrate to 1 from E0_xray to Emax_xray (int dE E * SED(E).
            E*SED is the power-law with index alpha_xray, so the output is divided by 1/E at the end to return number). 
        SED_LyA
            SED of our Lyman-alpha-continuum sources.
            Normalized to integrate to 1 (int d nu SED(nu), so SED is number per units energy (as opposed as E*SED, what was for Xrays).

    """
    ### Non-default parameters
    CosmoParams: InitVar[Cosmo_Parameters]


    ### Default and init=False parameters
    # Flags
    accretion_model: str = "exp"
    USE_POPIII: bool = False
    USE_LW_FEEDBACK: bool = True
    quadratic_SFRD_lognormal: bool = True

    # SFR(Mh) parameters 
    epsstar: float = 0.1
    dlog10epsstardz: float = 0.0 
    alphastar: float = 0.5
    betastar: float = -0.5
    Mc: float = 3e11
    sigmaUV: float = 0.5 # TODO: only used in UVLF not sfrd
    _zpivot: float = _field(init=False)
    fstarmax: float = _field(init=False)
    alphastar_III: float = 0
    betastar_III: float = 0
    fstar_III: float = 10**(-2.5)
    Mc_III: float = 1e7
    dlog10epsstardz_III: float = 0.0
    _zpivot_III: float = _field(init=False)

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
    _log10EMIN_INTEGRATE: float = _field(init=False) # to account for photons coming from higher z that redshift
    _log10EMAX_INTEGRATE: float = _field(init=False)
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
    
    ### Dust parameters for UVLFs
    C0dust: float = 4.43
    C1dust: float = 1.99 #4.43, 1.99 is Meurer99; 4.54, 2.07 is Overzier01
    _kappaUV: float = _field(init=False) #SFR/LUV, value from Madau+Dickinson14, fully degenerate with epsilon
    _kappaUV_III: float = _field(init=False) #SFR/LUV for PopIII. Assume X more efficient than PopII   


    def __post_init__(self, CosmoParams):
        schema = {
            "accretion_model": (str, {"EPS", "exp"}),
            "USE_POPIII": (bool, None),
            "USE_LW_FEEDBACK": (bool, None),
            "quadratic_SFRD_lognormal": (bool, None),
            "FLAG_MTURN_SHARP": (bool, None),
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

        ### Dust parameters for UVLFs
        self._kappaUV = 1.15e-28 #SFR/LUV, value from Madau+Dickinson14, fully degenerate with epsilon
        self._kappaUV_III = self._kappaUV #SFR/LUV for PopIII. Assume X more efficient than PopII



    def SED_XRAY(self, En, pop = 0): #pop set to zero as default, but it must be set to either 2 or 3
        "SED of our Xray sources, normalized to integrate to 1 from E0_xray to Emax_xray (int dE E * SED(E), and E*SED is the power-law with index alpha_xray, so the output is divided by 1/E at the end to return number). Takes energy En in eV"
        if pop == 2:
            alphaX = self.alpha_xray
        elif pop == 3:
            alphaX = self.alpha_xray_III
        else:
            print("Must set pop to either 2 or 3!")
            
        if np.abs(alphaX + 1.0) < 0.01: #log
            norm = 1.0/np.log(self.Emax_xray_norm/self.E0_xray) / self.E0_xray
        else:
            norm = (1.0 + alphaX)/((self.Emax_xray_norm/self.E0_xray)**(1 + alphaX) - 1.0) / self.E0_xray

        return np.power(En/self.E0_xray, alphaX)/En * norm * np.heaviside(En - self.E0_xray, 0.5)
        #do not cut at higher energies since they redshift into <2 keV band

    def SED_LyA(self, nu_in, pop = 0): #default pop set to zero so python doesn't complain, but must be 2 or 3 for this to work
        "SED of our Lyman-alpha-continuum sources, normalized to integrate to 1 (int d nu SED(nu), so SED is number per units energy (as opposed as E*SED, what was for Xrays) "

        nucut = constants.freqLyB #above and below this freq different power laws
        if pop == 2:
            amps = np.array([0.68,0.32]) #Approx following the stellar spectra of BL05. Normalized to unity
            indexbelow = 0.14 #if one of them zero worry about normalization
            normbelow = (1.0 + indexbelow)/(1.0 - (constants.freqLyA/nucut)**(1 + indexbelow)) * amps[0]
            indexabove = -8.0
            normabove = (1.0 + indexabove)/((constants.freqLyCont/nucut)**(1 + indexabove) - 1.0) * amps[1]
        elif pop == 3:
            amps = np.array([0.56,0.44]) #Approx following the stellar spectra of BL05. Normalized to unity
            indexbelow = 1.29 #if one of them zero worry about normalization
            normbelow = (1.0 + indexbelow)/(1.0 - (constants.freqLyA/nucut)**(1 + indexbelow)) * amps[0]
            indexabove = 0.2
            normabove = (1.0 + indexabove)/((constants.freqLyCont/nucut)**(1 + indexabove) - 1.0) * amps[1]
        else:
            print("Must set pop to 2 or 3!")
            
        nulist = np.asarray([nu_in]) if np.isscalar(nu_in) else np.asarray(nu_in)

        result = np.zeros_like(nulist)
        for inu, currnu in enumerate(nulist):
            if (currnu<constants.freqLyA or currnu>=constants.freqLyCont):
                result[inu] = 0.0
            elif (currnu < nucut): #between LyA and LyB
                result[inu] = normbelow * (currnu/nucut)**indexbelow
            elif (currnu >= nucut):  #between LyB and Continuum
                result[inu] = normabove * (currnu/nucut)**indexabove
            else:
                print("Error in SED_LyA, whats the frequency Kenneth?")


        return result/nucut #extra 1/nucut because dnu, normalizes the integral
        


def validate_fields(obj, schema: dict):
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