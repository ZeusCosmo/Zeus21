"""

Takes inputs and stores them in useful classes

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

from . import constants

import numpy as np
from classy import Class
from scipy.interpolate import interp1d

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
    FLAG_FORCE_LINEAR_CF: int (0 or 1)
        0 to do standard calculation, 1 to force linearization of correlation function.
    MIN_R_NONLINEAR: float
        Minimum radius R/cMpc in which we start doing the nonlinear calculation. 
        Below ~1 it will blow up because sigma > 1 eventually, and our exp(\delta) approximation breaks. 
        Check if you play with it and if you change Window().
    MAX_R_NONLINEAR: float
        Maximum radius R/cMpc in which we start doing the nonlinear calculation (above this it is very linear)
    FLAG_DO_DENS_NL: bool
        Whether to do the nonlinear (ie lognormal) calculation for the density field itself and its cross correlations. 
        Small (<3%) correction in dd, but non trivial (~10%) in d-xa and d-Tx
    FLAG_WF_ITERATIVE: bool
        Whether to iteratively do the WF correction as in Hirata2006.

    Attributes
    ----------
    C2_RENORMALIZATION_FLAG: int (0 or 1)
        Whether to renormalize the C2 oefficients (appendix in 2302.08506).
    """

    def __init__(self, precisionboost = 1.0, FLAG_FORCE_LINEAR_CF = 0, 
                 MIN_R_NONLINEAR = 2.0, MAX_R_NONLINEAR = 100.0,
                 FLAG_DO_DENS_NL = False, FLAG_WF_ITERATIVE = True):
        
        self.precisionboost = precisionboost
        self.FLAG_FORCE_LINEAR_CF = FLAG_FORCE_LINEAR_CF
        self.C2_RENORMALIZATION_FLAG = 1 - FLAG_FORCE_LINEAR_CF

        self.MIN_R_NONLINEAR = MIN_R_NONLINEAR
        self.MAX_R_NONLINEAR = MAX_R_NONLINEAR

        self.FLAG_DO_DENS_NL = FLAG_DO_DENS_NL

        self.FLAG_WF_ITERATIVE = FLAG_WF_ITERATIVE



class Cosmo_Parameters_Input:
    "Class to pass the 6 LCDM parameters as input"

    def __init__(self, omegab = 0.0223828, omegac = 0.1201075, h_fid = 0.67810, As = 2.100549e-09, ns = 0.9660499, 
                 tau_fid = 0.05430842, kmax_CLASS = 500., zmax_CLASS = 50.,zmin_CLASS = 5., Flag_emulate_21cmfast = False, 
                 USE_RELATIVE_VELOCITIES = False, HMF_CHOICE= "ST"):

        self.omegab = omegab
        self.omegac = omegac
        self.h_fid = h_fid
        self.As = As
        self.ns = ns
        self.tau_fid = tau_fid

        #other params for CLASS
        self.kmax_CLASS = kmax_CLASS
        self.zmax_CLASS = zmax_CLASS
        self.zmin_CLASS = zmin_CLASS
        #and whether to emulate 21cmFAST
        self.Flag_emulate_21cmfast = Flag_emulate_21cmfast #whether to emulate 21cmFAST in HMF, LyA, and X-ray opacity calculations
        
        ###HAC: Flag whether to use v_cb
        self.USE_RELATIVE_VELOCITIES = USE_RELATIVE_VELOCITIES

        #which HMF we use
        self.HMF_CHOICE = HMF_CHOICE #which HMF functional form we use.
        #options are "ST" the classic  Sheth-Tormen (f(nu)), "Yung" for the Tinker08 (f(sigma)) calibrated to Yung+23. Default ST



class Cosmo_Parameters:
    "Class that will keep the cosmo parameters throughout"

    def __init__(self, UserParams, CosmoParams_input, ClassCosmo):

        self.omegab = CosmoParams_input.omegab
        self.omegac = CosmoParams_input.omegac
        self.h_fid = CosmoParams_input.h_fid
        self.As = CosmoParams_input.As
        self.ns = CosmoParams_input.ns
        self.tau_fid = CosmoParams_input.tau_fid

        #other params in the input
        self.kmax_CLASS = CosmoParams_input.kmax_CLASS
        self.zmax_CLASS = CosmoParams_input.zmax_CLASS
        self.zmin_CLASS = CosmoParams_input.zmin_CLASS #when to start the HMF calcs., not an input strictly
        self.Flag_emulate_21cmfast = CosmoParams_input.Flag_emulate_21cmfast #whether to emulate 21cmFAST in HMF, LyA, and X-ray opacity calculations

        #derived params
        self.omegam = self.omegab + self.omegac
        self.OmegaM = ClassCosmo.Omega_m()
        self.rhocrit = 2.78e11*self.h_fid**2 #Msun/Mpc^3
        self.OmegaR = ClassCosmo.Omega_r()
        self.OmegaL = ClassCosmo.Omega_Lambda()
        self.OmegaB = ClassCosmo.Omega_b()
        
        self.z_rec = ClassCosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        
        ###HAC: added v_cb flag. JBM: moved to CosmoParams so user does not have to pass Class Cosmo all the time
        self.USE_RELATIVE_VELOCITIES = CosmoParams_input.USE_RELATIVE_VELOCITIES
        if self.USE_RELATIVE_VELOCITIES == True:
            self.sigma_vcb = ClassCosmo.pars['sigma_vcb']
            self.vcb_avg = ClassCosmo.pars['v_avg']
        else:  #set but not to random values, just something sensible in case the user wants pop3 but not relvel
            self.sigma_vcb = 30.0
            self.vcb_avg = 27.5
               
        ###n_H() stuff
        self.Y_He = ClassCosmo.get_current_derived_parameters(['YHe'])['YHe']
        self.x_He = self.Y_He/4.0/(1.0 - self.Y_He) #=nHe/nH
        self.f_H = (1.0 - self.Y_He)/(1.0 - 3.0/4.0 * self.Y_He) #=nH/nb
        self.f_He = self.Y_He/4.0/(1.0 - 3.0/4.0 * self.Y_He) #=nHe/nb
        
        self.mu_baryon = (1 + self.x_He * 4.)/(1 + self.x_He) * constants.mH_GeV #mproton ~ 0.94 GeV
        self.mu_baryon_Msun = self.mu_baryon/constants.MsuntoGeV
        
#        ###old dependencies of n_baryon() instead of n_H()
#        self.Y_He = ClassCosmo.get_current_derived_parameters(['YHe'])['YHe']
#        self.f_He = self.Y_He/4.0/(1.0 - 3.0/4.0 * self.Y_He) #=nHe/nb
#        self.f_H = (1.0 - self.Y_He)/(1.0 - 3.0/4.0 * self.Y_He) #=nH/nb
#        self.mu_baryon = (self.f_H + self.f_He * 4.) * 0.94 #mproton ~ 0.94 GeV
        


        #for R->M conversions for HMF. Used for CLASS input so assumes tophat.
        self.constRM = self.OmegaM*self.rhocrit * 4.0 * np.pi/3.0

        self.rho_M0 = self.OmegaM*self.rhocrit



        self._ztabinchi = np.linspace(0.0, 1100. , 10000) #cheap so do a lot
        # self._chitab = ClassCosmo.z_of_r(self._ztabinchi)[0]
        # self.zfofRint = interp1d(self._chitab, self._ztabinchi)
        self._chitab, self._Hztab = ClassCosmo.z_of_r(self._ztabinchi) #chi and dchi/dz
        self.zfofRint = interp1d(self._chitab, self._ztabinchi)
        self.chiofzint = interp1d(self._ztabinchi,self._chitab)
        self.Hofzint = interp1d(self._ztabinchi,self._Hztab)

        _thermo = ClassCosmo.get_thermodynamics()
        self.Tadiabaticint = interp1d(_thermo['z'], _thermo['Tb [K]'])
        self.xetanhint = interp1d(_thermo['z'], _thermo['x_e'])

        _ztabingrowth = np.linspace(0., 100. , 2000)
        _growthtabint = np.array([ClassCosmo.scale_independent_growth_factor(zz) for zz in _ztabingrowth])

        self.growthint = interp1d(_ztabingrowth,_growthtabint)


        #and define the shells that we integrate over at each z.
        self.Rsmmin = 0.5
        self.Rsmmax = 2000.

        if(self.Flag_emulate_21cmfast==True):
            self.Rsmmin = 0.62*1.5 #same as minmum R in 21cmFAST for their standard 1.5 Mpc cell resolution. 0.62 is their 'L_FACTOR'
            self.Rsmmax = 500. #same as R_XLy_MAX in 21cmFAST. Too low?
            
        self.NRs = np.floor(45*UserParams.precisionboost).astype(int)
        self._Rtabsmoo = np.logspace(np.log10(self.Rsmmin), np.log10(self.Rsmmax), self.NRs) # Smoothing Radii in Mpc com
        self._dlogRR = np.log(self.Rsmmax/self.Rsmmin)/(self.NRs-1.0)

        self.indexminNL = (np.log(UserParams.MIN_R_NONLINEAR/self.Rsmmin)/self._dlogRR).astype(int)
        self.indexmaxNL = (np.log(UserParams.MAX_R_NONLINEAR/self.Rsmmin)/self._dlogRR).astype(int) + 1 #to ensure it captures MAX_R


        #HMF-related constants
        self.HMF_CHOICE = CosmoParams_input.HMF_CHOICE 
        if(self.Flag_emulate_21cmfast == False): #standard, best fit ST from Schneider+
            self.a_ST = 0.707 #OG ST fit, or 0.85 to fit 1805.00021
            self.p_ST = 0.3
            self.Amp_ST = 0.3222
            self.delta_crit_ST = 1.686
            self.a_corr_EPS = self.a_ST #correction to the eps relation between nu and nu' when doing extended PS. Follows hi-z simulation results from Schneider+
        elif(self.Flag_emulate_21cmfast == True): #emulate 21cmFAST, including HMF from Jenkins 2001
            self.HMF_CHOICE = 'ST' #forced to match their functional form
            self.a_ST = 0.73
            self.p_ST = 0.175
            self.Amp_ST = 0.353
            self.delta_crit_ST = 1.68
            self.a_corr_EPS = 1.0
            
        else:
            print("Error! Have to set either Flag_emulate_21cmfast = True or False")



class Astro_Parameters:
    "Class to pass the astro parameters as input"

    def __init__(self, UserParams, Cosmo_Parameters, 
                    astromodel = 0,
                    accretion_model = 0,

                    alphastar = 0.5,
                    betastar = -0.5,
                    epsstar = 0.1,
                    Mc = 3e11,
                    dlog10epsstardz = 0.0,

                    fesc10 = 0.1,
                    alphaesc = 0.0,
                    L40_xray = 3.0,
                    E0_xray = 500.,
                    alpha_xray = -1.0,
                    Emax_xray_norm=2000,

                    Nalpha_lyA_II = 9690,
                    Nalpha_lyA_III = 17900,

                    Mturn_fixed = None,
                    FLAG_MTURN_SHARP= False,

                    C0dust = 4.43,
                    C1dust = 1.99,

                    sigmaUV=0.5,

                    USE_POPIII = False,

                    alphastar_III = 0, 
                    betastar_III = 0,
                    fstar_III = 10**(-2.5),
                    Mc_III = 1e7,
                    dlog10epsstardz_III = 0.0,
                 
                    fesc7_III = 10**(-1.35),
                    alphaesc_III = -0.3,
                    L40_xray_III = 3.0,
                    alpha_xray_III = -1.0,
                    
                    USE_LW_FEEDBACK = True,
                    A_LW = 2.0,
                    beta_LW = 0.6,
                 
                    A_vcb = 1.0,
                    beta_vcb = 1.8,
                 

                ):
        
        #for internal functions in SED_LyA
        self.Flag_emulate_21cmfast = Cosmo_Parameters.Flag_emulate_21cmfast

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True and astromodel == 0):
            print('ERROR, picked astromodel = 0 but tried to emulate 21cmFAST. They use astromodel = 1. Changing it!')
            self.astromodel = 1
        else:
            self.astromodel = astromodel # which SFR model we use. 0=Gallumi-like, 1=21cmfast-like

        ###HAC: PopIII parameters:
        self.USE_POPIII = USE_POPIII
        
        self.alphastar_III = alphastar_III
        self.betastar_III = betastar_III
        self.fstar_III = fstar_III
        self.Mc_III = Mc_III
        self.dlog10epsstardz_III = dlog10epsstardz_III
        self._zpivot_III = 8.0  #fixed, at which z we evaluate eps and dlogeps/dz
        
        self.fesc7_III = fesc7_III
        self.alphaesc_III = alphaesc_III
        self.L40_xray_III = L40_xray_III
        self.alpha_xray_III = alpha_xray_III
        
        ###HAC: Using LW feedback and fixing parameters
        self.USE_LW_FEEDBACK = USE_LW_FEEDBACK
        
        if self.USE_LW_FEEDBACK == True:
            self.A_LW = A_LW
            self.beta_LW = beta_LW
        else:
            self.A_LW = 0.0
            self.beta_LW = 0.0
        
        ###HAC: Using Relative Velocities and fixing parameters
        if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:
            self.A_vcb = A_vcb
            self.beta_vcb = beta_vcb
        else:
            self.A_vcb = 0.0
            self.beta_vcb = 0.0

        
        #SFR(Mh) parameters: 
        self.epsstar = epsstar #epsilon_* = f* at Mc
        self.dlog10epsstardz = dlog10epsstardz #dlog10epsilon/dz
        self._zpivot = 8.0 #fixed, at which z we evaluate eps and dlogeps/dz
        self.alphastar = alphastar #powerlaw index for lower masses
        self.betastar = betastar #powerlaw index for higher masses, only for model 0
        self.Mc = Mc # mass at which the power law cuts, only for model 0
        self.sigmaUV = sigmaUV #stochasticity (gaussian rms) in the halo-galaxy connection P(MUV | Mh) - TODO: only used in UVLF not sfrd

        self.fstarmax = 1.0 #where we cap it
        
        if self.astromodel == 0: #GALUMI-like
            self.accretion_model = accretion_model #0 = exponential, 1= EPS #choose the accretion model. Default = EPS
        elif self.astromodel == 1: #21cmfast-like, ignores Mc and beta and has a t* later in SFR()
            self.tstar = 0.5
            self.fstar10 = self.epsstar
        else:
            print('ERROR, need to pick astromodel')

        #fesc(M) parameter. Power law normalized (fesc10) at M=1e10 Msun with index alphaesc
        self.fesc10 = fesc10
        self.alphaesc = alphaesc
        self._clumping = 3.0 #clumping factor, z-independent and fixed for now
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self._clumping = 2.0 #this is the 21cmFAST value



        #xray parameters here, assumed power-law for now
        self.L40_xray = L40_xray # soft-band (E<2 keV) lum/SFR in Xrays in units of 10^40 erg/s/(Msun/yr)
        self.E0_xray = E0_xray #minimum energy in eV
        self.Emax_xray_norm = Emax_xray_norm #max energy in eV to normalize SED. Keep at 2000 eV normally
        self.Emax_xray_integral = 10000. #max energy in eV that we integrate up to. Higher than Emax_xray_norm since photons can redshift from higher z
        self.alpha_xray = alpha_xray #Xray SED power-law index

        if(self.E0_xray < constants.EN_ION_HI):
            print('What the heck? How can E0_XRAY < EN_ION_HI ?')



        #table with how many energies we integrate over
        self.Nen_xray = 30
        self._log10EMIN_INTEGRATE = np.log10(self.E0_xray/2.0) # to account for photons coming from higher z that redshift
        self._log10EMAX_INTEGRATE = np.log10(self.Emax_xray_integral)
        self.Energylist = np.logspace(self._log10EMIN_INTEGRATE,self._log10EMAX_INTEGRATE,self.Nen_xray) #in eV
        self.dlogEnergy = (self._log10EMAX_INTEGRATE - self._log10EMIN_INTEGRATE)/(self.Nen_xray-1.0)*np.log(10.) #to get dlog instead of dlog10


        self.N_alpha_perbaryon_II=Nalpha_lyA_II #number of photons between LyA and Ly Cont. per baryon (from LB05)
        self.N_alpha_perbaryon_III=Nalpha_lyA_III #number of photons between LyA and Ly Cont. per baryon (from LB05)
        
        #number of ionizing photons per baryon
        self.N_ion_perbaryon_II = 5000 #fixed for PopII-type (Salpeter)
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self.N_ion_perbaryon_III = 44000 #fixed for PopIII-type, from Klessen & Glover 2023 Table A2 (2303.12500)
        elif(Cosmo_Parameters.Flag_emulate_21cmfast==False):
            self.N_ion_perbaryon_III = 52480 #fixed for PopIII-type, from Klessen & Glover 2023 Table A2 (2303.12500)
            
        #number of LW photons per baryon
        if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
            self.N_LW_II = 6200.0 #assuming BL05 stellar spectrum, equal to N_alpha_perbaryon_II * fraction of photons that fall in the LW band
            self.N_LW_III = 12900.0 #assuming Intermediate IMF from 2202.02099, equal to 4.86e-22 / (11.9 * u.eV).to(u.erg).value * 5.8e14
            
        elif(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            popIIIcorrection = 0.7184627927009317/6.5 #scaling used by 21cmfast to get correct number of Pop III LW photons per baryon
            self.N_LW_III = popIIIcorrection * self.N_alpha_perbaryon_III

            popIIcorrection = 0.6415670418531249/2.5 #scaling used by 21cmfast to get correct number of Pop II LW photons per baryon
            self.N_LW_II = popIIcorrection * self.N_alpha_perbaryon_II


        if(Mturn_fixed == None): #The FIXED/SHARP routine below only applies to Pop II, not to Pop III
            self.FLAG_MTURN_FIXED = False #whether to fix Mturn or use Matom(z) at each z
        else:
            self.FLAG_MTURN_FIXED = True #whether to fix Mturn or use Matom(z) at each z
            self.Mturn_fixed = Mturn_fixed
            self.FLAG_MTURN_SHARP = FLAG_MTURN_SHARP #whether to do sharp cut at Mturn_fixed or regular exponential cutoff. Only active if FLAG_MTURN_FIXED and turned on by hand.

        #dust parameters for UVLFs:
        self.C0dust, self.C1dust = C0dust, C1dust #4.43, 1.99 is Meurer99; 4.54, 2.07 is Overzier01
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
        

        
###HAC: Original SED_LyA
#    def SED_LyA(self, nu_in):
#        "SED of our Lyman-alpha-continuum sources, normalized to integrate to 1 (int d nu SED(nu), so SED is number per units energy (as opposed as E*SED, what was for Xrays) "
#
#        nucut = constants.freqLyB #above and below this freq different power laws
#        amps = np.array([0.68,0.32]) #Approx following the stellar spectra of BL05. Normalized to unity
#
#        indexbelow = 0.14 #if one of them zero worry about normalization
#        normbelow = (1.0 + indexbelow)/(1.0 - (constants.freqLyA/nucut)**(1 + indexbelow)) * amps[0]
#        indexabove = -8.0
#        normabove = (1.0 + indexabove)/((constants.freqLyCont/nucut)**(1 + indexabove) - 1.0) * amps[1]
#
#        nulist = np.asarray([nu_in]) if np.isscalar(nu_in) else np.asarray(nu_in)
#
#        result = np.zeros_like(nulist)
#        for inu, currnu in enumerate(nulist):
#            if (currnu<constants.freqLyA or currnu>=constants.freqLyCont):
#                result[inu] = 0.0
#            elif (currnu < nucut): #between LyA and LyB
#                result[inu] = normbelow * (currnu/nucut)**indexbelow
#            elif (currnu >= nucut):  #between LyB and Continuum
#                result[inu] = normabove * (currnu/nucut)**indexabove
#            else:
#                print("Error in SED_LyA, whats the frequency Kenneth?")
#
#
#        return result/nucut #extra 1/nucut because dnu, normalizes the integral
