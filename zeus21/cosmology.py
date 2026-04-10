"""

Cosmology helper functions and other tools

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

"""

import numpy as np
from classy import Class
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

import mcfit

from . import constants
from .inputs import Cosmo_Parameters, Cosmo_Parameters_Input
from .correlations import Correlations

def cosmo_wrapper(User_Parameters, Cosmo_Parameters_Input):
    """
    Wrapper function for all the cosmology. It takes Cosmo_Parameters_Input and returns:
    Cosmo_Parameters, Class_Cosmo, Correlations, HMF_interpolator
    """

    #ClassCosmo = Class()
    #ClassCosmo.compute()

    ClassyCosmo = runclass(Cosmo_Parameters_Input)
    CosmoParams = Cosmo_Parameters(User_Parameters, Cosmo_Parameters_Input, ClassyCosmo) 
    CorrFClass = Correlations(User_Parameters, CosmoParams, ClassyCosmo)
    HMFintclass = HMF_interpolator(User_Parameters,CosmoParams,ClassyCosmo)

    return CosmoParams, ClassyCosmo, CorrFClass, HMFintclass



def runclass(CosmologyIn):
    "Set up CLASS cosmology. Takes CosmologyIn class input and returns CLASS Cosmology object"
    ClassCosmo = Class()
    ClassCosmo.set({'omega_b': CosmologyIn.omegab,'omega_cdm': CosmologyIn.omegac,
                    'h': CosmologyIn.h_fid,'A_s': CosmologyIn.As,'n_s': CosmologyIn.ns,'tau_reio': CosmologyIn.tau_fid})
    ClassCosmo.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':CosmologyIn.kmax_CLASS, 'z_max_pk': CosmologyIn.zmax_CLASS}) ###HAC: add vTK to outputs
    ClassCosmo.set({'gauge':'synchronous'})
    #hfid = ClassCosmo.h() # get reduced Hubble for conversions to 1/Mpc

    # and run it (see warmup for their doc)
    ClassCosmo.compute()
    
    ClassCosmo.pars['Flag_emulate_21cmfast'] = CosmologyIn.Flag_emulate_21cmfast
    
    ###HAC: Adding VCB feedback via a second run of CLASS:
    if CosmologyIn.USE_RELATIVE_VELOCITIES == True:
        
        kMAX_VCB = 50.0
        ###HAC: getting z_rec from first CLASS run
        z_rec = ClassCosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        z_drag = ClassCosmo.get_current_derived_parameters(['z_d'])['z_d']

        ###HAC: Running CLASS a second time just to get velocity transfer functions at recombination
        ClassCosmoVCB = Class()
        ClassCosmoVCB.set({'omega_b': CosmologyIn.omegab,'omega_cdm': CosmologyIn.omegac,
                        'h': CosmologyIn.h_fid,'A_s': CosmologyIn.As,'n_s': CosmologyIn.ns,'tau_reio': CosmologyIn.tau_fid})
        ClassCosmoVCB.set({'output':'vTk'})
        ClassCosmoVCB.set({'P_k_max_1/Mpc':kMAX_VCB, 'z_max_pk':12000})
        ClassCosmoVCB.set({'gauge':'newtonian'})
        ClassCosmoVCB.compute()
        velTransFunc = ClassCosmoVCB.get_transfer(z_drag)

        kVel = velTransFunc['k (h/Mpc)'] * CosmologyIn.h_fid
        theta_b = velTransFunc['t_b']
        theta_c = velTransFunc['t_cdm']

        sigma_vcb = np.sqrt(np.trapezoid(CosmologyIn.As * (kVel/0.05)**(CosmologyIn.ns-1) /kVel * (theta_b - theta_c)**2/kVel**2, kVel)) * constants.c_kms
        ClassCosmo.pars['sigma_vcb'] = sigma_vcb
        
        ###HAC: now computing average velocity assuming a Maxwell-Boltzmann distribution of velocities
        velArr = np.geomspace(0.01, constants.c_kms, 1000) #in km/s
        vavgIntegrand = (3 / (2 * np.pi * sigma_vcb**2))**(3/2) * 4 * np.pi * velArr**2 * np.exp(-3 * velArr**2 / (2 * sigma_vcb**2))
        ClassCosmo.pars['v_avg'] = np.trapezoid(vavgIntegrand * velArr, velArr)
        
        ###HAC: Computing Vcb Power Spectrum
        ClassCosmo.pars['k_vcb'] = kVel
        ClassCosmo.pars['theta_b'] = theta_b
        ClassCosmo.pars['theta_c'] = theta_c
        P_vcb = CosmologyIn.As * (kVel/0.05)**(CosmologyIn.ns-1) * (theta_b - theta_c)**2/kVel**2 * 2 * np.pi**2 / kVel**3
        
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

def time_at_redshift(ClassyCosmo,z):
    """
    Returns the age of the Universe (in Gyrs) corresponding to a given redshift.

    Parameters
    ----------
    ClassyCosmo: zeus21.runclass class
        Sets up Class cosmology.
    z: float
        Redshift.
    """
    background = ClassyCosmo.get_background()
    classy_t, classy_z = background['proper time [Gyr]'], background['z']
    classy_tinterp = interp1d(classy_z, classy_t)
    return classy_tinterp(z)

def redshift_at_time(ClassyCosmo,t):
    """
    Returns the redshift corresponding to a given age of the Universe (in Gyrs).

    Parameters
    ----------
    ClassyCosmo: zeus21.runclass class
        Sets up Class cosmology.
    t: float
        Age in Gyrs.
    """
    background = ClassyCosmo.get_background()
    classy_t, classy_z = background['proper time [Gyr]'], background['z']
    classy_tinterp = interp1d(classy_t, classy_z)
    return classy_tinterp(t)

def Hub(Cosmo_Parameters, z):
#Hubble(z) in km/s/Mpc
    return Cosmo_Parameters.h_fid * 100 * np.sqrt(Cosmo_Parameters.OmegaM * pow(1+z,3.)+Cosmo_Parameters.OmegaR * pow(1+z,4.)+Cosmo_Parameters.OmegaL)

def HubinvMpc(Cosmo_Parameters, z):
#H(z) in 1/Mpc
    return Hub(Cosmo_Parameters,z)/constants.c_kms

def Hubinvyr(Cosmo_Parameters,z):
#H(z) in 1/yr
    return Hub(Cosmo_Parameters,z)*constants.KmToMpc*constants.yrTos

def rho_baryon(Cosmo_Parameters,z):
#\rho_baryon in Msun/Mpc^3 as a function of z
    return Cosmo_Parameters.OmegaB * Cosmo_Parameters.rhocrit * pow(1+z,3.0)
    
def n_H(Cosmo_Parameters, z):
#density of hydrogen nuclei (neutral or ionized) in 1/cm^3
    return rho_baryon(Cosmo_Parameters, z) *( 1- Cosmo_Parameters.Y_He)/(constants.mH_GeV/constants.MsuntoGeV) / (constants.Mpctocm**3.0) 

#def n_baryon(Cosmo_Parameters, z):
##density of baryons in 1/cm^3
#    return rho_baryon(Cosmo_Parameters, z) / Cosmo_Parameters.mu_baryon_Msun / (constants.Mpctocm**3.0)



def Tcmb(ClassCosmo, z):
    T0CMB = ClassCosmo.T_cmb()
    return T0CMB*(1+z)

def Tadiabatic(CosmoParams, z):
    "Returns T_adiabatic as a function of z from thermodynamics in CLASS"
    return CosmoParams.Tadiabaticint(z)
def xefid(CosmoParams, z):
    "Returns fiducial x_e(z) w/o any sources. Uses thermodynamics in CLASS for z>15, and fixed below to avoid the tanh approx."
    _zcutCLASSxe = 15.
    _xecutCLASSxe = CosmoParams.xetanhint(_zcutCLASSxe)
    return CosmoParams.xetanhint(z) * np.heaviside(z - _zcutCLASSxe, 0.5) + _xecutCLASSxe * np.heaviside(_zcutCLASSxe - z, 0.5)

def adiabatic_index(z):
    "Returns adiabatic index (delta_Tad/delta) as a function of z. Fit from 1506.04152. to ~3% on z = 6 − 50)."
    return 0.58 - 0.005*(z-10.)


def MhofRad(Cosmo_Parameters,R):
    #convert input Radius in Mpc comoving to Mass in Msun
    return Cosmo_Parameters.constRM *pow(R, 3.0)

def RadofMh(Cosmo_Parameters,M):
    #convert input M halo in Msun radius in cMpc
    return pow(M/Cosmo_Parameters.constRM, 1/3.0)



def ST_HMF(Cosmo_Parameters, Mass, sigmaM, dsigmadM):
    A_ST = Cosmo_Parameters.Amp_ST
    a_ST = Cosmo_Parameters.a_ST
    p_ST = Cosmo_Parameters.p_ST
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST

    nutilde = np.sqrt(a_ST) * delta_crit_ST/sigmaM

    return -A_ST * np.sqrt(2./np.pi) * nutilde * (1. + nutilde**(-2.0*p_ST)) * np.exp(-nutilde**2/2.0) * (Cosmo_Parameters.rho_M0 / (Mass * sigmaM)) * dsigmadM


def Tink_HMF(Cosmo_Parameters, Mass, sigmaM, dsigmadM,z):
    #Tinker08 form of the HMF. All in physical (no h) units. Form from App.A of Yung+23 (2309.14408)
    f = f_GUREFT_physical(sigmaM,z)
    return f*(Cosmo_Parameters.rho_M0 / (Mass)) * np.abs(dsigmadM/sigmaM)

def f_GUREFT_physical(sigma,z):
    #Fit in eq A2 in Yung+23 (2309.14408), fit to z<20 (Implementation thanks to Aaron Yung). Physical because no h.
    #sigma(M,z) is the input, no growth here bc we use class with full sigma evolution
    k = np.array([ 1.37657725e-01, -1.00382125e-02,  1.02963559e-03,  1.06641384e+00,
        2.47557563e-02, -2.83342017e-03,  4.86693806e+00,  9.21235623e-02,
       -1.42628278e-02,  1.19837952e+00,  1.42966892e-03, -3.30740460e-04])
    A = lambda x: k[0] + k[1]*x  + k[2]*(x**2)
    a = lambda x: k[3] + k[4]*x  + k[5]*(x**2) 
    b = lambda x: k[6] + k[7]*x  + k[8]*(x**2)
    c = lambda x: k[9] + k[10]*x + k[11]*(x**2)
    sig = sigma
    #cap coefficients at z=20 to avoid extrapolation
    zuse = np.fmin(z,20.0)
    return A(zuse) * (((sig/b(zuse))**(-a(zuse))) + 1.0 ) * np.exp(-c(zuse)/(sig**2))

def PS_HMF_unnorm(Cosmo_Parameters, Mass, nu, dlogSdM):
    'Returns the Press-Schechter HMF (unnormalized since we will take ratios), given a halo Mass [Msun], nu = delta_tilde/S_tilde, with delta_tilde = delta_crit - delta_R, and variance S = sigma(M)^2 - sigma(R)^2. Used for 21cmFAST mode.'

    return nu * np.exp(-Cosmo_Parameters.a_corr_EPS*nu**2/2.0) * dlogSdM* (1.0 / Mass)
    #written so that dsigmasq/dM appears directly, since that is not modified by EPS, whereas sigma_tot^2 = sigma^2(M) - sigma^2(R). The sigma in denominator will be sigma_tot



class HMF_interpolator:
    "Class that builds an interpolator of the HMF. Returns an interpolator"

    def __init__(self, User_Parameters, Cosmo_Parameters, ClassCosmo):

        self._Mhmin = 1e5 #originally 1e5
        self._Mhmax = 1e14
        self._NMhs = np.floor(35*User_Parameters.precisionboost).astype(int)
        self.Mhtab = np.logspace(np.log10(self._Mhmin),np.log10(self._Mhmax),self._NMhs) # Halo mases in Msun
        self.RMhtab = RadofMh(Cosmo_Parameters, self.Mhtab)

        self.logtabMh = np.log(self.Mhtab)


        self._zmin=Cosmo_Parameters.zmin_CLASS
        self._zmax = Cosmo_Parameters.zmax_CLASS
        self._Nzs=np.floor(100*User_Parameters.precisionboost).astype(int)
        self.zHMFtab = np.linspace(self._zmin,self._zmax,self._Nzs)

        #check resolution
        if (Cosmo_Parameters.kmax_CLASS < 1.0/self.RMhtab[0]):
            print('Warning! kmax_CLASS may be too small! Run CLASS with higher kmax')

        self.sigmaMhtab = np.array([[ClassCosmo.sigma(RR,zz) for zz in self.zHMFtab] for RR in self.RMhtab])

        self._depsM=0.01 #for derivatives, relative to M
        self.dsigmadMMhtab = np.array([[(ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM*(1+self._depsM)),zz)-ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM*(1-self._depsM)),zz))/(MM*2.0*self._depsM) for zz in self.zHMFtab] for MM in self.Mhtab])


        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            #ADJUST BY HAND adjust sigmas to match theirs, since the CLASS TF they use is at a fixed cosmology from 21cmvFAST but the input cosmology is different
            self.sigmaMhtab*=np.sqrt(0.975)#/0.9845
            self.dsigmadMMhtab*=np.sqrt(0.975)#/0.9845

            #this correction is because 21cmFAST uses the dicke() function to compute growth, which is ~0.5% offset at high z. This offset makes our growth the same as dicke() for a Planck2018 cosmology. Has to be added separately to the growth(z) correction above since they come in different places
            _offsetgrowthdicke21cmFAST = 1-0.000248*(self.zHMFtab-5.)
            self.sigmaMhtab*=_offsetgrowthdicke21cmFAST
            self.dsigmadMMhtab*=_offsetgrowthdicke21cmFAST
            #Note that these two changes may be different if away from Planck2018



        self.HMFtab = np.zeros_like(self.sigmaMhtab)





        for iM, MM in enumerate(self.Mhtab):
            for iz, zz in enumerate(self.zHMFtab):
                sigmaM = self.sigmaMhtab[iM,iz]
                dsigmadM = self.dsigmadMMhtab[iM,iz]

                if(Cosmo_Parameters.HMF_CHOICE == 'ST'):
                    self.HMFtab[iM,iz] = ST_HMF(Cosmo_Parameters, MM, sigmaM, dsigmadM)
                elif(Cosmo_Parameters.HMF_CHOICE == 'Yung'):
                    self.HMFtab[iM,iz] = Tink_HMF(Cosmo_Parameters, MM, sigmaM, dsigmadM,zz)
                else:
                    print('ERROR, use a correct Cosmo_Parameters.HMF_CHOICE')
                    self.HMFtab[iM,iz] = 0.0





        _HMFMIN = np.exp(-300.) #min HMF to avoid overflowing
        logHMF_ST_trim = self.HMFtab
        logHMF_ST_trim[np.array(logHMF_ST_trim <= 0.)] = _HMFMIN
        logHMF_ST_trim = np.log(logHMF_ST_trim)


        self.fitMztab = [np.log(self.Mhtab), self.zHMFtab]
        self.logHMFint = RegularGridInterpolator(self.fitMztab, logHMF_ST_trim, bounds_error = False, fill_value = -np.inf) ###HAC: Changed to -np.inf so HMFint = exp(-np.inf)= zero to fix nans in sfrd.py

        self.sigmaintlog = RegularGridInterpolator(self.fitMztab, self.sigmaMhtab, bounds_error = False, fill_value = np.nan)# no need to log since it doesnt vary dramatically

        self.dsigmadMintlog = RegularGridInterpolator(self.fitMztab, self.dsigmadMMhtab, bounds_error = False, fill_value = np.nan)


        #also build an interpolator for sigma(R) of the R we integrate over (for CD and EoR). These R >> Rhalo typically, so need new table.
        self.sigmaofRtab = np.array([[ClassCosmo.sigma(RR,zz) for zz in self.zHMFtab] for RR in Cosmo_Parameters._Rtabsmoo])
        self.fitRztab = [np.log(Cosmo_Parameters._Rtabsmoo), self.zHMFtab]
        self.sigmaRintlog = RegularGridInterpolator(self.fitRztab, self.sigmaofRtab, bounds_error = False, fill_value = np.nan) #no need to log either




    def HMF_int(self, Mh, z):
        "Interpolator to find HMF(M,z), designed to take a single z but an array of Mh in Msun"
        _logMh = np.log(Mh)

        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])

        return np.exp(self.logHMFint(inarray) )



    def sigma_int(self,Mh,z):
        "Interpolator to find sigma(M,z), designed to take a single z but an array of Mh in Msun"
        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])
        return self.sigmaintlog(inarray)

    def sigmaR_int(self,RR,z):
        "Interpolator to find sigma(RR,z), designed to take a single z but an array of RR in cMpc"
        _logRR = np.log(RR)
        logRRvec = np.asarray([_logRR]) if np.isscalar(_logRR) else np.asarray(_logRR)
        inarray = np.array([[LR,z] for LR in logRRvec])
        return self.sigmaRintlog(inarray)


    def dsigmadM_int(self,Mh,z):
        "Interpolator to find dsigma/dM(M,z), designed to take a single z but an array of Mh in Msun. Used in 21cmFAST mode"
        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])
        return self.dsigmadMintlog(inarray)


def growth(Cosmo_Parameters, z):
    "Scale-independent growth factor, interpolated from CLASS"
    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    if (Cosmo_Parameters.Flag_emulate_21cmfast==True):
        _offsetgrowthdicke21cmFAST = 1-0.000248*(zlist-5.) #as in HMF, to fix growth. have to do it independently since it depends on z.
        return Cosmo_Parameters.growthint(zlist) * _offsetgrowthdicke21cmFAST
    else:
        return Cosmo_Parameters.growthint(zlist)


def dgrowth_dz(CosmoParams, z):
    "Derivative of growth factor growth() w.r.t. z"
    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    dzlist = zlist*0.001
    return (growth(CosmoParams, z+dzlist)-growth(CosmoParams, z-dzlist))/(2.0*dzlist)


def redshift_of_chi(CosmoParams, z):
    "Returns z(chi) for any input comoving distance from today chi in Mpc"
    return CosmoParams.zfofRint(z)



def T021(Cosmo_Parameters, z):
    "Prefactor in mK to T21 that only depends on cosmological parameters and z. Eg Eq.(21) in 2110.13919"
    return 34 * pow((1+z)/16.,0.5) * (Cosmo_Parameters.omegab/0.022) * pow(Cosmo_Parameters.omegam/0.14,-0.5)


#UNUSED bias, just for reference
def bias_ST(Cosmo_Parameters, sigmaM):
    # from https://arxiv.org/pdf/1007.4201.pdf Table 1
    a_ST = Cosmo_Parameters.a_ST
    p_ST = Cosmo_Parameters.p_ST
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST
    nu = delta_crit_ST/sigmaM
    nutilde = np.sqrt(a_ST) * nu
    
    return 1.0 + (nutilde**2 - 1.0 + 2. * p_ST/(1.0 + nutilde**(2. * p_ST) ) )/delta_crit_ST 

def bias_Tinker(Cosmo_Parameters, sigmaM):
    #from https://arxiv.org/pdf/1001.3162.pdf, Delta=200
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST
    nu = delta_crit_ST/sigmaM
    
    #Tinker fit
    _Deltahalo = 200;
    _yhalo = np.log10(_Deltahalo)
    _Abias = 1.0 + 0.24 * _yhalo * np.exp(-(4.0/_yhalo)**4.)
    _abias = 0.44*_yhalo-0.88
    _Bbias = 0.183
    _bbias = 1.5
    _Cbias = 0.019 + 0.107 * _yhalo + 0.19 * np.exp(-(4.0/_yhalo)**4.)
    _cbias = 2.4

    return 1.0 - _Abias*(nu**_abias/(nu**_abias + delta_crit_ST**_abias)) + _Bbias * nu**_bbias + _Cbias * nu**_cbias

#UNUSED:
# def interp2Dlinear_only_y(arrayxy, arrayz, x, y):
#     "2D interpolator where the x axis is assumed to be an array identical to the trained x. That is, an array of 1D linear interpolators. arrayxy is [x,y]. arrayz is result. x is the x input (=arrayxy[0]),  and y the y input. Returns z result (array)"
#     if((x != arrayxy[0]).all()):
#         print('ERROR on interp2Dlinear_only_y, x need be the same in interp and input')
#         return -1
#     Ny = len(arrayxy[1])
#     ymin, ymax = arrayxy[1][[0,-1]]
#     if((y > ymax or y<ymin).all()):
#         print('EXTRAPOLATION on interp2Dlinear_only_y on y axis. max={} curr={}'.format(ymax, y))
#
#     ystep = (ymax-ymin)/(Ny-1.)
#     ysign = np.sign(ystep).astype(int)
#
#     iy = np.floor((y-ymin)/ystep).astype(int)
#     iy=np.fmin(iy,Ny-1-(ysign+1)//2) #if positive ysign stop 1 lower
#     iy=np.fmax(iy,0-(ysign-1)//2) #if negative ysign stop 1 higher
#
#     y1 = ymin + iy * ystep
#     y2 = y1 + ystep
#
#     fy1 = arrayz[:,iy]
#     fy2 = arrayz[:,iy+ysign]
#
#     fy = fy1 + (fy2-fy1)/ystep * (y - y1)
#     return fy
