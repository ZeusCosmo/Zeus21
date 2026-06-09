"""

Cosmology functions and helper tools related with cosmology

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Emilie Thelie, Sarah Libanore
UT Austin - April 2026
BGU - June 2026
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from . import constants

def time_at_redshift(ClassyCosmo,z):
    """
    Returns the age of the Universe (in Gyrs) corresponding to a given redshift.

    Parameters
    ----------
    ClassyCosmo: zeus21.runclass class
        Sets up Class cosmology.
    z: float
        Redshift.

    Returns
    -------
    float
        Age of the Universe in Gyrs.
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

    Returns
    -------
    float   
        Redshift corresponding to the given age of the Universe.
    """

    background = ClassyCosmo.get_background()
    classy_t, classy_z = background['proper time [Gyr]'], background['z']
    classy_tinterp = interp1d(classy_t, classy_z)
    
    return classy_tinterp(t)




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
    """
    Hubble parameter H(z).

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Hubble parameter H(z) in km/s/Mpc.
    """

    return Cosmo_Parameters.h_fid * 100 * np.sqrt(Cosmo_Parameters.OmegaM * pow(1+z,3.)+Cosmo_Parameters.OmegaR * pow(1+z,4.)+Cosmo_Parameters.OmegaL)


def HubinvMpc(Cosmo_Parameters, z):
    """
    Converts Hubble parameter H(z) in inverse length units (1/Mpc).

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Hubble parameter H(z) in 1/Mpc.
    """

    return Hub(Cosmo_Parameters,z)/constants.c_kms


def Hubinvyr(Cosmo_Parameters, z):
    """
    Converts Hubble parameter H(z) in inverse time units (1/yr).

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Hubble parameter H(z) in 1/yr.
    """

    return Hub(Cosmo_Parameters,z)*constants.KmToMpc*constants.yrTos


def rho_baryon(Cosmo_Parameters, z):
    """
    Baryon density rho_baryon(z).

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Baryon density rho_baryon(z) in Msun/Mpc^3.
    """

    return Cosmo_Parameters.OmegaB * Cosmo_Parameters.rhocrit * pow(1+z,3.0)
    

def n_H(Cosmo_Parameters, z):
    """
    Number density of hydrogen nuclei (including both neutral or ionized).

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Number density of hydrogen nuclei in 1/cm^3.
    """

    return rho_baryon(Cosmo_Parameters, z) *( 1- Cosmo_Parameters.Y_He)/(constants.mH_GeV/constants.MsuntoGeV) / (constants.Mpctocm**3.0) 


def Tcmb(ClassCosmo, z):
    """
    CMB temperature T(z).

    Parameters
    ----------
    ClassCosmo : ClassCosmo
        CLASS cosmology object.
    z : float
        Redshift.

    Returns
    -------
    float
        CMB temperature T(z) in K.
    """

    T0CMB = ClassCosmo.T_cmb()

    return T0CMB*(1+z)
        

def Tadiabatic(CosmoParams, z):
    """
    Returns T_adiabatic as a function of z from thermodynamics in CLASS.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Adiabatic temperature T_adiabatic(z).
    """

    return CosmoParams.Tadiabaticint(z)


def xefid(CosmoParams, z):
    """
    Electron fraction x_e(z) without any sources. 
    Uses thermodynamics in CLASS for z>15, and fixed below to avoid the tanh approximation.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    z : float
        Redshift.

    Returns
    -------
    float
        Fiducial x_e(z).
    """

    _zcutCLASSxe = 15.
    _xecutCLASSxe = CosmoParams.xetanhint(_zcutCLASSxe)

    return CosmoParams.xetanhint(z) * np.heaviside(z - _zcutCLASSxe, 0.5) + _xecutCLASSxe * np.heaviside(_zcutCLASSxe - z, 0.5)


def adiabatic_index(z):
    """
    Returns adiabatic index (delta_Tad/delta) as a function of z. Fit from 1506.04152. to ~3% on z = 6 − 50).

    Parameters
    ----------
    z : float
        Redshift.

    Returns
    -------
    float
        Adiabatic index (delta_Tad/delta).
    """

    return 0.58 - 0.005*(z-10.)


def MhofRad(Cosmo_Parameters, R):
    """
    Convert input comoving Radius to virial Mass.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    R : float
        Comoving radius in cMpc.

    Returns
    -------
    float
        Mass in Msun.
    """
    
    return Cosmo_Parameters.constRM *pow(R, 3.0)


def RadofMh(Cosmo_Parameters, M):
    """
    Convert input virial Mass to comoving Radius.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    M : float
        Virial mass in Msun.

    Returns
    -------
    float
        Comoving radius in cMpc.
    """
    
    return pow(M/Cosmo_Parameters.constRM, 1/3.0)


def ST_HMF(Cosmo_Parameters, Mass, sigmaM, dsigmadM):
    """
    Sheth-Tormen Halo Mass Function.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    Mass : float
        Halo mass in Msun.
    sigmaM : float
        Variance of the density field on the scale of the halo mass.
    dsigmadM : float
        Derivative of sigmaM with respect to Mass.

    Returns
    -------
    float
        HMF value in 1/Mpc^3/Msun.
    """
    
    A_ST = Cosmo_Parameters.Amp_ST
    a_ST = Cosmo_Parameters.a_ST
    p_ST = Cosmo_Parameters.p_ST
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST

    nutilde = np.sqrt(a_ST) * delta_crit_ST/sigmaM

    return -A_ST * np.sqrt(2./np.pi) * nutilde * (1. + nutilde**(-2.0*p_ST)) * np.exp(-nutilde**2/2.0) * (Cosmo_Parameters.rho_M0 / (Mass * sigmaM)) * dsigmadM


def Tink_HMF(Cosmo_Parameters, Mass, sigmaM, dsigmadM, z):
    """
    Tinker 2008 Halo Mass Function.
    All in physical (no h) units. 
    Form from App.A of Yung+23 (2309.14408).

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    Mass : float
        Halo mass in Msun.
    sigmaM : float
        Variance of the density field on the scale of the halo mass at redshift z.
    dsigmadM : float
        Derivative of sigmaM with respect to Mass at redshift z.
    z : float
        Redshift.

    Returns
    -------
    float
        HMF value in 1/Mpc^3/Msun.
    """
    
    f = f_GUREFT_physical(sigmaM, z)

    return f*(Cosmo_Parameters.rho_M0 / (Mass)) * np.abs(dsigmadM/sigmaM)


def f_GUREFT_physical(sigmaM, z):
    """
    Fit in eq A2 in Yung+23 (2309.14408) to z < 20.
    Required by the Tinker 2008 HMF; all in physical units (no h). 
    Implementation thanks to Aaron Yung.

    Parameters
    ----------
    sigmaM : float
        Variance of the density field on the scale of the halo mass at redshift z.
    z : float
        Redshift.

    Returns
    -------
    float
        HMF value in 1/Mpc^3/Msun.
    """
        
    k = np.array([ 1.37657725e-01, -1.00382125e-02,  1.02963559e-03,  1.06641384e+00,
        2.47557563e-02, -2.83342017e-03,  4.86693806e+00,  9.21235623e-02,
       -1.42628278e-02,  1.19837952e+00,  1.42966892e-03, -3.30740460e-04])
    
    A = lambda x: k[0] + k[1]*x  + k[2]*(x**2)
    a = lambda x: k[3] + k[4]*x  + k[5]*(x**2) 
    b = lambda x: k[6] + k[7]*x  + k[8]*(x**2)
    c = lambda x: k[9] + k[10]*x + k[11]*(x**2)

    #cap coefficients at z=20 to avoid extrapolation
    zuse = np.fmin(z,20.0)

    return A(zuse) * (((sigmaM/b(zuse))**(-a(zuse))) + 1.0 ) * np.exp(-c(zuse)/(sigmaM**2))


def PS_HMF_unnorm(Cosmo_Parameters, Mass, nu, dlogSdM):
    """
    Unnormalized Press-Schechter HMF.
    Used to emulate 21cmFAST. 

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters.
    Mass : float
        Halo mass in Msun.
    nu : float
        Peak height, defined as nu = delta_tilde/S_tilde, with delta_tilde = delta_crit - delta_R, and variance S = sigma(M)^2 - sigma(R)^2.
    dlogSdM : float
        Derivative of log(S) with respect to Mass, where S = sigma(M)^2 - sigma(R)^2.  
    
    Returns
    -------
    float
        HMF value in 1/Mpc^3/Msun.
    """
        
    return nu * np.exp(-Cosmo_Parameters.a_corr_EPS*nu**2/2.0) * dlogSdM* (1.0 / Mass)


class HMF_interpolator:
    """
    Class that builds an interpolator of the HMF as function of the halo mass and redshift. 

    Parameters
    ----------

    User_Parameters : User_Parameters
        User parameters, used to set the resolution of the HMF table.
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters, used to compute the HMF table with CLASS.
    
    Attributes
    -------
    HMF_int : RegularGridInterpolator
        Interpolator for HMF value, takes (Mass, z) as arguments
    sigma_int : RegularGridInterpolator
        Interpolator for sigma value, takes (Mass, z) as arguments
    sigmaR_int : RegularGridInterpolator
        Interpolator for sigma(R) value, takes (R, z) as arguments 
    dsigmadM_int : RegularGridInterpolator
        Interpolator for dsigma/dM value, takes (Mass, z) as arguments
    """

    def __init__(self, User_Parameters, Cosmo_Parameters):

        self._Mhmin = 1e5 # minimum halo mass in Msun
        self._Mhmax = 1e14 # maximum halo mass in Msun
        self._NMhs = np.floor(35*User_Parameters.precisionboost).astype(int) # number of halo mass points in the table, set by precisionboost
        self.Mhtab = np.logspace(np.log10(self._Mhmin),np.log10(self._Mhmax),self._NMhs) # halo mass table in Msun
        self.logtabMh = np.log(self.Mhtab) # log of halo mass table, used for interpolation since the HMF varies more smoothly in log(M)
        
        self.RMhtab = RadofMh(Cosmo_Parameters, self.Mhtab) # comoving radius corresponding to the halo mass table, in cMpc


        self._zmin=Cosmo_Parameters.zmin_CLASS # minimum redshift for the HMF table, set by CLASS
        self._zmax = Cosmo_Parameters.zmax_CLASS # maximum redshift for the HMF table, set by CLASS
        self._Nzs=np.floor(100*User_Parameters.precisionboost).astype(int) # number of redshift points in the table, set by precisionboost. Note that the HMF is very steep at high z, so we need more points than for other tables to get good interpolation.
        self.zHMFtab = np.linspace(self._zmin,self._zmax,self._Nzs) # redshift table for the HMF

        # check resolution: make sure that the kmax_CLASS is high enough to resolve the small scales corresponding to the smallest halos. If not, warn the user
        if (Cosmo_Parameters.kmax_CLASS < 1.0/self.RMhtab[0]):
            print('Warning! kmax_CLASS may be too small! Run CLASS with higher kmax')

        # sigma(M,z) table, computed from CLASS
        self.sigmaMhtab = np.array([[Cosmo_Parameters.ClassCosmo.sigma(RR,zz) for zz in self.zHMFtab] for RR in self.RMhtab]) 

        # derivative of sigma with respect to M
        self._depsM = 0.01 # step 
        self.dsigmadMMhtab = np.array([[(Cosmo_Parameters.ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM*(1+self._depsM)),zz)-Cosmo_Parameters.ClassCosmo.sigma(RadofMh(Cosmo_Parameters, MM*(1-self._depsM)),zz))/(MM*2.0*self._depsM) for zz in self.zHMFtab] for MM in self.Mhtab])

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            print('WARNING!' \
            'You set Flag_emulate_21cmfast == True.' \
            'HMF_interpolator applyies corrections to sigma(M) and growth(z) to match the 21cmFAST cosmology. ' \
            'These corrections are only valid for a Planck2018 cosmology, and may be different if you use a different cosmology.')

            # CORRECTION #1 
            # 21cmFAST uses a fixed cosmology to compute the transfer function, which is different from our input Planck2018 cosmology. This leads to a mismatch in sigma(M); to fix it, we adjust our sigma(M) in a redshift-independent way to match theirs 
            # NOTE! This factor should be corrected if your cosmology is not Planck2018
            self.sigmaMhtab*=np.sqrt(0.975)
            self.dsigmadMMhtab*=np.sqrt(0.975)

            # CORRECTION #2
            # 21cmFAST uses the dicke() function to compute growth, which is ~0.5% offset at high z. This offset makes our growth the same as dicke() for a Planck2018 cosmology 
            # NOTE! This factor should be corrected if your cosmology is not Planck2018
            _offsetgrowthdicke21cmFAST = 1-0.000248*(self.zHMFtab-5.)
            self.sigmaMhtab*=_offsetgrowthdicke21cmFAST
            self.dsigmadMMhtab*=_offsetgrowthdicke21cmFAST

        self.HMFtab = np.zeros_like(self.sigmaMhtab)

        # fill HMF table (Mh,z) using either ST or Tinker, depending on the choice in Cosmo_Parameters, using the sigma(M,z) and dsigma/dM(M,z) from CLASS.
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

        # set min HMF to avoid overflowing
        _HMFMIN = np.exp(-300.) 
        logHMF_ST_trim = self.HMFtab
        logHMF_ST_trim[np.array(logHMF_ST_trim <= 0.)] = _HMFMIN
        logHMF_ST_trim = np.log(logHMF_ST_trim)

        # interpolator for log(HMF) as a function of log(Mh) and z, with bounds_error=False and fill_value=-inf to avoid extrapolation issues
        self.fitMztab = [np.log(self.Mhtab), self.zHMFtab]
        self.logHMFint = RegularGridInterpolator(self.fitMztab, logHMF_ST_trim, bounds_error = False, fill_value = -np.inf) 

        # interpolator for sigma(M,z) as a function of log(Mh) and z, with bounds_error=False and fill_value=np.nan to avoid extrapolation issues
        self.sigmaintlog = RegularGridInterpolator(self.fitMztab, self.sigmaMhtab, bounds_error = False, fill_value = np.nan)

        # interpolator for dsigma/dM(M,z) as a function of log(Mh) and z, with bounds_error=False and fill_value=np.nan to avoid extrapolation issues
        self.dsigmadMintlog = RegularGridInterpolator(self.fitMztab, self.dsigmadMMhtab, bounds_error = False, fill_value = np.nan)

        # interpolator for sigma(R); typically, R >> Rhalo, so we need a new table
        self.sigmaofRtab = np.array([[Cosmo_Parameters.ClassCosmo.sigma(RR,zz) for zz in self.zHMFtab] for RR in Cosmo_Parameters._Rtabsmoo])
        self.fitRztab = [np.log(Cosmo_Parameters._Rtabsmoo), self.zHMFtab]
        self.sigmaRintlog = RegularGridInterpolator(self.fitRztab, self.sigmaofRtab, bounds_error = False, fill_value = np.nan) 


    def HMF_int(self, Mh, z):
        """
        Interpolator to find HMF(M,z).

        Parameters
        ----------
        Mh : float or array
            Halo mass in Msun. Can be a single value or an array of values.
        z : float
        
        Returns
        -------
        float
            Interpolator for HMF value, takes (Mass, z) as arguments.
        """

        _logMh = np.log(Mh)

        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])

        return np.exp(self.logHMFint(inarray) )


    def sigma_int(self, Mh, z):
        """
        Interpolator to find sigma(M,z).

        Parameters
        ----------
        Mh : float or array
            Halo mass in Msun. Can be a single value or an array of values.
        z : float
        
        Returns
        -------
        float
            Interpolator for sigma value, takes (Mass, z) as arguments.
        """

        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])

        return self.sigmaintlog(inarray)


    def sigmaR_int(self, RR, z):
        """
        Interpolator to find sigma(R,z).

        Parameters
        ----------
        RR : float or array
            Comoving distance in Mpc. Can be a single value or an array of values.
        z : float
        
        Returns
        -------
        float
            Interpolator for sigma value, takes (R, z) as arguments.
        """
        _logRR = np.log(RR)
        logRRvec = np.asarray([_logRR]) if np.isscalar(_logRR) else np.asarray(_logRR)
        inarray = np.array([[LR,z] for LR in logRRvec])

        return self.sigmaRintlog(inarray)


    def dsigmadM_int(self, Mh, z):
        """
        Interpolator to find dsigma/dM.

        Parameters
        ----------
        Mh : float or array
            Halo mass in Msun. Can be a single value or an array of values.
        z : float
        
        Returns
        -------
        float
            Interpolator for dsigma/dM value, takes (Mass, z) as arguments.
        """

        _logMh = np.log(Mh)
        logMhvec = np.asarray([_logMh]) if np.isscalar(_logMh) else np.asarray(_logMh)
        inarray = np.array([[LM,z] for LM in logMhvec])

        return self.dsigmadMintlog(inarray)


def growth(Cosmo_Parameters, z):
    """
    Interpolator to find the scale-independent growth factor.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters, used to compute the growth factor with CLASS.
    z : float
        Redshift.
    
    Returns
    -------
    float
        Interpolator for the scale-independent growth factor, takes z as argument.
    """

    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    if (Cosmo_Parameters.Flag_emulate_21cmfast==True):
        print('WARNING!' \
        'You set Flag_emulate_21cmfast == True.' \
        'growth() applyies corrections to match the 21cmFAST cosmology. ' \
        'These corrections are only valid for a Planck2018 cosmology, and may be different if you use a different cosmology.')
        
        # 21cmFAST uses the dicke() function to compute growth, which is ~0.5% offset at high z. This offset makes our growth the same as dicke() for a Planck2018 cosmology 
        # NOTE! This factor should be corrected if your cosmology is not Planck2018
        _offsetgrowthdicke21cmFAST = 1-0.000248*(zlist-5.)

        return Cosmo_Parameters.growthint(zlist) * _offsetgrowthdicke21cmFAST
    
    else:
        return Cosmo_Parameters.growthint(zlist)


def dgrowth_dz(CosmoParams, z):
    """
    Derivative of growth factor w.r.t. z.

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters, used to compute the growth factor with CLASS.
    z : float
        Redshift.
    
    Returns
    -------
    float
        dgrowth/dz
    """

    zlist = np.asarray([z]) if np.isscalar(z) else np.asarray(z)
    dzlist = zlist*0.001

    return (growth(CosmoParams, z+dzlist)-growth(CosmoParams, z-dzlist))/(2.0*dzlist)


def T021(Cosmo_Parameters, z):
    """
    Prefactor in mK to T21 that only depends on cosmological parameters and z. See Eq.(21) in 2110.13919

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters, used to compute the growth factor with CLASS.
    z : float
        Redshift.
    
    Returns
    -------
    float
        Prefactor in mK to T21
    """

    return 34 * pow((1+z)/16.,0.5) * (Cosmo_Parameters.omegab/0.022) * pow(Cosmo_Parameters.omegam/0.14,-0.5)


def bias_ST(Cosmo_Parameters, sigmaM):
    """
    Bias of halos in the Sheth-Tormen model. 
    See https://arxiv.org/pdf/1007.4201.pdf Table 1

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters, used to compute the growth factor with CLASS.
    sigmaM : float
        Variance of the matter density field smoothed on a scale corresponding to the halo mass.

    Returns
    -------
    float
        Halo bias
    """

    a_ST = Cosmo_Parameters.a_ST
    p_ST = Cosmo_Parameters.p_ST
    delta_crit_ST = Cosmo_Parameters.delta_crit_ST
    nu = delta_crit_ST/sigmaM
    nutilde = np.sqrt(a_ST) * nu
    
    return 1.0 + (nutilde**2 - 1.0 + 2. * p_ST/(1.0 + nutilde**(2. * p_ST) ) )/delta_crit_ST 


def bias_Tinker(Cosmo_Parameters, sigmaM):
    """
    Bias of halos in the Tinker model. See https://arxiv.org/pdf/1001.3162.pdf for Delta = 200

    Parameters
    ----------
    Cosmo_Parameters : Cosmo_Parameters
        Cosmological parameters, used to compute the growth factor with CLASS.
    sigmaM : float
        Variance of the matter density field smoothed on a scale corresponding to the halo mass.

    Returns
    -------
    float
        Halo bias
    """

    delta_crit_ST = Cosmo_Parameters.delta_crit_ST # critical density for collapse
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

