"""

Compute UVLFs given our SFR and HMF models.

Author: Julian B. Muñoz
UT Austin - June 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Bug fix by Emily Bregou
UT Austin - June 2025
"""

from . import cosmology
from . import constants
from .sfrd import SFR_II, SFR_III
from .cosmology import bias_Tinker

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d






def MUV_of_SFR(SFRtab, kappaUV):
    'returns MUV, uses SFR. Dust added later in loglike.'
    #convert SFR to MUVs
    LUVtab = SFRtab/kappaUV
    MUVtab = 51.63 - 2.5 * np.log10(LUVtab) #AB magnitude
    return MUVtab


#and combine to get UVLF:
def UVLF_binned(Astro_Parameters,Cosmo_Parameters,HMF_interpolator, zcenter, zwidth, MUVcenters, MUVwidths, DUST_FLAG=True, RETURNBIAS = False):
    'Binned UVLF in units of 1/Mpc^3/mag, for bins at <zcenter> with a Gaussian width zwidth, centered at MUV centers with tophat width MUVwidths. z width only in HMF since that varies the most rapidly. If flag RETURNBIAS set to true it returns number-avgd bias instead of UVLF, still have to divide by UVLF'
    
    if(constants.NZ_TOINT>1):
        DZ_TOINT = np.linspace(-np.sqrt(constants.NZ_TOINT/3.),np.sqrt(constants.NZ_TOINT/3.),constants.NZ_TOINT) #in sigmas around zcenter
    else:
        DZ_TOINT = np.array([0.0])
    WEIGHTS_TOINT = np.exp(-DZ_TOINT**2/2.)/np.sum(np.exp(-DZ_TOINT**2/2.)) #assumed Gaussian in z, fair



    
    SFRlist = SFR_II(Astro_Parameters,Cosmo_Parameters,HMF_interpolator, HMF_interpolator.Mhtab, zcenter, zcenter)
    sigmaUV = Astro_Parameters.sigmaUV
    
    if (constants.FLAG_RENORMALIZE_LUV == True): #lower the LUV (or SFR) to recover the true avg, not log-avg
        SFRlist/= np.exp((np.log(10)/2.5*sigmaUV)**2/2.0)
        
    MUVbarlist = MUV_of_SFR(SFRlist, Astro_Parameters._kappaUV) #avg for each Mh
    MUVbarlist = np.fmin(MUVbarlist,constants._MAGMAX)
    

    if(RETURNBIAS==True): # weight by bias
        biasM = np.array([bias_Tinker(Cosmo_Parameters, HMF_interpolator.sigma_int(HMF_interpolator.Mhtab,zcenter+dz*zwidth)) for dz in DZ_TOINT])
    else: # do not weight by bias
        biasM = np.ones_like(WEIGHTS_TOINT)
 
        
    HMFtab = np.array([HMF_interpolator.HMF_int(HMF_interpolator.Mhtab,zcenter+dz*zwidth) for dz in DZ_TOINT])
    HMFcurr = np.sum(WEIGHTS_TOINT * HMFtab.T * biasM.T,axis=1)

    #cannot directly 'dust' the theory since the properties of the IRX-beta relation are calibrated on observed MUV. Recursion instead:
    currMUV = MUVbarlist
    if(DUST_FLAG==True):
        currMUV2 = np.ones_like(currMUV)
        while(np.sum(np.abs((currMUV2-currMUV)/currMUV)) > 0.02):
            currMUV2 = currMUV
            currMUV = MUVbarlist + AUV(Astro_Parameters,zcenter,currMUV)
           
    
    MUVcuthi = MUVcenters +  MUVwidths/2.
    MUVcutlo = MUVcenters -  MUVwidths/2.
    
    xhi = np.subtract.outer(MUVcuthi, currMUV)/(np.sqrt(2) * sigmaUV)
    xlo = np.subtract.outer(MUVcutlo, currMUV )/(np.sqrt(2) * sigmaUV)
    weights = (erf(xhi) - erf(xlo)).T/(2.0 * MUVwidths)
    
    UVLF_filtered = np.trapz(weights.T * HMFcurr, HMF_interpolator.Mhtab, axis=-1)

    if(Astro_Parameters.USE_POPIII==False):
        return UVLF_filtered
    else:
        _J21interptemp = interp1d(np.linspace(0,100,3), np.zeros(3), kind = 'linear', bounds_error = False, fill_value = 0,) #TODO: how to deal with J21, requires running get_21_coefficients
        SFRlist_III = SFR_III(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, HMF_interpolator.Mhtab, _J21interptemp, zcenter, zcenter, Cosmo_Parameters.vcb_avg)
    
        MUVbarlist_III = MUV_of_SFR(SFRlist_III, Astro_Parameters._kappaUV_III) #avg for each Mh
        MUVbarlist_III = np.fmin(MUVbarlist_III,constants._MAGMAX)
          
        #and the same for popIII, TODO: ignore dust for pop3 for now
        xhi = np.subtract.outer(MUVcuthi, MUVbarlist_III)/(np.sqrt(2) * sigmaUV)
        xlo = np.subtract.outer(MUVcutlo, MUVbarlist_III)/(np.sqrt(2) * sigmaUV)
        weights = (erf(xhi) - erf(xlo)).T/(2.0 * MUVwidths)

        UVLF_filtered_III = np.trapz(weights.T * HMFcurr, HMF_interpolator.Mhtab, axis=-1)
    
        return UVLF_filtered, UVLF_filtered_III





#####Here the dust attenuation
def AUV(Astro_Parameters, z, MUV, HIGH_Z_DUST = True, _zmaxdata=8.0):
    'Average attenuation A as a function of OBSERVED z and magnitude. If using on theory iterate until convergence. HIGH_Z_DUST is whether to do dust at higher z than 0 or set to 0. Fix at \beta(z=8) result if so'
    
    betacurr = beta(z,MUV)
    
    C0, C1 = Astro_Parameters.C0dust, Astro_Parameters.C1dust
    
    sigmabeta = 0.34 #from Bouwens 2014
    
    Auv = C0 + 0.2*np.log(10)*sigmabeta**2 * C1**2 + C1 * betacurr
    Auv=Auv.T
    if not (HIGH_Z_DUST):
        Auv*=np.heaviside(_zmaxdata - z,0.5)
    Auv=Auv.T
    return np.fmax(Auv, 0.0)

def beta(z, MUV):
    'Color as a function of redshift and mag, interpolated from Bouwens 2013-14 data.'

    zdatbeta = [2.5,3.8,5.0,5.9,7.0,8.0]
    betaMUVatM0 = [-1.7,-1.85,-1.91,-2.00,-2.05,-2.13]
    dbeta_dMUV = [-0.20,-0.11,-0.14,-0.20,-0.20,-0.15]

    _MUV0 = -19.5
    _c = -2.33

    betaM0 = np.interp(z, zdatbeta, betaMUVatM0, left=betaMUVatM0[0], right=betaMUVatM0[-1])
    dbetaM0 = (MUV - _MUV0).T * np.interp(z, zdatbeta, dbeta_dMUV, left=dbeta_dMUV[0], right=dbeta_dMUV[-1])
    
    sol1 = (betaM0-_c) * np.exp(dbetaM0/(betaM0-_c))+_c #for MUV > MUV0
    sol2 = dbetaM0 + betaM0 #for MUV < MUV0
    
    return sol1.T * np.heaviside(MUV - _MUV0, 0.5) + sol2.T * np.heaviside(_MUV0 - MUV, 0.5)
