"""

Xray structure, helper functions, and definitions

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

import numpy as np
from . import constants
from .cosmology import n_H, HubinvMpc


class Xray_class:
    "Class containing the X-ray functions that we want to pass to main calculation"

    def __init__(self, User_Parameters, Cosmo_Parameters):

        self.atomfractions = np.array([1,Cosmo_Parameters.x_He]) #fraction of baryons in HI and HeI, assumed to just be the avg cosmic
#        self.atomfractions = np.array([Cosmo_Parameters.f_H,Cosmo_Parameters.f_He]) #fraction of baryons in HI and HeI, assumed to just be the avg cosmic
        self.atomEnIon = np.array([constants.EN_ION_HI, constants.EN_ION_HeI]) #threshold energies for each, in eV
        self.TAUMAX=100. #max optical depth, cut to 0 after to avoid overflows


    def optical_depth(self, User_Parameters, Cosmo_Parameters, En,z,zp):
        "Function that calculates the optical depth for a photon of energy En/eV from z to zp"
        Nzinttau = np.floor(10*User_Parameters.precisionboost).astype(int)
        #surprisingly it converges very quickly, since things are smooth functions of nu/z. Warning, make sure to tweak if SED is not a powerlaw!

        Envec = np.asarray([En]) if np.isscalar(En) else np.asarray(En)

        zinttau = np.linspace(z,zp,Nzinttau)


        Eninttautab = np.outer((1+zinttau)/(1+z) , Envec)

        sigmatot = self.atomfractions[0] * sigma_HI(Eninttautab)
        sigmatot += self.atomfractions[1] * sigma_HeI(Eninttautab)
        sigmatot = sigmatot.T #to broadcast below
        # divided by factor of H(z')(1+z') because of variable of integration change from proper distance to redshift
        integrand = 1.0/HubinvMpc(Cosmo_Parameters, zinttau)/(1+zinttau) * sigmatot * n_H(Cosmo_Parameters, zinttau) * constants.Mpctocm
#        integrand = 1.0/HubinvMpc(Cosmo_Parameters, zinttau)/(1+zinttau) * sigmatot * n_baryon(Cosmo_Parameters, zinttau) * constants.Mpctocm
        taulist = np.trapezoid(integrand, zinttau, axis=1)

        #OLD: kept for reference only.
        # taulist = 1.0*np.zeros_like(Envec)
        # for iE, Energy in enumerate(Envec):
        #     Eninttau = (1+zinttau)/(1+z) * Energy
        #     sigmatot = self.atomfractions[0] * sigma_HI(Eninttau)
        #     sigmatot += self.atomfractions[1] * sigma_HeI(Eninttau)
        #     #we ignore HeII since it's a small correction (Pritchard and Furlanetto 06)
        #
        #     integrand = 1.0/HubinvMpc(Cosmo_Parameters, zinttau)/(1+zinttau) * sigmatot * n_baryon(Cosmo_Parameters, zinttau) * constants.Mpctocm
        #
        #     taulist[iE] = np.trapezoid(integrand, zinttau)

        indextautoolarge = np.array(taulist>=self.TAUMAX)
        taulist [indextautoolarge] = self.TAUMAX
        return taulist




    def opacity_Xray(self, User_Parameters, Cosmo_Parameters, En,z,zp):
        "Returns opacity, see optical_depth() for the hard calculation."

        XRAY_OPACITY_MODEL = Cosmo_Parameters.Flag_emulate_21cmfast
        #important, 0 = standard, 1=21cmfast-like (step at tau=1)


        if(XRAY_OPACITY_MODEL==0): #0 is standard/regular.
            return np.exp(-self.optical_depth(User_Parameters, Cosmo_Parameters,En,z,zp))
        elif (XRAY_OPACITY_MODEL==1): #1 is 21cmFAST-like (step-wise exp(-tau), either 1 or 0)
            return np.heaviside(1.0 - self.optical_depth(User_Parameters, Cosmo_Parameters,En,z,zp), 0.5)
        else:
            print('ERROR, choose a correct XRAY_OPACITY_MODEL')


    def lambda_Xray_com(self, Cosmo_Parameters, En,z):
        "Returns the mean free path in cMpc of an Xray of energy En/eV near z. Unused but good cross check"

        sigmatot = self.atomfractions[0] * sigma_HI(En)
        sigmatot += self.atomfractions[1] * sigma_HeI(En)
        return (1.0/(sigmatot * n_H(Cosmo_Parameters,z))/constants.Mpctocm*(1+z) )
#        return (1.0/(sigmatot * n_baryon(Cosmo_Parameters,z))/constants.Mpctocm*(1+z) )




def sigma_HI(Energyin):
    "cross section for Xray absorption for neutral HI, from astro-ph/9601009 and takes Energy in eV and returns cross sec in cm^2"
    E0 = 4.298e-1
    sigma0 =  5.475e4
    ya = 3.288e1
    P =  2.963
    yw =  0.0
    y0 =  0.0
    y1 = 0.0

    Energy = Energyin

    warning_lowE_HIXray = np.heaviside(13.6 - Energy, 0.5)
    if(np.sum(warning_lowE_HIXray) > 0):
        print('ERROR! Some energies for Xrays below HI threshold in sigma_HI. Too low!')


    x = Energy/E0 - y0
    y = np.sqrt(x**2 + y1**2)
    Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0+np.sqrt(y/ya))**(-P)

    return sigma0 * constants.sigma0norm * Fy



def sigma_HeI(Energyin):
    "same as sigma_HI but for HeI, parameters are:"
    E0 = 13.61
    sigma0 = 9.492e2
    ya = 1.469
    P =  3.188
    yw =  2.039
    y0 =  4.434e-1
    y1 = 2.136

    Energy = Energyin
    warning_lowE_HeIXray = np.heaviside(25. - Energy, 0.5)
    if(np.sum(warning_lowE_HeIXray) > 0):
        print('ERROR! Some energies for Xrays below HeI threshold in sigma_HeI. Too low!')


    x = Energy/E0 - y0
    y = np.sqrt(x**2 + y1**2)
    Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0+np.sqrt(y/ya))**(-P)

    return sigma0 * constants.sigma0norm * Fy
