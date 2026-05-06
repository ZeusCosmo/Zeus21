"""

Xray tests for Zeus21

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

"""

import pytest
import zeus21
import numpy as np

from zeus21.T21coefficients import Xrays_class

UserParams = zeus21.User_Parameters(zmin_T21=20.)

CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=100.) #to speed up
AstroParams = zeus21.Astro_Parameters(CosmoParams=CosmoParams)
HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams)

Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, AstroParams, HMFintclass)
Energylist = AstroParams.Energylist

def test_xrays():

    #test cross sections are positive
    assert( (np.zeros_like(Energylist) <= Coeffs.Xrays.sigma_HI(Energylist)).all())
    assert( (np.zeros_like(Energylist) <= Coeffs.Xrays.sigma_HeI(Energylist)).all())

    #test that X-ray heating is non-negative (allowing for small numerical noise)
    assert( (Coeffs.Tk_xray >= 0.0).all())

    #test that ionization from X-rays is non-negative
    assert( (Coeffs.Gammaion_II >= 0.0).all())
    assert( (Coeffs.Gammaion_III >= 0.0).all())

    #test that xe is between adiabatic value and 1
    assert( (Coeffs.xe_avg >= Coeffs.xe_avg_ad).all())
    assert( (Coeffs.xe_avg <= 1.0).all())
