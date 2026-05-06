"""

Test the inputs for Zeus21, cosmo (including CLASS) and astro

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

import pytest
import zeus21
import numpy as np
from scipy.interpolate import interp1d

def test_inputs():

    UserParams = zeus21.User_Parameters()

    paramscosmo = [0.022, 0.12, 0.07,2.1e-9, 0.96,0.05, 100., 10.]
    # omegab, omegac, h_fid, As, ns, tau_fid, kmax_CLASS, zmax_CLASS

    CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, omegab=paramscosmo[0], omegac=paramscosmo[1], h_fid=paramscosmo[2], As=paramscosmo[3], ns=paramscosmo[4], tau_fid=paramscosmo[5], kmax_CLASS=paramscosmo[6], zmax_CLASS=paramscosmo[7])

    #make sure all the input parameters are the same as we use throughout
    assert(CosmoParams.omegab == paramscosmo[0])
    assert(CosmoParams.omegac == paramscosmo[1])
    assert(CosmoParams.h_fid == paramscosmo[2])
    assert(CosmoParams.As == paramscosmo[3])
    assert(CosmoParams.ns == paramscosmo[4])
    assert(CosmoParams.tau_fid == paramscosmo[5])
    assert(CosmoParams.kmax_CLASS == paramscosmo[6])
    assert(CosmoParams.zmax_CLASS == paramscosmo[7])
    assert(CosmoParams.zmax_CLASS >= CosmoParams.zmin_CLASS >= 0.0)

    #make sure the Omegas add to 1
    OmegaToT = CosmoParams.OmegaM + CosmoParams.OmegaR + CosmoParams.OmegaL
    assert(OmegaToT == pytest.approx(1.0))

    #and the fH and fHe fractions
    assert(CosmoParams.f_He + CosmoParams.f_H == pytest.approx(1.0))

    #make sure the Rsmoo chosen are reasonable.
    assert(CosmoParams._Rtabsmoo[0] <= 3.0) # the smallest one is small enough
    assert(CosmoParams._Rtabsmoo[-1] >= 100.) # the largest one is large enough
    assert(CosmoParams.indexminNL <= CosmoParams.indexmaxNL)


    #Test the cosmo interpolators
    _indextest=1

    chitest = CosmoParams._chitab[_indextest]
    zlistchitest = CosmoParams.zfofRint(chitest)
    assert(zlistchitest == pytest.approx(CosmoParams._ztabinchi[_indextest]) )


    _thermo = CosmoParams.ClassCosmo.get_thermodynamics()
    ztestint_thermo = _thermo['z'][_indextest]
    Ttestint_thermo = CosmoParams.Tadiabaticint(ztestint_thermo)
    assert(Ttestint_thermo == pytest.approx(_thermo['Tb [K]'][_indextest], 0.01) )
    xetestint_thermo = CosmoParams.xetanhint(ztestint_thermo)
    assert(xetestint_thermo == pytest.approx(_thermo['x_e'][_indextest], 0.01) )

    #for growth we'll check that its 0 today
    assert(CosmoParams.growthint(0) == pytest.approx(1) )



    #NOW ASTRO INPUTS
    AstroParams = zeus21.Astro_Parameters(CosmoParams=CosmoParams)

    #also run the 21cmfast-like model
    CosmoParams_21cmfast = zeus21.Cosmo_Parameters(UserParams=UserParams, Flag_emulate_21cmfast=True)
    AstroParams_21cmfast = zeus21.Astro_Parameters(CosmoParams=CosmoParams_21cmfast)


    assert( 0.0 <= AstroParams_21cmfast.tstar <= 10.0)
    assert( 0.0 <= AstroParams_21cmfast.fstarmax <= 10.0)
    assert(AstroParams_21cmfast.fstar10 == pytest.approx(AstroParams_21cmfast.epsstar) )
    assert( 0.0 <= AstroParams.clumping <= 10.0 )
    assert( 0.0 <= AstroParams_21cmfast.clumping <= 10.0 )



    #test Pop II Xray SED
    Energylisttest = np.logspace(2,np.log10(AstroParams.Emax_xray_norm),100)
    SEDXtab_test = AstroParams.SED_XRAY(Energylisttest, 2) #same in both models
    normalization_XraySED = np.trapezoid(Energylisttest * SEDXtab_test,Energylisttest)
    assert( normalization_XraySED == pytest.approx(1.0, 0.05) ) #5% is enough here
    
    #test Pop III Xray SED
    SEDXtab_test = AstroParams.SED_XRAY(Energylisttest, 3) #same in both models
    normalization_XraySED = np.trapezoid(Energylisttest * SEDXtab_test,Energylisttest)
    assert( normalization_XraySED == pytest.approx(1.0, 0.05) ) #5% is enough here


    #test Pop II LyA SED
    nulisttest = np.linspace(zeus21.constants.freqLyA, zeus21.constants.freqLyCont, 100)
    SEDLtab_test = AstroParams.SED_LyA(nulisttest, 2) #same in both models
    normalization_LyASED = np.trapezoid(SEDLtab_test,nulisttest)
    assert( normalization_LyASED == pytest.approx(1.0, 0.05) ) #5% is enough here
    
    #test Pop III LyA SED
    nulisttest = np.linspace(zeus21.constants.freqLyA, zeus21.constants.freqLyCont, 100)
    SEDLtab_test = AstroParams.SED_LyA(nulisttest, 3) #same in both models
    normalization_LyASED = np.trapezoid(SEDLtab_test,nulisttest)
    assert( normalization_LyASED == pytest.approx(1.0, 0.05) ) #5% is enough here
