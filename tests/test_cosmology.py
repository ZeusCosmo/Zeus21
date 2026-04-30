"""

Cosmology tests for Zeus21

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

import pytest
import zeus21
import numpy as np

from zeus21.cosmology import *


def test_cosmo():

    UserParams = zeus21.User_Parameters()

    CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=10., zmax_CLASS=10., USE_RELATIVE_VELOCITIES=True) #to speed up

    #velocity component testing
    assert(10.0 <= CosmoParams.sigma_vcb <= 100.0)
    assert(10.0 <= CosmoParams.vcb_avg <= 100.0)
    
    #useful functions:

    #today
    H0test = Hub(CosmoParams,0.)
    assert(CosmoParams.h_fid * 100 == pytest.approx(H0test, 0.01))


    #test there is DE and M+DE~1 at z=0
    zzDE = 1e-1
    HzDEtest = Hub(CosmoParams,zzDE)
    assert(H0test * (1 + 1.5 * CosmoParams.OmegaM * zzDE) == pytest.approx(HzDEtest, 0.01))

    #matter domination
    zzMatter = 30.
    HzMattertest = Hub(CosmoParams,zzMatter)
    assert(np.sqrt(CosmoParams.OmegaM) * H0test * (1+zzMatter)**1.5 == pytest.approx(HzMattertest, 0.01))

    #rad domination
    zzRad = 1e9
    HzRadtest = Hub(CosmoParams,zzRad)
    assert(np.sqrt(CosmoParams.OmegaR) * H0test * (1+zzRad)**2.0 == pytest.approx(HzRadtest, 0.01))


    assert(0. <= n_H(CosmoParams,0.0) <= 1e-6) #make sure it's reasonable ~1e-7

    assert(2.5<= Tcmb(CosmoParams.ClassCosmo,0.0) <= 3.0) #make sure it's reasonable 2.725 K

    assert(Tcmb(CosmoParams.ClassCosmo,500.) == pytest.approx(Tadiabatic(CosmoParams,500.), 0.1)) #where they are coupled

    assert(0. <= xefid(CosmoParams,0) <= 1.0)
    assert(0. <= xefid(CosmoParams,10) <= 1.0)
    assert(0. <= xefid(CosmoParams,100) <= 1.0)

    Rintest=1.0 #cMpc
    assert(RadofMh (CosmoParams,MhofRad(CosmoParams,Rintest)) == pytest.approx(Rintest) ) #make sure they're the inverse of each other


    assert(growth(CosmoParams,0.) == pytest.approx(1.0,0.01) )
    assert(dgrowth_dz(CosmoParams,10.) <= 0. )




    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams)
    MM = HMFintclass.fitMztab[0][1]
    zz = HMFintclass.fitMztab[1][1]
    assert(HMFintclass.HMF_int(np.exp(MM),zz) == pytest.approx(HMFintclass.HMFtab[1,1],0.01))
    
    
