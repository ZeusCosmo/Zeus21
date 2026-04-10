"""

Astrophysics tests for Zeus21, SFRD, global, and 21-cm PS

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

import pytest
import zeus21
import numpy as np

from zeus21.sfrd import *
from zeus21.correlations import *
from zeus21.reionization import *

UserParams = zeus21.User_Parameters()

CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS = 100.) #to speed up a little
ClassyCosmo = zeus21.runclass(CosmoParams_input)
CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams,ClassyCosmo)



AstroParams = zeus21.Astro_Parameters(UserParams,CosmoParams)
AstroParams_popIII = zeus21.Astro_Parameters(UserParams,CosmoParams,USE_POPIII=True)
ZMIN = 20.0 #down to which z we compute the evolution
CorrFClass = zeus21.Correlations(UserParams,CosmoParams, ClassyCosmo)
Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
Coeffs_popIII = zeus21.get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams_popIII, HMFintclass, zmin=ZMIN)

#also for exponential accretion:
AstroParams_expacc = zeus21.Astro_Parameters(UserParams,CosmoParams, accretion_model=0)

#and for the 21cmfast mode:
CosmoParams_input_21cmfast = zeus21.Cosmo_Parameters_Input(Flag_emulate_21cmfast=True)
ClassyCosmo_21cmfast = zeus21.runclass(CosmoParams_input_21cmfast)
CosmoParams_21cmfast = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input_21cmfast, ClassyCosmo_21cmfast)
AstroParams_21cmfast = zeus21.Astro_Parameters(UserParams,CosmoParams_21cmfast, astromodel = 1)

#and for bubbles:
BMF_class = zeus21.BMF(Coeffs, HMFintclass, CosmoParams, AstroParams, ClassyCosmo)


ztest = 20.
iztest = min(range(len(Coeffs.zintegral)), key=lambda i: np.abs(Coeffs.zintegral[i]-ztest))

#test the background and globals
def test_background():

    #test SFR first
    sSFR = SFR_II(AstroParams, CosmoParams, HMFintclass, HMFintclass.Mhtab, ztest, ztest)/HMFintclass.Mhtab
    assert( (0 <= sSFR).all()) #positive
    assert( (sSFR/zeus21.cosmology.Hubinvyr(CosmoParams,ztest) <= 1).all()) #make sure sSFR/H < 1 (not all mass forms stars in a Hubble time)
    
    sSFR3 = SFR_III(AstroParams, CosmoParams, HMFintclass, HMFintclass.Mhtab, Coeffs_popIII.J21LW_interp_conv_avg, ztest, ztest, ClassyCosmo.pars['v_avg'])/HMFintclass.Mhtab
    assert( (0 <= sSFR3).all()) #positive
    assert( (sSFR3/zeus21.cosmology.Hubinvyr(CosmoParams,ztest) <= 1).all()) #make sure sSFR3/H < 1 (not all mass forms stars in a Hubble time)

    
    #repeat for Exp Accretion case
    sSFR_exp = SFR_II(AstroParams_expacc, CosmoParams, HMFintclass, HMFintclass.Mhtab, ztest, ztest)/HMFintclass.Mhtab
    assert( (0 <= sSFR_exp).all())
    assert( (sSFR_exp/zeus21.cosmology.Hubinvyr(CosmoParams,ztest) <= 1).all())
    
    sSFR_exp3 = SFR_III(AstroParams_expacc, CosmoParams, HMFintclass, HMFintclass.Mhtab, Coeffs_popIII.J21LW_interp_conv_avg, ztest, ztest, ClassyCosmo.pars['v_avg'])/HMFintclass.Mhtab
    assert( (0 <= sSFR_exp3).all())
    assert( (sSFR_exp3/zeus21.cosmology.Hubinvyr(CosmoParams,ztest) <= 1).all())

    
    #repeat for 21cmfast emulation case
    sSFR_21cmfast = SFR_II(AstroParams_21cmfast, CosmoParams_21cmfast, HMFintclass, HMFintclass.Mhtab, ztest, ztest)/HMFintclass.Mhtab
    assert( (0 <= sSFR_21cmfast).all())
    assert( (sSFR_21cmfast/zeus21.cosmology.Hubinvyr(CosmoParams_21cmfast,ztest) <= 1).all())
    
    sSFR_21cmfast3 = SFR_III(AstroParams_expacc, CosmoParams_21cmfast, HMFintclass, HMFintclass.Mhtab, Coeffs_popIII.J21LW_interp_conv_avg, ztest, ztest, ClassyCosmo.pars['v_avg'])/HMFintclass.Mhtab
    assert( (0 <= sSFR_21cmfast3).all())
    assert( (sSFR_21cmfast3/zeus21.cosmology.Hubinvyr(CosmoParams_21cmfast,ztest) <= 1).all())


    #test fesc
    assert( (0 <= fesc_II(AstroParams, HMFintclass.Mhtab)).all())
    assert( (fesc_II(AstroParams, HMFintclass.Mhtab <= 1)).all())

    assert( (0 <= fesc_III(AstroParams, HMFintclass.Mhtab)).all())
    assert( (fesc_III(AstroParams, HMFintclass.Mhtab <= 1)).all())


    #and sfrd calculation
    assert( (Coeffs.ztabRsmoo[iztest] >= Coeffs.zintegral[iztest]).all())

    assert( (Coeffs.sigmaofRtab >= 0.0).all()) #all Ts positive



    assert( (Coeffs.xe_avg >= Coeffs.xe_avg_ad).all()) #xrays only add ionization, not subtract

    assert( (Coeffs.Tk_ad >= 0.0).all()) #all Ts positive
    assert( (Coeffs.Tk_xray >= 0.0).all()) #all Ts positive
    assert( (Coeffs.Tk_avg >= Coeffs.Tk_ad).all()) #and it's a sum
    assert( (Coeffs.T_CMB >= 0.0).all()) #all Ts positive
    assert( (Coeffs._invTs_avg >= 0.0).all()) #all Ts positive
    assert( (Coeffs.invTcol_avg >= 0.0).all()) #all Ts positive
    assert( (Coeffs.Jalpha_avg >= 0.0).all())
    assert( (Coeffs.xa_avg >= 0.0).all())
    assert( (Coeffs._fheat >= 0.0).all())

    #make sure Ts is between Tcmb and Tc
    assert( (Coeffs.invTcol_avg >= Coeffs._invTs_avg).all()) #all Ts positive
    assert( (Coeffs._invTs_avg >= 1/Coeffs.T_CMB).all()) #all Ts positive

    assert( (Coeffs.SFRDbar2D_II >= 0.0).all())
    assert( (Coeffs.SFRD_II_avg >= 0.0).all())
    
    assert( (Coeffs.SFRDbar2D_III >= 0.0).all())
    assert( (Coeffs.SFRD_III_avg >= 0.0).all())
    
    assert( (Coeffs.niondot_avg_II >= 0.0).all())
    assert( (Coeffs.niondot_avg_III >= 0.0).all())
    
    assert( (Coeffs.xHI_avg >= 0.0).all())
    assert( (Coeffs.xHI_avg <= 1.0).all())

    assert( (Coeffs.T21avg <= zeus21.cosmology.T021(CosmoParams,Coeffs.zintegral)).all()) #capped from above
    assert( (Coeffs.T21avg >= (- 300.0) ).all()) #capped from below (IN LCDM!)



    assert( (Coeffs.gamma_index2D >= 0.0).all()) #effective biases have to be larger than 0 in reasonable models, since galaxies live in haloes that are more clustered than average matter (in other words, SFRD grows monotonically with density)

    assert( ((BMF_class.ion_frac >= 0.0)*(BMF_class.ion_frac <= 1.0) ).all())
    assert( (BMF_class.BMF >= 0.0).all())




#and test the PS too
PS21 = zeus21.Power_Spectra(UserParams, CosmoParams, AstroParams, ClassyCosmo, CorrFClass, Coeffs)


def test_pspec():

    assert((PS21._rs_input_mcfit == CorrFClass.rlist_CF).all())
    assert((PS21.klist_PS == CorrFClass._klistCF).all())
    assert((PS21.kwindow == PS21._kwindowX).all())

    ztest = 20.
    iztest = min(range(len(Coeffs.zintegral)), key=lambda i: np.abs(Coeffs.zintegral[i]-ztest))

    assert((PS21.windowalpha_II[iztest,0] >= PS21.windowalpha_II[iztest,-1]).all()) #at fixed z it should go down with k
    assert((PS21.windowxray_II[iztest,0] >= PS21.windowxray_II[iztest,-1]).all())
    
    assert((PS21.windowalpha_III[iztest,0] >= PS21.windowalpha_III[iztest,-1]).all()) #at fixed z it should go down with k
    assert((PS21.windowxray_III[iztest,0] >= PS21.windowxray_III[iztest,-1]).all())

    #make sure all correlations are sensible
    assert( (PS21.Deltasq_dxa[iztest]**2 <= 1.01* PS21.Deltasq_d[iztest] * PS21.Deltasq_xa[iztest]).all())
    assert( (PS21.Deltasq_dTx[iztest]**2 <= 1.01* PS21.Deltasq_d[iztest] * PS21.Deltasq_Tx[iztest]).all())
    assert( (PS21.Deltasq_xaTx[iztest]**2 <= 1.01* PS21.Deltasq_Tx[iztest] * PS21.Deltasq_xa[iztest]).all())

    assert( (PS21.Deltasq_dxa_lin[iztest]**2 <= 1.01* PS21.Deltasq_d_lin[iztest] * PS21.Deltasq_xa_lin[iztest]).all())
    assert( (PS21.Deltasq_dTx_lin[iztest]**2 <= 1.01* PS21.Deltasq_d_lin[iztest] * PS21.Deltasq_Tx_lin[iztest]).all())
    assert( (PS21.Deltasq_xaTx_lin[iztest]**2 <= 1.01* PS21.Deltasq_Tx_lin[iztest] * PS21.Deltasq_xa_lin[iztest]).all())

    #also make sure all Pk(k) < avg^2 for all quantities at some k~0.1
    ktest = 0.1
    iktest = min(range(len(PS21.klist_PS)), key=lambda i: np.abs(PS21.klist_PS[i]-ktest))

    assert( (PS21.Deltasq_xa[:,iktest] <= 1.01*Coeffs.xa_avg**2 ).all())
    assert( (PS21.Deltasq_Tx[:,iktest] <= 1.01*Coeffs.Tk_xray**2).all())
    assert( (PS21.Deltasq_T21[:,iktest] <= 1.01*(Coeffs.T21avg)**2).all()) #can fail near T21~0. If so add an offset outside the **2.
