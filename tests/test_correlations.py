"""

Correlations tests for Zeus21

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

import pytest
import zeus21
import numpy as np

from zeus21 import z21_utilities
import warnings
warnings.filterwarnings("ignore", category=UserWarning) #to silence annyoing warning in mcfit

UserParams = zeus21.User_Parameters()

CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=100., zmax_CLASS=10.) #to speed up


def test_corrfuncs():

    # Correlation arrays are now stored on CosmoParams (computed in run_correlations())
    assert len(CosmoParams._klistCF) > 0
    assert len(CosmoParams._PklinCF) > 0
    assert len(CosmoParams.rlist_CF) > 0
    assert np.all(np.isfinite(CosmoParams._PklinCF))
    assert CosmoParams.xi_RR_CF.shape == (CosmoParams.NRs, CosmoParams.NRs, len(CosmoParams.rlist_CF))
    assert np.all(np.isfinite(CosmoParams.xi_RR_CF))
    assert np.all(np.isfinite(CosmoParams.xiEta_RR_CF))

    assert(CosmoParams.xi_RR_CF[0][0][1] >= CosmoParams.xi_RR_CF[1][1][1]) #make sure smoothing goes the right direction
    assert(CosmoParams.xiEta_RR_CF[0][0][1] >= CosmoParams.xiEta_RR_CF[1][1][1]) #make sure smoothing goes the right direction

    #windows (now in z21_utilities)
    ktestwin = 1e-4
    Rtestwin = 1.0
    assert(z21_utilities._WinG(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))
    assert(z21_utilities._WinTH(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))
    assert(z21_utilities._WinTH1D(ktestwin,Rtestwin) == pytest.approx(1.0, 0.01))

    ktestwin = 3.
    assert(z21_utilities._WinG(ktestwin,Rtestwin) < 1.0)
    assert(z21_utilities._WinTH(ktestwin,Rtestwin) < 1.0)
    assert(z21_utilities._WinTH1D(ktestwin,Rtestwin) < 1.0)
