"""

Test UV luminosity functions for Zeus21

Author: Claude AI
April 2025

"""

import pytest
import zeus21
import numpy as np

from zeus21.UVLFs import UVLF_binned, MUV_of_SFR, AUV, beta

def test_MUV_of_SFR():
    """Test the conversion from SFR to UV magnitudes"""
    # Test a range of SFR values
    SFR_test = np.logspace(-3, 2, 10)  # M_sun/yr
    kappaUV_test = 1.15e-28  # Typical value
    
    # Calculate MUV
    MUV_result = MUV_of_SFR(SFR_test, kappaUV_test)
    
    # Check that increasing SFR leads to brighter (more negative) MUV
    assert np.all(np.diff(MUV_result) < 0)
    
    # Check specific value based on the formula M_UV = 51.63 - 2.5*log10(SFR/kappaUV)
    # For SFR = 1 M_sun/yr with kappaUV = 1.15e-28
    expected_MUV = 51.63 - 2.5 * np.log10(1.0/1.15e-28)
    assert MUV_of_SFR(np.array([1.0]), kappaUV_test)[0] == pytest.approx(expected_MUV)
    
    # Test different kappaUV values
    kappaUV_test2 = 2.0e-28
    MUV_result2 = MUV_of_SFR(SFR_test, kappaUV_test2)
    
    # Higher kappaUV should result in fainter magnitudes (more positive)
    assert np.all(MUV_result2 > MUV_result)

def test_beta_function():
    """Test the beta (UV slope) calculation"""
    # Test a single redshift and magnitude but use arrays as the function expects
    z_test = np.array([5.0])
    MUV_test = np.array([-20.0])
    
    # Calculate beta value
    beta_value = beta(z_test, MUV_test)
    
    # Check that beta value is reasonable (typical range is -3 to -1)
    assert beta_value > -3.0
    assert beta_value < -1.0
    
    # Test at pivot point
    MUV_pivot = np.array([-19.5])  # The pivot point defined in the code
    beta_at_pivot = beta(z_test, MUV_pivot)
    
    # Check that a value is returned
    assert isinstance(beta_at_pivot, np.ndarray)

def test_AUV_function():
    """Test the dust attenuation calculation"""
    # Set up parameters
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input()
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    
    # Test with arrays as the function expects
    z_test = np.array([5.0])
    MUV_test = np.array([-20.0])
    
    # Calculate dust attenuation
    A_UV = AUV(AstroParams, z_test, MUV_test)
    
    # Check that attenuation is non-negative
    assert np.all(A_UV >= 0.0)
    
    # Test the HIGH_Z_DUST flag behavior
    z_high = np.array([9.0])  # High redshift above _zmaxdata
    _zmaxdata = 8.0
    
    # Test with HIGH_Z_DUST=True (dust applied at high z)
    A_UV_high = AUV(AstroParams, z_high, MUV_test, HIGH_Z_DUST=True)
    
    # Test with HIGH_Z_DUST=False (no dust above _zmaxdata)
    A_UV_no_highz = AUV(AstroParams, z_high, MUV_test, HIGH_Z_DUST=False, _zmaxdata=_zmaxdata)
    
    # HIGH_Z_DUST=False should give zero attenuation for z > _zmaxdata
    assert np.all(A_UV_no_highz == 0.0)

def test_UVLF_binned():
    """Test the binned UV luminosity function calculation"""
    # Set up parameters
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=10., zmax_CLASS=20.)
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    
    # Test data
    z_center = 6.0
    z_width = 0.5
    MUV_centers = np.array([-22.0, -20.0, -18.0])
    MUV_widths = np.full_like(MUV_centers, 1.0)
    
    # Calculate UVLF
    uvlf = UVLF_binned(AstroParams, CosmoParams, HMFintclass, z_center, z_width, 
                       MUV_centers, MUV_widths, DUST_FLAG=True, RETURNBIAS=False)
    
    # Check dimensions
    assert uvlf.shape == (3,)
    
    # Check that values are positive
    assert np.all(uvlf >= 0.0)
    
    # Test that fainter (more positive MUV) bins typically have higher number densities
    # This is a general trend for LFs, but not strictly required
    # We'll do a weak test that they're not all identical
    assert len(np.unique(uvlf)) > 1
    
    # Test RETURNBIAS flag
    bias_values = UVLF_binned(AstroParams, CosmoParams, HMFintclass, z_center, z_width, 
                             MUV_centers, MUV_widths, DUST_FLAG=True, RETURNBIAS=True)
    
    # Check dimensions
    assert bias_values.shape == (3,)
    
    # Check that biases are positive
    assert np.all(bias_values >= 0.0)
    
    # Test without dust correction
    uvlf_nodust = UVLF_binned(AstroParams, CosmoParams, HMFintclass, z_center, z_width, 
                             MUV_centers, MUV_widths, DUST_FLAG=False, RETURNBIAS=False)
    
    # Check dimensions
    assert uvlf_nodust.shape == (3,)
    
    # Without dust, we expect different values than with dust
    assert not np.array_equal(uvlf, uvlf_nodust)


def test_UVLF_binned_with_min_t_formation():
    """Test that min_t_formation_Myr produces finite outputs and suppresses the bright end.

    When sigmaUV is large, scatter can push small halos into unphysically bright bins.
    Setting min_t_formation_Myr places a physical upper limit on each halo's SFR based on
    its maximum stellar mass (all baryons converted to stars) and the minimum formation time.
    This should suppress the very bright end of the UVLF without affecting the faint end.
    """
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=10., zmax_CLASS=20.)
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)

    # Use a large sigmaUV to create unphysical scatter into the bright end
    large_sigmaUV = 2.0
    min_t_Myr = 10.0

    # AstroParams with the physicality cutoff applied
    AstroParams_cut = zeus21.Astro_Parameters(
        UserParams, CosmoParams,
        sigmaUV=large_sigmaUV,
        min_t_formation_Myr=min_t_Myr
    )

    # AstroParams without the cutoff (default None)
    AstroParams_nocut = zeus21.Astro_Parameters(
        UserParams, CosmoParams,
        sigmaUV=large_sigmaUV
    )

    z_center = 6.0
    z_width = 0.5
    # Include a very bright bin (-25) where small-halo scatter is cut off,
    # a typical bin (-20), and a faint bin (-15) that should be unaffected
    MUV_centers = np.array([-25.0, -20.0, -15.0])
    MUV_widths = np.full_like(MUV_centers, 1.0)

    uvlf_cut = UVLF_binned(
        AstroParams_cut, CosmoParams, HMFintclass,
        z_center, z_width, MUV_centers, MUV_widths,
        DUST_FLAG=False, RETURNBIAS=False
    )
    uvlf_nocut = UVLF_binned(
        AstroParams_nocut, CosmoParams, HMFintclass,
        z_center, z_width, MUV_centers, MUV_widths,
        DUST_FLAG=False, RETURNBIAS=False
    )

    # Output must be finite (no NaNs or Infs) with the cutoff applied
    assert np.all(np.isfinite(uvlf_cut)), "UVLF with min_t_formation_Myr cutoff contains NaN or Inf values"

    # All values must be non-negative
    assert np.all(uvlf_cut >= 0.0), "UVLF with min_t_formation_Myr cutoff contains negative values"

    # The cutoff should suppress the very bright end: small halos that could not
    # physically produce MUV=-25 galaxies (min_MUV~-18.7 for 1e8 Msun with t_min=10 Myr)
    # no longer contribute via scatter, so the bright-end UVLF should be lower
    assert uvlf_cut[0] < uvlf_nocut[0], (
        "min_t_formation_Myr cutoff should suppress the very bright end (MUV=-25) of the UVLF"
    )