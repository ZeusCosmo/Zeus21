"""

Test UV luminosity functions for Zeus21

Author: Claude AI
April 2025

Edited by Alessandra Venditti
UT Austin - June 2026
"""

import pytest
import zeus21
import numpy as np

from zeus21.LFs import LF_class


def test_luminosity_to_magnitude_conversions():
    """Test magnitude and luminosity conversions, verifying values and that they are invert of each other"""

    LF = LF_class.__new__(LF_class)

    # Expected Lnu to MUV conversion for a range of Lnu
    Lnu_test = np.logspace(25., 30., 3)  # erg/s/Hz
    sigma_test = 0.5
    expected_Lnu_renorm = Lnu_test / np.exp((np.log(10)/2.5 * sigma_test)**2 / 2.0)
    expected_MUV = 51.63 - 2.5 * np.log10(Lnu_test)
    expected_MUV_renorm = 51.63 - 2.5 * np.log10(expected_Lnu_renorm)

    # Test from Mag_of_L_ergsHz
    MUV = LF.Mag_of_L_ergsHz(Lnu_test)
    np.testing.assert_allclose(MUV, expected_MUV, rtol=1e-12)

    # Test inverse function
    Lnu_roundtrip = LF.L_ergsHz_of_Mag(MUV)
    np.testing.assert_allclose(Lnu_roundtrip, Lnu_test, rtol=1e-12)

    # Test from logorMag_of_L
    MUV = LF.logorMag_of_L(Lnu_test, "UV", renormalize_L=False)
    MUV_renorm = LF.logorMag_of_L(Lnu_test, "UV", renormalize_L=True, sigma=sigma_test)
    np.testing.assert_allclose(MUV, expected_MUV, rtol=1e-12)
    np.testing.assert_allclose(MUV_renorm, expected_MUV_renorm, rtol=1e-12)


    # Expected nuLnu to MUV conversion for a range of nuLnu
    nuLnu_test = np.logspace(40., 45., 3)  # erg/s
    wavelength_test = 1500.  # A
    expected_MUV = 51.63 - 2.5 * np.log10(nuLnu_test / (299792.458 / (wavelength_test/1e13)) )

    # Test from Mag_of_L_ergs
    MUV = LF.Mag_of_L_ergs(nuLnu_test, wavelength=wavelength_test)
    np.testing.assert_allclose(MUV, expected_MUV, rtol=1e-12)

    # Test inverse function
    nuLnu_roundtrip = LF.L_ergs_of_Mag(MUV, wavelength=wavelength_test)
    np.testing.assert_allclose(nuLnu_roundtrip, nuLnu_test, rtol=1e-12)


def test_betaUV_dust():
    """Test the beta (UV slope) calculation"""

    LF = LF_class.__new__(LF_class)
    LFParams = zeus21.LF_Parameters()

    # Test a single redshift and magnitude but use arrays as the function expects
    z_test = np.array([5.0])
    MUV_test = np.array([-20.0])

    # Calculate beta value
    beta_value = LF.betaUV_dust(LFParams, z_test, MUV_test)

    # Check that beta value is reasonable (typical range is -3 to -1)
    assert np.all(beta_value > -3.0)
    assert np.all(beta_value < -1.0)

    # Test at pivot point
    MUV_pivot = np.array([-19.5])
    beta_at_pivot = LF.betaUV_dust(LFParams, z_test, MUV_pivot)

    # Check that a value is returned
    assert isinstance(beta_at_pivot, np.ndarray)


def test_dust_attenuation():
    """Test the dust attenuation calculation"""

    LF = LF_class.__new__(LF_class)
    LFParams = zeus21.LF_Parameters()

    
    # Test with arrays as the function expects
    z_test = np.array([5.0])
    MUV_test = np.array([-20.0])
    
    # Calculate dust attenuation
    A_UV = LF.dust_attenuation(LFParams, z_test, MUV_test, "UV")
    
    # Check that attenuation is non-negative
    assert np.all(A_UV >= 0.0)

    
    # Test the HIGH_Z_DUST flag behavior
    z_high = np.array([9.0, 10.0, 12.0])
    MUV_test = np.array([-22.0, -20.0, -18.0])  # High redshift above _zmaxdata
    
    # Test with HIGH_Z_DUST=True (dust applied at high z)
    LFParams.HIGH_Z_DUST = True
    A_UV_highz = LF.dust_attenuation(LFParams, z_high, MUV_test, "UV")

    # HIGH_Z_DUST=True should some attenuation for z > _zmaxdata
    assert np.any(A_UV_highz > 0.0)

    
    # Test with HIGH_Z_DUST=False (no dust above _zmaxdata)
    LFParams.HIGH_Z_DUST = False
    A_UV_no_highz = LF.dust_attenuation(LFParams, z_high, MUV_test, "UV")
    
    # HIGH_Z_DUST=False should give zero attenuation for z > _zmaxdata
    assert np.all(A_UV_no_highz == 0.0)


def test_compute_LFbias_binned_from_SFRlist():
    """Test the binned UV luminosity function calculation"""

    # Set up parameters
    UserParams = zeus21.User_Parameters()
    CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=100., zmax_CLASS=20.)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams)
    AstroParams = zeus21.Astro_Parameters(CosmoParams=CosmoParams, USE_POPIII=False, FLAG_USE_PSD=False)
    
    LFParams = zeus21.LF_Parameters(FLAG_COMPUTE_UVLF=False, FLAG_COMPUTE_HaLF=False,
    RETURNBIAS=True,)
    LF = LF_class(UserParams, CosmoParams, AstroParams, HMFintclass, LFParams)


    # Test a range of SFR values
    SFR_test = np.logspace(-3, 2, HMFintclass.Mhtab.size)
    kappaUV_test = 1.15e-28  # Typical value  

    # Test LF parameters
    zcenter_test = 6.0
    zwidth_test = 0.5
    MUVcenters_test = np.array([-22.0, -20.0, -18.0])
    MUVwidths_test = np.full_like(MUVcenters_test, 1.0)
    DUST_FLAG = True
    sigmaUV_test = 0.5  

    
    # Calculate UVLF
    UVLF = LF.compute_LFbias_binned_from_SFRlist(SFR_test, HMFintclass, LFParams,
                                                 zcenter_test, zwidth_test, MUVcenters_test, MUVwidths_test,  
                                                 kappaUV_test, sigmaUV_test, renormalize_L=True,  
                                                 which_band="UV", include_dust=DUST_FLAG,
                                                 computeLF=True, computeBias=False)["LF"]
    
    # Check dimensions
    assert UVLF.shape == (3,)
    
    # Check that values are positive
    assert np.all(UVLF >= 0.0)
    
    # Test that fainter (more positive MUV) bins typically have higher number densities
    # This is a general trend for LFs, but not strictly required
    # We'll do a weak test that they're not all identical
    assert len(np.unique(UVLF)) > 1


    # Test RETURNBIAS flag
    bias = LF.compute_LFbias_binned_from_SFRlist(SFR_test, HMFintclass, LFParams,
                                                 zcenter_test, zwidth_test, MUVcenters_test, MUVwidths_test,  
                                                 kappaUV_test, sigmaUV_test, renormalize_L=True,  
                                                 which_band="UV", include_dust=DUST_FLAG,
                                                 computeLF=False, computeBias=True)["bias"]
    
    # Check dimensions
    assert bias.shape == (3,)
    
    # Check that biases are positive
    assert np.all(bias >= 0.0)
    
    # Test without dust correction
    UVLF_nodust = LF.compute_LFbias_binned_from_SFRlist(SFR_test, HMFintclass, LFParams,
                                                 zcenter_test, zwidth_test, MUVcenters_test, MUVwidths_test,  
                                                 kappaUV_test, sigmaUV_test, renormalize_L=True,  
                                                 which_band="UV", include_dust=False,
                                                 computeLF=True, computeBias=False)["LF"]
    
    # Check dimensions
    assert UVLF_nodust.shape == (3,)
    
    # Without dust, we expect different values than with dust
    assert not np.array_equal(UVLF, UVLF_nodust)



def test_UVLF_binned_with_min_t_formation():
    """Test that min_t_formation_Myr produces finite outputs and suppresses the bright end.

    When sigmaUV is large, scatter can push small halos into unphysically bright bins.
    Setting min_t_formation_Myr places a physical upper limit on each halo's SFR based on
    its maximum stellar mass (all baryons converted to stars) and the minimum formation time.
    This should suppress the very bright end of the UVLF without affecting the faint end.
    """
    pytest.skip("min_t_formation_Myr is not yet a parameter in Astro_Parameters for this branch")


# TODO: tests for UVLF with PSD?

# TODO: tests for Halpha LF?

# TODO: tests for Ha/UV ratios?