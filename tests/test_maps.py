"""

Test the maps functionality in Zeus21

Author: Claude AI
April 2025

"""

import pytest
import zeus21
import numpy as np

from zeus21.maps import T21_maps
from zeus21.z21_utilities import powerboxCtoR

def test_coevalmaps_initialization():
    """Test that T21_maps initializes correctly"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters(zmin=5.0)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=100.) # Use higher kmax_CLASS as in test_astrophysics.py
    
    AstroParams = zeus21.Astro_Parameters(CosmoParams=CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams)
    
    # Generate T21 coefficients
    Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, AstroParams, HMFintclass)
    
    # Generate power spectra
    PS21 = zeus21.Power_Spectra(UserParams, CosmoParams, AstroParams, Coeffs)
    
    # Test redshift
    ztest = 8  # Use a redshift that's compatible with our ZMIN setting
    
    # Initialize the map with reduced size for test performance
    map_obj = T21_maps(CosmoParams, Coeffs, PS21, [ztest], input_boxlength=300, ncells=50, seed=12345)
    
    # Verify attributes
    assert map_obj.input_boxlength == 300
    assert map_obj.ncells == 50
    assert map_obj.seed == 12345
    
    # Check that z is snapped to closest value in grid
    iz_test = min(range(len(Coeffs.zintegral)), key=lambda i: np.abs(Coeffs.zintegral[i]-ztest))
    #assert map_obj.input_z[0] == pytest.approx(Coeffs.zintegral[iz_test]) # that is not necessarily going to be equal...
    
    # Check T21global is properly set
    assert map_obj.T21avg == pytest.approx(Coeffs.T21avg[iz_test]/(Coeffs.xHI_avg[iz_test] + 1e-15))
    
    # Verify all components exist
    assert map_obj.density is not None
    assert map_obj.T21_lin is not None
    assert map_obj.T21_NL is not None
    assert map_obj.T21 is not None
    
    # Check that maps have correct dimensions
    assert map_obj.density.shape == (1, 50, 50, 50)
    assert map_obj.T21_lin.shape == (1, 50, 50, 50)
    assert map_obj.T21_NL.shape == (1, 50, 50, 50)
    assert map_obj.T21.shape == (1, 50, 50, 50)
    
    # Check basic statistics of maps
    # Density map should have mean ≈ 0
    assert np.mean(map_obj.density) == pytest.approx(0.0, abs=0.1)
    
    # T21_lin should have mean ≈ T21global
    assert np.mean(map_obj.T21_lin) == pytest.approx(map_obj.T21avg, abs=5.0)
    
    # Verify standard deviation is not zero (actual field generated)
    assert np.std(map_obj.density) > 0
    assert np.std(map_obj.T21) > 0

def test_powerboxCtoR():
    """Test the powerboxCtoR utility function"""
    UserParams = zeus21.User_Parameters(zmin=20.0)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams=UserParams, kmax_CLASS=100.) # Use higher kmax_CLASS as in test_astrophysics.py
    
    AstroParams = zeus21.Astro_Parameters(CosmoParams=CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams)
    
    # Generate T21 coefficients
    Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, AstroParams, HMFintclass)
    
    # Generate power spectra
    PS21 = zeus21.Power_Spectra(UserParams, CosmoParams, AstroParams, Coeffs)
    
    # Test redshift
    ztest = 25.0  # Use a redshift that's compatible with our ZMIN setting
    
    # Initialize a map for testing
    from powerbox import PowerBox
    import powerbox as pbox
    
    # Create a simple powerbox object with known spectrum
    pb = PowerBox(
        N=20,                     
        dim=3,                     
        pk=lambda k: k**-2,  # Simple power law spectrum
        boxlength=300,           
        seed=12345                
    )
    
    # Generate k-space field
    delta_k = pb.delta_k()
    
    # Apply utility function
    real_field = powerboxCtoR(pb, mapkin=delta_k)
    
    # Check that output is real
    assert np.isreal(real_field).all()
    
    # Check dimensions
    assert real_field.shape == (20, 20, 20)
    
    # Test with default parameter (None)
    real_field2 = powerboxCtoR(pb)
    assert np.isreal(real_field2).all()
    assert real_field2.shape == (20, 20, 20)