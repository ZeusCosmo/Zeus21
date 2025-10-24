"""

Test the maps functionality in Zeus21

Author: Claude AI
April 2025

"""

import pytest
import zeus21
import numpy as np

from zeus21.maps import CoevalMaps, powerboxCtoR

def test_coevalmaps_initialization():
    """Test that CoevalMaps initializes correctly"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=100.) # Use higher kmax_CLASS as in test_astrophysics.py
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    CorrFClass = zeus21.Correlations(UserParams, CosmoParams, ClassyCosmo)
    
    # Generate T21 coefficients
    ZMIN = 20.0  # Use same ZMIN as in test_astrophysics.py
    Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
    
    # Generate power spectra
    PS21 = zeus21.Power_Spectra(UserParams, CosmoParams, AstroParams, ClassyCosmo, CorrFClass, Coeffs)
    
    # Test redshift
    ztest = 25.0  # Use a redshift that's compatible with our ZMIN setting
    
    # Initialize the map with reduced size for test performance
    map_obj = CoevalMaps(Coeffs, PS21, ztest, Lbox=300, Nbox=50, KIND=0, seed=12345)
    
    # Verify attributes
    assert map_obj.Lbox == 300
    assert map_obj.Nbox == 50
    assert map_obj.seed == 12345
    
    # Check that z is snapped to closest value in grid
    iz_test = min(range(len(Coeffs.zintegral)), key=lambda i: np.abs(Coeffs.zintegral[i]-ztest))
    assert map_obj.z == Coeffs.zintegral[iz_test]
    
    # Check T21global is properly set
    assert map_obj.T21global == pytest.approx(Coeffs.T21avg[iz_test])
    
    # Check map dimensions
    assert map_obj.T21map.shape == (50, 50, 50)
    
    # Check that density map is None for KIND=0
    assert map_obj.deltamap is None

def test_coevalmaps_kind1():
    """Test CoevalMaps with KIND=1 (correlated density and T21)"""
    # Set up the necessary objects
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=100.) # Use higher kmax_CLASS as in test_astrophysics.py
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    CorrFClass = zeus21.Correlations(UserParams, CosmoParams, ClassyCosmo)
    
    # Generate T21 coefficients
    ZMIN = 20.0  # Use same ZMIN as in test_astrophysics.py
    Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
    
    # Generate power spectra
    PS21 = zeus21.Power_Spectra(UserParams, CosmoParams, AstroParams, ClassyCosmo, CorrFClass, Coeffs)
    
    # Test redshift
    ztest = 25.0  # Use a redshift that's compatible with our ZMIN setting
    
    # Initialize the map with reduced size for test performance
    map_obj = CoevalMaps(Coeffs, PS21, ztest, Lbox=300, Nbox=50, KIND=1, seed=12345)
    
    # Verify all components exist
    assert map_obj.deltamap is not None
    assert map_obj.T21maplin is not None
    assert map_obj.T21mapNL is not None
    assert map_obj.T21map is not None
    
    # Check that maps have correct dimensions
    assert map_obj.deltamap.shape == (50, 50, 50)
    assert map_obj.T21maplin.shape == (50, 50, 50)
    assert map_obj.T21mapNL.shape == (50, 50, 50)
    assert map_obj.T21map.shape == (50, 50, 50)
    
    # Check that T21map is the sum of linear and non-linear components
    assert np.array_equal(map_obj.T21map, map_obj.T21maplin + map_obj.T21mapNL)
    
    # Check basic statistics of maps
    # Density map should have mean ≈ 0
    assert np.mean(map_obj.deltamap) == pytest.approx(0.0, abs=0.1)
    
    # T21maplin should have mean ≈ T21global
    assert np.mean(map_obj.T21maplin) == pytest.approx(map_obj.T21global, abs=5.0)
    
    # Verify standard deviation is not zero (actual field generated)
    assert np.std(map_obj.deltamap) > 0
    assert np.std(map_obj.T21map) > 0

def test_powerboxCtoR():
    """Test the powerboxCtoR utility function"""
    UserParams = zeus21.User_Parameters()
    CosmoParams_input = zeus21.Cosmo_Parameters_Input(kmax_CLASS=100.) # Use higher kmax_CLASS as in test_astrophysics.py
    ClassyCosmo = zeus21.runclass(CosmoParams_input)
    CosmoParams = zeus21.Cosmo_Parameters(UserParams, CosmoParams_input, ClassyCosmo)
    
    AstroParams = zeus21.Astro_Parameters(UserParams, CosmoParams)
    HMFintclass = zeus21.HMF_interpolator(UserParams, CosmoParams, ClassyCosmo)
    CorrFClass = zeus21.Correlations(UserParams, CosmoParams, ClassyCosmo)
    
    # Generate T21 coefficients
    ZMIN = 20.0  # Use same ZMIN as in test_astrophysics.py
    Coeffs = zeus21.get_T21_coefficients(UserParams, CosmoParams, ClassyCosmo, AstroParams, HMFintclass, zmin=ZMIN)
    
    # Generate power spectra
    PS21 = zeus21.Power_Spectra(UserParams, CosmoParams, AstroParams, ClassyCosmo, CorrFClass, Coeffs)
    
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