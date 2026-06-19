# TO BE DONE! 
# SL: for now I just move here the cosmo_wrapper from the cosmology.py, but we will need to add more wrapper functions here in the future. 

from .inputs import Cosmo_Parameters
from .cosmology import HMF_interpolator

def cosmo_wrapper(User_Parameters):
    """
    Wrapper function for all the cosmology. It takes Cosmo_Parameters_Input and returns:
    Cosmo_Parameters, Class_Cosmo, Correlations, HMF_interpolator
    """

    CosmoParams = Cosmo_Parameters(User_Parameters) 
    HMFintclass = HMF_interpolator(User_Parameters,CosmoParams)

    return CosmoParams, HMFintclass
