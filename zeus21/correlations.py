"""

Code to compute correlation functions from power spectra and functions of them. Holds two classes: Correlations (with matter correlation functions smoothed over different R), and Power_Spectra (which will compute and hold the 21-cm power spectrum and power for derived quantities like xa, Tk, etc.)

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Sarah Libanore
BGU - July 2025

Edited by Hector Afonso G. Cruz & Julian Munoz
UT Austin - May 2026
NYU - June 2026
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import mcfit
from scipy.special import gammaincc #actually very fast, no need to approximate
import numexpr as ne

from . import constants
from . import cosmology
from . import z21_utilities



class Power_Spectra:
    
    """
    Get the 21-cm power spectrum and its components from correlation functions and coefficients
    
    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    AstroParams : AstroParams class
    T21coeffs : T21coeffs class
    RSD_MODE : int
        Choice of redshift-space distortion mode.
        0 = None (mu=0), just for comparison with real-space
        1 = Spherical avg (like 21-cmFAST), standard assumption in sims
        2 = LoS only (mu=1), more observationally relevant
        Default is 1
    
    Attributes
    ----------
    Basic Setup Attributes
    
    self._rs_input_mcfit: array
        Input array of rs from mcfit P2xi used in inputs.py
    self.klist_PS: array
        Input array of wavenumbers used in inputs.py
    self.kwindow: array
        Output array of wavenumbers used in window function calls.
        Identical to klist_PS

    Window Function Attributes
    self.windowalpha_II: matrix
        Linear Pop II LyA window functions. Dimension (z, k)
    self.windowalpha_III
        Linear Pop III LyA window functions. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
    self.windowxray_II
        Linear Pop II Xray window functions. Dimension (z, k)
    self.windowxray_III
        Linear Pop III Xray window functions. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        
    Baryon Power Spectra Attributes (Used only if UserParams.USE_BARYON_FLAG == True)
    self.pK_bOnlyCLASS_intp: interpolator
        Baryon-only power spectrum, interpolated over redshift z and wavenumber k
    self.pK_bANDcbCLASS_intp: interpolator
        Baryon-CDM cross power spectrum, interpolated over redshift z and wavenumber k
        
    Linear Power Spectra
    self.Deltasq_xa_lin_II: matrix
        Linear Pop II contribution to LyA power spectrum. Dimension (z, k)
    self.Deltasq_xa_lin_III: matrix
        Linear Pop III contribution to LyA power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        Density-only power spectra used
    self.Deltasq_xa_lin_IIxIII: matrix
        Linear Pop II x III cross contribution to LyA power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        
    self.Deltasq_Tx_lin_II: matrix
        Linear Pop II contribution to Xray power spectrum. Dimension (z, k)
    self.Deltasq_Tx_lin_III: matrix
        Linear Pop III contribution to Xray power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        Density-only power spectra used (no linear eta power spectra)
    self.Deltasq_Tx_lin_IIxIII: matrix
        Linear Pop II x III cross contribution to Xray power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        
    self.Deltasq_xaTx_lin_II: matrix
        Linear Pop II contribution to LyA-Xray cross spectrum. Dimension (z, k)
    self.Deltasq_xaTx_lin_III: matrix
        Linear Pop III contribution to LyA-Xray cross spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        Density-only power spectra used (no linear eta power spectra)
    self.Deltasq_xaTx_lin_IIxIII: matrix
        Linear Pop II x III cross contribution to LyA-Xray cross spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
        
    self.Deltasq_d_lin: matrix
        Linear LSS power spectra. Dimension (z, k)
        Set to P_baryon(k) if UserParams.USE_BARYON_FLAG == True
    self.Deltasq_dxa_lin_II: matrix
        Linear Pop II density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
    self.Deltasq_dxa_lin_III: matrix
        Linear Pop III density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
        Set to zero if AstroParams.USE_POPIII == False
    self.Deltasq_dTx_lin_II: matrix
        Linear Pop II density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
    self.Deltasq_dTx_lin_III: matrix
        Linear Pop III density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
        Set to zero if AstroParams.USE_POPIII == False

    Total Power Spectra (including nonlinear corrections)
    self.Deltasq_xa_II: matrix
        Nonlinear Pop II contribution to LyA power spectrum. Dimension (z, k)
    self.Deltasq_xa_III: matrix
        Nonlinear Pop III contribution to LyA power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
    self.Deltasq_xa_IIxIII: matrix
        Nonlinear Pop II x III cross contribution to LyA power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False

    self.Deltasq_Tx_II: matrix
        Nonlinear Pop II contribution to Xray power spectrum. Dimension (z, k)
    self.Deltasq_Tx_III: matrix
        Nonlinear Pop III contribution to Xray power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
    self.Deltasq_Tx_IIxIII: matrix
        Nonlinear Pop II x III cross contribution to Xray power spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False

    self.Deltasq_xaTx_II: matrix
        Nonlinear Pop II contribution to LyA-Xray cross spectrum. Dimension (z, k)
    self.Deltasq_xaTx_III: matrix
        Nonlinear Pop III contribution to LyA-Xray cross spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False
    self.Deltasq_xaTx_IIxIII: matrix
        Nonlinear Pop II x III cross contribution to LyA-Xray cross spectrum. Dimension (z, k)
        Set to zero if AstroParams.USE_POPIII == False

    self.Deltasq_d: matrix
        Nonlinear LSS power spectra. Dimension (z, k)
        Set to P_baryon(k) if UserParams.USE_BARYON_FLAG == True
    self.Deltasq_dxa_II: matrix
        Nonlinear Pop II density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
    self.Deltasq_dxa_III: matrix
        Nonlinear Pop III density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
        Set to zero if AstroParams.USE_POPIII == False
    self.Deltasq_dTx_II: matrix
        Nonlinear Pop II density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
    self.Deltasq_dTx_III: matrix
        Nonlinear Pop III density-LyA cross power spectrum. Dimension (z, k)
        Uses P_baryonXcdm(k) if UserParams.USE_BARYON_FLAG == True
        Set to zero if AstroParams.USE_POPIII == False
    
    Combined (Pop II + Pop III) Total Power Spectra
    self.Deltasq_d: matrix
        Total density power spectrum. Dimension (z, k)
    self.Deltasq_dxa: matrix
        Total density-LyA cross power spectrum. Dimension (z, k)
    self.Deltasq_dTx: matrix
        Total density-Xray cross power spectrum. Dimension (z, k)
    self.Deltasq_xa: matrix
        Total LyA power spectrum. Dimension (z, k)
    self.Deltasq_xaTx: matrix
        Total LyA-Xray cross power spectrum. Dimension (z, k)
    self.Deltasq_Tx: matrix
        Total Xray power spectrum. Dimension (z, k)
    
    Ionization/Bubble Related Power Spectra
    self.Deltasq_xion: matrix
        Nonlinear ionization power spectrum. Dimension (z, k)
    self.Deltasq_xion_lin: matrix
        Linear ionization power spectrum. Dimension (z, k)
    self.Deltasq_dxion: matrix
        Nonlinear density-ionization cross power spectrum. Dimension (z, k)
    self.Deltasq_dxion_lin: matrix
        Linear density-ionization cross power spectrum. Dimension (z, k)
    self.Deltasq_xaxion: matrix
        Nonlinear LyA-ionization cross power spectrum. Dimension (z, k)
    self.Deltasq_xaxion_lin: matrix
        Linear LyA-ionization cross power spectrum. Dimension (z, k)
    self.Deltasq_Txxion: matrix
        Nonlinear Xray-ionization cross power spectrum. Dimension (z, k)
    self.Deltasq_Txxion_lin: matrix
        Linear Xray-ionization cross power spectrum. Dimension (z, k)
        
    Final 21-cm Power Spectra
    self.Deltasq_T21: matrix
        Total nonlinear 21-cm power spectrum. Dimension (z, k)
    self.Deltasq_T21_lin: matrix
        Total linear 21-cm power spectrum. Dimension (z, k)
    self.Deltasq_dT21: matrix
        Nonlinear Density-21 cm cross power spectrum. Dimension (z, k)
    self.Deltasq_dT21_lin: matrix
        Linear Density-21 cm cross power spectrum. Dimension (z, k)
    
    """

    def __init__(self, UserParams, CosmoParams, AstroParams, T21coeffs, RSD_MODE=1):

        #Variable set up
        self._rs_input_mcfit = CosmoParams.rlist_CF #just to make notation simpler
        self.klist_PS = CosmoParams._klistCF
        self.RSD_MODE = RSD_MODE #TODO: mode to save at different mu

        #first get the linear window functions -- note it already has growth factor in it, so it multiplies Pmatter(z=0)
        #fix some arrays: TYTYTY HERE

        self._zGreaterMatrix100, self._iRnonlinear, self._corrdNL =  self._prepare_corr_arrays(CosmoParams, T21coeffs)

        self.kwindow, self.windowalpha_II = self.get_xa_window(CosmoParams, AstroParams, T21coeffs, pop = 2)
        self._kwindowX, self.windowxray_II = self.get_Tx_window(CosmoParams, AstroParams, T21coeffs, pop = 2)
        

        if AstroParams.USE_POPIII == True:
        # SarahLibanore: add AstroParams to use flag on quadratic order
            self.kwindow, self.windowalpha_III = self.get_xa_window(CosmoParams,AstroParams, T21coeffs, pop = 3)
        # SarahLibanore: add AstroParams to use flag on quadratic order
            self._kwindowX, self.windowxray_III = self.get_Tx_window(CosmoParams, AstroParams, T21coeffs, pop = 3)
        else:
            self.windowalpha_III = np.zeros_like(self.windowalpha_II)
            self.windowxray_III = np.zeros_like(self.windowxray_II)
            
        #calculate some growth etc, and the bubble biases for the xHI linear window function:
        self._lingrowthd = cosmology.growth(CosmoParams, T21coeffs.zintegral)



        ##############################
        #If USE_BARYON_FLAG, use baryon and baryon-cdm power spectra for correlations involving delta_b

        if UserParams.USE_BARYON_FLAG:
            transfersMatrix = CosmoParams.ClassCosmo.get_transfer_and_k_and_z()

            fracB = CosmoParams.ClassCosmo.Om_b(0) / (CosmoParams.ClassCosmo.Om_b(0) + CosmoParams.ClassCosmo.Om_cdm(0))
            fracC = CosmoParams.ClassCosmo.Om_cdm(0) / (CosmoParams.ClassCosmo.Om_b(0) + CosmoParams.ClassCosmo.Om_cdm(0))

            zCLASS = transfersMatrix[2]
            kCLASS = transfersMatrix[1]; kCLASS[-1] = 0.999*kCLASS[-1] #to avoid interpolation errrors
            tCLASS = fracB * transfersMatrix[0]['d_b'] + fracC * transfersMatrix[0]['d_cdm']
            tCLASS_b = transfersMatrix[0]['d_b']

            pK_bOnlyCLASS = CosmoParams.ClassCosmo.pars['A_s'] * (kCLASS / 0.05)**(CosmoParams.ClassCosmo.pars['n_s']-1) * tCLASS_b.T**2 * (2 * np.pi**2/kCLASS**3)
            pK_bANDcbCLASS = CosmoParams.ClassCosmo.pars['A_s'] * (kCLASS / 0.05)**(CosmoParams.ClassCosmo.pars['n_s']-1) * tCLASS_b.T*tCLASS.T * (2 * np.pi**2/kCLASS**3)

            self.pK_bOnlyCLASS_intp = RegularGridInterpolator([zCLASS, kCLASS], pK_bOnlyCLASS, method = 'cubic')
            self.pK_bANDcbCLASS_intp = RegularGridInterpolator([zCLASS, kCLASS], pK_bANDcbCLASS, method = 'cubic')
        


        ##############################
        #Get all correlation functions

        # SarahLibanore: add AstroParams to use flag on quadratic order
        self.get_all_corrs_II(UserParams, CosmoParams, AstroParams, T21coeffs)
        
        if AstroParams.USE_POPIII == True:
            self.get_all_corrs_IIxIII(CosmoParams, T21coeffs) #compute Pop II x III dependent cross power spectra
            self.get_all_corrs_III(UserParams, CosmoParams, T21coeffs) #compute Pop III-dependent power spectra
        else:
            #bypases Pop III correlation routine and sets all Pop III-dependent correlations to zero
            self._IIxIII_deltaxi_xa = np.zeros_like(self._II_deltaxi_xa)
            self._IIxIII_deltaxi_Tx = np.zeros_like(self._II_deltaxi_xa)
            self._IIxIII_deltaxi_xaTx = np.zeros_like(self._II_deltaxi_xa)

            self._III_deltaxi_xa = np.zeros_like(self._II_deltaxi_xa)
            self._III_deltaxi_dxa = np.zeros_like(self._II_deltaxi_xa)

            self._III_deltaxi_Tx = np.zeros_like(self._II_deltaxi_xa)
            self._III_deltaxi_xaTx = np.zeros_like(self._II_deltaxi_xa)
            self._III_deltaxi_dTx = np.zeros_like(self._II_deltaxi_xa)
            
            
        self._k3over2pi2 = (self.klist_PS**3)/(2.0 * np.pi**2)

        #and now define power spectra:
        #for xalpha, first linear
        self._Pk_xa_lin_II = self.windowalpha_II**2 * CosmoParams._PklinCF
        self._Pk_xa_lin_III = self.windowalpha_III**2 * CosmoParams._PklinCF #doesn't include linear eta power spectra
        self._Pk_xa_lin_IIxIII = 2* self.windowalpha_II * self.windowalpha_III * CosmoParams._PklinCF #Pop IIxIII cross term doesn't have a velocity component

        self.Deltasq_xa_lin_II = self._Pk_xa_lin_II * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_lin_III = self._Pk_xa_lin_III * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_lin_IIxIII = self._Pk_xa_lin_IIxIII * self._k3over2pi2 #note that it still has units of xa_avg

        #nonlinear corrections too:
        self._d_Pk_xa_nl_II = self.get_list_PS(self._II_deltaxi_xa, T21coeffs.zintegral)
        self._d_Pk_xa_nl_III = self.get_list_PS(self._III_deltaxi_xa, T21coeffs.zintegral) #velocity correlations already embedded in nonlinear computation
        self._d_Pk_xa_nl_IIxIII = self.get_list_PS(self._IIxIII_deltaxi_xa, T21coeffs.zintegral)

        self.Deltasq_xa_II = self.Deltasq_xa_lin_II + self._d_Pk_xa_nl_II * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_III = self.Deltasq_xa_lin_III + self._d_Pk_xa_nl_III * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_IIxIII = self.Deltasq_xa_lin_IIxIII + self._d_Pk_xa_nl_IIxIII * self._k3over2pi2 #note that it still has units of xa_avg


        ##############################


        #and same for xray
        self._Pk_Tx_lin_II = self.windowxray_II**2 * CosmoParams._PklinCF
        self._Pk_Tx_lin_III = self.windowxray_III**2 * CosmoParams._PklinCF #doesn't include linear eta power spectra
        self._Pk_Tx_lin_IIxIII = 2* self.windowxray_II * self.windowxray_III * CosmoParams._PklinCF #Pop IIxIII cross term doesn't have a velocity component

        self.Deltasq_Tx_lin_II = self._Pk_Tx_lin_II * self._k3over2pi2
        self.Deltasq_Tx_lin_III = self._Pk_Tx_lin_III * self._k3over2pi2
        self.Deltasq_Tx_lin_IIxIII = self._Pk_Tx_lin_IIxIII * self._k3over2pi2

        self._d_Pk_Tx_nl_II = self.get_list_PS(self._II_deltaxi_Tx, T21coeffs.zintegral)
        self._d_Pk_Tx_nl_III = self.get_list_PS(self._III_deltaxi_Tx, T21coeffs.zintegral)
        self._d_Pk_Tx_nl_IIxIII = self.get_list_PS(self._IIxIII_deltaxi_Tx, T21coeffs.zintegral)

        self.Deltasq_Tx_II = self.Deltasq_Tx_lin_II + self._d_Pk_Tx_nl_II * self._k3over2pi2
        self.Deltasq_Tx_III = self.Deltasq_Tx_lin_III + self._d_Pk_Tx_nl_III * self._k3over2pi2
        self.Deltasq_Tx_IIxIII = self.Deltasq_Tx_lin_IIxIII + self._d_Pk_Tx_nl_IIxIII * self._k3over2pi2


        ##############################


        #and their cross correlation
        self._Pk_xaTx_lin_II = self.windowalpha_II * self.windowxray_II * CosmoParams._PklinCF
        self._Pk_xaTx_lin_III = self.windowalpha_III * self.windowxray_III * CosmoParams._PklinCF #doesn't include linear eta power spectra
        self._Pk_xaTx_lin_IIxIII = (self.windowalpha_II * self.windowxray_III + self.windowalpha_III * self.windowxray_II) * CosmoParams._PklinCF

        self.Deltasq_xaTx_lin_II = self._Pk_xaTx_lin_II * self._k3over2pi2
        self.Deltasq_xaTx_lin_III = self._Pk_xaTx_lin_III * self._k3over2pi2
        self.Deltasq_xaTx_lin_IIxIII = self._Pk_xaTx_lin_IIxIII * self._k3over2pi2

        self._d_Pk_xaTx_nl_II = self.get_list_PS(self._II_deltaxi_xaTx, T21coeffs.zintegral)
        self._d_Pk_xaTx_nl_III = self.get_list_PS(self._III_deltaxi_xaTx, T21coeffs.zintegral)
        self._d_Pk_xaTx_nl_IIxIII = self.get_list_PS(self._IIxIII_deltaxi_xaTx, T21coeffs.zintegral)

        self.Deltasq_xaTx_II = self.Deltasq_xaTx_lin_II + self._d_Pk_xaTx_nl_II * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xaTx_III = self.Deltasq_xaTx_lin_III + self._d_Pk_xaTx_nl_III * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xaTx_IIxIII = self.Deltasq_xaTx_lin_IIxIII + self._d_Pk_xaTx_nl_IIxIII * self._k3over2pi2 #note that it still has units of xa_avg


        ##############################
        
        
        #and the same for deltaNL and its cross terms:
        if UserParams.USE_BARYON_FLAG == 1:
            
            zInput, kInput = np.meshgrid(T21coeffs.zintegral, self.klist_PS, indexing='ij', sparse=True)

            self._PklinCF_bb = self.pK_bOnlyCLASS_intp((zInput, kInput))
            self._PklinCF_bm = self.pK_bANDcbCLASS_intp((zInput, kInput))
            
            self._Pk_d_lin = self._PklinCF_bb #No Pop II or III contribution
            self._Pk_dxa_lin_II = self.windowalpha_II * self._PklinCF_bm /np.transpose([self._lingrowthd])
            self._Pk_dxa_lin_III = self.windowalpha_III * self._PklinCF_bm /np.transpose([self._lingrowthd])#No velocity component
            self._Pk_dTx_lin_II = self.windowxray_II * self._PklinCF_bm /np.transpose([self._lingrowthd])
            self._Pk_dTx_lin_III = self.windowxray_III * self._PklinCF_bm /np.transpose([self._lingrowthd])#No velocity component
            
        else:
            self._Pk_d_lin = np.outer(self._lingrowthd**2, CosmoParams._PklinCF) #No Pop II or III contribution
            self._Pk_dxa_lin_II = (self.windowalpha_II.T * self._lingrowthd).T * CosmoParams._PklinCF
            self._Pk_dxa_lin_III = (self.windowalpha_III.T * self._lingrowthd).T * CosmoParams._PklinCF #No velocity component
            self._Pk_dTx_lin_II = (self.windowxray_II.T * self._lingrowthd).T * CosmoParams._PklinCF
            self._Pk_dTx_lin_III = (self.windowxray_III.T * self._lingrowthd).T * CosmoParams._PklinCF #No velocity component
        
        self.Deltasq_d_lin = self._Pk_d_lin * self._k3over2pi2 #note that it still has units of xa_avg
        
        self.Deltasq_dxa_lin_II = self._Pk_dxa_lin_II * self._k3over2pi2
        self.Deltasq_dxa_lin_III = self._Pk_dxa_lin_III * self._k3over2pi2 #No velocity component

        self.Deltasq_dTx_lin_II = self._Pk_dTx_lin_II * self._k3over2pi2
        self.Deltasq_dTx_lin_III = self._Pk_dTx_lin_III * self._k3over2pi2 #No velocity component

        self._Pk_d =  self._Pk_d_lin
        self._Pk_dxa_II =  self._Pk_dxa_lin_II
        self._Pk_dxa_III =  self._Pk_dxa_lin_III
        self._Pk_dTx_II =  self._Pk_dTx_lin_II
        self._Pk_dTx_III =  self._Pk_dTx_lin_III

        if(UserParams.FLAG_DO_DENS_NL): #note that the nonlinear terms (cross and auto) below here have the growth already accounted for

            self._d_Pk_d_nl = self.get_list_PS(self._II_deltaxi_d, T21coeffs.zintegral)
            self._Pk_d += self._d_Pk_d_nl

            self._d_Pk_dxa_nl_II = self.get_list_PS(self._II_deltaxi_dxa, T21coeffs.zintegral)
            self._d_Pk_dxa_nl_III = self.get_list_PS(self._III_deltaxi_dxa, T21coeffs.zintegral)
            self._Pk_dxa_II += self._d_Pk_dxa_nl_II
            self._Pk_dxa_III += self._d_Pk_dxa_nl_III

            self._d_Pk_dTx_nl_II = self.get_list_PS(self._II_deltaxi_dTx, T21coeffs.zintegral)
            self._d_Pk_dTx_nl_III = self.get_list_PS(self._III_deltaxi_dTx, T21coeffs.zintegral)

            self._Pk_dTx_II += self._d_Pk_dTx_nl_II
            self._Pk_dTx_III += self._d_Pk_dTx_nl_III

        self.Deltasq_d = self._Pk_d * self._k3over2pi2

        self.Deltasq_dxa_II = self._Pk_dxa_II * self._k3over2pi2
        self.Deltasq_dxa_III = self._Pk_dxa_III * self._k3over2pi2

        self.Deltasq_dTx_II = self._Pk_dTx_II * self._k3over2pi2
        self.Deltasq_dTx_III = self._Pk_dTx_III * self._k3over2pi2


        ##############################


        #and xHI too. Linear part does not have bubbles, only delta part
        if(constants.FLAG_DO_BUBBLES):
            #auto
            self._Pk_xion_lin = self.windowxion**2 * CosmoParams._PklinCF
            self.Deltasq_xion_lin = self._Pk_xion_lin * self._k3over2pi2

            self._d_Pk_xion_nl = self.get_list_PS(self._deltaxi_xi, T21coeffs.zintegral)
            self.Deltasq_xion = self.Deltasq_xion_lin + self._d_Pk_xion_nl * self._k3over2pi2

            #cross with density
            self._Pk_dxion_lin = (self.windowxion.T * self._lingrowthd).T  * CosmoParams._PklinCF
            self.Deltasq_dxion_lin = self._Pk_dxion_lin * self._k3over2pi2

            self._d_Pk_dxion_nl = self.get_list_PS(self._deltaxi_dxi, T21coeffs.zintegral)
            self.Deltasq_dxion = self.Deltasq_dxion_lin + self._d_Pk_dxion_nl * self._k3over2pi2

            #cross with xa
            self._Pk_xaxion_lin = self.windowxion * self.windowalpha  * CosmoParams._PklinCF
            self.Deltasq_xaxion_lin = self._Pk_xaxion_lin * self._k3over2pi2

            self._d_Pk_xaxion_nl = self.get_list_PS(self._deltaxi_xaxi, T21coeffs.zintegral)
            self.Deltasq_xaxion = self.Deltasq_xaxion_lin + self._d_Pk_xaxion_nl * self._k3over2pi2

            #and cross with Tx
            self._Pk_Txxion_lin = self.windowxion * self.windowxray  * CosmoParams._PklinCF
            self.Deltasq_Txxion_lin = self._Pk_Txxion_lin * self._k3over2pi2

            self._d_Pk_Txxion_nl = self.get_list_PS(self._deltaxi_Txxi, T21coeffs.zintegral)
            self.Deltasq_Txxion = self.Deltasq_Txxion_lin + self._d_Pk_Txxion_nl * self._k3over2pi2
        else:
            self.Deltasq_xion =  np.zeros_like(self.Deltasq_d)
            self.Deltasq_xion_lin = np.zeros_like(self.Deltasq_d)
            self.Deltasq_dxion =  np.zeros_like(self.Deltasq_d)
            self.Deltasq_dxion_lin = np.zeros_like(self.Deltasq_d)
            self.Deltasq_xaxion =  np.zeros_like(self.Deltasq_d)
            self.Deltasq_xaxion_lin = np.zeros_like(self.Deltasq_d)
            self.Deltasq_Txxion = np.zeros_like(self.Deltasq_d)
            self.Deltasq_Txxion_lin = np.zeros_like(self.Deltasq_d)
            #These have to be defined even if no EoR bubbles


        ##############################
        #Compute the 21-cm power spectrum

        self._betaT = T21coeffs.T_CMB/T21coeffs.Tk_avg /(T21coeffs.invTcol_avg**-1 - T21coeffs.T_CMB) #multiplies \delta T_x and \delta T_ad [both dimensionful, not \deltaT/T]
        self._betaxa = 1./(1. + T21coeffs.xa_avg)/T21coeffs.xa_avg #multiplies \delta x_a [again not \delta xa/xa]

        #calculate beta_adiabatic
        self._dlingrowthd_dz = cosmology.dgrowth_dz(CosmoParams, T21coeffs.zintegral)

        _factor_adi_ = (1+T21coeffs.zintegral)**2
        _integrand_adi = T21coeffs.Tk_avg*self._dlingrowthd_dz/_factor_adi_ * T21coeffs.dlogzint*T21coeffs.zintegral

        if(CosmoParams.Flag_emulate_21cmfast==True):
            _hizintegral = 0.0 #they do not account for the adiabatic history prior to starting their evolution. It misses ~half of the adiabatic flucts.
        else:
            #the z>zmax part of the integral we do aside. Assume Tk=Tadiabatic from CLASS.
            _zlisthighz_ = np.linspace(T21coeffs.zintegral[-1], 99., 100) #beyond z=100 need to explictly tell CLASS to save growth
            _dgrowthhighz_ = cosmology.dgrowth_dz(CosmoParams, _zlisthighz_)
            _hizintegral = np.trapezoid(cosmology.Tadiabatic(CosmoParams,_zlisthighz_)
            /(1+_zlisthighz_)**2 * _dgrowthhighz_, _zlisthighz_)

        self._betaTad_ = -2./3. * _factor_adi_/self._lingrowthd * (np.cumsum(_integrand_adi[::-1])[::-1] + _hizintegral) #units of Tk_avg. Internal sum goes from high to low z (backwards), minus sign accounts for it properly so it's positive.
        self._betaTad_ *= self._betaT #now it's dimensionless, since it multiplies \delta_m(k,z)



        self._betad = (1.0 + self._betaTad_)# this includes both the usual (1+d) and the adiabatic Tk contribution. Now we add RSD
        if(self.RSD_MODE==0): #no RSD (real space)
            pass #nothing to change
        elif(self.RSD_MODE==1): #spherically avg'd RSD
            self._betad += constants.MU_AVG ** 2
        elif(self.RSD_MODE==2): #LoS RSD (mu=1)
            self._betad += constants.MU_LoS ** 2
        else:
            print('Error, have to choose an RSD mode! RSD_MODE')

        if(constants.FLAG_DO_BUBBLES):
            self._betaxion = - 1.0/T21coeffs.xHI_avg * np.heaviside(constants.ZMAX_Bubbles - T21coeffs.zintegral, 0.5) # xion = 1 - xHI, only for z<ZMAX_Bubbles. 1/xHI_avg since P_xHI has units of xHI
        else:
            self._betaxion = np.zeros_like(T21coeffs.xHI_avg) # do not do EoR bubbles at all


        ##############################


        #To first order: dT21/T0 = (1+cT * betaTad) * delta_m + betaT * deltaTX + betaxa * delta xa + betaxion * delta xion

        self._allbetas = np.array([self._betad, self._betaxa, self._betaT, self._betaxion])
        self._allbetamatrix = np.einsum('ij,kj->ikj', self._allbetas, self._allbetas)

        #Sum Pop II and Pop III contributions
        self.Deltasq_d = self.Deltasq_d
        self.Deltasq_dxa = self.Deltasq_dxa_II + self.Deltasq_dxa_III
        self.Deltasq_dTx = self.Deltasq_dTx_II + self.Deltasq_dTx_III

        self.Deltasq_xa = self.Deltasq_xa_II + self.Deltasq_xa_III + self.Deltasq_xa_IIxIII
        self.Deltasq_xaTx = self.Deltasq_xaTx_II + self.Deltasq_xaTx_III + self.Deltasq_xaTx_IIxIII
        self.Deltasq_Tx = self.Deltasq_Tx_II + self.Deltasq_Tx_III + self.Deltasq_Tx_IIxIII


        self._allcorrs = np.array( [[self.Deltasq_d, self.Deltasq_dxa, self.Deltasq_dTx, self.Deltasq_dxion], \
                                    [self.Deltasq_dxa, self.Deltasq_xa, self.Deltasq_xaTx, self.Deltasq_xaxion], \
                                    [self.Deltasq_dTx, self.Deltasq_xaTx, self.Deltasq_Tx, self.Deltasq_Txxion], \
                                    [self.Deltasq_dxion, self.Deltasq_xaxion, self.Deltasq_Txxion, self.Deltasq_xion]]\
                                        )

        self.Deltasq_T21 = np.einsum('ijk...,ijkl...->kl...', self._allbetamatrix, self._allcorrs)
        self.Deltasq_T21 = (self.Deltasq_T21.T*T21coeffs.T21avg**2).T
        
        self.Deltasq_dT21 = (np.einsum('ik...,ikl...->kl...',self._allbetas,self._allcorrs[0]).T*T21coeffs.T21avg).T


        #Sum Linear Pop II and Pop III contributions
        self.Deltasq_d_lin = self.Deltasq_d_lin
        self.Deltasq_dxa_lin = self.Deltasq_dxa_lin_II + self.Deltasq_dxa_lin_III
        self.Deltasq_dTx_lin = self.Deltasq_dTx_lin_II + self.Deltasq_dTx_lin_III

        self.Deltasq_xa_lin = self.Deltasq_xa_lin_II + self.Deltasq_xa_lin_III + self.Deltasq_xa_lin_IIxIII
        self.Deltasq_xaTx_lin = self.Deltasq_xaTx_lin_II + self.Deltasq_xaTx_lin_III + self.Deltasq_xaTx_lin_IIxIII
        self.Deltasq_Tx_lin = self.Deltasq_Tx_lin_II + self.Deltasq_Tx_lin_III + self.Deltasq_Tx_lin_IIxIII


        self._allcorrs_lin = np.array( [[self.Deltasq_d_lin, self.Deltasq_dxa_lin, self.Deltasq_dTx_lin, self.Deltasq_dxion_lin], \
                                    [self.Deltasq_dxa_lin, self.Deltasq_xa_lin, self.Deltasq_xaTx_lin, self.Deltasq_xaxion_lin], \
                                    [self.Deltasq_dTx_lin, self.Deltasq_xaTx_lin, self.Deltasq_Tx_lin, self.Deltasq_Txxion_lin], \
                                    [self.Deltasq_dxion_lin, self.Deltasq_xaxion_lin, self.Deltasq_Txxion_lin, self.Deltasq_xion_lin]]\
                                        )

        self.Deltasq_T21_lin = np.einsum('ijk...,ijkl...->kl...', self._allbetamatrix, self._allcorrs_lin)
        self.Deltasq_T21_lin = (self.Deltasq_T21_lin.T*T21coeffs.T21avg**2).T

        self.Deltasq_dT21_lin = (np.einsum('ik...,ikl...->kl...',self._allbetas,self._allcorrs_lin[0]).T*T21coeffs.T21avg).T
#        print("Power Spectral Routine Done!")



    def _prepare_corr_arrays(self, CosmoParams, T21coeffs):
        """
        Prepare correlation arrays to save computation time

        Parameters
        ----------
        CosmoParams : CosmoParams class
        T21coeffs : T21coeffs class

        Returns
        ----------
        zGM : matrix
            Identical to zGreaterMatrix but with NaNs replaced with 100 for computational ease. Dimension (z, k)
        iR: array
            Defines which indices are nonlinear
        corr: matrix
            Matter correlation function at iR indices. Dimensions (1, iR, iR, k)
        """
        
        zGM = np.copy(T21coeffs.zGreaterMatrix)
        zGM[np.isnan(zGM)] = 100
        iR = np.arange(CosmoParams.indexmaxNL)
        corr = CosmoParams.xi_RR_CF[np.ix_(iR, iR)]
        corr[:CosmoParams.indexminNL, :CosmoParams.indexminNL] = \
            corr[CosmoParams.indexminNL, CosmoParams.indexminNL]
        return zGM, iR, corr.reshape((1, *corr.shape))

    # SarahLibanore: add AstroParams to use flag on quadratic order
    def get_xa_window(self, CosmoParams, AstroParams, T21coeffs, pop = 0): #set pop to 2 or 3, default zero just so python doesn't complain
        """
        Computes the LyA window functions for each stellar population across z and k.

        Parameters
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        T21coeffs : T21coeffs class
        pop: int
            Which stellar population to use. 2 for Pop II, 3 for Pop III.

        Returns
        ----------
        _kwinalpha : array
            Array of wavenumbers
        _win_alpha: matrix
            Matrix of LyA window functions. Dimension (z, k)

        """

        coeffzp = T21coeffs.coeff1LyAzp
        coeffJaxa = T21coeffs.coeff_Ja_xa

        growthRmatrix = cosmology.growth(CosmoParams, self._zGreaterMatrix100)

        if pop == 2:
            coeffRmatrix = T21coeffs.coeff2LyAzpRR_II
            gammaRmatrix = T21coeffs.gamma_II_index2D * growthRmatrix
        elif pop == 3:
            coeffRmatrix = T21coeffs.coeff2LyAzpRR_III
            gammaRmatrix = T21coeffs.gamma_III_index2D * growthRmatrix
        else:
            print("Must set pop to either 2 or 3!")

        _wincoeffsMatrix = coeffRmatrix * gammaRmatrix
        # SarahLibanore: quadratic order in the lognormal
        if AstroParams.quadratic_SFRD_lognormal:
           _wincoeffsMatrix *= 1./(1-2.*T21coeffs.gamma2_II_index2D*T21coeffs.sigmaofRtab**2)

        if(CosmoParams.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
            _wincoeffsMatrix /=(4*np.pi * CosmoParams._Rtabsmoo**2) * (CosmoParams._Rtabsmoo * CosmoParams._dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2 * Delta r)
            _kwinalpha, _win_alpha = self.get_Pk_from_xi(CosmoParams._Rtabsmoo, _wincoeffsMatrix)

        else:
            _kwinalpha = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RtabsmooMesh, kWinAlphaMesh = np.meshgrid(T21coeffs.zintegral, CosmoParams._Rtabsmoo, _kwinalpha, indexing = 'ij', sparse = True)

            _win_alpha = coeffRgammaRmatrix * z21_utilities._WinTH(RtabsmooMesh, kWinAlphaMesh)
            _win_alpha = np.sum(_win_alpha, axis = 1)

        _win_alpha *= np.array([coeffzp*coeffJaxa]).T
        
        return _kwinalpha, _win_alpha


    # SarahLibanore: add AstroParams to use flag on quadratic order
    def get_Tx_window(self, CosmoParams, AstroParams, T21coeffs, pop = 0): #set pop to 2 or 3, default zero just so python doesn't complain
        """
        Computes the Xray window functions for each stellar population across z and k.

        Parameters
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        T21coeffs : T21coeffs class
        pop: int
            Which stellar population to use. 2 for Pop II, 3 for Pop III.

        Returns
        ----------
        _kwinTx : array
            Array of wavenumbers
        _win_Tx: matrix
            Matrix of Xray window functions. Dimension (z, k)

        """

        coeffzp = np.array([T21coeffs.coeff1Xzp]).T
        growthRmatrix = cosmology.growth(CosmoParams, self._zGreaterMatrix100)

        if pop == 2:
            coeffRmatrix = T21coeffs.coeff2XzpRR_II
            gammaRmatrix = T21coeffs.gamma_II_index2D * growthRmatrix
            _coeffTx_units = T21coeffs.coeff_Gammah_Tx_II#z-dependent, includes 10^40 erg/s/SFR normalizaiton and erg/K conversion factor, and the 1/(1+z)^2 factor to compensate the adiabatic cooling of the Tx olny part
        elif pop == 3:
            coeffRmatrix = T21coeffs.coeff2XzpRR_III
            gammaRmatrix = T21coeffs.gamma_III_index2D * growthRmatrix
            _coeffTx_units = T21coeffs.coeff_Gammah_Tx_III
        else:
            print("Must set pop to either 2 or 3!")

        # SarahLibanore: quadratic order in the lognormal
        if AstroParams.quadratic_SFRD_lognormal:
           gammaRmatrix *= (1/(1-2.*T21coeffs.gamma2_II_index2D*T21coeffs.sigmaofRtab**2))

        if(CosmoParams.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
            _wincoeffs = coeffRmatrix * gammaRmatrix #array in logR space
            _wincoeffs /=(4*np.pi * CosmoParams._Rtabsmoo**2) * (CosmoParams._Rtabsmoo * CosmoParams._dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2) * Delta r
            _kwinTx, _win_Tx_curr = self.get_Pk_from_xi(CosmoParams._Rtabsmoo, _wincoeffs)

        else:
            _kwinTx = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RtabsmooMesh, kWinTxMesh = np.meshgrid(T21coeffs.zintegral, CosmoParams._Rtabsmoo, _kwinTx, indexing = 'ij', sparse = True)

            _win_Tx_curr = coeffRgammaRmatrix * z21_utilities._WinTH(RtabsmooMesh, kWinTxMesh)
            _win_Tx_curr = np.sum(_win_Tx_curr , axis = 1)

        _win_Tx = _win_Tx_curr * coeffzp
        _win_Tx = np.cumsum(_win_Tx[::-1], axis = 0)[::-1]

        _win_Tx =_win_Tx * np.array([_coeffTx_units]).T

        return _kwinTx, _win_Tx


    # SarahLibanore: function modified to include quadratic order
    def get_all_corrs_II(self, UserParams, CosmoParams, AstroParams, T21coeffs):

        """
        Computes the Pop II correlation functions across z and R.

        Parameters
        ----------
        UserParams : UserParams class
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        T21coeffs : T21coeffs class

        Returns
        ----------
        Attributes stored in Power_Spectra

        """
    

        _coeffTx_units = T21coeffs.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(CosmoParams,self._zGreaterMatrix100[:, self._iRnonlinear])

        coeffzp1xa = T21coeffs.coeff1LyAzp * T21coeffs.coeff_Ja_xa
        coeffzp1Tx = T21coeffs.coeff1Xzp

        coeffR1xa = T21coeffs.coeff2LyAzpRR_II[:,self._iRnonlinear]
        coeffR1Tx = T21coeffs.coeff2XzpRR_II[:,self._iRnonlinear]

        coeffmatrixxa = coeffR1xa.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1) * coeffR1xa.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)

        # gammaR1 = T21coeffs.gamma_II_index2D[:, self._iRnonlinear] * growthRmatrix
        # gammamatrixR1R1 = gammaR1.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1) * gammaR1.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)

        # gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL')#np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster


        # SarahLibanore : change to introduce quantities required in the second order correction
        # --- #
        growthRmatrix1 = growthRmatrix.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1)
        growthRmatrix2 = growthRmatrix.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)
        growth_corr = growthRmatrix1 * growthRmatrix2

        gammaR1 = T21coeffs.gamma_II_index2D[:, self._iRnonlinear]
        sigmaR1 = T21coeffs.sigmaofRtab[:, self._iRnonlinear]
        sR1 = (sigmaR1).reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1)
        sR2 = (sigmaR1).reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)

        g1 = (gammaR1 * sigmaR1).reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1)
        g2 = (gammaR1 * sigmaR1).reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)
        gammamatrixR1R1 = g1 * g2

        corrdNL = self._corrdNL
        corrdNL_gs = ne.evaluate('corrdNL * growth_corr/ (sR1 * sR2)')
        gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL_gs')

        if AstroParams.quadratic_SFRD_lognormal:

            gammaR1NL = T21coeffs.gamma2_II_index2D[:, self._iRnonlinear]
            g1NL = (gammaR1NL * sigmaR1**2).reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1)
            g2NL = (gammaR1NL * sigmaR1**2).reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)

            numerator_NL = ne.evaluate('gammaTimesCorrdNL+ g1 * g1 * (0.5 - g2NL * (1 - corrdNL_gs * corrdNL_gs)) + g2 * g2 * (0.5 - g1NL * (1 - corrdNL_gs * corrdNL_gs))')
            
            denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - corrdNL_gs * corrdNL_gs)')
            
            norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)')
            norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)')
            
            log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')
            nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)')

            # use second order in SFRD lognormal approx
            expGammaCorrMinusLinear = ne.evaluate('nonlinearcorrelation - 1-gammaTimesCorrdNL/((-1+2.*g1NL)*(-1+2.*g2NL))')
        else:
            expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')

        self._II_deltaxi_xa = np.einsum('ijkl->il', coeffmatrixxa * expGammaCorrMinusLinear, optimize = True)
        self._II_deltaxi_xa *= np.array([coeffzp1xa]).T**2 #brings it to xa units

        if (UserParams.FLAG_DO_DENS_NL):
            D_coeffR1xa = coeffR1xa.reshape(*coeffR1xa.shape, 1)
            DDgammaR1 = T21coeffs.gamma_II_index2D[:, self._iRnonlinear]
            D_gammaR1 = DDgammaR1.reshape(*DDgammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            # SarahLibanore
            if AstroParams.quadratic_SFRD_lognormal:

                DDsigmaR1 = T21coeffs.sigmaofRtab[:, self._iRnonlinear]
                D_sigmaR1 = DDsigmaR1.reshape(*DDsigmaR1.shape , 1)
                DDgammaR1N = T21coeffs.gamma2_II_index2D[:, self._iRnonlinear]
                D_gammaR1N = DDgammaR1N.reshape(*DDgammaR1N.shape , 1)
  
                gammaTimesCorrdNL = ne.evaluate('D_gammaR1 * D_growthRmatrix* D_growthRmatrix * D_corrdNL')
                numerator_NL = ne.evaluate('gammaTimesCorrdNL+ D_gammaR1 * D_gammaR1 * D_sigmaR1* D_sigmaR1 /2 + D_gammaR1N * D_growthRmatrix * D_growthRmatrix* D_growthRmatrix * D_growthRmatrix * (D_corrdNL * D_corrdNL)')
                
                denominator_NL = ne.evaluate('1. - 2 * D_gammaR1N*D_sigmaR1*D_sigmaR1')
                
                norm1 = ne.evaluate('exp(D_gammaR1 * D_gammaR1 * D_sigmaR1* D_sigmaR1 * D_gammaR1 * D_gammaR1 * D_sigmaR1* D_sigmaR1 / (2 - 4 * D_gammaR1N*D_sigmaR1*D_sigmaR1)) / sqrt(1 - 2 * D_gammaR1N*D_sigmaR1*D_sigmaR1)')
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1)')
                nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)')

                self._II_deltaxi_dxa = np.sum(D_coeffR1xa * (
                    nonlinearcorrelation - 1 - D_gammaR1 * D_growthRmatrix**2 * D_corrdNL/(1-2.*D_gammaR1N*D_sigmaR1**2)
                     ), axis = 1)

            else:
                self._II_deltaxi_dxa = np.sum(D_coeffR1xa * ((np.exp(D_gammaR1 * D_growthRmatrix**2 * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix**2 * D_corrdNL), axis = 1)

            self._II_deltaxi_d = (np.exp(growthRmatrix[:,:1]**2 * corrdNL[0,0,0,:]) - 1.0) - growthRmatrix[:,:1]**2 * corrdNL[0,0,0,:]
    
            self._II_deltaxi_dxa *= np.array([coeffzp1xa]).T


        ### To compute Tx quantities, I'm broadcasting arrays such that the axes are zp1, R1, zp2, R2, and looping over r
        # gammaR2 = np.copy(gammaR1) #already has growth factor in this
        # gammamatrixR1R2 = gammaR1.reshape(*gammaR1.shape, 1, 1) * gammaR2.reshape(1, 1, *gammaR2.shape)

        coeffzp1Tx = np.copy(T21coeffs.coeff1Xzp).reshape(*T21coeffs.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21coeffs.coeff1Xzp).reshape(1, 1, *T21coeffs.coeff1Xzp.shape, 1)

        coeffR2Tx = np.copy(coeffR1Tx)
        coeffmatrixTxTx = coeffR1Tx.reshape(*coeffR1Tx.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffmatrixxaTx = coeffR1xa.reshape(*coeffR1xa.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffsTxALL =  coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx
        coeffsXaTxALL = coeffzp2Tx * coeffmatrixxaTx

        gammaR2 = np.copy(gammaR1) #already has growth factor in this
        sigmaR2 = np.copy(sigmaR1) #already has growth factor in this

        growthRmatrix1 = growthRmatrix.reshape(*gammaR1.shape, 1, 1)
        growthRmatrix2 = growthRmatrix.reshape(1, 1, *gammaR2.shape)
        growth_corr = growthRmatrix1 * growthRmatrix2

        g1 = (gammaR1 * sigmaR1).reshape(*gammaR1.shape, 1, 1)
        sR1 = (sigmaR1).reshape(*gammaR1.shape, 1, 1)
        g2 = (gammaR2 * sigmaR2).reshape(1, 1, *gammaR2.shape)
        sR2 = (sigmaR2).reshape(1, 1, *gammaR2.shape)
        if AstroParams.quadratic_SFRD_lognormal:
            gammaR2NL = np.copy(gammaR1NL)
            g1NL = (gammaR1NL * sigmaR1**2).reshape(*gammaR1NL.shape, 1, 1)
            g2NL = (gammaR2NL * sigmaR2**2).reshape(1, 1, *gammaR2NL.shape)

        gammamatrixR1R2 = g1 * g2

        self._II_deltaxi_Tx = np.zeros_like(self._II_deltaxi_xa)
        self._II_deltaxi_xaTx = np.zeros_like(self._II_deltaxi_xa)
        corrdNLBIG = corrdNL[:,:, np.newaxis, :,:] #dimensions zp1, R1, zp2, R2, and r which will be looped over below
        for ir in range(len(CosmoParams._Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            
            corrdNL_gs = ne.evaluate('corrdNL * growth_corr / (sR1 * sR2)')

            #HAC: Computations using ne.evaluate(...) use numexpr, which speeds up computations of massive numpy arrays
            gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R2 * corrdNL_gs')
            if AstroParams.quadratic_SFRD_lognormal:

                numerator_NL = ne.evaluate('gammaTimesCorrdNL + g1 * g1 * (0.5 - g2NL * (1 - corrdNL_gs * corrdNL_gs)) + g2 * g2 * (0.5 - g1NL * (1 - corrdNL_gs * corrdNL_gs))')
                denominator_NL = ne.evaluate('1. - 2 * g1NL - 2 * g2NL + 4 * g1NL * g2NL * (1 - corrdNL_gs * corrdNL_gs)')
                norm1 = ne.evaluate('exp(g1 * g1 / (2 - 4 * g1NL)) / sqrt(1 - 2 * g1NL)')
                norm2 = ne.evaluate('exp(g2 * g2 / (2 - 4 * g2NL)) / sqrt(1 - 2 * g2NL)')
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm1 * norm2)')
                nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)')

                # use second order in SFRD lognormal approx
                expGammaCorrMinusLinear = ne.evaluate('nonlinearcorrelation - 1-gammaTimesCorrdNL/((-1+2.*g1NL)*(-1+2.*g2NL))')
            else:
                expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')

            deltaXiTxAddend = ne.evaluate('coeffsTxALL * expGammaCorrMinusLinear')
            deltaXiTxAddend = np.einsum('ijkl->ik', deltaXiTxAddend, optimize = True) #equivalent to np.sum(deltaXiTxAddend, axis = (1, 3))
            deltaXiTxAddend = np.cumsum(deltaXiTxAddend[::-1], axis = 0)[::-1]
            deltaXiTxAddend = np.moveaxis(deltaXiTxAddend, 1, 0)
            deltaXiTxAddend = np.cumsum(deltaXiTxAddend[::-1], axis = 0)[::-1]
            self._II_deltaxi_Tx[:,ir] = np.einsum('ii->i', deltaXiTxAddend, optimize = True)

            deltaXiXaTxAddend = ne.evaluate('coeffsXaTxALL * expGammaCorrMinusLinear')
            deltaXiXaTxAddend = np.einsum('ijkl->ik', deltaXiXaTxAddend, optimize = True) #equivalent to np.sum(deltaXiXaTxAddend, axis = (1, 3))
            deltaXiXaTxAddend = np.moveaxis(deltaXiXaTxAddend, 1, 0)
            deltaXiXaTxAddend = np.cumsum(deltaXiXaTxAddend[::-1], axis = 0)[::-1]
            self._II_deltaxi_xaTx[:,ir] = np.einsum('ii->i', deltaXiXaTxAddend, optimize = True)


        self._II_deltaxi_Tx *= np.array([_coeffTx_units]).T**2
        self._II_deltaxi_xaTx *= np.array([coeffzp1xa * _coeffTx_units]).T


        if (UserParams.FLAG_DO_DENS_NL):
            D_coeffR2Tx = coeffR2Tx.reshape(1, *coeffR2Tx.shape, 1)
            D_coeffzp2Tx = coeffzp2Tx.flatten().reshape(1, *coeffzp2Tx.flatten().shape, 1)
            DDgammaR2 = np.copy(DDgammaR1)
            D_gammaR2 = DDgammaR2.reshape(1, *DDgammaR2.shape , 1)
            D_growthRmatrix = growthRmatrix[:,0].reshape(*growthRmatrix[:,0].shape, 1, 1, 1)
            D_corrdNL = corrdNLBIG.squeeze()[0].reshape(1, 1, *corrdNLBIG.squeeze()[0].shape)
        
            if AstroParams.quadratic_SFRD_lognormal:

                DDsigmaR2 = np.copy(DDsigmaR1)
                D_sigmaR2 = DDsigmaR2.reshape(1, *DDsigmaR2.shape , 1)
                DDgammaR2N = np.copy(DDgammaR1N)
                D_gammaR2N = DDgammaR2N.reshape(1, *DDgammaR2N.shape , 1)
  
                gammaTimesCorrdNL = ne.evaluate('D_gammaR2 * D_growthRmatrix* D_growthRmatrix * D_corrdNL')
                numerator_NL = ne.evaluate('gammaTimesCorrdNL+ D_gammaR2 * D_gammaR2 * D_sigmaR2* D_sigmaR2 /2 + D_gammaR2N * D_growthRmatrix* D_growthRmatrix* D_growthRmatrix* D_growthRmatrix * (D_corrdNL * D_corrdNL)')
                
                denominator_NL = ne.evaluate('1. - 2 * D_gammaR2N*D_sigmaR2*D_sigmaR2')
                
                norm2 = ne.evaluate('exp(D_gammaR2 * D_gammaR2 * D_sigmaR2* D_sigmaR2 * D_gammaR2 * D_gammaR2 * D_sigmaR2* D_sigmaR2 / (2 - 4 * D_gammaR2N*D_sigmaR2*D_sigmaR2)) / sqrt(1 - 2 * D_gammaR2N*D_sigmaR2*D_sigmaR2)')
                
                log_norm = ne.evaluate('log(sqrt(denominator_NL) * norm2)')
                nonlinearcorrelation = ne.evaluate('exp(numerator_NL/denominator_NL - log_norm)')

                self._II_deltaxi_dTx = D_coeffzp2Tx * np.sum(D_coeffR2Tx * (nonlinearcorrelation -1 -D_gammaR2*D_growthRmatrix**2 *D_corrdNL/(1-2.*D_gammaR2N*D_sigmaR2**2)), axis = 2)

            else:
                self._II_deltaxi_dTx =  D_coeffzp2Tx * np.sum(D_coeffR2Tx * ((np.exp(D_gammaR2 * D_growthRmatrix**2 * D_corrdNL)-1.0) - D_gammaR2 * D_growthRmatrix**2 * D_corrdNL), axis = 2)


            self._II_deltaxi_dTx = np.moveaxis(self._II_deltaxi_dTx, 1, 0)
            self._II_deltaxi_dTx = np.cumsum(self._II_deltaxi_dTx[::-1], axis = 0)[::-1]
            self._II_deltaxi_dTx = np.moveaxis(self._II_deltaxi_dTx, 1, 0)
            self._II_deltaxi_dTx = np.einsum('iik->ik', self._II_deltaxi_dTx, optimize = True)
            self._II_deltaxi_dTx *= np.array([_coeffTx_units]).T
            
        return 1

    def get_all_corrs_IIxIII(self, CosmoParams, T21coeffs):
        """
        Computes the Pop IIxIII cross correlation functions across z and R.

        Parameters
        ----------
        CosmoParams : CosmoParams class
        T21coeffs : T21coeffs class

        Returns
        ----------
        Attributes stored in Power_Spectra

        """


        corrdNL = self._corrdNL


        _coeffTx_units_II = T21coeffs.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor
        _coeffTx_units_III = T21coeffs.coeff_Gammah_Tx_III #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(CosmoParams,self._zGreaterMatrix100[:, self._iRnonlinear])
        gammaR1_II = T21coeffs.gamma_II_index2D[:, self._iRnonlinear] * growthRmatrix
        gammaR1_III = T21coeffs.gamma_III_index2D[:, self._iRnonlinear] * growthRmatrix

        coeffzp1xa = T21coeffs.coeff1LyAzp * T21coeffs.coeff_Ja_xa
        coeffzp1Tx = T21coeffs.coeff1Xzp

        coeffR1xa_II = T21coeffs.coeff2LyAzpRR_II[:,self._iRnonlinear]
        coeffR1xa_III = T21coeffs.coeff2LyAzpRR_III[:,self._iRnonlinear]

        coeffR1Tx_II = T21coeffs.coeff2XzpRR_II[:,self._iRnonlinear]
        coeffR1Tx_III = T21coeffs.coeff2XzpRR_III[:,self._iRnonlinear]

        gammamatrix_R1II_R1III = gammaR1_II.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1) * gammaR1_III.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)
        coeffmatrixxa_R1II_R1III = coeffR1xa_II.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1) * coeffR1xa_III.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)

        gammaTimesCorrdNL = ne.evaluate('gammamatrix_R1II_R1III * corrdNL') #np.einsum('ijkl,ijkl->ijkl', gammamatrix_R1II_R1III, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster
        expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')

        self._IIxIII_deltaxi_xa = 2 * np.einsum('ijkl->il', coeffmatrixxa_R1II_R1III * expGammaCorrMinusLinear, optimize = True) #factor of 2 to account for cross-term
        self._IIxIII_deltaxi_xa *= np.array([coeffzp1xa]).T**2 #brings it to xa units

        ###No density cross-term because density by itself doesn't have a Pop II + III contribution; the xa and Tx contribution is already accounted for in the Pop II- and Pop III-only get_all_corrs

        ### To compute Tx quantities, I'm broadcasting arrays such that the axes are zp1, R1, zp2, R2, r

        gammaR2_II = np.copy(gammaR1_II) #already has growth factor in this
        gammaR2_III = np.copy(gammaR1_III) #already has growth factor in this

        gammamatrix_R1II_R2III = gammaR1_II.reshape(*gammaR1_II.shape, 1, 1) * gammaR2_III.reshape(1, 1, *gammaR2_III.shape)
        gammamatrix_R1III_R2II = gammaR1_III.reshape(*gammaR1_III.shape, 1, 1) * gammaR2_II.reshape(1, 1, *gammaR2_II.shape)

        coeffzp1Tx = np.copy(T21coeffs.coeff1Xzp).reshape(*T21coeffs.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21coeffs.coeff1Xzp).reshape(1, 1, *T21coeffs.coeff1Xzp.shape, 1)

        coeffR2Tx_II = np.copy(coeffR1Tx_II)
        coeffR2Tx_III = np.copy(coeffR1Tx_III)

        coeffmatrixTxTx_R1II_R2III = coeffR1Tx_II.reshape(*coeffR1Tx_II.shape, 1, 1) * coeffR2Tx_III.reshape(1, 1, *coeffR2Tx_III.shape)
        coeffmatrixTxTx_R1III_R2II = coeffR1Tx_III.reshape(*coeffR1Tx_III.shape, 1, 1) * coeffR2Tx_II.reshape(1, 1, *coeffR2Tx_II.shape)

        coeffmatrixxaTx_R1II_R2III = coeffR1xa_II.reshape(*coeffR1xa_II.shape, 1, 1) * coeffR2Tx_III.reshape(1, 1, *coeffR2Tx_III.shape)
        coeffmatrixxaTx_R1III_R2II = coeffR1xa_III.reshape(*coeffR1xa_III.shape, 1, 1) * coeffR2Tx_II.reshape(1, 1, *coeffR2Tx_II.shape)

        coeffsTxALL_R1II_R2III = coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx_R1II_R2III
        coeffsTxALL_R1III_R2II = coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx_R1III_R2II
        coeffsXaTxALL_R1II_R2III = coeffzp2Tx * coeffmatrixxaTx_R1II_R2III
        coeffsXaTxALL_R1III_R2II = coeffzp2Tx * coeffmatrixxaTx_R1III_R2II

        self._IIxIII_deltaxi_Tx = np.zeros_like(self._IIxIII_deltaxi_xa)
        _IIxIII_deltaxi_xaTx1 = np.zeros_like(self._IIxIII_deltaxi_xa)
        _IIxIII_deltaxi_xaTx2 = np.zeros_like(self._IIxIII_deltaxi_xa)
        corrdNLBIG = corrdNL[:,:, np.newaxis, :,:] #dimensions zp1, R1, zp2, R2, and r, the last of which will be looped over below

        for ir in range(len(CosmoParams._Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            
            #HAC: Computations using ne.evaluate(...) use numexpr, which speeds up computations of massive numpy arrays

            gamma_R1II_R2III_CorrdNL = ne.evaluate('gammamatrix_R1II_R2III * corrdNL')
            expGamma_R1II_R2III_CorrdNL = ne.evaluate('exp(gamma_R1II_R2III_CorrdNL) - 1 - gamma_R1II_R2III_CorrdNL')

            gamma_R1III_R2II_CorrdNL = ne.evaluate('gammamatrix_R1III_R2II * corrdNL')
            expGamma_R1III_R2II_CorrdNL = ne.evaluate('exp(gammamatrix_R1III_R2II * corrdNL) - 1 - gammamatrix_R1III_R2II * corrdNL')

            deltaXiTxAddend = ne.evaluate('coeffsTxALL_R1II_R2III * expGamma_R1II_R2III_CorrdNL + coeffsTxALL_R1III_R2II * expGamma_R1III_R2II_CorrdNL')
            deltaXiTxAddend = np.einsum('ijkl->ik', deltaXiTxAddend, optimize = True)# equivalent to np.sum(deltaXiTxAddend, axis = (1, 3))
            deltaXiTxAddend = np.cumsum(deltaXiTxAddend[::-1], axis = 0)[::-1]
            deltaXiTxAddend = np.moveaxis(deltaXiTxAddend, 1, 0)
            deltaXiTxAddend = np.cumsum(deltaXiTxAddend[::-1], axis = 0)[::-1]
            self._IIxIII_deltaxi_Tx[:,ir] = np.einsum('ii->i', deltaXiTxAddend, optimize = True)

            #Tx in R2 uses Pop III quantities
            deltaXiXaTxAddend1 = ne.evaluate('coeffsXaTxALL_R1II_R2III * expGamma_R1II_R2III_CorrdNL')
            deltaXiXaTxAddend1 = np.einsum('ijkl->ik', deltaXiXaTxAddend1, optimize = True) # equivalent to np.sum(deltaXiXaTxAddend, axis = (1, 3))
            deltaXiXaTxAddend1 = np.moveaxis(deltaXiXaTxAddend1, 1, 0)
            deltaXiXaTxAddend1 = np.cumsum(deltaXiXaTxAddend1[::-1], axis = 0)[::-1]
            _IIxIII_deltaxi_xaTx1[:, ir] = np.einsum('ii->i', deltaXiXaTxAddend1, optimize = True)
            
            #Tx in R2 uses Pop II quantities
            deltaXiXaTxAddend2 = ne.evaluate('coeffsXaTxALL_R1III_R2II * expGamma_R1III_R2II_CorrdNL')
            deltaXiXaTxAddend2 = np.einsum('ijkl->ik', deltaXiXaTxAddend2, optimize = True) # equivalent to np.sum(deltaXiXaTxAddend, axis = (1, 3))
            deltaXiXaTxAddend2 = np.moveaxis(deltaXiXaTxAddend2, 1, 0)
            deltaXiXaTxAddend2 = np.cumsum(deltaXiXaTxAddend2[::-1], axis = 0)[::-1]
            _IIxIII_deltaxi_xaTx2[:, ir] = np.einsum('ii->i', deltaXiXaTxAddend2, optimize = True)

        self._IIxIII_deltaxi_Tx *= np.array([_coeffTx_units_II * _coeffTx_units_III]).T
        
        _IIxIII_deltaxi_xaTx1 *=  np.array([coeffzp1xa * _coeffTx_units_III]).T
        _IIxIII_deltaxi_xaTx2 *=  np.array([coeffzp1xa * _coeffTx_units_II]).T
        self._IIxIII_deltaxi_xaTx =  _IIxIII_deltaxi_xaTx1 + _IIxIII_deltaxi_xaTx2
        
        return 1
        
        
    def get_xi_Sum_2ExpEta(self, xiEta, etaCoeff1, etaCoeff2):
        """
        Computes the correlation function of the VCB portion of the SFRD, expressed using sums of two exponentials
        if rho(z1, x1) / rhobar = Ae^-b tilde(eta) + Ce^-d tilde(eta) and rho(z2, x2) / rhobar = Fe^-g tilde(eta) + He^-k tilde(eta)
        Then this computes <rho(z1, x1) * rho(z2, x2)> - <rho(z1, x1)> <rho(z2, x2)>
        Refer to eq. A12 in 2407.18294 for more details
        
        Parameters
        ----------
        xiEta: matrix
            Matrix of Eta correlation function. Dimension (corrEtaNL)
        etaCoeff1: matrix
            Stored Eta parameters in T21coeffs.vcb_expFitParams. Dimension (vcbCoeffsR1)
        etaCoeff2: matrix
            Stored Eta parameters in T21coeffs.vcb_expFitParams. Dimension (vcbCoeffsR2)
        
        Returns
        ----------
        xiTotal: matrix
            Total <rho(z1, x1) * rho(z2, x2)> - <rho(z1, x1)> <rho(z2, x2)> power spectra
        """
        
        aa, bb, cc, dd = etaCoeff1
        ff, gg, hh, kk = etaCoeff2
        
        normBB = ne.evaluate('(1+2*bb)**(3/2)')
        normGG = ne.evaluate('(1+2*gg)**(3/2)')
        normDD = ne.evaluate('(1+2*dd)**(3/2)')
        normKK = ne.evaluate('(1+2*kk)**(3/2)')
        
        afBG = ne.evaluate('aa * ff / normBB / normGG')
        ahBK = ne.evaluate('aa * hh / normBB / normKK')
        cfDG = ne.evaluate('cc * ff / normDD / normGG')
        chDK = ne.evaluate('cc * hh / normDD / normKK')
        
        #The below involves horribly long writing, but breaking this into pieces makes for slightly longer computation time
        xiNumerator  = ne.evaluate('afBG * (1 / (1 - 6*bb * gg * xiEta / ((1+2*bb)*(1+2*gg)))**(3/2) - 1) + ahBK * (1 / (1 - 6*bb * kk * xiEta / ((1+2*bb)*(1+2*kk)))**(3/2) - 1) + cfDG * (1 / (1 - 6*dd * gg * xiEta / ((1+2*dd)*(1+2*gg)))**(3/2) - 1) + chDK * (1 / (1 - 6*dd * kk * xiEta / ((1+2*dd)*(1+2*kk)))**(3/2) - 1)')
        xiDenominator  = ne.evaluate('afBG + ahBK + cfDG + chDK')
        
        xiTotal = ne.evaluate('xiNumerator / xiDenominator')
        
        return xiTotal


    def get_all_corrs_III(self, UserParams, CosmoParams, T21coeffs):
        """
        Computes the Pop III correlation functions across z and R.

        Parameters
        ----------
        UserParams : UserParams class
        CosmoParams : CosmoParams class
        T21coeffs : T21coeffs class

        Returns
        ----------
        Attributes stored in Power_Spectra

        """


        corrdNL = self._corrdNL


        corrEtaNL = CosmoParams.xiEta_RR_CF[np.ix_(self._iRnonlinear,self._iRnonlinear)]
        corrEtaNL[0:CosmoParams.indexminNL,0:CosmoParams.indexminNL] = corrEtaNL[CosmoParams.indexminNL,CosmoParams.indexminNL]
        corrEtaNL = corrEtaNL.reshape(1, *corrEtaNL.shape)


        _coeffTx_units = T21coeffs.coeff_Gammah_Tx_III #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(CosmoParams,self._zGreaterMatrix100[:, self._iRnonlinear])
        gammaR1 = T21coeffs.gamma_III_index2D[:, self._iRnonlinear] * growthRmatrix
        
        vcbCoeffs1 = T21coeffs.vcb_expFitParams[:, self._iRnonlinear]
        vcbCoeffsR1 = np.transpose(vcbCoeffs1, (2, 0, 1))
        vcbCoeffsR1 = vcbCoeffsR1[:,:,:,np.newaxis,np.newaxis]
        vcbCoeffsR2 = np.moveaxis(vcbCoeffsR1, 3, 2)

        coeffzp1xa = T21coeffs.coeff1LyAzp * T21coeffs.coeff_Ja_xa
        coeffzp1Tx = T21coeffs.coeff1Xzp

        coeffR1xa = T21coeffs.coeff2LyAzpRR_III[:,self._iRnonlinear]
        coeffR1Tx = T21coeffs.coeff2XzpRR_III[:,self._iRnonlinear]

        gammamatrixR1R1 = gammaR1.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1) * gammaR1.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)
        coeffmatrixxa = coeffR1xa.reshape(len(T21coeffs.zintegral), 1, len(self._iRnonlinear),1) * coeffR1xa.reshape(len(T21coeffs.zintegral), len(self._iRnonlinear), 1,1)

        gammaCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL') #np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster
        expGammaCorr = ne.evaluate('exp(gammaCorrdNL) - 1') # equivalent to np.exp(gammaTimesCorrdNL)-1.0

        if CosmoParams.USE_RELATIVE_VELOCITIES == True:
            etaCorr_xa = self.get_xi_Sum_2ExpEta(corrEtaNL, vcbCoeffsR1, vcbCoeffsR2)
            totalCorr = ne.evaluate('expGammaCorr * etaCorr_xa + expGammaCorr + etaCorr_xa - gammaCorrdNL') ###TO DO (linearized VCB flucts): - etaCorr_xa_lin #note that the Taylor expansion of the cross-term is 0 to linear order
        else:
            totalCorr = ne.evaluate('expGammaCorr - gammaCorrdNL') ###TO DO (linearized VCB flucts): - etaCorr_xa_lin #note that the Taylor expansion of the cross-term is 0 to linear order

        self._III_deltaxi_xa = np.einsum('ijkl->il', coeffmatrixxa * totalCorr , optimize = True)  # equivalent to self._III_deltaxi_xa = np.sum(coeffmatrixxa * ((np.exp(gammaTimesCorrdNL)-1.0) - gammaTimesCorrdNL), axis = (1,2))
        self._III_deltaxi_xa *= np.array([coeffzp1xa]).T**2 #brings it to xa units

        if (UserParams.FLAG_DO_DENS_NL): #no velocity contribution to density
            D_coeffR1xa = coeffR1xa.reshape(*coeffR1xa.shape, 1)
            D_gammaR1 = gammaR1.reshape(*gammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            self._III_deltaxi_dxa = np.sum(D_coeffR1xa * ((np.exp(D_gammaR1 * D_growthRmatrix * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)
            self._III_deltaxi_dxa *= np.array([coeffzp1xa]).T

        ### To compute Tx quantities, I'm broadcasting arrays such that the axes are zp1, R1, zp2, R2, r

        gammaR2 = np.copy(gammaR1) #already has growth factor in this
        gammamatrixR1R2 = gammaR1.reshape(*gammaR1.shape, 1, 1) * gammaR2.reshape(1, 1, *gammaR2.shape)

        coeffzp1Tx = np.copy(T21coeffs.coeff1Xzp).reshape(*T21coeffs.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21coeffs.coeff1Xzp).reshape(1, 1, *T21coeffs.coeff1Xzp.shape, 1)
        coeffR2Tx = np.copy(coeffR1Tx)
        coeffmatrixTxTx = coeffR1Tx.reshape(*coeffR1Tx.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffmatrixxaTx = coeffR1xa.reshape(*coeffR1xa.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffsTxALL = coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx
        coeffsXaTxALL = coeffzp2Tx * coeffmatrixxaTx

        corrdNLBIG = corrdNL[:,:, np.newaxis, :, :]
        corrEtaNLBIG = corrEtaNL[:,:, np.newaxis, :, :]

        vcbCoeffsR1 = vcbCoeffsR1[:,:,:,:,:]
        vcbCoeffsR2 = np.transpose(vcbCoeffsR1, (0,3,4,1,2))

        self._III_deltaxi_Tx = np.zeros_like(self._III_deltaxi_xa)
        self._III_deltaxi_xaTx = np.zeros_like(self._III_deltaxi_xa)
        self._III_deltaxi_dTx = np.zeros_like(self._III_deltaxi_xa)

        for ir in range(len(CosmoParams._Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            corrEtaNL = corrEtaNLBIG[:,:,:,:,ir]

            gammaCorrdNL = ne.evaluate('gammamatrixR1R2 * corrdNL')
            expGammaCorrdNL = ne.evaluate('exp(gammaCorrdNL) - 1')
            
            if CosmoParams.USE_RELATIVE_VELOCITIES == True:
                etaCorr_Tx = self.get_xi_Sum_2ExpEta(corrEtaNL, vcbCoeffsR1, vcbCoeffsR2)
                totalCorr = ne.evaluate('expGammaCorrdNL * etaCorr_Tx + expGammaCorrdNL + etaCorr_Tx - gammaCorrdNL') ###TO DO (linearized VCB flucts): - etaCorr_xa_lin #note that the Taylor expansion of the cross-term is 0 to linear order
            else:
                totalCorr = ne.evaluate('expGammaCorrdNL - gammaCorrdNL') ###TO DO (linearized VCB flucts): - etaCorr_xa_lin #note that the Taylor expansion of the cross-term is 0 to linear order

            deltaXiTxAddend = ne.evaluate('coeffsTxALL * totalCorr') # equivalent to np.multiply(coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx, totalCorr, out = outDummy)
            deltaXiTxAddend = np.einsum('ijkl->ik', deltaXiTxAddend, optimize=True) # equivalent to np.sum(deltaXiTxAddend, axis = (1, 3))
            deltaXiTxAddend = np.cumsum(deltaXiTxAddend[::-1], axis = 0)[::-1]
            deltaXiTxAddend = np.moveaxis(deltaXiTxAddend, 1, 0)
            deltaXiTxAddend = np.cumsum(deltaXiTxAddend[::-1], axis = 0)[::-1]
            self._III_deltaxi_Tx[:, ir] = np.einsum('ii->i', deltaXiTxAddend, optimize = True)

            deltaXiXaTxAddend = ne.evaluate('coeffsXaTxALL * totalCorr') # equivalent to np.multiply(coeffzp2Tx * coeffmatrixxaTx, totalCorr, out = outDummy)
            deltaXiXaTxAddend = np.einsum('ijkl->ik', deltaXiXaTxAddend, optimize=True) # equivalent to np.sum(deltaXiXaTxAddend, axis = (1, 3))
            deltaXiXaTxAddend = np.moveaxis(deltaXiXaTxAddend, 1, 0)
            deltaXiXaTxAddend = np.cumsum(deltaXiXaTxAddend[::-1], axis = 0)[::-1]
            self._III_deltaxi_xaTx[:, ir] = np.einsum('ii->i', deltaXiXaTxAddend, optimize = True)

        if (UserParams.FLAG_DO_DENS_NL): #no velocity contribution to density
            D_coeffR2Tx = coeffR2Tx.reshape(1, *coeffR2Tx.shape, 1)
            D_coeffzp2Tx = coeffzp2Tx.flatten().reshape(1, *coeffzp2Tx.flatten().shape, 1)
            D_gammaR2 = gammaR2.reshape(1, *gammaR2.shape , 1)
            D_growthRmatrix = growthRmatrix[:,0].reshape(*growthRmatrix[:,0].shape, 1, 1, 1)
            D_corrdNL = corrdNLBIG.squeeze()[0].reshape(1, 1, *corrdNLBIG.squeeze()[0].shape)

            self._III_deltaxi_dTx =  D_coeffzp2Tx * np.sum(D_coeffR2Tx * ((np.exp(D_gammaR2 * D_growthRmatrix * D_corrdNL)-1.0) - D_gammaR2 * D_growthRmatrix * D_corrdNL), axis = 2)

            self._III_deltaxi_dTx = np.moveaxis(self._III_deltaxi_dTx, 1, 0)
            self._III_deltaxi_dTx = np.cumsum(self._III_deltaxi_dTx[::-1], axis = 0)[::-1]
            self._III_deltaxi_dTx = np.moveaxis(self._III_deltaxi_dTx, 1, 0)
            self._III_deltaxi_dTx = np.einsum('iik->ik', self._III_deltaxi_dTx, optimize = True)
            self._III_deltaxi_dTx *= np.array([_coeffTx_units]).T

        self._III_deltaxi_Tx *= np.array([_coeffTx_units]).T**2
        self._III_deltaxi_xaTx *= np.array([coeffzp1xa * _coeffTx_units]).T

        return 1
        
        
    def get_list_PS(self, xi_list, zlisttoconvert):
        """
        Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input

        Parameters
        ----------
        xi_list : matrix
            list of correlation functions
        zlisttoconvert: array
            which redshifts xi_list is evaluated at

        Returns
        ----------
        _Pk_list: matrix
            Matrix of power spectra. Dimension (z, K)

        """
        _Pk_list = []

        for izp,zp in enumerate(zlisttoconvert):

            _kzp, _Pkzp = self.get_Pk_from_xi(self._rs_input_mcfit,xi_list[izp])
            _Pk_list.append(_Pkzp)
            #can ignore _kzp, it's the same as klist_PS above by construction


        return np.array(_Pk_list)


    def get_Pk_from_xi(self, rsinput, xiinput):
        """
        Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF

        Parameters
        ----------
        rsinput : array
            Array of Rs used to evaluate xiinput
        xiinput: matrix
            Matrix of values you are Fourier Transforming. Dimension (z, R)

        Returns
        ----------
        kPf: list
            List of wavenumbers
        Pf: matrix
            Resultant Fourier Transform of xiinput. Dimension (z, k)

        """
        
        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf
