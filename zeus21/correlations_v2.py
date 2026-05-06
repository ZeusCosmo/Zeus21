"""

Code to compute correlation functions from power spectra and functions of them. Holds two classes: Correlations (with matter correlation functions smoothed over different R), and Power_Spectra (which will compute and hold the 21-cm power spectrum and power for derived quantities like xa, Tk, etc.)

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Sarah Libanore
BGU - July 2025

"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import mcfit
from scipy.special import gammaincc #actually very fast, no need to approximate
import numexpr as ne

from . import constants
from . import cosmology
from . import z21_utilities



class Power_Spectra:
    "Get power spetrum from correlation functions and coefficients"

    def __init__(self, User_Parameters, Cosmo_Parameters, Astro_Parameters, T21_coefficients, RSD_MODE=1):

#        print("STEP 0: Variable Setup")
        #set up some variables
        self._rs_input_mcfit = Cosmo_Parameters.rlist_CF #just to make notation simpler
        self.klist_PS = Cosmo_Parameters._klistCF
        self.RSD_MODE = RSD_MODE #redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space #TODO: mode to save at different mu

        #first get the linear window functions -- note it already has growth factor in it, so it multiplies Pmatter(z=0)
        #fix some arrays: TYTYTY HERE

        self._zGreaterMatrix100, self._iRnonlinear, self._corrdNL =  self._prepare_corr_arrays(Cosmo_Parameters, T21_coefficients)

        self.kwindow, self.windowalpha_II = self.get_xa_window(Astro_Parameters, Cosmo_Parameters, T21_coefficients, pop = 2)
        self._kwindowX, self.windowxray_II = self.get_Tx_window(Astro_Parameters, Cosmo_Parameters, T21_coefficients, pop = 2)
        

        if Astro_Parameters.USE_POPIII == True:
        # SarahLibanore: add AstroParams to use flag on quadratic order
            self.kwindow, self.windowalpha_III = self.get_xa_window(Astro_Parameters, Cosmo_Parameters, T21_coefficients, pop = 3)
        # SarahLibanore: add AstroParams to use flag on quadratic order
            self._kwindowX, self.windowxray_III = self.get_Tx_window(Astro_Parameters, Cosmo_Parameters, T21_coefficients, pop = 3)
        else:
            self.windowalpha_III = np.zeros_like(self.windowalpha_II)
            self.windowxray_III = np.zeros_like(self.windowxray_II)
            
        #calculate some growth etc, and the bubble biases for the xHI linear window function:
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, T21_coefficients.zintegral)



        ##############################

#        print("STEP 1: Computing Nonlinear Power Spectra")
        #finally, get all the nonlinear correlation functions:
#        print("Computing Pop II-dependent power spectra")
        # SarahLibanore: add AstroParams to use flag on quadratic order
        self.get_all_corrs_II(Astro_Parameters, User_Parameters, Cosmo_Parameters, T21_coefficients)
        
        if Astro_Parameters.USE_POPIII == True:
#            print("Computing Pop IIxIII-dependent cross power spectra")
            self.get_all_corrs_IIxIII(Cosmo_Parameters, T21_coefficients)
            
#            print("Computing Pop III-dependent power spectra")
            self.get_all_corrs_III(User_Parameters, Cosmo_Parameters, T21_coefficients)
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
        self._Pk_xa_lin_II = self.windowalpha_II**2 * Cosmo_Parameters._PklinCF
        self._Pk_xa_lin_III = self.windowalpha_III**2 * Cosmo_Parameters._PklinCF ###TO DO (linearized VCB flucts):+ self.windowalphaVel_III**2 * Cosmo_Parameters._PkEtaCF
        self._Pk_xa_lin_IIxIII = 2* self.windowalpha_II * self.windowalpha_III * Cosmo_Parameters._PklinCF #Pop IIxIII cross term doesn't have a velocity component

        self.Deltasq_xa_lin_II = self._Pk_xa_lin_II * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_lin_III = self._Pk_xa_lin_III * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_lin_IIxIII = self._Pk_xa_lin_IIxIII * self._k3over2pi2 #note that it still has units of xa_avg

        #nonlinear corrections too:
        self._d_Pk_xa_nl_II = self.get_list_PS(self._II_deltaxi_xa, T21_coefficients.zintegral)
        self._d_Pk_xa_nl_III = self.get_list_PS(self._III_deltaxi_xa, T21_coefficients.zintegral) #velocity correlations already embedded in nonlinear computation
        self._d_Pk_xa_nl_IIxIII = self.get_list_PS(self._IIxIII_deltaxi_xa, T21_coefficients.zintegral)

        self.Deltasq_xa_II = self.Deltasq_xa_lin_II + self._d_Pk_xa_nl_II * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_III = self.Deltasq_xa_lin_III + self._d_Pk_xa_nl_III * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xa_IIxIII = self.Deltasq_xa_lin_IIxIII + self._d_Pk_xa_nl_IIxIII * self._k3over2pi2 #note that it still has units of xa_avg


        ##############################


        #and same for xray
        self._Pk_Tx_lin_II = self.windowxray_II**2 * Cosmo_Parameters._PklinCF
        self._Pk_Tx_lin_III = self.windowxray_III**2 * Cosmo_Parameters._PklinCF ###TO DO (linearized VCB flucts):+ self.windowxrayVel_III**2 * Cosmo_Parameters._PkEtaCF
        self._Pk_Tx_lin_IIxIII = 2* self.windowxray_II * self.windowxray_III * Cosmo_Parameters._PklinCF #Pop IIxIII cross term doesn't have a velocity component

        self.Deltasq_Tx_lin_II = self._Pk_Tx_lin_II * self._k3over2pi2
        self.Deltasq_Tx_lin_III = self._Pk_Tx_lin_III * self._k3over2pi2
        self.Deltasq_Tx_lin_IIxIII = self._Pk_Tx_lin_IIxIII * self._k3over2pi2

        self._d_Pk_Tx_nl_II = self.get_list_PS(self._II_deltaxi_Tx, T21_coefficients.zintegral)
        self._d_Pk_Tx_nl_III = self.get_list_PS(self._III_deltaxi_Tx, T21_coefficients.zintegral)
        self._d_Pk_Tx_nl_IIxIII = self.get_list_PS(self._IIxIII_deltaxi_Tx, T21_coefficients.zintegral)

        self.Deltasq_Tx_II = self.Deltasq_Tx_lin_II + self._d_Pk_Tx_nl_II * self._k3over2pi2
        self.Deltasq_Tx_III = self.Deltasq_Tx_lin_III + self._d_Pk_Tx_nl_III * self._k3over2pi2
        self.Deltasq_Tx_IIxIII = self.Deltasq_Tx_lin_IIxIII + self._d_Pk_Tx_nl_IIxIII * self._k3over2pi2


        ##############################


        #and their cross correlation
        self._Pk_xaTx_lin_II = self.windowalpha_II * self.windowxray_II * Cosmo_Parameters._PklinCF
        self._Pk_xaTx_lin_III = self.windowalpha_III * self.windowxray_III * Cosmo_Parameters._PklinCF ###TO DO (linearized VCB flucts):+ self.windowalphaVel_III * self.windowxrayVel_III * Cosmo_Parameters._PkEtaCF
        self._Pk_xaTx_lin_IIxIII = (self.windowalpha_II * self.windowxray_III + self.windowalpha_III * self.windowxray_II) * Cosmo_Parameters._PklinCF

        self.Deltasq_xaTx_lin_II = self._Pk_xaTx_lin_II * self._k3over2pi2
        self.Deltasq_xaTx_lin_III = self._Pk_xaTx_lin_III * self._k3over2pi2
        self.Deltasq_xaTx_lin_IIxIII = self._Pk_xaTx_lin_IIxIII * self._k3over2pi2

        self._d_Pk_xaTx_nl_II = self.get_list_PS(self._II_deltaxi_xaTx, T21_coefficients.zintegral)
        self._d_Pk_xaTx_nl_III = self.get_list_PS(self._III_deltaxi_xaTx, T21_coefficients.zintegral)
        self._d_Pk_xaTx_nl_IIxIII = self.get_list_PS(self._IIxIII_deltaxi_xaTx, T21_coefficients.zintegral)

        self.Deltasq_xaTx_II = self.Deltasq_xaTx_lin_II + self._d_Pk_xaTx_nl_II * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xaTx_III = self.Deltasq_xaTx_lin_III + self._d_Pk_xaTx_nl_III * self._k3over2pi2 #note that it still has units of xa_avg
        self.Deltasq_xaTx_IIxIII = self.Deltasq_xaTx_lin_IIxIII + self._d_Pk_xaTx_nl_IIxIII * self._k3over2pi2 #note that it still has units of xa_avg


        ##############################
        
        
        #and the same for deltaNL and its cross terms:
        self._Pk_d_lin = np.outer(self._lingrowthd**2, Cosmo_Parameters._PklinCF) #No Pop II or III contribution
        self.Deltasq_d_lin = self._Pk_d_lin * self._k3over2pi2 #note that it still has units of xa_avg

        self._Pk_dxa_lin_II = (self.windowalpha_II.T * self._lingrowthd).T * Cosmo_Parameters._PklinCF
        self._Pk_dxa_lin_III = (self.windowalpha_III.T * self._lingrowthd).T * Cosmo_Parameters._PklinCF #No velocity component

        self._Pk_dTx_lin_II = (self.windowxray_II.T * self._lingrowthd).T * Cosmo_Parameters._PklinCF
        self._Pk_dTx_lin_III = (self.windowxray_III.T * self._lingrowthd).T * Cosmo_Parameters._PklinCF #No velocity component

        self.Deltasq_dxa_lin_II = self._Pk_dxa_lin_II * self._k3over2pi2
        self.Deltasq_dxa_lin_III = self._Pk_dxa_lin_III * self._k3over2pi2 #No velocity component

        self.Deltasq_dTx_lin_II = self._Pk_dTx_lin_II * self._k3over2pi2
        self.Deltasq_dTx_lin_III = self._Pk_dTx_lin_III * self._k3over2pi2 #No velocity component

        self._Pk_d =  self._Pk_d_lin

        self._Pk_dxa_II =  self._Pk_dxa_lin_II
        self._Pk_dxa_III =  self._Pk_dxa_lin_III

        self._Pk_dTx_II =  self._Pk_dTx_lin_II
        self._Pk_dTx_III =  self._Pk_dTx_lin_III

        if(User_Parameters.FLAG_DO_DENS_NL): #note that the nonlinear terms (cross and auto) below here have the growth already accounted for

            self._d_Pk_d_nl = self.get_list_PS(self._II_deltaxi_d, T21_coefficients.zintegral)
            self._Pk_d += self._d_Pk_d_nl

            self._d_Pk_dxa_nl_II = self.get_list_PS(self._II_deltaxi_dxa, T21_coefficients.zintegral)
            self._d_Pk_dxa_nl_III = self.get_list_PS(self._III_deltaxi_dxa, T21_coefficients.zintegral)
            self._Pk_dxa_II += self._d_Pk_dxa_nl_II
            self._Pk_dxa_III += self._d_Pk_dxa_nl_III

            self._d_Pk_dTx_nl_II = self.get_list_PS(self._II_deltaxi_dTx, T21_coefficients.zintegral)
            self._d_Pk_dTx_nl_III = self.get_list_PS(self._III_deltaxi_dTx, T21_coefficients.zintegral)

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
            self._Pk_xion_lin = self.windowxion**2 * Cosmo_Parameters._PklinCF
            self.Deltasq_xion_lin = self._Pk_xion_lin * self._k3over2pi2

            self._d_Pk_xion_nl = self.get_list_PS(self._deltaxi_xi, T21_coefficients.zintegral)
            self.Deltasq_xion = self.Deltasq_xion_lin + self._d_Pk_xion_nl * self._k3over2pi2

            #cross with density
            self._Pk_dxion_lin = (self.windowxion.T * self._lingrowthd).T  * Cosmo_Parameters._PklinCF
            self.Deltasq_dxion_lin = self._Pk_dxion_lin * self._k3over2pi2

            self._d_Pk_dxion_nl = self.get_list_PS(self._deltaxi_dxi, T21_coefficients.zintegral)
            self.Deltasq_dxion = self.Deltasq_dxion_lin + self._d_Pk_dxion_nl * self._k3over2pi2

            #cross with xa
            self._Pk_xaxion_lin = self.windowxion * self.windowalpha  * Cosmo_Parameters._PklinCF
            self.Deltasq_xaxion_lin = self._Pk_xaxion_lin * self._k3over2pi2

            self._d_Pk_xaxion_nl = self.get_list_PS(self._deltaxi_xaxi, T21_coefficients.zintegral)
            self.Deltasq_xaxion = self.Deltasq_xaxion_lin + self._d_Pk_xaxion_nl * self._k3over2pi2

            #and cross with Tx
            self._Pk_Txxion_lin = self.windowxion * self.windowxray  * Cosmo_Parameters._PklinCF
            self.Deltasq_Txxion_lin = self._Pk_Txxion_lin * self._k3over2pi2

            self._d_Pk_Txxion_nl = self.get_list_PS(self._deltaxi_Txxi, T21_coefficients.zintegral)
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

#        print('STEP 2: Computing 21-cm Power Spectrum')
        #and get the PS of T21 too.
        self._betaT = T21_coefficients.T_CMB/T21_coefficients.Tk_avg /(T21_coefficients.invTcol_avg**-1 - T21_coefficients.T_CMB) #multiplies \delta T_x and \delta T_ad [both dimensionful, not \deltaT/T]
        self._betaxa = 1./(1. + T21_coefficients.xa_avg)/T21_coefficients.xa_avg #multiplies \delta x_a [again not \delta xa/xa]

        #calculate beta_adiabatic
        self._dlingrowthd_dz = cosmology.dgrowth_dz(Cosmo_Parameters, T21_coefficients.zintegral)

        _factor_adi_ = (1+T21_coefficients.zintegral)**2
        _integrand_adi = T21_coefficients.Tk_avg*self._dlingrowthd_dz/_factor_adi_ * T21_coefficients.dlogzint*T21_coefficients.zintegral

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            _hizintegral = 0.0 #they do not account for the adiabatic history prior to starting their evolution. It misses ~half of the adiabatic flucts.
        else:
            #the z>zmax part of the integral we do aside. Assume Tk=Tadiabatic from CLASS.
            _zlisthighz_ = np.linspace(T21_coefficients.zintegral[-1], 99., 100) #beyond z=100 need to explictly tell CLASS to save growth
            _dgrowthhighz_ = cosmology.dgrowth_dz(Cosmo_Parameters, _zlisthighz_)
            _hizintegral = np.trapezoid(cosmology.Tadiabatic(Cosmo_Parameters,_zlisthighz_)
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
            self._betaxion = - 1.0/T21_coefficients.xHI_avg * np.heaviside(constants.ZMAX_Bubbles - T21_coefficients.zintegral, 0.5) # xion = 1 - xHI, only for z<ZMAX_Bubbles. 1/xHI_avg since P_xHI has units of xHI
        else:
            self._betaxion = np.zeros_like(T21_coefficients.xHI_avg) # do not do EoR bubbles at all


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
        self.Deltasq_T21 = (self.Deltasq_T21.T*T21_coefficients.T21avg**2).T
        
        self.Deltasq_dT21 = (np.einsum('ik...,ikl...->kl...',self._allbetas,self._allcorrs[0]).T*T21_coefficients.T21avg).T


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
        self.Deltasq_T21_lin = (self.Deltasq_T21_lin.T*T21_coefficients.T21avg**2).T

        self.Deltasq_dT21_lin = (np.einsum('ik...,ikl...->kl...',self._allbetas,self._allcorrs_lin[0]).T*T21_coefficients.T21avg).T
#        print("Power Spectral Routine Done!")



    def _prepare_corr_arrays(self, Cosmo_Parameters, T21_coefficients):
        zGM = np.copy(T21_coefficients.zGreaterMatrix)
        zGM[np.isnan(zGM)] = 100
        iR = np.arange(Cosmo_Parameters.indexmaxNL)
        corr = Cosmo_Parameters.xi_RR_CF[np.ix_(iR, iR)]
        corr[:Cosmo_Parameters.indexminNL, :Cosmo_Parameters.indexminNL] = \
            corr[Cosmo_Parameters.indexminNL, Cosmo_Parameters.indexminNL]
        return zGM, iR, corr.reshape((1, *corr.shape))

    # SarahLibanore: add AstroParams to use flag on quadratic order
    def get_xa_window(self, Astro_Parameters, Cosmo_Parameters, T21_coefficients, pop = 0): #set pop to 2 or 3, default zero just so python doesn't complain
        "Returns the xa window function for all z in zintegral"

        coeffzp = T21_coefficients.coeff1LyAzp
        coeffJaxa = T21_coefficients.coeff_Ja_xa

        growthRmatrix = cosmology.growth(Cosmo_Parameters, self._zGreaterMatrix100)

        if pop == 2:
            coeffRmatrix = T21_coefficients.coeff2LyAzpRR_II
            gammaRmatrix = T21_coefficients.gamma_II_index2D * growthRmatrix
        elif pop == 3:
            coeffRmatrix = T21_coefficients.coeff2LyAzpRR_III
            gammaRmatrix = T21_coefficients.gamma_III_index2D * growthRmatrix
        else:
            print("Must set pop to either 2 or 3!")

        _wincoeffsMatrix = coeffRmatrix * gammaRmatrix
        # SarahLibanore: quadratic order in the lognormal
        if Astro_Parameters.quadratic_SFRD_lognormal:
           _wincoeffsMatrix *= 1./(1-2.*T21_coefficients.gamma2_II_index2D*T21_coefficients.sigmaofRtab**2)

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
            _wincoeffsMatrix /=(4*np.pi * Cosmo_Parameters._Rtabsmoo**2) * (Cosmo_Parameters._Rtabsmoo * Cosmo_Parameters._dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2 * Delta r)
            _kwinalpha, _win_alpha = self.get_Pk_from_xi(Cosmo_Parameters._Rtabsmoo, _wincoeffsMatrix)

        else:
            _kwinalpha = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RtabsmooMesh, kWinAlphaMesh = np.meshgrid(T21_coefficients.zintegral, Cosmo_Parameters._Rtabsmoo, _kwinalpha, indexing = 'ij', sparse = True)

            _win_alpha = coeffRgammaRmatrix * z21_utilities._WinTH(RtabsmooMesh, kWinAlphaMesh)
            _win_alpha = np.sum(_win_alpha, axis = 1)

        _win_alpha *= np.array([coeffzp*coeffJaxa]).T
        
        return _kwinalpha, _win_alpha


    # SarahLibanore: add AstroParams to use flag on quadratic order
    def get_Tx_window(self, Astro_Parameters, Cosmo_Parameters, T21_coefficients, pop = 0): #set pop to 2 or 3, default zero just so python doesn't complain
        "Returns the Tx window function for all z in zintegral"

        coeffzp = np.array([T21_coefficients.coeff1Xzp]).T
        growthRmatrix = cosmology.growth(Cosmo_Parameters, self._zGreaterMatrix100)

        if pop == 2:
            coeffRmatrix = T21_coefficients.coeff2XzpRR_II
            gammaRmatrix = T21_coefficients.gamma_II_index2D * growthRmatrix
            _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_II#z-dependent, includes 10^40 erg/s/SFR normalizaiton and erg/K conversion factor, and the 1/(1+z)^2 factor to compensate the adiabatic cooling of the Tx olny part
        elif pop == 3:
            coeffRmatrix = T21_coefficients.coeff2XzpRR_III
            gammaRmatrix = T21_coefficients.gamma_III_index2D * growthRmatrix
            _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_III
        else:
            print("Must set pop to either 2 or 3!")

        # SarahLibanore: quadratic order in the lognormal
        if Astro_Parameters.quadratic_SFRD_lognormal:
           gammaRmatrix *= (1/(1-2.*T21_coefficients.gamma2_II_index2D*T21_coefficients.sigmaofRtab**2))

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
            _wincoeffs = coeffRmatrix * gammaRmatrix #array in logR space
            _wincoeffs /=(4*np.pi * Cosmo_Parameters._Rtabsmoo**2) * (Cosmo_Parameters._Rtabsmoo * Cosmo_Parameters._dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2) * Delta r
            _kwinTx, _win_Tx_curr = self.get_Pk_from_xi(Cosmo_Parameters._Rtabsmoo, _wincoeffs)

        else:
            _kwinTx = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RtabsmooMesh, kWinTxMesh = np.meshgrid(T21_coefficients.zintegral, Cosmo_Parameters._Rtabsmoo, _kwinTx, indexing = 'ij', sparse = True)

            _win_Tx_curr = coeffRgammaRmatrix * z21_utilities._WinTH(RtabsmooMesh, kWinTxMesh)
            _win_Tx_curr = np.sum(_win_Tx_curr , axis = 1)

        _win_Tx = _win_Tx_curr * coeffzp
        _win_Tx = np.cumsum(_win_Tx[::-1], axis = 0)[::-1]

        _win_Tx =_win_Tx * np.array([_coeffTx_units]).T

        return _kwinTx, _win_Tx


    # SarahLibanore: function modified to include quadratic order
    def get_all_corrs_II(self, Astro_Parameters, User_Parameters, Cosmo_Parameters, T21_coefficients):

        "Returns the Pop II components of the correlation functions of all observables at each z in zintegral"
        #HAC: I deleted the bubbles and EoR part, to be done later.....
        #self._iRnonlinear = np.arange(Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexmaxNL)
    

        _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(Cosmo_Parameters,self._zGreaterMatrix100[:, self._iRnonlinear])

        coeffzp1xa = T21_coefficients.coeff1LyAzp * T21_coefficients.coeff_Ja_xa
        coeffzp1Tx = T21_coefficients.coeff1Xzp

        coeffR1xa = T21_coefficients.coeff2LyAzpRR_II[:,self._iRnonlinear]
        coeffR1Tx = T21_coefficients.coeff2XzpRR_II[:,self._iRnonlinear]

        coeffmatrixxa = coeffR1xa.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1) * coeffR1xa.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)

        # gammaR1 = T21_coefficients.gamma_II_index2D[:, self._iRnonlinear] * growthRmatrix
        # gammamatrixR1R1 = gammaR1.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1) * gammaR1.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)

        # gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL')#np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster


        # SarahLibanore : change to introduce quantities required in the second order correction
        # --- #
        growthRmatrix1 = growthRmatrix.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1)
        growthRmatrix2 = growthRmatrix.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)
        growth_corr = growthRmatrix1 * growthRmatrix2

        gammaR1 = T21_coefficients.gamma_II_index2D[:, self._iRnonlinear] 
        sigmaR1 = T21_coefficients.sigmaofRtab[:, self._iRnonlinear] 
        sR1 = (sigmaR1).reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1)
        sR2 = (sigmaR1).reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)

        g1 = (gammaR1 * sigmaR1).reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1)
        g2 = (gammaR1 * sigmaR1).reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)
        gammamatrixR1R1 = g1 * g2

        corrdNL = self._corrdNL
        corrdNL_gs = ne.evaluate('corrdNL * growth_corr/ (sR1 * sR2)')
        gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL_gs')

        if Astro_Parameters.quadratic_SFRD_lognormal:

            gammaR1NL = T21_coefficients.gamma2_II_index2D[:, self._iRnonlinear] 
            g1NL = (gammaR1NL * sigmaR1**2).reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1)
            g2NL = (gammaR1NL * sigmaR1**2).reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)

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

        if (User_Parameters.FLAG_DO_DENS_NL):
            D_coeffR1xa = coeffR1xa.reshape(*coeffR1xa.shape, 1)
            DDgammaR1 = T21_coefficients.gamma_II_index2D[:, self._iRnonlinear] 
            D_gammaR1 = DDgammaR1.reshape(*DDgammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            # SarahLibanore
            if Astro_Parameters.quadratic_SFRD_lognormal:  

                DDsigmaR1 = T21_coefficients.sigmaofRtab[:, self._iRnonlinear] 
                D_sigmaR1 = DDsigmaR1.reshape(*DDsigmaR1.shape , 1)
                DDgammaR1N = T21_coefficients.gamma2_II_index2D[:, self._iRnonlinear] 
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

        coeffzp1Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(*T21_coefficients.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(1, 1, *T21_coefficients.coeff1Xzp.shape, 1)

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
        if Astro_Parameters.quadratic_SFRD_lognormal:
            gammaR2NL = np.copy(gammaR1NL)
            g1NL = (gammaR1NL * sigmaR1**2).reshape(*gammaR1NL.shape, 1, 1)
            g2NL = (gammaR2NL * sigmaR2**2).reshape(1, 1, *gammaR2NL.shape)

        gammamatrixR1R2 = g1 * g2

        self._II_deltaxi_Tx = np.zeros_like(self._II_deltaxi_xa)
        self._II_deltaxi_xaTx = np.zeros_like(self._II_deltaxi_xa)
        corrdNLBIG = corrdNL[:,:, np.newaxis, :,:] #dimensions zp1, R1, zp2, R2, and r which will be looped over below
        for ir in range(len(Cosmo_Parameters._Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            
            corrdNL_gs = ne.evaluate('corrdNL * growth_corr / (sR1 * sR2)')

            #HAC: Computations using ne.evaluate(...) use numexpr, which speeds up computations of massive numpy arrays
            gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R2 * corrdNL_gs')
            if Astro_Parameters.quadratic_SFRD_lognormal:

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


        if (User_Parameters.FLAG_DO_DENS_NL):
            D_coeffR2Tx = coeffR2Tx.reshape(1, *coeffR2Tx.shape, 1)
            D_coeffzp2Tx = coeffzp2Tx.flatten().reshape(1, *coeffzp2Tx.flatten().shape, 1)
            DDgammaR2 = np.copy(DDgammaR1)
            D_gammaR2 = DDgammaR2.reshape(1, *DDgammaR2.shape , 1)
            D_growthRmatrix = growthRmatrix[:,0].reshape(*growthRmatrix[:,0].shape, 1, 1, 1)
            D_corrdNL = corrdNLBIG.squeeze()[0].reshape(1, 1, *corrdNLBIG.squeeze()[0].shape)
        
            if Astro_Parameters.quadratic_SFRD_lognormal:                

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

    def get_all_corrs_IIxIII(self, Cosmo_Parameters, T21_coefficients):
        """
        Returns the Pop IIxIII cross-correlation function of all observables at each z in zintegral
        """
        #HAC: I deleted the bubbles and EoR part, to be done later.....


        corrdNL = self._corrdNL


        _coeffTx_units_II = T21_coefficients.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor
        _coeffTx_units_III = T21_coefficients.coeff_Gammah_Tx_III #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(Cosmo_Parameters,self._zGreaterMatrix100[:, self._iRnonlinear])
        gammaR1_II = T21_coefficients.gamma_II_index2D[:, self._iRnonlinear] * growthRmatrix
        gammaR1_III = T21_coefficients.gamma_III_index2D[:, self._iRnonlinear] * growthRmatrix

        coeffzp1xa = T21_coefficients.coeff1LyAzp * T21_coefficients.coeff_Ja_xa
        coeffzp1Tx = T21_coefficients.coeff1Xzp

        coeffR1xa_II = T21_coefficients.coeff2LyAzpRR_II[:,self._iRnonlinear]
        coeffR1xa_III = T21_coefficients.coeff2LyAzpRR_III[:,self._iRnonlinear]

        coeffR1Tx_II = T21_coefficients.coeff2XzpRR_II[:,self._iRnonlinear]
        coeffR1Tx_III = T21_coefficients.coeff2XzpRR_III[:,self._iRnonlinear]

        gammamatrix_R1II_R1III = gammaR1_II.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1) * gammaR1_III.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)
        coeffmatrixxa_R1II_R1III = coeffR1xa_II.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1) * coeffR1xa_III.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)

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

        coeffzp1Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(*T21_coefficients.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(1, 1, *T21_coefficients.coeff1Xzp.shape, 1)

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

        for ir in range(len(Cosmo_Parameters._Rtabsmoo)):
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
        
        
    # === BEGIN EDIT (par/perp eta split): generalized signature.
    # Original signature was get_xi_Sum_2ExpEta(self, xiEta, etaCoeff1, etaCoeff2)
    # using the isotropic factor (1 - K*xiEta)**(3/2). The new signature accepts
    # both modes; pass xiEtaperp=None for the legacy isotropic calculation. ===
    def get_xi_Sum_2ExpEta(self, xiEtapar, xiEtaperp, etaCoeff1, etaCoeff2):
        """
        Computes the correlation function of the VCB portion of the SFRD, expressed using sums of two exponentials
        if rho(z1, x1) / rhobar = Ae^-b tilde(eta) + Ce^-d tilde(eta) and rho(z2, x2) / rhobar = Fe^-g tilde(eta) + He^-k tilde(eta)
        Then this computes <rho(z1, x1) * rho(z2, x2)> - <rho(z1, x1)> <rho(z2, x2)>
        Refer to eq. A12 in 2407.18294 for more details

        Two computational modes are supported via the second argument:

          * ``xiEtaperp is None`` (legacy / isotropic):
                The first argument ``xiEtapar`` is treated as the single
                isotropic eta correlation function ``xiEta``, and the
                original four-term factor ``(1 - K * xiEta)**(3/2)`` is
                used for each term.

          * ``xiEtaperp is not None`` (parallel/perpendicular split):
                The first argument is the parallel eta correlation function
                and the second is the perpendicular one. Each
                ``(1 - K * xiEta)**(3/2)`` factor is replaced by
                ``((1 - K * xiEtapar) * (1 - K * xiEtaperp)**2)**(1/2)``
                with K = 6*K1*K2/((1+2*K1)*(1+2*K2)).

        Callers should select exactly one mode and pass the matching
        correlation function(s); the alternative array(s) need not (and
        should not) be constructed.
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

        # --- EDIT (par/perp eta split): mode dispatch on xiEtaperp ---
        if xiEtaperp is None:
            # Legacy isotropic eta correlation: single xiEta argument.
            xiEta = xiEtapar
            xiNumerator  = ne.evaluate('afBG * (1 / (1 - 6*bb * gg * xiEta / ((1+2*bb)*(1+2*gg)))**(3/2) - 1) + ahBK * (1 / (1 - 6*bb * kk * xiEta / ((1+2*bb)*(1+2*kk)))**(3/2) - 1) + cfDG * (1 / (1 - 6*dd * gg * xiEta / ((1+2*dd)*(1+2*gg)))**(3/2) - 1) + chDK * (1 / (1 - 6*dd * kk * xiEta / ((1+2*dd)*(1+2*kk)))**(3/2) - 1)')
        else:
            # Parallel/perpendicular split:
            # (1 - K * xi)**(3/2)  ->  ((1 - K * xipar) * (1 - K * xiperp)**2)**(1/2)
            xiNumerator  = ne.evaluate(
                'afBG * (1 / ((1 - 6*bb*gg*xiEtapar/((1+2*bb)*(1+2*gg))) * (1 - 6*bb*gg*xiEtaperp/((1+2*bb)*(1+2*gg)))**2)**(1/2) - 1)'
                ' + ahBK * (1 / ((1 - 6*bb*kk*xiEtapar/((1+2*bb)*(1+2*kk))) * (1 - 6*bb*kk*xiEtaperp/((1+2*bb)*(1+2*kk)))**2)**(1/2) - 1)'
                ' + cfDG * (1 / ((1 - 6*dd*gg*xiEtapar/((1+2*dd)*(1+2*gg))) * (1 - 6*dd*gg*xiEtaperp/((1+2*dd)*(1+2*gg)))**2)**(1/2) - 1)'
                ' + chDK * (1 / ((1 - 6*dd*kk*xiEtapar/((1+2*dd)*(1+2*kk))) * (1 - 6*dd*kk*xiEtaperp/((1+2*dd)*(1+2*kk)))**2)**(1/2) - 1)'
            )
        # --- end EDIT (par/perp eta split) ---

        xiDenominator  = ne.evaluate('afBG + ahBK + cfDG + chDK')

        xiTotal = ne.evaluate('xiNumerator / xiDenominator')

        return xiTotal
    # === END EDIT (par/perp eta split): get_xi_Sum_2ExpEta ===


    def get_all_corrs_III(self, User_Parameters, Cosmo_Parameters, T21_coefficients):
        "Returns the Pop III components of the correlation functions of all observables at each z in zintegral"
        #HAC: I deleted the bubbles and EoR part, to be done later.....


        corrdNL = self._corrdNL

        # === BEGIN EDIT (par/perp eta split): runtime dispatch.
        # Switch between legacy isotropic eta correlation and the
        # parallel/perpendicular split. Default False (legacy) so existing
        # behavior is preserved if the flag is absent on Cosmo_Parameters.
        # The code computes EITHER xiEta OR (xiEtapar, xiEtaperp), never both. ===
        USE_ETA_PARPERP_SPLIT = getattr(Cosmo_Parameters, 'USE_ETA_PARPERP_SPLIT', False)

        # Build EITHER the isotropic eta CF matrix OR the par/perp pair,
        # but not both, so the unused branch never allocates memory.
        if USE_ETA_PARPERP_SPLIT:
            corrEtaNLpar = Cosmo_Parameters.xiEtaPar_RR_CF[np.ix_(self._iRnonlinear,self._iRnonlinear)]
            corrEtaNLpar[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrEtaNLpar[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
            corrEtaNLpar = corrEtaNLpar.reshape(1, *corrEtaNLpar.shape)

            corrEtaNLperp = Cosmo_Parameters.xiEtaPerp_RR_CF[np.ix_(self._iRnonlinear,self._iRnonlinear)]
            corrEtaNLperp[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrEtaNLperp[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
            corrEtaNLperp = corrEtaNLperp.reshape(1, *corrEtaNLperp.shape)
        else:
            corrEtaNL = Cosmo_Parameters.xiEta_RR_CF[np.ix_(self._iRnonlinear,self._iRnonlinear)]
            corrEtaNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrEtaNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
            corrEtaNL = corrEtaNL.reshape(1, *corrEtaNL.shape)
        # === END EDIT (par/perp eta split): eta-CF matrix construction ===


        _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_III #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(Cosmo_Parameters,self._zGreaterMatrix100[:, self._iRnonlinear])
        gammaR1 = T21_coefficients.gamma_III_index2D[:, self._iRnonlinear] * growthRmatrix
        
        vcbCoeffs1 = T21_coefficients.vcb_expFitParams[:, self._iRnonlinear]
        vcbCoeffsR1 = np.transpose(vcbCoeffs1, (2, 0, 1))
        vcbCoeffsR1 = vcbCoeffsR1[:,:,:,np.newaxis,np.newaxis]
        vcbCoeffsR2 = np.moveaxis(vcbCoeffsR1, 3, 2)

        coeffzp1xa = T21_coefficients.coeff1LyAzp * T21_coefficients.coeff_Ja_xa
        coeffzp1Tx = T21_coefficients.coeff1Xzp

        coeffR1xa = T21_coefficients.coeff2LyAzpRR_III[:,self._iRnonlinear]
        coeffR1Tx = T21_coefficients.coeff2XzpRR_III[:,self._iRnonlinear]

        gammamatrixR1R1 = gammaR1.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1) * gammaR1.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)
        coeffmatrixxa = coeffR1xa.reshape(len(T21_coefficients.zintegral), 1, len(self._iRnonlinear),1) * coeffR1xa.reshape(len(T21_coefficients.zintegral), len(self._iRnonlinear), 1,1)

        gammaCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL') #np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster
        expGammaCorr = ne.evaluate('exp(gammaCorrdNL) - 1') # equivalent to np.exp(gammaTimesCorrdNL)-1.0

        if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:
            # --- EDIT (par/perp eta split): mode dispatch for xa term ---
            if USE_ETA_PARPERP_SPLIT:
                etaCorr_xa = self.get_xi_Sum_2ExpEta(corrEtaNLpar, corrEtaNLperp, vcbCoeffsR1, vcbCoeffsR2)
            else:
                etaCorr_xa = self.get_xi_Sum_2ExpEta(corrEtaNL, None, vcbCoeffsR1, vcbCoeffsR2)
            # --- end EDIT ---
            totalCorr = ne.evaluate('expGammaCorr * etaCorr_xa + expGammaCorr + etaCorr_xa - gammaCorrdNL') ###TO DO (linearized VCB flucts): - etaCorr_xa_lin #note that the Taylor expansion of the cross-term is 0 to linear order
        else:
            totalCorr = ne.evaluate('expGammaCorr - gammaCorrdNL') ###TO DO (linearized VCB flucts): - etaCorr_xa_lin #note that the Taylor expansion of the cross-term is 0 to linear order

        self._III_deltaxi_xa = np.einsum('ijkl->il', coeffmatrixxa * totalCorr , optimize = True)  # equivalent to self._III_deltaxi_xa = np.sum(coeffmatrixxa * ((np.exp(gammaTimesCorrdNL)-1.0) - gammaTimesCorrdNL), axis = (1,2))
        self._III_deltaxi_xa *= np.array([coeffzp1xa]).T**2 #brings it to xa units

        if (User_Parameters.FLAG_DO_DENS_NL): #no velocity contribution to density
            D_coeffR1xa = coeffR1xa.reshape(*coeffR1xa.shape, 1)
            D_gammaR1 = gammaR1.reshape(*gammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            self._III_deltaxi_dxa = np.sum(D_coeffR1xa * ((np.exp(D_gammaR1 * D_growthRmatrix * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)
            self._III_deltaxi_dxa *= np.array([coeffzp1xa]).T

        ### To compute Tx quantities, I'm broadcasting arrays such that the axes are zp1, R1, zp2, R2, r

        gammaR2 = np.copy(gammaR1) #already has growth factor in this
        gammamatrixR1R2 = gammaR1.reshape(*gammaR1.shape, 1, 1) * gammaR2.reshape(1, 1, *gammaR2.shape)

        coeffzp1Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(*T21_coefficients.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(1, 1, *T21_coefficients.coeff1Xzp.shape, 1)
        coeffR2Tx = np.copy(coeffR1Tx)
        coeffmatrixTxTx = coeffR1Tx.reshape(*coeffR1Tx.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffmatrixxaTx = coeffR1xa.reshape(*coeffR1xa.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffsTxALL = coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx
        coeffsXaTxALL = coeffzp2Tx * coeffmatrixxaTx

        corrdNLBIG = corrdNL[:,:, np.newaxis, :, :]
        # --- EDIT (par/perp eta split): expand only the eta arrays we built ---
        if USE_ETA_PARPERP_SPLIT:
            corrEtaNLparBIG  = corrEtaNLpar[:,:, np.newaxis, :, :]
            corrEtaNLperpBIG = corrEtaNLperp[:,:, np.newaxis, :, :]
        else:
            corrEtaNLBIG = corrEtaNL[:,:, np.newaxis, :, :]
        # --- end EDIT ---

        vcbCoeffsR1 = vcbCoeffsR1[:,:,:,:,:]
        vcbCoeffsR2 = np.transpose(vcbCoeffsR1, (0,3,4,1,2))

        self._III_deltaxi_Tx = np.zeros_like(self._III_deltaxi_xa)
        self._III_deltaxi_xaTx = np.zeros_like(self._III_deltaxi_xa)
        self._III_deltaxi_dTx = np.zeros_like(self._III_deltaxi_xa)

        for ir in range(len(Cosmo_Parameters._Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            # --- EDIT (par/perp eta split): per-ir slice of the active eta CF ---
            if USE_ETA_PARPERP_SPLIT:
                corrEtaNLpar  = corrEtaNLparBIG[:,:,:,:,ir]
                corrEtaNLperp = corrEtaNLperpBIG[:,:,:,:,ir]
            else:
                corrEtaNL = corrEtaNLBIG[:,:,:,:,ir]
            # --- end EDIT ---

            gammaCorrdNL = ne.evaluate('gammamatrixR1R2 * corrdNL')
            expGammaCorrdNL = ne.evaluate('exp(gammaCorrdNL) - 1')
            
            if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:
                # --- EDIT (par/perp eta split): mode dispatch for Tx term ---
                if USE_ETA_PARPERP_SPLIT:
                    etaCorr_Tx = self.get_xi_Sum_2ExpEta(corrEtaNLpar, corrEtaNLperp, vcbCoeffsR1, vcbCoeffsR2)
                else:
                    etaCorr_Tx = self.get_xi_Sum_2ExpEta(corrEtaNL, None, vcbCoeffsR1, vcbCoeffsR2)
                # --- end EDIT ---
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

        if (User_Parameters.FLAG_DO_DENS_NL): #no velocity contribution to density
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
        "Returns the power spectrum given a list of CFs (xi_list) evaluated at z=zlisttoconvert as input"

        _Pk_list = []

        for izp,zp in enumerate(zlisttoconvert):

            _kzp, _Pkzp = self.get_Pk_from_xi(self._rs_input_mcfit,xi_list[izp])
            _Pk_list.append(_Pkzp)
            #can ignore _kzp, it's the same as klist_PS above by construction


        return np.array(_Pk_list)


    def get_Pk_from_xi(self, rsinput, xiinput):
        "Generic Fourier Transform, returns Pk from an input Corr Func xi. kPf should be the same as _klistCF"

        kPf, Pf = mcfit.xi2P(rsinput, l=0, lowring=True)(xiinput, extrap=False)

        return kPf, Pf