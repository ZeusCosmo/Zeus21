"""

Code to compute correlation functions from power spectra and functions of them. Holds two classes: Correlations (with matter correlation functions smoothed over different R), and Power_Spectra (which will compute and hold the 21-cm power spectrum and power for derived quantities like xa, Tk, etc.)

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import mcfit
from scipy.special import gammaincc #actually very fast, no need to approximate
import numexpr as ne

from . import constants
from . import cosmology



class Correlations:
    "Class that calculates and keeps the correlation functions."

    def __init__(self, UserParams, Cosmo_Parameters, ClassCosmo):


        #we choose the k to match exactly the log FFT of input Rtabsmoo.

        self._klistCF, _dummy_ = mcfit.xi2P(Cosmo_Parameters._Rtabsmoo, l=0, lowring=True)(0*Cosmo_Parameters._Rtabsmoo, extrap=False)
        self.NkCF = len(self._klistCF)

        self._PklinCF = np.zeros(self.NkCF) # P(k) in 1/Mpc^3
        for ik, kk in enumerate(self._klistCF):
            self._PklinCF[ik] = ClassCosmo.pk(kk, 0.0) # function .pk(k,z)



        self._xif = mcfit.P2xi(self._klistCF, l=0, lowring=True)

        self.WINDOWTYPE = 'TOPHAT'
        #options are 'TOPHAT', 'TOPHAT1D' and 'GAUSS' (for now). TOPHAT is calibrated for EPS, but GAUSS has less ringing

        self.xi_RR_CF = self.get_xi_R1R2_z0(Cosmo_Parameters)
        ClassCosmo.pars['xi_RR_CF'] = np.copy(self.xi_RR_CF) #store correlation function for gamma_III correction in SFRD

        ###HAC: Interpolated object for eta power spectrum
        if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:
            P_eta_interp = interp1d(ClassCosmo.pars['k_eta'], ClassCosmo.pars['P_eta'], bounds_error = False, fill_value = 0)
            self._PkEtaCF = P_eta_interp(self._klistCF)
            self.xiEta_RR_CF = self.get_xiEta_R1R2(Cosmo_Parameters)
        else:
            self._PkEtaCF = np.zeros_like(self._PklinCF)
            self.xiEta_RR_CF = np.zeros_like(self.xi_RR_CF)
    def _WinTH(self,k,R):
        x = k * R
        return 3.0/x**2 * (np.sin(x)/x - np.cos(x))

    def _WinTH1D(self,k,R):
        x = k * R
        return  np.sin(x)/x

    def _WinG(self,k,R):
        x = k * R * constants.RGauss_factor
        return np.exp(-x**2/2.0)

    def Window(self, k, R):
        if self.WINDOWTYPE == 'TOPHAT':
            return self._WinTH(k, R)
        elif self.WINDOWTYPE == 'GAUSS':
            return self._WinG(k, R)
        elif self.WINDOWTYPE == 'TOPHAT1D':
            return self._WinTH1D(k, R)
        else:
            print('ERROR in Window. Wrong type')


    def get_xi_z0_lin(self):
        "Get correlation function of density, linearly extrapolated to z=0"
        ##Warning: definitely check if beyond LCDM!
        #currenetly unused, just for refernce and plots

        rslinCF, xilinCF = self._xif(self._PklinCF, extrap=False)

        return rslinCF, xilinCF

    def get_xi_R1R2_z0 (self, Cosmo_Parameters):
        "same as get_xi_z0_lin but smoothed over two different radii with Window(k,R) \
        same separations rs as get_xi_z0_lin so it does not output them."
        
        ###HAC: Broadcasted to improve efficiency
        ###HAC: dim 0 is R1, dim 1 is R2, dim 2 is r, where R1 and R2 are smoothing radii and r is the argument of xi(r)
        lengthRarray = Cosmo_Parameters.NRs
        windowR1 = self.Window(self._klistCF.reshape(lengthRarray, 1, 1), Cosmo_Parameters._Rtabsmoo.reshape(1, 1, lengthRarray))
        windowR2 = self.Window(self._klistCF.reshape(1, lengthRarray,1), Cosmo_Parameters._Rtabsmoo.reshape(1, 1, lengthRarray))
        
        _PkRR = np.array([[self._PklinCF]]) * windowR1 * windowR2
        
        self.rlist_CF, xi_RR_CF = self._xif(_PkRR, extrap = False)

        return xi_RR_CF
        
    ###HAC: The next two are the same, but for
    def get_xiEta(self, Cosmo_Parameters, ClassCosmo):
        "Get correlation function of v^2 at z_drag (~1060 for LCDM parameters)"
        ##Warning: definitel check if beyond LCDM!
        #currently unused, just for reference and plots
        
        rsEtaCF, xiEtaCF = self._xif(self._PkEtaCF, extrap=False)
        
        return rsEtaCF, xiEtaCF
        
    def get_xiEta_R1R2(self, Cosmo_Parameters):
        "same as get_xiEta but smoothed over two different radii with Window"
        
        ###HAC: Broadcasted to improve efficiency
        ###HAC: dim 0 is R1, dim 1 is R2, dim 2 is r, where R1 and R2 are smoothing radii and r is the argument of xi(r)
        lengthRarray = len(Cosmo_Parameters._Rtabsmoo)
        
        windowR1 = self.Window(self._klistCF.reshape(lengthRarray, 1, 1), Cosmo_Parameters._Rtabsmoo.reshape(1, 1, lengthRarray))
        windowR2 = self.Window(self._klistCF.reshape(1, lengthRarray,1), Cosmo_Parameters._Rtabsmoo.reshape(1, 1, lengthRarray))

        _PkEtaRR = np.array([[self._PkEtaCF]]) * windowR1 * windowR2

        self.rlist_CF, xiEta_RR_CF = self._xif(_PkEtaRR, extrap = False)

        return xiEta_RR_CF
        



class Power_Spectra:
    "Get power spetrum from correlation functions and coefficients"

    def __init__(self, User_Parameters, Cosmo_Parameters, Astro_Parameters, ClassCosmo, Correlations, T21_coefficients, RSD_MODE=1):

#        print("STEP 0: Variable Setup")
        #set up some variables
        self._rs_input_mcfit = Correlations.rlist_CF #just to make notation simpler
        self.klist_PS = Correlations._klistCF
        self.RSD_MODE = RSD_MODE #redshift-space distortion mode. 0 = None (mu=0), 1 = Spherical avg (like 21-cmFAST), 2 = LoS only (mu=1). 2 is more observationally relevant, whereas 1 the standard assumption in sims. 0 is just for comparison with real-space #TODO: mode to save at different mu

        #first get the linear window functions -- note it already has growth factor in it, so it multiplies Pmatter(z=0)
        self.kwindow, self.windowalpha_II = self.get_xa_window(Cosmo_Parameters, Correlations, T21_coefficients, pop = 2)
        self._kwindowX, self.windowxray_II = self.get_Tx_window(Cosmo_Parameters, Correlations, T21_coefficients, pop = 2)
        
        if Astro_Parameters.USE_POPIII == True:
            self.kwindow, self.windowalpha_III = self.get_xa_window(Cosmo_Parameters, Correlations, T21_coefficients, pop = 3)
            self._kwindowX, self.windowxray_III = self.get_Tx_window(Cosmo_Parameters, Correlations, T21_coefficients, pop = 3)
        else:
            self.windowalpha_III = np.zeros_like(self.windowalpha_II)
            self.windowxray_III = np.zeros_like(self.windowxray_II)
            
        #calculate some growth etc, and the bubble biases for the xHI linear window function:
        self._lingrowthd = cosmology.growth(Cosmo_Parameters, T21_coefficients.zintegral)

        #We don't care about bubbles at the moment
        # if(constants.FLAG_DO_BUBBLES):
        #     self..calculate_barrier(Cosmo_Parameters, T21_coefficients)
        #     self..get_bubbles(Cosmo_Parameters, Correlations, T21_coefficients)
        #     self..windowxion = np.array([Correlations.Window(self..Rbub_star[iz]) for iz in range(T21_coefficients.Nzintegral)]) #Window returns a k-array. Smooths at the peak of the BMF

        #     self..windowxion = (self..windowxion.T*T21_coefficients.Qstar * self..bias_bub_avg * self.._lingrowthd * np.exp(-T21_coefficients.Qstar) ).T #normalize
        # else:
        #     self..windowxion = np.zeros_like(self..windowalpha)


        ##############################

#        print("STEP 1: Computing Nonlinear Power Spectra")
        #finally, get all the nonlinear correlation functions:
#        print("Computing Pop II-dependent power spectra")
        self.get_all_corrs_II(User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients)
        
        if Astro_Parameters.USE_POPIII == True:
#            print("Computing Pop IIxIII-dependent cross power spectra")
            self.get_all_corrs_IIxIII(User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients)
            
#            print("Computing Pop III-dependent power spectra")
            self.get_all_corrs_III(User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients)
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
        self._Pk_xa_lin_II = self.windowalpha_II**2 * Correlations._PklinCF
        self._Pk_xa_lin_III = self.windowalpha_III**2 * Correlations._PklinCF ###TO DO (linearized VCB flucts):+ self.windowalphaVel_III**2 * Correlations._PkEtaCF
        self._Pk_xa_lin_IIxIII = 2* self.windowalpha_II * self.windowalpha_III * Correlations._PklinCF #Pop IIxIII cross term doesn't have a velocity component

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
        self._Pk_Tx_lin_II = self.windowxray_II**2 * Correlations._PklinCF
        self._Pk_Tx_lin_III = self.windowxray_III**2 * Correlations._PklinCF ###TO DO (linearized VCB flucts):+ self.windowxrayVel_III**2 * Correlations._PkEtaCF
        self._Pk_Tx_lin_IIxIII = 2* self.windowxray_II * self.windowxray_III * Correlations._PklinCF #Pop IIxIII cross term doesn't have a velocity component

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
        self._Pk_xaTx_lin_II = self.windowalpha_II * self.windowxray_II * Correlations._PklinCF
        self._Pk_xaTx_lin_III = self.windowalpha_III * self.windowxray_III * Correlations._PklinCF ###TO DO (linearized VCB flucts):+ self.windowalphaVel_III * self.windowxrayVel_III * Correlations._PkEtaCF
        self._Pk_xaTx_lin_IIxIII = (self.windowalpha_II * self.windowxray_III + self.windowalpha_III * self.windowxray_II) * Correlations._PklinCF

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
        self._Pk_d_lin = np.outer(self._lingrowthd**2, Correlations._PklinCF) #No Pop II or III contribution
        self.Deltasq_d_lin = self._Pk_d_lin * self._k3over2pi2 #note that it still has units of xa_avg

        self._Pk_dxa_lin_II = (self.windowalpha_II.T * self._lingrowthd).T * Correlations._PklinCF
        self._Pk_dxa_lin_III = (self.windowalpha_III.T * self._lingrowthd).T * Correlations._PklinCF #No velocity component

        self._Pk_dTx_lin_II = (self.windowxray_II.T * self._lingrowthd).T * Correlations._PklinCF
        self._Pk_dTx_lin_III = (self.windowxray_III.T * self._lingrowthd).T * Correlations._PklinCF #No velocity component

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
            self._Pk_xion_lin = self.windowxion**2 * Correlations._PklinCF
            self.Deltasq_xion_lin = self._Pk_xion_lin * self._k3over2pi2

            self._d_Pk_xion_nl = self.get_list_PS(self._deltaxi_xi, T21_coefficients.zintegral)
            self.Deltasq_xion = self.Deltasq_xion_lin + self._d_Pk_xion_nl * self._k3over2pi2

            #cross with density
            self._Pk_dxion_lin = (self.windowxion.T * self._lingrowthd).T  * Correlations._PklinCF
            self.Deltasq_dxion_lin = self._Pk_dxion_lin * self._k3over2pi2

            self._d_Pk_dxion_nl = self.get_list_PS(self._deltaxi_dxi, T21_coefficients.zintegral)
            self.Deltasq_dxion = self.Deltasq_dxion_lin + self._d_Pk_dxion_nl * self._k3over2pi2

            #cross with xa
            self._Pk_xaxion_lin = self.windowxion * self.windowalpha  * Correlations._PklinCF
            self.Deltasq_xaxion_lin = self._Pk_xaxion_lin * self._k3over2pi2

            self._d_Pk_xaxion_nl = self.get_list_PS(self._deltaxi_xaxi, T21_coefficients.zintegral)
            self.Deltasq_xaxion = self.Deltasq_xaxion_lin + self._d_Pk_xaxion_nl * self._k3over2pi2

            #and cross with Tx
            self._Pk_Txxion_lin = self.windowxion * self.windowxray  * Correlations._PklinCF
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



    def get_xa_window(self, Cosmo_Parameters, Correlations, T21_coefficients, pop = 0): #set pop to 2 or 3, default zero just so python doesn't complain
        "Returns the xa window function for all z in zintegral"
        
        zGreaterMatrix100 = np.copy(T21_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        coeffzp = T21_coefficients.coeff1LyAzp
        coeffJaxa = T21_coefficients.coeff_Ja_xa

        growthRmatrix = cosmology.growth(Cosmo_Parameters, zGreaterMatrix100)

        if pop == 2:
            coeffRmatrix = T21_coefficients.coeff2LyAzpRR_II
            gammaRmatrix = T21_coefficients.gamma_II_index2D * growthRmatrix
        elif pop == 3:
            coeffRmatrix = T21_coefficients.coeff2LyAzpRR_III
            gammaRmatrix = T21_coefficients.gamma_III_index2D * growthRmatrix
        else:
            print("Must set pop to either 2 or 3!")

        _wincoeffsMatrix = coeffRmatrix * gammaRmatrix

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
            _wincoeffsMatrix /=(4*np.pi * T21_coefficients.Rtabsmoo**2) * (T21_coefficients.Rtabsmoo * T21_coefficients.dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2 * Delta r)
            _kwinalpha, _win_alpha = self.get_Pk_from_xi(T21_coefficients.Rtabsmoo, _wincoeffsMatrix)

        else:
            _kwinalpha = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RtabsmooMesh, kWinAlphaMesh = np.meshgrid(T21_coefficients.zintegral, T21_coefficients.Rtabsmoo, _kwinalpha, indexing = 'ij', sparse = True)

            _win_alpha = coeffRgammaRmatrix * Correlations._WinTH(RtabsmooMesh, kWinAlphaMesh)
            _win_alpha = np.sum(_win_alpha, axis = 1)

        _win_alpha *= np.array([coeffzp*coeffJaxa]).T
        
        return _kwinalpha, _win_alpha


    def get_Tx_window(self, Cosmo_Parameters,  Correlations, T21_coefficients, pop = 0): #set pop to 2 or 3, default zero just so python doesn't complain
        "Returns the Tx window function for all z in zintegral"

        zGreaterMatrix100 = np.copy(T21_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        coeffzp = np.array([T21_coefficients.coeff1Xzp]).T
        growthRmatrix = cosmology.growth(Cosmo_Parameters, zGreaterMatrix100)

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

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #do the standard 1D TopHat
            _wincoeffs = coeffRmatrix * gammaRmatrix #array in logR space
            _wincoeffs /=(4*np.pi * T21_coefficients.Rtabsmoo**2) * (T21_coefficients.Rtabsmoo * T21_coefficients.dlogRR) # so we can just use mcfit for logFFT, 1/(4pir^2) * Delta r
            _kwinTx, _win_Tx_curr = self.get_Pk_from_xi(T21_coefficients.Rtabsmoo, _wincoeffs)

        else:
            _kwinTx = self.klist_PS

            coeffRgammaRmatrix = coeffRmatrix * gammaRmatrix
            coeffRgammaRmatrix = coeffRgammaRmatrix.reshape(*coeffRgammaRmatrix.shape, 1)

            dummyMesh, RtabsmooMesh, kWinTxMesh = np.meshgrid(T21_coefficients.zintegral, T21_coefficients.Rtabsmoo, _kwinTx, indexing = 'ij', sparse = True)

            _win_Tx_curr = coeffRgammaRmatrix * Correlations._WinTH(RtabsmooMesh, kWinTxMesh)
            _win_Tx_curr = np.sum(_win_Tx_curr , axis = 1)

        _win_Tx = _win_Tx_curr * coeffzp
        _win_Tx = np.cumsum(_win_Tx[::-1], axis = 0)[::-1]

        _win_Tx =_win_Tx * np.array([_coeffTx_units]).T

        return _kwinTx, _win_Tx



    def get_all_corrs_II(self, User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients):
        "Returns the Pop II components of the correlation functions of all observables at each z in zintegral"
        #HAC: I deleted the bubbles and EoR part, to be done later.....
        #_iRnonlinear = np.arange(Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexmaxNL)
    
        zGreaterMatrix100 = np.copy(T21_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        _iRnonlinear = np.arange(Cosmo_Parameters.indexmaxNL)
        corrdNL = Correlations.xi_RR_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
        #for R<RNL fix at RNL, avoids corelations blowing up at low R
        corrdNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrdNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
        corrdNL = corrdNL.reshape((1, *corrdNL.shape))

        _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(Cosmo_Parameters,zGreaterMatrix100[:, _iRnonlinear])
        gammaR1 = T21_coefficients.gamma_II_index2D[:, _iRnonlinear] * growthRmatrix

        coeffzp1xa = T21_coefficients.coeff1LyAzp * T21_coefficients.coeff_Ja_xa
        coeffzp1Tx = T21_coefficients.coeff1Xzp

        coeffR1xa = T21_coefficients.coeff2LyAzpRR_II[:,_iRnonlinear]
        coeffR1Tx = T21_coefficients.coeff2XzpRR_II[:,_iRnonlinear]

        gammamatrixR1R1 = gammaR1.reshape(len(T21_coefficients.zintegral), 1, len(_iRnonlinear),1) * gammaR1.reshape(len(T21_coefficients.zintegral), len(_iRnonlinear), 1,1)
        coeffmatrixxa = coeffR1xa.reshape(len(T21_coefficients.zintegral), 1, len(_iRnonlinear),1) * coeffR1xa.reshape(len(T21_coefficients.zintegral), len(_iRnonlinear), 1,1)

        gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL')#np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster
        expGammaCorrMinusLinear = ne.evaluate('exp(gammaTimesCorrdNL) - 1 - gammaTimesCorrdNL')
        self._II_deltaxi_xa = np.einsum('ijkl->il', coeffmatrixxa * expGammaCorrMinusLinear, optimize = True)
        self._II_deltaxi_xa *= np.array([coeffzp1xa]).T**2 #brings it to xa units

        if (User_Parameters.FLAG_DO_DENS_NL):
            D_coeffR1xa = coeffR1xa.reshape(*coeffR1xa.shape, 1)
            D_gammaR1 = gammaR1.reshape(*gammaR1.shape , 1)
            D_growthRmatrix = growthRmatrix[:,:1].reshape(*growthRmatrix[:,:1].shape, 1)
            D_corrdNL = corrdNL[:1,0,:,:]

            self._II_deltaxi_dxa = np.sum(D_coeffR1xa * ((np.exp(D_gammaR1 * D_growthRmatrix * D_corrdNL )-1.0 ) - D_gammaR1 * D_growthRmatrix * D_corrdNL), axis = 1)
            self._II_deltaxi_dxa *= np.array([coeffzp1xa]).T

            self._II_deltaxi_d = (np.exp(growthRmatrix[:,:1]**2 * corrdNL[0,0,0,:]) - 1.0) - growthRmatrix[:,:1]**2 * corrdNL[0,0,0,:]



        ### To compute Tx quantities, I'm broadcasting arrays such that the axes are zp1, R1, zp2, R2, and looping over r
        gammaR2 = np.copy(gammaR1) #already has growth factor in this
        gammamatrixR1R2 = gammaR1.reshape(*gammaR1.shape, 1, 1) * gammaR2.reshape(1, 1, *gammaR2.shape)

        coeffzp1Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(*T21_coefficients.coeff1Xzp.shape, 1, 1, 1)
        coeffzp2Tx = np.copy(T21_coefficients.coeff1Xzp).reshape(1, 1, *T21_coefficients.coeff1Xzp.shape, 1)

        coeffR2Tx = np.copy(coeffR1Tx)
        coeffmatrixTxTx = coeffR1Tx.reshape(*coeffR1Tx.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffmatrixxaTx = coeffR1xa.reshape(*coeffR1xa.shape, 1, 1) * coeffR2Tx.reshape(1, 1, *coeffR2Tx.shape)
        coeffsTxALL =  coeffzp1Tx * coeffzp2Tx * coeffmatrixTxTx
        coeffsXaTxALL = coeffzp2Tx * coeffmatrixxaTx


        self._II_deltaxi_Tx = np.zeros_like(self._II_deltaxi_xa)
        self._II_deltaxi_xaTx = np.zeros_like(self._II_deltaxi_xa)
        corrdNLBIG = corrdNL[:,:, np.newaxis, :,:] #dimensions zp1, R1, zp2, R2, and r which will be looped over below
        for ir in range(len(T21_coefficients.Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            
            #HAC: Computations using ne.evaluate(...) use numexpr, which speeds up computations of massive numpy arrays
            gammaTimesCorrdNL = ne.evaluate('gammamatrixR1R2 * corrdNL')
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
            D_gammaR2 = gammaR2.reshape(1, *gammaR2.shape , 1)
            D_growthRmatrix = growthRmatrix[:,0].reshape(*growthRmatrix[:,0].shape, 1, 1, 1)
            D_corrdNL = corrdNLBIG.squeeze()[0].reshape(1, 1, *corrdNLBIG.squeeze()[0].shape)

            self._II_deltaxi_dTx =  D_coeffzp2Tx * np.sum(D_coeffR2Tx * ((np.exp(D_gammaR2 * D_growthRmatrix * D_corrdNL)-1.0) - D_gammaR2 * D_growthRmatrix * D_corrdNL), axis = 2)

            self._II_deltaxi_dTx = np.moveaxis(self._II_deltaxi_dTx, 1, 0)
            self._II_deltaxi_dTx = np.cumsum(self._II_deltaxi_dTx[::-1], axis = 0)[::-1]
            self._II_deltaxi_dTx = np.moveaxis(self._II_deltaxi_dTx, 1, 0)
            self._II_deltaxi_dTx = np.einsum('iik->ik', self._II_deltaxi_dTx, optimize = True)
            self._II_deltaxi_dTx *= np.array([_coeffTx_units]).T
            
        return 1

    def get_all_corrs_IIxIII(self, User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients):
        "Returns the Pop IIxIII cross-correlation function of all observables at each z in zintegral"
        #HAC: I deleted the bubbles and EoR part, to be done later.....
        #_iRnonlinear = np.arange(Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexmaxNL)

        zGreaterMatrix100 = np.copy(T21_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        _iRnonlinear = np.arange(Cosmo_Parameters.indexmaxNL)
        corrdNL = Correlations.xi_RR_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
        #for R<RNL fix at RNL, avoids corelations blowing up at low R
        corrdNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrdNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
        corrdNL = corrdNL.reshape((1, *corrdNL.shape))

        _coeffTx_units_II = T21_coefficients.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor
        _coeffTx_units_III = T21_coefficients.coeff_Gammah_Tx_III #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(Cosmo_Parameters,zGreaterMatrix100[:, _iRnonlinear])
        gammaR1_II = T21_coefficients.gamma_II_index2D[:, _iRnonlinear] * growthRmatrix
        gammaR1_III = T21_coefficients.gamma_III_index2D[:, _iRnonlinear] * growthRmatrix

        coeffzp1xa = T21_coefficients.coeff1LyAzp * T21_coefficients.coeff_Ja_xa
        coeffzp1Tx = T21_coefficients.coeff1Xzp

        coeffR1xa_II = T21_coefficients.coeff2LyAzpRR_II[:,_iRnonlinear]
        coeffR1xa_III = T21_coefficients.coeff2LyAzpRR_III[:,_iRnonlinear]

        coeffR1Tx_II = T21_coefficients.coeff2XzpRR_II[:,_iRnonlinear]
        coeffR1Tx_III = T21_coefficients.coeff2XzpRR_III[:,_iRnonlinear]

        gammamatrix_R1II_R1III = gammaR1_II.reshape(len(T21_coefficients.zintegral), 1, len(_iRnonlinear),1) * gammaR1_III.reshape(len(T21_coefficients.zintegral), len(_iRnonlinear), 1,1)
        coeffmatrixxa_R1II_R1III = coeffR1xa_II.reshape(len(T21_coefficients.zintegral), 1, len(_iRnonlinear),1) * coeffR1xa_III.reshape(len(T21_coefficients.zintegral), len(_iRnonlinear), 1,1)

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

        for ir in range(len(T21_coefficients.Rtabsmoo)):
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
        # Computes the correlation function of the VCB portion of the SFRD, expressed using sums of two exponentials
        # if rho(z1, x1) / rhobar = Ae^-b tilde(eta) + Ce^-d tilde(eta)
        # and rho(z2, x2) / rhobar = Fe^-g tilde(eta) + He^-k tilde(eta)
        # then this computes <rho(z1, x1) * rho(z2, x2)> - <rho(z1, x1)> <rho(z2, x2)>
        # Refer to eq. A12 in 2407.18294 for more details
        
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


    def get_all_corrs_III(self, User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients):
        "Returns the Pop III components of the correlation functions of all observables at each z in zintegral"
        #HAC: I deleted the bubbles and EoR part, to be done later.....
        #_iRnonlinear = np.arange(Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexmaxNL)
        zGreaterMatrix100 = np.copy(T21_coefficients.zGreaterMatrix)
        zGreaterMatrix100[np.isnan(zGreaterMatrix100)] = 100

        _iRnonlinear = np.arange(Cosmo_Parameters.indexmaxNL) #for R<RNL fix at RNL, avoids corelations blowing up at low R

        corrdNL = Correlations.xi_RR_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
        corrdNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrdNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
        corrdNL = corrdNL.reshape((1, *corrdNL.shape))

        corrEtaNL = Correlations.xiEta_RR_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
        corrEtaNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrEtaNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
        corrEtaNL = corrEtaNL.reshape(1, *corrEtaNL.shape)


        _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_III #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor

        growthRmatrix = cosmology.growth(Cosmo_Parameters,zGreaterMatrix100[:, _iRnonlinear])
        gammaR1 = T21_coefficients.gamma_III_index2D[:, _iRnonlinear] * growthRmatrix
        
        vcbCoeffs1 = T21_coefficients.vcb_expFitParams[:, _iRnonlinear]
        vcbCoeffsR1 = np.transpose(vcbCoeffs1, (2, 0, 1))
        vcbCoeffsR1 = vcbCoeffsR1[:,:,:,np.newaxis,np.newaxis]
        vcbCoeffsR2 = np.moveaxis(vcbCoeffsR1, 3, 2)

        coeffzp1xa = T21_coefficients.coeff1LyAzp * T21_coefficients.coeff_Ja_xa
        coeffzp1Tx = T21_coefficients.coeff1Xzp

        coeffR1xa = T21_coefficients.coeff2LyAzpRR_III[:,_iRnonlinear]
        coeffR1Tx = T21_coefficients.coeff2XzpRR_III[:,_iRnonlinear]

        gammamatrixR1R1 = gammaR1.reshape(len(T21_coefficients.zintegral), 1, len(_iRnonlinear),1) * gammaR1.reshape(len(T21_coefficients.zintegral), len(_iRnonlinear), 1,1)
        coeffmatrixxa = coeffR1xa.reshape(len(T21_coefficients.zintegral), 1, len(_iRnonlinear),1) * coeffR1xa.reshape(len(T21_coefficients.zintegral), len(_iRnonlinear), 1,1)

        gammaCorrdNL = ne.evaluate('gammamatrixR1R1 * corrdNL') #np.einsum('ijkl,ijkl->ijkl', gammamatrixR1R1, corrdNL, optimize = True) #same thing as gammamatrixR1R1 * corrdNL but faster
        expGammaCorr = ne.evaluate('exp(gammaCorrdNL) - 1') # equivalent to np.exp(gammaTimesCorrdNL)-1.0

        if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:
            etaCorr_xa = self.get_xi_Sum_2ExpEta(corrEtaNL, vcbCoeffsR1, vcbCoeffsR2)
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
        corrEtaNLBIG = corrEtaNL[:,:, np.newaxis, :, :]

        vcbCoeffsR1 = vcbCoeffsR1[:,:,:,:,:]
        vcbCoeffsR2 = np.transpose(vcbCoeffsR1, (0,3,4,1,2))

        self._III_deltaxi_Tx = np.zeros_like(self._III_deltaxi_xa)
        self._III_deltaxi_xaTx = np.zeros_like(self._III_deltaxi_xa)
        self._III_deltaxi_dTx = np.zeros_like(self._III_deltaxi_xa)

        for ir in range(len(T21_coefficients.Rtabsmoo)):
            corrdNL = corrdNLBIG[:,:,:,:,ir]
            corrEtaNL = corrEtaNLBIG[:,:,:,:,ir]

            gammaCorrdNL = ne.evaluate('gammamatrixR1R2 * corrdNL')
            expGammaCorrdNL = ne.evaluate('exp(gammaCorrdNL) - 1')
            
            if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:
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



# Below is the old get_all_corrs function for reference. It has some EoR bubbles functions that are incomplete (I think)
#def get_all_corrs(self, User_Parameters, Cosmo_Parameters, Correlations, T21_coefficients):
#    "Returns the correlation function of all observable at each z in zintegral"
#
#    #_iRnonlinear = np.arange(Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexmaxNL)
#    _iRnonlinear = np.arange(Cosmo_Parameters.indexmaxNL)
#    corrdNL = Correlations.xi_RR_CF[np.ix_(_iRnonlinear,_iRnonlinear)]
#
#    #for R<RNL fix at RNL, avoids corelations blowing up at low R
#    corrdNL[0:Cosmo_Parameters.indexminNL,0:Cosmo_Parameters.indexminNL] = corrdNL[Cosmo_Parameters.indexminNL,Cosmo_Parameters.indexminNL]
#
#    corr_deltaR1R2z0 = np.transpose(corrdNL, (2,0,1)) #so we can broadcast it to the sum below, first index is zp now
#
#    self._deltaxi_Tx = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_xa = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_xaTx = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#
#    self._deltaxi_d = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_dxa = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_dTx = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#
#
#    self._deltaxi_xi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_dxi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_xaxi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#    self._deltaxi_Txxi = np.zeros((T21_coefficients.Nzintegral,Correlations.NkCF))
#
#
#
#    _coeffTx_units = T21_coefficients.coeff_Gammah_Tx_II #includes -10^40 erg/s/SFR normalizaiton and erg/K conversion factor
#
#    for izp1,zp1 in reversed(list(enumerate(T21_coefficients.zintegral))): #reversed order sum to go from high to low z
#        if (izp1 < len(T21_coefficients.zintegral)-1):#start by summing over the previous one. izp1 decreases between steps
#            self._deltaxi_Tx[izp1] = self._deltaxi_Tx[izp1+1]
#
#        zpRlist1 = T21_coefficients.ztabRsmoo[izp1,_iRnonlinear]
#        growthRlist1 = cosmology.growth(Cosmo_Parameters,zpRlist1)
#        gammaR1 = T21_coefficients.gamma_index2D[izp1,_iRnonlinear] * growthRlist1
#
#        coeffzp1xa = T21_coefficients.coeff1LyAzp[izp1] * T21_coefficients.coeff_Ja_xa[izp1]
#        coeffzp1Tx = T21_coefficients.coeff1Xzp[izp1] #we can't multiply at the end because each object is summed over previous z too.
#
#        coeffR1xa = T21_coefficients.coeff2LyAzpRR[izp1,_iRnonlinear]
#        coeffR1Tx = T21_coefficients.coeff2XzpRR[izp1,_iRnonlinear]
#
#
#
#        #for xa auto corr and delta we don't need the second zp2 sum:
#        gammamatrixR1R1 = np.outer(gammaR1,gammaR1)
#        coeffmatrixxa = np.outer(coeffR1xa,coeffR1xa)
#        self._deltaxi_xa[izp1] = np.sum(coeffmatrixxa * ((np.exp(gammamatrixR1R1 * corr_deltaR1R2z0)-1.0) - gammamatrixR1R1 * corr_deltaR1R2z0) , axis=(1,2))
#        self._deltaxi_xa[izp1] *= (coeffzp1xa)**2 #brings it to xa units
#
#
#
#        if(User_Parameters.FLAG_DO_DENS_NL):
#            self._deltaxi_dxa[izp1] = np.sum(coeffR1xa * ((np.exp(gammaR1 * growthRlist1[0] * corr_deltaR1R2z0[:,0])-1.0) - gammaR1 * growthRlist1[0] * corr_deltaR1R2z0[:,0]) , axis=(1))
#            self._deltaxi_dxa[izp1] *= coeffzp1xa #brings it to xa units
#            #for d-d autocorrelation only keep the most local term
#            self._deltaxi_d[izp1] = (np.exp(growthRlist1[0]**2 * corr_deltaR1R2z0[:,0,0])-1.0) - growthRlist1[0]**2 * corr_deltaR1R2z0[:,0,0]
#
#
#        if(constants.FLAG_DO_BUBBLES):
#            _indexRbub = self._Rbub_star_index[izp1] - Cosmo_Parameters.indexminNL #to get the correct array element, if too large/small for nonlinearities just skip the nonlinear part, outside of k range
#        else:
#            _indexRbub = -1
#
#        _flag_doEoRNL = constants.FLAG_DO_BUBBLES and zp1 < constants.ZMAX_Bubbles and T21_coefficients.Qion_avg[izp1] < 1.0 and _indexRbub >= 0 and _indexRbub < len(_iRnonlinear)
#        #all these things have to be true for us to run the nonlinear+bubble part
#
#        if(_flag_doEoRNL):
#            _eminusQstar = np.exp(-T21_coefficients.Qstar[izp1])
#            gammaeffxHI = -T21_coefficients.Qstar[izp1] * self.bias_bub_avg[izp1] * growthRlist1[0] #effective bias of the xion term. includes growth
#
#            self._deltaxi_xaxi[izp1] = np.sum(coeffR1xa * ((np.exp(gammaR1 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub])-1.0) - gammaR1 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub]) , axis=(1))
#            self._deltaxi_xaxi[izp1] *= coeffzp1xa * _eminusQstar #brings it to xa units
#
#            self._deltaxi_dxi[izp1] =  (1.0 - np.exp(gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,0,_indexRbub]) ) -  gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,0,_indexRbub]
#            self._deltaxi_dxi[izp1] *= _eminusQstar
#
#            #for autocorrelation we have a density and a bubble/random term. first density
#            self._deltaxi_xi[izp1] =  (np.exp(-2.0 * gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,_indexRbub,_indexRbub]) -1.0) -  (-2.0) * gammaeffxHI * growthRlist1[0] * corr_deltaR1R2z0[:,_indexRbub,_indexRbub]
#            #plus the bubble part, fully nonlinear, no "correction wrt linear"
#            self._deltaxi_xi[izp1] += (np.exp(self.Qo_tab[izp1]) - 1.0)
#
#            self._deltaxi_xi[izp1] *= _eminusQstar**2
#
#
#        for izp2,zp2 in reversed(list(enumerate(T21_coefficients.zintegral))): #double loop because nonlocal in time sum.
#
#            _factorzp1equalzp2 = 2.0 #factor for 2 or 1 depending on whether they are the same for the sum below
#            if (izp2 < izp1): #sum only for z >= zp1, not below
#                continue
#            elif (izp2 == izp1):
#                _factorzp1equalzp2 = 1.0
#
#
#            coeffzp2Tx = T21_coefficients.coeff1Xzp[izp2] #inside zp2 it's always Tx since it's the nonlocal-in-time one
#            zpRlist2 = T21_coefficients.ztabRsmoo[izp2,_iRnonlinear]
#            growthRlist2 = cosmology.growth(Cosmo_Parameters,zpRlist2)
#
#            gammaR2 = T21_coefficients.gamma_index2D[izp2,_iRnonlinear] * growthRlist2
#            gammamatrixR1R2 = np.outer(gammaR1,gammaR2)
#
#
#            coeffR2Tx = T21_coefficients.coeff2XzpRR[izp2,_iRnonlinear]
#            coeffmatrixTxTx = np.outer(coeffR1Tx,coeffR2Tx)
#            coeffmatrixxaTx = np.outer(coeffR1xa,coeffR2Tx)
#
#            self._deltaxi_Tx[izp1] += _factorzp1equalzp2 * coeffzp1Tx * coeffzp2Tx * np.sum(coeffmatrixTxTx * ((np.exp(gammamatrixR1R2 * corr_deltaR1R2z0)-1.0) - gammamatrixR1R2 * corr_deltaR1R2z0) , axis=(1,2))
#
#            self._deltaxi_xaTx[izp1] += coeffzp2Tx * np.sum(coeffmatrixxaTx * ((np.exp(gammamatrixR1R2 * corr_deltaR1R2z0)-1.0) - gammamatrixR1R2 * corr_deltaR1R2z0) , axis=(1,2))
#
#            if(User_Parameters.FLAG_DO_DENS_NL):
#                self._deltaxi_dTx[izp1] += coeffzp2Tx * np.sum(coeffR2Tx * ((np.exp(gammaR2* growthRlist1[0] * corr_deltaR1R2z0[:,0])-1.0) - gammaR2* growthRlist1[0] * corr_deltaR1R2z0[:,0]) , axis=(1))
#
#            if(_flag_doEoRNL):
#                self._deltaxi_Txxi[izp1] += coeffzp2Tx * np.sum(coeffR2Tx * ((np.exp(gammaR2 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub])-1.0) - gammaR2 * gammaeffxHI * corr_deltaR1R2z0[:,_indexRbub]) , axis=(1))
#
#
#        self._deltaxi_xaTx[izp1]*= coeffzp1xa
#        self._deltaxi_xaTx[izp1]*=_coeffTx_units[izp1]
#
#        if(User_Parameters.FLAG_DO_DENS_NL):
#            self._deltaxi_dTx[izp1]*=_coeffTx_units[izp1]
#
#        if(_flag_doEoRNL):
#            self._deltaxi_Txxi[izp1]*=_coeffTx_units[izp1] * _eminusQstar
#
#
#
#    self._deltaxi_Tx=(self._deltaxi_Tx.T*_coeffTx_units**2).T #we cannot easily do this in the loop because it sums over previous ones
#
#    return 1
#
#     def calculate_barrier(self, Cosmo_Parameters, T21_coefficients):
#         "Caclulate the barrier B(z, sigmaR) that the density \delta has to cross to ionize"
#
#         self.Barrier0list = np.zeros_like(T21_coefficients.zintegral)
#         self.Barrier1list = np.zeros_like(T21_coefficients.zintegral)
#
#         sigmaminsqlist = (T21_coefficients.sigmaMatom * self._lingrowthd/self._lingrowthd[0])**2
#         sigmapivotsqlist = (T21_coefficients.sigmaMpivot * self._lingrowthd/self._lingrowthd[0])**2
#         #notice sigmaMatom depends on z and sigmaMpivot doesn't. For now at least. Code doesn't care since _lingrowthd does depend on z anyway
#
#
#         sigmaRref = np.sqrt(sigmaminsqlist/20.)
#         #pick this one for reference to take d/dsigmaR^2
#
#         alphaeff = T21_coefficients._alphaeff #note that if alpha_eff = 0 you recover erfc. For negative it can behave weird so beware (for instance voids reionize first. Not physical)
#
#         plindex = -T21_coefficients.dlogMdlogsigma
#         #M~sigma^-plindex
#
#         totalindex = plindex * alphaeff
#         sindex = 1./2. + totalindex
#
#         for izp, zp in enumerate(T21_coefficients.zintegral):
#
#             if zp>constants.ZMAX_Bubbles:
#                 continue
#
#             _invQbar = 1.0/T21_coefficients.Qion_avg[izp] #we need Nion/<Nion> > 1/invQbar to ionize the region. larger delta at higher z
#
#
#
#             dtab = np.linspace(-3.0 * sigmaRref[izp] , 3.0 * sigmaRref[izp] , 99)
#             dtabhi = np.linspace(3.3 * sigmaRref[izp], 1.5, 30)
#             dtab = np.append(dtab, dtabhi)
#
#             dtildetabsq = (constants.delta_crit_ST - dtab)**2
#
#
#             tabsigmasqit = [0.8*sigmaRref[izp]**2, 1.4*sigmaRref[izp]**2] #to get derivatives wrt sigma^2
#
#             barrier = np.zeros_like(tabsigmasqit)
#
#             for isigma, sigmaRRsq in enumerate(tabsigmasqit):
#
#                 mumintildesq = dtildetabsq/(sigmaminsqlist[izp] - sigmaRRsq)
#                 mupivottildesq = dtildetabsq/sigmapivotsqlist[izp]
#
#
#                 NionEPS = pow(dtildetabsq, - totalindex) * (gammaincc(sindex,mumintildesq/2.0) - gammaincc(sindex,mupivottildesq/2.0))
#
#                 Probdtab = np.exp(-dtab**2/sigmaRRsq/2.0)
#
#                 norm = np.trapezoid(NionEPS * Probdtab, dtab)
#                 NionEPS/=norm
#
#                 bindex = min(range(len(NionEPS)), key=lambda i: abs(NionEPS[i]-_invQbar))
#
#                 barrier[isigma] = dtab[bindex]
#
#             self.Barrier0list[izp] = np.sum(barrier)/len(barrier) #sigma-indep
#             self.Barrier1list[izp] = (barrier[-1] - barrier[0])/(tabsigmasqit[-1] - tabsigmasqit[0]) #linear in sigmaR^2
#
#     def get_bubbles(self, Cosmo_Parameters, Correlations, T21_coefficients):
#         "Returns the Bubble mass function for EoR"
#
#
#         _Rtab = T21_coefficients.Rtabsmoo
#         _rhob0 = cosmology.rho_baryon(Cosmo_Parameters, 0.)
#         _Mtab = _rhob0 * 4.0 * np.pi * _Rtab**3/3.0  #at z=0 because comoving
#         _dMdR = _rhob0 * 4.0 * np.pi * _Rtab**2 #at z=0 because comoving
#         _dlogMdlogR = 3.0
#         _dlog_Mtab = _dlogMdlogR * T21_coefficients.dlogRR
#
#
#         self.BMF_array = np.zeros_like(T21_coefficients.gamma_Niondot_index2D)
#         #bubble mass function, dn/dm in 1/cMpc^3/Msun
#
#         self.Qo_tab = np.zeros_like(self.BMF_array)
#         #Q_overlap, integral of [BMF * Voverlap(r)] at _Rtab
#         _Voverlap = np.array([[Voverlap(Rbb, rr) for Rbb in _Rtab] for rr in _Rtab])
#         #index is [ir, iRb]
#
#
#         self.Q_infer_BMF = np.zeros(T21_coefficients.Nzintegral)
#
#
#         self.Rbub_star = np.zeros(T21_coefficients.Nzintegral) #peak of BMF
#         self._Rbub_star_index = np.zeros(T21_coefficients.Nzintegral, dtype=int) #its index in Rsmoo
#         self.bias_bub_avg = np.zeros(T21_coefficients.Nzintegral) #avg (mass-weighted) bias
#
#
#         for izp, zp in enumerate(T21_coefficients.zintegral):
#
#             if (zp > constants.ZMAX_Bubbles or T21_coefficients.Qion_avg[izp] >= 1.0): #only do below a threshold and before EoR is complete to avoid numerical noise
#                 continue
#
#             sigmaofRtab = T21_coefficients.sigmaofRtab[izp]
#             logsigmaoflogR_f = UnivariateSpline(np.log(_Rtab),np.log(sigmaofRtab) )
#             dlogsigmadlogR_f = logsigmaoflogR_f.derivative()
#             dlogsigmadlogRtab = dlogsigmadlogR_f(np.log(_Rtab) )
#
#
#
#             B0 = self.Barrier0list[izp] #Fit is Barrier = B0 + B1 sigma^2
#             B1 = self.Barrier1list[izp]
#             Btab = B0 + B1 * sigmaofRtab**2
#
#             dlogsigmadlogMtab = dlogsigmadlogRtab / _dlogMdlogR
#
#             self.BMF_array[izp] = np.sqrt(2.0/np.pi) * _rhob0/(_Mtab**2) * np.abs(dlogsigmadlogMtab) * B0/sigmaofRtab * np.exp(-Btab**2/(2.0 * sigmaofRtab**2))
#
#
#             self.Q_infer_BMF[izp] = np.sum(self.BMF_array[izp] * _Mtab/_rhob0 * _Mtab)*_dlog_Mtab
#
#
#             self.BMF_array[izp] *= T21_coefficients.Qstar[izp]/self.Q_infer_BMF[izp] #renormalized now
#
#             self._bias_bubbles_zp = 1.0 + B0**2/(Btab * sigmaofRtab**2) #Eulerian bias of a bubble of some mass/radius at zp
#
#             self.bias_bub_avg[izp] = np.sum(self.BMF_array[izp] * self._bias_bubbles_zp * _Mtab/_rhob0 * _Mtab)*_dlog_Mtab/T21_coefficients.Qstar[izp]
#             #average bias
#
#
#             _dimlessBMF = _Mtab**2 * self.BMF_array[izp]
#             self._Rbub_star_index[izp] = max(range(len(_Mtab)), key=lambda i: _dimlessBMF[i])
#             self.Rbub_star[izp] = _Rtab[self._Rbub_star_index[izp]] #the maximum of the BMF
#
#
#             self.Qo_tab[izp] = np.array([np.sum(self.BMF_array[izp] * Vtab * _Mtab)*_dlog_Mtab for Vtab in _Voverlap])
#         self.Rbub_star = np.fmax(self.Rbub_star, 1e-3) #to avoid Nans in other functions
#
#
# def Voverlap(Rb, r):
#     "Overlapping volume of two bubbles of radius Rb separated by r. From FZH04"
#     return ((4 * np.pi/3.0) * Rb**3 - np.pi * r * (Rb**2 - r**2/12.)) * np.heaviside( 2*Rb - r , 0.5)
