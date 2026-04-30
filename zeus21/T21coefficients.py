"""

Bulk of the Zeus21 calculation. Determines Lyman-alpha and X-ray fluxes, and evolves the cosmic-dawn IGM state (WF coupling and heating). From that we get the 21-cm global signal and the effective biases gammaR to determine the 21-cm power spectrum.

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Emily Bregou
UT Austin - October 2025

Edited by Sarah Libanore, Emilie Thelie, Hector Afonso G. Cruz
BGU, UT Austin - April 2026 
"""

from . import cosmology
from . import constants

import numpy as np

from scipy import interpolate


from .sfrd import Z_init, SFRD_class, PopIII_relvel
from .reionization import reionization_global


class LyAlpha_class:

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None, SFRD_Init = None):

        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams) 

        if SFRD_Init is None:
            SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init) 

        self.coeff1LyAzp = (1+z_Init.zintegral)**2/(4*np.pi)

        nuLYA = np.geomspace(constants.freqLyA, constants.freqLyCont, 128)
        sedLYAII_interp = interpolate.interp1d(nuLYA, AstroParams.SED_LyA(nuLYA, pop = 2), kind = 'linear', bounds_error = False, fill_value = 0) #interpolate LyA SED

        n_recArray = np.arange(0,constants.n_max_recycle-1 )
        zpCube, rCube, n_recCube = np.meshgrid(z_Init.zintegral, CosmoParams._Rtabsmoo, n_recArray, indexing='ij', sparse=True) #for broadcasting purposes
        n_lineCube = n_recCube + 2
        zmax_lineCube = (1+zpCube) * (1 - pow(1+n_lineCube,-2.0))/(1-pow(n_lineCube,-2.0) ) - 1.0 #maximum redshift Lyman series photons can redshift before falling into a Ly-n resonance

        nu_linezpCube = constants.freqLyCont * (1 - (1.0/n_lineCube)**2)
        zGreaterCube = z_Init.zGreaterMatrix_nonan.reshape(len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1)
        nu_lineRRCube = nu_linezpCube * (1.+zGreaterCube)/(1+zpCube)
        
        eps_alphaRR_II_Cube = AstroParams.N_alpha_perbaryon_II/CosmoParams.mu_baryon_Msun  * sedLYAII_interp(nu_lineRRCube)
        
        #the last nonzero index of the array is overestimated since only part of the spherical shell is within zmax_line. Correct by by dz/Delta z
        weights_recCube = np.heaviside(zmax_lineCube - zGreaterCube, 0.0)
        index_first0_weightsCube = np.where(np.diff(weights_recCube, axis = 1) == -1) #find index of last nonzero value. equals zero if two consecutive elements are 1 or 0, and -1 if two consecutive elements are [1,0]
        i0Z, i0R, i0N = index_first0_weightsCube
        weights_recCube[i0Z, i0R, i0N] *= (zmax_lineCube[i0Z, 0, i0N] - zGreaterCube[i0Z, i0R, 0])/ (zGreaterCube[i0Z, i0R+1, 0] - zGreaterCube[i0Z, i0R, 0])

        Jalpha_II = np.array(constants.fractions_recycle)[:len(n_recArray)].reshape(1,1,len(n_recArray)) * weights_recCube * eps_alphaRR_II_Cube #just resizing f_recycle; it is length 29,we only consider up to n=22
        LyAintegral_II = np.sum(Jalpha_II,axis=2) #sum over axis 2, over all possible n transitions
        self.coeff2LyAzpRR_II = CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_II * LyAintegral_II/ constants.yrTos/constants.Mpctocm**2

        if AstroParams.USE_POPIII:
            sedLYAIII_interp = interpolate.interp1d(nuLYA, AstroParams.SED_LyA(nuLYA, pop = 3), kind = 'linear', bounds_error = False, fill_value = 0)
            eps_alphaRR_III_Cube = AstroParams.N_alpha_perbaryon_III/CosmoParams.mu_baryon_Msun  * sedLYAIII_interp(nu_lineRRCube)
            
            Jalpha_III = np.array(constants.fractions_recycle)[:len(n_recArray)].reshape(1,1,len(n_recArray)) * weights_recCube * eps_alphaRR_III_Cube
            LyAintegral_III = np.sum(Jalpha_III,axis=2)
            self.coeff2LyAzpRR_III = CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_III * LyAintegral_III/ constants.yrTos/constants.Mpctocm**2
        else:
            self.coeff2LyAzpRR_III = np.zeros_like(self.coeff2LyAzpRR_II)

        # Non-Linear Correction Factors
        # Correct for nonlinearities in <(1+d)SFRD>, only if doing nonlinear stuff. 
        # We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. 
        # We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).
        if UserParams.C2_RENORMALIZATION_FLAG:
            self.coeff2LyAzpRR_II = self.coeff2LyAzpRR_II* SFRD_Init._corrfactorEulerian_II.T
            if AstroParams.USE_POPIII:
                self.coeff2LyAzpRR_III = self.coeff2LyAzpRR_III * SFRD_Init._corrfactorEulerian_III.T
        

class Xrays_class:

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None, SFRD_Init = None):

        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams) 

        if SFRD_Init is None:
            SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init) 

        self.atomfractions = np.array([1,CosmoParams.x_He]) #fraction of baryons in HI and HeI, assumed to just be the avg cosmic
        self.atomEnIon = np.array([constants.EN_ION_HI, constants.EN_ION_HeI]) #threshold energies for each, in eV
        self.TAUMAX=100. #max optical depth, cut to 0 after to avoid overflows

        _Energylist = AstroParams.Energylist
        Nzinttau = np.floor(10*UserParams.precisionboost).astype(int)

        zGreaterCube = z_Init.zGreaterMatrix_nonan.reshape(len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1, 1) #redefine this just for x-ray routine

        self.coeff1Xzp = -2/3 * z_Init.zintegral * z_Init.dlogzint / cosmology.Hubinvyr(CosmoParams,z_Init.zintegral) / (1+z_Init.zintegral) * (1+z_Init.zintegral)**2
        self.coeff1Xzp = self.coeff1Xzp / (1+z_Init.zintegral)**2 * constants.yrTos #this accounts for adiabatic cooling. compensated by the inverse at the end

        zpCube, rCube, eCube, zPPCube = np.meshgrid(z_Init.zintegral, CosmoParams._Rtabsmoo, _Energylist, np.arange(Nzinttau), indexing='ij', sparse=True)
        currentEnergyTable = eCube * (1+zGreaterCube) / (1+zpCube)
        SEDCube = AstroParams.SED_XRAY(currentEnergyTable, pop = 2)
        SEDCube_III = AstroParams.SED_XRAY(currentEnergyTable, pop = 3)

        ######## Broadcasted routine to find X-ray optical depths, modeled after but does not use xrays.optical_depth
        zPPCube = np.array([np.linspace(np.transpose([z_Init.zintegral]), z_Init.zGreaterMatrix, Nzinttau, axis = 2)])
        zPPCube = zPPCube.reshape(len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1, Nzinttau) #to have 4D dimensions, default shape = (64,45, 1, 10)

        ePPCube = eCube * (1+ zPPCube) / (1+zpCube) #E'' = E(1+z'')/(1+z)
        sigmatot = self.atomfractions[0] * self.sigma_HI(ePPCube)
        sigmatot += self.atomfractions[1] * self.sigma_HeI(ePPCube)

        opticalDepthIntegrand = 1 / cosmology.HubinvMpc(CosmoParams, zPPCube) / (1+zPPCube) * sigmatot * cosmology.n_H(CosmoParams, zPPCube) * constants.Mpctocm #this uses atom fractions of 1 for HI and x_He for HeI
        tauCube = np.trapezoid(opticalDepthIntegrand, zPPCube, axis = 3)

        indextautoolarge = np.array(tauCube>=self.TAUMAX)
        tauCube[indextautoolarge] = self.TAUMAX

        if CosmoParams.Flag_emulate_21cmfast:
            weights_X_zCube = np.heaviside(1.0 - tauCube, 0.5)
        else:
            weights_X_zCube = np.exp(-tauCube)
            
        SEDCube = SEDCube[:,:,:,0] #rescale dimensions of energy and SED cubes back to 3D, so we can integrate over energy
        SEDCube_III = SEDCube_III[:,:,:,0] #rescale dimensions of energy and SED cubes back to 3D, so we can integrate over energy
        
        eCube = eCube[:,:,:,0]
        ######## end of optical depth routine

        JX_coeffsCube = SEDCube * weights_X_zCube
        JX_coeffsCube_III = SEDCube_III * weights_X_zCube

        sigma_times_en = self.atomfractions[0] * self.sigma_HI(eCube) * (eCube - self.atomEnIon[0])
        sigma_times_en += self.atomfractions[1] * self.sigma_HeI(eCube) * (eCube - self.atomEnIon[1])
        sigma_times_en /= np.sum(self.atomfractions)#to normalize per baryon, instead of per Hydrogen nucleus
                #HI and HeII separate. Notice Energy (and not Energy'), since they get absorbed at the zp frame
        
        xrayEnergyTable = np.sum(JX_coeffsCube * sigma_times_en * eCube * AstroParams.dlogEnergy,axis=2)
        self.coeff2XzpRR_II = np.nan_to_num(CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_II * xrayEnergyTable * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST, nan = 0)
        
        if AstroParams.USE_POPIII:
            xrayEnergyTable_III = np.sum(JX_coeffsCube_III * sigma_times_en * eCube * AstroParams.dlogEnergy,axis=2)
            self.coeff2XzpRR_III = np.nan_to_num(CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_III * xrayEnergyTable_III * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST, nan = 0)
        else:
            self.coeff2XzpRR_III = np.zeros_like(self.coeff2XzpRR_II)

        # Non-Linear Correction Factors
        # Correct for nonlinearities in <(1+d)SFRD>, only if doing nonlinear stuff. 
        # We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. 
        # We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).
        if UserParams.C2_RENORMALIZATION_FLAG:
            self.coeff2XzpRR_II = self.coeff2XzpRR_II* SFRD_Init._corrfactorEulerian_II.T
            if AstroParams.USE_POPIII:
                self.coeff2XzpRR_III = self.coeff2XzpRR_III * SFRD_Init._corrfactorEulerian_III.T

        self._GammaXray_II = self.coeff1Xzp * np.sum( self.coeff2XzpRR_II ,axis=1) #notice units are modified (eg 1/H) so it's simplest to sum
        self._GammaXray_III = self.coeff1Xzp * np.sum( self.coeff2XzpRR_III ,axis=1) #notice units are modified (eg 1/H) so it's simplest to sum
        
        fion = 0.4 * np.exp(-cosmology.xefid(CosmoParams, z_Init.zintegral)/0.2)#partial ionization from Xrays. Fit to Furlanetto&Stoever
        atomEnIonavg = (self.atomfractions[0] *  self.atomEnIon[0] + self.atomfractions[1] *  self.atomEnIon[1]) / (self.atomfractions[0] + self.atomfractions[1] ) #to turn this ratio into one over n_b instead of n_H
        
        self.coeff_Gammah_Tx_II = -AstroParams.L40_xray * constants.ergToK * (1.0+z_Init.zintegral)**2
        self.coeff_Gammah_Tx_III = -AstroParams.L40_xray_III * constants.ergToK * (1.0+z_Init.zintegral)**2 #convert from one to the other, last factors accounts for adiabatic cooling. compensated by the inverse at zp in coeff1Xzp. Minus because integral goes from low to high z, but we'll be summing from high to low everywhere.
        
        self.Gammaion_II = self.coeff_Gammah_Tx_II *constants.KtoeV * self._GammaXray_II * fion/atomEnIonavg * 3/2
        self.Gammaion_III = self.coeff_Gammah_Tx_III *constants.KtoeV * self._GammaXray_III * fion/atomEnIonavg * 3/2 #atomEnIonavg makes it approximate. No adiabatic cooling (or recombinations) so no 1+z factors. Extra 3/2 bc temperature has a 2/3
        
        #TODO: Improve model for xe

        self.xe_avg_ad = cosmology.xefid(CosmoParams, z_Init.zintegral)
        self.xe_avg = self.xe_avg_ad + np.cumsum((self.Gammaion_II+self.Gammaion_III)[::-1])[::-1]
        if CosmoParams.Flag_emulate_21cmfast:
            self.xe_avg = 2e-4 * np.ones_like(self.Gammaion_II) #we force this when we emualte 21cmdast to compare both codes on the same footing
        self.xe_avg = np.fmin(self.xe_avg, 1.0-1e-9)

        #and heat from Xrays
        self._fheat = pow(self.xe_avg,0.225)
        self.coeff1Xzp*=self._fheat #since this is what we use for the power spectrum (and not Gammaheat) we need to upate it
        self.Gammaheat_II = self._GammaXray_II * self._fheat
        self.Gammaheat_III = self._GammaXray_III * self._fheat

        #Computing avg kinetic temperature as sum of adiabatic & xray temperature
        self.Tk_xray = self.coeff_Gammah_Tx_II * np.cumsum(self.Gammaheat_II[::-1])[::-1] + self.coeff_Gammah_Tx_III * np.cumsum(self.Gammaheat_III[::-1])[::-1]#in K, cumsum reversed because integral goes from high to low z. Only heating part
        self.Tk_ad = cosmology.Tadiabatic(CosmoParams, z_Init.zintegral)
        if CosmoParams.Flag_emulate_21cmfast:
            self.Tk_ad*=0.95 #they use recfast, so their 'cosmo' temperature is slightly off
        self.Tk_avg = self.Tk_ad + self.Tk_xray
        

    def sigma_HI(self, Energyin):
        "cross section for Xray absorption for neutral HI, from astro-ph/9601009 and takes Energy in eV and returns cross sec in cm^2"
        E0 = 4.298e-1
        sigma0 =  5.475e4
        ya = 3.288e1
        P =  2.963
        yw =  0.0
        y0 =  0.0
        y1 = 0.0

        Energy = Energyin

        warning_lowE_HIXray = np.heaviside(13.6 - Energy, 0.5)
        if(np.sum(warning_lowE_HIXray) > 0):
            print('ERROR! Some energies for Xrays below HI threshold in sigma_HI. Too low!')


        x = Energy/E0 - y0
        y = np.sqrt(x**2 + y1**2)
        Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0+np.sqrt(y/ya))**(-P)

        return sigma0 * constants.sigma0norm * Fy



    def sigma_HeI(self, Energyin):
        "same as sigma_HI but for HeI, parameters are:"
        E0 = 13.61
        sigma0 = 9.492e2
        ya = 1.469
        P =  3.188
        yw =  2.039
        y0 =  4.434e-1
        y1 = 2.136

        Energy = Energyin
        warning_lowE_HeIXray = np.heaviside(25. - Energy, 0.5)
        if(np.sum(warning_lowE_HeIXray) > 0):
            print('ERROR! Some energies for Xrays below HeI threshold in sigma_HeI. Too low!')


        x = Energy/E0 - y0
        y = np.sqrt(x**2 + y1**2)
        Fy = ((x-1.0)**2 + yw**2) * y**(0.5*P - 5.5) * (1.0+np.sqrt(y/ya))**(-P)

        return sigma0 * constants.sigma0norm * Fy




class get_T21_coefficients:
    "Loops through SFRD integrals and obtains avg T21 and the coefficients for its power spectrum. Takes input zmin, which minimum z we integrate down to. It accounts for: \
    -Xray heating \
    -LyA coupling. \
    TODO: reionization/EoR"

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp):
        #####################################################################################################
        ### Initialize redshift tables
        self.z_Init = Z_init(UserParams, CosmoParams)


        #####################################################################################################
        ### Initialize and compute the SFRD approximation
        # With recursive routine to compute average Pop II and III SFRDs with LW feedback
        # Will only perform 1 iteration; if Astro_Parameters.USE_LW_FEEDBACK = False, then inputs.py sets A_LW = 0.0
        # With broadcasted prescription to Compute gammas
        # Including LW correction to Pop III gammas
        self.SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init) 
        

        #####################################################################################################
        ### Computing lambdas in velocity anisotropies
        # Because we found the SFRD vcb dependence to be delta independent, we compute quantities below for a variety of R's and delta_R = 0
        self.USE_POPIII = AstroParams.USE_POPIII
        if self.USE_POPIII:
            self.relvel = PopIII_relvel(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init)
            ### TODO to debug: compare the output with old version
        else:
            self.relvel = None


        #####################################################################################################
        ### Lyman-Alpha Anisotropies
        # Makes heavy use of broadcasting to make computations faster
        # 3D cube will be summed over one axis. Dimensions are (z,R,n) = (64, 45, 21)
        self.LyA = LyAlpha_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init)


        #####################################################################################################
        ### X-ray Anisotropies
        self.Xrays = Xrays_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init)

        
        #####################################################################################################
        ### Computing free-electron fraction and Salpha correction factors in the Bulk IGM  
        self.evolve_T21_fields(UserParams, CosmoParams)   

        
        #####################################################################################################
        ### Reionization
        self.ReioGlobal = reionization_global(CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init, PRINT_SUCCESS=False)
        self.xHI_avg = 1. - self.ReioGlobal.ion_frac ### TODO this one is volume weighted for now, maybe need to be rethought

        #####################################################################################################
        ### Compute the 21cm Global Signal
        self.T21avg = cosmology.T021(CosmoParams,self.z_Init.zintegral) * self.xa_avg/(1.0 + self.xa_avg) * (1.0 - self.T_CMB * self.invTcol_avg) * self.xHI_avg

        self.tau_reio_val = self.tau_reio(CosmoParams, self.z_Init.zintegral, self.xHI_avg)


    def __getattr__(self, name):
        list_of_cls = [self.z_Init, self.SFRD_Init, self.LyA, self.Xrays, self.ReioGlobal]
        if self.USE_POPIII:
            list_of_cls += [self.relvel]
        for cls in list_of_cls:
            try:
                return getattr(cls, name)
            except AttributeError:
                pass
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    #def __setattr__(self, name, value):
    #    list_of_cls = [self.z_Init, self.SFRD_Init, self.LyA, self.Xrays]
    #    if self.USE_POPIII:
    #        list_of_cls += [self.relvel]
    #
    #    for cls in list_of_cls:
    #        if hasattr(cls, name):
    #            setattr(cls, name, value)
    #            return
    #
    #    # If the attribute does not belong to any child, set it on the Parent
    #    object.__setattr__(self, name, value) ### TODO debug? remove?



    def evolve_T21_fields(self, UserParams, CosmoParams):   
        # LyA stuff to find components of Salpha correction factor
        self.Jalpha_avg = self.LyA.coeff1LyAzp*np.sum(self.LyA.coeff2LyAzpRR_II + self.LyA.coeff2LyAzpRR_III,axis=1) #units of 1/(cm^2 s Hz sr)
        self.T_CMB = cosmology.Tcmb(CosmoParams.ClassCosmo, self.z_Init.zintegral)

        _tau_GP = 3./2. * cosmology.n_H(CosmoParams,self.z_Init.zintegral) * constants.Mpctocm / cosmology.HubinvMpc(CosmoParams,self.z_Init.zintegral) * (constants.wavelengthLyA/1e7)**3 * constants.widthLyAcm * (1.0 - self.Xrays.xe_avg)  #~3e5 at z=6

        if CosmoParams.Flag_emulate_21cmfast:
            _tau_GP/=CosmoParams.f_H #for some reason they multiuply by N0 (all baryons) and not NH0.

        _xiHirata = pow(_tau_GP*1e-7,1/3.)*pow(self.Xrays.Tk_avg,-2./3)
        _factorxi = (1.0 + constants.a_Hirata*_xiHirata + constants.b_Hirata * _xiHirata**2 + constants.c_Hirata * _xiHirata**3) 


        #prefactor without the Salpha correction from Hirata2006
        if CosmoParams.Flag_emulate_21cmfast:
            self._coeff_Ja_xa_0 = 1.66e11/(1+self.z_Init.zintegral) #They use a fixed (and slightly ~10% off) value.
        else:
            self._coeff_Ja_xa_0 = 8.0*np.pi*(constants.wavelengthLyA/1e7)**2 * constants.widthLyA * constants.Tstar_21/(9.0*constants.A10_21*self.T_CMB) #units of (cm^2 s Hz sr), convert from Ja to xa. should give 1.81e11/(1+z_Init.zintegral) for Tcmb_0=2.725 K

        self.coeff_Ja_xa = self._coeff_Ja_xa_0 * self.Salpha_exp(self.z_Init.zintegral, self.Xrays.Tk_avg, self.Xrays.xe_avg)
        self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg
        self.invTcol_avg = 1.0 / self.Xrays.Tk_avg
        self._invTs_avg = (1.0/self.T_CMB+self.xa_avg*self.invTcol_avg)/(1+self.xa_avg)
        if UserParams.FLAG_WF_ITERATIVE: #iteratively find Tcolor and Ts. Could initialize one to zero, but this should converge faster
            ### iteration routine to find Tcolor and Ts
            _invTs_tryfirst = 1.0/self.T_CMB
            while(np.sum(np.fabs(_invTs_tryfirst/self._invTs_avg - 1.0))>0.01): #no more than 1% error total
                _invTs_tryfirst = self._invTs_avg

                #update xalpha
                _Salphatilde = (1.0 - 0.0632/self.Xrays.Tk_avg + 0.116/self.Xrays.Tk_avg**2 - 0.401/self.Xrays.Tk_avg*self._invTs_avg + 0.336*self._invTs_avg/self.Xrays.Tk_avg**2)/_factorxi
                self.coeff_Ja_xa = self._coeff_Ja_xa_0 * _Salphatilde
                self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg

                #and Tcolor^-1
                self.invTcol_avg = 1.0/self.Xrays.Tk_avg + constants.gcolorfactorHirata * 1.0/self.Xrays.Tk_avg * (_invTs_tryfirst - 1.0/self.Xrays.Tk_avg)

                #and finally Ts^-1
                self._invTs_avg = (1.0/self.T_CMB+self.xa_avg * self.invTcol_avg)/(1+self.xa_avg)


        
    def tau_reio(self, CosmoParams, zlist, xHI):
        "Returns the optical depth to reionization given a neutral frac xHI as a func of zlist"
        #assume HeII at z=4, can be varied with zHeIIreio

        #first integrate for z<zmin, assumed xHI=0
        _lowestz = np.min(zlist)
        _zlistlowz = np.linspace(0,_lowestz,100)
        _nelistlowz = cosmology.n_H(CosmoParams,_zlistlowz)*(1 + CosmoParams.x_He + CosmoParams.x_He * np.heaviside(constants.zHeIIreio - _zlistlowz,0.5))
        _distlistlowz = 1.0/cosmology.HubinvMpc(CosmoParams,_zlistlowz)/(1+_zlistlowz)
        _lowzint = constants.sigmaT * np.trapezoid(_nelistlowz*_distlistlowz,_zlistlowz) * constants.Mpctocm

        #now the rest of integral
        xHIint = np.fmin(np.fmax(xHI,0.0),1.0) #at min 0%, at max 100%
        _zlisthiz = zlist
        _nelistlhiz = cosmology.n_H(CosmoParams,_zlisthiz) * (1 + CosmoParams.x_He) * (1.0 - xHIint)
        _distlisthiz = 1.0/cosmology.HubinvMpc(CosmoParams,_zlisthiz)/(1+_zlisthiz)

        _hizint = constants.sigmaT * np.trapezoid(_nelistlhiz*_distlisthiz,_zlisthiz) * constants.Mpctocm

        return(_lowzint + _hizint)

    #Kept for reference purposes. Does not correct x_alpha as a function of Ts iteratively, but some old works don't either so this allows for comparison. Only used if FLAG_WF_ITERATIVE == False
    def Salpha_exp(self, z, T, xe):
        "correction from Eq 55 in astro-ph/0608032, Tk in K evaluated for the IGM where there is small reionization (xHI~1 and xe<<1) during LyA coupling era"
        tau_GP_noreio = 3e5*pow((1+z)/7,3./2.)*(1-xe)
        gamma_Sobolev = 1.0/tau_GP_noreio
        return np.exp( - 0.803 * pow(T,-2./3.) * pow(1e-6/gamma_Sobolev,-1.0/3.0))
