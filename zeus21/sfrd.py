"""

Bulk of the Zeus21 calculation. Compute sSFRD from cosmology.

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
import astropy
from astropy import units as u

import scipy
from scipy import interpolate


class Z_init:

    def __init__(self, UserParams, CosmoParams):

        zmax_integral = constants.ZMAX_INTEGRAL
        zmin_integral = UserParams.zmin_T21
        
        Nzintegral = np.ceil(1.0 + np.log(zmax_integral/zmin_integral)/UserParams.dlogzint_target).astype(int)
        
        self.dlogzint = np.log(zmax_integral/zmin_integral)/(Nzintegral-1.0) #exact value rather than input target above
        self.zintegral = np.logspace(np.log10(zmin_integral), np.log10(zmax_integral), Nzintegral) #note these are also the z at which we "observe", to share computational load
        
        #define table of redshifts 
        rGreaterMatrix = np.transpose([CosmoParams.chiofzint(self.zintegral)]) + CosmoParams._Rtabsmoo
        self.zGreaterMatrix = CosmoParams.zfofRint(rGreaterMatrix)

        if CosmoParams.Flag_emulate_21cmfast: #they take the redshift to be at the midpoint of the two shells. In dr really.
            # HECTOR CHANGES
            self.zGreaterMatrix = np.append(self.zintegral.reshape(len(self.zGreaterMatrix), 1), self.zGreaterMatrix, axis = 1)
            self.zGreaterMatrix = (self.zGreaterMatrix[:, 1:] + self.zGreaterMatrix[:, :-1])/2              
        else:
            self.zGreaterMatrix[rGreaterMatrix > CosmoParams.chiofzint(constants.zmax_AstroBreak)] = np.nan
            
        self.zGreaterMatrix_nonan = np.nan_to_num(self.zGreaterMatrix, nan = 100)


class SFRD_class:

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None):

        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams)  

        ### Will only perform 1 iteration; if AstroParams.USE_LW_FEEDBACK = False, then inputs.py sets A_LW = 0.0
        zSFRDflat = np.geomspace(UserParams.zmin_T21, constants.zmax_AstroBreak, 128) #extend to z = constants.zmax_AstroBreak for extrapolation purposes. Higher in z than zInit.zintegral
        zSFRD, mArray = np.meshgrid(zSFRDflat, HMFinterp.Mhtab, indexing = 'ij', sparse = True)

        init_J21LW_interp = interpolate.interp1d(zSFRDflat, np.zeros_like(zSFRDflat), kind = 'linear', bounds_error = False, fill_value = 0,) #no LW background. Controls only Mmol() function, NOT the individual Pop II and III LW background

        SFRD_II_avg = np.trapezoid(self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=2), HMFinterp.logtabMh, axis = 1) #never changes with J_LW
        self.SFRD_II_interp = interpolate.interp1d(zSFRDflat, SFRD_II_avg, kind = 'cubic', bounds_error = False, fill_value = 0,)

        J21LW_II = self.J_LW_21(CosmoParams, AstroParams, SFRD_II_avg, zSFRDflat, pop=2) #this never changes; only Pop III Quanties change
        self.J_21_LW_II = interpolate.interp1d(zSFRDflat, J21LW_II, kind = 'cubic')(z_Init.zintegral) #different from J21LW_interp

        if AstroParams.USE_POPIII:

            SFRD_III_Iter_Matrix = [np.trapezoid(self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=3, vCB=CosmoParams.vcb_avg, J21LW_interp=init_J21LW_interp), HMFinterp.logtabMh, axis = 1)] #changes with each iteration

            errorTolerance = 0.001 # 0.1 percent accuracy
            recur_iterate_Flag = True
            while recur_iterate_Flag:
                J21LW_III_iter = self.J_LW_21(CosmoParams, AstroParams, SFRD_III_Iter_Matrix[-1], zSFRDflat, pop=3)
                loop_J21LW_interp = interpolate.interp1d(zSFRDflat, J21LW_II + J21LW_III_iter, kind = 'linear', fill_value = 0, bounds_error = False)

                SFRD_III_avg_n = np.trapezoid(self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=3, vCB=CosmoParams.vcb_avg, J21LW_interp= loop_J21LW_interp), HMFinterp.logtabMh, axis = 1)
                SFRD_III_Iter_Matrix.append(SFRD_III_avg_n)

                if max(SFRD_III_Iter_Matrix[-1]/SFRD_III_Iter_Matrix[-2]) < 1.0 + errorTolerance and min(SFRD_III_Iter_Matrix[-1]/SFRD_III_Iter_Matrix[-2]) > 1.0 - errorTolerance:
                    recur_iterate_Flag = False

            self.J21LW_interp_conv_avg = loop_J21LW_interp

            self.SFRD_III_cnvg_interp = interpolate.interp1d(zSFRDflat, SFRD_III_Iter_Matrix[-1], kind = 'cubic', bounds_error = False, fill_value = 0)
            self.J_21_LW_III = interpolate.interp1d(zSFRDflat, J21LW_III_iter, kind = 'cubic')(z_Init.zintegral)

        else:

            self.SFRD_III_cnvg_interp = interpolate.interp1d(zSFRDflat, np.zeros_like(zSFRDflat), kind = 'cubic', bounds_error = False, fill_value = 0)

        self.SFRD_II_avg = self.SFRD_II_interp(z_Init.zintegral)
        self.SFRD_III_avg = self.SFRD_III_cnvg_interp(z_Init.zintegral)
        self.SFRD_avg = self.SFRD_II_avg + self.SFRD_III_avg

        self.SFRDbar2D_II = self.SFRD_II_interp(np.nan_to_num(z_Init.zGreaterMatrix, nan = 100))
        
        self.SFRDbar2D_III = self.SFRD_III_cnvg_interp(np.nan_to_num(z_Init.zGreaterMatrix, nan = 100))
            
        # Reionization
        self.fesctab_II = self.fesc_II(AstroParams, HMFinterp.Mhtab) #prepare fesc(M) table -- z independent for now so only once
        self.fesctab_III = self.fesc_III(AstroParams, HMFinterp.Mhtab) #PopIII prepare fesc(M) table -- z independent for now so only once
        reio_integrand_II = self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=2)
        reio_integrand_III = self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=3)
        niondot_avg_II = AstroParams.N_ion_perbaryon_II/cosmology.rho_baryon(CosmoParams,0.) * np.trapezoid(reio_integrand_II * self.fesctab_II, HMFinterp.logtabMh, axis = 1)
        niondot_avg_III = AstroParams.N_ion_perbaryon_III/cosmology.rho_baryon(CosmoParams,0.) * np.trapezoid(reio_integrand_III * self.fesctab_III, HMFinterp.logtabMh, axis = 1)
        self.reio_integrand_II_interp = interpolate.interp1d(zSFRDflat, niondot_avg_II, kind = 'cubic', bounds_error = False, fill_value = 0)
        self.reio_integrand_III_interp = interpolate.interp1d(zSFRDflat, niondot_avg_III, kind = 'cubic', bounds_error = False, fill_value = 0)
        self.niondot_avg_II = self.reio_integrand_II_interp(z_Init.zintegral)
        self.niondot_avg_III = self.reio_integrand_III_interp(z_Init.zintegral)
        self.niondot_avg = self.niondot_avg_II + self.niondot_avg_III

        if not UserParams.DO_ONLY_GLOBAL:

            self.sigmaofRtab = np.array([HMFinterp.sigmaR_int(CosmoParams._Rtabsmoo, zz) for zz in z_Init.zintegral]) #to be used in correlations.py, in get_bubbles()

            self.compute_gamma(CosmoParams, AstroParams, HMFinterp, z_Init.zintegral, CosmoParams._Rtabsmoo, HMFinterp.Mhtab, self.sigmaofRtab, self.fesctab_II)
            

    #fstar = Mstardot/Mhdot, parametrizes as you wish
    def fstarofz_II(self, CosmoParams, AstroParams, z, Mhlist):
        eps = AstroParams.epsstar
        dlog10eps = AstroParams.dlog10epsstardz
        zpiv = AstroParams._zpivot
        Mc = AstroParams.Mc
        alphastar = AstroParams.alphastar
        betastar = AstroParams.betastar

        epsstar_ofz = eps * 10**(dlog10eps * (z-zpiv) )
        
        if CosmoParams.Flag_emulate_21cmfast:
            return CosmoParams.OmegaB/CosmoParams.OmegaM * np.clip(epsstar_ofz /(pow(Mhlist/Mc, -alphastar)), 0, AstroParams.fstarmax)

        else:
            return CosmoParams.OmegaB/CosmoParams.OmegaM * np.clip(2.0 * epsstar_ofz\
            /(pow(Mhlist/Mc,- alphastar) + pow(Mhlist/Mc,-betastar) ), 0, AstroParams.fstarmax)


    # popIII fstar = Mstardot/Mhdot, parametrizes as you wish
    def fstarofz_III(self, CosmoParams, AstroParams, z, Mhlist):

        eps = AstroParams.fstar_III
        dlog10eps = AstroParams.dlog10epsstardz_III
        zpiv = AstroParams._zpivot_III
        Mc = AstroParams.Mc_III
        alphastar = AstroParams.alphastar_III 
        betastar = AstroParams.betastar_III

        epsstar_ofz = eps * 10**(dlog10eps * (z-zpiv) )
        
        if CosmoParams.Flag_emulate_21cmfast:
            return CosmoParams.OmegaB/CosmoParams.OmegaM * np.clip(epsstar_ofz /(pow(Mhlist/Mc, -alphastar)), 0, AstroParams.fstarmax)

        else:
            return CosmoParams.OmegaB/CosmoParams.OmegaM * np.clip(2.0 * epsstar_ofz\
            /(pow(Mhlist/Mc,- alphastar) + pow(Mhlist/Mc,-betastar) ), 0, AstroParams.fstarmax)

    def Matom(self, z):
        "Returns Matom as a function of z"
        return 3.3e7 * pow((1.+z)/(21.),-3./2)

    ###HAC: Added Mmol split by contributions with no, vcb, and LW feecback
    def Mmol_0(self, z):
        "Returns Mmol as a function of z WITHOUT LW or VCB feedback"
        return 3.3e7 * (1.+z)**(-1.5)

    def Mmol_vcb(self, CosmoParams, AstroParams, z, vCB):
        "Returns Mmol as a function of z WITHOUT LW feedback"
        mmolBase = self.Mmol_0(z)
        vcbFeedback = pow(1 + AstroParams.A_vcb * vCB / CosmoParams.sigma_vcb, AstroParams.beta_vcb)
        return mmolBase * vcbFeedback

    def Mmol_LW(self, AstroParams, J21LW_interp, z):
        "Returns Mmol as a function of z WITHOUT VCB feedback"
        mmolBase = self.Mmol_0(z)
        lwFeedback = 1 + AstroParams.A_LW*pow(J21LW_interp(z), AstroParams.beta_LW)
        return mmolBase * lwFeedback
        
    def Mmol(self, CosmoParams, AstroParams, J21LW_interp, z, vCB):
        "Returns Mmol as a function of z WITH LW AND VCB feedback"
        mmolBase = self.Mmol_0(z)
        vcbFeedback = pow(1 + AstroParams.A_vcb * vCB / CosmoParams.sigma_vcb, AstroParams.beta_vcb)
        lwFeedback = 1 + AstroParams.A_LW*pow(J21LW_interp(z), AstroParams.beta_LW)
        
        return mmolBase * vcbFeedback * lwFeedback


    def fduty(self, CosmoParams, AstroParams, massVector, z, pop, vCB, J21LW_interp):

        if pop == 2:
            #The FIXED/SHARP routine below only applies to Pop II, not to Pop III
            if AstroParams.USE_POPIII:
                fduty = np.exp(-self.Matom(z)/massVector) 

            else:

                if not AstroParams.FLAG_MTURN_FIXED:
                    fduty = np.exp(-self.Matom(z)/massVector) 
                elif not AstroParams.FLAG_MTURN_SHARP: #whether to do regular exponential turn off or a sharp one at Mturn
                    fduty = np.exp(-AstroParams.Mturn_fixed/massVector)
                else:
                    fduty = np.heaviside(massVector - AstroParams.Mturn_fixed, 0.5)


        elif pop == 3:

            duty_matom_component = np.exp(-massVector/self.Matom(z)) 

            fduty =  np.exp(-self.Mmol(CosmoParams, AstroParams, J21LW_interp, z, vCB)/massVector) * duty_matom_component 

        return fduty


    def dMh_dt(self, CosmoParams, AstroParams, HMFinterp, massVector, z):
        'Mass accretion rate, in units of M_sun/yr'
        
        if not CosmoParams.Flag_emulate_21cmfast: #GALLUMI-like
            if AstroParams.accretion_model == "exp": #exponential accretion
                dMhdz = massVector * constants.ALPHA_accretion_exponential
                
            elif AstroParams.accretion_model == "EPS": #EPS accretion
                
                Mh2 = massVector* constants.EPSQ_accretion
                indexMh2low = Mh2 < massVector.flatten()[0]
                Mh2[indexMh2low] = massVector.flatten()[0]
                
                sigmaMh = HMFinterp.sigmaintlog((np.log(massVector), z))
                sigmaMh2 = HMFinterp.sigmaintlog((np.log(Mh2), z))
                sigmaMh2[np.full_like(sigmaMh2, fill_value=True, dtype = bool) * indexMh2low] = 1e99
                
                growth = cosmology.growth(CosmoParams,z)
                dzgrow = z*0.01
                dgrowthdz = (cosmology.growth(CosmoParams,z+dzgrow) - cosmology.growth(CosmoParams,z-dzgrow))/(2.0 * dzgrow)
                dMhdz = - massVector * np.sqrt(2/np.pi)/np.sqrt(sigmaMh2**2 - sigmaMh**2) *dgrowthdz/growth * CosmoParams.delta_crit_ST
                
            else:
                print("ERROR! Have to choose an accretion model in AstroParams (accretion_model)")
            Mhdot = dMhdz*cosmology.Hubinvyr(CosmoParams,z)*(1.0+z)
            return Mhdot

        else: #21cmfast-like
            return massVector/AstroParams.tstar*cosmology.Hubinvyr(CosmoParams,z)


    def SFR(self, CosmoParams, AstroParams, HMFinterp, massVector, z, pop, vCB = False, J21LW_interp = False):
        "SFR in Msun/yr at redshift z. Evaluated at the halo masses Mh [Msun] of the HMFinterp, given AstroParams"
        
        if (pop == 3 and not AstroParams.USE_POPIII):
            return 0 #skip whole routine if NOT using PopIII stars
        
        if pop == 2:
            fstarM = self.fstarofz_II(CosmoParams, AstroParams, z, massVector)
        else:
            fstarM = self.fstarofz_III(CosmoParams, AstroParams, z, massVector)

        fduty = self.fduty(CosmoParams, AstroParams, massVector, z, pop, vCB, J21LW_interp)
        
        return self.dMh_dt(CosmoParams, AstroParams, HMFinterp, massVector, z)  * fstarM * fduty


    def SFRD_integrand(self, CosmoParams, AstroParams, HMFinterp, massVector, z, pop, vCB = False, J21LW_interp = False):
        
        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(massVector), z)))
        SFRtab_curr = self.SFR(CosmoParams, AstroParams, HMFinterp, massVector, z, pop, vCB, J21LW_interp)
        integrand = HMF_curr * SFRtab_curr * massVector

        return integrand
    

    def J_LW_21(self, CosmoParams, AstroParams, sfrdIter, z, pop):
        #specific intensity, units of erg/s/cm^2/Hz/sr
        #for units to work, c must be in Mpc/s and proton mass in solar masses
        #and convert from 1/Mpc^2 to 1/cm^2
        
        Elw = (constants.Elw_eV * u.eV).to(u.erg).value
        
        if pop == 3:
            Nlw = AstroParams.N_LW_III
        elif pop == 2:
            Nlw = AstroParams.N_LW_II

        zIntMatrix = np.linspace(z, constants.redshiftFactor_Visbal*(1+z)-1, 20)
                
        if CosmoParams.Flag_emulate_21cmfast:##HAC ACAUSAL: This if statement allows for acausal Mmol
            sfrdIterMatrix_LW = sfrdIter * np.ones_like(zIntMatrix) 
        else:
            sfrdIterMatrix_LW = interpolate.interp1d(z, sfrdIter, kind = 'linear', bounds_error=False, fill_value=0)(zIntMatrix)
    
        integrandLW = constants.c_Mpcs / 4 / np.pi
        integrandLW *= (1+z)**2 / cosmology.Hubinvyr(CosmoParams,zIntMatrix)
        integrandLW *= Nlw * Elw / constants.mprotoninMsun / constants.deltaNulw
        integrandLW = integrandLW * sfrdIterMatrix_LW * (1 /u.Mpc**2).to(1/u.cm**2).value #broadcasting doesn't like augmented assignment operations (like *=) for some reason

        return 1e21 *np.trapezoid(integrandLW, x = zIntMatrix, axis = 0)


    def J_LW_Discrete(self, CosmoParams, AstroParams, z, pop, rGreater, SFRD_interp_input):
        #specific intensity, units of erg/s/cm^2/Hz/sr
        #for units to work, c must be in Mpc/s and proton mass in solar masses
        #and convert from 1/Mpc^2 to 1/cm^2
        
        Elw = (constants.Elw_eV * u.eV).to(u.erg).value
        
        rTable = np.transpose([CosmoParams.chiofzint(z)]) + rGreater
        rTable[rTable > CosmoParams.chiofzint(constants.zmax_AstroBreak)] = CosmoParams.chiofzint(constants.zmax_AstroBreak) #cut down so that nothing exceeds zmax = constants.zmax_AstroBreak
        zTable = CosmoParams.zfofRint(rTable)
        
        ##HAC ACAUSAL: The below if statement allows for acausal Mmol
        if CosmoParams.Flag_emulate_21cmfast:
            zTable = np.array([z]).T * np.ones_like(rTable) #HAC: This fixes J_LW(z) = int SFRD(z) dz' such that no z' dependence in the integral (for some reason 21cmFAST does this). Delete when comparing J_LW() with Visbal+14 and Mebane+17
            
        zMax = np.transpose([constants.redshiftFactor_Visbal*(1+z)-1])
        rMax = CosmoParams.chiofzint(zMax)
        
        c1 = (1+z)**2/4/np.pi
        
        if pop == 3:
            Nlw = AstroParams.N_LW_III

        elif pop == 2:
            Nlw = AstroParams.N_LW_II
        
        c2r = SFRD_interp_input(zTable)
                
        c2r *= Nlw * Elw / constants.deltaNulw / constants.mprotoninMsun * 0.5*(1 - np.tanh((rTable - rMax)/10)) * (1 /u.yr/u.Mpc**2).to(1/u.s/u.cm**2).value #smooth tanh cutoff, smoother function within 2-3% agreement with J_LW()
        
        return np.transpose([c1]), c2r


    def dSFRDIII_dJ(self,CosmoParams, AstroParams, HMFinterp, z, vCB, J21LW_interp):

        Mh = HMFinterp.Mhtab
        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z)))

        SFRtab_currIII = self.SFR(CosmoParams, AstroParams, HMFinterp, HMFinterp.Mhtab, z, pop=3, vCB = vCB, J21LW_interp=J21LW_interp)

        integrand_III = HMF_curr * SFRtab_currIII * HMFinterp.Mhtab
        integrand_III *= AstroParams.A_LW * AstroParams.beta_LW * J21LW_interp(z)**(AstroParams.beta_LW - 1)
        integrand_III *= -1 * self.Mmol_vcb(CosmoParams, AstroParams, z, CosmoParams.vcb_avg)/ HMFinterp.Mhtab

        return np.trapezoid(integrand_III, HMFinterp.logtabMh)


    def fesc_II(self,AstroParams, Mh):
        "f_escape for a halo of mass Mh [Msun] given AstroParams" #The pivot scale here for Pop II stars is at 1e10 solar masses
        return np.fmin(1.0, AstroParams.fesc10 * pow(Mh/1e10,AstroParams.alphaesc) )

    def fesc_III(self,AstroParams, Mh):
        "f_escape for a PopIII halo of mass Mh [Msun] given AstroParams" #The pivot scale here for Pop III stars is at 1e7 solar masses
        return np.fmin(1.0, AstroParams.fesc7_III * pow(Mh/1e7,AstroParams.alphaesc_III) )

    def compute_sigmaR_nu(self, CosmoParams, HMFinterp, z_array, R_array, Mh_array, dorv_array, dorv):

        zArray, rArray, mArray, dorvNormArray = np.meshgrid(z_array, R_array, Mh_array, dorv_array, indexing = 'ij', sparse = True)

        rGreaterArray = np.zeros_like(zArray) + rArray
        rGreaterArray[CosmoParams.chiofzint(zArray) + rArray >= CosmoParams.chiofzint(constants.zmax_AstroBreak)] = np.nan
        zGreaterArray = CosmoParams.zfofRint(CosmoParams.chiofzint(zArray) + rGreaterArray)

        whereNotNans = np.invert(np.isnan(rGreaterArray))

        sigmaR = np.zeros((len(z_array), len(R_array), 1, 1))
        sigmaR[whereNotNans] = HMFinterp.sigmaRintlog((np.log(rGreaterArray)[whereNotNans], zGreaterArray[whereNotNans]))

        sigmaM = HMFinterp.sigmaintlog((np.log(mArray), zGreaterArray))

        modSigmaSq = sigmaM**2 - sigmaR**2
        indexTooBig = (modSigmaSq <= 0.0)
        modSigmaSq[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigma = np.sqrt(modSigmaSq)

        nu0 = CosmoParams.delta_crit_ST / sigmaM
        nu0[indexTooBig] = 1.0

        dsigmadMcurr = HMFinterp.dsigmadMintlog((np.log(mArray),zGreaterArray)) ###HAC: Check this works when emulating 21cmFAST
        dlogSdMcurr = (dsigmadMcurr*sigmaM*2.0)/(modSigmaSq)

        if dorv == "delta":
            deltaArray = dorvNormArray * sigmaR
        elif dorv == "vel":
            deltaArray = np.zeros_like(sigmaR) # deltaZero 

        modd = CosmoParams.delta_crit_ST - deltaArray
        nu = modd / modSigma
        
        if not CosmoParams.Flag_emulate_21cmfast:

            # EPS_HMF_corr
            HMF_corr = (nu/nu0) * (sigmaM/modSigma)**2.0 * np.exp(-CosmoParams.a_corr_EPS * (nu**2-nu0**2)/2.0 ) * (1.0 + deltaArray)

        else: #as 21cmFAST, use PS HMF, integrate and normalize at the end

            # PS_HMF_corr
            HMF_corr = cosmology.PS_HMF_unnorm(CosmoParams, Mh_array.reshape(len(Mh_array),1),nu,dlogSdMcurr) * (1.0 + deltaArray)

        if dorv == "delta":
            out = deltaArray
        elif dorv == "vel":
            out = dorvNormArray

        return HMF_corr, mArray, zGreaterArray, out
    

    def compute_gamma(self, CosmoParams, AstroParams, HMFinterp, z_array, R_array, Mh_array, input_sigmaofRtab, fesctab_II):

        #and EPS factors
        Nsigmad = 1.0 #how many sigmas we explore
        Nds = 3 #how many deltas

        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)
        
        HMF_corr, mArray, zGreaterArray, deltaArray = self.compute_sigmaR_nu(CosmoParams, HMFinterp, z_array, R_array, Mh_array, deltatab_norm, "delta")

        #PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
        if not CosmoParams.Flag_emulate_21cmfast:
        #Normalized PS(d)/<PS(d)> at each mass. 21cmFAST instead integrates it and does SFRD(d)/<SFRD(d)>
        # last 1+delta product converts from Lagrangian to Eulerian
            
            integrand_II = HMF_corr * self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=2)

            if AstroParams.USE_POPIII:
                integrand_III = HMF_corr * self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3, vCB=CosmoParams.vcb_avg,J21LW_interp=self.J21LW_interp_conv_avg)
            
        else: #as 21cmFAST, use PS HMF, integrate and normalize at the end
                        
            integrand_II = HMF_corr * self.SFR(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=2) * mArray

            if AstroParams.USE_POPIII:
                integrand_III = HMF_corr * self.SFR(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3,  vCB=CosmoParams.vcb_avg,J21LW_interp=self.J21LW_interp_conv_avg) * mArray

        ########
        # Compute SFRD quantities
        SFRD_II_dR = np.trapezoid(integrand_II, HMFinterp.logtabMh, axis = 2)

        niondot_II_dR = np.trapezoid(integrand_II*fesctab_II[None, None, :, None], HMFinterp.logtabMh, axis = 2)

        if AstroParams.USE_POPIII:
                
            SFRD_III_dR = np.trapezoid(integrand_III, HMFinterp.logtabMh, axis = 2)
        else:
            SFRD_III_dR = np.zeros_like(SFRD_II_dR)

        self.gamma_II_index2D = self.compute_numerical_der_gamma(SFRD_II_dR, deltaArray, 1) 
        
        self.gamma2_II_index2D = self.compute_numerical_der_gamma(SFRD_II_dR, deltaArray, 2) 
        
        self.gamma_niondot_II_index2D = self.compute_numerical_der_gamma(niondot_II_dR, deltaArray, 1)

        self.gamma2_niondot_II_index2D = self.compute_numerical_der_gamma(niondot_II_dR, deltaArray, 2)

        if AstroParams.USE_POPIII:
            self.gamma_III_index2D = self.compute_numerical_der_gamma(SFRD_III_dR, deltaArray, 1)
            self.gamma2_III_index2D = self.compute_numerical_der_gamma(SFRD_III_dR, deltaArray, 2)
        else:
            self.gamma_III_index2D = np.zeros_like(self.gamma_II_index2D)
            self.gamma2_III_index2D = np.zeros_like(self.gamma2_II_index2D)

        gamma_II_index2D_Lag = self.gamma_II_index2D - 1.
        gamma_III_Lagrangian = self.gamma_III_index2D-1.0
        
        if AstroParams.quadratic_SFRD_lognormal:

            gamma2_II_index2D_Lag = self.gamma2_II_index2D + 1/2. 
            
            _corrfactorEulerian_II = (1+(gamma_II_index2D_Lag-2*gamma2_II_index2D_Lag)*self.sigmaofRtab**2)/(1-2*gamma2_II_index2D_Lag*self.sigmaofRtab**2)


            if AstroParams.USE_POPIII:
                gamma2_III_Lagrangian = self.gamma2_III_index2D + 1/2.
                _corrfactorEulerian_III = (1+(gamma_III_Lagrangian-2*gamma2_III_Lagrangian)*self.sigmaofRtab**2)/(1-2*gamma2_III_Lagrangian*self.sigmaofRtab**2)                
            else:
                _corrfactorEulerian_III = np.zeros_like(_corrfactorEulerian_II)

        else:
            _corrfactorEulerian_II = 1.0 + gamma_II_index2D_Lag * input_sigmaofRtab**2

            if AstroParams.USE_POPIII:
                _corrfactorEulerian_III = 1.0 + gamma_III_Lagrangian*self.sigmaofRtab**2
            else:
                _corrfactorEulerian_III = np.zeros_like(_corrfactorEulerian_II)


        self._corrfactorEulerian_II=_corrfactorEulerian_II.T

        self._corrfactorEulerian_II[0:CosmoParams.indexminNL] = self._corrfactorEulerian_II[CosmoParams.indexminNL] #for R<R_NL we just fix it to the RNL value, as we do for the correlation function. We could cut the sum but this keeps those scales albeit approximately

        self._corrfactorEulerian_III=_corrfactorEulerian_III.T ### TODO check if I should've just added the self here.
        _corrfactorEulerian_III[0:CosmoParams.indexminNL] = _corrfactorEulerian_III[CosmoParams.indexminNL] #for R<R_NL we just fix it to the RNL value, as we do for the correlation function. We could cut the sum but this keeps those scales albeit approximately


        ### LW correction to Pop III gammas
        if AstroParams.USE_POPIII:
            if AstroParams.USE_LW_FEEDBACK:
                #get the zero-lag correlation function (zero distance separation)
                xi_RR_CF_zerolag = np.copy(CosmoParams.ClassCosmo.pars['xi_RR_CF'][:,:,0])

                #compute LW coefficients for Pop II and III stars
                coeff1LWzp_II, coeff2LWzpRR_II = self.J_LW_Discrete(CosmoParams, AstroParams, z_array, 2, R_array, self.SFRD_II_interp)

                coeff1LWzp_III, coeff2LWzpRR_III = self.J_LW_Discrete(CosmoParams, AstroParams, z_array, 3, R_array, self.SFRD_III_cnvg_interp)

                # Corrections WITH Rmax smoothing
                deltaGamma_R = 1 / np.transpose([self.SFRD_III_cnvg_interp(z_array)])
                deltaGamma_R *= np.array([self.dSFRDIII_dJ(CosmoParams, AstroParams, HMFinterp, np.array([z_array]).T, vCB=CosmoParams.vcb_avg, J21LW_interp=self.J21LW_interp_conv_avg)]).T
                
                deltaGamma_R = deltaGamma_R * (coeff1LWzp_II * coeff2LWzpRR_II * self.gamma_II_index2D + coeff1LWzp_III * coeff2LWzpRR_III * self.gamma_III_index2D) * 1e21

                #choose only max of r and R; since growth factors cancel out, none are used here
                xi_R_maxrR = np.tril(np.ones_like(xi_RR_CF_zerolag)) * np.transpose([np.diag(xi_RR_CF_zerolag)])
                xi_R_maxrR  = xi_R_maxrR  + np.triu(xi_RR_CF_zerolag, k = 1)

                self.deltaGamma_R_Matrix = xi_R_maxrR.reshape(len(R_array), 1, len(R_array)) * (deltaGamma_R * CosmoParams._dlogRR * R_array).reshape(1, len(z_array), len(R_array))
                deltaGamma_R_z = np.transpose(   np.sum(self.deltaGamma_R_Matrix, axis = 2) / np.transpose([np.diagonal(xi_RR_CF_zerolag[:,:])])    )
                deltaGamma_R_z[ self.gamma_III_index2D == 0 ] = 0 #don't correct gammas if gammas are zero
                self.deltaGamma_R_z = deltaGamma_R_z
                self.gamma_III_index2D += deltaGamma_R_z #correct Pop III gammas with LW correction factor
                

        return 1


    def compute_numerical_der_gamma(self, arr1, arr2, order):

        midpoint = arr2.shape[-1]//2 #midpoint of deltaArray at delta = 0

        if order == 1:
           darr1_darr2 =  np.log(arr1[:,:,midpoint+1]/arr1[:,:,midpoint-1]) / (arr2[:,:,0,midpoint+1] - arr2[:,:,0,midpoint-1])

        elif order == 2:

            der1_II =  np.log(arr1[:,:,midpoint]/arr1[:,:,midpoint-1])/(arr2[:,:,0,midpoint] - arr2[:,:,0,midpoint-1]) #ln(y2/y1)/(x2-x1)
            der2_II =  np.log(arr1[:,:,midpoint+1]/arr1[:,:,midpoint])/(arr2[:,:,0,midpoint+1] - arr2[:,:,0,midpoint]) #ln(y3/y2)/(x3-x2)
            darr1_darr2 = (der2_II - der1_II)/(arr2[:,:,0,midpoint+1] - arr2[:,:,0,midpoint-1]) #second derivative: (der2-der1)/((x3-x1)/2)

        else:
            print('Check derivation order for gammas')
            return 0 

        darr1_darr2[np.isnan(darr1_darr2)] = 0.0

        return darr1_darr2


class PopIII_relvel:

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None, SFRD_Init = None):

        if z_Init is None:
            z_Init = Z_init(UserParams, CosmoParams)

        if SFRD_Init is None:
            SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init) 


        if AstroParams.USE_POPIII:
            self.vcb_expFitParams = np.zeros((len(z_Init.zintegral),len(CosmoParams._Rtabsmoo), 4)) #for the 4 exponential parameters
            
            if CosmoParams.USE_RELATIVE_VELOCITIES:

                v_avg0 = CosmoParams.ClassCosmo.pars['v_avg']
                vAvg_array = v_avg0 * np.array([0.2, 0.7, 1, 1.25, 2.0])
                etaTilde_array = 3 * vAvg_array**2 / CosmoParams.ClassCosmo.pars['sigma_vcb']**2

                HMF_corr, mArray, zGreaterArray, velArray  = SFRD_Init.compute_sigmaR_nu(CosmoParams, HMFinterp, z_Init.zintegral, CosmoParams._Rtabsmoo, HMFinterp.Mhtab, vAvg_array, 'vel') 
                

                #PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
                if not CosmoParams.Flag_emulate_21cmfast:
                #Normalized PS(d)/<PS(d)> at each mass. 21cmFAST instead integrates it and does SFRD(d)/<SFRD(d)>
                # last 1+delta product converts from Lagrangian to Eulerian

                    integrand_III = HMF_corr * SFRD_Init.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3, vCB=velArray, J21LW_interp=SFRD_Init.J21LW_interp_conv_avg)
                    
                else: #as 21cmFAST, use PS HMF, integrate and normalize at the end

                    integrand_III = HMF_corr * SFRD_Init.SFR(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3, vCB=velArray, J21LW_interp=SFRD_Init.J21LW_interp_conv_avg) * mArray

                SFRD_III_dR_V = np.trapezoid(integrand_III, HMFinterp.logtabMh, axis = 2)

                SFRDIII_Ratio = SFRD_III_dR_V / SFRD_III_dR_V[:,:,len(vAvg_array)//2].reshape((len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1))
                SFRDIII_Ratio[np.isnan(SFRDIII_Ratio)] = 0.0

                #temporarily turning off divide warnings; will turn them on again after exponential fitting routine
                divideErr = np.seterr(divide = 'ignore')
                divideErr2 = np.seterr(invalid = 'ignore')
                
                ###HAC: The next few lines fits for rho(z, v) / rhoavg = Ae^-b tilde(eta) + Ce^-d tilde(eta).
                ### To expedite the computation, instead of using scipy.optimize.curve_fit, I choose two points where one
                ### exponential dominates to fit for C and d, subtract Ce^-d tilde(eta) from rho(z, v) / rhoavg, then fit for A and b
                
                dParams = -1 * np.log(SFRDIII_Ratio[:,:,-1]/SFRDIII_Ratio[:,:,-2]) / (etaTilde_array[-1]-etaTilde_array[-2])

                cParams = np.exp(np.log(SFRDIII_Ratio[:,:,-1]) + dParams *  etaTilde_array[-1])

                SFRDIII_RatioNew = SFRDIII_Ratio - cParams.reshape(*cParams.shape, 1) * np.exp(-1 * dParams.reshape(*dParams.shape, 1)* etaTilde_array.reshape(1,1,*etaTilde_array.shape) )
                bParams = -1 * np.log(SFRDIII_RatioNew[:,:,0]/SFRDIII_RatioNew[:,:,1]) / (etaTilde_array[0]-etaTilde_array[1])
                aParams = np.exp(np.log(SFRDIII_RatioNew[:,:,0]) + bParams *  etaTilde_array[0])
                
                divideErr = np.seterr(divide = 'warn')
                divideErr2 = np.seterr(invalid = 'warn')
                
                self.vcb_expFitParams[:,:,0] = aParams
                self.vcb_expFitParams[:,:,1] = bParams
                self.vcb_expFitParams[:,:,2] = cParams
                self.vcb_expFitParams[:,:,3] = dParams

                self.vcb_expFitParams[np.isnan(self.vcb_expFitParams)] = 0.0
                