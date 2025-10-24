"""

Bulk of the Zeus21 calculation. Compute sSFRD from cosmology, determines Lyman-alpha and X-ray fluxes, and evolves the cosmic-dawn IGM state (WF coupling and heating). From that we get the 21-cm global signal and the effective biases gammaR to determine the 21-cm power spectrum.

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024
"""

from . import cosmology
from .xrays import Xray_class, sigma_HI, sigma_HeI
from . import constants

import numpy as np
import astropy
from astropy import units as u
from astropy import constants as const

import scipy
from scipy import interpolate

import pickle



class get_T21_coefficients:
    "Loops through SFRD integrals and obtains avg T21 and the coefficients for its power spectrum. Takes input zmin, which minimum z we integrate down to. It accounts for: \
    -Xray heating \
    -LyA coupling. \
    TODO: reionization/EoR"

    def __init__(self, User_Parameters, Cosmo_Parameters, ClassCosmo, Astro_Parameters, HMF_interpolator, zmin = 10.0):
    
        #####################################################################################################
        ### STEP 0: Defining Constants and storage variables

        #define comoving distance quantities
        self.Rtabsmoo = Cosmo_Parameters._Rtabsmoo
        self.dlogRR = Cosmo_Parameters._dlogRR
        
        #define the integration redshifts, goes as log(z) (1+ doesn't change sampling much)
        self.zmax_integral = constants.ZMAX_INTEGRAL
        self.zmin = zmin
        self._dlogzint_target = 0.02/User_Parameters.precisionboost
        self.Nzintegral = np.ceil(1.0 + np.log(self.zmax_integral/self.zmin)/self._dlogzint_target).astype(int)
        self.dlogzint = np.log(self.zmax_integral/self.zmin)/(self.Nzintegral-1.0) #exact value rather than input target above
        self.zintegral = np.logspace(np.log10(self.zmin), np.log10(self.zmax_integral), self.Nzintegral) #note these are also the z at which we "observe", to share computational load
        
        #define table of redshifts and distances
        self.rGreaterMatrix = np.transpose([Cosmo_Parameters.chiofzint(self.zintegral)]) + self.Rtabsmoo
        self.zGreaterMatrix = Cosmo_Parameters.zfofRint(self.rGreaterMatrix)
        
        self.ztabRsmoo = np.nan_to_num(np.copy(self.zGreaterMatrix), nan = 100)#HAC: patch fix for now. Later, figure out how to reconcile zGreaterMatrix with zGreaterMatrix_nonan
        if(Cosmo_Parameters.Flag_emulate_21cmfast == True): #they take the redshift to be at the midpoint of the two shells. In dr really.
        
            self.zGreaterMatrix = np.append(self.zintegral.reshape(len(self.zGreaterMatrix), 1), self.zGreaterMatrix, axis = 1)
            self.zGreaterMatrix = (self.zGreaterMatrix[:, 1:] + self.zGreaterMatrix[:, :-1])/2
            
#            self.zGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = 50.0 ###HAC: Check if I can actually comment this out or not
            self.rGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = Cosmo_Parameters.chiofzint(50.0)
            
            
            self.ztabRsmoo = np.append(self.zintegral.reshape(len(self.ztabRsmoo), 1), self.ztabRsmoo, axis = 1)###HAC: no longer necessary!
            self.ztabRsmoo = (self.ztabRsmoo[:, 1:] + self.ztabRsmoo[:, :-1])/2###HAC: no longer necessary!
        else:
            self.zGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = np.nan
            self.rGreaterMatrix[self.rGreaterMatrix > Cosmo_Parameters.chiofzint(50.0)] = np.nan #replace z > 50 = np.nan so that nothing exceeds zmax = 50
            self.ztabRsmoo = np.nan_to_num(np.copy(self.zGreaterMatrix), nan = 100)#HAC: patch fix for now. Later, figure out how to reconcile zGreaterMatrix with zGreaterMatrix_nonan
            
        zGreaterMatrix_nonan = np.nan_to_num(self.zGreaterMatrix, nan = 100)
#        self.ztabRsmoo = np.zeros_like(self.SFRDbar2D) #z's that correspond to each Radius R around each zp #HAC: No longer needed

        ###HAC: added SFRD & J21LW variables for pop II and III stars TO BE DELETED (not needed)
        self.SFRD_avg = np.zeros_like(self.zintegral)
        self.SFRD_II_avg = np.zeros_like(self.zintegral)
        self.SFRD_III_avg = np.zeros_like(self.zintegral)
        self.J_21_LW_II = np.zeros_like(self.zintegral)
        self.J_21_LW_III = np.zeros_like(self.zintegral)
        
        self.SFRDbar2D = np.zeros((self.Nzintegral, Cosmo_Parameters.NRs)) #SFR at z=zprime when averaged over a radius R (so up to a higher z)
        
        self.gamma_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta + \gamma_2 delta^2)
        self.gamma_II_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta + \gamma_2 delta^2)
        self.gamma2_II_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta + \gamma_2 delta^2)
        self.gamma_III_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta + \gamma_2 delta^2)
        self.gamma2_III_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma \delta + \gamma_2 \delta^2)
        
        self.niondot_avg = np.zeros_like(self.zintegral) #\dot nion at each z (int d(SFRD)/dM *fesc(M) dM)/rhobaryon
        self.gamma_Niondot_index2D = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta + \gamma_2 delta^2)


        
#        ###HAC: OLD, TO BE DELETED, added SFRD variables for pop II and III stars
#        self.gamma_index2D_old = np.zeros_like(self.SFRDbar2D) #index of SFR ~ exp(\gamma delta + \gamma_2 delta^2)

        #EoR coeffs
        self.sigmaofRtab = np.array([HMF_interpolator.sigmaR_int(self.Rtabsmoo, zz) for zz in self.zintegral]) #to be used in correlations.py, in get_bubbles()

        fesctab_II = fesc_II(Astro_Parameters, HMF_interpolator.Mhtab) #prepare fesc(M) table -- z independent for now so only once
        fesctab_III = fesc_III(Astro_Parameters, HMF_interpolator.Mhtab) #PopIII prepare fesc(M) table -- z independent for now so only once

        #Xray coeffs
        self.coeff1Xzp = np.zeros_like(self.zintegral) #zp-dependent coeff in Xray calculation
        self.coeff2XzpRR = np.zeros_like(self.SFRDbar2D) #zp and R-dependent coeff in Xray calculation
        self.Tk_avg = np.zeros_like(self.zintegral) #average kinetic temperature

        #LyA coeffs
        self.coeff1LyAzp = np.zeros_like(self.zintegral) # Same but for LyA
#        self.coeff2LyAzpRR = np.zeros_like(self.SFRDbar2D)# Same but for LyA
        self.Jalpha_avg = np.zeros_like(self.zintegral) #avg Jalpha (we compute xa at the end)
        _Jalpha_coeffs = np.zeros([constants.n_max_recycle-1,Cosmo_Parameters.NRs]) #the line recycled coeffs

        #and EPS factors
        Nsigmad = 1.0 #how many sigmas we explore
        Nds = 3 #how many deltas; must be an odd integer
        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)

        #initialize Xrays
        Xrays = Xray_class(User_Parameters, Cosmo_Parameters)
        _Energylist = Astro_Parameters.Energylist
        Nzinttau = np.floor(10*User_Parameters.precisionboost).astype(int)
        
        #####################################################################################################
        ### STEP 1: Recursive routine to compute average Pop II and III SFRDs with LW feedback
        ### Will only perform 1 iteration; if Astro_Parameters.USE_LW_FEEDBACK = False, then inputs.py sets A_LW = 0.0
        zSFRDflat = np.geomspace(self.zmin, 50, 128) #extend to z = 50 for extrapolation purposes. Higher in z than self.zintegral
        zSFRD, mArray = np.meshgrid(zSFRDflat, HMF_interpolator.Mhtab, indexing = 'ij', sparse = True)

        J21LW_interp = interpolate.interp1d(zSFRDflat, np.zeros_like(zSFRDflat), kind = 'linear', bounds_error = False, fill_value = 0,) #no LW background. Controls only Mmol() function, NOT the individual Pop II and III LW background
        SFRD_II_avg = np.trapz(SFRD_II_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, zSFRD, zSFRD), HMF_interpolator.logtabMh, axis = 1) #never changes with J_LW
        SFRD_II_interp = interpolate.interp1d(zSFRDflat, SFRD_II_avg, kind = 'cubic', bounds_error = False, fill_value = 0,)

        J21LW_II = 1e21 * J_LW(Astro_Parameters, Cosmo_Parameters, SFRD_II_avg, zSFRDflat, 2) #this never changes; only Pop III Quanties change
        self.J_21_LW_II = interpolate.interp1d(zSFRDflat, J21LW_II, kind = 'cubic')(self.zintegral) #different from J21LW_interp

        if Astro_Parameters.USE_POPIII == True:
            SFRD_III_Iter_Matrix = [np.trapz(SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, J21LW_interp, zSFRD, zSFRD, ClassCosmo.pars['v_avg']), HMF_interpolator.logtabMh, axis = 1)] #changes with each iteration

            errorTolerance = 0.001 # 0.1 percent accuracy
            recur_iterate_Flag = True
            while recur_iterate_Flag == True:
                J21LW_III_iter = 1e21 * J_LW(Astro_Parameters, Cosmo_Parameters, SFRD_III_Iter_Matrix[-1], zSFRDflat, 3)
                J21LW_interp = interpolate.interp1d(zSFRDflat, J21LW_II + J21LW_III_iter, kind = 'linear', fill_value = 0, bounds_error = False)

                SFRD_III_avg_n = np.trapz(SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, J21LW_interp, zSFRD, zSFRD, ClassCosmo.pars['v_avg']), HMF_interpolator.logtabMh, axis = 1)
                SFRD_III_Iter_Matrix.append(SFRD_III_avg_n)

                if max(SFRD_III_Iter_Matrix[-1]/SFRD_III_Iter_Matrix[-2]) < 1.0 + errorTolerance and min(SFRD_III_Iter_Matrix[-1]/SFRD_III_Iter_Matrix[-2]) > 1.0 - errorTolerance:
                    recur_iterate_Flag = False

            self.J21LW_interp_conv_avg = J21LW_interp
            SFRD_III_cnvg_interp = interpolate.interp1d(zSFRDflat, SFRD_III_Iter_Matrix[-1], kind = 'cubic', bounds_error = False, fill_value = 0)
            self.J_21_LW_III = interpolate.interp1d(zSFRDflat, J21LW_III_iter, kind = 'cubic')(self.zintegral)

        elif Astro_Parameters.USE_POPIII == False:
            self.SFRD_III_avg = np.zeros_like(self.zintegral)
            SFRD_III_cnvg_interp = interpolate.interp1d(zSFRDflat, np.zeros_like(zSFRDflat), kind = 'cubic', bounds_error = False, fill_value = 0)

        self.SFRD_II_avg = SFRD_II_interp(self.zintegral)
        self.SFRD_III_avg = SFRD_III_cnvg_interp(self.zintegral)
        self.SFRD_avg = self.SFRD_II_avg + self.SFRD_III_avg

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
            self.SFRDbar2D_II = SFRD_II_interp(np.nan_to_num(self.zGreaterMatrix, nan = 100))
            self.SFRDbar2D_III = SFRD_III_cnvg_interp(np.nan_to_num(self.zGreaterMatrix, nan = 100))
            
        elif(Cosmo_Parameters.Flag_emulate_21cmfast==True): ###HAC ACAUSAL: This accounts for the acausal Mmol effect in 21cmfast
            zpTable, tempTable, mTable = np.meshgrid(self.zintegral, self.Rtabsmoo, HMF_interpolator.Mhtab, indexing = 'ij', sparse = True)
            zppTable = self.zGreaterMatrix.reshape((len(self.zintegral), len(self.Rtabsmoo), 1))

            self.SFRDbar2D_II = np.trapz(SFRD_II_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mTable, zppTable, zpTable), HMF_interpolator.logtabMh, axis = 2)
            self.SFRDbar2D_III = np.trapz(SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mTable, J21LW_interp, zppTable, zpTable, ClassCosmo.pars['v_avg']), HMF_interpolator.logtabMh, axis = 2)

            self.SFRDbar2D_II[np.isnan(self.SFRDbar2D_II)] = 0.0
            self.SFRDbar2D_III[np.isnan(self.SFRDbar2D_III)] = 0.0
                
        
        #####################################################################################################
        ### STEP 2: Broadcasted Prescription to Compute gammas
        zArray, rArray, mArray, deltaNormArray = np.meshgrid(self.zintegral, self.Rtabsmoo, HMF_interpolator.Mhtab, deltatab_norm, indexing = 'ij', sparse = True)

        rGreaterArray = np.zeros_like(zArray) + rArray

        rGreaterArray[Cosmo_Parameters.chiofzint(zArray) + rArray >= Cosmo_Parameters.chiofzint(50)] = np.nan
        zGreaterArray = Cosmo_Parameters.zfofRint(Cosmo_Parameters.chiofzint(zArray) + rGreaterArray)

        whereNotNans = np.invert(np.isnan(rGreaterArray))

        sigmaR = np.zeros((len(self.zintegral), len(self.Rtabsmoo), 1, 1))
        sigmaR[whereNotNans] = HMF_interpolator.sigmaRintlog((np.log(rGreaterArray)[whereNotNans], zGreaterArray[whereNotNans]))

        sigmaM = np.zeros((len(self.zintegral), len(self.Rtabsmoo), len(HMF_interpolator.Mhtab), 1)) ###HAC: Is this necessary?
        sigmaM = HMF_interpolator.sigmaintlog((np.log(mArray), zGreaterArray))

        modSigmaSq = sigmaM**2 - sigmaR**2
        indexTooBig = (modSigmaSq <= 0.0)
        modSigmaSq[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigma = np.sqrt(modSigmaSq)

        nu0 = Cosmo_Parameters.delta_crit_ST / sigmaM
        nu0[indexTooBig] = 1.0

        dsigmadMcurr = HMF_interpolator.dsigmadMintlog((np.log(mArray),zGreaterArray)) ###HAC: Check this works when emulating 21cmFAST
        dlogSdMcurr = (dsigmadMcurr*sigmaM*2.0)/(modSigmaSq)

        deltaArray = deltaNormArray * sigmaR
        # sMax = 0.3
        # deltaArray[Nsigmad * sigmaR > 1.0] = deltaNormArray * sMax

        modd = Cosmo_Parameters.delta_crit_ST - deltaArray
        nu = modd / modSigma

        #PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
        if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
        #Normalized PS(d)/<PS(d)> at each mass. 21cmFAST instead integrates it and does SFRD(d)/<SFRD(d)>
        # last 1+delta product converts from Lagrangian to Eulerian
            EPS_HMF_corr = (nu/nu0) * (sigmaM/modSigma)**2.0 * np.exp(-Cosmo_Parameters.a_corr_EPS * (nu**2-nu0**2)/2.0 ) * (1.0 + deltaArray)
            integrand_II = EPS_HMF_corr * SFRD_II_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, zGreaterArray, zGreaterArray)
            
        elif(Cosmo_Parameters.Flag_emulate_21cmfast==True): #as 21cmFAST, use PS HMF, integrate and normalize at the end
            PS_HMF_corr = cosmology.PS_HMF_unnorm(Cosmo_Parameters, HMF_interpolator.Mhtab.reshape(len(HMF_interpolator.Mhtab),1),nu,dlogSdMcurr) * (1.0 + deltaArray)
            integrand_II = PS_HMF_corr * SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, zGreaterArray, zGreaterArray) * mArray
            
        else:
            print("ERROR: Need to set FLAG_EMULATE_21CMFAST at True or False in the self.gamma_index2D calculation.")

        ########
        # Compute SFRD and niondot quantities
        SFRD_II_dR = np.trapz(integrand_II, HMF_interpolator.logtabMh, axis = 2)
        niondot_II_dR = np.trapz(integrand_II*fesctab_II[None, None, :, None], HMF_interpolator.logtabMh, axis = 2)

        ###
        if Astro_Parameters.USE_POPIII == True:
            if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
                integrand_III = EPS_HMF_corr * SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, J21LW_interp, zGreaterArray, zGreaterArray, ClassCosmo.pars['v_avg'])
            elif(Cosmo_Parameters.Flag_emulate_21cmfast==True):
                integrand_III = PS_HMF_corr * SFR_III(Astro_Parameters, Cosmo_Parameters, ClassCosmo, HMF_interpolator, mArray, J21LW_interp, zGreaterArray, zGreaterArray, ClassCosmo.pars['v_avg']) * mArray

            SFRD_III_dR = np.trapz(integrand_III, HMF_interpolator.logtabMh, axis = 2)
            niondot_III_dR = np.trapz(integrand_III*fesctab_III[None, None, :, None], HMF_interpolator.logtabMh, axis = 2)
        else:
            SFRD_III_dR = np.zeros_like(SFRD_II_dR)
            
        #compute gammas
        midpoint = deltaArray.shape[-1]//2 #midpoint of deltaArray at delta = 0

        self.gamma_II_index2D = np.log(SFRD_II_dR[:,:,midpoint+1]/SFRD_II_dR[:,:,midpoint-1]) / (deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1])
        self.gamma_II_index2D[np.isnan(self.gamma_II_index2D)] = 0.0

        self.gamma_niondot_II_index2D = np.log(niondot_II_dR[:,:,midpoint+1]/niondot_II_dR[:,:,midpoint-1]) / (deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1])
        self.gamma_niondot_II_index2D[np.isnan(self.gamma_niondot_II_index2D)] = 0.0

        #compute second-order derivative gammas by computing two first-order derivatives #TODO: functionalize derivatives
        der1_II =  np.log(SFRD_II_dR[:,:,midpoint]/SFRD_II_dR[:,:,midpoint-1])/(deltaArray[:,:,0,midpoint] - deltaArray[:,:,0,midpoint-1]) #ln(y2/y1)/(x2-x1)
        der2_II =  np.log(SFRD_II_dR[:,:,midpoint+1]/SFRD_II_dR[:,:,midpoint])/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint]) #ln(y3/y2)/(x3-x2)
        self.gamma2_II_index2D = (der2_II - der1_II)/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1]) #second derivative: (der2-der1)/((x3-x1)/2)
        self.gamma2_II_index2D[np.isnan(self.gamma2_II_index2D)] = 0.0
        
        der1_niondot_II =  np.log(niondot_II_dR[:,:,midpoint]/niondot_II_dR[:,:,midpoint-1])/(deltaArray[:,:,0,midpoint] - deltaArray[:,:,0,midpoint-1]) #ln(y2/y1)/(x2-x1)
        der2_niondot_II =  np.log(niondot_II_dR[:,:,midpoint+1]/niondot_II_dR[:,:,midpoint])/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint]) #ln(y3/y2)/(x3-x2)
        self.gamma2_niondot_II_index2D = (der2_niondot_II - der1_niondot_II)/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1]) #second derivative: (der2-der1)/((x3-x1)/2)
        self.gamma2_niondot_II_index2D[np.isnan(self.gamma2_niondot_II_index2D)] = 0.0

        if Astro_Parameters.USE_POPIII == True:
            self.gamma_III_index2D = np.log(SFRD_III_dR[:,:,midpoint+1]/SFRD_III_dR[:,:,midpoint-1]) / (deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1])
            self.gamma_III_index2D[np.isnan(self.gamma_III_index2D)] = 0.0

            self.gamma_niondot_III_index2D = np.log(niondot_III_dR[:,:,midpoint+1]/niondot_III_dR[:,:,midpoint-1]) / (deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1])
            self.gamma_niondot_III_index2D[np.isnan(self.gamma_niondot_III_index2D)] = 0.0

            der1_III =  np.log(SFRD_III_dR[:,:,midpoint]/SFRD_III_dR[:,:,midpoint-1])/(deltaArray[:,:,0,midpoint] - deltaArray[:,:,0,midpoint-1]) #ln(y2/y1)/(x2-x1)
            der2_III =  np.log(SFRD_III_dR[:,:,midpoint+1]/SFRD_III_dR[:,:,midpoint])/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint]) #ln(y3/y2)/(x3-x2)
            self.gamma2_III_index2D = (der2_III - der1_III)/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1]) #second derivative: (der2-der1)/((x3-x1)/2)
            self.gamma2_III_index2D[np.isnan(self.gamma2_III_index2D)] = 0.0

            der1_niondot_III =  np.log(niondot_III_dR[:,:,midpoint]/niondot_III_dR[:,:,midpoint-1])/(deltaArray[:,:,0,midpoint] - deltaArray[:,:,0,midpoint-1]) #ln(y2/y1)/(x2-x1)
            der2_niondot_III =  np.log(niondot_III_dR[:,:,midpoint+1]/niondot_III_dR[:,:,midpoint])/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint]) #ln(y3/y2)/(x3-x2)
            self.gamma2_niondot_III_index2D = (der2_niondot_III - der1_niondot_III)/(deltaArray[:,:,0,midpoint+1] - deltaArray[:,:,0,midpoint-1]) #second derivative: (der2-der1)/((x3-x1)/2)
            self.gamma2_niondot_III_index2D[np.isnan(self.gamma2_niondot_III_index2D)] = 0.0

        else:
            self.gamma_III_index2D = np.zeros_like(self.gamma_II_index2D)
            self.gamma2_III_index2D = np.zeros_like(self.gamma2_II_index2D)
            self.gamma_niondot_III_index2D = np.zeros_like(self.gamma_niondot_II_index2D)
            self.gamma2_niondot_III_index2D = np.zeros_like(self.gamma2_niondot_II_index2D)
           
        #####################################################################################################
        ### STEP 3: Computing lambdas in velocity anisotropies
        ### Because we found the SFRD vcb dependence to be delta independent, we compute quantities below for a variety of R's and delta_R = 0
        
        if Astro_Parameters.USE_POPIII == True:
            self.vcb_expFitParams = np.zeros((len(self.zintegral),len(self.Rtabsmoo), 4)) #for the 4 exponential parameters
            
            if Cosmo_Parameters.USE_RELATIVE_VELOCITIES == True:

                v_avg0 = ClassCosmo.pars['v_avg']
                vAvg_array = v_avg0 * np.array([0.2, 0.7, 1, 1.25, 2.0])
                etaTilde_array = 3 * vAvg_array**2 / ClassCosmo.pars['sigma_vcb']**2

                zArray, rArray, mArray, velArray = np.meshgrid(self.zintegral, self.Rtabsmoo, HMF_interpolator.Mhtab, vAvg_array, indexing = 'ij', sparse = True)

                rGreaterArray = np.zeros_like(zArray) + rArray

                rGreaterArray[Cosmo_Parameters.chiofzint(zArray) + rArray >= Cosmo_Parameters.chiofzint(50)] = np.nan
                zGreaterArray = Cosmo_Parameters.zfofRint(Cosmo_Parameters.chiofzint(zArray) + rGreaterArray)

                whereNotNans = np.invert(np.isnan(rGreaterArray))

                sigmaR = np.zeros((len(self.zintegral), len(self.Rtabsmoo), 1, 1))
                sigmaR[whereNotNans] = HMF_interpolator.sigmaRintlog((np.log(rGreaterArray)[whereNotNans], zGreaterArray[whereNotNans]))

                sigmaM = np.zeros((len(self.zintegral), len(self.Rtabsmoo), len(HMF_interpolator.Mhtab), 1)) ###HAC: Is this necessary?
                sigmaM = HMF_interpolator.sigmaintlog((np.log(mArray), zGreaterArray))

                modSigmaSq = sigmaM**2 - sigmaR**2
                indexTooBig = (modSigmaSq <= 0.0)
                modSigmaSq[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
                modSigma = np.sqrt(modSigmaSq)

                nu0 = Cosmo_Parameters.delta_crit_ST / sigmaM
                nu0[indexTooBig] = 1.0

                dsigmadMcurr = HMF_interpolator.dsigmadMintlog((np.log(mArray),zGreaterArray)) ###HAC: Check this works when emulating 21cmFAST
                dlogSdMcurr = (dsigmadMcurr*sigmaM*2.0)/(modSigmaSq)

                deltaZero = np.zeros_like(sigmaR)
                # sMax = 0.3
                # deltaArray[Nsigmad * sigmaR > 1.0] = deltaNormArray * sMax

                modd = Cosmo_Parameters.delta_crit_ST - deltaZero
                nu = modd / modSigma

                #PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
                if(Cosmo_Parameters.Flag_emulate_21cmfast==False):
                #Normalized PS(d)/<PS(d)> at each mass. 21cmFAST instead integrates it and does SFRD(d)/<SFRD(d)>
                # last 1+delta product converts from Lagrangian to Eulerian
                    EPS_HMF_corr = (nu/nu0) * (sigmaM/modSigma)**2.0 * np.exp(-Cosmo_Parameters.a_corr_EPS * (nu**2-nu0**2)/2.0 ) * (1.0 + deltaZero)
                    integrand_III = EPS_HMF_corr * SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, J21LW_interp, zGreaterArray, zGreaterArray, velArray)
                    
                elif(Cosmo_Parameters.Flag_emulate_21cmfast==True): #as 21cmFAST, use PS HMF, integrate and normalize at the end
#                    PS_HMF_corr = cosmology.PS_HMF_unnorm(Cosmo_Parameters, HMF_interpolator.Mhtab.reshape(len(HMF_interpolator.Mhtab),1),nu,dlogSdMcurr) * (1.0 + deltaZero)
                    PS_HMF_corr = cosmology.PS_HMF_unnorm(Cosmo_Parameters, HMF_interpolator.Mhtab.reshape(len(HMF_interpolator.Mhtab),1), nu, dlogSdMcurr) * (1.0 + deltaZero)
                    integrand_III = PS_HMF_corr * SFR_III(Astro_Parameters, Cosmo_Parameters, ClassCosmo, HMF_interpolator, mArray, J21LW_interp, zGreaterArray, zGreaterArray, velArray) * mArray
                    
                else:
                    print("ERROR: Need to set FLAG_EMULATE_21CMFAST at True or False in the self.gamma_index2D calculation.")

                SFRD_III_dR_V = np.trapz(integrand_III, HMF_interpolator.logtabMh, axis = 2)

                SFRDIII_Ratio = SFRD_III_dR_V / SFRD_III_dR_V[:,:,len(vAvg_array)//2].reshape((len(self.zintegral), len(self.Rtabsmoo), 1))
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
                
        
        #####################################################################################################
        ### STEP 4: LW correction to Pop III gammas
        if Astro_Parameters.USE_POPIII == True:
            if Astro_Parameters.USE_LW_FEEDBACK == True:
                #get the zero-lag correlation function (zero distance separation)
                xi_RR_CF_zerolag = np.copy(ClassCosmo.pars['xi_RR_CF'][:,:,0])

                #compute LW coefficients for Pop II and III stars
                coeff1LWzp_II, coeff2LWzpRR_II = J_LW_Discrete(Astro_Parameters, Cosmo_Parameters, ClassCosmo, self.zintegral, 2, self.Rtabsmoo, SFRD_II_interp, SFRD_III_cnvg_interp)
                coeff1LWzp_III, coeff2LWzpRR_III = J_LW_Discrete(Astro_Parameters, Cosmo_Parameters, ClassCosmo, self.zintegral, 3, self.Rtabsmoo, SFRD_II_interp, SFRD_III_cnvg_interp)

                # Corrections WITH Rmax smoothing
                deltaGamma_R = 1 / np.transpose([SFRD_III_cnvg_interp(self.zintegral)])
                deltaGamma_R *= np.array([dSFRDIII_dJ(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, J21LW_interp, np.array([self.zintegral]).T, np.array([self.zintegral]).T, ClassCosmo.pars['v_avg'])]).T
                deltaGamma_R = deltaGamma_R * (coeff1LWzp_II * coeff2LWzpRR_II * self.gamma_II_index2D + coeff1LWzp_III * coeff2LWzpRR_III * self.gamma_III_index2D) * 1e21

                #choose only max of r and R; since growth factors cancel out, none are used here
                xi_R_maxrR = np.tril(np.ones_like(xi_RR_CF_zerolag)) * np.transpose([np.diag(xi_RR_CF_zerolag)])
                xi_R_maxrR  = xi_R_maxrR  + np.triu(xi_RR_CF_zerolag, k = 1)

                self.deltaGamma_R_Matrix = xi_R_maxrR.reshape(len(self.Rtabsmoo), 1, len(self.Rtabsmoo)) * (deltaGamma_R * self.dlogRR * self.Rtabsmoo).reshape(1, len(self.zintegral), len(self.Rtabsmoo))
                deltaGamma_R_z = np.transpose(   np.sum(self.deltaGamma_R_Matrix, axis = 2) / np.transpose([np.diagonal(xi_RR_CF_zerolag[:,:])])    )
                deltaGamma_R_z[ self.gamma_III_index2D == 0 ] = 0 #don't correct gammas if gammas are zero
                self.deltaGamma_R_z = deltaGamma_R_z
                self.gamma_III_index2D += deltaGamma_R_z #correct Pop III gammas with LW correction factor
                

        #####################################################################################################
        ### STEP 5: Lyman-Alpha Anisotropies
        
        #Makes heavy use of broadcasting to make computations faster
        #3D cube will be summed over one axis. Dimensions are (z,R,n) = (64, 45, 21)

        self.coeff1LyAzp = (1+self.zintegral)**2/(4*np.pi)

        nuLYA = np.geomspace(constants.freqLyA, constants.freqLyCont, 128)
        sedLYAII_interp = interpolate.interp1d(nuLYA, Astro_Parameters.SED_LyA(nuLYA, pop = 2), kind = 'linear', bounds_error = False, fill_value = 0) #interpolate LyA SED

        n_recArray = np.arange(0,constants.n_max_recycle-1 )
        zpCube, rCube, n_recCube = np.meshgrid(self.zintegral, self.Rtabsmoo, n_recArray, indexing='ij', sparse=True) #for broadcasting purposes
        n_lineCube = n_recCube + 2
        zmax_lineCube = (1+zpCube) * (1 - pow(1+n_lineCube,-2.0))/(1-pow(n_lineCube,-2.0) ) - 1.0 #maximum redshift Lyman series photons can redshift before falling into a Ly-n resonance

        nu_linezpCube = constants.freqLyCont * (1 - (1.0/n_lineCube)**2)
        zGreaterCube = zGreaterMatrix_nonan.reshape(len(self.zintegral), len(self.Rtabsmoo), 1)
        nu_lineRRCube = nu_linezpCube * (1.+zGreaterCube)/(1+zpCube)
        
        eps_alphaRR_II_Cube = Astro_Parameters.N_alpha_perbaryon_II/Cosmo_Parameters.mu_baryon_Msun  * sedLYAII_interp(nu_lineRRCube)
        
        #the last nonzero index of the array is overestimated since only part of the spherical shell is within zmax_line. Correct by by dz/Delta z
        weights_recCube = np.heaviside(zmax_lineCube - zGreaterCube, 0.0)
        index_first0_weightsCube = np.where(np.diff(weights_recCube, axis = 1) == -1) #find index of last nonzero value. equals zero if two consecutive elements are 1 or 0, and -1 if two consecutive elements are [1,0]
        i0Z, i0R, i0N = index_first0_weightsCube
        weights_recCube[i0Z, i0R, i0N] *= (zmax_lineCube[i0Z, 0, i0N] - zGreaterCube[i0Z, i0R, 0])/ (zGreaterCube[i0Z, i0R+1, 0] - zGreaterCube[i0Z, i0R, 0])

        Jalpha_II = np.array(constants.fractions_recycle)[:len(n_recArray)].reshape(1,1,len(n_recArray)) * weights_recCube * eps_alphaRR_II_Cube #just resizing f_recycle; it is length 29,we only consider up to n=22
        LyAintegral_II = np.sum(Jalpha_II,axis=2) #sum over axis 2, over all possible n transitions
        self.coeff2LyAzpRR_II = self.Rtabsmoo * self.dlogRR * self.SFRDbar2D_II * LyAintegral_II/ constants.yrTos/constants.Mpctocm**2

        if Astro_Parameters.USE_POPIII == True:
            sedLYAIII_interp = interpolate.interp1d(nuLYA, Astro_Parameters.SED_LyA(nuLYA, pop = 3), kind = 'linear', bounds_error = False, fill_value = 0)
            eps_alphaRR_III_Cube = Astro_Parameters.N_alpha_perbaryon_III/Cosmo_Parameters.mu_baryon_Msun  * sedLYAIII_interp(nu_lineRRCube)
            
            Jalpha_III = np.array(constants.fractions_recycle)[:len(n_recArray)].reshape(1,1,len(n_recArray)) * weights_recCube * eps_alphaRR_III_Cube
            LyAintegral_III = np.sum(Jalpha_III,axis=2)
            self.coeff2LyAzpRR_III = self.Rtabsmoo * self.dlogRR * self.SFRDbar2D_III * LyAintegral_III/ constants.yrTos/constants.Mpctocm**2
        else:
            self.coeff2LyAzpRR_III = np.zeros_like(self.coeff2LyAzpRR_II)
        
        
        #####################################################################################################
        ### STEP 6: X-ray Anisotropies
        
        zGreaterCube = zGreaterMatrix_nonan.reshape(len(self.zintegral), len(self.Rtabsmoo), 1, 1) #redefine this just for x-ray routine

        self.coeff1Xzp = -2/3 * self.zintegral * self.dlogzint / cosmology.Hubinvyr(Cosmo_Parameters,self.zintegral) / (1+self.zintegral) * (1+self.zintegral)**2
        self.coeff1Xzp = self.coeff1Xzp / (1+self.zintegral)**2 * constants.yrTos #this accounts for adiabatic cooling. compensated by the inverse at the end

        zpCube, rCube, eCube, zPPCube = np.meshgrid(self.zintegral, self.Rtabsmoo, _Energylist, np.arange(Nzinttau), indexing='ij', sparse=True)
        currentEnergyTable = eCube * (1+zGreaterCube) / (1+zpCube)
        SEDCube = Astro_Parameters.SED_XRAY(currentEnergyTable, pop = 2)
        SEDCube_III = Astro_Parameters.SED_XRAY(currentEnergyTable, pop = 3)

        ######## Broadcasted routine to find X-ray optical depths, modeled after but does not use xrays.optical_depth
        zPPCube = np.array([np.linspace(np.transpose([self.zintegral]), self.zGreaterMatrix, Nzinttau, axis = 2)])
        zPPCube = zPPCube.reshape(len(self.zintegral), len(self.Rtabsmoo), 1, Nzinttau) #to have 4D dimensions, default shape = (64,45, 1, 10)

        ePPCube = eCube * (1+ zPPCube) / (1+zpCube) #E'' = E(1+z'')/(1+z)
        sigmatot = Xrays.atomfractions[0] * sigma_HI(ePPCube)
        sigmatot += Xrays.atomfractions[1] * sigma_HeI(ePPCube)

        opticalDepthIntegrand = 1 / cosmology.HubinvMpc(Cosmo_Parameters, zPPCube) / (1+zPPCube) * sigmatot * cosmology.n_H(Cosmo_Parameters, zPPCube) * constants.Mpctocm #this uses atom fractions of 1 for HI and x_He for HeI
#        opticalDepthIntegrand = 1 / cosmology.HubinvMpc(Cosmo_Parameters, zPPCube) / (1+zPPCube) * sigmatot * cosmology.n_baryon(Cosmo_Parameters, zPPCube) * constants.Mpctocm
        tauCube = np.trapz(opticalDepthIntegrand, zPPCube, axis = 3)

        indextautoolarge = np.array(tauCube>=Xrays.TAUMAX)
        tauCube[indextautoolarge] = Xrays.TAUMAX

        if Cosmo_Parameters.Flag_emulate_21cmfast == False:
            weights_X_zCube = np.exp(-tauCube)
        elif Cosmo_Parameters.Flag_emulate_21cmfast == True:
            weights_X_zCube = np.heaviside(1.0 - tauCube, 0.5)
        else:
            print("Error, choose a correct XRAY_OPACITY_MODEL")
            
        SEDCube = SEDCube[:,:,:,0] #rescale dimensions of energy and SED cubes back to 3D, so we can integrate over energy
        SEDCube_III = SEDCube_III[:,:,:,0] #rescale dimensions of energy and SED cubes back to 3D, so we can integrate over energy
        
        eCube = eCube[:,:,:,0]
        ######## end of optical depth routine

        JX_coeffsCube = SEDCube * weights_X_zCube
        JX_coeffsCube_III = SEDCube_III * weights_X_zCube

        sigma_times_en = Xrays.atomfractions[0] * sigma_HI(eCube) * (eCube - Xrays.atomEnIon[0])
        sigma_times_en += Xrays.atomfractions[1] * sigma_HeI(eCube) * (eCube - Xrays.atomEnIon[1])
        sigma_times_en /= np.sum(Xrays.atomfractions)#to normalize per baryon, instead of per Hydrogen nucleus
                #HI and HeII separate. Notice Energy (and not Energy'), since they get absorbed at the zp frame
        
        xrayEnergyTable = np.sum(JX_coeffsCube * sigma_times_en * eCube * Astro_Parameters.dlogEnergy,axis=2)
        self.coeff2XzpRR_II = np.nan_to_num(self.Rtabsmoo * self.dlogRR * self.SFRDbar2D_II * xrayEnergyTable * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST, nan = 0)
        
        if Astro_Parameters.USE_POPIII == True:
            xrayEnergyTable_III = np.sum(JX_coeffsCube_III * sigma_times_en * eCube * Astro_Parameters.dlogEnergy,axis=2)
            self.coeff2XzpRR_III = np.nan_to_num(self.Rtabsmoo * self.dlogRR * self.SFRDbar2D_III * xrayEnergyTable_III * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST, nan = 0)
        else:
            self.coeff2XzpRR_III = np.zeros_like(self.coeff2XzpRR_II)

        
        #####################################################################################################
        ### STEP 7: Non-Linear Correction Factors
        #correct for nonlinearities in <(1+d)SFRD>, only if doing nonlinear stuff. We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).

        if(User_Parameters.C2_RENORMALIZATION_FLAG==True):
            _corrfactorEulerian_II = 1.0 + (self.gamma_II_index2D-1.0)*self.sigmaofRtab**2
            _corrfactorEulerian_II=_corrfactorEulerian_II.T
            _corrfactorEulerian_II[0:Cosmo_Parameters.indexminNL] = _corrfactorEulerian_II[Cosmo_Parameters.indexminNL] #for R<R_NL we just fix it to the RNL value, as we do for the correlation function. We could cut the sum but this keeps those scales albeit approximately
            self.coeff2LyAzpRR_II*= _corrfactorEulerian_II.T
            self.coeff2XzpRR_II*= _corrfactorEulerian_II.T
            if Astro_Parameters.USE_POPIII == True:
                _corrfactorEulerian_III = 1.0 + (self.gamma_III_index2D-1.0)*self.sigmaofRtab**2
                _corrfactorEulerian_III=_corrfactorEulerian_III.T
                _corrfactorEulerian_III[0:Cosmo_Parameters.indexminNL] = _corrfactorEulerian_III[Cosmo_Parameters.indexminNL] #for R<R_NL we just fix it to the RNL value, as we do for the correlation function. We could cut the sum but this keeps those scales albeit approximately
                self.coeff2LyAzpRR_III*= _corrfactorEulerian_III.T
                self.coeff2XzpRR_III*= _corrfactorEulerian_III.T
        ## alternative expression below: if you take (1+d)~exp(d) throughout.
        #self.coeff2LyAzpRR *= np.exp(self.sigmaofRtab**2/2.0 * (2.0 *self.gamma_index2D-1.0) )
        #self.coeff2XzpRR *= np.exp(self.sigmaofRtab**2/2.0 * (2.0 *self.gamma_index2D-1.0) )
        
        
        #####################################################################################################
        ### STEP 8: Computing free-electron fraction and Salpha correction factors in the Bulk IGM
        
        self._GammaXray_II = self.coeff1Xzp * np.sum( self.coeff2XzpRR_II ,axis=1) #notice units are modified (eg 1/H) so it's simplest to sum
        self._GammaXray_III = self.coeff1Xzp * np.sum( self.coeff2XzpRR_III ,axis=1) #notice units are modified (eg 1/H) so it's simplest to sum
        
        fion = 0.4 * np.exp(-cosmology.xefid(Cosmo_Parameters, self.zintegral)/0.2)#partial ionization from Xrays. Fit to Furlanetto&Stoever
        atomEnIonavg = (Xrays.atomfractions[0] *  Xrays.atomEnIon[0] + Xrays.atomfractions[1] *  Xrays.atomEnIon[1]) / (Xrays.atomfractions[0] + Xrays.atomfractions[1] ) #to turn this ratio into one over n_b instead of n_H
        
        self.coeff_Gammah_Tx_II = -Astro_Parameters.L40_xray * constants.ergToK * (1.0+self.zintegral)**2
        self.coeff_Gammah_Tx_III = -Astro_Parameters.L40_xray_III * constants.ergToK * (1.0+self.zintegral)**2 #convert from one to the other, last factors accounts for adiabatic cooling. compensated by the inverse at zp in coeff1Xzp. Minus because integral goes from low to high z, but we'll be summing from high to low everywhere.
        
        self.Gammaion_II = self.coeff_Gammah_Tx_II *constants.KtoeV * self._GammaXray_II * fion/atomEnIonavg * 3/2
        self.Gammaion_III = self.coeff_Gammah_Tx_III *constants.KtoeV * self._GammaXray_III * fion/atomEnIonavg * 3/2 #atomEnIonavg makes it approximate. No adiabatic cooling (or recombinations) so no 1+z factors. Extra 3/2 bc temperature has a 2/3
        
        #TODO: Improve model for xe

        self.xe_avg_ad = cosmology.xefid(Cosmo_Parameters, self.zintegral)
        self.xe_avg = self.xe_avg_ad + np.cumsum((self.Gammaion_II+self.Gammaion_III)[::-1])[::-1]
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self.xe_avg = 2e-4 * np.ones_like(self.Gammaion_II) #we force this when we emualte 21cmdast to compare both codes on the same footing
        self.xe_avg = np.fmin(self.xe_avg, 1.0-1e-9)

        #and heat from Xrays
        self._fheat = pow(self.xe_avg,0.225)
        self.coeff1Xzp*=self._fheat #since this is what we use for the power spectrum (and not Gammaheat) we need to upate it
        self.Gammaheat_II = self._GammaXray_II * self._fheat
        self.Gammaheat_III = self._GammaXray_III * self._fheat

        #Computing avg kinetic temperature as sum of adiabatic & xray temperature
        self.Tk_xray = self.coeff_Gammah_Tx_II * np.cumsum(self.Gammaheat_II[::-1])[::-1] + self.coeff_Gammah_Tx_III * np.cumsum(self.Gammaheat_III[::-1])[::-1]#in K, cumsum reversed because integral goes from high to low z. Only heating part
        self.Tk_ad = cosmology.Tadiabatic(Cosmo_Parameters, self.zintegral)
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self.Tk_ad*=0.95 #they use recfast, so their 'cosmo' temperature is slightly off
        self.Tk_avg = self.Tk_ad + self.Tk_xray


        # LyA stuff to find components of Salpha correction factor
        self.Jalpha_avg = self.coeff1LyAzp*np.sum(self.coeff2LyAzpRR_II + self.coeff2LyAzpRR_III,axis=1) #units of 1/(cm^2 s Hz sr)
        self.T_CMB = cosmology.Tcmb(ClassCosmo, self.zintegral)

        _tau_GP = 3./2. * cosmology.n_H(Cosmo_Parameters,self.zintegral) * constants.Mpctocm / cosmology.HubinvMpc(Cosmo_Parameters,self.zintegral) * (constants.wavelengthLyA/1e7)**3 * constants.widthLyAcm * (1.0 - self.xe_avg)  #~3e5 at z=6
#        _tau_GP = 3./2.*Cosmo_Parameters.f_H * cosmology.n_baryon(Cosmo_Parameters,self.zintegral)*constants.Mpctocm/cosmology.HubinvMpc(Cosmo_Parameters,self.zintegral) * (constants.wavelengthLyA/1e7)**3 * constants.widthLyAcm * (1.0 - self.xe_avg)  #~3e5 at z=6
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            _tau_GP/=Cosmo_Parameters.f_H #for some reason they multiuply by N0 (all baryons) and not NH0.

        _xiHirata = pow(_tau_GP*1e-7,1/3.)*pow(self.Tk_avg,-2./3)
        _factorxi = (1.0 + 2.98394*_xiHirata + 1.53583 * _xiHirata**2 + 3.8528 * _xiHirata**3)


        #prefactor without the Salpha correction from Hirata2006
        if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
            self._coeff_Ja_xa_0 = 1.66e11/(1+self.zintegral) #They use a fixed (and slightly ~10% off) value.
        else:
            self._coeff_Ja_xa_0 = 8.0*np.pi*(constants.wavelengthLyA/1e7)**2 * constants.widthLyA * constants.Tstar_21/(9.0*constants.A10_21*self.T_CMB) #units of (cm^2 s Hz sr), convert from Ja to xa. should give 1.81e11/(1+self.zintegral) for Tcmb_0=2.725 K

        #ADD AFTER CHECKING W JULIAN
        #collision coefficient fh(1-x_e) 
        #self.xc_HH = (1.0 - self.xe_avg) * cosmology.n_H(Cosmo_Parameters, self.zintegral) * np.interp(self.Tk_avg, constants.Tk_HH, constants.k_HH) / constants.A10_21 * constants.Tstar_21 / cosmology.Tcmb(ClassCosmo, self.zintegral)
        #self.xc_He = self.xe_avg * cosmology.n_H(Cosmo_Parameters, self.zintegral) * np.interp(self.Tk_avg, constants.Tk_He, constants.k_He) / constants.A10_21 * constants.Tstar_21 / cosmology.Tcmb(ClassCosmo, self.zintegral) #xe
        #self.xc_avg = self.xc_HH + self.xc_He

        self.coeff_Ja_xa = self._coeff_Ja_xa_0 * Salpha_exp(self.zintegral, self.Tk_avg, self.xe_avg)
        self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg
        self.invTcol_avg = 1.0 / self.Tk_avg
        self._invTs_avg = (1.0/self.T_CMB+(self.xa_avg)*self.invTcol_avg)/(1+self.xa_avg)
        #AFTER CHECKING ALSO UNCOMMENT THIS:
        #self._invTs_avg = (1.0/self.T_CMB+(self.xa_avg+self.xc_avg)*self.invTcol_avg)/(1+self.xa_avg+self.xc_avg)
        if(User_Parameters.FLAG_WF_ITERATIVE==True): #iteratively find Tcolor and Ts. Could initialize one to zero, but this should converge faster
            ### iteration routine to find Tcolor and Ts
            _invTs_tryfirst = 1.0/self.T_CMB
            while(np.sum(np.fabs(_invTs_tryfirst/self._invTs_avg - 1.0))>0.01): #no more than 1% error total
                _invTs_tryfirst = self._invTs_avg

                #update xalpha
                _Salphatilde = (1.0 - 0.0632/self.Tk_avg + 0.116/self.Tk_avg**2 - 0.401/self.Tk_avg*self._invTs_avg + 0.336*self._invTs_avg/self.Tk_avg**2)/_factorxi
                self.coeff_Ja_xa = self._coeff_Ja_xa_0 * _Salphatilde
                self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg

                #and Tcolor^-1
                self.invTcol_avg = 1.0/self.Tk_avg + constants.gcolorfactorHirata * 1.0/self.Tk_avg * (_invTs_tryfirst - 1.0/self.Tk_avg)

                #and finally Ts^-1
                self._invTs_avg = (1.0/self.T_CMB+self.xa_avg * self.invTcol_avg)/(1+self.xa_avg)

        
        
        #####################################################################################################
        ### STEP 9: Reionization
        _trec0 = 1.0/(constants.alphaB * cosmology.n_H(Cosmo_Parameters,0) *(1 + Cosmo_Parameters.x_He) * Astro_Parameters._clumping)#t_recombination  at z=0, in sec
#        _trec0 = 1.0/(constants.alphaB * cosmology.n_baryon(Cosmo_Parameters,0) * Astro_Parameters._clumping)#t_recombination  at z=0, in sec
        _recexp = 1.0/(_trec0 * np.sqrt(Cosmo_Parameters.OmegaM) * cosmology.Hubinvyr(Cosmo_Parameters,0) / constants.yrTos)# = 1/(_trec0 * H0 * sqrt(OmegaM) ), dimless. Assumes matter domination and constant clumping. Can be modified to power-law clumping changing the powerlaw below from 3/2

        self.coeffQzp = self.dlogzint*self.zintegral/cosmology.Hubinvyr(Cosmo_Parameters,self.zintegral)/(1+self.zintegral) #Deltaz * dt/dz. Units of 1/yr, inverse of niondot

        ###HAC: Added N_ion rate contribution from Pop II and III stars. Note that I am using rho_b(z=0) because it's a comoving volume
        zArray, mArray = np.meshgrid(self.zintegral, HMF_interpolator.Mhtab, indexing = 'ij', sparse = True)

        integrand_II_table = SFRD_II_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, zArray, zArray)
        integrand_III_table = SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, mArray, J21LW_interp, zArray, zArray, ClassCosmo.pars['v_avg'])
        
        self.niondot_avg_II = Astro_Parameters.N_ion_perbaryon_II/cosmology.rho_baryon(Cosmo_Parameters,0.) * np.trapz(integrand_II_table * fesctab_II, HMF_interpolator.logtabMh, axis = 1)
        self.niondot_avg_III = Astro_Parameters.N_ion_perbaryon_II/cosmology.rho_baryon(Cosmo_Parameters,0.) * np.trapz(integrand_III_table * fesctab_III, HMF_interpolator.logtabMh, axis = 1)
        self.niondot_avg = self.niondot_avg_II + self.niondot_avg_III

        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #regular calculation, integrating over time and accounting for recombinations in the exponent

            self.Qfactrecomb = np.exp(-2/3 * _recexp * pow(1+self.zintegral,3/2))
            self.Qion_avg = 1/self.Qfactrecomb*np.cumsum(self.coeffQzp[::-1] * self.Qfactrecomb[::-1] * self.niondot_avg[::-1])[::-1]

        if(Cosmo_Parameters.Flag_emulate_21cmfast==True): #21cmfast instead uses nion (rather than niondot and integrating). We can emulate that here. there nion = niondot * t_star/H(z) [see Park+19]. In that case we can iteratively solve for Q=nion - nrecom(Q), where nrecom = int dt Q/t_recom to correct for recombinations. Easier than ODE.

            #self._nion = np.cumsum(self.coeffQzp[::-1] * self.niondot_avg[::-1])[::-1]
            self._nion = self.niondot_avg * Astro_Parameters.tstar/cosmology.Hubinvyr(Cosmo_Parameters,self.zintegral)
            self._Q0iteration = self._nion #0th iteration has no recombinations
            self.trec = _trec0/(1+self.zintegral)**3/constants.yrTos #in yr at each time t
            self._Q1iteration = 0.0
            while(np.sum(np.abs(self._Q1iteration-self._Q0iteration))>0.001):
                self._Q1iteration = self._Q0iteration
                self._nrecombinations = np.cumsum(self.coeffQzp[::-1] * (self._Q1iteration/self.trec)[::-1])[::-1] #coeffQzp = dt/dz as before
                self._Q0iteration = self._nion - self._nrecombinations
            self.Qion_avg = self._Q0iteration

        #common to both methods.
        self.Qion_avg = np.fmin(1.0, self.Qion_avg)
        self._xHII_avg = self.Qion_avg + (1.0 - self.Qion_avg) * self.xe_avg #accounts for partial ionization, small effect
        self.xHI_avg = (1.0 - self._xHII_avg)
        self.xHI_avg = np.fmin(1.0, self.xHI_avg)
        
        
        #####################################################################################################
        ### STEP 10: Compute the 21cm Global Signal
        self.T21avg = cosmology.T021(Cosmo_Parameters,self.zintegral) * self.xa_avg/(1.0 + self.xa_avg) * (1.0 - self.T_CMB * self.invTcol_avg) * self.xHI_avg
        
        

        
        
        
        

def tau_reio(Cosmo_Parameters, T21_coefficients):
    "Returns the optical depth to reionization given a model. It assumes xHI=1 for z<zmin."
    #we separate into a low- and hi-z parts (z< or > zmini)

    _zlistlowz = np.linspace(0,T21_coefficients.zmin,100)
    
    _nelistlowz = cosmology.n_H(Cosmo_Parameters,_zlistlowz)*(1.0 + Cosmo_Parameters.x_He + Cosmo_Parameters.x_He * np.heaviside(constants.zHeIIreio - _zlistlowz,0.5))
#    _nelistlowz = cosmology.n_baryon(Cosmo_Parameters,_zlistlowz)*(Cosmo_Parameters.f_H + Cosmo_Parameters.f_He + Cosmo_Parameters.f_He * np.heaviside(constants.zHeIIreio - _zlistlowz,0.5))
    _distlistlowz = 1.0/cosmology.HubinvMpc(Cosmo_Parameters,_zlistlowz)/(1+_zlistlowz)
    _lowzint = constants.sigmaT * np.trapz(_nelistlowz*_distlistlowz,_zlistlowz) * constants.Mpctocm

    _zlisthiz = T21_coefficients.zintegral
    
    _nelistlhiz = cosmology.n_H(Cosmo_Parameters,_zlisthiz) * (1 + Cosmo_Parameters.x_He) * (1.0 - T21_coefficients.xHI_avg)
#    _nelistlhiz = cosmology.n_baryon(Cosmo_Parameters,_zlisthiz) * (1.0 - T21_coefficients.xHI_avg)
    _distlisthiz = 1.0/cosmology.HubinvMpc(Cosmo_Parameters,_zlisthiz)/(1+_zlisthiz)

    _hizint = constants.sigmaT * np.trapz(_nelistlhiz*_distlisthiz,_zlisthiz) * constants.Mpctocm

    return(_lowzint + _hizint)

def Matom(z):
    "Returns Matom as a function of z"
    return 3.3e7 * pow((1.+z)/(21.),-3./2)

###HAC: Added Mmol split by contributions with no, vcb, and LW feecback
def Mmol_0(z):
    "Returns Mmol as a function of z WITHOUT LW or VCB feedback"
    return 3.3e7 * (1.+z)**(-1.5)

def Mmol_vcb(Astro_Parameters, Cosmo_Parameters, z, vCB):
    "Returns Mmol as a function of z WITHOUT LW feedback"
    mmolBase = Mmol_0(z)
    vcbFeedback = pow(1 + Astro_Parameters.A_vcb * vCB / Cosmo_Parameters.sigma_vcb, Astro_Parameters.beta_vcb)
    return mmolBase * vcbFeedback

def Mmol_LW(Astro_Parameters, J21LW_interp, z):
    "Returns Mmol as a function of z WITHOUT VCB feedback"
    mmolBase = Mmol_0(z)
    lwFeedback = 1 + Astro_Parameters.A_LW*pow(J21LW_interp(z), Astro_Parameters.beta_LW)
    return mmolBase * lwFeedback
    
def Mmol(Astro_Parameters, Cosmo_Parameters, J21LW_interp, z, vCB):
    "Returns Mmol as a function of z WITH LW AND VCB feedback"
    mmolBase = Mmol_0(z)
    vcbFeedback = pow(1 + Astro_Parameters.A_vcb * vCB / Cosmo_Parameters.sigma_vcb, Astro_Parameters.beta_vcb)
    lwFeedback = 1 + Astro_Parameters.A_LW*pow(J21LW_interp(z), Astro_Parameters.beta_LW)
    
    return mmolBase * vcbFeedback * lwFeedback
    
    
#fstar = Mstardot/Mhdot, parametrizes as you wish
def fstarofz(Astro_Parameters, Cosmo_Parameters, z, Mhlist):
    epsstar_ofz = Astro_Parameters.epsstar * 10**(Astro_Parameters.dlog10epsstardz * (z-Astro_Parameters._zpivot) )
    if Cosmo_Parameters.Flag_emulate_21cmfast == False:
        return 2.0 * Cosmo_Parameters.OmegaB/Cosmo_Parameters.OmegaM * epsstar_ofz\
        /(pow(Mhlist/Astro_Parameters.Mc,- Astro_Parameters.alphastar) + pow(Mhlist/Astro_Parameters.Mc,- Astro_Parameters.betastar) )
    elif Cosmo_Parameters.Flag_emulate_21cmfast == True:
        return Cosmo_Parameters.OmegaB/Cosmo_Parameters.OmegaM * epsstar_ofz /(pow(Mhlist/Astro_Parameters.Mc,- Astro_Parameters.alphastar))

    
###HAC: Added fstar for PopIII
def fstarofz_III(Astro_Parameters, Cosmo_Parameters, z, Mhlist):
    epsstar_ofz_III = Astro_Parameters.fstar_III * 10**(Astro_Parameters.dlog10epsstardz_III * (z-Astro_Parameters._zpivot_III) )
    if Cosmo_Parameters.Flag_emulate_21cmfast == False:
        return 2 * Cosmo_Parameters.OmegaB/Cosmo_Parameters.OmegaM * epsstar_ofz_III\
    /(pow(Mhlist/Astro_Parameters.Mc_III, -Astro_Parameters.alphastar_III) + pow(Mhlist/Astro_Parameters.Mc_III, -Astro_Parameters.betastar_III))
    elif Cosmo_Parameters.Flag_emulate_21cmfast == True:
        return Cosmo_Parameters.OmegaB/Cosmo_Parameters.OmegaM * epsstar_ofz_III/(pow(Mhlist/Astro_Parameters.Mc_III, -Astro_Parameters.alphastar_III))
    
    
def J_LW(Astro_Parameters, Cosmo_Parameters, sfrdIter, z, pop):
    #specific intensity, units of erg/s/cm^2/Hz/sr
    #for units to work, c must be in Mpc/s and proton mass in solar masses
    #and convert from 1/Mpc^2 to 1/cm^2
    
    Elw = (constants.Elw_eV * u.eV).to(u.erg).value
    deltaNulw = constants.deltaNulw #Hz
    speedLight = constants.c_Mpcs
    massProton = constants.mprotoninMsun
    redshiftFactor = 1.04 #max amount LW photons can redshift before being scattered, as in Visbal+1402.0882
    
    if pop == 3:
        Nlw = Astro_Parameters.N_LW_III
    elif pop == 2:
        Nlw = Astro_Parameters.N_LW_II

    zIntMatrix = np.linspace(z, redshiftFactor*(1+z)-1, 20)
    
    sfrdIterMatrix_LW = interpolate.interp1d(z, sfrdIter, kind = 'linear', bounds_error=False, fill_value=0)(zIntMatrix)
    
    if(Cosmo_Parameters.Flag_emulate_21cmfast==True):##HAC ACAUSAL: This if statement allows for acausal Mmol
        sfrdIterMatrix_LW = sfrdIter * np.ones_like(zIntMatrix) #HAC: This fixes J_LW(z) = int SFRD(z) dz' such that no z' dependence in the integral (for some reason 21cmFAST does this). Delete when comparing J_LW() with Visbal+14 and Mebane+17
        
    integrandLW = speedLight / 4 / np.pi
    integrandLW *= (1+z)**2 / cosmology.Hubinvyr(Cosmo_Parameters,zIntMatrix)
#    integrandLW *= (1+z)**3 / cosmology.Hubinvyr(Cosmo_Parameters,zIntMatrix) / (1+zIntMatrix) #HAC: delete this and comment above back in!!!
    integrandLW *= Nlw * Elw / massProton / deltaNulw
    integrandLW = integrandLW * sfrdIterMatrix_LW * (1 /u.Mpc**2).to(1/u.cm**2).value #broadcasting doesn't like augmented assignment operations (like *=) for some reason
    return np.trapz(integrandLW, x = zIntMatrix, axis = 0)


def SFRD_II_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z, z2):
    Mh = massVector
    
    HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(Mh), z)))
    SFRtab_currII = SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, z, z2)
    integrand_II = HMF_curr * SFRtab_currII * Mh
    return integrand_II


def SFRD_III_integrand(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, J21LW_interp, z, z2, vCB):
    Mh = massVector
    HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(Mh), z)))
    SFRtab_currIII = SFR_III(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, J21LW_interp, z, z2, vCB)
    integrand_III = HMF_curr * SFRtab_currIII * Mh
    return integrand_III


def SFR_II(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z, z2):
    "SFR in Msun/yr at redshift z. Evaluated at the halo masses Mh [Msun] of the HMF_interpolator, given Astro_Parameters"
    Mh = massVector
    
    #The FIXED/SHARP routine below only applies to Pop II, not to Pop III
    if Astro_Parameters.USE_POPIII == False:
        if(Astro_Parameters.FLAG_MTURN_FIXED == False):
            fduty = np.exp(-Matom(z)/Mh)
        elif(Astro_Parameters.FLAG_MTURN_SHARP == False): #whether to do regular exponential turn off or a sharp one at Mturn
            fduty = np.exp(-Astro_Parameters.Mturn_fixed/Mh)
        else:
            fduty = np.heaviside(Mh - Astro_Parameters.Mturn_fixed, 0.5)
    elif Astro_Parameters.USE_POPIII == True:
        fduty = np.exp(-Matom(z)/Mh)

    fstarM = fstarofz(Astro_Parameters, Cosmo_Parameters, z, Mh)
    fstarM = np.fmin(fstarM, Astro_Parameters.fstarmax)
    
    return dMh_dt(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, z)  * fstarM * fduty


def SFR_III(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, J21LW_interp, z, z2, vCB):
    "PopIII SFR in Msun/yr at redshift z. Evaluated at the halo masses Mh [Msun] of the HMF_interpolator, given Astro_Parameters"
    if(Astro_Parameters.USE_POPIII == False):
        return 0 #skip whole routine if NOT using PopIII stars
    else:
        Mh = massVector
        
        if(Cosmo_Parameters.Flag_emulate_21cmfast==False): #in 21cmfast it uses a backwarsd time z2>z, but in general it should not
            z2 = z    
        duty_matom_component = np.exp(-Mh/Matom(z2))
        fduty_III =  np.exp(-Mmol(Astro_Parameters, Cosmo_Parameters, J21LW_interp, z2, vCB)/Mh) * duty_matom_component
        
        fstarM_III = fstarofz_III(Astro_Parameters, Cosmo_Parameters, z, Mh)
        fstarM_III = np.fmin(fstarM_III, Astro_Parameters.fstarmax)
        
        return dMh_dt(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, Mh, z) * fstarM_III * fduty_III
    
    
def dMh_dt(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, massVector, z):
    'Mass accretion rate, in units of M_sun/yr'
    Mh = massVector
    
    if(Astro_Parameters.astromodel == False): #GALLUMI-like
        if(Astro_Parameters.accretion_model == False): #exponential accretion
            dMhdz = massVector * constants.ALPHA_accretion_exponential
            
        elif(Astro_Parameters.accretion_model == True): #EPS accretion
            
            Mh2 = Mh * constants.EPSQ_accretion
            indexMh2low = Mh2 < Mh.flatten()[0]
            Mh2[indexMh2low] = Mh.flatten()[0]
            
            sigmaMh = HMF_interpolator.sigmaintlog((np.log(Mh), z))
            sigmaMh2 = HMF_interpolator.sigmaintlog((np.log(Mh2), z))
            sigmaMh2[np.full_like(sigmaMh2, fill_value=True, dtype = bool) * indexMh2low] = 1e99
            
            growth = cosmology.growth(Cosmo_Parameters,z)
            dzgrow = z*0.01
            dgrowthdz = (cosmology.growth(Cosmo_Parameters,z+dzgrow) - cosmology.growth(Cosmo_Parameters,z-dzgrow))/(2.0 * dzgrow)
            dMhdz = - Mh * np.sqrt(2/np.pi)/np.sqrt(sigmaMh2**2 - sigmaMh**2) *dgrowthdz/growth * Cosmo_Parameters.delta_crit_ST
            
        else:
            print("ERROR! Have to choose an accretion model in Astro_Parameters (accretion_model)")
        Mhdot = dMhdz*cosmology.Hubinvyr(Cosmo_Parameters,z)*(1.0+z)
        return Mhdot

    elif(Astro_Parameters.astromodel == True): #21cmfast-like
        return Mh/Astro_Parameters.tstar*cosmology.Hubinvyr(Cosmo_Parameters,z)
    else:
        print('ERROR, MODEL is not defined')
        

def J_LW_Discrete(Astro_Parameters, Cosmo_Parameters, ClassCosmo, z, pop, rGreater, SFRD_II_interp, SFRD_III_cnvg_interp):
    #specific intensity, units of erg/s/cm^2/Hz/sr
    #for units to work, c must be in Mpc/s and proton mass in solar masses
    #and convert from 1/Mpc^2 to 1/cm^2
    
    Elw = (constants.Elw_eV * u.eV).to(u.erg).value
    deltaNulw = constants.deltaNulw
    massProton = constants.mprotoninMsun
    redshiftFactor = 1.04 #max amount LW photons can redshift before being scattered, as in Visbal+1402.0882
    
    rTable = np.transpose([Cosmo_Parameters.chiofzint(z)]) + rGreater
    rTable[rTable > Cosmo_Parameters.chiofzint(50)] = Cosmo_Parameters.chiofzint(50) #cut down so that nothing exceeds zmax = 50
    zTable = Cosmo_Parameters.zfofRint(rTable)
    
    ##HAC ACAUSAL: The below if statement allows for acausal Mmol
    if(Cosmo_Parameters.Flag_emulate_21cmfast==True):
        zTable = np.array([z]).T * np.ones_like(rTable) #HAC: This fixes J_LW(z) = int SFRD(z) dz' such that no z' dependence in the integral (for some reason 21cmFAST does this). Delete when comparing J_LW() with Visbal+14 and Mebane+17
        
    zMax = np.transpose([redshiftFactor*(1+z)-1])
    rMax = Cosmo_Parameters.chiofzint(zMax)
    
    c1 = (1+z)**2/4/np.pi
    
    if pop == 3:
        Nlw = Astro_Parameters.N_LW_III
        c2r = SFRD_III_cnvg_interp(zTable)
    elif pop == 2:
        Nlw = Astro_Parameters.N_LW_II
        c2r = SFRD_II_interp(zTable)
            
#    c2r *= Nlw * Elw / deltaNulw / massProton * (1 - np.heaviside(rTable - rMax, 1)) * (1 /u.yr/u.Mpc**2).to(1/u.s/u.cm**2).value #hard Heaviside cutoff, leads to instabilities & discontinuities
    c2r *= Nlw * Elw / deltaNulw / massProton * 0.5*(1 - np.tanh((rTable - rMax)/10)) * (1 /u.yr/u.Mpc**2).to(1/u.s/u.cm**2).value #smooth tanh cutoff, smoother function within 2-3% agreement with J_LW()
    return np.transpose([c1]), c2r

def dSFRDIII_dJ(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, J21LW_interp, z, z2, vCB):
    Mh = HMF_interpolator.Mhtab
    HMF_curr = np.exp(HMF_interpolator.logHMFint((np.log(Mh), z)))
    SFRtab_currIII = SFR_III(Astro_Parameters, Cosmo_Parameters, HMF_interpolator, HMF_interpolator.Mhtab, J21LW_interp, z, z2, vCB)
    integrand_III = HMF_curr * SFRtab_currIII * HMF_interpolator.Mhtab
    integrand_III *= Astro_Parameters.A_LW * Astro_Parameters.beta_LW * J21LW_interp(z)**(Astro_Parameters.beta_LW - 1)
    integrand_III *= -1 * Mmol_vcb(Astro_Parameters, Cosmo_Parameters, z, Cosmo_Parameters.vcb_avg)/ HMF_interpolator.Mhtab
    return np.trapz(integrand_III, HMF_interpolator.logtabMh)


def fesc_II(Astro_Parameters, Mh):
    "f_escape for a halo of mass Mh [Msun] given Astro_Parameters" #The pivot scale here for Pop II stars is at 1e10 solar masses
    return np.fmin(1.0, Astro_Parameters.fesc10 * pow(Mh/1e10,Astro_Parameters.alphaesc) )

def fesc_III(Astro_Parameters, Mh):
    "f_escape for a PopIII halo of mass Mh [Msun] given Astro_Parameters" #The pivot scale here for Pop III stars is at 1e7 solar masses
    return np.fmin(1.0, Astro_Parameters.fesc7_III * pow(Mh/1e7,Astro_Parameters.alphaesc_III) )
    
def vFit_2(vel2, aVel, bVel, cVel, dVel):
    #fitting 2 exponentials to SFRD(z | \delta_r, v_cb) / SFRD(z | \delta_r, v_avg)
    vel = 3*vel2
    return aVel * np.exp(-bVel * vel) + cVel* np.exp(-dVel * vel)

#Kept for reference purposes. Does not correct x_alpha as a function of Ts iteratively, but some old works don't either so this allows for comparison. Only used if FLAG_WF_ITERATIVE == False
def Salpha_exp(z, T, xe):
    "correction from Eq 55 in astro-ph/0608032, Tk in K evaluated for the IGM where there is small reionization (xHI~1 and xe<<1) during LyA coupling era"
    tau_GP_noreio = 3e5*pow((1+z)/7,3./2.)*(1-xe)
    gamma_Sobolev = 1.0/tau_GP_noreio
    return np.exp( - 0.803 * pow(T,-2./3.) * pow(1e-6/gamma_Sobolev,-1.0/3.0))

