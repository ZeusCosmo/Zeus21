"""
Bulk of the Zeus21 calculation: 
    - Determines Lyman-alpha and X-ray fluxes,
    - Evolves the cosmic-dawn IGM state (WF coupling and heating),
    - Computes the 21-cm global signal and the effective biases gammaR to determine the 21-cm power spectrum.

Authors: zeus21 v2 collaboration - June 2026
    Emily Bregou
    Hector Afonso G. Cruz
    Sarah Libanore
    Julian B. Muñoz
    Yonny Sklansky
    Emilie Thélie
    Alessandra Venditti
arXiv:2302.08506, arXiv:2306.09403, arXiv:2407.18294, Sklansky et al. (in prep)
"""

from . import cosmology
from . import constants

import numpy as np

from scipy import interpolate


from .sfrd import Z_init, SFRD_class, PopIII_relvel
from .reionization import reionization_global

from .SED import SED_LyA, SED_XRAY


class LyAlpha_class:
    """
    Determines Lyman-alpha properties and fluxes.

    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    AstroParams : AstroParams class
    HMFinterp : HMFinterp class
    z_Init : Z_init class, optional
        Initial redshift matrices for the calculation (see sfrd.py for details).
        Default is None.
    SFRD_Init : SFRD_class class, optional
        Initial star formation rate density for the calculation (see sfrd.py for details). 
        Default is None.

    Attributes
    ----------
    coeff1LyAzp : array 
        Redshift-dependent coefficient in the J_alpha flux computation, see Eq. 29 in arXiv:2302.08506.
    coeff2LyAzpRR_II : array
        Coefficient that multiplies the SFRD in the integral for J_alpha for Pop II stars, see Eq. 30 of arXiv:2302.08506. 
    coeff2LyAzpRR_III : array
        Coefficient that multiplies the SFRD in the integral for J_alpha for Pop III stars.

    """

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None, SFRD_Init = None):

        # if z_Init and SFRD_Init are not provided, we initialize them here. This allows us to avoid redundant computations if they were already initialized in the parent class and passed as arguments.
        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams) 

        if SFRD_Init is None:
            SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init) 

        self.coeff1LyAzp = (1+z_Init.zintegral)**2/(4*np.pi) # redshift-dependent coefficient in the J_alpha flux computation, see Eq. 29 in arXiv:2302.08506 

        nuLYA = np.geomspace(constants.freqLyA, constants.freqLyCont, 128) # frequencies to compute the SED at, between LyA and the Lyman limit.  We only consider these photons since above they are absorbed by the IGM through photoionization, and below they do not redshift into LyA. 
        sedLYAII_interp = interpolate.interp1d(nuLYA, SED_LyA(nuLYA, pop = 2), kind = 'linear', bounds_error = False, fill_value = 0) # interpolate LyA SED to compute the contribution of higher Lyman series photons that redshift into LyA after being emitted at higher frequencies.

        n_recArray = np.arange(0,constants.n_max_recycle-1 ) # array of n levels from which photons are emitted after recombinations
        zpCube, rCube, n_recCube = np.meshgrid(z_Init.zintegral, CosmoParams._Rtabsmoo, n_recArray, indexing='ij', sparse=True) # 3D cube for the recombination contribution to LyA. Dimensions are (z,R,n), where n is the Lyman series level from which photons are emitted after recombinations
        n_lineCube = n_recCube + 2 
        zmax_lineCube = (1+zpCube) * (1 - pow(1+n_lineCube,-2.0))/(1-pow(n_lineCube,-2.0) ) - 1.0 # maximum redshift Lyman series photons can redshift before falling into a Ly-n resonance

        nu_linezpCube = constants.freqLyCont * (1 - (1.0/n_lineCube)**2) 
        zGreaterCube = z_Init.zGreaterMatrix_nonan.reshape(len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1) # redefine this just for LyA routine, to have the right dimensions for the recombination contribution. Dimensions are (z,R,1), where z is the redshift at which we want to compute the flux, and R is the smoothing scale at which we want to compute the SFRD. We will be summing over n_recCube, so we need to have the same zGreater for all n's.
        nu_lineRRCube = nu_linezpCube * (1.+zGreaterCube)/(1+zpCube) # frequency at which photons emitted at the Lyman series lines are observed at redshift zGreaterCube
        
        eps_alphaRR_II_Cube = AstroParams.N_alpha_perbaryon_II/CosmoParams.mu_baryon_Msun  * sedLYAII_interp(nu_lineRRCube) # emissivity of Lyman-alpha photons from recombinations, converted from per SFR to per baryon by dividing by the mean mass per baryon in Msun, and multiplying by the number of LyA photons emitted per baryon in stars. We then multiply by the SED at the frequency at which these photons are observed at redshift zGreaterCube, to account for the fact that not all photons emitted at the Lyman series lines will redshift into LyA, but some will redshift into lower frequencies and be absorbed by dust or redshift out of the band.
        
        #the last nonzero index of the array is overestimated since only part of the spherical shell is within zmax_line. Correct by by dz/Delta z
        weights_recCube = np.heaviside(zmax_lineCube - zGreaterCube, 0.0)
        index_first0_weightsCube = np.where(np.diff(weights_recCube, axis = 1) == -1) #find index of last nonzero value. equals zero if two consecutive elements are 1 or 0, and -1 if two consecutive elements are [1,0]
        i0Z, i0R, i0N = index_first0_weightsCube
        weights_recCube[i0Z, i0R, i0N] *= (zmax_lineCube[i0Z, 0, i0N] - zGreaterCube[i0Z, i0R, 0])/ (zGreaterCube[i0Z, i0R+1, 0] - zGreaterCube[i0Z, i0R, 0])

        Jalpha_II = np.array(constants.fractions_recycle)[:len(n_recArray)].reshape(1,1,len(n_recArray)) * weights_recCube * eps_alphaRR_II_Cube # just resizing f_recycle; it is length 29,we only consider up to n=22

        LyAintegral_II = np.sum(Jalpha_II,axis=2) #sum over axis 2, over all possible n transitions, see Eq. 25 of arXiv:2302.08506
        self.coeff2LyAzpRR_II = CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_II * LyAintegral_II/ constants.yrTos/constants.Mpctocm**2 # This is the coefficient that multiplies the SFRD in the integral for J_alpha, see Eq. 30 of arXiv:2302.08506. It has dimensions of s^-1 cm^-3, so when multiplied by the SFRD in Msun/year/Mpc^3 and integrated over R, it gives the correct units of s^-1 cm^-3 for J_alpha. 

        if AstroParams.USE_POPIII:
            # if required, we repeat the same for Pop III stars, where we change the SED and number of LyA photons per baryon in stars. We use the same weights_recCube since they only depend on the redshift at which photons are emitted and observed
            sedLYAIII_interp = interpolate.interp1d(nuLYA, SED_LyA(nuLYA, pop = 3), kind = 'linear', bounds_error = False, fill_value = 0)
            eps_alphaRR_III_Cube = AstroParams.N_alpha_perbaryon_III/CosmoParams.mu_baryon_Msun  * sedLYAIII_interp(nu_lineRRCube)
            
            Jalpha_III = np.array(constants.fractions_recycle)[:len(n_recArray)].reshape(1,1,len(n_recArray)) * weights_recCube * eps_alphaRR_III_Cube
            LyAintegral_III = np.sum(Jalpha_III,axis=2)
            self.coeff2LyAzpRR_III = CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_III * LyAintegral_III/ constants.yrTos/constants.Mpctocm**2
        else:
            self.coeff2LyAzpRR_III = np.zeros_like(self.coeff2LyAzpRR_II)

        if UserParams.C2_RENORMALIZATION_FLAG:
            # If required, correct for nonlinearities in <(1+d)SFRD>
            # We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. 
            # We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).
            self.coeff2LyAzpRR_II = self.coeff2LyAzpRR_II * SFRD_Init._corrfactorEulerian_II.T
            if AstroParams.USE_POPIII:
                self.coeff2LyAzpRR_III = self.coeff2LyAzpRR_III * SFRD_Init._corrfactorEulerian_III.T
        

class Xrays_class:
    """
    Determines X-ray properties and fluxes.

    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    AstroParams : AstroParams class
    HMFinterp : HMFinterp class
    z_Init : Z_init class, optional
        Initial redshift matrices for the calculation (see sfrd.py for details).
        Default is None.
    SFRD_Init : SFRD_class class, optional
        Initial star formation rate density for the calculation (see sfrd.py for details). 
        Default is None.

    Attributes
    ----------
    atomfractions : array   
        Fraction of baryons in HI and HeI, assumed to just be the avg cosmic. Used to compute X-ray absorption.
    atomEnIon : array
        Threshold energies for HI and HeI, in eV.
    TAUMAX : float
        Maximum optical depth, cut to 0 after to avoid overflows.
    coeff1Xzp : array   
        Redshift-dependent coefficient in the X-ray flux computation, with extra factors to account for adiabatic cooling and the fact that we compute the integral in redshift instead of time.
    coeff2XzpRR_II : array
        Coefficient that multiplies the SFRD in the integral for the X-ray flux for Pop II stars
    coeff2XzpRR_III : array
        Coefficient that multiplies the SFRD in the integral for the X-ray flux for Pop III stars
    _GammaXray_II : array
        X-ray ionization rate for Pop II stars, in s^-1, see Eq. 37 in arXiv:2302.08506
    _GammaXray_III : array
        X-ray ionization rate for Pop III stars, in s^-1
    coeff_Gammah_Tx_II : array
        Coefficient that multiplies the X-ray ionization rate to get the X-ray heating rate for Pop II stars, in K/s, see Eq. 41 in arXiv:2302.08506.
    coeff_Gammah_Tx_III : array 
        Coefficient that multiplies the X-ray ionization rate to get the X-ray heating rate for Pop III stars, in K/s
    Gammaion_II : array
        X-ray ionization rate for Pop II stars, in s^-1
    Gammaion_III : array
        X-ray ionization rate for Pop III stars, in s^-1
    _xe_avg_ad : array
        Average ionization fraction of the IGM from adiabatic cooling and recombinations
    _xe_avg: array
        Average ionization fraction of the IGM, including both the contribution from UV photons and the partial ionization from X-rays
    _fheat : array
        Fraction of X-ray energy that goes into heating, as opposed to ionization
    Gammaheat_II : array
        X-ray heating rate for Pop II stars, in K/s
    Gammaheat_III : array
        X-ray heating rate for Pop III stars, in K/s
    Tk_xray : array
        Average kinetic temperature of the IGM from X-ray heating, in K
    Tk_ad : array
        Average kinetic temperature of the IGM from adiabatic cooling only, in K
    Tk_avg : array
        Average kinetic temperature of the IGM, including both adiabatic cooling and X-ray heating, in K
    sigma_HI : function
        Cross section for X-ray absorption by HI, as a function of energy in eV, in cm^2
    sigma_HeI : function
        Cross section for X-ray absorption by HeI, as a function of energy in eV, in cm^2
    """

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None, SFRD_Init = None):

        # if z_Init and SFRD_Init are not provided, we initialize them here. This allows us to avoid redundant computations if they were already initialized in the parent class and passed as arguments.
        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams) 

        if SFRD_Init is None:
            SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init) 

        self.atomfractions = np.array([1,CosmoParams.x_He]) #fraction of baryons in HI and HeI, assumed to just be the avg cosmic
        self.atomEnIon = np.array([constants.EN_ION_HI, constants.EN_ION_HeI]) #threshold energies for each, in eV
        self.TAUMAX=100. #max optical depth, cut to 0 after to avoid overflows

        _Energylist = AstroParams.Energylist # list of energies at which we compute the SED and optical depth. We use a fixed list of energies instead of integrating over energy, to speed up the computation
        Nzinttau = np.floor(10*UserParams.precisionboost).astype(int) # number of redshift points to compute the optical depth integral. We use a fixed number of points instead of integrating over redshift, to speed up the computation

        zGreaterCube = z_Init.zGreaterMatrix_nonan.reshape(len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1, 1) #redefine this just for x-ray routine

        self.coeff1Xzp = -2/3 * z_Init.zintegral * z_Init.dlogzint / cosmology.Hubinvyr(CosmoParams,z_Init.zintegral) / (1+z_Init.zintegral) * (1+z_Init.zintegral)**2
        self.coeff1Xzp = self.coeff1Xzp / (1+z_Init.zintegral)**2 * constants.yrTos # this accounts for adiabatic cooling. compensated by the inverse at the end

        zpCube, rCube, eCube, zPPCube = np.meshgrid(z_Init.zintegral, CosmoParams._Rtabsmoo, _Energylist, np.arange(Nzinttau), indexing='ij', sparse=True) # 4D cube for the X-ray contribution. Dimensions are (z,R,E,z'), where z is the redshift at which we want to compute the flux, R is the smoothing scale at which we want to compute the SFRD, E is the energy at which we want to compute the SED and optical depth, and z' is the redshift at which we want to compute the optical depth integral
        currentEnergyTable = eCube * (1+zGreaterCube) / (1+zpCube) # Energy at which photons observed at redshift zGreaterCube were emitted at redshift zpCube, since E' = E(1+z')/(1+z). 
        SEDCube = SED_XRAY(AstroParams, currentEnergyTable, pop = 2) # SED of our X-ray sources for popII
        SEDCube_III = SED_XRAY(AstroParams, currentEnergyTable, pop = 3) # SED  of our X-ray sources for popIII, we compute it even if we don't use it, to speed up the computation in case we do use it later.

        ######## Broadcasted routine to find X-ray optical depths, modeled after but does not use xrays.optical_depth

        zPPCube = np.array([np.linspace(np.transpose([z_Init.zintegral]), z_Init.zGreaterMatrix, Nzinttau, axis = 2)])
        zPPCube = zPPCube.reshape(len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1, Nzinttau) # to have 4D dimensions, default shape = (64, 45, 1, 10) 

        ePPCube = eCube * (1+ zPPCube) / (1+zpCube) # E'' = E(1+z'')/(1+z)
        sigmatot = self.atomfractions[0] * self.sigma_HI(ePPCube) # cross section for X-ray absorption. determined by the energy at which they are absorbed. We multiply by the atom fractions to get the total cross section per baryon
        sigmatot += self.atomfractions[1] * self.sigma_HeI(ePPCube) # we only consider HeI since HeII is negligible at the redshifts we're interested in, and it has a much higher ionization energy so it does not contribute much to the absorption of X-rays

        opticalDepthIntegrand = 1 / cosmology.HubinvMpc(CosmoParams, zPPCube) / (1+zPPCube) * sigmatot * cosmology.n_H(CosmoParams, zPPCube) * constants.Mpctocm # this uses atom fractions of 1 for HI and x_He for HeI
        tauCube = np.trapezoid(opticalDepthIntegrand, zPPCube, axis = 3) # integrate over z' to get the optical depth. This gives us a 3D cube with dimensions (z, R, E), where z is the redshift at which we want to compute the flux, R is the smoothing scale at which we want to compute the SFRD, and E is the energy at which we want to compute the SED and optical depth.

        # cap tau to avoid overflows in the exponential
        indextautoolarge = np.array(tauCube>=self.TAUMAX) 
        tauCube[indextautoolarge] = self.TAUMAX

        if CosmoParams.Flag_emulate_21cmfast:
            # if we're emulating 21cmfast, we use a step function for the absorption
            weights_X_zCube = np.heaviside(1.0 - tauCube, 0.5)
        else:
            weights_X_zCube = np.exp(-tauCube)
            
        SEDCube = SEDCube[:,:,:,0] # rescale dimensions of energy and SED cubes back to 3D, so we can integrate over energy
        SEDCube_III = SEDCube_III[:,:,:,0] # same for popIII
        
        eCube = eCube[:,:,:,0]
        ######## end of optical depth routine

        JX_coeffsCube = SEDCube * weights_X_zCube # this is the coefficient that multiplies the SFRD in the integral for the X-ray flux, before integrating over energy. It has dimensions of number of photons per energy per baryon, multiplied by the absorption factor, so when multiplied by the SFRD in Msun/year/Mpc^3 and integrated over R and E, it gives the correct units of number of photons per second per baryon for the X-ray flux. We keep it as a cube for now to integrate over energy later.
        JX_coeffsCube_III = SEDCube_III * weights_X_zCube # same for popIII

        sigma_times_en = self.atomfractions[0] * self.sigma_HI(eCube) * (eCube - self.atomEnIon[0]) # we multiply the cross section by (E - E_ion) to account for the fact that only the energy above the ionization threshold goes into heating and ionization, while the rest is lost to secondary electrons. We also multiply by the atom fractions to get the total contribution per baryon, instead of per Hydrogen nucleus.
        sigma_times_en += self.atomfractions[1] * self.sigma_HeI(eCube) * (eCube - self.atomEnIon[1]) # same for HeI
        sigma_times_en /= np.sum(self.atomfractions) # to normalize per baryon, instead of per Hydrogen nucleus HI and HeII separate. Notice Energy (and not Energy'), since they get absorbed at the zp frame
        
        xrayEnergyTable = np.sum(JX_coeffsCube * sigma_times_en * eCube * AstroParams.dlogEnergy,axis=2) # integrate over energy to get the coefficient that multiplies the SFRD in the integral for the X-ray flux
        self.coeff2XzpRR_II = np.nan_to_num(CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_II * xrayEnergyTable * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST, nan = 0) # see Eq. 39 in arXiv:2302.08506. We multiply by normLX_CONST to convert from number of photons to energy, and by 1/Mpc^2 to convert from per area to per volume, since the SFRD is in Msun/year/Mpc^3 and we want the X-ray flux in energy per second per baryon
        
        if AstroParams.USE_POPIII:
            # same for popIII
            xrayEnergyTable_III = np.sum(JX_coeffsCube_III * sigma_times_en * eCube * AstroParams.dlogEnergy,axis=2)
            self.coeff2XzpRR_III = np.nan_to_num(CosmoParams._Rtabsmoo * CosmoParams._dlogRR * SFRD_Init.SFRDbar2D_III * xrayEnergyTable_III * (1.0/constants.Mpctocm**2.0) * constants.normLX_CONST, nan = 0)
        else:
            self.coeff2XzpRR_III = np.zeros_like(self.coeff2XzpRR_II)

        if UserParams.C2_RENORMALIZATION_FLAG:
            # if required, correct for nonlinearities in <(1+d)SFRD>, only if doing nonlinear stuff. 
            # We're assuming that (1+d)SFRD ~ exp(gamma*d), so the "Lagrangian" gamma was gamma-1. 
            # We're using the fact that for a lognormal variable X = log(Z), with  Z=\gamma \delta, <X> = exp(\gamma^2 \sigma^2/2).
            self.coeff2XzpRR_II = self.coeff2XzpRR_II* SFRD_Init._corrfactorEulerian_II.T
            if AstroParams.USE_POPIII:
                self.coeff2XzpRR_III = self.coeff2XzpRR_III * SFRD_Init._corrfactorEulerian_III.T

        self._GammaXray_II = self.coeff1Xzp * np.sum( self.coeff2XzpRR_II ,axis=1) # eq. 37 in 2302.08506; notice units are modified (eg 1/H) so it's simplest to sum
        self._GammaXray_III = self.coeff1Xzp * np.sum( self.coeff2XzpRR_III ,axis=1) # same for popIII
        
        fion = 0.4 * np.exp(-cosmology.xefid(CosmoParams, z_Init.zintegral)/0.2) # partial ionization from Xrays. Fit to Furlanetto&Stoever
        atomEnIonavg = (self.atomfractions[0] *  self.atomEnIon[0] + self.atomfractions[1] *  self.atomEnIon[1]) / (self.atomfractions[0] + self.atomfractions[1] ) # convert from 1/n_H to 1 / n_b 

        self.coeff_Gammah_Tx_II = -AstroParams.L40_xray * constants.ergToK * (1.0+z_Init.zintegral)**2  # coefficient to convert from Gamma_X to T_X, last factors accounts for adiabatic cooling. compensated by the inverse at zp in coeff1Xzp. Minus because integral goes from low to high z, but we'll be summing from high to low everywhere.
        self.coeff_Gammah_Tx_III = -AstroParams.L40_xray_III * constants.ergToK * (1.0+z_Init.zintegral)**2 # same for popIII
        
        self.Gammaion_II = self.coeff_Gammah_Tx_II *constants.KtoeV * self._GammaXray_II * fion/atomEnIonavg * 3/2 # ionization rate from X-rays for Pop II stars, in s^-1. We multiply by fion to account for the fact that only a fraction of the energy goes into ionization, and divide by the average ionization energy per baryon to convert from energy to number of ionizations ; atomEnIonavg makes it approximate. No adiabatic cooling (or recombinations) so no 1+z factors. Extra 3/2 bc temperature has a 2/3
        self.Gammaion_III = self.coeff_Gammah_Tx_III *constants.KtoeV * self._GammaXray_III * fion/atomEnIonavg * 3/2 # same for popIII
        
        # TODO: Improve model for xe

        self.xe_avg_ad = cosmology.xefid(CosmoParams, z_Init.zintegral) # average ionization fraction from adiabatic cooling and recombinations, without X-ray ionization.
        self.xe_avg = self.xe_avg_ad + np.cumsum((self.Gammaion_II+self.Gammaion_III)[::-1])[::-1] # average ionization fraction including X-ray ionization
        if CosmoParams.Flag_emulate_21cmfast:
            # if we're emulating 21cmfast, we use a fixed ionization fraction
            self.xe_avg = 2e-4 * np.ones_like(self.Gammaion_II) 

        self.xe_avg = np.fmin(self.xe_avg, 1.0-1e-9)

        # heat from Xrays
        self._fheat = pow(self.xe_avg,0.225)
        self.coeff1Xzp *= self._fheat # since this is what we use for the power spectrum, we need to upate it
        self.Gammaheat_II = self._GammaXray_II * self._fheat
        self.Gammaheat_III = self._GammaXray_III * self._fheat

        # Computing avg kinetic temperature as sum of adiabatic & xray temperature
        self.Tk_xray = self.coeff_Gammah_Tx_II * np.cumsum(self.Gammaheat_II[::-1])[::-1] + self.coeff_Gammah_Tx_III * np.cumsum(self.Gammaheat_III[::-1])[::-1] # in K, cumsum reversed because integral goes from high to low z. Only heating part
        self.Tk_ad = cosmology.Tadiabatic(CosmoParams, z_Init.zintegral)
        if CosmoParams.Flag_emulate_21cmfast:
        # if we're emulating 21cmfast, we use a fixed kinetic temperature, since they use recfast, so their 'cosmo' temperature is slightly off
            self.Tk_ad*=0.95 
        
        self.Tk_avg = self.Tk_ad + self.Tk_xray
        

    def sigma_HI(self, Energyin):
        """
        Cross section for Xray absorption for neutral HI, from astro-ph/9601009

        Parameters
        ----------
        Energyin: float
            Energy in eV

        Returns
        -------
        float
            Cross section in cm^2.
        """

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
        """
        Cross section for Xray absorption for neutral HI

        Parameters
        ----------
        Energyin: float
            Energy in eV

        Returns
        -------
        float
            Cross section in cm^2.
        """

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
    """
    Loops through SFRD integrals and accounts for LyA coupling and Xray heating to obtain the average T21 and the coefficients for its power spectrum

    Parameters
    ----------
    UserParams : object
        User-defined parameters
    CosmoParams : object
        Cosmological parameters
    AstroParams : object
        Astrophysical parameters
    HMFinterp : object
        Halo mass function interpolator

    Attributes
    ----------
    z_Init : object
        Redshift tables for the calculation (see sfrd.py for details)
    SFRD_Init : object
        Initial star formation rate density for the calculation (see sfrd.py for details)
    USE_POPIII : bool
        Whether to include Pop III stars in the calculation or not, determined by AstroParams
    relvel : object or None
        Relative velocity between baryons and dark matter, which affects the SFRD and therefore the LyA and X-ray fluxes. Only computed if USE_POPIII is True
    LyA : object
        Lyman-alpha anisotropies, which depend on the SFRD and the redshift tables; see LyAlpha_class for details
    Xrays : object
        X-ray anisotropies, which depend on the SFRD and the redshift tables; see Xrays_class for details
    ReioGlobal : object
        Global reionization history, which depends on the SFRD and the redshift tables; see reionization.py for details
    xHI_avg : array
        Average neutral hydrogen fraction volume-weighted, computed from the global reionization history 
    T21avg : array
        Average 21cm brightness temperature
    tau_reio_val : float
        Optical depth to reionization, computed from the global reionization history and the average neutral hydrogen fraction
    __ getattr__ : method
        This method allows us to access the attributes of the classes that we initialized directly from the get_T21_coefficients class, without having to specify which class they come from
    evolve_T21_fields : method
        Compute evolution of the LyA flux, LyA coupling coefficient, color temperature and spin temperature
    Jalpha_avg : array  
        Average LyA flux at all redshifts and radii
    _coeff_Ja_xa_0 : array
        Normalization of the LyA flux at all redshifts
    coeff_Ja_xa : array
        LyA flux corrected with Hirata2006 prescription
    xa_avg : array 
        LyA coupling coefficient
    TCMB : array
        CMB temperature at all redshifts 
    invTcol_avg : array
        Inverse of the color temperature (equal 1/Tk)
    _invTs_avg : array 
        Inverse of the spin temperature
    tau_reio : method
        Compute the optical depth to reionization 
    Salpha_exp : method
        Hirata2006 correction to the LyA flux (Eq 55 in astro-ph/0608032)
    """

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init =None, SFRD_Init=None):

        # if z_Init and SFRD_Init are not provided, we initialize them here. This allows us to avoid redundant computations if they were already initialized in the parent class and passed as arguments.
        if z_Init is None:
            # to perform cross-correlation studies, the redshift array has to be the same as in zeus21
            self.z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams) 
        else:
            self.z_Init = z_Init

        if SFRD_Init is None:
            self.SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init) 
        else: 
            self.SFRD_Init = SFRD_Init
        
        # Computing lambdas in velocity anisotropies
        # The SFRD vcb dependence is delta independent, therefore we compute quantities below for a variety of R's and delta_R = 0
        self.USE_POPIII = AstroParams.USE_POPIII
        if self.USE_POPIII:
            self.relvel = PopIII_relvel(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init)
        else:
            self.relvel = None

        # Lyman-Alpha Anisotropies
        self.LyA = LyAlpha_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init)

        ### X-ray Anisotropies
        self.Xrays = Xrays_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init)

        # Computing free-electron fraction and Salpha correction factors in the Bulk IGM  
        self.evolve_T21_fields(UserParams, CosmoParams)   
    
        # Reionization
        self.ReioGlobal = reionization_global(CosmoParams, AstroParams, HMFinterp, self.z_Init, self.SFRD_Init, PRINT_SUCCESS=False)
        self.xHI_avg = 1. - self.ReioGlobal.ion_frac ### TODO this one is volume weighted for now

        # Compute the 21cm Global Signal
        self.T21avg = cosmology.T021(CosmoParams,self.z_Init.zintegral) * self.xa_avg/(1.0 + self.xa_avg) * (1.0 - self.T_CMB * self.invTcol_avg) * self.xHI_avg #TODO

        self.tau_reio_val = self.tau_reio(CosmoParams, self.z_Init.zintegral, self.xHI_avg)


    def __getattr__(self, name):
        """
        Access the attributes of the classes that we initialized directly from the get_T21_coefficients class, without having to specify which class they come from

        Parameters
        ----
        name: str
            Name of the attribute to get
        
        Returns
        -------
        attribute            
        The attribute with the given name, if it exists in any of the classes that we initialized. If it does not exist in any of them, raises an AttributeError.
        """

        list_of_cls = [self.z_Init, self.SFRD_Init, self.LyA, self.Xrays, self.ReioGlobal]

        if self.USE_POPIII:
            list_of_cls += [self.relvel]
        for cls in list_of_cls:
            try:
                return getattr(cls, name)
            except AttributeError:
                pass

        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")


    def evolve_T21_fields(self, UserParams, CosmoParams):  
        """
        Compute evolution of the LyA flux, LyA coupling coefficient, color temperature and spin temperature

        Parameters
        ----------
        UserParams : object
            User-defined parameters
        CosmoParams : object
            Cosmological parameters

        Returns
        -------
        None
        """ 

        # LyA stuff to find components of Salpha correction factor
        self.Jalpha_avg = self.LyA.coeff1LyAzp*np.sum(self.LyA.coeff2LyAzpRR_II + self.LyA.coeff2LyAzpRR_III,axis=1) #units of 1/(cm^2 s Hz sr)

        # CMB temperature
        self.T_CMB = cosmology.Tcmb(CosmoParams.ClassCosmo, self.z_Init.zintegral)

        _tau_GP = 3./2. * cosmology.n_H(CosmoParams,self.z_Init.zintegral) * constants.Mpctocm / cosmology.HubinvMpc(CosmoParams,self.z_Init.zintegral) * (constants.wavelengthLyA/1e7)**3 * constants.widthLyAcm * (1.0 - self.Xrays.xe_avg)  #~3e5 at z=6

        if CosmoParams.Flag_emulate_21cmfast:
            # 21cmFAST multiplies by N0 (all baryons) instead of NH0
            _tau_GP/=CosmoParams.f_H 

        # compute correction coefficients from Hirata2006
        _xiHirata = pow(_tau_GP*1e-7,1/3.)*pow(self.Xrays.Tk_avg,-2./3)
        _factorxi = (1.0 + constants.a_Hirata*_xiHirata + constants.b_Hirata * _xiHirata**2 + constants.c_Hirata * _xiHirata**3) 


        #prefactor without the Salpha correction from Hirata2006
        if CosmoParams.Flag_emulate_21cmfast:
            # 21cmFAST uses a fixed (and slightly ~10% off) value.
            self._coeff_Ja_xa_0 = 1.66e11/(1+self.z_Init.zintegral) 
        else:
            self._coeff_Ja_xa_0 = 8.0*np.pi*(constants.wavelengthLyA/1e7)**2 * constants.widthLyA * constants.Tstar_21/(9.0*constants.A10_21*self.T_CMB) #units of (cm^2 s Hz sr), convert from Ja to xa. should give 1.81e11/(1+z_Init.zintegral) for Tcmb_0=2.725 K

        self.coeff_Ja_xa = self._coeff_Ja_xa_0 * self.Salpha_exp(self.z_Init.zintegral, self.Xrays.Tk_avg, self.Xrays.xe_avg)

        # LyA flux, see Eq. 27 in 2302.08506
        self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg
        
        # color temperature 
        self.invTcol_avg = 1.0 / self.Xrays.Tk_avg

        # spin temperature
        self._invTs_avg = (1.0/self.T_CMB+self.xa_avg*self.invTcol_avg)/(1+self.xa_avg)

        if UserParams.FLAG_WF_ITERATIVE: 
            # iteration routine to find Tcolor and Ts
            _invTs_tryfirst = 1.0/self.T_CMB

            while(np.sum(np.fabs(_invTs_tryfirst/self._invTs_avg - 1.0))>0.01): #no more than 1% error total
                _invTs_tryfirst = self._invTs_avg

                #update xalpha
                _Salphatilde = (1.0 - 0.0632/self.Xrays.Tk_avg + 0.116/self.Xrays.Tk_avg**2 - 0.401/self.Xrays.Tk_avg*self._invTs_avg + 0.336*self._invTs_avg/self.Xrays.Tk_avg**2)/_factorxi
                self.coeff_Ja_xa = self._coeff_Ja_xa_0 * _Salphatilde
                self.xa_avg = self.coeff_Ja_xa * self.Jalpha_avg

                #and Tcolor^-1
                self.invTcol_avg = 1.0/self.Xrays.Tk_avg + constants.gcolorfactorHirata * 1.0/self.Xrays.Tk_avg * (_invTs_tryfirst - 1.0/self.Xrays.Tk_avg)

                # finally Ts^-1
                self._invTs_avg = (1.0/self.T_CMB+self.xa_avg * self.invTcol_avg)/(1+self.xa_avg)

        
    def tau_reio(self, CosmoParams, zlist, xHI):
        """
        Compute the optical depth to reionization 

        Parameters
        ----------
        CosmoParams : object
            Cosmological parameters
        zlist : list
            Redshifts
        xHI : array 
            Neutral fraction as function of redshift
            
        Returns
        -------
        tau_reio : list
        """ 

        # set EoR end
        _lowestz = np.min(zlist)
        _zlistlowz = np.linspace(0,_lowestz,100)

        _nelistlowz = cosmology.n_H(CosmoParams,_zlistlowz)*(1 + CosmoParams.x_He + CosmoParams.x_He * np.heaviside(constants.zHeIIreio - _zlistlowz,0.5)) # free electrons in post-EoR, assume HeII at z=4, can be varied with zHeIIreio

        _distlistlowz = 1.0/cosmology.HubinvMpc(CosmoParams,_zlistlowz)/(1+_zlistlowz) # comoving distances

        # integrate z < zmin (post-reio), assuming xHI=0
        _lowzint = constants.sigmaT * np.trapezoid(_nelistlowz*_distlistlowz,_zlistlowz) * constants.Mpctocm

        xHIint = np.fmin(np.fmax(xHI,0.0),1.0) # cap neutral fraction, at min 0%, at max 100%
        _zlisthiz = zlist
        _nelistlhiz = cosmology.n_H(CosmoParams,_zlisthiz) * (1 + CosmoParams.x_He) * (1.0 - xHIint) # free electrons in EoR
        _distlisthiz = 1.0/cosmology.HubinvMpc(CosmoParams,_zlisthiz)/(1+_zlisthiz) # comoving distances

        # integrate z > zmin
        _hizint = constants.sigmaT * np.trapezoid(_nelistlhiz*_distlisthiz,_zlisthiz) * constants.Mpctocm

        tau_reio = (_lowzint + _hizint)

        return tau_reio


    def Salpha_exp(self, z, T, xe):
        """
        Hirata2006 correction to the LyA flux (Eq 55 in astro-ph/0608032). This function is used to initialize the xalpha computation, but then overwritten. Only used if FLAG_WF_ITERATIVE == False

        Parameters
        ----------
        z : float
            Redshifts
        T : float 
            Temperature in K
        xe : float
            Free electron fraction with small reionization (xe << 1)
            
        Returns
        -------
        Salpha : float
            Correction to the LyA flux
        """ 

        tau_GP_noreio = 3e5*pow((1+z)/7,3./2.)*(1-xe)
        gamma_Sobolev = 1.0/tau_GP_noreio

        Salpha = np.exp( - 0.803 * pow(T,-2./3.) * pow(1e-6/gamma_Sobolev,-1.0/3.0))

        return Salpha
