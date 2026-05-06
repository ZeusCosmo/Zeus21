"""

Compute Star Formation Histories with Burstiness.

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2026

Edited by Sarah Libanore
BGU - April 2026 

"""

from .sfrd import * 

class SFH_class:

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, tage, zobs, z_Init = None, SFRD_Init = None):
        "Returns the star formation history at age tage [in Myr] of a galaxy in a halo of mass Mh at age tage, in Msun/yr"    

        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams)  

        if SFRD_Init is None:
            SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init) 

        self.SFH_II = self.SFH(CosmoParams, AstroParams, HMFinterp, SFRD_Init, tage, zobs, pop = 2)

        if AstroParams.USE_POPIII:
            raise ValueError('Burstiness is not implemented for PopIII')


    def SFH(self, CosmoParams, AstroParams, HMFinterp, SFRD_Init, tage, zobs, pop):

        tobs = CosmoParams.tageofzMyr(zobs)

        _tearlier = np.fmax(0.0,tobs-tage) #time before the observation
        zage = CosmoParams.zfoftageMyr(_tearlier) #z of the earlier times 
        massVector = HMFinterp.Mhtab #has to be the same due to the meanSFRnormalization below

        ###ASDASD TYTY - TODO this is just for comparing w bagpipes one run
        if(AstroParams.FLAG_COMPARE_BAGPIPES == True):
            texp=-120 #Myr, minus because backwards
            Mstar = 3e7*(massVector/7e9)**1.5 #made up but fits the usual power-law
            Lbox = 1000 #Myr, age of universe (so it integrates to Mstar)
            return np.outer(Mstar/(texp*1e6) * np.exp(-SFRD_Init.Matom(zobs)/massVector) * AstroParams.mean_SFR_normalization , (np.exp(tage/texp) / (np.exp((Lbox)/texp) -1)) )  #in Msun/yr

        ### This is a decent approximation,but only for exponential accretion
        alphatime_invMyr = constants.ALPHA_accretion_exponential * cosmology.Hubinvyr(CosmoParams, zage) * (1+zage) * 1e6
        Mhhistory = np.outer(massVector, np.exp(-alphatime_invMyr * tage))

        # Get the mass accretion rate at all the past redshifts z
        dMhdot = SFRD_Init.dMh_dt(CosmoParams, AstroParams, HMFinterp, Mhhistory, zage) #in Msun/yr

        fstar = self.fstarofz_scaled_Mz(AstroParams, CosmoParams, SFRD_Init, zage, Mhhistory, pop)

        if(AstroParams.FLAG_RENORMALIZE_AVG_SFH==True):
            _meanSFRnormalization = self._get_mean_SFR_normalization(AstroParams, HMFinterp.Mhtab) #normalization of the SFR, in Msun/yr, at each Mh
        else:
            _meanSFRnormalization = 1.0 #no normalization, just return the SFR

        SFH_val = fstar * dMhdot * _meanSFRnormalization[:, None] #in Msun/yr

        return SFH_val


    def fstarofz_scaled_Mz(self, AstroParams, CosmoParams, SFRD_Init, z, Mhlist, pop):
        'Approximates fstarofz so its not ran over a huge array Nm x Nz, but only over Nm and Nz and multiplied. Exact for M<Mc (which matters most anyway), offset can be reabsorbed in beta*'

        if pop == 2:
            eps = AstroParams.epsstar 
            dlog10eps = AstroParams.dlog10epsstardz
            zpiv = AstroParams._zpivot
            Mc = AstroParams.Mc
            alphastar = AstroParams.alphastar
            betastar = AstroParams.betastar
            fstarmax = AstroParams.fstarmax

        fofMh = SFRD_Init.fstar_ofz(CosmoParams, z[0], Mhlist[:,0], eps, dlog10eps, zpiv, Mc, alphastar, betastar, fstarmax)
        fofz = np.pow(Mhlist[0]/Mhlist[0,0],AstroParams.alphastar) 
                
        return np.outer(fofMh,fofz) #this is a very good approximation for the fstarofz function, but only for exponential accretion


    def sigmaPSD_at_Mh(self, AstroParams, Mh):

        "Returns the sigma of the power spectrum of lnSFR at Mh, in units of ln(SFR), so to convert to log10(SFR) multiply by np.log(10)"
        Mh = np.atleast_1d(Mh)
        sigma_at_Mh = AstroParams.sigmaPSD + AstroParams.dsigmaPSDdlog10Mh * np.log10(Mh/1e10)

        return np.fmin(np.fmax(sigma_at_Mh,AstroParams._minsigmaPSD),AstroParams._maxsigmaPSD) #make sure it's within the limits set by UserParams
    
    def tauPSD_at_Mh(self, AstroParams, Mh):

        "Returns the tau of the power spectrum of lnSFR at Mh, in Myr"
        Mh = np.atleast_1d(Mh)
        tau_at_Mh = AstroParams.tauPSD * 10**(AstroParams.dlog10tauPSDdlog10Mh * np.log10(Mh/1e10))

        return np.fmin(np.fmax(tau_at_Mh,AstroParams._mintauPSD),
        AstroParams._maxtauPSD) #make sure it's within the limits set by UserParams


    def PowerlnSFR(self, AstroParams, omega, Mh):
        "Returns the power spectrum of lnSFR - damped (alpha=2) random walk, with a timescale tau in Myr, and sigma (no units) the amplitude of the walk. "
        "As in 2410.21409 (and refs therein), but notice lnSFR and not log10SFR, can convert to other references by sigma->sigma*np.log(10)"
        omega = np.atleast_1d(omega) #make sure omega is a vector
        Mh = np.atleast_1d(Mh) #make sure Mh is a vector
        sigma_at_Mh = self.sigmaPSD_at_Mh(AstroParams, Mh) 
        tau_at_Mh = self.tauPSD_at_Mh(AstroParams, Mh)
        _tau_times_omega = np.outer(omega,tau_at_Mh) #omega is a vector length of omega, tau has the Mh length
        return (sigma_at_Mh**2 * tau_at_Mh / (1.0 + _tau_times_omega**2.0)).T # NM x Nomega; secretely there's a 1*Myr in the amplitude
    
    def Wink_TH(self,omega, T):
        "Returns a tophat temporal window function for a given frequency omega and timescale T"
        x = omega*T/2 + 1e-16
        return np.sin(x) / (x)
    
    def Variance_of_lnSFR(self, AstroParams, T, Mh):
        "Returns the root mean square of lnSFR when averaged over a timescale T, basically integrate Power times wink**2"

        omegalist = np.logspace(np.log10(AstroParams._omegamin),np.log10(AstroParams._omegamax), 999) # in 1/Myr
        power = self.PowerlnSFR(AstroParams, omegalist, Mh)
        wink = self.Wink_TH(omegalist, T)

        return np.trapezoid(power * np.abs(wink)**2, omegalist) *2/(2*np.pi) 
    
    def _get_mean_SFR_normalization(self, AstroParams, Mh):
        "Returns the boost to <SFR> due to stochasticity, i.e. the ratio of <SFR> to SFR(Mh) with no burstiness"
        _varlnSFR = self.Variance_of_lnSFR(AstroParams, 0.,Mh) #T=0 since it's at integrated over all timescales
        meanSFRnormalization = np.exp(_varlnSFR/2.) #<e^d> = <e^sigma^2/2> for a gaussian d
        return meanSFRnormalization


    def _get_PowerSFR_NL_FFT_vectorized(self, AstroParams, Mh_array):
        '''
        This is the power spectrum of SFR, which is nonlinearly related to that of lnSFR. 
        We obtain it thru FFTing the correlation function of lnSFR, which is a damped random walk with timescale tau and amplitude sigma.
        Mh_array is an array of halo masses, shape (NMhs,)
        Returns omegalist and powerNL, where powerNL is the power spectrum of lnSFR for all masses in Mh_array
        '''
        
        Mh_array = np.atleast_1d(Mh_array)  # Ensure Mh_array is a numpy array
        # dt is the time resolution for FFT, Nfft is the number of points in FFT
        dt = AstroParams._dt_FFT
        Nfft = AstroParams._N_FFT
        half_Nfft = Nfft // 2
        t_corr = dt * np.arange(-half_Nfft, Nfft - half_Nfft)
        
        # Vectorize the parameter calculations
        sigma_array = self.sigmaPSD_at_Mh(AstroParams, Mh_array)  # Shape: (NMhs,)
        tau_array = self.tauPSD_at_Mh(AstroParams, Mh_array)     # Shape: (NMhs,)
        
        # Broadcast for correlation function calculation
        t_corr_2d = t_corr[np.newaxis, :]  # Shape: (1, Nfft)
        tau_2d = tau_array[:, np.newaxis]   # Shape: (NMhs, 1)
        sigma_2d = sigma_array[:, np.newaxis]  # Shape: (NMhs, 1)
        
        # Vectorized correlation function
        corrF = np.exp(-np.abs(t_corr_2d)/tau_2d) * sigma_2d**2/(2.0)
        corrFNL = np.exp(corrF) - 1.0  # Shape: (NMhs, Nfft)
        
        # FFT along the time axis for all masses at once
        powerNL = np.fft.rfft(corrFNL, axis=1) * dt  # Shape: (NMhs, Nfft//2+1)
        
        # Frequency axis (same for all masses)
        omegalist = np.fft.rfftfreq(len(t_corr), d=dt) * 2 * np.pi
        
        return omegalist, np.abs(powerNL)



    def WindowFourier(self, CosmoParams, AstroParams, HMFinterp, SFRD_Init, GreensFunction, zobs, tage, pop):
        "Fourier transform of GreensFunction * SFH."
        "Inputs are AstroParams, CosmoParams, HMFinterp, GreensFunction,  Mh, and zobs."
        "Returns the angular frequency list and the Fourier transform of the window function in erg/s/Msun."

        dt = AstroParams._dt_FFT
        Nfft = AstroParams._N_FFT 
        _tFFT = dt*np.arange(Nfft)  

        tobs = CosmoParams.tageofzMyr(zobs)
        _tearlier = np.fmax(0.0,tobs-tage) #time before the observation
        zage = CosmoParams.zfoftageMyr(_tearlier) #z of the earlier times 

        SFHarray = self.SFH(CosmoParams,AstroParams, HMFinterp, SFRD_Init, _tFFT, zobs, pop)

        _integrand = GreensFunction(AstroParams, _tFFT, HMFinterp.Mhtab)*SFHarray*1e6 #convert SFR to 1/Myr for FFT
        _windowFourier = np.fft.rfft(_integrand,axis=1)*dt #for correct normalization
        omegalist = np.fft.rfftfreq(len(_tFFT), d=dt) * 2 * np.pi  # Convert to angular frequency

        return omegalist, _windowFourier
