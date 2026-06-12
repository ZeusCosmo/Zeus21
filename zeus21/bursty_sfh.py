"""

Compute Star Formation Histories with Burstiness.

Author: Julian B. Muñoz
UT Austin - January 2026

Edited by Sarah Libanore
BGU - April 2026 

"""

from .sfrd import * 

class SFH_class:
    """
    Star formation histories (SFHs) with stochastic burstiness for Pop II galaxies.

    Computes the SFR(t) of a galaxy in a halo of mass Mh, including a
    power-spectral-density (PSD) model for log-normal SFR fluctuations
    (damped random walk in ln SFR). See 2601.07912 for the burstiness model.

    Only Pop II is currently supported; Pop III raises a ValueError.

    Parameters
    ----------
    UserParams : User_Parameters
        Global run settings.
    CosmoParams : Cosmo_Parameters
        Cosmological parameters, including tageofzMyr and zfoftageMyr.
    AstroParams : Astro_Parameters
        Astrophysical parameters. Key attributes used here:

        - USE_POPIII : bool — if True, raises ValueError (not implemented).
        - FLAG_COMPARE_BAGPIPES : bool — use a toy exponential SFH for
          comparison with BAGPIPES fits instead of the full model.
        - FLAG_RENORMALIZE_AVG_SFH : bool — if True, boosts mean SFR by
          exp(σ²/2) to account for log-normal stochasticity (<e^x>).
        - sigmaPSD, dsigmaPSDdlog10Mh : PSD amplitude and its mass slope.
        - tauPSD, dlog10tauPSDdlog10Mh : PSD timescale (Myr) and mass slope.
        - _minsigmaPSD, _maxsigmaPSD, _mintauPSD, _maxtauPSD : clamp limits.
        - _omegamin, _omegamax : frequency integration range (1/Myr).
        - _dt_FFT, _N_FFT : time resolution (Myr) and number of points for FFT.
        - epsstar, dlog10epsstardz, _zpivot, Mc, alphastar, betastar,
          fstarmax, mean_SFR_normalization : star-formation efficiency params.

    HMFinterp : HMF_interpolator
        Halo mass function interpolator; must expose Mhtab.
    tage : float or array-like
        Lookback time(s) in Myr at which to evaluate the SFH.
    zobs : float
        Observed redshift.
    z_Init : float, optional
        Initialisation redshift for the SFRD. Computed via Z_init if not given.
    SFRD_Init : SFRD_class, optional
        Pre-computed SFRD object. Constructed internally if not given.

    Attributes
    ----------
    SFH_II : ndarray, shape (NMh, Ntage)
        Pop II star formation rate in M☉ yr⁻¹ as a function of halo mass
        and lookback time.
    """

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
        """
        Compute the star formation history at a set of lookback times.

        Reconstructs halo mass history assuming exponential accretion, then
        multiplies the mass accretion rate by the stellar efficiency fstar(z, Mh).
        Optionally renormalises the mean SFR to account for log-normal scatter
        (see FLAG_RENORMALIZE_AVG_SFH).

        Parameters
        ----------
        CosmoParams : Cosmo_Parameters
        AstroParams : Astro_Parameters
        HMFinterp : HMF_interpolator
        SFRD_Init : SFRD_class
        tage : float or array-like
            Lookback times in Myr relative to zobs.
        zobs : float
            Observed redshift.
        pop : {2}
            Stellar population. Only Pop II (2) is implemented.

        Returns
        -------
        ndarray, shape (NMh, Ntage)
            SFR in M☉ yr⁻¹ at each (halo mass, lookback time).
        """

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
        """
        Factorized approximation to fstar(z, Mh) over a 2-D (Mh, z) grid.

        Computes fstar(Mh, z) ≈ fstar(Mh, z0) · (Mh / Mh0)^α, separating the
        mass and redshift dependences. This avoids evaluating fstar_ofz on the
        full NMh × Nz array; it is exact for Mh < Mc and any residual offset
        is absorbed into β*.

        Only valid under exponential accretion. Use with care if the accretion
        history deviates strongly from exponential.

        Parameters
        ----------
        AstroParams : Astro_Parameters
        CosmoParams : Cosmo_Parameters
        SFRD_Init : SFRD_class
        z : array-like, shape (Nz,)
            Redshift array (z[0] is used as the reference epoch).
        Mhlist : ndarray, shape (NMh, Nz)
            Halo mass grid at each redshift step.
        pop : {2}
            Stellar population.

        Returns
        -------
        ndarray, shape (NMh, Nz)
            Approximate fstar values over the full (Mh, z) grid.
        """

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
        """
        PSD amplitude σ(Mh) for the ln SFR damped random walk.

        Linear in log10(Mh), clamped to [_minsigmaPSD, _maxsigmaPSD].
        To convert to log10(SFR) units, multiply by log(10).

        Parameters
        ----------
        AstroParams : Astro_Parameters
        Mh : float or array-like
            Halo mass(es) in M☉.

        Returns
        -------
        ndarray
            σ in units of ln SFR, same shape as Mh.
        """

        "Returns the sigma of the power spectrum of lnSFR at Mh, in units of ln(SFR), so to convert to log10(SFR) multiply by np.log(10)"
        Mh = np.atleast_1d(Mh)
        sigma_at_Mh = AstroParams.sigmaPSD + AstroParams.dsigmaPSDdlog10Mh * np.log10(Mh/1e10)

        return np.fmin(np.fmax(sigma_at_Mh,AstroParams._minsigmaPSD),AstroParams._maxsigmaPSD) #make sure it's within the limits set by UserParams
    
    def tauPSD_at_Mh(self, AstroParams, Mh):
        """
        PSD correlation timescale τ(Mh) for the ln SFR damped random walk.

        Power-law in Mh, clamped to [_mintauPSD, _maxtauPSD].

        Parameters
        ----------
        AstroParams : Astro_Parameters
        Mh : float or array-like
            Halo mass(es) in M☉.

        Returns
        -------
        ndarray
            τ in Myr, same shape as Mh.
        """

        "Returns the tau of the power spectrum of lnSFR at Mh, in Myr"
        Mh = np.atleast_1d(Mh)
        tau_at_Mh = AstroParams.tauPSD * 10**(AstroParams.dlog10tauPSDdlog10Mh * np.log10(Mh/1e10))

        return np.fmin(np.fmax(tau_at_Mh,AstroParams._mintauPSD),
        AstroParams._maxtauPSD) #make sure it's within the limits set by UserParams


    def PowerlnSFR(self, AstroParams, omega, Mh):
        """
        Power spectrum of ln SFR under a damped (α=2) random walk model.

        P(ω, Mh) = σ²(Mh) · τ(Mh) / [1 + (ω τ)²]

        As in 2601.07912. Note that compared to other refs (eg 2410.21409) 
        we do not have an implicit factor of 1 Myr in the
        amplitude; σ here is in ln SFR units; can convert by sigma->sigma*np.log(10)
        Index for random walk us alpha = 2 by default, can be enhanced to modify. 

        Parameters
        ----------
        AstroParams : Astro_Parameters
        omega : float or array-like, shape (Nω,)
            Angular frequency in Myr⁻¹.
        Mh : float or array-like, shape (NMh,)
            Halo mass(es) in M☉.

        Returns
        -------
        ndarray, shape (NMh, Nω)
            Power spectrum in (ln SFR)² Myr.
        """
        
        omega = np.atleast_1d(omega) #make sure omega is a vector
        Mh = np.atleast_1d(Mh) #make sure Mh is a vector
        sigma_at_Mh = self.sigmaPSD_at_Mh(AstroParams, Mh) 
        tau_at_Mh = self.tauPSD_at_Mh(AstroParams, Mh)
        _tau_times_omega = np.outer(omega,tau_at_Mh) #omega is a vector length of omega, tau has the Mh length
        return (sigma_at_Mh**2 * tau_at_Mh / (1.0 + _tau_times_omega**2.0)).T # NM x Nomega; secretely there's a 1*Myr in the amplitude
    
    def Wink_TH(self,omega, T):
        """
        Top-hat temporal window function in Fourier space.

        W(ω, T) = sinc(ωT/2) = sin(ωT/2) / (ωT/2)

        Used to compute the variance of ln SFR averaged over a timescale T.
        A small offset (1e-16) is added to the argument to avoid 0/0 at ω=0.

        Parameters
        ----------
        omega : array-like
            Angular frequency in Myr⁻¹.
        T : float
            Averaging timescale in Myr.

        Returns
        -------
        ndarray
            Dimensionless window, same shape as omega. Equals 1 at ω→0.
        """
        
        x = omega*T/2 + 1e-16
        return np.sin(x) / (x)
    
    def Variance_of_lnSFR(self, AstroParams, T, Mh):
        """
        Variance of ln SFR averaged over a timescale T.

        Var[ln SFR]_T = (2/2π) ∫ P(ω) |W(ω,T)|² dω

        The factor 2/(2π) converts the one-sided integral over positive ω to
        the full two-sided variance.

        Parameters
        ----------
        AstroParams : Astro_Parameters
        T : float
            Averaging timescale in Myr. Use T=0 for the unsmoothed variance
            (integrates over all frequencies).
        Mh : float or array-like
            Halo mass(es) in M☉.

        Returns
        -------
        ndarray
            Variance in (ln SFR)², same shape as Mh.
        """
        
        omegalist = np.logspace(np.log10(AstroParams._omegamin),np.log10(AstroParams._omegamax), 999) # in 1/Myr
        power = self.PowerlnSFR(AstroParams, omegalist, Mh)
        wink = self.Wink_TH(omegalist, T)

        return np.trapezoid(power * np.abs(wink)**2, omegalist) *2/(2*np.pi) 
    
    def _get_mean_SFR_normalization(self, AstroParams, Mh):
        """
        Log-normal boost to the mean SFR from stochastic burstiness.

        For a Gaussian variable δ = ln SFR with variance σ²,
        ⟨SFR⟩ = SFR_smooth · exp(σ²/2).

        i.e. the ratio of mean <SFR> to SFR(Mh) with no burstiness (ie median)

        Parameters
        ----------
        AstroParams : Astro_Parameters
        Mh : array-like
            Halo mass(es) in M☉.

        Returns
        -------
        ndarray
            Multiplicative boost factor exp(σ²/2), same shape as Mh.
        """
        
        _varlnSFR = self.Variance_of_lnSFR(AstroParams, 0.,Mh) #T=0 since it's at integrated over all timescales
        meanSFRnormalization = np.exp(_varlnSFR/2.) #<e^d> = <e^sigma^2/2> for a gaussian d
        return meanSFRnormalization


    def _get_PowerSFR_NL_FFT_vectorized(self, AstroParams, Mh_array):
        """
        Non-linear power spectrum of SFR via FFT of the ln SFR correlation function.

        Because SFR = exp(ln SFR) is a non-linear transformation, its power
        spectrum differs from P_lnSFR. This method computes it by:

        1. Building the auto-correlation of ln SFR (damped random walk):
               C(t) = σ² / 2 · exp(−|t| / τ)
        2. Exponentiating to get the SFR correlation:
               C_SFR(t) = exp(C(t)) − 1
        3. FFT → one-sided power spectrum of SFR fluctuations.

        All NMh masses are processed simultaneously via broadcasting.

        Parameters
        ----------
        AstroParams : Astro_Parameters
            Must expose: _dt_FFT, _N_FFT.
        Mh_array : float or array-like, shape (NMh,)
            Halo mass(es) in M☉.

        Returns
        -------
        omegalist : ndarray, shape (Nfft//2 + 1,)
            Angular frequencies in Myr⁻¹.
        powerNL : ndarray, shape (NMh, Nfft//2 + 1)
            Non-linear SFR power spectrum in (M☉ yr⁻¹)² Myr.
        """
        
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
        """
        Fourier transform of the convolution G(t) * SFH(t).

        Computes W̃(ω, Mh) = FFT[ G(t) · SFH(t) ], where G is a Green's
        function (e.g. Greens_function_LUV) and SFH(t) is the star formation
        history. The result enters the luminosity power spectrum as
        P_L(ω) = |W̃(ω)|² · P_SFR(ω).

        Parameters
        ----------
        CosmoParams : Cosmo_Parameters
        AstroParams : Astro_Parameters
            Must expose: _dt_FFT, _N_FFT.
        HMFinterp : HMF_interpolator
        SFRD_Init : SFRD_class
        GreensFunction : callable
            A function with signature ``G(AstroParams, tage_Myr, Mhtab)``
            returning an array of shape (NMh, Nt) in erg s⁻¹ M☉⁻¹.
            Typically one of the Greens_function_L* functions from sed.py.
        zobs : float
            Observed redshift.
        tage : float
            Maximum lookback time in Myr (sets the FFT window).
        pop : {2}
            Stellar population.

        Returns
        -------
        omegalist : ndarray, shape (Nfft//2 + 1,)
            Angular frequencies in Myr⁻¹.
        windowFourier : ndarray, shape (NMh, Nfft//2 + 1)
            Complex Fourier transform of G·SFH in erg s⁻¹ M☉⁻¹ Myr.

        Notes
        -----
        SFH is converted from M☉ yr⁻¹ to M☉ Myr⁻¹ (×10⁶) before the FFT
        so that the output is in consistent Myr-based units.
        """

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
