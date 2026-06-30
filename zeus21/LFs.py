"""
Compute LFs given our SFR and HMF models.

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

from . import constants
from .sfrd import Z_init, SFRD_class
from .cosmology import bias_Tinker

import numpy as np
from scipy.special import erf
from copy import copy

from .SED import Greens_function_LHa, Greens_function_LUV_Short, Greens_function_LUV_Long

from .bursty_sfh import SFH_class
from .z21_utilities import pdf_fft_convolution, pdf_log_transform, normal_pdf, lognormal_pdf, sigma_log10, mean_log10


class LF_class:
    """
    Compute all quantities and methods associated with luminosity functions 
    
    Parameters
    ----------
    UserParams : User_Parameters
    CosmoParams : Cosmo_Parameters
    AstroParams : Astro_Parameters
    HMFinterp : HMF_interpolator
    LFParams : LF_Parameters
    z_Init : Z_init or None, optional
        Initial redshift matrices to be used in full-SFRD and PSD calculations.
        Only instantiated when a full SFRD object is required. If None (default), initialized internally.
    SFRD_Init : SFRD_class or None, optional
        Precomputed SFRD object. 
        Only instantiated when a full SFRD object is required i.e. for PSD calculations, and for Pop III non-PSD calculations when LW is not provided explicitly. If None (default), initialized internally.
    SFH_Init : SFH_class or None, optional
        Precomputed SFH object to be used in PSD calculations.
        Only instantiated for PSD calculations. If None (default), initialized internally.
    vCB : float, None or False, optional
        Baryon-CDM relative streaming velocity used for Pop III non-PSD SFR feedback. 
        If None (default), cosmological mean from ``CosmoParams`` is used. 
        False to fully disable streaming velocity feedback.
    J21LW_interp : interpolator, None or False, optional
        LW background interpolator as a function of redshift used for Pop III SFR feedback. 
        If None (default), the converged background from ``SFRD_Init`` is used. 
        False to fully disable LW feedback.

    Attributes
    ----------
    z_Init : Z_init class
        Initial redshift matrices used by full-SFRD and PSD calculations, when initialized.
    SFRD_Init : SFRD_class
        Full or lightweight SFRD object used to evaluate SFRs and, when available, self-consistent LW backgrounds.
    SFH_Init : SFH_class
        SFH object used for PSD-based observables.
    DZ_TOINT : array
        Redshift offsets, in units of ``LFParams.zwidth``, used to average the HMF over the redshift bin.
    WEIGHTS_TOINT : array
        Gaussian weights associated with ``DZ_TOINT``.
    biasM : array
        Tinker halo bias evaluated at the redshift samples used for the LF bin.
        Only defined when bias is requested as an output in ``LFParams``.
    UVLFbias_outputs : dict
        UVLF output nested dictionary, when requested as in ``LFParams``.
        Possible top-level keys are "tot", "popII", and "popIII".
        Each component can contain "LF" and/or "bias".
    HaLFbias_outputs : dict
        Halpha LF output nested dictionary, when requested as in ``LFParams``.
        Possible top-level keys are "tot", "popII", and "popIII" for different population types.
        Each component can contain "LF" and/or "bias" (with "bias" the numerator of the HMF-averaged halo bias, to be normalized by the LF to recover average bias).
    """

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, LFParams, z_Init=None, SFRD_Init=None, SFH_Init=None, vCB=None, J21LW_interp=None):

        # Evaluate whether or not a full SFRD init is needed or light init is enough
        need_full_SFRD = (
            AstroParams.FLAG_USE_PSD  # TODO: here for safety as PSD branch has not been tested, check if actually needed
            or (AstroParams.USE_POPIII and not LFParams.SKIP_POPIII and J21LW_interp is None)  # In standard computation, full init is only needed if we want self-consistent LW background for Pop IIIs; if no Pop IIIs, or LW given as input, light init is enough
        )

        # SFRD instantiation
        if z_Init is not None:
            self.z_Init = z_Init
        elif need_full_SFRD:                
            self.z_Init = Z_init(UserParams, CosmoParams)

        if SFRD_Init is not None:
            self.SFRD_Init = SFRD_Init
        elif need_full_SFRD:                
            self.SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init)
        else:
            # Lightweight SFRD object, enabling to import relevant methods for SFR calculation without computing global SFRD, LW background, reionization, gamma coefficients, etc...
            self.SFRD_Init = SFRD_class.light_init()


        # SFH instantiation
        if AstroParams.FLAG_USE_PSD:
            if SFH_Init is not None:
                self.SFH_Init = SFH_Init
            else:
                self.SFH_Init = SFH_class(UserParams, CosmoParams, AstroParams, HMFinterp, AstroParams._tagesMyr, LFParams.zcenter, self.z_Init, self.SFRD_Init)
                
        
        # Set redshift offsets and weights
        if(constants.NZ_TOINT>1):
            self.DZ_TOINT = np.linspace(-np.sqrt(constants.NZ_TOINT/3.), np.sqrt(constants.NZ_TOINT/3.),constants.NZ_TOINT)  # in sigmas around zcenter
        else:
            self.DZ_TOINT = np.array([0.0])
            
        self.WEIGHTS_TOINT = np.exp(-self.DZ_TOINT**2/2.)/np.sum(np.exp(-self.DZ_TOINT**2/2.))  # Assumed Gaussian in z, fair
        

        # Save Tinker halo bias only when bias requested as output
        if LFParams.RETURNBIAS:
            self.biasM = np.array([bias_Tinker(CosmoParams, HMFinterp.sigma_int(HMFinterp.Mhtab,LFParams.zcenter+dz*LFParams.zwidth)) for dz in self.DZ_TOINT])
        
        # Compute UVLF/bias if requested and save outputs
        if LFParams.FLAG_COMPUTE_UVLF:
            self.UVLFbias_outputs = self.compute_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, "UV", vCB, J21LW_interp)

        # Compute LF/bias if requested and save outputs
        if LFParams.FLAG_COMPUTE_HaLF:
            self.HaLFbias_outputs = self.compute_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, "Ha", vCB, J21LW_interp)


    ### Luminosity to log-luminosity/magnitude convert functions and vice versa
    # TODO: unify in two simple Mag_of_L and L_of_Mag functions giving the type of input (Lnu or nuLnu) and units as input to the function to avoid duplication?

    def Hz_from_angstrom(self, wavelength=1500.):
        """
        Convert rest-frame wavelength in Angstrom to frequency in Hz.

        Parameters
        ----------
        wavelength : float, optional
            Rest-frame wavelength in Angstrom. Default is 1500.

        Returns
        -------
        nu : float
            Frequency in Hz.
        """

        return constants.c_kms / (wavelength/1e13)
        

    def Mag_of_L_ergsHz(self, L):
        """
        Convert specific luminosity in erg/s/Hz to AB absolute magnitude (from 1703.02913).

        Parameters
        ----------
        L : float or array
            Specific luminosity (L_nu) in erg/s/Hz.

        Returns
        -------
        Mag : float or array
            AB absolute magnitude.
        """

        return constants.zeropoint_ABmag_ergsHz - 2.5 * np.log10(L)  # AB magnitude

    def Mag_of_L_ergs(self, L, wavelength=1500.):
        """
        Convert specific luminosity in erg/s to AB absolute magnitude (from 1703.02913).

        Parameters
        ----------
        L : float or array
            Specific luminosity (nuL_nu) in erg/s.
        wavelength : float, optional
            Rest-frame wavelength in Angstrom used to convert nuL_nu to L_nu. Default is 1500.

        Returns
        -------
        Mag : float or array
            AB absolute magnitude.
        """

        return self.Mag_of_L_ergsHz(L/self.Hz_from_angstrom(wavelength))


    def L_ergsHz_of_Mag(self, Mag):
        """
        Convert AB absolute magnitude to specific luminosity in erg/s/Hz (from 1703.02913).

        Parameters
        ----------
        Mag : float or array
            AB absolute magnitude.

        Returns
        -------
        L : float or array
            Specific luminosity (L_nu) in erg/s/Hz.
        """

        return 10**(0.4 * (constants.zeropoint_ABmag_ergsHz - Mag))

    def L_ergs_of_Mag(self, Mag, wavelength=1500.):
        """
        Convert AB absolute magnitude to specific luminosity in erg/s (from 1703.02913).

        Parameters
        ----------
        Mag : float or array
            AB absolute magnitude.
        wavelength : float, optional
            Rest-frame wavelength in Angstrom. Default is 1500.

        Returns
        -------
        L : float or array
            specific luminosity (nuL_nu) in erg/s.
        """

        return self.L_ergsHz_of_Mag(Mag) * self.Hz_from_angstrom(wavelength)
    

    def logorMag_of_L(self, L, which_band, renormalize_L, sigma=None):
        """
        Convert luminosity to the LF observable coordinate (log-luminosity/magnitude).
        Mean luminosity can be shifted so that a lognormal scatter preserves the linear mean.

        Parameters
        ----------
        L : float or array
            Average luminosity.
            For ``which_band`` = "UV", it is interpreted as specific luminosity in erg/s/Hz (L_nu).
            For ``which_band`` = "Ha", it is intepreted as integrated line luminosity (L_Halpha) in erg/s.
        which_band : {"UV", "Ha"}
            Observable band to compute.
        renormalize_L : bool
            Whether to renormalize the linear luminosity before converting to the LF coordinate.
        sigma : float or array, optional
            Scatter used for the renormalization. Required when ``renormalize_L`` is True.

        Returns
        -------
        logL_or_mag : float or array
            UV magnitude for ``which_band`` = "UV" or log10(L_Halpha) for ``which_band`` = "Ha". Non-finite values are
            replaced by arbitrarily faint values.
        """

        # Select desired band
        if which_band not in ["UV", "Ha"]:
            raise ValueError("Conversion from luminosity to log-luminosity/magnitude only implemented for 'UV' and 'Ha'.")

        # Lower the avg luminosity to recover the true avg, instead of log-avg, when applying lognormal scatter sigma
        # TODO: Note that in-place modification of L if np.ndarray here! If this is not what we want, we should have a local copy of L in the function instead
        if renormalize_L:
            if sigma is None:
                raise ValueError("Requested luminosity renormalization after converting to log-luminosity/magnitude when applying lognormal scatter, but no provided value for the scatter.")
            
            if which_band == "UV":
                L /= np.exp((np.log(10)/2.5 * sigma)**2 / 2.0)

            elif which_band == "Ha":
                L /= np.exp((np.log(10) * sigma)**2 / 2.0)


        # Convert luminosity to log-luminosity/magnitude
        if which_band == "UV":
            logLormag = self.Mag_of_L_ergsHz(L)  # TODO: also implement nuFnu conversion for UV, as right now this assumes specific luminosity in erg/s/Hz --> note that for Halpha and other emission lines this is not needed, as line luminosities are always an integrated flux measure emerging from continuum, not continuum per unit frequency/wavelength
            bad_value_fix = 100.0  # TODO: adjustable parameter in LFParameters, or constants?
        
        elif which_band == "Ha":
            logLormag = np.log10(L)  # Note that this a safe fallback for emission lines in general, not only Halpha
            bad_value_fix = -100.0  # TODO: adjustable parameter in LFParameters, or constants?


        # Replace "bad values" in return
        return np.where(np.isfinite(logLormag), logLormag, bad_value_fix)



    def compute_LFbias_binned(self, CosmoParams, AstroParams, HMFinterp, LFParams, which_band, vCB=None, J21LW_interp=None):
        """
        Compute binned LF and/or bias for the requested band.

        The method dispatches to Pop II and Pop III component calculations and optionally adds a total component. Output keys are controlled by ``LFParams.SKIP_POPII``, ``AstroParams.SKIP_POPIII`` and ``LFParams.SKIP_TOT``.

        Parameters
        ----------
        CosmoParams : Cosmo_Parameters
        AstroParams : Astro_Parameters
        HMFinterp : HMF_interpolator
        LFParams : LF_Parameters
        which_band : {"UV", "Ha"}
            Which LF band to compute.
        vCB : float, None or False, optional
            Baryon-CDM relative streaming velocity used for Pop III non-PSD SFR feedback. 
            If None (default), cosmological mean from ``CosmoParams`` is used. 
            False to fully disable streaming velocity feedback.
        J21LW_interp : interpolator, None or False, optional
            LW background interpolator as a function of redshift used for Pop III SFR feedback. 
            If None (default), the converged background from ``SFRD_Init`` is used. 
            False to fully disable LW feedback.

        Returns
        -------
        outputs : dict
            Nested dictionary with population keys and entries ``"LF"`` and/or
            ``"bias"``. The ``"bias"`` entry is the bias numerator.
            Possible top-level keys are "tot", "popII", and "popIII" for different population types.
            Each component can contain "LF" and/or "bias" (with "bias" the numerator of the HMF-averaged halo bias, to be normalized by the LF to recover average bias).
        """

        outputs = {}

        computePopII = not LFParams.SKIP_POPII
        computePopIII = not LFParams.SKIP_POPIII
        computeTot = not LFParams.SKIP_TOT

        # PopII
        if computePopII:
            outputs["popII"] = self.compute_pop_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, which_band, 2)

        # PopIII
        if computePopIII:
            if not AstroParams.USE_POPIII:
                raise ValueError("Attempting to compute Pop III LF/bias with AstroParams.USE_POPIII=False.")
            outputs["popIII"] = self.compute_pop_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, which_band, 3, vCB, J21LW_interp)  # Note that vCB and LW are only needed for Pop III

        # Total 
        if computeTot:
            
            if computePopIII and computePopII:
                outputs["tot"] = {key: outputs["popII"][key] + outputs["popIII"][key] for key in outputs["popII"]}

            elif computePopII:
                outputs["tot"] = outputs["popII"]

            elif computePopIII:
                outputs["tot"] = outputs["popIII"]
        
        return outputs
            

    def compute_pop_LFbias_binned(self, CosmoParams, AstroParams, HMFinterp, LFParams, which_band, pop, vCB=None, J21LW_interp=None):
        """
        Compute a single population (Pop III or Pop II) contribution to a binned luminosity function.

        Parameters
        ----------
        CosmoParams : Cosmo_Parameters
        AstroParams : Astro_Parameters
        HMFinterp : HMF_interpolator
        LFParams : LF_Parameters
        which_band : {"UV", "Ha"}
            Which LF band to compute.
        pop : int
            Stellar population, 2 for Pop II or 3 for Pop III.
        vCB : float, None or False, optional
            Baryon-CDM relative streaming velocity used for Pop III non-PSD SFR feedback. 
            If None (default), cosmological mean from ``CosmoParams`` is used. 
            False to fully disable streaming velocity feedback.
        J21LW_interp : interpolator, None or False, optional
            LW background interpolator as a function of redshift used for Pop III SFR feedback. 
            If None (default), the converged background from ``SFRD_Init`` is used. 
            False to fully disable LW feedback.

        Returns
        -------
        outputs : dict
            Dictionary containing "LF" and/or "bias" for the selected population.
        """

        # Error if not using Pop III stars in the AstroParams but Pop III LF calculation requested
        if pop == 3 and not AstroParams.USE_POPIII:
            raise ValueError('Attempting to compute Pop III LFs with USE_POPIII = False')

        # Error for Halpha calculation requested for Pop III
        if which_band == "Ha" and pop == 3:
            raise ValueError('PopIII are not implemented for Ha')  
        
        
        # PSD calculation, in which MUV and sigmaUV derived from integrating SFH
        # TODO: check, simply refactored from previous version with no functional change
        if AstroParams.FLAG_USE_PSD:

            if pop == 3 and AstroParams.DETACH_III_ACH:
                raise ValueError("LF calculation from PSD not implemented for Pop IIIs with a detached atomic-cooling component")

            if which_band == "UV":
                return self.compute_LFbias_binned_from_PSD(CosmoParams, AstroParams, HMFinterp, LFParams,
                                                           LFParams.zcenter, LFParams.zwidth, LFParams.MUVcenters, LFParams.MUVwidths,
                                                           which_band, pop, LFParams.DUST_FLAG,
                                                           LFParams.RETURNLF, LFParams.RETURNBIAS)

            elif which_band == "Ha":
                return self.compute_LFbias_binned_from_PSD(CosmoParams, AstroParams, HMFinterp, LFParams,
                                                           LFParams.zcenter, LFParams.zwidth, LFParams.log10LHacenters, LFParams.log10LHawidths,
                                                           which_band, pop, LFParams.DUST_FLAG,  
                                                           LFParams.RETURNLF, LFParams.RETURNBIAS)

            else:
                raise ValueError('Only UV and Ha LF can be computed.')
            

        # Standard Munoz+23 model: mean LUV \propto SFR \propto Mhdot*fstar + lognormal/gaussian scatter set by sigma
        else:  

            if which_band == "UV":

                # For Pop III case, read vCB from CosmoParams and LW background from SFRD init if not explicitly provided
                # TODO: implement vCB and LW feedback in PSD case too?
                if pop == 3:

                    # TODO: None option with defaults can be implemented directly in sfrd.Mmol
                    if vCB is None:
                        vCB = CosmoParams.vcb_avg 

                    if J21LW_interp is None:
                        # Safety check: if LW not explicitly provided, we raise an error if it's not available in SFRD_Init (this should not happen: if USE_POPIII = True and LW not explicitly provided, SFRD should be fully instantiated in the LF init)
                        if hasattr(self.SFRD_Init, "J21LW_interp_conv_avg"):
                            J21LW_interp = self.SFRD_Init.J21LW_interp_conv_avg
                        else:
                            raise ValueError("Pop III LF without full SFRD initialization requires an external J21LW_interp passed to the LF_class")     


                # Case of Pop IIIs with a detached ACH component: here we explicitly compute SFRs for the two components separately so that we can apply different LFParams to them and compute separate LFs to be summed at the end
                # TODO: for now, the case with detached Pop III ACH component is isolated here, but we could integrate it in the general case below by implementing a separate poulation type, e.g. 3.II? This could be extended to other population types e.g. different morphological types etc... (to be consistently modified in sfrd.py)
                if pop == 3 and AstroParams.DETACH_III_ACH:  

                    # To detach ACH component, we use a separate set of AstroParams for the main component (minihalo-only component, extended up to Mup_III="Matom") and the additional ACH component (with the main Pop III component put to 0 through its epsstar value)
                    AstroParams_main = copy(AstroParams)
                    AstroParams_main.DETACH_III_ACH = False
                    AstroParams_main.Mup_III = "Matom"

                    AstroParams_ACH = copy(AstroParams)
                    AstroParams_ACH.DETACH_III_ACH = True
                    AstroParams_ACH.epsstar_III = 0.0

                    SFRlist_main = self.SFRD_Init.SFR(CosmoParams, AstroParams_main, HMFinterp, HMFinterp.Mhtab, LFParams.zcenter, pop, vCB, J21LW_interp)
                    SFRlist_ACH = self.SFRD_Init.SFR(CosmoParams, AstroParams_ACH, HMFinterp, HMFinterp.Mhtab,LFParams.zcenter, pop, vCB, J21LW_interp)

                    outputs_main = self.compute_LFbias_binned_from_SFRlist(SFRlist_main, HMFinterp, LFParams, 
                                                                           LFParams.zcenter, LFParams.zwidth, LFParams.MUVcenters, LFParams.MUVwidths,  
                                                                           LFParams._kappaUV_III, LFParams.sigmaUV_III, LFParams.FLAG_RENORMALIZE_LUV, which_band, LFParams.DUST_FLAG_III,  
                                                                           LFParams.RETURNLF, LFParams.RETURNBIAS)
                    outputs_ACH = self.compute_LFbias_binned_from_SFRlist(SFRlist_ACH, HMFinterp, LFParams, 
                                                                          LFParams.zcenter, LFParams.zwidth, LFParams.MUVcenters, LFParams.MUVwidths, 
                                                                          LFParams._kappaUV_III_ACH, LFParams.sigmaUV_III_ACH, LFParams.FLAG_RENORMALIZE_LUV, which_band, LFParams.DUST_FLAG_III_ACH,
                                                                          LFParams.RETURNLF, LFParams.RETURNBIAS)

                    return {key: outputs_main[key] + outputs_ACH[key] for key in outputs_main}


                # General case, applying population-specific LFParams
                else:

                    SFRlist = self.SFRD_Init.SFR(CosmoParams, AstroParams, HMFinterp, HMFinterp.Mhtab, LFParams.zcenter, pop, vCB, J21LW_interp)

                    if pop == 3:
                        sigmaUV = LFParams.sigmaUV_III
                        kappaUV = LFParams._kappaUV_III
                        include_dust = LFParams.DUST_FLAG_III  # TODO: custom dust model for Pop IIIs + enforce separate population dust flags also in the PSD case?
                    elif pop == 2:
                        sigmaUV = LFParams.sigmaUV
                        kappaUV = LFParams._kappaUV
                        include_dust = LFParams.DUST_FLAG

                    return self.compute_LFbias_binned_from_SFRlist(SFRlist, HMFinterp, LFParams, 
                                                                   LFParams.zcenter, LFParams.zwidth, LFParams.MUVcenters, LFParams.MUVwidths,  
                                                                   kappaUV, sigmaUV, LFParams.FLAG_RENORMALIZE_LUV, which_band, include_dust,
                                                                   LFParams.RETURNLF, LFParams.RETURNBIAS)


            elif which_band == "Ha":
                raise ValueError('FLAG_USE_PSD=False not implemented for Halpha LF.')

            else:
                raise ValueError('Only UV and Ha LF can be computed.')
    

    def compute_LFbias_binned_from_SFRlist(self, SFRlist, HMFinterp, LFParams,  zcenter, zwidth, logLcenters, logLwidths,  kappa, sigma, renormalize_L,  which_band, include_dust, computeLF, computeBias):
        """
        Compute binned LF outputs from a precomputed SFR list.

        Parameters
        ----------
        SFRlist : array
            Star formation rate evaluated on ``HMFinterp.Mhtab``.
        HMFinterp : HMF_interpolator
        LFParams : LF_Parameters
        zcenter, zwidth : float
            Redshift bin center and width.
        logLcenters, logLwidths : array
            LF bin centers and widths. These are UV magnitudes bins for ``which_band`` = "UV" and log10(L_Ha) bins for ``which_band`` = "Halpha".
        kappa : float
            SFR-to-luminosity conversion factor.
        sigma : float or array
            Scatter in the LF observable coordinate.
        renormalize_L : bool
            Whether to renormalize the luminosity before adding lognormal scatter to preserve the linear mean.
        which_band : {"UV", "Ha"}
            Which LF band to compute.
        include_dust : bool
            Whether to apply dust corrections.
        computeLF, computeBias : bool
            Select which output entries to compute.

        Returns
        -------
        outputs : dict
            Dictionary containing output "LF" and/or "bias", depending on selected output types.
        """

        if not computeLF and not computeBias:
            raise ValueError("No return options for LF computation from SFRlist.")  


        # Average luminosity
        L_avglist = SFRlist / kappa  # SFR to luminosity conversion for each Mh

        # Luminosity to log-luminosity/magnitude conversion
        logL_avglist = self.logorMag_of_L(L_avglist, which_band, renormalize_L, sigma)

        # Avoid numerical issues with zero sigma --> TODO: min. safe sigma in LFParams? Note that potential inconsistency with PSD calculation, in which the min. is set to 0.1
        sigma = np.fmax(sigma, 0.2)


        # Dust correction applied to log-luminosity/magnitude list and sigma if necessary
        if include_dust:
            logL_avglist, sigma = self.apply_dust_correction(LFParams, zcenter, logL_avglist, sigma, which_band)  # TODO: note that all of the core parameters of the LF calculation here are given explicitely for a single population types, so LFParams is only used for the dust properties, see comment in apply_dust_correction


        # LF core computation
        return self.compute_LFbias_binned_from_avgsigma(logL_avglist, sigma,  HMFinterp, zcenter, zwidth, logLcenters, logLwidths,  computeLF, computeBias)
    

    def compute_LFbias_binned_from_PSD(self, CosmoParams, AstroParams, HMFinterp, LFParams,  zcenter, zwidth, logLcenters, logLwidths,  which_band, pop, include_dust,  computeLF, computeBias):
        """
        Compute binned LF outputs from PSD-derived observable statistics.

        Parameters
        ----------
        CosmoParams : Cosmo_Parameters
        AstroParams : Astro_Parameters
        HMFinterp : HMF_interpolator
        LFParams : LF_Parameters
        zcenter, zwidth : float
            Redshift bin center and width.
        logLcenters, logLwidths : array
            LF bin centers and widths.
        which_band : {"UV", "Ha"}
            Which LF band to compute.
        pop : int
            Stellar population, 2 for Pop II or 3 for Pop III.
        include_dust : bool
            Whether to apply dust corrections.
        computeLF, computeBias : bool
            Select which output entries to compute.

        Returns
        -------
        outputs : dict
            Dictionary containing output "LF" and/or "bias", depending on selected output types.
        """
        # TODO: check, simply refactored from previous version with no functional change
        
        if which_band == "UV":
            LUV_short, sigmaLUV_short = self.meanandsigma_observable_PSD(CosmoParams, AstroParams, HMFinterp, LFParams, Greens_function_LUV_Short, pop)

            LUV_long, sigmaLUV_long = self.meanandsigma_observable_PSD(CosmoParams, AstroParams, HMFinterp, LFParams, Greens_function_LUV_Long, pop)

            logL_avglist, sigma = self.sigma_MUV_from_meansandsigmas(LUV_short, LUV_long, sigmaLUV_short, sigmaLUV_long)

            logL_avglist = np.fmin(logL_avglist, constants._MAGMAX_UV)


        elif which_band == "Ha":

            L_avglist, sigma_ln = self.meanandsigma_observable_PSD(CosmoParams, AstroParams, HMFinterp, LFParams, Greens_function_LHa, pop)
    
            logL_avglist = mean_log10(L_avglist, sigma_ln) 
            sigma = sigma_log10(L_avglist, sigma_ln)

            logL_avglist = np.fmax(logL_avglist,constants._MAGMIN_Ha)  # Cut to avoid -inf    

            sigma = np.fmax(sigma, 0.1)  # Avoid numerical issues with zero sigma --> TODO: min. safe sigma in LFParams? Note that potential inconsistency with standard calculation, in which the min. is set to 0.2

        else:
            raise ValueError('Only UV and Ha LF can be computed.')
        

        # Dust correction applied to log-luminosity/magnitude list and sigma if necessary
        if include_dust:
            logL_avglist, sigma = self.apply_dust_correction(LFParams, zcenter, logL_avglist, sigma, which_band)


        # LF core computation
        return self.compute_LFbias_binned_from_avgsigma(logL_avglist, sigma,  HMFinterp, zcenter, zwidth, logLcenters, logLwidths,  computeLF, computeBias)
        

    def compute_LFbias_binned_from_avgsigma(self, logL_avglist, sigma,  HMFinterp, zcenter, zwidth, logLcenters, logLwidths,  computeLF, computeBias):
        """
        Compute LF and/or bias given the average log-luminosity/magnitude associated with each halo mass and its lognormal scatter.
        This is the common numerical core for LF computation, shared by the SFR-list and PSD branches: it convolves the mean LF coordinate at each halo mass with a Gaussian scatter and integrates over the HMF.

        Parameters
        ----------
        logL_avglist : array
            Mean LF coordinate (UV magnitude or log10(L_Halpha)) at each halo mass.
        sigma : float or array
            Scatter in the same coordinate.
        HMFinterp : HMF_interpolator
        zcenter, zwidth : float
            Redshift bin center and width.
        logLcenters, logLwidths : array
            LF bin centers and widths.
        computeLF, computeBias : bool
            Select which output entries to compute.

        Returns
        -------
        outputs : dict
            Dictionary containing output "LF" and/or "bias", depending on selected output types.
        """
         
        cuthi = logLcenters + logLwidths/2.
        cutlo = logLcenters - logLwidths/2.

        xhi = np.subtract.outer(cuthi, logL_avglist)/(np.sqrt(2) * sigma)
        xlo = np.subtract.outer(cutlo, logL_avglist)/(np.sqrt(2) * sigma)
        weights = (erf(xhi) - erf(xlo)).T/(2.0 * logLwidths)
   
        HMFtab = np.array([HMFinterp.HMF_int(HMFinterp.Mhtab, zcenter + dz*zwidth) for dz in self.DZ_TOINT])

        outputs = {}
        if computeLF:
            HMFcurr = np.sum(self.WEIGHTS_TOINT * HMFtab.T, axis=1)
            outputs["LF"] = np.trapezoid(weights.T * HMFcurr, HMFinterp.Mhtab, axis=-1)  # TODO: check consistency without fduty
        if computeBias:  # TODO: compute actual average bias, already dividing by LF here?
            halobiascurr = np.sum(self.WEIGHTS_TOINT * HMFtab.T * self.biasM.T, axis=1)
            outputs["bias"] = np.trapezoid(weights.T * halobiascurr, HMFinterp.Mhtab, axis=-1)  # TODO: check consistency without fduty

        return outputs
        


    #####Here the dust attenuation
    def apply_dust_correction(self, LFParams, z, logL_or_mag, sigma, which_band):
        """
        Apply dust attenuation to intrinsic LF coordinate(s) and scatter.

        Parameters
        ----------
        LFParams : LF_Parameters
        z : float
            Redshift.
        logL_or_mag : array
            Intrinsic LF coordinate.
        sigma : float or array
            Intrinsic scatter.
        which_band : {"UV", "Ha"}
            Observable band.

        Returns
        -------
        logL_or_mag_dust : array
            Dust-corrected observed coordinate(s).
        sigma_dust : float or array
            Scatter after optional dust contribution.
        """

        # TODO: right now global dust correction parameters, we cannot chose different parameters for different components, we may want to isolate the dust model parameters in a separate class (also these may be useful for other observables other than LFs)

        # Cannot directly 'dust' the theoretical intrinsic magnitudes, since the properties of the IRX-beta relation are calibrated on observed MUV. Recursion instead, solving MUV_obs = MUV_intrinsic + A_UV(MUV_obs) iteratively
        curr_logLormag = logL_or_mag

        curr2 = np.ones_like(curr_logLormag)
        while(np.sum(np.abs((curr2-curr_logLormag)/curr_logLormag)) > 0.02):
            curr2 = curr_logLormag
            curr_logLormag = logL_or_mag + self.dust_attenuation(LFParams, z, curr_logLormag, which_band)

        if LFParams.sigma_times_AUV_dust != 0.:
            sigma_dust = np.fmax(0.0, LFParams.sigma_times_AUV_dust) * self.dust_attenuation(LFParams, z, curr_logLormag, which_band)
            sigma = np.sqrt(sigma**2 + sigma_dust**2)  # Add dust sigma, if any, to the UV sigma

        return curr_logLormag, sigma
    

    def dust_attenuation(self, LFParams, z, logL_or_mag, which_band):
        """
        Return the mean attenuation for the requested observable.
        For UV, this is given as a function of OBSERVED z and magnitude. If using on theoretical intrinsice magnitudes, iterate until convergence. 
        The ``LFParams.HIGH_Z_DUST`` flag controls whether to apply dust attenuation at higher z than 0 or set attenuation to 0. If true, fix betaUV for attenuation calculation at ``LFParams._zmaxdata`` redshift value.
        Halpha dust attenuation not currently implemented.

        Parameters
        ----------
        LFParams : LF_Parameters
        z : float
            Redshift.
        logL_or_mag : array
            Observed LF coordinate to be dust attenuated.
        which_band : {"UV", "Ha"}
            Observable band.

        Returns
        -------
        Adust : array
            Attenuation in magnitudes for UV. Halpha currently raises a
            ``ValueError`` because no calibrated model is implemented.
        """

        if which_band == "UV":

            MUV = logL_or_mag
            betacurr = self.betaUV_dust(LFParams, z, MUV)
                        
            sigmabeta = 0.34 #from Bouwens 2014
            
            Auv = LFParams.C0dust + 0.2*np.log(10)*sigmabeta**2 * LFParams.C1dust**2 + LFParams.C1dust * betacurr  # TODO: ref?
            Auv=Auv.T
            if not (LFParams.HIGH_Z_DUST):
                Auv*=np.heaviside(LFParams._zmaxdata - z, 0.5)
            
            Adust = np.fmax(Auv.T, 0.0)

        elif which_band == "Ha":

            raise ValueError("Halpha dust attenuation not implemented. Set DUST_FLAG=False when computing Halpha LF, or implement a calibrated A_Ha model.")

            'Average attenuation A as a function of z and log10LHa.'
            # TODO: made-up to see how to calibrate it. Unused in current implementation (set Ha DUST = False)
            # Conjured approximation - lower at high z and fainter 

            log10LHa = logL_or_mag
            AHa = 0.5 * (1 + 0.3 * (log10LHa - 42.0))
            Adust = -0.4 * np.fmax(AHa, 0.0)  # no negative dust attenuation
            #-0.4* instead of +1* here since its log10L not mag

        return Adust


    def betaUV_dust(self, LFParams, z, MUV):
        """
        Compute the UV continuum slope used by the dust model, currently implementing Bowens+13,14 or Zhao+24 model.
    
        Parameters
        ----------
        LFParams : LF_Parameters
        z : float or array
            Redshift.
        MUV : float or array
            UV absolute magnitude.

        Returns
        -------
        beta : array
            UV slope from the selected dust model.
        """

        if LFParams.DUST_model == "Bouwens13":

            'Color as a function of redshift and mag, interpolated from Bouwens 2013-14 data.'

            zdatbeta = [2.5,3.8,5.0,5.9,7.0,8.0]
            betaMUVatM0 = [-1.7,-1.85,-1.91,-2.00,-2.05,-2.13]
            dbeta_dMUV = [-0.20,-0.11,-0.14,-0.20,-0.20,-0.15]

            _MUV0 = -19.5
            _c = -2.33

            betaM0 = np.interp(z, zdatbeta, betaMUVatM0, left=betaMUVatM0[0], right=betaMUVatM0[-1])
            dbetaM0 = (MUV - _MUV0).T * np.interp(z, zdatbeta, dbeta_dMUV, left=dbeta_dMUV[0], right=dbeta_dMUV[-1])
            
            sol1 = (betaM0-_c) * np.exp(dbetaM0/(betaM0-_c))+_c #for MUV > MUV0
            sol2 = dbetaM0 + betaM0 #for MUV < MUV0
            
            return sol1.T * np.heaviside(MUV - _MUV0, 0.5) + sol2.T * np.heaviside(_MUV0 - MUV, 0.5)

        elif LFParams.DUST_model == "Zhao24": 

            'from https://arxiv.org/pdf/2401.07893.pdf, table 1'

            betaM0z0 = -1.58
            dbetaM0dz = -0.081

            dbetaM0dMUVz0 = -0.216
            ddbetaM0dMUVdz = 0.012

            MUV0 = -19.5
            betaM0 = betaM0z0 + z * dbetaM0dz
            dbetaM0 = (MUV - MUV0).T * (dbetaM0dMUVz0 + ddbetaM0dMUVdz * z)
            
            sol2 = dbetaM0 + betaM0 #beta_M0 + db/dMUV|M0 (DeltaMUV)at M0=-19.5
            
            return np.fmax(-3.0, sol2) #cap at -3 just in case


    def correct_AP_LF(self, z, Deltaz, CosmoParams_data, CosmoParams, logLormag_data, Phi_data, errPhi_data, errPhi_asy_data = None, which_band = "UV"): 
        "Corrects the observed LF from the assumed cosmology CosmoParams to another with CosmoParams_out. Note: no dust correction since it's applied directly to theory->model" 
        # TODO: check if needed, function never used in LFs.py

        r_data = CosmoParams_data.chiofzint(z) #comoving distance
        Vol_data = CosmoParams_data.chiofzint(z+Deltaz/2.0)**3 - CosmoParams_data.chiofzint(z-Deltaz/2.0)**3 #no need for 4pi/3 since it'll be a ratio
        
        r_out = CosmoParams.chiofzint(z)
        Vol_out = CosmoParams.chiofzint(z+Deltaz/2.0)**3 - CosmoParams.chiofzint(z-Deltaz/2.0)**3 
        
        Phi_out = Phi_data * Vol_data/Vol_out
        errPhi_out = errPhi_data * Vol_data/Vol_out
        val = -5. if which_band == "UV" else 2.
        logLormag_out = logLormag_data + val * np.log10(r_out/r_data) #linear change so it doesn't affect bin sizes

        if (errPhi_asy_data is not None): #for asymmetric errorbars, optional arg
            errPhi_asy_out = errPhi_asy_data * Vol_data/Vol_out
            return logLormag_out, Phi_out, errPhi_out, errPhi_asy_out
        else:
            return logLormag_out, Phi_out, errPhi_out


    def meanandsigma_observable_PSD(self, CosmoParams, AstroParams, HMFinterp, LFParams, GreensFunction, pop):
        """
        Compute PSD-derived mean and scatter for a given observable (e.g. LUV, Ha, etc...) from the power spectrum of the SFRD.

        Parameters
        ----------
        CosmoParams : Cosmo_Parameters
        AstroParams : Astro_Parameters
        HMFinterp : HMF_interpolator
        LFParams : LF_Parameters
        GreensFunction : callable
            Time-domain response function G(t) for the observable of interest.
        pop : int
            Stellar population, 2 for Pop II or 3 for Pop III.

        Returns
        -------
        avgobs : array
            Mean observable at each halo mass.
        sigmaobs : array
            Scatter of the observable at each halo mass.
        """
        # TODO: check, simply refactored from previous version with no functional change

        #First get the mean observable at the given redshift and halo mass
        _windowintages = GreensFunction(AstroParams, AstroParams._tagesMyr, HMFinterp.Mhtab) 

        if pop == 2:
            _SFHinages = self.SFH_Init.SFH_II
        elif pop == 3:
            _SFHinages = self.SFH_Init.SFH_III
        
        avgobs = np.trapezoid(_SFHinages*_windowintages, AstroParams._tagesMyr*1e6, axis=1) 

        #Now get the sigma, first compute the power spectrum of the SFR from that of lnSFR:   
        #use the omegalist from the FFT, which is the same for all observables
        #use the power spectrum from the FFT, which is the same for all observables

        omegalist, powerNL = self.SFH_Init._get_PowerSFR_NL_FFT_vectorized(AstroParams, HMFinterp.Mhtab) #freq in 1/Myr and power of SFR=e^x (x=lnSFR). First array is Nfft, second is Nm x Nfft

        #And FFT the window function for the integral:
        _, windowfourier = self.SFH_Init.WindowFourier(CosmoParams, AstroParams, HMFinterp, self.SFRD_Init, GreensFunction, LFParams.zcenter, AstroParams._tagesMyr, pop)

        _whichomegakeep = np.logical_and(omegalist > AstroParams._omegamin, omegalist < AstroParams._omegamax)

        sigmaobs = np.sqrt(np.trapezoid(powerNL * np.abs(windowfourier)**2*_whichomegakeep, omegalist, axis=1)*2/(2*np.pi)) #times 2 because + and - freqs

        return avgobs, sigmaobs


    def sigma_MUV_from_meansandsigmas(self, LUV1mean, LUV2mean, sigmaLUV1, sigmaLUV2):
        """
        Combine short- and long-timescale UV statistics into mean UV magnitudes and scatter. These components are considered mostly uncorrelated.

        Parameters
        ----------
        LUV1mean, LUV2mean : array
            Mean UV luminosity components.
        sigmaLUV1, sigmaLUV2 : array
            Scatter of the UV luminosity components.

        Returns
        -------
        MUV_avg : array
            Mean UV magnitude.
        sigma_MUV : array
            UV-magnitude scatter.
        """
        # TODO: check, simply refactored from previous version with no functional change

        #vectorize the inputs so it can read either scalar or array inputs
        LUV1mean = np.asarray(LUV1mean)
        LUV2mean = np.asarray(LUV2mean)
        sigmaLUV1 = np.asarray(sigmaLUV1)
        sigmaLUV2 = np.asarray(sigmaLUV2)

        _numberofMhs = len(LUV1mean) 
        if len(LUV2mean) != _numberofMhs or len(sigmaLUV1) != _numberofMhs or len(sigmaLUV2) != _numberofMhs:
            raise ValueError("All input arrays must have the same length.")
        # Initialize arrays to hold the results
        MUVbar = np.zeros(_numberofMhs)
        sigmaMUV = np.zeros(_numberofMhs)
        for imh in range(_numberofMhs):
            sigmaUV1 = sigma_log10(sigmaLUV1[imh],LUV1mean[imh])*np.log(10)
            mu1 = mean_log10(sigmaLUV1[imh],LUV1mean[imh])*np.log(10)
            sigmaUV2= sigma_log10(sigmaLUV2[imh],LUV2mean[imh])*np.log(10)
            mu2 = mean_log10(sigmaLUV2[imh],LUV2mean[imh])*np.log(10)

            yvalues, PDF_y = pdf_fft_convolution(mu1, sigmaUV1, mu2, sigmaUV2)   
            lnyvalues, PDF_lny = pdf_log_transform(yvalues, PDF_y) 
            MUVvalues, PDF_MUV = self.Mag_of_L_ergs(np.exp(lnyvalues)), -2.5*np.log(10)*PDF_lny
            _norm = np.trapezoid(PDF_MUV, MUVvalues) #normalization, should always be 1 but just in case
            MUVbar[imh] = np.trapezoid(MUVvalues * PDF_MUV, MUVvalues)/_norm
            sigmaMUV[imh] = np.sqrt(np.trapezoid((MUVvalues - MUVbar[imh])**2 * PDF_MUV, MUVvalues)/_norm)
        
        return MUVbar, sigmaMUV #NOTE: can be enhanced to return full PDF, but needs to know the size of the MUVvalues array. We dont need it yet so just return the mean and sigma
        




class Ha_UV_ratio:  # TODO: different file e.g. line_ratios.py?
    """
    Compute the Halpha-to-UV luminosity ratio distribution.

    This helper combines the UV luminosity function, Halpha luminosities and PSD-derived covariance terms to evaluate the probability distribution of ``log10(L_Ha / L_UV)`` or the equivalent ``xi_ion`` quantity.

    Parameters
    ----------
    UserParams : User_Parameters
    CosmoParams : Cosmo_Parameters
    AstroParams : Astro_Parameters
    HMFinterp : HMF_interpolator
    LFParams : LF_Parameters
    z_Init : Z_init, optional
        Precomputed redshift matrices.
    SFRD_Init : SFRD_class, optional
        Precomputed SFRD object.
    SFH_Init : SFH_class, optional
        Precomputed SFH object.
    LF_Init : LF_class, optional
        Precomputed LF object. If None, initialized internally.
    """
    # TODO: check class and documentation

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, LFParams, z_Init = None, SFRD_Init = None, SFH_Init = None, LF_Init = None):

        if not AstroParams.FLAG_USE_PSD:
            raise ValueError('FLAG_USE_PSD=False not implemented in PDF_HaUV_ratio()')

        if AstroParams.USE_POPIII:
            raise ValueError('USE_POPIII=True not implemented in PDF_HaUV_ratio()')


        if z_Init is None:
            self.z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams)
        else:
            self.z_Init = z_Init

        if SFRD_Init is None:
            self.SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, self.z_Init)
        else:
            self.SFRD_Init = SFRD_Init

        if LF_Init is None:
            self.LF_Init = LF_class(UserParams, CosmoParams, AstroParams, HMFinterp, LFParams, z_Init=self.z_Init, SFRD_Init=self.SFRD_Init, vCB_input=False, J21LW_interp_input=False)
        else:
            self.LF_Init = LF_Init

        if SFH_Init is None:
            self.SFH_Init = SFH_class(UserParams, CosmoParams, AstroParams, HMFinterp, AstroParams._tagesMyr, LFParams.zcenter, self.z_Init, self.SFRD_Init, )
        else:
            self.SFH_Init = SFH_Init


    def PDF_log10HaUVratio(self, LUV_mean, LHa_mean, sigmaLHa, sigmasquaredcross, log10etavalues):
        """
        Returns the PDF of log10(LHa/LUV) at fixed Mh (NOTE: integrated over all MUVs). Assumed dust corrected!
        
        Parameters:
        -----------
        LUVmean : array_like
            The mean LUV value
        LHa_mean : array_like
            The mean LHa value
        sigmaLHa : array_like
            The sigma of LHa value
        sigmasquaredcross : array_like
            The cross sigma squared (sigma^2) of LHa and LUV
            This is the covariance between LHa and LUV, computed from their window functions. From cross_sigma_squared_PSD.
            
        Returns:
        --------
        log10etavalues : ndarray
            The log10eta values (same for all input array elements)
        PDFlog10eta : ndarray
            Array of shape (len(LUVmean), len(log10etavalues)) with PDFs
        """

        AconstantLUVLHa = sigmasquaredcross/sigmaLHa**2 
        BconstantLUVLHa = LUV_mean - AconstantLUVLHa * LHa_mean 
        _A, _B = AconstantLUVLHa, BconstantLUVLHa 

        mean_of_log10LHa = np.log10(LHa_mean)- 1/2 * np.log10(1 + sigmaLHa**2/LHa_mean**2)
        sigma_of_log10LHa = sigma_log10(sigmaLHa, LHa_mean)

        if log10etavalues is None:  # If not provided, create a default range
            log10etavalues = np.linspace(-3.5,-1.3,55)

        etavalues = 10**log10etavalues
        _LHavalues = np.outer(etavalues,_B)/(1-np.outer(etavalues,_A)) #recalculate the LHa values from eta values

        muHa, sigmaHa = mean_of_log10LHa*np.log(10), sigma_of_log10LHa*np.log(10) #mean and std of ln(Ha), a gaussian varible
        PDFLHalognormal = lognormal_pdf(_LHavalues, muHa, sigmaHa)
        dydx = _B/(_B+_A*_LHavalues) * 1/(_LHavalues * np.log(10))
        
        log10etavalues = log10etavalues
        PDFlog10eta = PDFLHalognormal/np.abs(dydx)

        return log10etavalues, PDFlog10eta


    def PDF_HaUV_ratio(self, CosmoParams, AstroParams, HMFinterp, LFParams, log10etavalues, FLAG_supersample_MUV = False):
        '''
        Returns Ha/UV ratio PDF, binned in MUV_bin_edges and log10eta bins, if provided.

        Parameters:
        -----------
        AstroParams : Astro Parameters
        CosmoParams : Cosmo Parameters
        HMFinterp : HMFinterp
        zcenter, zwidth: the z where it is calculated (no width for now, ignored)
        log10etavalues: the log10(Ha/UV) ratios where the PDF is computed. Assigned by function if None
        FLAG_supersample_MUV : Whether to super-sample to integrate within MUV_bin_edges better. If =False then just computes at the center of each bin
        
        Returns:
        --------

        log10etavalues: bins of log10Ha/UV
        pdf_binned: the PDF(log10etavalues) in those log10etavalues bins, and the MUV bins chosen
        UVLFvalues: the UVLF at the MUV binned, so the user can sum stuff easily

        '''
        
        if log10etavalues is None:  # If not provided, create a default range
            log10etavalues = np.linspace(-3.5,-1.0,20)

        HMFtab = HMFinterp.HMF_int(HMFinterp.Mhtab,LFParams.zcenter) 

        meanLUVshort, meanLHa, sigmaLUVshort, sigmaLHa, sigmasqcross = self.cross_sigma_squared_PSD(AstroParams, CosmoParams, HMFinterp, Greens_function_LUV_Short,Greens_function_LHa, LFParams.zcenter)

        _Acoeff = sigmasqcross/sigmaLHa**2 

        meanLUVlong, sigmaLUVlong = self.LF_Init.meanandsigma_observable_PSD(CosmoParams, AstroParams, HMFinterp, LFParams, Greens_function_LUV_Long, pop=2)

        #get the UV-A*Ha "excess" UV luminosity, not exactly lognormal so use the same trick as for MUV:
        meanLUV_excess_short = meanLUVshort - _Acoeff * meanLHa
        sigmaLUV_excess_short  = np.sqrt(sigmaLUVshort**2 + _Acoeff**2 * sigmaLHa**2 - 2.0 * _Acoeff * sigmasqcross)
        MUVbar_excess, sigmaMUV_excess = self.LF_Init.sigma_MUV_from_meansandsigmas(meanLUV_excess_short, meanLUVlong, sigmaLUV_excess_short, sigmaLUVlong)


        MUVavglist, sigmaMUV = self.LF_Init.sigma_MUV_from_meansandsigmas(meanLUVshort, meanLUVlong, sigmaLUVshort, sigmaLUVlong)
        MUVavglist = np.fmin(MUVavglist,constants._MAGMAX_UV)
        sigma_times_AUV_dust = np.fmax(0.0, LFParams.sigma_times_AUV_dust) #assumed constant, not derived from SFH

        #these are the parameters of the closest lognormal to each Ha, UV, and UVx
        muHa, sigmaHa = mean_log10(sigmaLHa, meanLHa)*np.log(10), sigma_log10(sigmaLHa, meanLHa)*np.log(10) #mean and std of ln(LUVx), a gaussian varible

        muUV, sigmaUV = np.log(self.LF_Init.L_ergsHz_of_Mag(MUVavglist)), sigmaMUV*np.log(10)/2.5 
        muUVx, sigmaUVx = np.log(self.LF_Init.L_ergsHz_of_Mag(MUVbar_excess)), sigmaMUV_excess*np.log(10)/2.5 

        if (FLAG_supersample_MUV==True):
            _NLuvsupersample = 99 #number of LUV values to supersample
            _LUVlist = np.logspace(35,46,_NLuvsupersample) #in case you want to integrate and then bin
        else:
            _LUVlist = self.LF_Init.L_ergsHz_of_Mag(LFParams.MUVcenters) #this will give mean of <log10(LUV)> for each MUV bin


        #This is used for assigning galaxies to MUV bins, so we add dust correction since the Ha/UV ratios are dust corrected but they're binned in MUVobs
        currMUV = MUVavglist
        currMUV2 = np.ones_like(currMUV)
        while(np.sum(np.abs((currMUV2-currMUV)/currMUV)) > 0.02):
            currMUV2 = currMUV
            currMUV = MUVavglist + self.LF_Init.dust_attenuation(LFParams,LFParams.zcenter,currMUV,"UV")

        sigmaUV_dust = sigma_times_AUV_dust * self.LF_Init.dust_attenuation(LFParams,LFParams.zcenter,currMUV,"UV")

        sigmaMUV_obs = np.sqrt(sigmaMUV**2 + sigmaUV_dust**2) #add dust sigma, if any, to the UV sigma
        sigmaMUV_obs = np.fmax(sigmaMUV_obs, 0.2) #avoid numerical issues with zero sigma

        muUV_obs, sigmaUV_obs = np.log(self.LF_Init.L_ergsHz_of_Mag(currMUV)), sigmaMUV_obs*np.log(10)/2.5
        PlnLUV = normal_pdf(np.log(_LUVlist), muUV_obs, sigmaUV_obs, dimy=1)+1e-99 #to avoid Nans
        UVLFvalues = np.trapezoid(HMFtab[:,None] * PlnLUV, HMFinterp.Mhtab, axis=0)


        #we exploit the fact that P(log10LHa - log10LHabar | LUV) doesnt change for LUV > LUVbar(Mh)
        #so we set LUV = LUVbar(Mh) for each Mh. (we dont modify P(LUV) since that is the UVLF, not the Ha/UV ratio)
        _meanLUV_ofMh = np.exp(muUV[:, None])
        _LUVforcalculation = np.minimum(_LUVlist[None,:], _meanLUV_ofMh) #Mh x LUVs, so that for LUV>LUVbar we recover the LUVbar result
        PlnLUVforcalculation = normal_pdf(np.log(_LUVforcalculation), muUV, sigmaUV, dimy=1)+1e-99 #to avoid Nans

        _LHacalc = _LUVforcalculation[:,:,None] * 10**log10etavalues[None,None,:]
        PlnLHa = normal_pdf(np.log(_LHacalc), muHa, sigmaHa, dimy=2)

        _LUVx = np.fmax(1.0, _LUVforcalculation[:,:,None] - _LHacalc [:,:,:] * _Acoeff[:, None,None]) #LUVx = _LUVforcalculation - A * LHa, where A is the coefficient for each MUV
        PlnLUVx_fixedHa = normal_pdf(np.log(_LUVx), muUVx, sigmaUVx,dimy=2)
        dlnLUV_dlnLUVx = _LUVx/_LUVforcalculation[:,:,None] 

        PDF_lnLUV_fixedHa = PlnLUVx_fixedHa/np.abs(dlnLUV_dlnLUVx)

        Plog10eta_fixedLUV = PDF_lnLUV_fixedHa * PlnLHa/PlnLUVforcalculation[:,:,None] * np.log(10) #Plog10eta_fixedLUV = Plog10LHa_fixedLUV. Note PlnLUVforcalculation, since it is the PDF of LUV that we use in Bayes rule. Below its P(LUV) since we sum over the Prob that that Mh is in the MUV bin
        
        pdf_binned = np.trapezoid(HMFtab[:,None,None] * Plog10eta_fixedLUV * PlnLUV[:,:,None], HMFinterp.Mhtab, axis=0)/UVLFvalues[:,None]
        
        #if supersampling, re-bin in MUVs:
        if (FLAG_supersample_MUV==True):
            #weights is NMUVcenters x _NLuvsupersample, multiply pdf_binned which is _NLuvsupersample x Nlog10etavalues

            MUVleft_edges  = LFParams.MUVcenters - LFParams.MUVwidths / 2
            MUVright_edges = LFParams.MUVcenters + LFParams.MUVwidths / 2

            # full edges (length N+1)
            MUV_bin_edges = np.concatenate([MUVleft_edges, [MUVright_edges[-1]]])

            MUVcuthi = MUV_bin_edges[1:]
            MUVcutlo = MUV_bin_edges[:-1]
            MUVs = self.LF_Init.Mag_of_L_ergs(_LUVlist) #here we use _LUVlist since its for the P(LUV) not the Ha/UV ratio
            xhi = np.heaviside(np.subtract.outer(MUVcuthi, MUVs),0.5)
            xlo = np.heaviside(np.subtract.outer(MUVcutlo, MUVs),0.5)
            MUVwidths = MUV_bin_edges[1:] - MUV_bin_edges[:-1]
            weights = (xhi - xlo).T/(MUVwidths)

            UVLFvalues_binned = np.einsum('ij,i->j', weights, UVLFvalues)
            pdf_binned = np.einsum('ji,jk->ik', weights, pdf_binned*UVLFvalues[:,None]) / UVLFvalues_binned[:,None]
            

        return log10etavalues, pdf_binned, UVLFvalues


    def log10eta_fromlog10xiion(self, log10xiion):
        'Returns log10(LHa/LUV) [both in erg/s] given xiion'

        _constxiionHaUV = 7.28e11 #erg/s/Hz
        HatoUV = 1.0/self.LF_Init.L_ergs_of_Mag(self.LF_Init.Mag_of_L_ergsHz(1.))*10**log10xiion/_constxiionHaUV

        return np.log10(HatoUV)


    def log10xiion_fromlog10eta(self, log10eta):
        'Returns log10(xiion) [in Hz/erg] given log10(LHa/LUV) [both in erg/s]'

        _constxiionHaUV = 7.28e11 #erg/s/Hz
        xiion = self.LF_Init.L_ergs_of_Mag(self.LF_Init.Mag_of_L_ergsHz(1.))*10**log10eta*_constxiionHaUV

        return np.log10(xiion)


    def cross_sigma_squared_PSD(self, GreensFunction1, GreensFunction2, CosmoParams, AstroParams, HMFinterp, LFParams):
        "Returns the cross sigma squared of two observables, given their window functions and SFH"
        

        _, windowfourier1= self.SFH_Init.WindowFourier(CosmoParams, AstroParams, HMFinterp, self.SFRD_Init, GreensFunction1, LFParams.zcenter, AstroParams._tagesMyr, pop = 2)
        
        _, windowfourier2= self.SFH_Init.WindowFourier(CosmoParams, AstroParams, HMFinterp, self.SFRD_Init, GreensFunction2, LFParams.zcenter, AstroParams._tagesMyr, pop = 2)

        omegalist = AstroParams.omega_PSFR #use the omegalist from the FFT, which is the same for all observables
        powerNL = AstroParams.PSFR_table #use the power spectrum from the FFT, which is the same for all observables

        _whichomegakeep = np.logical_and(omegalist > AstroParams._omegamin, omegalist < AstroParams._omegamax)

        sigmasqcross = np.trapezoid(powerNL * np.real(windowfourier1*np.conjugate(windowfourier2) )*_whichomegakeep, omegalist,axis=1)*2/(2*np.pi) #times 2 because + and - freqs
        
        #Also return the sigmas for each observable, which are needed for the PDF
        sigma1 = np.sqrt(np.trapezoid(powerNL * np.abs(windowfourier1)**2*_whichomegakeep, omegalist,axis=1)*2/(2*np.pi) )
        sigma2 = np.sqrt(np.trapezoid(powerNL * np.abs(windowfourier2)**2*_whichomegakeep, omegalist,axis=1)*2/(2*np.pi) )
        #And the means of the observables
        
        mean1 = np.trapezoid(GreensFunction1(AstroParams, AstroParams._tagesMyr, HMFinterp.Mhtab) * self.SFH_Init.SFH_II, AstroParams._tagesMyr*1e6, axis=1)
        mean2 = np.trapezoid(GreensFunction2(AstroParams, AstroParams._tagesMyr, HMFinterp.Mhtab) * self.SFH_Init.SFH_II, AstroParams._tagesMyr*1e6, axis=1)

        return mean1, mean2, sigma1, sigma2, sigmasqcross


