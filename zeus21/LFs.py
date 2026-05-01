"""

Compute UVLFs given our SFR and HMF models.

Author: Julian B. Muñoz
.UT Austin - June 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Sarah Libanore, Alessandra Venditti
BGU - April 2026 
"""

from . import cosmology
from . import constants
from .sfrd import Z_init, SFRD_class
from .cosmology import bias_Tinker

import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d


class LF:

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, LFParams, z_Init=None, SFRD_Init=None, vCB_input=False, J21LW_interp_input=False): 

        if z_Init is None:
            self.z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams)

        if SFRD_Init is None:
            self.SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init)  # TODO: wasting memory, add method overload for instantiating without initializing

        if(constants.NZ_TOINT>1):
            self.DZ_TOINT = np.linspace(-np.sqrt(constants.NZ_TOINT/3.), np.sqrt(constants.NZ_TOINT/3.),constants.NZ_TOINT)  # in sigmas around zcenter
        else:
            self.DZ_TOINT = np.array([0.0])
            
        self.WEIGHTS_TOINT = np.exp(-self.DZ_TOINT**2/2.)/np.sum(np.exp(-self.DZ_TOINT**2/2.))  # assumed Gaussian in z, fair


        self.biasM = np.array([bias_Tinker(CosmoParams, HMFinterp.sigma_int(HMFinterp.Mhtab,LFParams.zcenter+dz*LFParams.zwidth)) for dz in self.DZ_TOINT])
        
        if LFParams.FLAG_COMPUTE_UVLF:
            self.compute_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, "UV", vCB_input, J21LW_interp_input)

        if LFParams.FLAG_COMPUTE_HaLF:
            if AstroParams.USE_POPIII:
                raise ValueError('PopIII are not implemented for Ha')

            self.compute_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, "Ha", vCB_input, J21LW_interp_input)


    def MUV_of_SFR(self, SFRtab, kappaUV):
        'returns MUV, uses SFR. Dust added later in loglike.'
        # convert SFR to MUVs
        LUVtab = SFRtab/kappaUV
        MUVtab = constants.LUV1500A_toMUV - 2.5 * np.log10(LUVtab)  # AB magnitude
        return MUVtab
    

    def compute_LFbias_binned(self, CosmoParams, AstroParams, HMFinterp, LFParams, which_band="UV", vCB_input=False, J21LW_interp_input=False):

        output = self.compute_pop_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, pop=2, vCB=False, J21LW_interp=False, which_band=which_band)

        self.UVLF_pop2_binned = output[0]
        self.UVbias_pop2_binned = output[1]

        if AstroParams.USE_POPIII:
            if not vCB_input:
                vCB = CosmoParams.vcb_avg
            else:
                vCB = vCB_input 

            if not J21LW_interp_input:
                J21LW_interp = self.SFRD_Init.J21LW_interp_conv_avg
            else:
                J21LW_interp = J21LW_interp_input

            outputIII = self.compute_pop_LFbias_binned(CosmoParams, AstroParams, HMFinterp, LFParams, pop=3, vCB=vCB, J21LW_interp=J21LW_interp, which_band=which_band)
            self.UVLF_pop3_binned= outputIII[0]
            self.UVbias_pop3_binned= outputIII[1]

        else: 
            self.UVLF_pop3_binned = np.zeros_like(self.UVLF_pop2_binned)
            self.UVbias_pop3_binned = np.zeros_like(self.UVbias_pop2_binned)


        self.UVLF_binned = self.UVLF_pop2_binned + self.UVLF_pop3_binned
        self.UVbias_binned = self.UVbias_pop2_binned + self.UVbias_pop3_binned


        return 1 
            


    def compute_pop_LFbias_binned(self, CosmoParams, AstroParams, HMFinterp, LFParams, pop, which_band, vCB=False, J21LW_interp=False):
        'Binned UVLF in units of 1/Mpc^3/mag, for bins at <zcenter> with a Gaussian width zwidth, centered at MUV centers with tophat width MUVwidths. z width only in HMF since that varies the most rapidly. If flag RETURNBIAS set to true it returns number-avgd bias instead of UVLF, still have to divide by UVLF'
        

        if(AstroParams.FLAG_USE_PSD == True):  # MUV and sigmaUV derived from integrating SFH --> TODO: fix

            if which_band == "UV":
                LUV_short, sigmaLUV_short = sfrd.meanandsigma_observable_PSD(AstroParams, CosmoParams, HMFinterp,  AstroParams.Greens_function_LUV_Short, LFParams.zcenter)

                LUV_long, sigmaLUV_long = sfrd.meanandsigma_observable_PSD(AstroParams, CosmoParams, HMFinterp,  AstroParams.Greens_function_LUV_Long, LFParams.zcenter)

                logLormag_avglist, sigma_dex = sfrd.sigma_MUV_from_meansandsigmas(LUV_short, LUV_long, sigmaLUV_short, sigmaLUV_long)

                logLormag_avglist = np.fmin(logLormag_avglist, constants._MAGMAX_UV)

            elif which_band == "Ha":

                L_avglist, sigma_ln = sfrd.meanandsigma_observable_PSD(AstroParams, CosmoParams, HMFinterp, AstroParams.Greens_function_LHa, LFParams.zcenter)
       
                logLormag_avglist = sfrd.mean_log10(L_avglist, sigma_ln) 
                sigma_dex = sfrd.sigma_log10(L_avglist, sigma_ln)

                logLormag_avglist = np.fmax(logLormag_avglist,constants._MAGMIN_Ha) #cut to avoid -inf    

                sigma_dex = np.fmax(sigma_dex, 0.1) #avoid numerical issues with zero sigma

            else:
                raise ValueError('Only UV and Ha LF can be computed.')

        else:  # standard Munoz+23 model LUV \propto SFR \propto Mgdot*fstar

            if which_band == "UV":

                SFRlist = self.SFRD_Init.SFR(AstroParams, CosmoParams, HMFinterp, HMFinterp.Mhtab, LFParams.zcenter, pop, vCB, J21LW_interp)   

                sigma_dex = LFParams.sigmaUV  

                if (LFParams.FLAG_RENORMALIZE_LUV):  # lower the LUV (or SFR) to recover the true avg, not log-avg
                    SFRlist/= np.exp((np.log(10)/2.5*sigma_dex)**2/2.0)
                    
                logLormag_avglist = self.MUV_of_SFR(SFRlist, LFParams._kappaUV)  # avg for each Mh

            elif which_band == "Ha":
                raise ValueError('FLAG_USE_PSD=False not implemented in HaLF_binned()')

            else:
                raise ValueError('Only UV and Ha LF can be computed.')

            
        HMFtab = np.array([HMFinterp.HMF_int(HMFinterp.Mhtab, LFParams.zcenter+dz*LFParams.zwidth) for dz in self.DZ_TOINT])

        HMFcurr = np.sum(self.WEIGHTS_TOINT * HMFtab.T, axis=1)
        halobiascurr = np.sum(self.WEIGHTS_TOINT * HMFtab.T * self.biasM.T, axis=1)

        # cannot directly 'dust' the theory since the properties of the IRX-beta relation are calibrated on observed MUV. Recursion instead:

        logLormag_avglist = np.where(np.isfinite(logLormag_avglist), logLormag_avglist, 0.)
        curr_logLormag = logLormag_avglist
        

        if (LFParams.DUST_FLAG):
            curr2 = np.ones_like(curr_logLormag)
            while(np.sum(np.abs((curr2-curr_logLormag)/curr_logLormag)) > 0.02):
                curr2 = curr_logLormag
                curr_logLormag = logLormag_avglist + self.dust_attenuation(LFParams, LFParams.zcenter, curr_logLormag, "UV")
            
            if LFParams.sigma_times_AUV_dust != 0.:
                sigma_dust = np.fmax(0.0, LFParams.sigma_times_AUV_dust) * self.dust_attenuation(LFParams, LFParams.zcenter, curr_logLormag, "UV")
            else:
                sigma_dust = 0.
        else:
            sigma_dust = 0.0

        sigma = np.sqrt(sigma_dex**2 + sigma_dust**2) #add dust sigma, if any, to the UV sigma
        sigma = np.fmax(sigma, 0.2) #avoid numerical issues with zero sigma

            
        if which_band == "UV":
            cuthi = LFParams.MUVcenters + LFParams.MUVwidths/2.
            cutlo = LFParams.MUVcenters - LFParams.MUVwidths/2.
        elif which_band == "Ha":
            cuthi = LFParams.log10LHacenters +  LFParams.log10LHawidths/2.
            cutlo = LFParams.log10LHacenters -  LFParams.log10LHawidths/2.

        xhi = np.subtract.outer(cuthi, curr_logLormag)/(np.sqrt(2) * sigma)
        xlo = np.subtract.outer(cutlo, curr_logLormag)/(np.sqrt(2) * sigma)
        weights = (erf(xhi) - erf(xlo)).T/(2.0 * LFParams.MUVwidths)
        

        self.test = cuthi

        LF = np.trapezoid(weights.T * HMFcurr, HMFinterp.Mhtab, axis=-1)  # TODO: check consistency without fduty
        bias = np.trapezoid(weights.T * halobiascurr, HMFinterp.Mhtab, axis=-1)  # TODO: check consistency without fduty

        return LF, bias



    #####Here the dust attenuation
    def dust_attenuation(self, LFParams, z, logL_or_mag, which_band):
        'Average attenuation A as a function of OBSERVED z and magnitude. If using on theory iterate until convergence. HIGH_Z_DUST is whether to do dust at higher z than 0 or set to 0. Fix at \beta(z=8) result if so'

        if which_band == "UV":

            MUV = logL_or_mag
            betacurr = self.betaUV_dust(LFParams, z, MUV)
                        
            sigmabeta = 0.34 #from Bouwens 2014
            
            Auv = LFParams.C0dust + 0.2*np.log(10)*sigmabeta**2 * LFParams.C1dust**2 + LFParams.C1dust * betacurr
            Auv=Auv.T
            if not (LFParams.HIGH_Z_DUST):
                Auv*=np.heaviside(LFParams._zmaxdata - z,0.5)
            
            Adust = np.fmax(Auv.T, 0.0)

        elif which_band == "Ha":

            'Average attenuation A as a function of z and log10LHa.'
            #TODO: made up see how to calibrate it. Unused in current implementation (set Ha DUST = False)
            #conjured approximation - lower at high z and fainter 

            log10LHa = logL_or_mag
            AHa = 0.5 * (1 + 0.3 * (log10LHa - 42.0))
            Adust = -0.4 * np.fmax(AHa, 0.0) #no negative dust attenuation
            #-0.4* instead of +1* here since its log10L not mag

        return Adust


    def betaUV_dust(self, LFParams, z, MUV):

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

        elif LFParams.DUST_model == "Bouwens13": 

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
        "Corrects the observed UVLF from the assumed cosmology CosmoParams to another with CosmoParams_out. Note: no dust correction since it's applied directly to theory->model"

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


'''
EXTRA FUNCTIONS
'''


def PDF_log10HaUVratio(LUV_mean, LHa_mean, sigmaLHa, sigmasquaredcross, log10etavalues = None):
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
    sigma_of_log10LHa = sfrd.sigma_log10(sigmaLHa, LHa_mean)

    if log10etavalues is None:  # If not provided, create a default range
        log10etavalues = np.linspace(-3.5,-1.3,55)
    etavalues = 10**log10etavalues
    _LHavalues = np.outer(etavalues,_B)/(1-np.outer(etavalues,_A)) #recalculate the LHa values from eta values

    muHa, sigmaHa = mean_of_log10LHa*np.log(10), sigma_of_log10LHa*np.log(10) #mean and std of ln(Ha), a gaussian varible
    PDFLHalognormal = sfrd.lognormal_pdf(_LHavalues, muHa, sigmaHa)
    dydx = _B/(_B+_A*_LHavalues) * 1/(_LHavalues * np.log(10))
    PDFlog10eta = PDFLHalognormal/np.abs(dydx)

    return log10etavalues, PDFlog10eta



def PDF_HaUV_ratio(UserParams, CosmoParams, AstroParams, HMFinterp, LFParams, z_Init = None, SFRD_Init = None, LFclass=None, log10etavalues=None, FLAG_supersample_MUV = False):
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
    
    if (AstroParams.FLAG_USE_PSD == False):
        raise ValueError('FLAG_USE_PSD=False not implemented in PDF_HaUV_ratio()')

    if z_Init is None:
        z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams)

    if SFRD_Init is None:
        SFRD_Init = SFRD_class(UserParams, CosmoParams, AstroParams, HMFinterp, z_Init)

    if LFclass is None:
        LFclass = LF(UserParams, CosmoParams, AstroParams, HMFinterp, LFParams, z_Init=z_Init, SFRD_Init=SFRD_Init, vCB_input=False, J21LW_interp_input=False)

    if log10etavalues is None:  # If not provided, create a default range
        log10etavalues = np.linspace(-3.5,-1.0,20)

    HMFtab = HMFinterp.HMF_int(HMFinterp.Mhtab,LFParams.zcenter) 

    meanLUVshort, meanLHa, sigmaLUVshort, sigmaLHa, sigmasqcross = sfrd.cross_sigma_squared_PSD(AstroParams, CosmoParams, HMFinterp, AstroParams.Greens_function_LUV_Short,AstroParams.Greens_function_LHa, LFParams.zcenter)

    _Acoeff = sigmasqcross/sigmaLHa**2 

    meanLUVlong, sigmaLUVlong = sfrd.meanandsigma_observable_PSD(AstroParams, CosmoParams, HMFinterp, AstroParams.Greens_function_LUV_Long, LFParams.zcenter)

    #get the UV-A*Ha "excess" UV luminosity, not exactly lognormal so use the same trick as for MUV:
    meanLUV_excess_short = meanLUVshort - _Acoeff * meanLHa
    sigmaLUV_excess_short  = np.sqrt(sigmaLUVshort**2 + _Acoeff**2 * sigmaLHa**2 - 2.0 * _Acoeff * sigmasqcross)
    MUVbar_excess, sigmaMUV_excess = sfrd.sigma_MUV_from_meansandsigmas(meanLUV_excess_short, meanLUVlong, sigmaLUV_excess_short, sigmaLUVlong)


    MUVavglist, sigmaMUV = sfrd.sigma_MUV_from_meansandsigmas(meanLUVshort, meanLUVlong, sigmaLUVshort, sigmaLUVlong)
    MUVavglist = np.fmin(MUVavglist,constants._MAGMAX_UV)
    sigma_times_AUV_dust = np.fmax(0.0, LFParams.sigma_times_AUV_dust) #assumed constant, not derived from SFH

    #these are the parameters of the closest lognormal to each Ha, UV, and UVx
    muHa, sigmaHa = sfrd.mean_log10(sigmaLHa, meanLHa)*np.log(10), sfrd.sigma_log10(sigmaLHa, meanLHa)*np.log(10) #mean and std of ln(LUVx), a gaussian varible

    muUV, sigmaUV = np.log(sfrd.LUV_of_MUV(MUVavglist)), sigmaMUV*np.log(10)/2.5 
    muUVx, sigmaUVx = np.log(sfrd.LUV_of_MUV(MUVbar_excess)), sigmaMUV_excess*np.log(10)/2.5 

    if (FLAG_supersample_MUV==True):
        _NLuvsupersample = 99 #number of LUV values to supersample
        _LUVlist = np.logspace(35,46,_NLuvsupersample) #in case you want to integrate and then bin
    else:
        _LUVlist = sfrd.LUV_of_MUV(LFParams.MUVcenters) #this will give mean of <log10(LUV)> for each MUV bin


    #This is used for assigning galaxies to MUV bins, so we add dust correction since the Ha/UV ratios are dust corrected but they're binned in MUVobs
    currMUV = MUVavglist
    currMUV2 = np.ones_like(currMUV)
    while(np.sum(np.abs((currMUV2-currMUV)/currMUV)) > 0.02):
        currMUV2 = currMUV
        currMUV = MUVavglist + LFclass.dust_attenuation(LFParams,LFParams.zcenter,currMUV,"UV")

    sigmaUV_dust = sigma_times_AUV_dust * LFclass.dust_attenuation(LFParams,LFParams.zcenter,currMUV,"UV")

    sigmaMUV_obs = np.sqrt(sigmaMUV**2 + sigmaUV_dust**2) #add dust sigma, if any, to the UV sigma
    sigmaMUV_obs = np.fmax(sigmaMUV_obs, 0.2) #avoid numerical issues with zero sigma

    muUV_obs, sigmaUV_obs = np.log(sfrd.LUV_of_MUV(currMUV)), sigmaMUV_obs*np.log(10)/2.5
    PlnLUV = sfrd.normal_pdf(np.log(_LUVlist), muUV_obs, sigmaUV_obs, dimy=1)+1e-99 #to avoid Nans
    UVLFvalues = np.trapezoid(HMFtab[:,None] * PlnLUV, HMFinterp.Mhtab, axis=0)


    #we exploit the fact that P(log10LHa - log10LHabar | LUV) doesnt change for LUV > LUVbar(Mh)
    #so we set LUV = LUVbar(Mh) for each Mh. (we dont modify P(LUV) since that is the UVLF, not the Ha/UV ratio)
    _meanLUV_ofMh = np.exp(muUV[:, None])
    _LUVforcalculation = np.minimum(_LUVlist[None,:], _meanLUV_ofMh) #Mh x LUVs, so that for LUV>LUVbar we recover the LUVbar result
    PlnLUVforcalculation = sfrd.normal_pdf(np.log(_LUVforcalculation), muUV, sigmaUV, dimy=1)+1e-99 #to avoid Nans

    _LHacalc = _LUVforcalculation[:,:,None] * 10**log10etavalues[None,None,:]
    PlnLHa = sfrd.normal_pdf(np.log(_LHacalc), muHa, sigmaHa, dimy=2)

    _LUVx = np.fmax(1.0, _LUVforcalculation[:,:,None] - _LHacalc [:,:,:] * _Acoeff[:, None,None]) #LUVx = _LUVforcalculation - A * LHa, where A is the coefficient for each MUV
    PlnLUVx_fixedHa = sfrd.normal_pdf(np.log(_LUVx), muUVx, sigmaUVx,dimy=2)
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
        MUVs = sfrd.MUV_of_LUV(_LUVlist) #here we use _LUVlist since its for the P(LUV) not the Ha/UV ratio
        xhi = np.heaviside(np.subtract.outer(MUVcuthi, MUVs),0.5)
        xlo = np.heaviside(np.subtract.outer(MUVcutlo, MUVs),0.5)
        MUVwidths = MUV_bin_edges[1:] - MUV_bin_edges[:-1]
        weights = (xhi - xlo).T/(MUVwidths)

        UVLFvalues_binned = np.einsum('ij,i->j', weights, UVLFvalues)
        pdf_binned = np.einsum('ji,jk->ik', weights, pdf_binned*UVLFvalues[:,None]) / UVLFvalues_binned[:,None]
        
    return log10etavalues, pdf_binned, UVLFvalues


