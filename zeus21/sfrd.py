"""

Bulk of the Zeus21 calculation. Compute sSFRD from cosmology.

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

Edited by Sarah Libanore, Emilie Thelie, Hector Afonso G. Cruz, Alessandra Venditti, Emily Bregou
UT Austin - April 2026 
BGU - June 2026
"""

from . import cosmology
from . import constants

import numpy as np
import astropy
from astropy import units as u

import scipy
from scipy import interpolate


class Z_init:
    """
    Initial redshift matrices for the calculation
    
    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    
    Attributes
    ----------
    dlogzint : array 
        Set the log step for the redshift binning, based on the required input 
    zintegral : array 
        Redshift array over which will be performed all integration and for which the output will be computed
    zGreaterMatrix : matrix
        Redshift associated with the distance at radius R. Dimension (z, R)
    zGreaterMatrix_nonan : matrix 
        Redshift associated with the distance at radius R; when z > zmax_AstroBreak (50 by default in constants), we set z > 100 to prevent computing things where we don't trust the astrophysical model. Dimension (z, R)
    """ 

    def __init__(self, UserParams, CosmoParams):

        zmin_integral = UserParams.zmin
        zmax_integral = constants.ZMAX_INTEGRAL
        
        Nzintegral = np.ceil(1.0 + np.log(zmax_integral/zmin_integral)/UserParams.dlogzint_target).astype(int)
        
        self.dlogzint = np.log(zmax_integral/zmin_integral)/(Nzintegral-1.0) #exact value rather than input target above
        self.zintegral = np.geomspace(zmin_integral, zmax_integral, Nzintegral) #note these are also the z at which we "observe", to share computational load
        
        # define table of redshifts 
        rGreaterMatrix = np.transpose([CosmoParams.chiofzint(self.zintegral)]) + CosmoParams._Rtabsmoo
        self.zGreaterMatrix = CosmoParams.zfofRint(rGreaterMatrix)

        if CosmoParams.Flag_emulate_21cmfast: 
            #   21cmFAST takes the redshift to be at the midpoint of the two shells
            # TODO: HECTOR CHANGES
            self.zGreaterMatrix = np.append(self.zintegral.reshape(len(self.zGreaterMatrix), 1), self.zGreaterMatrix, axis = 1)
            self.zGreaterMatrix = (self.zGreaterMatrix[:, 1:] + self.zGreaterMatrix[:, :-1])/2              
        else:
            self.zGreaterMatrix[rGreaterMatrix > CosmoParams.chiofzint(constants.zmax_AstroBreak)] = np.nan
            
        self.zGreaterMatrix_nonan = np.nan_to_num(self.zGreaterMatrix, nan = 100) # prevent calculation where the astro model is not trusted 


class SFRD_class:
    """
    Compute all quantities and methods associated with the star formation rate density and the astrophysical model 
    
    Parameters
    ----------
    UserParams : UserParams class
    CosmoParams : CosmoParams class
    AstroParams : AstroParams class
    HMFinterp : HMFinterp class
    z_Init : Z_init class, optional
        Initial redshift matrices for the calculation (see sfrd.py for details).
        Default is None.

    Attributes
    ----------
    SFRD_II_interp : interpolator
        Average SFRD for popII stars, interpolated over redshift.
    J_21_LW_II : interpolator
        Lyman-Werner flux from popII stars, units of erg/s/cm^2/Hz/s,, interpolated over redshift
    J21LW_interp_conv_avg : interpolator
        Lyman-Werner flux iteratively computed to account for popIII contribution, interpolated over redshift
    SFRD_III_cnvg_interp : : interpolator
        Average SFRD for popIII stars, determines part-of and is affected by the LW flux; interpolated over redshift.
    J_21_LW_III : interpolator
        Lyman-Werner flux, units of erg/s/cm^2/Hz/s from popIII stars,, interpolated over redshift
    SFRD_II_avg : array 
        Average SFRD for popII stars, units Msun/yr
    SFRD_III_avg : array 
        Average SFRD for popIII stars, units Msun/yr
    SFRD_avg : array 
        Total zverage SFRD, units Msun/yr
    SFRDbar2D_II : matrix
        Average SFRD for popII computed at z corresponding to each shell.
    SFRDbar2D_III : matrix
        Average SFRD for popIII computed at z corresponding to each shell
<    fesctab_II : array
        Escape fraction for popII, z-independent, as function of the halo mass
     fesctab_III : array
        Escape fraction for popIII, z-independent, as function of the halo mass
    reio_integrand_II_interp : integrand
        Number of ionizing photons produced by popII, interpolated in redshift
    reio_integrand_II_interp : integrand
        Number of ionizing photons produced by popIII, interpolated in redshift
    niondot_avg_II : array
        Number of ionizing photons produced by popII computed at the redshifts of the analysis
    niondot_avg_III : array
        Number of ionizing photons produced by popIII computed at the redshifts of the analysis
    niondot_avg : array
        Number of ionizing photons produce at the redshifts of the analysis
    sigmaofRtab : matrix
        Variance of the matter field on scales associated with the shells and at the observed rerdshift 
    Matom : method
        Minimum mass for atomic cooling halos at given redshift
    Mmol_0 : method
        Minimum mass for molecular halos without LW or VCB feedback
    Mmol_vcb : method
        Minimum mass for molecular halos without LW feedback
    Mmol_LW : method
        Minimum mass for molecular halos without VCB feedback
    Mmol : method
        Minimum mass for molecular halos with LW and VCB feedback
    dMh_dt : method
        Mass accretion rate, in units of M_sun/yr
    fstar_ofz : method
        Star formation efficiency generative function
    fduty : method
        Duty cycle to damp star formation in low or high mass halos or both 
    SFE_II : method
        Star formation efficiency for popII stars
    SFE_III : method
        Star formation efficiency for popIII stars
    SFE : method
        Total star formation efficiency 
    SFR : method 
        Star formation rate
    SRFD_integrand : method
        Integrand to compute the star formation rate density
    J_LW_21 : method
        Mean cosmological LW background specific intensity
    J_LW_Discrete : method 
        Radial kernel of the LW specific intensity before R integration
    dSFRDIII_dJ : method
        Response of the popIII star formation rate density to the LW background
    fesc_II : method
        Escape fraction of ionizing photons in halos hosting popII stars
    fesc_III : method
        Escape fraction of ionizing photons in halos hosting popIII stars
    compute_sigmaR_nu : method
        Compute the local mass function conditioned over the environment
    compute_gamma : method 
        Compute linear and quadratic gamma exponents for the SFRD-delta (popII+popIII) and niondot-delta (only popII) lognormal approximations
    gamma_II_index2D : array 
        Linear gamma exponent for SFRD in popII
    gamma2_II_index2D : array 
        Quadratic gamma exponent for SFRD in popII
    gamma_niondot_II_index2D : array 
        Linear gamma exponent for niondot in popII
    gamma2_niondot_II_index2D : array 
        Quadratic gamma exponent for niondot in popII
    gamma_III_index2D : array 
        Linear gamma exponent for SFRD in popIII
    gamma2_III_index2D : array 
        Quadratic gamma exponent for SFRD in popIII
    compute_numerical_der_gamma : method
        Compute first and second numerical derivatives of an array wrt the other (used for SFRD and niondot wrt delta)
    """ 

    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None):

        # if z_Init is not provided, we initialize it here. This allows us to avoid redundant computations if they were already initialized in the parent class and passed as arguments.
        if z_Init is None:
            z_Init = Z_init(UserParams=UserParams, CosmoParams=CosmoParams)  

        zSFRDflat = np.geomspace(UserParams.zmin, constants.zmax_AstroBreak, 128) # extend to z = constants.zmax_AstroBreak for extrapolation purposes. Higher in z than zInit.zintegral
        zSFRD, mArray = np.meshgrid(zSFRDflat, HMFinterp.Mhtab, indexing = 'ij', sparse = True) # create redshift and halo mass matrices, dimension (z, Mh)

        init_J21LW_interp = interpolate.interp1d(zSFRDflat, np.zeros_like(zSFRDflat), kind = 'linear', bounds_error = False, fill_value = 0,) # initialize no LW background, used to compute Mmol() function, NOT the individual Pop II and III LW background

        SFRD_II_avg = np.trapezoid(self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=2), HMFinterp.logtabMh, axis = 1) # average SFRD 
        self.SFRD_II_interp = interpolate.interp1d(zSFRDflat, SFRD_II_avg, kind = 'cubic', bounds_error = False, fill_value = 0,) 

        J21LW_II = self.J_LW_21(CosmoParams, AstroParams, SFRD_II_avg, zSFRDflat, pop=2) # LW specific intensity from popII
        self.J_21_LW_II = interpolate.interp1d(zSFRDflat, J21LW_II, kind = 'cubic')(z_Init.zintegral) 

        if AstroParams.USE_POPIII:

            # initialize popIII SFRD, update iteratively to account for LW feedback
            SFRD_III_Iter_Matrix = [np.trapezoid(self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=3, vCB=CosmoParams.vcb_avg, J21LW_interp=init_J21LW_interp), HMFinterp.logtabMh, axis = 1)] 

            errorTolerance = 0.001 # 0.1 percent accuracy

            recur_iterate_Flag = True
            while recur_iterate_Flag:

                J21LW_III_iter = self.J_LW_21(CosmoParams, AstroParams, SFRD_III_Iter_Matrix[-1], zSFRDflat, pop=3)
                loop_J21LW_interp = interpolate.interp1d(zSFRDflat, J21LW_II + J21LW_III_iter, kind = 'linear', fill_value = 0, bounds_error = False)

                SFRD_III_avg_n = np.trapezoid(self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=3, vCB=CosmoParams.vcb_avg, J21LW_interp= loop_J21LW_interp), HMFinterp.logtabMh, axis = 1) # correct through LW feedback
                SFRD_III_Iter_Matrix.append(SFRD_III_avg_n)

                if max(SFRD_III_Iter_Matrix[-1]/SFRD_III_Iter_Matrix[-2]) < 1.0 + errorTolerance and min(SFRD_III_Iter_Matrix[-1]/SFRD_III_Iter_Matrix[-2]) > 1.0 - errorTolerance:
                    recur_iterate_Flag = False

            self.J21LW_interp_conv_avg = loop_J21LW_interp

            self.SFRD_III_cnvg_interp = interpolate.interp1d(zSFRDflat, SFRD_III_Iter_Matrix[-1], kind = 'cubic', bounds_error = False, fill_value = 0) # SFRD for popIII
            self.J_21_LW_III = interpolate.interp1d(zSFRDflat, J21LW_III_iter, kind = 'cubic')(z_Init.zintegral) # LW flux from popIIII

        else:
            self.SFRD_III_cnvg_interp = interpolate.interp1d(zSFRDflat, np.zeros_like(zSFRDflat), kind = 'cubic', bounds_error = False, fill_value = 0)

        self.SFRD_II_avg = self.SFRD_II_interp(z_Init.zintegral)
        self.SFRD_III_avg = self.SFRD_III_cnvg_interp(z_Init.zintegral)
        self.SFRD_avg = self.SFRD_II_avg + self.SFRD_III_avg

        self.SFRDbar2D_II = self.SFRD_II_interp(np.nan_to_num(z_Init.zGreaterMatrix, nan = 100)) # dimension (z,R)
        self.SFRDbar2D_III = self.SFRD_III_cnvg_interp(np.nan_to_num(z_Init.zGreaterMatrix, nan = 100)) # dimension (z,R)
            
        # Reionization
        self.fesctab_II = self.fesc_II(AstroParams, HMFinterp.Mhtab) # prepare fesc(M) table -- z independent for now 
        self.fesctab_III = self.fesc_III(AstroParams, HMFinterp.Mhtab) #PopIII prepare fesc(M) table -- z independent for now 

        # prepare integrand to compute number of ionizing photons
        reio_integrand_II = self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=2) 
        reio_integrand_III = self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zSFRD, pop=3, vCB=CosmoParams.vcb_avg, J21LW_interp=init_J21LW_interp)
        niondot_avg_II = AstroParams.N_ion_perbaryon_II/cosmology.rho_baryon(CosmoParams,0.) * np.trapezoid(reio_integrand_II * self.fesctab_II, HMFinterp.logtabMh, axis = 1)  # number of ionizing photons produced by popII 
        niondot_avg_III = AstroParams.N_ion_perbaryon_III/cosmology.rho_baryon(CosmoParams,0.) * np.trapezoid(reio_integrand_III * self.fesctab_III, HMFinterp.logtabMh, axis = 1) # number of ionizing photons produced by popIIII

        self.reio_integrand_II_interp = interpolate.interp1d(zSFRDflat, niondot_avg_II, kind = 'cubic', bounds_error = False, fill_value = 0)
        self.reio_integrand_III_interp = interpolate.interp1d(zSFRDflat, niondot_avg_III, kind = 'cubic', bounds_error = False, fill_value = 0)

        self.niondot_avg_II = self.reio_integrand_II_interp(z_Init.zintegral)
        self.niondot_avg_III = self.reio_integrand_III_interp(z_Init.zintegral)
        self.niondot_avg = self.niondot_avg_II + self.niondot_avg_III

        if not UserParams.DO_ONLY_GLOBAL:
            # compute gamma coefficients required by the power spectrum (see correlations.py for detail)
            self.sigmaofRtab = np.array([HMFinterp.sigmaR_int(CosmoParams._Rtabsmoo, zz) for zz in z_Init.zintegral]) 

            self.compute_gamma(CosmoParams, AstroParams, HMFinterp, z_Init.zintegral, CosmoParams._Rtabsmoo, HMFinterp.Mhtab, self.sigmaofRtab, self.fesctab_II)


    def Matom(self, z):
        """
        Compute minimum mass for atomic halos

        Parameters 
        ----------
        z : float
            Redshift

        Returns
        ----------
        Matom : float 
            Minimum halo mass, in Msun         
        """

        Matom = 3.3e7 * pow((1.+z)/(21.),-3./2)

        return Matom 


    def Mmol_0(self, z):
        """
        Compute minimum mass for molecular halos without LW or VCB feedback

        Parameters 
        ----------
        z : float
            Redshift

        Returns
        ----------
        Mmol_0 : float 
            Minimum halo mass, in Msun         
        """

        Mmol_0 = 3.3e7 * (1.+z)**(-1.5)

        return Mmol_0


    def Mmol_vcb(self, CosmoParams, AstroParams, z, vCB):
        """
        Compute minimum mass for molecular halos without LW feedback

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        z : float
            Redshift
        vCB : float
            Baryon-DM relative velocity

        Returns
        ----------
        Mmol_vcb : float 
            Minimum halo mass, in Msun         
        """

        mmolBase = self.Mmol_0(z)
        vcbFeedback = pow(1 + AstroParams.A_vcb * vCB / CosmoParams.sigma_vcb, AstroParams.beta_vcb)

        Mmol_vcb = mmolBase * vcbFeedback

        return Mmol_vcb


    def Mmol_LW(self, AstroParams, J21LW_interp, z):
        """
        Compute minimum mass for molecular halos without VCB feedback

        Parameters 
        ----------
        AstroParams : AstroParams class
        J21LWinterp : interpolator
            Interpolator of the LW flux, function of z 
        z : float
            Redshift

        Returns
        ----------
        Mmol_LW : float 
            Minimum halo mass, in Msun         
        """

        mmolBase = self.Mmol_0(z)
        lwFeedback = 1 + AstroParams.A_LW*pow(J21LW_interp(z), AstroParams.beta_LW)

        Mmol_LW = mmolBase * lwFeedback

        return Mmol_LW 


    def Mmol(self, CosmoParams, AstroParams, J21LW_interp, z, vCB):
        """
        Compute minimum mass for molecular halos with LW and VCB feedback

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        J21LWinterp : interpolator
            Interpolator of the LW flux, function of z 
        z : float
            Redshift
        vCB : float
            Baryon-DM relative velocity

        Returns
        ----------
        Mmol : float 
            Minimum halo mass, in Msun         
        """

        mmolBase = self.Mmol_0(z)
        vcbFeedback = pow(1 + AstroParams.A_vcb * vCB / CosmoParams.sigma_vcb, AstroParams.beta_vcb)
        lwFeedback = 1 + AstroParams.A_LW*pow(J21LW_interp(z), AstroParams.beta_LW)
        
        Mmol = mmolBase * vcbFeedback * lwFeedback

        return Mmol


    def dMh_dt(self, CosmoParams, AstroParams, HMFinterp, massVector, z):
        """
        Compute halo mass accretion rate, in units of M_sun/yr

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        HMFinterp : HMFinterp class
        massVector : array
            Halo masses
        z : array
            Redshift

        Returns
        ----------
        Mhdot : array 
            Halo mass accretion rate       
        """

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

            elif(AstroParams.accretion_model == 'RP16'): # Fitting function to Rodríguez-Puebla+16 N-body simulations (eq. 11, dynamically averaged parameters from table 2)
                a = (1+z)**-1
                beta = 10**(2.73-(1.828*a)+(0.654*a**2))
                alpha = 1 + (0.329*a) - (0.206*a**2)

                # factors of h are accounted for to give units of M_sun/year for halo masses in units of M_sun:
                Mhdot = beta * (massVector/1e12)**alpha * cosmology.Hub(CosmoParams, z) / (100*CosmoParams.h_fid)
                
            else:
                print("ERROR! Have to choose an accretion model in AstroParams (accretion_model)")

                return -1 
            
            Mhdot = dMhdz*cosmology.Hubinvyr(CosmoParams,z)*(1.0+z)
            
        else: #21cmfast-like
            Mhdot = massVector/AstroParams.tstar*cosmology.Hubinvyr(CosmoParams,z)

        return Mhdot


    def fstar_ofz(self, CosmoParams, z, massVector, eps, dlog10eps, zpiv, Mc, alphastar, betastar, fstarmax):  
        """
        Compute star formation efficiency as function of z -- changing the parameters the user can run both popII and popIII

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        z : float
            Redshift
        massVector : array
            Halo masses
        eps : float
            Star formation efficiency at pivot redshift and mass
        dlog10eps : float
            Logharitmic redshift evolution 
        zpiv : float
            Pivot reference redshift
        Mc : float
            Pivot reference halo mass
        alphastar : float
            Power-law coefficient
        betastar : float 
            Second power-law coefficient
        fstarmax : float
            Cap 

        Returns
        ----------
        fstar : array 
            Star formation efficiency  
        """

        epsstar_ofz = eps * 10**(dlog10eps * (z-zpiv))

        if CosmoParams.Flag_emulate_21cmfast:
            # 21cmFAST-like
            fstar = CosmoParams.OmegaB/CosmoParams.OmegaM * np.clip(epsstar_ofz\
            /(pow(massVector/Mc, -alphastar)), 0, fstarmax)

        else:
            # GALLUMI-like 
            fstar = CosmoParams.OmegaB/CosmoParams.OmegaM * np.clip(2.0 * epsstar_ofz\
            /(pow(massVector/Mc,- alphastar) + pow(massVector/Mc,-betastar)), 0, fstarmax)
        
        return fstar 
        

    def fduty(self, CosmoParams, AstroParams, massVector, z,  lower_cutoff=False, upper_cutoff=False, is_sharp_cutoff=False, vCB=False, J21LW_interp=False):  
        """
        Compute duty fraction to damp star formation 

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        massVector : array
            Halo masses
        z : float
            Redshift
        lower_cutoff : str or bool or float 
            Apply cutoff on low masses; if False does not apply; if str == {Mmol, Matom} computes the minimum mas; if float uses it as minimym mass 
        upper_cutoff : str or bool or float 
            Apply cutoff on high masses; if False does not apply; if str == {Matom} computes the minimum mas; if float uses it as minimym mass 
        is_sharp_cutoff : bool
            Use sharp cutoff 
        vCB : bool
            Include contribution from baryon-CDM relative velocity (popIII) or not (popII)
        J21LW_interp : interpolator
            Include contribution from LW feedback (popIII) or not (popII)

        Returns
        ----------
        fduty : array 
            Duty cycle 
        """
        
        # cutoff on the low mass end 
        if lower_cutoff:
            if lower_cutoff == "Mmol":
                Mlow = self.Mmol(CosmoParams, AstroParams, J21LW_interp, z, vCB)
            elif lower_cutoff == "Matom":
                Mlow = self.Matom(z)
            else:
                Mlow = lower_cutoff

            if is_sharp_cutoff:
                fduty_low = np.heaviside(massVector - Mlow, 0.5)
            else:
                fduty_low = np.exp(-Mlow/massVector)
        else:
            fduty_low = 1.

        # cutoff on the high mass end 
        if upper_cutoff:
            if upper_cutoff == "Matom":
                Mup = self.Matom(z)
            else:
                Mup = upper_cutoff

            if is_sharp_cutoff:
                fduty_up = np.heaviside(Mup - massVector, 0.5)
            else:
                fduty_up = np.exp(-massVector/Mup)
        else:
            fduty_up = 1.

        fduty = fduty_low * fduty_up
        
        return fduty


    def SFE_II(self, CosmoParams, AstroParams, massVector, z): 
        """
        Star formation efficiency for popII stars

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        massVector : array
            Halo masses
        z : float
            Redshift

        Returns
        ----------
        SFE : array 
            Star formation efficiency
        """
        
        fstarM = self.fstar_ofz(CosmoParams, z, massVector,   
                                AstroParams.epsstar, AstroParams.dlog10epsstardz, AstroParams._zpivot, 
                                AstroParams.Mc, AstroParams.alphastar, AstroParams.betastar, AstroParams.fstarmax)

        if not AstroParams.FLAG_MTURN_FIXED:
            fduty = self.fduty(CosmoParams, AstroParams, massVector, z, lower_cutoff="Matom", upper_cutoff=False, is_sharp_cutoff=AstroParams.FLAG_MTURN_SHARP)
        else:
            fduty = self.fduty(CosmoParams, AstroParams, massVector, z, lower_cutoff=AstroParams.Mturn_fixed, upper_cutoff=False, is_sharp_cutoff=AstroParams.FLAG_MTURN_SHARP)

        SFE = fstarM * fduty

        return SFE
    

    def SFE_III(self, CosmoParams, AstroParams, massVector, z, vCB, J21LW_interp):  
        """
        Star formation efficiency for popIII stars; includes both mini halos (default) and additional atomic cooling halo component

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        massVector : array
            Halo masses
        z : float
            Redshift
        vCB : bool
            Include contribution from baryon-CDM relative velocity (popIII) or not (popII)
        J21LW_interp : bool
            Include contribution from LW feedback (popIII) or not (popII)

        Returns
        ----------
        SFE_tot : array 
            Star formation efficiency
        """
        
        # default mini halo population 
        eps = AstroParams.epsstar_III  # TODO: fstar_III to epssstar_III?
        dlog10eps = AstroParams.dlog10epsstardz_III
        zpiv = AstroParams._zpivot_III
        Mc = AstroParams.Mc_III
        alphastar = AstroParams.alphastar_III  # TODO: decide if we want to keep (same for ACH component)
        betastar = AstroParams.betastar_III
        fstarM = self.fstar_ofz(CosmoParams, z, massVector, 
                                eps, dlog10eps, zpiv,
                                Mc, alphastar, betastar, AstroParams.fstarmax)
        fduty = self.fduty(CosmoParams, AstroParams, massVector, z,  lower_cutoff="Mmol", upper_cutoff="Matom", is_sharp_cutoff=False,  vCB=vCB, J21LW_interp=J21LW_interp)  # TODO: Do we want to allow the cut-off to not be sharp? 
        SFE = fstarM * fduty

        if AstroParams.USE_POPIII_ACH:
            # atomic cooling halo component from ??? TODO: add reference
            if not AstroParams.DETACH_III_ACH:
                eps_ACH = eps  # TODO: check consistency with MC component (defined at pivot mass?)
                dlog10eps_ACH = dlog10eps
                zpiv_ACH = zpiv
                Mc_ACH = Mc
                alphastar_ACH = alphastar
                betastar_ACH = betastar
            else:
                eps_ACH = AstroParams.epssstar_III_ACH
                dlog10eps_ACH = AstroParams.dlog10epsstardz_III_ACH
                zpiv_ACH = AstroParams._zpivot_III_ACH
                Mc_ACH = AstroParams.Mc_III_ACH
                alphastar_ACH = AstroParams.alphastar_III_ACH
                betastar_ACH = AstroParams.betastar_III_ACH       
                 
            fstarM_ACH = self.fstar_ofz(CosmoParams, z, massVector,
                                        eps_ACH, dlog10eps_ACH, zpiv_ACH,
                                        Mc_ACH, alphastar_ACH, betastar_ACH, AstroParams.fstarmax)
            fduty_ACH = self.fduty(CosmoParams, AstroParams, massVector, z,  lower_cutoff="Matom", upper_cutoff=AstroParams.Mup_III, is_sharp_cutoff=False)
            SFE_ACH = fstarM_ACH * fduty_ACH
        else:
            SFE_ACH = np.zeros_like(SFE)

        SFE_tot = SFE + SFE_ACH

        return SFE_tot


    def SFE(self, CosmoParams, AstroParams, massVector, z, pop, vCB = False, J21LW_interp = False):  
        """
        Total tar formation efficiency 

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        massVector : array
            Halo masses
        z : float
            Redshift
        pop : int
            Which population (2 for popII or 3 for popIII)
        vCB : bool
            Include contribution from baryon-CDM relative velocity (popIII) or not (popII)
        J21LW_interp : bool
            Include contribution from LW feedback (popIII) or not (popII)

        Returns
        ----------
        SFE : array 
            Star formation efficiency for the input population 
        """
        

        if (pop == 3 and not AstroParams.USE_POPIII):
            return 0  # skip whole routine if NOT using PopIII stars
        
        if pop == 2:
            SFE = self.SFE_II(CosmoParams, AstroParams, massVector, z)
        else:
            SFE = self.SFE_III(CosmoParams, AstroParams, massVector, z, vCB, J21LW_interp)

        return SFE 
    

    def SFR(self, CosmoParams, AstroParams, HMFinterp, massVector, z, pop, vCB = False, J21LW_interp = False):
        """
        Star formation rate in Msun/yr for given population 

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        HMFinterp : HMFinterp class
        massVector : array
            Halo masses
        z : float
            Redshift
        pop : int
            Which population (2 for popII or 3 for popIII)
        vCB : bool
            Include contribution from baryon-CDM relative velocity (popIII) or not (popII)
        J21LW_interp : bool
            Include contribution from LW feedback (popIII) or not (popII)

        Returns
        ----------
        SFR : array 
            Star formation rate
        """

        SFR = self.dMh_dt(CosmoParams, AstroParams, HMFinterp, massVector, z) * self.SFE(CosmoParams, AstroParams, massVector, z, pop, vCB, J21LW_interp)        
        
        return SFR


    def SFRD_integrand(self, CosmoParams, AstroParams, HMFinterp, massVector, z, pop, vCB = False, J21LW_interp = False):
        """
        Integrand for the star formation rate density for a given population 

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        HMFinterp : HMFinterp class
        massVector : array
            Halo masses
        z : float
            Redshift
        pop : int
            Which population (2 for popII or 3 for popIII)
        vCB : bool
            Include contribution from baryon-CDM relative velocity (popIII) or not (popII)
        J21LW_interp : bool
            Include contribution from LW feedback (popIII) or not (popII)

        Returns
        ----------
        integrand : array 
            Integrand to be used in the main class
        """

        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(massVector), z)))
        SFRtab_curr = self.SFR(CosmoParams, AstroParams, HMFinterp, massVector, z, pop, vCB, J21LW_interp)
        integrand = HMF_curr * SFRtab_curr * massVector

        return integrand
    

    def J_LW_21(self, CosmoParams, AstroParams, sfrdIter, z, pop):
        """
        Mean background specific intensity, units of erg/s/cm^2/Hz/sr

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        sfrdIter : array
            Star formation rate density
        z : float
            Redshift
        pop : int
            Which population (2 for popII or 3 for popIII)

        Returns
        ----------
        JW : array 
            LW specific intensity
        """

        Elw = (constants.Elw_eV * u.eV).to(u.erg).value 
        
        # photons produced per baryon 
        if pop == 3:
            Nlw = AstroParams.N_LW_III
        elif pop == 2:
            Nlw = AstroParams.N_LW_II

        zIntMatrix = np.linspace(z, constants.redshiftFactor_Visbal*(1+z)-1, 20) # LW horizon from Visbal et al 2014
               
        sfrdIterMatrix_LW = interpolate.interp1d(z, sfrdIter, kind = 'linear', bounds_error=False, fill_value=0)(zIntMatrix)
    
        integrandLW = constants.c_Mpcs / 4 / np.pi # for units to work, c must be in Mpc/s and proton mass in solar masses
        integrandLW *= (1+z)**2 / cosmology.Hubinvyr(CosmoParams,zIntMatrix)
        integrandLW *= Nlw * Elw / constants.mprotoninMsun / constants.deltaNulw # specific emissivity
        integrandLW = integrandLW * sfrdIterMatrix_LW * (1 /u.Mpc**2).to(1/u.cm**2).value # convert from 1/Mpc^2 to 1/cm^2 

        JLW = 1e21 *np.trapezoid(integrandLW, x = zIntMatrix, axis = 0) # convert from cgs untis to units commonly used  

        return JLW


    def J_LW_Discrete(self, CosmoParams, AstroParams, z, pop, rGreater, SFRD_interp_input):
        """
        Radial kernel of the LW specific intensity before R integration, units of erg/s/cm^2/Hz/sr

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        z : float or array
            Redshift
        pop : int
            Which population (2 for popII or 3 for popIII)
        rGreater : matrix 
            Radii
        SFRD_interp_input : interpolator
            Interpolator for the star formation rate density in redshift

        Returns
        ----------
        RK : array 
            Radial kernel of the LW specific intensity
        """
        
        Elw = (constants.Elw_eV * u.eV).to(u.erg).value
        
        rTable = np.transpose([CosmoParams.chiofzint(z)]) + rGreater # while we compute the intensity at z, the source of the LW field is at redshift z' corresponding to a shell located R away from the comoving redshft associated with the source
        rTable[rTable > CosmoParams.chiofzint(constants.zmax_AstroBreak)] = CosmoParams.chiofzint(constants.zmax_AstroBreak) #c ut down so that nothing exceeds zmax where we do not trust the astrophysical model 
        zTable = CosmoParams.zfofRint(rTable)
            
        zMax = np.transpose([constants.redshiftFactor_Visbal*(1+z)-1])
        rMax = CosmoParams.chiofzint(zMax)
        
        c1 = (1+z)**2/4/np.pi
        
        if pop == 3:
            Nlw = AstroParams.N_LW_III

        elif pop == 2:
            Nlw = AstroParams.N_LW_II
        
        c2r = SFRD_interp_input(zTable)
                
        c2r *= Nlw * Elw / constants.deltaNulw / constants.mprotoninMsun * 0.5*(1 - np.tanh((rTable - rMax)/10)) * (1 /u.yr/u.Mpc**2).to(1/u.s/u.cm**2).value # smooth tanh cutoff, smoother function within 2-3% agreement with J_LW()
        
        RK = np.transpose([c1]), c2r

        return RK 


    def dSFRDIII_dJ(self,CosmoParams, AstroParams, HMFinterp, z, vCB, J21LW_interp):
        """
        Response of the popIII star formation rate density to the LW background

        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        HMFinterp : HMFinterp class
        z : float
            Redshift
        vCB : bool
            Include contribution from baryon-CDM relative velocity (popIII) or not (popII)
        J21LW_interp : bool
            Include contribution from LW feedback (popIII) or not (popII)

        Returns
        ----------
        integral : array 
            Response integrated over the mass array
        """

        Mh = HMFinterp.Mhtab
        HMF_curr = np.exp(HMFinterp.logHMFint((np.log(Mh), z)))

        SFRtab_currIII = self.SFR(CosmoParams, AstroParams, HMFinterp, HMFinterp.Mhtab, z, pop=3, vCB = vCB, J21LW_interp=J21LW_interp)

        integrand_III = HMF_curr * SFRtab_currIII * HMFinterp.Mhtab
        integrand_III *= AstroParams.A_LW * AstroParams.beta_LW * J21LW_interp(z)**(AstroParams.beta_LW - 1)
        integrand_III *= -1 * self.Mmol_vcb(CosmoParams, AstroParams, z, CosmoParams.vcb_avg)/ HMFinterp.Mhtab

        integral = np.trapezoid(integrand_III, HMFinterp.logtabMh)

        return integral 


    def fesc_II(self,AstroParams, Mh):
        """
        Escape fraction of ionizing photons in halos hosting popII stars

        Parameters 
        ----------
        AstroParams : AstroParams class
        Mh : array
            Halo masses
        
        Returns
        ----------
        fesc : array 
            Escape fraction 
        """

        fesc = np.fmin(1.0, AstroParams.fesc10 * pow(Mh/1e10,AstroParams.alphaesc) )

        return fesc 
    

    def fesc_III(self,AstroParams, Mh):
        """
        Escape fraction of ionizing photons in halos hosting popIII stars

        Parameters 
        ----------
        AstroParams : AstroParams class
        Mh : array
            Halo masses
        
        Returns
        ----------
        fesc : array 
            Escape fraction 
        """

        fesc = np.fmin(1.0, AstroParams.fesc7_III * pow(Mh/1e7,AstroParams.alphaesc_III) )

        return fesc
    

    def compute_sigmaR_nu(self, CosmoParams, HMFinterp, z_array, R_array, Mh_array, dorv_array, dorv):
        """
        Compute the local mass function conditioned over the environment
        
        Parameters 
        ----------
        CosmoParams : CosmoParams class
        HMFinterp : HMFinterp class
        z_array : array 
            Redshifts
        R_array : array 
            Shell radii
        Mh_array : array
            Halo masses
        dorv_array : array
            Input values of either the denisty or velocity field
        dorv : str
            Compute the output wrt the density field ("delta") or the velocity field ("vel")
        
        Returns
        ----------
        HMF_corr : array 
            Local HMF in Eulerian space
        mArray : array 
            Halo masses, dimension (z,R,Mh,delta or v)
        zGreaterArray : array
            Redshifts of the sources, dimension (z,R,Mh,delta or v)
        out : array 
           Either delta_R (if dorv == delta) or velocity, dimension (z,R,Mh,delta or v)
        """

        zArray, rArray, mArray, dorvNormArray = np.meshgrid(z_array, R_array, Mh_array, dorv_array, indexing = 'ij', sparse = True) # reshape

        rGreaterArray = np.zeros_like(zArray) + rArray
        rGreaterArray[CosmoParams.chiofzint(zArray) + rArray >= CosmoParams.chiofzint(constants.zmax_AstroBreak)] = np.nan
        zGreaterArray = CosmoParams.zfofRint(CosmoParams.chiofzint(zArray) + rGreaterArray) # redshift of the source

        whereNotNans = np.invert(np.isnan(rGreaterArray))

        sigmaR = np.zeros((len(z_array), len(R_array), 1, 1))
        sigmaR[whereNotNans] = HMFinterp.sigmaRintlog((np.log(rGreaterArray)[whereNotNans], zGreaterArray[whereNotNans])) # mass field variance on R (environment scale)

        sigmaM = HMFinterp.sigmaintlog((np.log(mArray), zGreaterArray)) # mass field variance on Mh

        modSigmaSq = sigmaM**2 - sigmaR**2
        indexTooBig = (modSigmaSq <= 0.0)
        modSigmaSq[indexTooBig] = np.inf #if sigmaR > sigmaM the halo does not fit in the radius R. Cut the sum
        modSigma = np.sqrt(modSigmaSq)

        # variables of the EPS theory 
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
            # EPS_HMF corrected with (1+delta) for Eulerian space
            HMF_corr = (nu/nu0) * (sigmaM/modSigma)**2.0 * np.exp(-CosmoParams.a_corr_EPS * (nu**2-nu0**2)/2.0 ) * (1.0 + deltaArray)

        else: 
            # as 21cmFAST, use PS HMF, integrate and normalize at the end
            HMF_corr = cosmology.PS_HMF_unnorm(CosmoParams, Mh_array.reshape(len(Mh_array),1),nu,dlogSdMcurr) * (1.0 + deltaArray)

        if dorv == "delta":
            out = deltaArray
        elif dorv == "vel":
            out = dorvNormArray

        return HMF_corr, mArray, zGreaterArray, out
    

    def compute_gamma(self, CosmoParams, AstroParams, HMFinterp, z_array, R_array, Mh_array, input_sigmaofRtab, fesctab_II):
        """
        Compute linear and quadratic gamma exponents for the SFRD-delta (popII+popIII) and niondot-delta (onlypopII) lognormal approximations
        
        Parameters 
        ----------
        CosmoParams : CosmoParams class
        AstroParams : AstroParams class
        HMFinterp : HMFinterp class
        z_array : array 
            Redshifts
        R_array : array r
            Shell radii
        Mh_array : array
            Halo masses
        input_sigmaofRtab : array
            Variance of the matter field smoothed over R
        fesctab_II : 
            Escape fraction for popII stars      
                          
        Returns
        ----------
        Empty
        """

        Nsigmad = 1.0 # how many sigmas we explore 
        Nds = 3 # how many deltas

        deltatab_norm = np.linspace(-Nsigmad,Nsigmad,Nds)
        
        HMF_corr, mArray, zGreaterArray, deltaArray = self.compute_sigmaR_nu(CosmoParams, HMFinterp, z_array, R_array, Mh_array, deltatab_norm, "delta") # compute local HMF 

        # PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
        if not CosmoParams.Flag_emulate_21cmfast:

            # Normalized PS(d)/<PS(d)> at each mass
            integrand_II = HMF_corr * self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=2)
            if AstroParams.USE_POPIII:
                integrand_III = HMF_corr * self.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3, vCB=CosmoParams.vcb_avg,J21LW_interp=self.J21LW_interp_conv_avg)
            
        else: 
            # 21cmFAST uses PS HMF, integrates and normalizes as SFRD(d)/<SFRD(d)>
            integrand_II = HMF_corr * self.SFR(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=2) * mArray
            if AstroParams.USE_POPIII:
                integrand_III = HMF_corr * self.SFR(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3,  vCB=CosmoParams.vcb_avg,J21LW_interp=self.J21LW_interp_conv_avg) * mArray

        # Local popII SFRD
        SFRD_II_dR = np.trapezoid(integrand_II, HMFinterp.logtabMh, axis = 2)

        # Local popII niondot
        niondot_II_dR = np.trapezoid(integrand_II*fesctab_II[None, None, :, None], HMFinterp.logtabMh, axis = 2)

        if AstroParams.USE_POPIII:
            # Local popIII SFRD            
            SFRD_III_dR = np.trapezoid(integrand_III, HMFinterp.logtabMh, axis = 2)
        else:
            SFRD_III_dR = np.zeros_like(SFRD_II_dR)

        # compute all required gammas 
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

        # LW correction to Pop III gammas
        if AstroParams.USE_POPIII and AstroParams.USE_LW_FEEDBACK:

            # get the zero-lag correlation function (zero distance separation)
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
            self.deltaGamma_R_z = np.transpose(   np.sum(self.deltaGamma_R_Matrix, axis = 2) / np.transpose([np.diagonal(xi_RR_CF_zerolag[:,:])])    )
            self.deltaGamma_R_z[ self.gamma_III_index2D == 0 ] = 0 #don't correct gammas if gammas are zero
            self.gamma_III_index2D += self.deltaGamma_R_z #correct Pop III gammas with LW correction factor

        # Non-Linear Correction Factors to convert from Lagrangian to Eulerian space and to normalize the integral of the SFRD (see sec 3A in 2507.15922)
        gamma_II_index2D_Lag = self.gamma_II_index2D - 1.
        gamma_III_Lagrangian = self.gamma_III_index2D - 1.
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
        self._corrfactorEulerian_III=_corrfactorEulerian_III.T
        self._corrfactorEulerian_III[0:CosmoParams.indexminNL] = self._corrfactorEulerian_III[CosmoParams.indexminNL] #for R<R_NL we just fix it to the RNL value, as we do for the correlation function. We could cut the sum but this keeps those scales albeit approximately
                
        return 1


    def compute_numerical_der_gamma(self, arr1, arr2, order):
        """
        Compute first and second numerical derivatives of arr1 wrt arr2

        Parameters 
        ----------
        arr1 : array
            Quantity to be derived 
        arr2 : array
            Quantity to perform the numerical derivative
        order : int 
            Compute either first (1) or second (2) order derivative

        Returns
        ----------
        darr1_darr2 : array 
            Numerical derivative
        """

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
    """
    Pre-compute velocity–dependent Pop III star formation suppression (see sec 4A in arXiv:2407.18294)

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
    vcb_expFitParams : float 
        Coefficients to approximate SFRD(vCB) / SFRD

    """
    def __init__(self, UserParams, CosmoParams, AstroParams, HMFinterp, z_Init = None, SFRD_Init = None):

        # if z_Init and SFRD_Init are not provided, we initialize them here. This allows us to avoid redundant computations if they were already initialized in the parent class and passed as arguments.
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

                HMF_corr, mArray, zGreaterArray, velArray  = SFRD_Init.compute_sigmaR_nu(CosmoParams, HMFinterp, z_Init.zintegral, CosmoParams._Rtabsmoo, HMFinterp.Mhtab, vAvg_array, 'vel') # local HMF 
                
                # PS_HMF~ delta/sigma^3 *exp(-delta^2/2sigma^2) * consts(of M including dsigma^2/dm)
                if not CosmoParams.Flag_emulate_21cmfast:
                    # Normalized PS(d)/<PS(d)> at each mass
                    integrand_III = HMF_corr * SFRD_Init.SFRD_integrand(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3, vCB=velArray, J21LW_interp=SFRD_Init.J21LW_interp_conv_avg)
                    
                else: 
                    # 21cmFAST uses PS HMF, integrates and normalizes as SFRD(d)/<SFRD(d)>
                    integrand_III = HMF_corr * SFRD_Init.SFR(CosmoParams, AstroParams, HMFinterp, mArray, zGreaterArray, pop=3, vCB=velArray, J21LW_interp=SFRD_Init.J21LW_interp_conv_avg) * mArray

                SFRD_III_dR_V = np.trapezoid(integrand_III, HMFinterp.logtabMh, axis = 2) # local SFRD corrected by vCB

                SFRDIII_Ratio = SFRD_III_dR_V / SFRD_III_dR_V[:,:,len(vAvg_array)//2].reshape((len(z_Init.zintegral), len(CosmoParams._Rtabsmoo), 1))
                SFRDIII_Ratio[np.isnan(SFRDIII_Ratio)] = 0.0

                # temporarily turning off divide warnings; will turn them on again after exponential fitting routine
                divideErr = np.seterr(divide = 'ignore')
                divideErr2 = np.seterr(invalid = 'ignore')
                
                ### TODO: The next few lines fits for rho(z, v) / rhoavg = Ae^-b tilde(eta) + Ce^-d tilde(eta).
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
                
