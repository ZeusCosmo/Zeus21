"""

Keep here all global flags, numerical constants, and conversion factors/units.

Author: Julian B. Muñoz
UT Austin and Harvard CfA - January 2023

Edited by Hector Afonso G. Cruz
JHU - July 2024

"""

###############################
###   Units and constants   ###
###############################

MpcToKm = (3.086e+13 * 1e6)
KmToMpc = 1.0/MpcToKm
Mpctocm = MpcToKm*1e5

MsunToKm = 1.48
MsunToMpc = MsunToKm * KmToMpc

Msuntogram = 1.989e33
GramtoGeV = 1 / 5.6095886e23
MsuntoGeV = Msuntogram/GramtoGeV
mH_GeV = 0.93827209
mprotoninMsun = mH_GeV/MsuntoGeV

c_kms = 299792.458
c_Mpcs = c_kms / MpcToKm

yrTos = 3.154e7
Mpctoyr = MpcToKm/c_kms/yrTos


ergToK = 7.24e15
KtoeV = 8.62e-5

#LW related constants
Elw_eV = 11.9 #eV
deltaNulw = 5.8e14 #Hz

#X-ray related constants
normLX_CONST = 1e40 #normalization of Xray luminosity, in units of erg/s/SFR = erg/s/(Msun/yr)
EN_ION_HI = 13.6 #eV
EN_ION_HeI = 24.59 #eV
sigma0norm = 1e-18 #cm^2 normalization of xray cross sections

ZMAX_INTEGRAL = 35.0 #at which z we start the integrals. We take 35. as fiducial since there is not much SFRD. Make sure to test if you have exotic cosmology/astrophysics

#LyA related constants
wavelengthLyC = 91.1753 ##lyman continuum in nm
wavelengthLyA = wavelengthLyC/(1.-1./4.) #lyman alpha, in nm
wavelengthLyB = wavelengthLyC/(1.-1./9.) #lyman beta, in nm

freqLyA = c_kms/wavelengthLyA * 1e12 #lyman alpha, in Hz
freqLyB = c_kms/wavelengthLyB * 1e12 #lyman beta, in Hz
freqLyCont = c_kms/wavelengthLyC * 1e12 #lyman continuum in Hz

n_max_recycle = 22 #we actually have 25 but they matter less and less
fractions_recycle = [1.0, 0.0, 0.2609 ,0.3078 ,0.3259 ,0.3353 ,0.3410 ,0.3448 , 0.3476, 0.3496, 0.3512, 0.3524, 0.3535, 0.3543,0.3550 ,0.3556 ,0.3561 ,0.3565 ,0.3569 ,0.3572 ,0.3575 ,0.3578 ,0.3580 , 0.3582, 0.3584, 0.3586, 0.3587, 0.3589, 0.3590]
#Table 5 in astro-ph/0608032

gcolorfactorHirata = 0.405535 #factor to calculate Tcolor from Tk iteratively from Hirata 2005 (astro-ph/0507102 Eq. (42))
widthLyA=50*1e6 #Hz, from astro-ph/0507102
widthLyAcm = widthLyA/(c_kms*1e5) #cm
Tstar_21 = 0.0682 #T* in K for the 21cm transition
A10_21 = 2.85e-15 #1/s, Einstein 10 coeff for HI


#whether to renormalize the C2 oefficients (appendix in 2302.08506)
#C2_RENORMALIZATION_FLAG = 1 - FLAG_FORCE_LINEAR_CF


RGauss_factor = 0.633 #factor RG = F * RTH between gaussian and Tophat masses = 4^1/3/sqrt(2*np.pi)

#This is only relevant for astromodel = 0
ALPHA_accretion_exponential = 0.79 #index of M(z) ~ e(-a z) for exponential growth. From Schneider+20 Eq (5). For accretion_model = 0
EPSQ_accretion = 0.6 #to calculate EPS accretion. Variance ratio. For EPS, ie, accretion_model = 1


#EoR related
alphaB = 4.31e-13/(1.0 + 0.67) #in cm3 s-1 at 1e4 K as usual
ZMAX_Bubbles = 20. #where to begin bubbles, since at high z it's very noisy (need dcrit = 1.69 exact)
sigmaT = 6.6524e-25 #Thomson cross section in cm^2, for tau_reio
zHeIIreio = 3.0 #when we assume HeII fully reionizes, for tau_reio
FLAG_DO_BUBBLES = False #whether to do xHI fluctuations (TODO. Global xHI is always calculated)


#RSD related
MU_AVG = 0.6 #recovers (1+mu^2)^2 = 1.87, and very close for (1+mu^2) [for cross terms]
MU_LoS = 1.0 #only fully LoS modes


#UVLF related
_MAGMAX_UV = 10. #max abs magnitude to avoid infs
_MAGMIN_Ha = -50. #max abs magnitude to avoid infs
NZ_TOINT = 3 #how many zs around <z> with z_rms we use to predict. Only in HMF since the rest do not vary much.

zeropoint_ABmag_ergsHz = 51.63 # pivot value for specific luminosity (erg/s/Hz) to magnitude conversion -- constant flat in wavelength

# SarahLibanore
zmax_AstroBreak = 50. # max redshift above which we do not trust astro computation

redshiftFactor_Visbal = 1.04 #max amount LW photons can redshift before being scattered, as in Visbal+1402.0882

a_Hirata = 2.98394
b_Hirata = 1.53583
c_Hirata = 3.8528