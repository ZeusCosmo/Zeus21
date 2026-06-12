"""
SEDs and Green's functions for first-galaxy emission models.

Two families of functions:

  X-ray / Lyman-alpha SEDs (used in 21cm calculations)
  ---------------------------------------------------------
  SED_XRAY      – power-law X-ray SED, normalized so ∫ E·SED(E) dE = 1
                  over [E0_xray, Emax_xray]. Returns photon number spectrum. 
                  E*SED is the power-law with index alpha_xray, so the output is divided by 1/E at the end to return number). 
  SED_LyA       – Lyman-alpha continuum SED, normalized so ∫ SED(ν) dν = 1 (as opposed as E*SED, what was for Xrays).
                  over [νLyA, νLyCont]. Returns number per unit frequency.

  Green's functions (used in UVLFs, Hα/UV ratios, etc.)
  ---------------------------------------------------------
  Greens_function_LUV        – UV luminosity per unit SFR as a function of
                               stellar population age. Integrate against SFR(t)
                               to get instantaneous L_UV.
  Greens_function_LUV_Short  – Same, windowed to ages < t_cut_LUV_short.
  Greens_function_LUV_Long   – Same, windowed to ages > t_cut_LUV_short.
  Greens_function_LHa        – Hα luminosity Green's function, analogous to LUV.

Supported SED stellar-population models (AstroParams.SEDMODEL):
  'bagpipes', 'BPASS', 'BPASS_binaries'

Population flags (pop):
  2 → Pop II stars
  3 → Pop III stars
"""


import numpy as np 
from . import constants




def SED_XRAY(AstroParams, En, pop = 0): #pop set to zero as default, but it must be set to either 2 or 3
    """
    X-ray SED for Pop II or Pop III sources.

    Normalized so that ∫_{E0}^{Emax} E · SED(E) dE = 1, i.e. E·SED is a
    power law with index alpha_xray. The function returns the *photon number*
    spectrum (divided by E at the end).

    The high-energy cutoff is intentionally omitted because photons redshift
    down into the <2 keV observing band.

    Parameters
    ----------
    AstroParams : object
        Must expose: alpha_xray, alpha_xray_III, E0_xray, Emax_xray_norm.
    En : float or array-like
        Photon energy in eV.
    pop : {2, 3}
        Stellar population. 2 = Pop II, 3 = Pop III.

    Returns
    -------
    ndarray
        SED in units of eV⁻¹, same shape as En.
        Zero below E0_xray.
    """
    if pop == 2:
        alphaX = AstroParams.alpha_xray
    elif pop == 3:
        alphaX = AstroParams.alpha_xray_III
    else:
        print("Must set pop to either 2 or 3!")
        
    if np.abs(alphaX + 1.0) < 0.01: #log
        norm = 1.0/np.log(AstroParams.Emax_xray_norm/AstroParams.E0_xray) / AstroParams.E0_xray
    else:
        norm = (1.0 + alphaX)/((AstroParams.Emax_xray_norm/AstroParams.E0_xray)**(1 + alphaX) - 1.0) / AstroParams.E0_xray

    return np.power(En/AstroParams.E0_xray, alphaX)/En * norm * np.heaviside(En - AstroParams.E0_xray, 0.5)
    #do not cut at higher energies since they redshift into <2 keV band



def SED_LyA(nu_in, pop = 0): #default pop set to zero so python doesn't complain, but must be 2 or 3 for this to work
    """
    Lyman-alpha continuum SED for Pop II or Pop III sources.

    A two-segment power law in frequency, joined at ν_LyB:
      • νLyA  ≤ ν < νLyB  : index indexbelow  (flatter)
      • νLyB  ≤ ν < νLyCont: index indexabove  (steeper)

    Normalized so that ∫_{νLyA}^{νLyCont} SED(ν) dν = 1 (photon number
    per unit frequency). Contrast with SED_XRAY, which normalizes E·SED.

    Parameters
    ----------
    nu_in : float or array-like
        Frequency in the same units as constants.freqLyA / freqLyCont.
    pop : {2, 3}
        Stellar population. Uses BL05 stellar spectra as reference.
        Pop II: amps = [0.68, 0.32], Pop III: amps = [0.56, 0.44].

    Returns
    -------
    ndarray
        SED value(s), same shape as nu_in. Zero outside [νLyA, νLyCont).
    """

    nucut = constants.freqLyB #above and below this freq different power laws
    if pop == 2:
        amps = np.array([0.68,0.32]) #Approx following the stellar spectra of BL05. Normalized to unity
        indexbelow = 0.14 #if one of them zero worry about normalization
        normbelow = (1.0 + indexbelow)/(1.0 - (constants.freqLyA/nucut)**(1 + indexbelow)) * amps[0]
        indexabove = -8.0
        normabove = (1.0 + indexabove)/((constants.freqLyCont/nucut)**(1 + indexabove) - 1.0) * amps[1]
    elif pop == 3:
        amps = np.array([0.56,0.44]) #Approx following the stellar spectra of BL05. Normalized to unity
        indexbelow = 1.29 #if one of them zero worry about normalization
        normbelow = (1.0 + indexbelow)/(1.0 - (constants.freqLyA/nucut)**(1 + indexbelow)) * amps[0]
        indexabove = 0.2
        normabove = (1.0 + indexabove)/((constants.freqLyCont/nucut)**(1 + indexabove) - 1.0) * amps[1]
    else:
        print("Must set pop to 2 or 3!")
        
    nulist = np.asarray([nu_in]) if np.isscalar(nu_in) else np.asarray(nu_in)

    result = np.zeros_like(nulist)
    for inu, currnu in enumerate(nulist):
        if (currnu<constants.freqLyA or currnu>=constants.freqLyCont):
            result[inu] = 0.0
        elif (currnu < nucut): #between LyA and LyB
            result[inu] = normbelow * (currnu/nucut)**indexbelow
        elif (currnu >= nucut):  #between LyB and Continuum
            result[inu] = normabove * (currnu/nucut)**indexabove
        else:
            print("Error in SED_LyA, whats the frequency Kenneth?")


    return result/nucut #extra 1/nucut because dnu, normalizes the integral




def Greens_function_LUV(AstroParams, ageMyrin, Mhalos):
"""
    UV luminosity Green's function for a 1 M☉/yr instantaneous burst at some time t.

    Convolve with SFR(t) to get L_UV(t):
        L_UV(t) = ∫ G_UV(t - t') · SFR(t') dt'

    The shape is a double-power-law in age (fast rise, slow decline) with a
    Gaussian bump at very young ages (~2–4 Myr) capturing the brief Wolf-Rayet
    and OB-supergiant phase. A Gaussian exponential cutoff suppresses
    contributions beyond ~650–1100 Myr depending on the SED model.

    Parameters
    ----------
    AstroParams : object
        Must choose SEDMODEL ∈ {'bagpipes', 'BPASS', 'BPASS_binaries'}.
    ageMyrin : float or array-like, shape (Nt,)
        Stellar population age in Myr. A small offset (1e-4 Myr) is added
        internally to avoid division by zero at age = 0.
    Mhalos : array-like, shape (NMh,)
        Halo masses in M☉. Currently enter only through IMFZcorrection,
        which is unity for UV (correction absorbed into ε*).

    Returns
    -------
    ndarray, shape (NMh, Nt)
        Green's function in erg s⁻¹ M☉⁻¹.
    """

    if AstroParams.SEDMODEL == 'bagpipes':
        _amp = 3.1e36
        _agepivot = 4 #Myr
        _agepivot2 = 650 #Myr
        _agebump, _widthbump, Ampbump = 3.4, 0.1, 0.33 #Myr, log10width, relative amplitude
        _alpha, _beta = 1.4, -0.3
    elif AstroParams.SEDMODEL == 'BPASS': #BPASS single stars 
        _agepivot = 4.2 #Myr
        _agepivot2 = 1100 #Myr
        _agebump, _widthbump, Ampbump = 2.2, 0.2, 0.7 #Myr, log10width, relative amplitude
        _alpha, _beta = 1.2, 0.0
        _amp = 1.8e36
    elif AstroParams.SEDMODEL =='BPASS_binaries': #BPASS with binarity fraction built in (default). Pretty similar in UV
        _agepivot = 4.0 #Myr
        _agepivot2 = 1100 #Myr
        _agebump, _widthbump, Ampbump = 2.2, 0.2, 0.6 #Myr, log10width, relative amplitude
        _alpha, _beta = 1.2, -0.2
        _amp = 2.2e36

    ageMyr = ageMyrin+1e-4 #to avoid complaints about division by zero
    IMFZcorrection = np.ones_like(Mhalos) #no correction on UV, absorbed by eps*
    massindepresult = _amp*( Ampbump*np.exp(-(np.log10(ageMyr)-np.log10(_agebump))**2/2/_widthbump**2) + 1/((ageMyr/_agepivot)**_alpha+(ageMyr/_agepivot)**(_beta))* np.exp(-(ageMyr/_agepivot2)**2) ) #erg/s/Msun
    return np.outer(IMFZcorrection,massindepresult) #erg/s/Msun, Nt x NMh

def Selection_Timescales_LUV(times, time1, time2):
    "Returns the selection function from t1 to t2, to keep t1 < t < t2 smoothly"
    _tanhwidth = 0.2
    return (1 + np.tanh( np.log(times/time1)/_tanhwidth))/2. * (1 + np.tanh( np.log(time2/times)/_tanhwidth))/2.

def Greens_function_LUV_Short(AstroParams,time, mass):
    "Age in Myr, window in erg/s/Msun for the short timescale LUV window"
    return Greens_function_LUV(AstroParams, time, mass) * Selection_Timescales_LUV(time+1e-10, 0.0, AstroParams._tcut_LUV_short)[None,:] #+1e-10 to avoid division by zero in selectionLUV

def Greens_function_LUV_Long(AstroParams,time, mass):
    "Age in Myr, window in erg/s/Msun for the long timescale LUV window"
    return Greens_function_LUV(AstroParams, time, mass) * Selection_Timescales_LUV(time+1e-10, AstroParams._tcut_LUV_short, 3000)[None,:]


def Greens_function_LHa(AstroParams, ageMyrin, Mhalos):
"""
    Hα luminosity Green's function for a 1 M☉/yr instantaneous burst at some past time t.

    Analogous to Greens_function_LUV but for the Hα recombination line.
    Hα traces ionizing photons and therefore falls off much faster with age
    (~few Myr vs. ~Gyr for UV). For BPASS_binaries there is a second component at ~20 Myr to account
    for delayed ionizing flux.

    The IMF/metallicity correction scales with halo mass as a power law,
    clamped to [0.1, 10] to prevent runaway corrections.


    Parameters
    ----------
    AstroParams : object
        Must expose: SEDMODEL, normLHa_ZIMF, alphanormLHa_ZIMF.
    ageMyrin : float or array-like, shape (Nt,)
        Stellar population age in Myr.
    Mhalos : array-like, shape (NMh,)
        Halo masses in M☉. Used in the IMF/Z mass-dependent correction.

    Returns
    -------
    ndarray, shape (NMh, Nt)
        Green's function in erg s⁻¹ M☉⁻¹.
    """
    if AstroParams.SEDMODEL == 'bagpipes':
        _amp = 1.2e35
        _exp = 2.0
        _agepivot = 4.4 #Myr
        _alpha = 0.33
    elif AstroParams.SEDMODEL == 'BPASS':
        _amp = 3.4e35
        _exp = 0.9
        _agepivot = 1.2 #Myr
        _alpha = 0.5
    elif AstroParams.SEDMODEL == 'BPASS_binaries':
        _amp = 3.4e35
        _exp = 0.78
        _agepivot = 1.2 #Myr
        _alpha = 0.5
    else:
        raise ValueError("SEDMODEL must be 'bagpipes', 'BPASS' or 'BPASS_binaries'")
    ageMyr = ageMyrin+1e-4 #to avoid complaints about division by zero
    IMFZcorrection =  AstroParams.normLHa_ZIMF * (Mhalos/1e10)**AstroParams.alphanormLHa_ZIMF
    IMFZcorrection = np.fmin(np.fmax(IMFZcorrection, 0.1),10.) #make sure it's not too low or high
    massindepresult = _amp * np.exp(-(ageMyr/_agepivot)**_exp)*(ageMyr/_agepivot)**_alpha #erg/s/Msun
    if AstroParams.SEDMODEL == 'BPASS_binaries':
        _amp2 = 8e32
        _agepivot2 = 20 #Myr
        massindepresult += _amp2 * np.exp(-(ageMyr/_agepivot2)) #extra component due to binaries
    return np.outer(IMFZcorrection,massindepresult) #erg/s/Msun, Nt x NMh, so we can multiply by SFR to get LHa
