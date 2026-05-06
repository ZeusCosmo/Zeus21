import numpy as np 
from . import constants


'''
        SED_XRAY
            SED of our Xray sources. Takes energy En in eV.
            Normalized to integrate to 1 from E0_xray to Emax_xray (int dE E * SED(E).
            E*SED is the power-law with index alpha_xray, so the output is divided by 1/E at the end to return number). 
        SED_LyA
            SED of our Lyman-alpha-continuum sources.
            Normalized to integrate to 1 (int d nu SED(nu), so SED is number per units energy (as opposed as E*SED, what was for Xrays).
'''

def SED_XRAY(AstroParams, En, pop = 0): #pop set to zero as default, but it must be set to either 2 or 3
    "SED of our Xray sources, normalized to integrate to 1 from E0_xray to Emax_xray (int dE E * SED(E), and E*SED is the power-law with index alpha_xray, so the output is divided by 1/E at the end to return number). Takes energy En in eV"
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
    "SED of our Lyman-alpha-continuum sources, normalized to integrate to 1 (int d nu SED(nu), so SED is number per units energy (as opposed as E*SED, what was for Xrays) "

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




'''
UV and Halpha Green Functions
'''
def Greens_function_LUV(AstroParams, ageMyrin, Mhalos):
    "Age in Myr, green's function in erg/s/Msun (so LUV = \int dAge Greens_function_LUV(Age) * SFR(Age))"

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
    "Age in Myr, green's function in erg/s/Msun (so LHa = \int dAge Greens_function_LHa(Age) * SFR(Age))"
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
    IMFZcorrection =  self.normLHa_ZIMF * (Mhalos/1e10)**self.alphanormLHa_ZIMF
    IMFZcorrection = np.fmin(np.fmax(IMFZcorrection, 0.1),10.) #make sure it's not too low or high
    massindepresult = _amp * np.exp(-(ageMyr/_agepivot)**_exp)*(ageMyr/_agepivot)**_alpha #erg/s/Msun
    if AstroParams.SEDMODEL == 'BPASS_binaries':
        _amp2 = 8e32
        _agepivot2 = 20 #Myr
        massindepresult += _amp2 * np.exp(-(ageMyr/_agepivot2)) #extra component due to binaries
    return np.outer(IMFZcorrection,massindepresult) #erg/s/Msun, Nt x NMh, so we can multiply by SFR to get LHa