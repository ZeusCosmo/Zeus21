"""
Helper functions to be used across zeus21

Authors: Yonatan Sklansky, Emilie Thelie
UT Austin - February 2025

"""

import numpy as np
import powerbox as pbox
from pyfftw import empty_aligned as empty
import time
import gc

def powerboxCtoR(pbobject,mapkin = None):
    'Function to convert a complex field to real 3D (eg density, T21...) on the powerbox notation'
    'Takes a powerbox object pbobject, and a map in k space (mapkin), or otherwise assumes its pbobject.delta_k() (tho in that case it should be delta_x() so...'

    realmap = empty((pbobject.N,) * pbobject.dim, dtype='complex128')
    if (mapkin is None):
        realmap[...] = pbobject.delta_k()
    else:
        realmap[...] = mapkin
    realmap[...] = pbobject.V * pbox.dft.ifft(realmap, L=pbobject.boxlength, a=pbobject.fourier_a, b=pbobject.fourier_b)[0]
    realmap = np.real(realmap)

    return realmap

def tophat_smooth(rr, ks, dk):
    x = ks * rr + 1e-5
    win_k = 3/(x**3) * (np.sin(x) - x*np.cos(x))
    deltakfilt = dk * win_k
    return np.real(np.fft.ifftn(deltakfilt))

def find_nearest_idx(array, values):
    array = np.atleast_1d(array)
    values = np.atleast_1d(values)
    idx = []
    for i in range(len(values)):
        idx.append((np.abs(array - values[i])).argmin())
    return np.unique(idx)

def print_timer(start_time, text_before="", text_after=""):
    elapsed_time = time.time() - start_time
    mins = int(elapsed_time//60)
    secs = int(elapsed_time - mins*60)
    print(f"{text_before}{mins}min {secs}s{text_after}")

def v2r(v):
    return (3/4/np.pi * v)**(1/3)
    
def r2v(r):
    return 4/3 * np.pi * r**3

def delete_class_attributes(class_instance): # delete all attributes of the class instance
    for attr in list(class_instance.__dict__):    
        delattr(class_instance, attr)
    gc.collect()