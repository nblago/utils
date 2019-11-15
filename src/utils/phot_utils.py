#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:44:20 2019

@author: nadiablago
"""
import pysynphot as ps
import numpy as np
import os


if not 'PYSYN_CDBS' in os.environ.keys():
    print ("Adding the Pysynphot environment:")
    try:
        os.environ['PYSYN_CDBS'] = "PATH_TO_YOUR_PYSYNPHOT"
    except KeyError:
        os.environ.putenv('PYSYN_CDBS', "PATH_TO_YOUR_PYSYNPHOT")
        
 
    
banddic = {"J": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_j_004_syn.fits"),
           "H": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_h_004_syn.fits"),
           "K": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_k_004_syn.fits"),
           "3.6": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac1_3.6.dat"),
           "4.5": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac2_4.5.dat"),
           "5.8": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac3_5.8.dat"),               
           "8.0": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac4_8.0.dat"),
           "swift,uvw2": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_uvw2_uvot.dat"),
           "swift,uvm2": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_uvm2_uvot.dat"),
           "swift,uvw1": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_uvw1_uvot.dat"),
           "swift,u": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_u_uvot.dat"),
           "swift,b": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_b_uvot.dat"),
           "swift,v": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_v_uvot.dat"),
           "paranal,Y":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_VISTA.Y.dat"),
           "paranal,Z":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_VISTA.Z.dat"),
           "paranal,J":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_VISTA.J.dat"),
           "paranal,H":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_VISTA.H.dat"),
           "paranal,Ks":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_VISTA.Ks.dat"),
           "omegacam,u":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.u_SDSS.dat"),
           "omegacam,g":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.g_SDSS.dat"),
           "omegacam,r":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.r_SDSS.dat"),
           "omegacam,i":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.i_SDSS.dat"),
           "omegacam,z":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.z_SDSS.dat"),                    
           "omegacam,Halpha":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.Halpha.dat"),
           "gaia,G":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/gaia_G.txt"),
           "gaia,RP":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/gaia_RP.txt"),
           "gaia,BP":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/gaia_BP.txt"),
           "gaia,RVS":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/gaia_RVS.txt")

           }


def get_synthetic_mag_spec(wave, flux, filt="sdss,g", system="abmag"):
    '''
    Provided a spectrum for a source, it computes its synthetic magnitude in the required
    band and photometric system.
    
    Parameters
    ----------
    wave : 1-D array containing the wavelength of the spectrum (in Angstrom).
    flux: 1-D array containing the flux (in erg/s/cm**2/A)
    filt: photometric band the synthetic mangitude needs to be computed.
    system: photometric system for the output mangitude (agmag, vegamag, stmag...)
    
    Returns
    -------
    mag : float
        The synthetic magnitude of the input spectrum for that photometric band and system.
    
    
    Examples
    --------
    >>> import pysynphot as ps
    >>> vega = ps.Vega
    >>> get_synthetic_phot(vega.wave, vega.flux, filt="V", system="vegamag")
    >>> 0.0
    '''
    
    sp = ps.ArraySpectrum(wave, flux, fluxunits="flam")
    
    #If the band is a custom one that we have added to the dictionary of filters,
    # we just create a new passband from the file. Otherwise, it selects it automatically.
    if (filt in banddic.keys()): 
        filt_file = banddic.get(filt, filt)
        bp = ps.FileBandpass(filt_file)
    else:
        bp = ps.ObsBandpass(filt)

    obs = ps.Observation(sp, bp, force='extrap', binset=wave)
    mag = obs.effstim(system)
    return mag 
    
def get_synthetic_mag_bb(T, mag_in, filt_in="sdss,r", system_in="vegamag", filt_out="sdss,g", system_out="vegamag"):
    '''
    Provided a black body temperature, it normalizes the emission to one input magnitude
    in a given photometric system, and produces another synthetic magnitude in the output
    band and photometric system.
    
    Parameters
    ----------
    T : float
        Black body temperature
    mag_in: float
        magnitude that the black body spectrum will be normalized to
    filt_in: string
        Name of the band where the input magnitude is in.
    system_in: photometric system for the input mangitude (agmag, vegamag, stmag...)
    filt_out: string
        Name of the band where the out magnitude will be computed.
    system_out: photometric system for the output mangitude (agmag, vegamag, stmag...)
    
    Returns
    -------
    mag : float
        The synthetic magnitude of the black body spectrum for that photometric band and system.
    
    
    Examples
    --------
    >>> phot_utils.get_filter_mag_bb(6000, 19, "V", "vegamag", "sdss,g", "abmag")
    >>> 19.311307758370692
    '''
    
    sp = ps.BlackBody(T)
    sp.convert('flam')
        
    if (filt_in in banddic.keys()):
        bp_in = ps.FileBandpass(banddic[filt_out])
    else:
        bp_in = ps.ObsBandpass(filt_out)
        
    if (filt_out in banddic.keys()):
        bp_out = ps.FileBandpass(banddic[filt_out])
    else:
        bp_out = ps.ObsBandpass(filt_out)
          
    sp_norm = sp.renorm(mag_in, system_in, bp_in)

    obs = ps.Observation(sp_norm, bp_out, binset=sp_norm.wave)
    mag = obs.effstim(system_out)
    return mag 

    
def get_synthetic_mag_file(filename, filt_out, system_out, filt_in=None, mag_in=None, system_in=None, ebv=0, force=None):
    '''
    Scales the input spectrum (read from the filename file) to the magnitude 
    provided in the filter_in band, and obtains the synthetic photometry in the filter_out band.
    If no scaling magnitude is provided, the original spectrum is used to compute the synthetic mangitude.
    
    Parameters
    ----------
    filename : string
            The name of the file which contains the spectrum.
    filt_out : string
            The name of the filter the system will compute the output of.
    system_out : string
            The system in which the output magntiudes are measured. "abmag" or "vegamag" or "stmag"
    filt_in : string
            The name of the filter we want to scale the spectrum.
    mag_in : float
            The mangtude of the object in the filter_in band.
    system_in : string
            The system in which the input magntiude is measured. "abmag" or "vegamag" or "stmag"
    ebv : float, default: 0
            The reddening to be applied to the model spectrum.    
    force : string, default: None
            Whether the magnitude should be extrapolated in case the filter and the spectrum wavelengths do not
            completely overlap. You may use force=[extrap|taper] to force this Observation anyway.
            
    Returns
    -------
    
    mag_out : float
                The synthetic photometry magnitude of the input spectrum once 
                scaled to its mag_in value in filter_out band.
    '''
    sp = ps.FileSpectrum(filename)
    sp_ext = sp * ps.Extinction(ebv, 'mwavg')
    
    if not( filt_in is None or mag_in is None or system_in is None):
        sp_norm = sp_ext.renorm(mag_in, system_in, ps.ObsBandpass(filt_in))
    else:
        sp_norm = sp_ext
        
    try:
        obs = ps.Observation(sp_norm, ps.ObsBandpass(filt_out), binset=sp_norm.wave, force=force)
    except ValueError:
        bp = ps.FileBandpass(banddic[filt_out])
        obs = ps.Observation(sp_norm, bp, binset=sp.wave)
        
    mag_out = obs.effstim(system_out)
    
    return mag_out


def get_model_spectra(teff, mh, logg, model='ck04models', L=1, ebv=0):
    '''
    
    For example: Temperature Teff=10000K, [M/H] = +0.1 and gravity log=3.0.
    model is the stellar model: 
    Castelli-Kurucz - ck04models
    Kurucz 1993 Atlas - k93models
        
    Parameters
    ----------
    teff : float
            The effective temperature of the star (in K).
    mh : string
            The metallicity as [M/H].
    logg : float
            The log g of the stellar model.
    model : string
            Gride stellar models that can be used to retrieve the models for stellar spectra.
            See: https://pysynphot.readthedocs.io/en/latest/appendixa.html
            for a description.
            
            Castelli-Kurucz - ck04models
            Kurucz 1993 Atlas - k93models
            Phoenix (F. Allard et al.) - phoenix
    L : float
            The bolometric luminosity of the star (in Lsun). It will be used to normalize the spectrum.
    ebv : float, default: 0
            The reddening to be applied to the model spectrum.    
            
    Returns
    -------
    s: pysynphot ArraySpectrum 
        The spectrum corresponding to the stellar model, scaled to the luminosity.
    
    Examples
    --------
    Temperature Teff=10000K, [M/H] = +0.1 and gravity log=3.0 for a star of 1 Lsun
    and no extinction.
    
    s = get_model_spectra(10000, 0.1, 3.0, model='ck04models', L=1, ebv=0)
    
    '''
    
    sp = ps.Icat(model, teff, mh, logg)

    if ebv != 0:
        sp = sp * ps.Extinction(ebv, 'lmcavg')
    
    F_model = sp.trapezoidIntegration(sp.wave, sp.flux)*(u.erg/u.s/u.cm**2) #In erg / s / cm**2
    Lmodel = (4 * np.pi * ((10*u.pc).to(u.cm))**2 * F_model ).to(u.Lsun)
    norm = L/Lmodel

    newflux = sp.flux * norm
    
    s = ps.ArraySpectrum(sp.wave, newflux, fluxunits="flam")
    
    return s

def get_synthetic_mag_model(teff, mh, logg, filt, system="abmag", model='ck04models', L=1, ebv=0):
    '''
    
    Returns a synthetic mangitude for a stellar spectral model from one of the grid models,
    which is scaled to the luminosity and added the extinction specified 
    as a paramter. 
        
    Parameters
    ----------
    teff : float
            The effective temperature of the star (in K).
    mh : string
            The metallicity as [M/H].
    logg : float
            The log g of the stellar model.
    filt:  string
            Name of the filter to compute the synthetic magntiude.
    system:  string
            Photometric system to compute the synthetic magntiude ("abmag", "vegamag", "stmag"...).            
    model : string
            Gride stellar models that can be used to retrieve the models for stellar spectra.
            See: https://pysynphot.readthedocs.io/en/latest/appendixa.html
            for a description.
            
            Castelli-Kurucz - ck04models
            Kurucz 1993 Atlas - k93models
            Phoenix (F. Allard et al.) - phoenix
    L : float
            The bolometric luminosity of the star (in Lsun). It will be used to normalize the spectrum.
    ebv : float, default: 0
            The reddening to be applied to the model spectrum.    
            
    Returns
    -------
    m: float
            Synthetic mangitude of the model spectrum in the filter "filt".
    
    Examples
    --------
    
    Gaia BP magnitude for a star with temperature Teff=10000K, [M/H] = +0.1 and gravity 
    log g =3.0 for a star of 1 Lsun and no extinction.

    >>> get_synthetic_mag_model(9550, -0.5, 3.95, 'gaia,BP', 'vegamag', model='ck04models', L=40, ebv=0)
    >>> 0.9094232520689509
    '''
    
    sp = get_model_spectra(teff, mh, logg, model='ck04models', L=L, ebv=ebv)
    mag = get_synthetic_mag_spec(sp.wave, sp.flux, filt=filt, system=system)
    
    return mag