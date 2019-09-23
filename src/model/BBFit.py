# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:57:34 2018

Class that enables to fit a black body function to a set of magntidues.

@author: nadiablago
"""

from __future__ import print_function

import matplotlib
from matplotlib import pylab as plt
import corner
from astropy import units as u
import astropy.constants as cnt
import pysynphot as ps
import os, sys
import numpy as np
import emcee
from scipy import stats
import extinction
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
import warnings

os.environ['PYSYN_CDBS'] = "/scratch/Software/pysynphot_files/cdbs/"
# Add the environment variable which points to the filter response files for the bands we are interested in.
if not 'PYSYN_CDBS' in os.environ.keys():
    print("Adding the Pysynphot environment:")
    os.environ['PYSYN_CDBS'] = "/scratch/Software/pysynphot_files/cdbs/"
print('PYSYN_CDBS environment variable set to: ', os.environ['PYSYN_CDBS'])

class BBFit:            
    
    def __init__(self):
        '''
        Constructor initializes all the parameters to defaults.os.environ['PYSYN_CDBS'] = "/scratch/Software/pysynphot_files/cdbs/"
        '''
        
        #Some predefined constants in the units we need them
        self.c =   cnt.c.to(u.cm/u.s).value #2.99792458e+10 #cm / s
        self.h = cnt.h.to(u.erg * u.s).value #6.62607004e-27 #erg s
        self.k_B = cnt.k_B.to(u.erg / u.K).value#1.38064852e-16 #erg / K
        
        #Source parameters
        self.av_host = 0
        self.av_mw = 0
        self.law = "Fitzpatrick"
        self.law_mw = "Fitzpatrick"
        
        #Black body models
        self.initT1 = 10000 #K
        self.initR1 = 1 # Rsun
        self.initT2 = 3000 #K
        self.initR2 = 1 # Rsun
        self.z = None
        self.distMpc = None #in Mpc
        self.mjd = 0 
        
        #Power law models
        self.alpha = 0.75
        self.alphaerr1 = 0
        self.alphaerr2 = 0
        self.scale = 1
        self.scaleerr1 = 0.1
        self.scaleerr2 = 0.1

        #Disk model (scale is already in the power law model)
        #Stellar mass, radius, log accretion mass per year, outer radius of accretion disk
        self.Mstar = 1
        self.Mstarerr1 = 0.1
        self.Mstarerr2 = 0.1
        self.Rstar = 1
        self.Rstarerr1 = 0.1
        self.rstarerr2 = 0.1
        self.logMacc = -8
        self.logMaccerr1 = -9
        self.logMaccerr2 = -9
        self.R_out = 3
        self.R_outerr1 = 1
        self.R_outerr2 = 1
        
        #Location for plots
        self.plotdir = "../../data/plots"

        #Location for fit results
        self.resdir = "../../data/modelfits"
        self.resfile = "fit_results.txt"
        
        #MCMC parameters
        self.method = 'ensemble' #or HA for Hastings
        self.mhtune = True # tuning of the Metropolis-Hastings 
        self.niterations = 10000
        self.burnin = 5000
        self.threads = 10
        self.nwalkers = 20
        self.sampler = None
        self.model = "BlackBody" #others are "BlackBody_Av" or "BlackBody2_Av", "PowerLaw", "PowerLaw_BlackBody"

        #Input data parameters.
        #The fitter will run either with magnitudes or with fluxes
        self.mags = None
        self.magerrs = None
        self.bands = None
        #Indicates whether the magnitude is in AB or Vega
        self.photsys = None
        
        self.wls = None
        self.fluxes = None
        self.fluxerrs = None
        
        #Output
        self.T = None
        self.Terr1 = None
        self.Terr2 = None
        
        self.R = None
        self.Rerr1 = None
        self.Rerr2 = None
        
        self.L = None
        self.Lerr1 = None
        self.Lerr2 = None

        #Output for the secondary star
        self.Tsec = None
        self.Tsecerr1 = None
        self.Tsecerr2 = None
        
        self.Rsec = None
        self.Rsecerr1 = None
        self.Rsecerr2 = None
        
        self.Lsec = None
        self.Lsecerr1 = None
        self.Lsecerr2 = None
        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
        
        #Set the plotting characteristics
        self._matplotlib_init()
        

    
        self.banddic = {"Y": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/ctio_y_andicam.dat"),
                   "J": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_j_002.fits"),
                   "H": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_h_002.fits"),
                   "K": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_k_002.fits"),
                   "keck,J": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.J.dat"),
                   "keck,H": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.H.dat"),
                   "keck,Ks": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.Ks.dat"),
                   "keck,K": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.K.dat"),
                   "3.6": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac1_3.6.txt"),
                   "4.5": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac2_4.5.txt"),
                   "5.8": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac3_5.8.txt"),               
                   "8.0": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac4_8.0.txt"),
                   "spitzer,3.6": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac1_3.6.dat"),
                   "spitzer,4.5": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac2_4.5.dat"),
                   "spitzer,5.8": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac3_5.8.dat"),               
                   "spitzer,8.0": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac4_8.0.dat"),
                   "wise,w1":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/WISE_WISE.W1.dat"),
                   "wise,w2":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/WISE_WISE.W2.dat"),
                   "wise,w3":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/WISE_WISE.W3.dat"),
                   "wise,w4":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/WISE_WISE.W4.dat"),
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
                   "nirc2,j":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.J.dat"),
                   "nirc2,h":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.H.dat"),
                   "nirc2,ks":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Keck_NIRC2.Ks.dat")

           }


    def _matplotlib_init(self):
        '''
        Set up preferences on matplotlib plot appearance.
        '''
        matplotlib.rcParams['xtick.minor.size'] = 6
        matplotlib.rcParams['xtick.major.size'] = 6
        matplotlib.rcParams['ytick.major.size'] = 6
        matplotlib.rcParams['xtick.minor.size'] = 4
        matplotlib.rcParams['ytick.minor.size'] = 4
        matplotlib.rcParams['lines.linewidth'] = 0.5
        matplotlib.rcParams['axes.linewidth'] = 1.5
        matplotlib.rcParams['font.size']= 14.0
        matplotlib.rcParams['font.family']= 'sans-serif'
        matplotlib.rcParams['xtick.major.width']= 2.
        matplotlib.rcParams['ytick.major.width']= 2.
        matplotlib.rcParams['ytick.direction']='in'
        matplotlib.rcParams['xtick.direction']='in'
        
    def _band2flux(self):
        '''
        Will transform the magnitude measurement into a flux measurement. 
        '''

        wls = np.array([])        
        fluxes = np.array([])
        fluxerr = np.array([])
        
        #Create a black body spectrum with an arbitrary value
        lam = np.linspace(100, 80000, 10000)
        sp = ps.BlackBody(10000)
        sp.convert('flam')

        sp2 = self._model_2(lam, 10000, 1)
        sp2 = sp2 * np.max(sp.flux) / np.max(sp2)
        
        sp = ps.ArraySpectrum(lam, sp2)

        
        for b, m, me, psys in zip(self.bands, self.mags, self.magerrs, self.photsys):
            
            print ("Band,",b)
            #Create the observation bandpass
            try:
                band = ps.ObsBandpass(b)
            except ValueError:
                #The band is not in the standard list
                #We need to go to the dictionary to retrieve the transmission function.
                band = ps.FileBandpass(self.banddic[b])
                    #band.waveunits.convert("angstrom")
                #else:
                #    band.waveunits = ps.units.Angstrom
            
            #Oftain the effective (average) wavelength
            effwave = band.avgwave()
            

            
            #Correct for Milky Way extinction
            m = m - extinction.fitzpatrick99(np.array([effwave]), a_v=self.av_mw, unit='aa')[0]
            
            #Normalize the spectrum to the magnitude of the observation
            sp_norm = sp.renorm(m, psys, band)
            #Observe with the band
            obs = ps.Observation(sp_norm, band)
            
            #Get the flux
            flux = obs.effstim('flam')

            wls = np.append(wls, effwave)
            fluxes = np.append(fluxes, flux) 
            
            #Compute the error bars
            flux_high = flux * 10**(0.4*me)
            flux_low = flux * 10**(-0.4*me)
            
            fluxerr = np.append(fluxerr, np.average([flux - flux_low, flux_high-flux]))

        return wls, fluxes, fluxerr
    
           
    def _model(self, lam, p):
        '''
        Returns the flux for the single BlackBody model for the wavelength introduced.
        lam is in A.
        p = (T, R)
        '''
        
        lam = lam * u.Angstrom
        
        T = p[0] * u.K
        R = (p[1] * u.Rsun).to(u.cm)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Area = np.pi * (4 * np.pi * R**2)
            flam =  Area * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
                (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T))-1)
        
        return flam.to(u.erg/u.s/u.Angstrom).value

    def _model_2(self, lam, T, R):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        lam = lam * u.Angstrom
        
        T = T * u.K
        R = (R * u.Rsun).to(u.cm)
        Area = np.pi * (4 * np.pi * R**2)
        
        flam =  Area * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T))-1)
        
        return flam.to(u.erg/u.s/u.Angstrom).value
        
    def _model_av_r(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        T = p[0] * u.K
        R = (p[1] * u.Rsun).to(u.cm)
        a_v = p[2]

        if a_v < 0:
            return lam * np.inf
            
        #Compute the effect of reddening as a flux factor
        flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam, a_v, unit='aa'))
        lam = lam * u.Angstrom
        

        area = np.pi * (4 * np.pi * R**2)
        flam =  area * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T))-1)
        
        #Apply the reddening
        flam = flam.to(u.erg/u.s/u.Angstrom).value * flux_red
        
                
        return flam

    def _model_av_r_2(self, lam, T, R, a_v):
        '''
        Return units: erg s-1 A-1
        '''

        return self._model_av_r(lam, (T, R, a_v))

        
    def _model2_av(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        T1 = p[0] * u.K
        R1 = (p[1] * u.Rsun).to(u.cm)
        a_v = p[2]
        T2 = p[3] * u.K
        R2 = (p[4] * u.Rsun).to(u.cm)

        #Compute the effect of reddening as a flux factor
        flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam, a_v, unit='aa'))
        lam = lam * u.Angstrom
        

        area1 = np.pi * (4 * np.pi * R1**2)
        area2 = np.pi * (4 * np.pi * R2**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flam1 =  area1 * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T1))-1)
            flam2 =  area2 * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T2))-1)
            
        flam = flam1 + flam2
        #Apply the reddening
        flam = flam.to(u.erg/u.s/u.Angstrom).value * flux_red
        
                
        return flam

    def _model2_av_2(self, lam, T1, R1, a_v, T2, R2):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model2_av(lam, (T1, R1, a_v, T2, R2))

    def _model2_av_r(self, lam, p):
        '''
        Return units: erg s-1 A-1
        '''

    
        T1 = p[0] #In K
        R1 = p[1]*69570000000.0 #From Rsun to cm
        a_v = p[2]
        T2 = p[3]
        R2 = p[4]*69570000000.0 #From Rsun to cm
        
        lam = lam * 1e-8 #To cm
    
        if a_v < 0:
            return lam * np.inf
            
        #We need an extra pi as it is integrated across all steradians
        #The second factor is the surface of the black body
        #The third ones is the Plank law
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flam1 =  np.pi * (4 * np.pi * R1**2) * ( (2*self.h*self.c**2)/( lam**5))/ (np.exp((self.h*self.c)/(lam*self.k_B*T1))-1)
            flam2 =  np.pi * (4 * np.pi * R2**2) * ( (2*self.h*self.c**2)/( lam**5))/ (np.exp((self.h*self.c)/(lam*self.k_B*T2))-1)
    
        #Compute the effect of reddening as a flux factor
        flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam*1e8, a_v, unit='aa'))
        
        flam = (flam1 + flam2) * flux_red *1e-8 #to erg / s / A

        #Apply the reddening and transform to erg /s/ A from cm
        return flam 

    def _model2_av_r_2(self, lam, T1, R1, a_v, T2, R2):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model2_av_r(lam, (T1, R1, a_v, T2, R2))

    def _model2_r(self, lam, p):
        '''
        Return units: erg s-1 A-1
        '''

    
        T1 = p[0] #In K
        R1 = p[1]*69570000000.0 #From Rsun to cm
        T2 = p[2]
        R2 = p[3]*69570000000.0 #From Rsun to cm
        
        lam = lam * 1e-8 #To cm

        #We need an extra pi as it is integrated across all steradians
        #The second factor is the surface of the black body
        #The third ones is the Plank law
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            flam1 =  np.pi * (4 * np.pi * R1**2) * ( (2*self.h*self.c**2)/( lam**5))/ (np.exp((self.h*self.c)/(lam*self.k_B*T1))-1)
            flam2 =  np.pi * (4 * np.pi * R2**2) * ( (2*self.h*self.c**2)/( lam**5))/ (np.exp((self.h*self.c)/(lam*self.k_B*T2))-1)
     
        flam = (flam1 + flam2)*1e-8 #to erg / s / A

        return flam 

    def _model2_r_2(self, lam, T1, R1, T2, R2):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model2_r(lam, (T1, R1, T2, R2))
        
    def _model_powerlaw(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        lam = lam * u.Angstrom
        
        w0 = 4000 #p[0] #Refernce wavelength
        alpha = p[0]
        scale = p[1]
        a_v = p[2]
            
        f = ps.PowerLaw(w0, alpha)
        f.convert('flam')
        
        flam = np.interp(lam, f.wave, f.flux)
        
        flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam, a_v, unit='aa'))
        area = 10**scale

        return area * flam * flux_red #.to(u.erg/u.s/u.Angstrom).value

    def _model_powerlaw_2(self, lam, alpha, scale, a_v):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model_powerlaw(lam, (alpha, scale, a_v))
        
    def _model_powerlaw_bb(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        
        w0 = 4000 #p[0] #Refernce wavelength
        alpha = p[0]
        scale = p[1]
        T_bb = p[2]
        R_bb = p[3]
            
        bb_flux = self._model_2(lam, T_bb, R_bb)

        lam = lam * u.Angstrom
        f = ps.PowerLaw(w0, alpha)
        f.convert('flam')
        
        flam = np.interp(lam, f.wave, f.flux)
        
        area = 10**scale

        
        return area * flam + bb_flux

    def _model_powerlaw_bb_2(self, lam, alpha, scale, T_bb, R_bb):
        '''
        Return units: erg s-1 A-1
        '''

        return self._model_powerlaw_bb(lam, (alpha, scale, T_bb, R_bb))
        


    def _model_accretion_disk_old2(self, lam, Mstar, Rstar, logMacc, scale, R_out):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model_accretion_disk_old(lam, (Mstar, Rstar, logMacc, scale, R_out))

        
    def _model_accretion_disk_old(self, lam, p):
        '''
        Equation 1 from Kenyon, Hartmann, Hewett 1988.
        '''    
        

        Mstar = p[0]
        Rstar = p[1]
        Macc = p[2]
        scale = p[3]
        R_out = p[4]
        
        if Mstar<0 or Macc<-12 or Rstar<0.001 or scale<0 or R_out < Rstar:
            return np.ones(len(lam))*np.inf
            
        Macc = 10**Macc

        R = np.linspace(Rstar,R_out,20)
        dR = R[1] - R[0]

        F_r = (3 * cnt.G * Mstar * u.Msun * Macc * u.Msun/u.year / 8 / np.pi / (u.Rsun*Rstar)**3) * (Rstar/R)**3 * (1 - (Rstar/R)**0.5)
        F_r = F_r.to(u.erg/u.cm**2/u.s)
        T_r = ((F_r / cnt.sigma_sb)**0.25).to(u.K)
        
        T_max = 13000 * u.K *(Mstar)**0.25 * (Macc / 1e-5)**0.25 * (Rstar)**-0.75
        
        #Cretae the disk model
        #For each differential radii, we compute the black body spectra corresponding 
        # to the temperature at that radius, and scale it by the flux expected at that
        # radius.
        
        disk_model = []
        for i, ri in enumerate(R):
            if ri>Rstar and ri<=1.5*Rstar:
                sp = ps.BlackBody(T_max.value)
                #sp = ps.BlackBody(T_r[i].value)
            else:
                sp = ps.BlackBody(T_r[i].value)
            sp.convert('flam')
            tot_flux = sp.trapezoidIntegration(sp.wave, sp.flux)
            #Compute the total emitted flux for the spherical area.
            #Adopt the outer radius as the 
            dist_flux_fac = np.pi * ((ri+dR)**2 - ri**2) * (u.Rsun.to(u.cm))**2
            scaled_flux = sp.flux / tot_flux * F_r[i].value #*  dist_flux_fac
            disk_model.append(scaled_flux)
            
        disk = np.array(disk_model)
        disk = np.nansum(disk, axis=0)
        
        sp = ps.ArraySpectrum(sp.wave, disk)
    
        #int_flux = sp.trapezoidIntegration(sp.wave, sp.flux)
        int_flux = np.max(sp.flux)


        #Normalize (recover) the integral flux from 1kpc
        flux_norm= sp.flux #/int_flux
        #sp_norm = ps.ArraySpectrum(sp.wave, flux_norm)
                
        flux_norm =  np.interp(lam, sp.wave, flux_norm)
        #flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam, a_v, unit='aa'))

        return flux_norm #* scale #* flux_red

    def _model_disk_T(self, R, Mstar, Rstar, logMacc):
        
        
        F_r = (3 * cnt.G * Mstar * 10**float(logMacc) * (u.Msun**2/u.year)) \
            / (8 * np.pi * (u.Rsun*R)**3) \
            * (1 - (Rstar/R)**0.5)
        T_r = ((F_r / cnt.sigma_sb)**0.25).to(u.K)
        
        #print (F_r, T_r)
        mask = (R>=Rstar) * (R<=1.5*Rstar)
        if np.count_nonzero(mask)>0:
            T_max = 13000 * u.K *(Mstar)**0.25 * (10**float(logMacc) / 1e-5)**0.25 * (Rstar)**-0.75
            T_r[mask] = T_max
            #print (mask, "Tmax", T_max, np.count_nonzero(mask))
        
        return T_r.value
     

    def _model_accretion_disk2(self,  lam, Mstar, Rstar, logMacc, R_out):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model_accretion_disk(lam, (Mstar, Rstar, logMacc, R_out))
        
    def _model_accretion_disk(self, lam, p):

        Mstar = np.maximum(1e-6, p[0])
        Rstar = np.maximum(1e-6, p[1])
        logMacc = np.maximum(-12, np.minimum(-7, p[2]))
        R_out = np.maximum(1e-6, p[3])
        i = 45.0
        #Deg to radians
        i = np.deg2rad(i%360)
        d = self.distMpc*(u.Mpc).to(u.cm)
        
        
        R = np.linspace(Rstar, R_out, 30)*u.Rsun
        nu = (cnt.c / (lam*u.Angstrom)).to(u.Hz)
        T_r = self._model_disk_T(R.value, Mstar, Rstar, logMacc)
        
        
        F_nu_arr = []
        for ni in nu:
            I_nu_r = R / (np.exp(cnt.h * ni/(cnt.k_B*T_r*u.K)) - 1)
            I_flux = np.trapz(I_nu_r, R)
            F_nu = (4 * np.pi * cnt.h * np.cos(i)*ni**3)/(cnt.c**2 * d**2) * I_flux
            F_nu_arr.append(F_nu.to(u.erg/u.s/u.Hz).value)
            
        F_nu_arr = np.array(F_nu_arr)
        s = ps.ArraySpectrum(lam, F_nu_arr, fluxunits='fnu', waveunits='Angstrom')
        s.convert('flam')
        
        fluxFactor = 4*np.pi*d**2


        return s.flux*fluxFactor

                
    #likelihood function
    def _like(self, p, xdat, ydat, errdat, debug=False):
        '''
        p: function parameters 
        args: carry anything we want to pass to our function (e.g. the data) 
        '''   
        
        if self.model == "BlackBody":
            ymod = self._model(xdat, p)
        elif self.model == "BlackBody_Av":
            ymod = self._model_av_r(xdat, p)
        elif self.model == "BlackBody2_Av":
            ymod = self._model2_av_r(xdat, p)
        elif self.model == "BlackBody2":
            ymod = self._model2_r(xdat, p)
        elif self.model == "PowerLaw":
            ymod = self._model_powerlaw(xdat, p)
        elif self.model == "PowerLaw_BlackBody":
            ymod = self._model_powerlaw_bb(xdat, p)
        elif self.model == "Disk":
            ymod = self._model_accretion_disk(xdat, p)
        else:
            print ("Unknown model", self.model)
            return np.nan

        #Discard models which exceed the upper limits
        if (np.any(ymod[errdat<0] > ydat[errdat<0])):
            prob = 1e-320
        else:
            prob = stats.norm.pdf(ydat, ymod, errdat) 
    
        # log probcilities
        # we add tiny number to avoid NaNs
        mylike = np.log(prob + 1e-320).sum() 
    
    	
        return mylike	
    
    
    
    def _logposterior(self, p, xdat, ydat, errdat):
        '''
        Returns the posterior of the observations. In essence the likelihood and the prior:
        #log(likelihood) + log(prior)
        '''
        lp = self._logprior(p)
        if (not np.isinf(lp)):
            lp= self._like(p, xdat, ydat, errdat) + lp
        return lp
    
    
    def _logprior(self, p):
        '''
        Returns the prior probability distribution for each model.

        '''
        
        if self.model == "BlackBody":
            
            T1 = p[0] 
            R1 = p[1] 
    
            if T1 < 0 or R1 < 0:
                return -np.inf
    
            logp = stats.uniform.logpdf(T1, 1000, 20000)
            logp = logp + stats.uniform.logpdf(R1, 0,  100)
        
        if self.model =="BlackBody_Av":
            
            T1 = p[0] 
            R1 = p[1] 
            av = p[2]
    
            if T1 < 0 or R1 < 0 or av < 0:
                return -np.inf
            else:
                logp = stats.uniform.logpdf(T1, 1000, 20000)
                logp = logp + stats.uniform.logpdf(R1, 0,  100)
                logp = logp + stats.uniform.logpdf(av, 0,  3)

        elif self.model == "BlackBody2":
            T1 = p[0] 
            R1 = p[1] 
            T2 = p[2]
            R2 = p[3]

            if T1 < 0 or T2 > T1 or T2 < 0:
                return - np.inf
            else:
                logp = stats.uniform.logpdf(T1, 1000, 60000)
                logp = logp + stats.uniform.logpdf(R1, 0,  70000)
                logp = logp + stats.uniform.logpdf(T2, 100, 5000)
                logp = logp + stats.uniform.logpdf(R2, 0, 5000000)

        elif self.model == "BlackBody2_Av":
            T1 = p[0] 
            R1 = p[1] 
            av = p[2]
            T2 = p[3]
            R2 = p[4]

            if T1 < 0 or T2 > T1 or T2 < 0 or av < 0 or av > 10:
                return - np.inf

            else:
                logp = stats.uniform.logpdf(T1, 1000, 20000)
                logp = logp + stats.uniform.logpdf(R1, 0,  100)
                logp = logp + stats.uniform.logpdf(av, 0, 3)
                logp = logp + stats.uniform.logpdf(T2, 500, 20000)
                logp = logp + stats.uniform.logpdf(R2, 0, 500)
                
        elif self.model == "PowerLaw":
            alpha = p[0]
            scale = p[1] 
            av = p[2]
            
            if av < 0:
                logp = -np.inf
            else:
                logp = stats.uniform.logpdf(alpha, 0, 3)
                logp = logp + stats.uniform.logpdf(scale, 0.1, 100)
                logp = logp + stats.uniform.logpdf(av, 0, 3)

        elif self.model == "PowerLaw_BlackBody":
            alpha = p[0]
            scale = p[1] 
            T1 = p[2]
            R1 = p[3]
            
            if R1 < 0 or T1 < 0 or alpha < 0:
                logp = -np.inf
            else:
                logp = stats.uniform.logpdf(alpha, 0, 3)
                logp = logp + stats.uniform.logpdf(scale, 0.1, 100)
                logp = logp + stats.uniform.logpdf(T1, 500, 20000)
                logp = logp + stats.uniform.logpdf(R1, 0, 500)
                
        elif self.model == "Disk":
            
            Mstar = p[0]
            Rstar = p[1] 
            logMacc = p[2]
            R_out = p[3]
            
            if Rstar < 0 or Mstar < 0 or logMacc < -12 or R_out<0 or R_out < Rstar:
                logp = -np.inf            
            else:
                logp = stats.uniform.logpdf(Mstar, 0, 1.44)
                logp = logp + stats.uniform.logpdf(Rstar, 0, 10)
                logp = logp + stats.uniform.logpdf(logMacc, -12, 7)
                logp = logp + stats.uniform.logpdf(R_out, 0, 50)

        return logp	

    def _get_max_and_intervals(self, x):
        '''
        Provided a chain of samples, finds the average value and returns the values
        for a 1 sigma distribution following the 34 and 66 percentiles.
        '''


        return np.percentile(x, 34), np.percentile(x, 50), np.percentile(x, 66)
        #return percent1, maxp, percent2

    def _area2rsun(self, A):
        '''
        Given the area of the black body in cm2 returns the radius for the object in solar radius.
        
        '''
        Aream2 = A * u.cm**2 # add units    
        Rad = np.sqrt(Aream2/(4*(np.pi)**2)).to(u.Rsun) #in Rsun
        
        return Rad.value
        
    def _fill_output(self):
        '''
        Computes the confidence intervals from the MCMC distribution.
        Transforms the temperature ad radius into a black body luminosity.
        '''
        
        if self.model.startswith("BlackBody"):
            T1, T, T2 =  self._get_max_and_intervals(self.sampler.flatchain[:,0])
            R1, R, R2 =  self._get_max_and_intervals(self.sampler.flatchain[:,1])
            
            self.T = T
            self.Terr1 = T - T1
            self.Terr2 = T2 - T
                 
            self.R = R
            self.Rerr1 = R - R1
            self.Rerr2 = R2 - R
            
            self.L = self._get_bol_lum(T, R)
            self.Lerr1 = self.L - self._get_bol_lum(T1, R1)
            self.Lerr2 = self._get_bol_lum(T2, R2) - self.L
            
            if self.model == "BlackBody_Av":
                Av1, Av, Av2 = self._get_max_and_intervals(self.sampler.flatchain[:,2])
                self.Av = Av
                self.Averr1 = Av - Av1
                self.Averr2 = Av2 - Av

            elif self.model == "BlackBody2_Av":
                
                Av1, Av, Av2 = self._get_max_and_intervals(self.sampler.flatchain[:,2])
                Tsec1, Tsec, Tsec2 =  self._get_max_and_intervals(self.sampler.flatchain[:,3])
                Rsec1, Rsec, Rsec2 =  self._get_max_and_intervals(self.sampler.flatchain[:,4])
                
                self.Av = Av
                self.Averr1 = Av - Av1
                self.Averr2 = Av2 - Av
                
                self.Tsec = Tsec
                self.Tsecerr1 = Tsec - Tsec1
                self.Tsecerr2 = Tsec2 - Tsec
                     
                self.Rsec = Rsec
                self.Rsecerr1 = Rsec - Rsec1
                self.Rsecerr2 = Rsec2 - Rsec
                
            elif self.model == "BlackBody2":
                
                Tsec1, Tsec, Tsec2 =  self._get_max_and_intervals(self.sampler.flatchain[:,2])
                Rsec1, Rsec, Rsec2 =  self._get_max_and_intervals(self.sampler.flatchain[:,3])
                
                self.Tsec = Tsec
                self.Tsecerr1 = Tsec - Tsec1
                self.Tsecerr2 = Tsec2 - Tsec
                     
                self.Rsec = Rsec
                self.Rsecerr1 = Rsec - Rsec1
                self.Rsecerr2 = Rsec2 - Rsec

                Lsec = self._get_bol_lum(Tsec, Rsec)
                Lerr1 = Lsec - self._get_bol_lum(Tsec1, Rsec1)
                Lerr2 = self._get_bol_lum(Tsec2, Rsec2) - Lsec
                
                self.L = self.L + Lsec
                self.Lerr1 = np.sqrt(self.Lerr1**2 + Lerr1**2)
                self.Lerr2 = np.sqrt(self.Lerr2**2 + Lerr2**2)
            
        elif self.model=="PowerLaw":
            alpha1, alpha, alpha2 =  self._get_max_and_intervals(self.sampler.flatchain[:,0])
            R1, R, R2 =  self._get_max_and_intervals(self.sampler.flatchain[:,1])
            Av1, Av, Av2 = self._get_max_and_intervals(self.sampler.flatchain[:,2])
        
            self.alpha = alpha
            self.alphaerr1 = alpha - alpha1
            self.alphaerr2 = alpha2 - alpha
            
            self.R = R
            self.Rerr1 = R - R1
            self.Rerr2 = R2 - R
            
            self.Av = Av
            self.Averr1 = Av - Av1
            self.Averr2 = Av2 - Av
            
        elif self.model=="PowerLaw_BlackBody":
            alpha1, alpha, alpha2 =  self._get_max_and_intervals(self.sampler.flatchain[:,0])
            scale1, scale, scale2 =  self._get_max_and_intervals(self.sampler.flatchain[:,1])
            T1, T, T2 = self._get_max_and_intervals(self.sampler.flatchain[:,2])
            R1, R, R2 = self._get_max_and_intervals(self.sampler.flatchain[:,3])
        
            self.alpha = alpha
            self.alphaerr1 = alpha - alpha1
            self.alphaerr2 = alpha2 - alpha
            
            self.scale = scale
            self.scaleerr1 = scale - scale1
            self.scaleerr2 = scale2 - scale
            
            self.T = T
            self.Terr1 = T - T1
            self.Terr2 = T2 - T

            self.R = R
            self.Rerr1 = R - R1
            self.Rerr2 = R2 - R
            
        elif self.model=="Disk":
            Mstar1, Mstar, Mstar2 =  self._get_max_and_intervals(self.sampler.flatchain[:,0])
            Rstar1, Rstar, Rstar2 = self._get_max_and_intervals(self.sampler.flatchain[:,1])
            logMacc1, logMacc, logMacc2 = self._get_max_and_intervals(self.sampler.flatchain[:,2])
            R_out1, R_out, R_out2 =  self._get_max_and_intervals(self.sampler.flatchain[:,3])
            #scale1, scale, scale2 =  self._get_max_and_intervals(self.sampler.flatchain[:,3])
        
            self.Mstar = Mstar
            self.Mstarerr1 = Mstar - Mstar1
            self.Mstarerr2 = Mstar2 - Mstar
            
            self.Rstar = Rstar
            self.Rstarerr1 = Rstar - Rstar1
            self.Rstarerr2 = Rstar2 - Rstar
            
            self.logMacc = logMacc
            self.logMaccerr1 = logMacc - logMacc1
            self.logMaccerr2 = logMacc2 - logMacc
            
            self.R_out = R_out
            self.R_outerr1 = R_out - R_out1
            self.R_outerr2 = R_out2 - R_out
            
    def _save_output(self):
        '''
        Saves in a results file.
        '''
        
        exists = os.path.isfile(self.resfile)

        with open(self.resfile, 'a') as outfile:
            
            print ("Saving results to %s"%self.resfile)
            if self.model == "BlackBody":
                
                if not exists:
                    outfile.write("mjd T Terr1 Terr2 R Rerr1 Rerr2 L Lerr1 Lerr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3e %.3e %.3e %.3f\n"%\
                    (self.mjd, self.T, self.Terr1, self.Terr2, self.R, self.Rerr1, self.Rerr2, self.L, self.Lerr1, self.Lerr2, self.av_mw))
                    
            elif self.model == "BlackBody_Av":

                if not exists:
                    outfile.write("mjd T Terr1 Terr2 R Rerr1 Rerr2 L Lerr1 Lerr2 Av Averr1 Averr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3e %.3e %.3e %.3f %.3f %.3f %.3f\n"%\
                    (self.mjd, self.T, self.Terr1, self.Terr2, self.R, self.Rerr1, self.Rerr2, \
                        self.L, self.Lerr1, self.Lerr2, self.Av, self.Averr1, self.Averr2, self.av_mw))
    
            elif self.model == "BlackBody2":
                    
                if not exists:
                    outfile.write("mjd T Terr1 Terr2 R Rerr1 Rerr2 L Lerr1 Lerr2 Tsec Tsecerr1 Tsecerr2 Rsec Rsecerr1 Rsecerr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3e %.3e %.3e %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"%\
                    (self.mjd, self.T, self.Terr1, self.Terr2, self.R, self.Rerr1, self.Rerr2, \
                        self.L, self.Lerr1, self.Lerr2, 
                        self.Tsec, self.Tsecerr1, self.Tsecerr2, self.Rsec, self.Rsecerr1, self.Rsecerr2, self.av_mw))
                        
            elif self.model == "BlackBody2_Av":
                    
                if not exists:
                    outfile.write("mjd T Terr1 Terr2 R Rerr1 Rerr2 L Lerr1 Lerr2 Av Averr1 Averr2 Tsec Tsecerr1 Tsecerr2 Rsec Rsecerr1 Rsecerr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3e %.3e %.3e %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"%\
                    (self.mjd, self.T, self.Terr1, self.Terr2, self.R, self.Rerr1, self.Rerr2, \
                        self.L, self.Lerr1, self.Lerr2, self.Av, self.Averr1, self.Averr2,\
                        self.Tsec, self.Tsecerr1, self.Tsecerr2, self.Rsec, self.Rsecerr1, self.Rsecerr2, self.av_mw))
                    
            elif self.model == "PowerLaw":
                
                if not exists:
                    outfile.write("mjd alpha alphaerr1 alphaerr2 scale scaleerr1 scaleerr2 Av Averr1 Averr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"%\
                    (self.mjd, self.alpha, self.alphaerr1, self.alphaerr2, self.scale, self.scaleerr1, self.scaleerr2, \
                        self.Av, self.Averr1, self.Averr2, self.av_mw))

            elif self.model == "PowerLaw_BlackBody":
                
                if not exists:
                    outfile.write("mjd alpha alphaerr1 alphaerr2 scale scaleerr1 scaleerr2 T Terr1 Terr2 R Rerr1 Rerr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"%\
                    (self.mjd, self.alpha, self.alphaerr1, self.alphaerr2, self.scale, self.scaleerr1, self.scaleerr2, \
                    self.T, self.Terr1, self.Terr2, \
                    self.R, self.Rerr1, self.Rerr2, \
                        self.av_mw))
                        
            elif self.model == "Disk":
                
                if not exists:
                    outfile.write("mjd M Merr1 Merr2 Rstar Rerr1 Rerr2 Macc Maccerr1 Maccerr2 R_out R_outerr1 R_outerr2 Av_MW\n")
                outfile.write("%.5f %.3f %.3f %.3f %.3f %.3f %.3f %.3e %.3e %.3e %.3e %.3e %.3e %.3f\n"%\
                    (self.mjd, self.Mstar, self.Mstarerr1, self.Mstarerr1, \
                    self.Rstar, self.Rstarerr1, self.Rstarerr2,\
                    self.logMacc, self.logMaccerr1, self.logMaccerr2,\
                     #self.scale, self.scaleerr1, self.scaleerr2, \
                     self.R_out, self.R_outerr1, self.R_outerr2,\
                        self.av_mw))
            else:
                print ("Unknown model! %s"%self.model)
            
    def _get_bol_lum(self, T, R):
        '''
        T is in K
        R in R_sun.
        
        Gives the Lbol in Lsun
        '''

        L =  cnt.sigma_sb * (T * u.K)**4 * 4 * np.pi * (R*u.Rsun)**2
        
        return (L.to(u.Lsun)).value

    def _get_save_path(self, savefile, plot_name=""):
        '''
        Checks what savefile name has been given.
        If there is a value, then it jsut stores it in the plot directory provided.
        If there is no name, then it creates a filename with the suffix provided.
        It also checks if there is already a file named like that, and it that is the case,
        it increases the suffix so that it has a higher number, avoiding collision.
        '''
        #If there is a given name to store the file, then we use that one
        if (not savefile is None):
            if os.path.dirname(savefile) == "":
                name = os.path.join(self.plotdir, os.path.basename(savefile))
        #If there is no name, then we will save the plots in the plot directory 
        #with an automatic name.
        # This name will increase a count if the name exists already.
        else:            
            i = 0
            name = os.path.join(self.plotdir, "%s_%.1f_%d.pdf"%(plot_name, self.mjd, i))

            while (os.path.isfile(name)):
                i = i+1
                name = os.path.join(self.plotdir, "%s_%.1f_%d.pdf"%(plot_name, self.mjd, i))
                
        return name
                

    def _initialize_parameters(self, plot=False):
        '''
        Runs the least squares optimiztion routine to find the best initial parameters 
        to start the MCMC with.
        '''

        lam = np.linspace(3000, 25000, 2000)
        a_v_wls = extinction.fitzpatrick99(self.wls, a_v=self.av_mw, unit='aa')
        reddening = 10**(0.4*a_v_wls)
        
        if self.model == "BlackBody":
            flux_ini = self._model_2(lam, self.initT1, self.initR1)

            p0 = (self.initT1, self.initR1)
            
            print ("Initial parameters given:", p0)
            
            #Perform a LSQ fit
            #params, covar = curve_fit(self._model_2, self.wls , self.fluxes, \
            #p0 = p0, sigma=self.fluxerrs, absolute_sigma=True, maxfev = 20000)
            #flux_end = self._model_2(lam, *params)
            
            if plot:
                plt.clf()
                plt.plot(lam, flux_ini, "r--", label="Fit initial parameters")
                #plt.plot(lam, flux_end, label="Best fit LSQ")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$F_{\\lambda}$ [erg/s/cm2/A]")
                plt.ylim(0.8*np.min(self.fluxes), 1.2*np.max(self.fluxes))
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs_bb")
                plt.savefig(name, dpi=200)
                
        elif self.model == "BlackBody_Av":
            flux_ini = self._model_av_r_2(lam, self.initT1, self.initR1, self.av_host)

            #p0 = (self.initT1, self.initR1, self.av_host)
            
            print ("Initial ", p0)
            
            #params, covar = curve_fit(self._model_av_r_2, self.wls , self.fluxes, \
            #p0 = p0, sigma=self.fluxerrs, absolute_sigma=True, maxfev = 20000)
            #flux_end = self._model_av_r_2(lam, *params)
            
            if plot:
                plt.clf()
                plt.plot(lam, flux_ini, "r--", label="Fit initial parameters")
                #plt.plot(lam, flux_end, label="Best fit LSQ")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.ylim(0.8*np.min(self.fluxes), 1.2*np.max(self.fluxes))
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs_bb_av")
                plt.savefig(name, dpi=200)
                
        elif self.model == "BlackBody2_Av":
            flux_ini = self._model2_av_r_2(lam, self.initT1, self.initR1, self.av_host, self.initT2, self.initR2)

            p0 = (self.initT1, self.initR1, self.av_host, self.initT2, self.initR2)
            
            print ("Initial ", p0)
            
            #params, covar = curve_fit(self._model2_av_r_2, self.wls , self.fluxes, \
            #p0 = p0, sigma=self.fluxerrs, absolute_sigma=True, maxfev = 20000)
            #flux_end = self._model2_av_r_2(lam, *params)
            
            if plot:
                plt.clf()
                plt.plot(lam, flux_ini, "r--", label="Fit initial parameters")
                #plt.plot(lam, flux_end, label="Best fit LSQ")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.ylim(0.8*np.min(self.fluxes), 1.2*np.max(self.fluxes))
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs")
                plt.savefig(name, dpi=200)

        elif self.model == "BlackBody2":
            flux_ini = self._model2_r_2(lam, self.initT1, self.initR1, self.initT2, self.initR2)

            p0 = (self.initT1, self.initR1, self.initT2, self.initR2)
            
            print ("Initial ", p0)
            
            #params, covar = curve_fit(self._model2_r_2, self.wls , self.fluxes, \
            #p0 = p0, sigma=self.fluxerrs, absolute_sigma=True, maxfev = 20000)
            #flux_end = self._model2_r_2(lam, *params)
            #flux_1 = self._model_2(lam, *params[0:2])
            #flux_2 = self._model_2(lam, *params[2:])
            
            
            if plot:
                plt.clf()
                plt.figure(figsize=(6,4))
                plt.plot(lam, flux_ini, "r--", label="Fit initial parameters")
                #plt.plot(lam, flux_end, label="Best fit LSQ")
                #plt.plot(lam, flux_1, label="BB1")
                #plt.plot(lam, flux_2, label="BB2")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.legend(loc="best", fontsize=10)
                plt.ylim(0.8*np.min(self.fluxes), 1.2*np.max(self.fluxes))
                plt.yscale("log")
                name = self._get_save_path(None, "fluxes_obs_2bb")
                plt.savefig(name, dpi=200)
                
        elif self.model == "PowerLaw":

            #params, covar = curve_fit(self._model_powerlaw_2, self.wls , self.fluxes, \
            #p0=(self.alpha, self.initR1, self.av_host), sigma=self.fluxerrs, absolute_sigma=True, maxfev = 10000)
            
            lam = np.linspace(3000, 25000, 2000)
            fluxpw = self._model_powerlaw_2(lam, self.alpha, self.scale, self.av_host)
                        
            if plot:
                plt.clf()
                plt.plot(lam, fluxpw, label="Fit initial parameters")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.ylim(0.8*np.min(self.fluxes), 1.2*np.max(self.fluxes))
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs_powerlaw")
                plt.savefig(name, dpi=200)
                print ("Saved fit as %s"%name)

        elif self.model == "PowerLaw_BlackBody":

            #params, covar = curve_fit(self._model_powerlaw_2, self.wls , self.fluxes, \
            #p0=(self.alpha, self.initR1, self.av_host), sigma=self.fluxerrs, absolute_sigma=True, maxfev = 10000)
            
            lam = np.linspace(3000, 25000, 2000)
            fluxpw = self._model_powerlaw_bb_2(lam, self.alpha, self.scale, self.initT1, self.initR1)
                        
            if plot:
                plt.clf()
                plt.plot(lam, fluxpw, label="Fit initial parameters")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="MW ext. corr")
                plt.errorbar(self.wls, self.fluxes/reddening, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.ylim(0.8*np.min(self.fluxes/reddening), 1.2*np.max(self.fluxes))
                plt.legend(loc="best")
                name = self._get_save_path(None, "fluxes_obs_powerlaw_bb")
                plt.savefig(name, dpi=200)
                print ("Saved fit as %s"%name)
                
        if self.model == 'Disk':
            
             
            #params = (0.5, 0.2, 5e-9, 1, 2)
            p0 = (self.Mstar,  self.Rstar, self.logMacc, self.R_out)

            #params, covar = curve_fit(self._model_accretion_disk2, self.wls , self.fluxes, \
            #p0 = p0, sigma=self.fluxerrs, absolute_sigma=True, maxfev = 20000)
            
            #print ("LSQ fit:  Mstar:", params[0], " Rstar", params[1], "logMacc ", \
            #   params[2], "R_out", params[3])

            lam = np.linspace(3000, 25000, 2000)

            #flux_disk = self._model_accretion_disk2(lam, params[0], params[1], params[2], params[3])
            
            if plot:
                plt.clf()
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                #plt.plot(lam, flux_disk, lw=3)
                plt.xlabel("Wavelength [$\\mu$m]")
                plt.ylabel("Flux [erg/cm$^2$/s]")
                plt.ylim(np.nanmin(self.fluxes)*0.9, np.nanmax(self.fluxes)*1.2)
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs_disk")
                plt.savefig(name, dpi=200)
                print ("Saved fit as %s"%name)

    def  initialize(self, plot=False):
        '''
        Will transform the magnitudes to fluxes and use the distance to the object to
        calculate the luminosity at each wavelength.
        '''

        if (not os.path.isdir(self.plotdir)):
            os.makedirs(self.plotdir)
            print ("Created plot directory %s"%self.plotdir)

        #Directory where to store the results
        if (not os.path.isdir(self.resdir)):
            os.makedirs(self.resdir)
            print ("Created result directory %s"%(self.resdir))
        self.resfile = os.path.join(self.resdir, self.model + os.path.basename(self.resfile))

        # generate the data  
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   
            self.wls, self.fluxes, self.fluxerrs = self._band2flux()

        #Plot the raw fluxes before correcting them.
        '''if (plot):
            plt.figure(figsize=(8,6))
            plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0)
            for i in range(len(self.wls)):
                plt.text(self.wls[i], self.fluxes[i]*1.01, self.bands[i].split(",")[-1], alpha=.4)
            name = self._get_save_path(None, "fluxes_observed")
            plt.yscale("log")
            plt.xlabel("Wavelength [A]")
            plt.ylabel("log (Flux/[erg/cm2/s])")
            plt.tight_layout()
            plt.savefig(name, dpi=200)'''
            
        if not self.distMpc is None and self.distMpc !=0:
            print ("Using distance to the source of %.1e Mpc"%self.distMpc)
            fluxFactor = (4*np.pi*((self.distMpc*u.Mpc).to(u.cm) )**2).value

        elif (self.distMpc is None or self.distMpc==0 )and (not self.z is None and self.z != 0):
            self.distMpc = self.cosmo.luminosity_distance(self.z)
                
            #Compute the flux multiplication factor for the object if it is at distance distMpc
            #We transform that to cm, as the flux is in erg cm-2 s-1
            fluxFactor = (4*np.pi*(self.distMpc.to(u.cm) )**2).value
        
        else: # self.distMpc is None and self.z is None:
            #Here we do not use any multiplication flux factor
            print ("Warning: no redshift or distance provided!")
            fluxFactor = 1
            
        self.fluxes = self.fluxes * fluxFactor
        self.fluxerrs = self.fluxerrs * fluxFactor

        self._initialize_parameters(plot)

    
    def run(self):
        '''
        Runs the main MCMC process. 
        Retrieves the priors, the likelihood process and computes the posterior probability.
        '''
    
        xs = self.wls
        ys = self.fluxes
        errs = self.fluxerrs
        
        if self.model == "BlackBody":
            p0 = np.array([ self.initT1, self.initR1])
            sigs = np.array([2000, 10])
        elif self.model == "BlackBody_Av":
            p0 = np.array([ self.initT1, self.initR1, self.av_host])
            sigs = np.array([2000, 10, 0.5]) 
        elif self.model == "BlackBody2":
            p0 = np.array([ self.initT1, self.initR1, self.initT2, self.initR2])
            sigs = np.array([2000, 5, 2000, 5]) 
        elif self.model == "BlackBody2_Av":
            p0 = np.array([ self.initT1, self.initR1, self.av_host, self.initT2, self.initR2])
            sigs = np.array([2000, 5, 1,  2000, 5])   
        elif self.model == "PowerLaw":
            p0 = np.array([ self.alpha, self.scale, self.av_host])
            sigs = np.array([2, 3, 2])
        elif self.model == "PowerLaw_BlackBody":
            p0 = np.array([ self.alpha, self.scale, self.initT1, self.initR1])
            sigs = np.array([2, 3, 2000, 2])
        elif self.model == "Disk":
            p0 = np.array([ self.Mstar, self.Rstar, self.logMacc, self.R_out])
            sigs = np.array([0.1, 0.01, 1, 0.1])
            print ("Initialized with p0", p0, " and sigmas ", sigs)
        else:
            print ("-------------------CRITICAL ERROR!----------------------")
            print ("-------------------UNKNOWN model! %s----------------------"%self.model)
            print ("-------------------CRITICAL ERROR!----------------------")

            sys.exit()
            
        ndim = len(p0)
        
        # emsemble MCMC
        p0s = emcee.utils.sample_ball(p0, sigs, self.nwalkers)
        # initialize the ball of initial conditions
        #Supports the threads=X argument for parallelization		
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self._logposterior,\
            args=(xs, ys, errs), threads=10)
        pos, lnprob, state = sampler.run_mcmc(p0s, self.burnin)
        print ("Burning phase finished")
        sampler.reset()
        pos, lnprob, state = sampler.run_mcmc(pos, self.niterations)
        print ('Acceptance ratio', sampler.acceptance_fraction)
    
        self.sampler = sampler

        print ("MCMC main phase finished")
        self._fill_output()
        self._save_output()


        
    def plot_corner_posteriors(self, savefile=None):
        '''
        Plots the corner plot of the MCMC results.
        '''

        if self.model == "BlackBody2":
            labels=["T1", "R1", "T2", "R2"]        
        elif self.model.startswith("BlackBody"):
            labels=["T1", "R1", "Av", "T2", "R2"]
        elif self.model == "PowerLaw":
            labels=["alpha", "scale", "Av"]
        elif self.model == "PowerLaw_BlackBody":
            labels = ["alpha", "scale", "T", "R"]
        elif self.model == "Disk":
            labels = ["Mstar", "Rstar", "logMacc", "R_out"]
        
        ndim = len(self.sampler.flatchain[0,:])
        chain = self.sampler
        samples = chain.flatchain
        
        samples = samples[:,0:ndim]  
        plt.figure(figsize=(8,8))
        fig = corner.corner(samples, labels=labels[0:ndim])
        plt.title("MJD: %.2f"%self.mjd)
        name = self._get_save_path(savefile, "mcmc_posteriors")
        plt.savefig(name)
        plt.close("all")
        

        plt.figure(figsize=(8,ndim*3))
        for n in range(ndim):
            plt.subplot(ndim,1,n+1)
            chain = self.sampler.chain[:,:,n]
            nwalk, nit = chain.shape
            
            for i in np.arange(nwalk):
                plt.plot(chain[i], lw=0.1)
                plt.ylabel(labels[n])
                plt.xlabel("Iteration")
        name_walkers = self._get_save_path(savefile, "mcmc_walkers")
        plt.tight_layout()
        plt.savefig(name_walkers)
        plt.close("all")  
        
            

    def plot_fit(self):
        '''
        Plots the best fit model to the data.
        '''
        
        lam = np.linspace( np.min(self.wls) -500 , np.max(self.wls) + 500, 1000)
        
        plt.clf()
        plt.figure(figsize=(8,6))
        plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, fmt="o")
        for i in range(len(self.wls)):
                plt.text(self.wls[i], self.fluxes[i]*1.01, self.bands[i].split(",")[-1], alpha=.4)
        
        if self.model == "BlackBody":
            fluxbb = self._model(lam, (self.T, self.R))
            plt.plot(lam, fluxbb, "k-", label="BB fit")
            plt.title("T: %.2f K R:%.2f R$_{\odot}$ Lumiosity %.2e L$_{\odot}$"%(self.T, self.R, self.L))    

        elif self.model == "BlackBody_Av":
            fluxbb = self._model(lam, (self.T, self.R))
            fluxbb_red = self._model_av_r(lam, (self.T, self.R, self.Av))
            plt.plot(lam, fluxbb, "k-", label="BB fit")
            plt.plot(lam, fluxbb_red, "red", label="BB fit + reddening")
            plt.title("T: %.1f K R:%.1f R$_{\odot}$ Lumiosity %.1e L$_{\odot}$ Av: %.2f"%(self.T, self.R, self.L, self.Av))    

        elif self.model == "BlackBody2_Av":
            fluxbb_red = self._model_av(lam, (self.T, self.R, self.Av))
            fluxbb_secondary_red = self._model_av(lam, (self.Tsec, self.Rsec, self.Av))
            fluxbb_with_seconday = self._model2_av(lam, (self.T, self.R, self.Av, self.Tsec, self.Rsec))

            plt.plot(lam, fluxbb_red, "k-", label="BB1 fit + reddening")
            plt.plot(lam, fluxbb_secondary_red, "k--", label="BB2 fit + reddening")
            plt.plot(lam, fluxbb_with_seconday, "green", label="BB1 + BB2")
            plt.title("T: %.1f K R:%.1f R$_{\odot}$ Lumiosity %.1e L$_{\odot}$ Av: %.2f\n T2: %.1f R2: %.1f"%(self.T, \
                      self.R, self.L, self.Av, self.Tsec, self.Rsec)) 

        elif self.model == "BlackBody2":
            fluxbb_primary = self._model(lam, (self.T, self.R))
            fluxbb_secondary = self._model(lam, (self.Tsec, self.Rsec))
            fluxbb_with_seconday = self._model2_r(lam, (self.T, self.R, self.Tsec, self.Rsec))

            plt.plot(lam, fluxbb_primary, "k-", label="BB1 fit")
            plt.plot(lam, fluxbb_secondary, "k--", label="BB2 fit")
            plt.plot(lam, fluxbb_with_seconday, "green", label="BB1 + BB2")
            plt.title("T: %.1f K R:%.1f R$_{\odot}$ T2: %.1f R2: %.1f"%(self.T, \
                      self.R, self.Tsec, self.Rsec)) 
                      
        elif self.model == "PowerLaw":
            flux = self._model_powerlaw(lam, (self.alpha, self.scale, self.Av))
            plt.plot(lam, flux, "k-", label="PowerLaw + reddening")
            plt.title("$\\alpha$: %.1f Av: %.2f"%(self.alpha, self.Av))    

        elif self.model == "PowerLaw_BlackBody":
            flux = self._model_powerlaw_bb(lam, (self.alpha, self.scale, self.T, self.R))
            flux_pw = self._model_powerlaw(lam, (self.alpha, self.scale, 0))
            flux_bb = self._model(lam, (self.T, self.R))
            plt.plot(lam, flux, "k-", label="PowerLaw + BlackBody")
            plt.plot(lam, flux_pw, "b--", label="PowerLaw")
            plt.plot(lam, flux_bb, "g:", label="BlackBody")
            plt.title("$\\alpha$: %.1f scale: %.2e T: %.1f R:%.1f"%(self.alpha, self.scale, self.T, self.R))   

        elif self.model == "Disk":
            fluxdisk = self._model_accretion_disk(lam, (self.Mstar, self.Rstar, self.logMacc, self.R_out))
            plt.plot(lam, fluxdisk, "k-", label="Disk fit")
            plt.title("M:%.3f M$_{\\odot}$ R:%.3f R$_{\odot}$ M$_{acc}$:%.2f R_out: %.2f"%(self.Mstar, self.Rstar, self.logMacc, self.R_out))    


        ymin, ymax = plt.ylim()
        #plt.ylim(np.max([ymin, np.min(self.fluxes)*0.01]), ymax)
        plt.xlabel("Wavelength [$\\AA$]")
        plt.ylabel("log Flux")
        plt.ylim(ymin=np.min(self.fluxes) * 0.1)
        plt.yscale("log")
        plt.legend()
        name = self._get_save_path(None, "mcmc_best_fit_model")
        plt.savefig(name)
        plt.close("all")
                    

    def write_fit_params(self):
        '''
        Write the best fit parameters of the model to the standard output.
        '''
        
        if self.model.startswith("BlackBody"):
        
            #Prints the best parameters
            print ('''
                        Temperature:    %.3f -%.3f +%.3f K
                        Radius:         %.3f -%.3f +%.3f R$_{\odot}$
                        Luminosity       %.3e -%.3e +%.3e L$_{\odot}$'''%(\
            self.T, self.Terr1, self.Terr2, \
            self.R, self.Rerr1, self.Rerr2, \
            self.L, self.Lerr1, self.Lerr2))
            
        
        if self.model == "BlackBody_Av":
            print ("            Av:    %.1f -%.1f +%.1f K"%(self.Av, self.Averr1, self.Averr2))
            
        if self.model == "BlackBody2":
            print ("                        Temperature2:    %.1f -%.1f +%.1f K"%(self.Tsec, self.Tsecerr1, self.Tsecerr2))
            print ("                        Radius2:         %.1f -%.1f +%.1f R$_{\odot}$"%(self.Rsec, self.Rsecerr1, self.Rsecerr2))        
        
        if self.model == "BlackBody2_Av":
            print ("                        Av:    %.1f -%.1f +%.1f K"%(self.Av, self.Averr1, self.Averr2))
            print ("                        Temperature2:    %.1f -%.1f +%.1f K"%(self.Tsec, self.Tsecerr1, self.Tsecerr2))
            print ("                        Radius2:         %.1f -%.1f +%.1f R$_{\odot}$"%(self.Rsec, self.Rsecerr1, self.Rsecerr2))
        
        if (self.model == "PowerLaw"):
            print ('''
                        alpha:    %.2f -%.2f +%.2f
                        Scale  :  %.2e -%.2e +%.2e
                        Av        %.2f -%.2f +%.2f'''%(\
            self.alpha, self.alphaerr1, self.alphaerr2, \
            self.scale, self.scaleerr1, self.scaleerr2, \
            self.Av, self.Averr1, self.Averr2))
            
        if (self.model == "PowerLaw_BlackBody"):
            print ('''
                        alpha:    %.2f -%.2f +%.2f
                        Scale (R):  %.2e -%.2e +%.2e
                        T        %.2f -%.2f +%.2f
                        R        %.2f -%.2f +%.2f '''%(\
            self.alpha, self.alphaerr1, self.alphaerr2, \
            self.scale, self.scaleerr1, self.scaleerr2,\
            self.T, self.Terr1, self.Terr2,\
            self.R, self.Rerr1, self.Rerr2    ))
            
        if (self.model == "Disk"):
            print ('''
                        Mstar:    %.3f$_{-%.3f}^{+%.3f}$
                        Rstar (10^8 cm):  %.3f -%.3f +%.3f
                        logMacc        %.3f$_{-%.3f}^{+%.3f}$
                        R_out        %.3f$_{-%.3f}^{+%.3f}$ '''%(\
            self.Mstar, self.Mstarerr1, self.Mstarerr2, \
            self.Rstar*(u.Rsun.to(u.cm))/1e8, self.Rstarerr1*(u.Rsun.to(u.cm))/1e8, self.Rstarerr2*(u.Rsun.to(u.cm))/1e8,\
            self.logMacc, self.logMaccerr1, self.logMaccerr2,\
            self.R_out, self.R_outerr1, self.R_outerr2 ))

