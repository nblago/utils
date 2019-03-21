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
import os
import numpy as np
import emcee
import scipy
from scipy import stats
import extinction
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import curve_fit
import warnings


class BBFit:            
    
    def __init__(self):
        '''
        Constructor initializes all the parameters to defaults.
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

        #Location for plots
        self.plotdir = "data/plots"
        
        #MCMC parameters
        self.method = 'ensemble' #or HA for Hastings
        self.mhtune = True # tuning of the Metropolis-Hastings 
        self.niterations = 10000
        self.burnin = 5000
        self.threads = 10
        self.nwalkers = 20
        self.sampler = None
        self.model = "BlackBody" #others are "BlackBody_Av" or "BlackBody2_Av"

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
        
        #Add the environment variable which points to the filter response files for the bands we are interested in.
        if not 'PYSYN_CDBS' in os.environ.keys():
            print ("Adding the Pysynphot environment:")
            os.environ['PYSYN_CDBS'] = "/Users/nadiablago/Documents/Software/pysynphot_files"
        print ('PYSYN_CDBS environment variable set to: ', os.environ['PYSYN_CDBS'])
    
        self.banddic = {"Y": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/ctio_y_andicam.dat"),
                    "J": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_j_004_syn.fits"),
                   "H": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_h_004_syn.fits"),
                   "K": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/bessell_k_004_syn.fits"),
                   "3.6": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac1_3.6.txt"),
                   "4.5": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac2_4.5.txt"),
                   "5.8": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac3_5.8.txt"),               
                   "8.0": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Spitzer_irac4_8.0.txt"),
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
                   "omegacam,Halpha":  os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/Paranal_OmegaCAM.Halpha.dat")
           }


    def _matplotlib_init(self):

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
        sp = ps.BlackBody(10000)
        sp.convert('flam')

        
        for b, m, me, psys in zip(self.bands, self.mags, self.magerrs, self.photsys):
            
            #Create the observation bandpass
            try:
                band = ps.ObsBandpass(b)
            except ValueError:
                #The band is not in the standard list
                #We need to go to the dictionary to retrieve the transmission function.
                band = ps.FileBandpass(self.banddic[b])
                #band.waveunits = ps.units.Angstrom
            
            #Oftain the effective (average) wavelength
            effwave = band.avgwave()
            wls = np.append(wls, effwave)
            
            #Correct for Milky Way extinction
            m = m - extinction.fitzpatrick99(np.array([effwave]), a_v=self.av_mw, unit='aa')[0]
            
            #Normalize the spectrum to the magnitude of the observation
            sp_norm = sp.renorm(m, psys, band)
            #Observe with the band
            obs = ps.Observation(sp_norm, band)
            #Get the flux
            flux = obs.effstim('flam')
            fluxes = np.append(fluxes, flux) 
            
            #Compute the error bars
            flux_high = flux * 10**(0.4*me)
            flux_low = flux * 10**(-0.4*me)
            
            fluxerr = np.append(fluxerr, np.average([flux - flux_low, flux_high-flux]))

        return wls, fluxes, fluxerr
    
           
    def _model(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
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

    def _model_simple(self, lam, T, R):
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
        
    def _model_av(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        T = p[0] * u.K
        R = (p[1] * u.Rsun).to(u.cm)
        a_v = p[2]

        #Compute the effect of reddening as a flux factor
        flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam, a_v, unit='aa'))
        lam = lam * u.Angstrom
        

        area = np.pi * (4 * np.pi * R**2)
        flam =  area * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T))-1)
        
        #Apply the reddening
        flam = flam.to(u.erg/u.s/u.Angstrom).value * flux_red
        
                
        return flam
        
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
        R1 = p[1]
        a_v = p[2]
            
        f = ps.PowerLaw(w0, alpha)
        f.convert('flam')
        
        flam = np.interp(lam, f.wave, f.flux)
        
        flux_red =  10**(-0.4 * extinction.fitzpatrick99(lam, a_v, unit='aa'))
        area = 10**R1

        return area * flam * flux_red #.to(u.erg/u.s/u.Angstrom).value

    def _model_powerlaw_2(self, lam, alpha, R1, a_v):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''

        return self._model_powerlaw(lam, (alpha, R1, a_v))
        
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
        lp= self._like(p, xdat, ydat, errdat) + self._logprior(p)
        return lp
    
    
    def _logprior(self, p):
        '''
        Creates the priors for each initial parameter
        
        T: temperature
        A: log area (in cm2)
        fraction: fraction of outliers in our datapoints.
        '''
        
        if self.model == "BlackBody" or self.model =="BlackBody_Av" or self.model == "BlackBody2_Av" or self.model == "BlackBody2":
            
            T1 = p[0] 
            R1 = p[1] 
    
            if T1 < 0:
                return -np.inf
    
            # normal mean= 0 , sig = 2
            logp = stats.norm.logpdf(T1, 10000, 10000)
            #logp = stats.uniform.logpdf(T1, 500, 70000)
    
            #uniform from the initial value until 3 orders of mangitude up and down
            #logp = logp + stats.uniform.logpdf(R1, 0,  1000)
            logp = logp + stats.gamma.logpdf(R1, 1, loc=0, scale=50)
        
        if self.model =="BlackBody_Av":
            av = p[2]
            if av < 0 or av > 8:
                logp = -np.inf
            else:
                logp = logp + stats.norm.logpdf(av, self.av_host, self.av_host)
        elif self.model == "BlackBody2_Av":
            av = p[2]
            T2 = p[3]
            R2 = p[4]

            if T1 < 0 or T2 > T1 or T2 < 0 or av < 0 or av > 10:
                return - np.inf

            else:
                logp = logp + stats.uniform.logpdf(av, 0, 3.6)

                #logp = logp + stats.uniform.logpdf(T2, 1000, 50000)
                logp = logp + stats.norm.logpdf(T2, 10000, 10000)

                logp = logp + stats.gamma.logpdf(R2, 1, loc=0, scale=50)

        elif self.model == "BlackBody2":
            T2 = p[2]
            R2 = p[3]

            if T1 < 0 or T2 > T1 or T2 < 0:
                return - np.inf

            else:
                #logp = logp + stats.uniform.logpdf(T2, 1000, 50000)
                logp = logp + stats.norm.logpdf(T2, 10000, 10000)

                logp = logp + stats.gamma.logpdf(R2, 1, loc=0, scale=100)
                
        elif self.model == "PowerLaw":
            alpha = p[0]
            R1 = p[1] 
            av = p[2]
            
            logp = stats.uniform.logpdf(alpha, -3, 3)
            logp = logp + stats.norm.logpdf(R1, 45, 10)
            
            if av < 0 or av > 10:
                logp = -np.inf
            else:
                logp = logp + stats.uniform.logpdf(av, 0, 3.6)
        return logp	

    def _get_max_and_intervals(self, x):
        '''
        Provided a chain of samples, finds the average value and returns the values
        for a 1 sigma distribution.
        
        '''
        '''h = np.histogram(x, density=True, bins=51)
        dx= h[1][1]-h[1][0]
        centx = (h[1][1:] + h[1][0:-1])/2.
        maxp = centx[np.argmax(h[0])]
        lower = h[0][centx<=maxp]
        upper = h[0][centx>maxp]
        sum1 = np.cumsum(lower[::-1]*dx)
        sum2 = np.cumsum(upper*dx)
        
        try:
            percent1 = np.interp(0.34, sum1, h[1][centx<=maxp][::-1])
        except:
            percent1 = np.min(x)
        try:    
            percent2 = np.interp(0.34, sum2, h[1][centx>maxp])
        except:
            percent2 = np.max(x)'''

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
        Transforms the log area into a radius.
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
        else:
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
                
    def  initialize(self, plot=False):
        '''
        Will transform the magnitudes to fluxes and use the distance to the object to
        calculate the luminosity at each wavelength.
        '''

        if (not os.path.isdir(self.plotdir)):
            os.makedirs(self.plotdir)

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
            name = self._get_save_path(None, "fluxes_obs")
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


    def _initialize_parameters(self, plot=True):
        '''
        Runs the least squares optimiztion routine to find the best initial parameters 
        to start the MCMC with.
        '''

        lam = np.linspace(3000, 25000, 2000)
        
        if self.model == "BlackBody2_Av":
            flux_ini = self._model2_av_r_2(lam, self.initT1, self.initR1, self.av_host, self.initT2, self.initR2)

            p0 = (self.initT1, self.initR1, self.av_host, self.initT2, self.initR2)
            
            print ("Initial ", p0, "Fluxerrs", self.fluxes/self.fluxerrs)
            
            params, covar = curve_fit(self._model2_av_r_2, self.wls , self.fluxes, \
            p0 = p0, sigma=self.fluxerrs, absolute_sigma=True, maxfev = 20000)
            
            self.initT1 = params[0]
            self.initR1 = params[1]
            self.av_host = params[2]
            self.initT2 = params[3]
            self.initR2 = params[4]

            print ("LSQ fit ", params)
            
            if plot:
                flux_end = self._model2_av_r_2(lam, *params)
                
                plt.clf()
                plt.plot(lam, flux_ini, "r--", label="initial parameters")
                plt.plot(lam, flux_end, label="Best fit LSQ")
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs")
                plt.savefig(name, dpi=200)
                
        if self.model == "PowerLaw":

            #params, covar = curve_fit(self._model_powerlaw_2, self.wls , self.fluxes, \
            #p0=(self.alpha, self.initR1, self.av_host), sigma=self.fluxerrs, absolute_sigma=True, maxfev = 10000)
            
            #alpha = params[0]
            #R = params[1]
            #av = params[2]
            
            #self.alpha = alpha
            #self.R1 = R
            #self.av_host = av
                        
            if plot:
                lam = np.linspace(3000, 25000, 2000)
                fluxpw = self._model_powerlaw_2(lam, self.alpha, self.initR1, self.av_host)
                
                plt.clf()
                plt.plot(lam, fluxpw, label="Best fit %s"%self.model)
                plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o", lw=0, label="Measurements")
                plt.xlabel("Wavelength [A]")
                plt.ylabel("$L_{\\lambda}$ [erg/s/A]")
                plt.legend()
                name = self._get_save_path(None, "fluxes_obs_LSQ_fit")
                plt.savefig(name, dpi=200)
                print ("Saved fit as %s"%name)
    
    
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
            sigs = np.array([2000, 10, 2]) 
        elif self.model == "BlackBody2":
            p0 = np.array([ self.initT1, self.initR1, self.initT2, self.initR2])
            sigs = np.array([5000, 5, 5000, 5]) 
        elif self.model == "BlackBody2_Av":
            p0 = np.array([ self.initT1, self.initR1, self.av_host, self.initT2, self.initR2])
            sigs = np.array([5000, 5, 1,  5000, 5])   
        elif self.model == "PowerLaw":
            p0 = np.array([ self.alpha, self.initR1, self.av_host])
            sigs = np.array([2, 10, 2])
            
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


        
    def plot_corner_posteriors(self, savefile=None, labels=["T1", "R1", "Av", "T2", "R2"]):
        '''
        Plots the corner plot of the MCMC results.
        '''
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
        
        lam = np.linspace( np.min(self.wls) -1000 , np.max(self.wls) + 500, 1000)
        
        plt.clf()
        plt.figure(figsize=(8,6))
        plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, fmt="o")
        for i in range(len(self.wls)):
                plt.text(self.wls[i], self.fluxes[i]*1.01, self.bands[i].split(",")[-1], alpha=.4)
        
        if self.model == "BlackBody":
            fluxbb = self._model(lam, (self.T, self.R))
            plt.plot(lam, fluxbb, "k-", label="BB fit")
            plt.title("T: %.1f K R:%.1f R$_{\odot}$ Lumiosity %.1e L$_{\odot}$"%(self.T, self.R, self.L))    

        elif self.model == "BlackBody_Av":
            fluxbb = self._model(lam, (self.T, self.R))
            fluxbb_red = self._model_av(lam, (self.T, self.R, self.Av))
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
            
            plt.title("T: %.1f K R:%.1f R$_{\odot}$ Lumiosity %.1e L$_{\odot}$ T2: %.1f R2: %.1f"%(self.T, \
                      self.R, self.L, self.Tsec, self.Rsec)) 
                      
                      
        elif self.model == "PowerLaw":
            flux = self._model_powerlaw(lam, (self.alpha, self.R, self.Av))
            plt.plot(lam, flux, "k-", label="PowerLaw + reddening")
            plt.title("$\\alpha$: %.1f Av: %.2f"%(self.alpha, self.Av))    

         
        plt.xlabel("Wavelength [$\\AA$]")
        plt.ylabel("log Flux")
        #plt.ylim(ymin=np.min(self.fluxes) * 0.8)
        plt.yscale("log")
        plt.legend()
        name = self._get_save_path(None, "mcmc_best_fit_model")
        plt.savefig(name)
        plt.close("all")
                    

    


