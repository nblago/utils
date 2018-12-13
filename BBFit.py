# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:57:34 2018

Class that enables to fit a black body function to a set of magntidues.

@author: nadiablago
"""

from __future__ import print_function


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


class BBFit:    

    
    def __init__(self):
        '''
        Constructor initializes all the parameters to defaults.
        '''
        
        #Source parameters
        self.av_host = 0
        self.av_mw = 0
        self.law = "Fitzpatrick"
        self.law_mw = "Fitzpatrick"
        self.initT1 = 10000 #K
        self.initA1 = 30 # log10(cm2)
        self.z = None
        self.distMpc = None #in Mpc
        self.mjd = 0 

        #Location for plots
        self.plotdir = "."
        
        #MCMC parameters
        self.method = 'ensemble' #or HA for Hastings
        self.mhtune = True # tuning of the Metropolis-Hastings 
        self.niterations = 10000
        self.burnin = 5000
        self.threads = 10
        self.nwalkers = 8
        self.sampler = None

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
        
        self.A = None
        self.Aerr1 = None
        self.Aerr2 = None
        
        self.R = None
        self.Rerr1 = None
        self.Rerr2 = None
        
        self.L = None
        self.Lerr1 = None
        self.Lerr2 = None

        self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

        
        #Add the environment variable which points to the filter response files for the bands we are interested in.
        if not 'PYSYN_CDBS' in os.environ.keys():
            print ("Adding the Pysynphot environment:")
            os.environ['PYSYN_CDBS'] == "/Users/nadiablago/Documents/Software/pysynphot_files"
    
        self.banddic = {"Y": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/ctio_y_003.dat"),
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
                   "swift,v": os.path.join(os.environ['PYSYN_CDBS'], "comp/nonhst/swift_v_uvot.dat")
           }
               
       
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
            m = m - extinction.fitzpatrick99(np.array([effwave]), a_v=self.av_mw, r_v=3.1, unit='aa')[0]
            
            #Normalize the spectrum to the AB mangitude of the observation
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
        logA = p[1] 

        Area = 10**logA * u.cm**2
        Blam =  Area * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T))-1)
        
        return Blam.to(u.erg/u.s/u.Angstrom).value

    def _model_av(self, lam, p):
        '''
        Return units: erg s-1 A-1
        As we multiply by the area of the emitting source (in cm**2)
        '''
        
        lam = lam * u.Angstrom
        
        T = p[0] * u.K
        logA = p[1] 
        av = p[2]

        area = 10**logA * u.cm**2
        flam =  area * (2*cnt.h*((cnt.c).to(u.cm/u.s))**2/( (lam.to(u.cm))**5))/ \
            (np.exp((cnt.h*cnt.c)/(lam.to(u.m)*cnt.k_B*T))-1)
        
        flam = flam.to(u.erg/u.s/u.Angstrom).value
        
        ext = extinction.fitzpatrick99(lam, av, unit='aa')
        
        return 
        
    #likelihood function
    def _like(self, p, xdat, ydat, errdat, debug=False):
        '''
        p: function parameters 
        args: carry anything we want to pass to our function (e.g. the data) 
        '''   
        
        ymod = self._model(xdat, p)
        
         
        #In case we want to cosider some outlier fraction as a parameter of the MCMC
        #outlierfrac = p[-1] # fraction of dataset in outliers
        #if outlierfrac < 0:
        #    return np.log(1e-320)
        #if outlierfrac >= 1:
        #    return np.log(1e-320)
        #prob = ( (1 - outlierfrac) * stats.norm.pdf(ydat, ymod, errdat) + outlierfrac * stats.norm.pdf(ydat, ymod, errdat*100) )	
        
        #Discard models which exceed the upper limits
        if (np.any(ymod[errdat<0] > ydat[errdat<0])):
            prob = 1e-320
        else:
            prob = stats.norm.pdf(ydat, ymod, errdat) 
    
        # log probabilities
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
        
        T = p[0] 
        A = p[1] 

        #if T<0:
        #    return np.log(1e-323)

        # normal mean= 0 , sig = 2
        logp = stats.norm.logpdf(T, loc=self.initT1, scale=10000)
        #logp = stats.uniform.logpdf(T, 500, 150000)

        #uniform from the initial value until 3 orders of mangitude up and down
        logp = logp + stats.uniform.logpdf(A, 20, 40)
        #logp = logp + stats.norm.logpdf(A, self.initA1, 10)
        
        #fraction = p[2] #uniform from 0 to 0.05
        #logp = logp + stats.uniform.logpdf(fraction,0,.05)

    
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
        T1, T, T2 =  self._get_max_and_intervals(self.sampler.flatchain[:,0])
        A1, A, A2 =  self._get_max_and_intervals(self.sampler.flatchain[:,1])
        
        R = self._area2rsun(10**A) #Rsun
        R1 = self._area2rsun(10**A1)
        R2 = self._area2rsun(10**A2)
        
        self.T = T
        self.Terr1 = T - T1
        self.Terr2 = T2 - T
        
        self.A = A
        self.Aerr1 = A - A1
        self.Aerr2 = A2 - A
             
        self.R = R
        self.Rerr1 = R - R1
        self.Rerr2 = R2 - R
        
        self.L = self._get_bol_lum(T, R)
        self.Lerr1 = self.L - self._get_bol_lum(T1, R1)
        self.Lerr2 = self._get_bol_lum(T2, R2) - self.L
        

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
            name = os.path.join(self.plotdir, "%s_%.1f_%d.png"%(plot_name, self.mjd, i))

            while (os.path.isfile(name)):
                i = i+1
                name = os.path.join(self.plotdir, "%s_%.1f_%d.png"%(plot_name, self.mjd, i))
                
        return name
                
    def  initialize(self, plot=False):
        '''
        Will transform the magnitudes to fluxes and use the distance to the object to
        calculate the luminosity at each wavelength.
        '''
        # generate the data     
        self.wls, self.fluxes, self.fluxerrs = self._band2flux()


        if not self.distMpc is None:
            print ("Using distance to the source of %.1e Mpc"%self.distMpc)
            fluxFactor = (4*np.pi*((self.distMpc*u.Mpc).to(u.cm) )**2).value

        elif self.distMpc is None and not self.z is None:
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

        if (plot):
            plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, marker="o")
            name = self._get_save_path(None, "fluxes_obs")
        
    def run(self):
        '''
        Runs the main MCMC process. 
        Retrieves the priors, the likelihood process and computes the posterior probability.
        '''
    
        xs = self.wls
        ys = self.fluxes
        errs = self.fluxerrs
        
        p0 = np.array([ self.initT1, self.initA1])
        sigs = np.array([self.initT1*0.2, 0.2])
    	
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

        self._fill_output()


        
    def plot_corner_posteriors(self, savefile=None, labels=["T", "log(Area)"]):
        '''
        Plots the corner plot of the MCMC results.
        '''
        ndim = len(self.sampler.flatchain[0,:])
        chain = self.sampler
        samples = chain.flatchain
        
        samples = samples[:,0:2]    
        fig = corner.corner(samples, labels=labels[0:2])
        plt.title("MJD: %.2f"%self.mjd)
        name = self._get_save_path(savefile, "mcmc_posteriors")
        plt.savefig(name)
        plt.close("all")
        
        #self.sampler.chain.shape = (nwalkers, niterations, ndim)
        chain0 = self.sampler.chain[:,:,0]
        chain1 = self.sampler.chain[:,:,1]

        nwalk, nit = chain0.shape
        
        plt.subplot(2,1,1)
        for i in np.arange(nwalk):
            plt.plot(chain0[i], 'r-', lw=0.1)
            plt.ylabel(labels[0])
            plt.xlabel("Iteration")
            
        plt.subplot(2,1,2)
        for i in np.arange(nwalk):
            plt.plot(chain1[i], 'b-', lw=0.1)
            plt.ylabel(labels[1])
            plt.xlabel("Iteration")
            
        name_walkers = self._get_save_path(savefile, "mcmc_walkers")
        plt.tight_layout()
        plt.savefig(name_walkers)
        plt.close("all")  
        
            

    def plot_fit(self):
        '''
        Plots the best fit model to the data.
        '''
        
        lam = np.linspace( np.min(self.wls) -500 , np.max(self.wls) + 500, 100)
        
        plt.clf()
        plt.figure(figsize=(8,6))
        plt.errorbar(self.wls, self.fluxes, yerr=self.fluxerrs, fmt="o")
        fluxbb = self._model(lam, (self.T, self.A))
        plt.plot(lam, fluxbb, "k-", label="BB fit")
        plt.title("T: %.1f K R:%.1f R$_{\odot}$ Lumiosity %.1e L$_{\odot}$"%(self.T, self.R, self.L))    
        plt.xlabel("Wavelength [$\\AA$]")
        plt.ylabel("log Flux")
        plt.yscale("log")
        plt.legend()
        name = self._get_save_path(None, "mcmc_best_fit_model")
        plt.savefig(name)
        plt.close("all")
                    

    


