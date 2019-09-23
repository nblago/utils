# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:09:28 2018

@author: nadiablago
"""
import BBFit
import numpy as np
from matplotlib import pylab as plt
from astropy import units as u

    
def set_2bb_phot(bb):


    #Source photometry
    bb.bands = np.array(["omegacam,u", "omegacam,g", "omegacam,r",   \
    "omegacam,i", "paranal,J", "paranal,H"])
    bb.mags = np.array([19.46, 20.13,  19.5, 19.18, 18.58, 18.41])
    bb.magerrs = np.array([0.05, 0.03,  0.03, 0.04, 0.06, 0.1])
    bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag"])
    
    #Set some initial parameters for the fit
    bb.initT1 = 10000
    bb.initR1 = 5
    bb.initT2 = 5000
    bb.initR2 = 5
    
    #Distance in Mpc
    bb.distMpc = 2.3e-3
    #Fixed reddening inside of MW. It will be corrected before the fit.
    bb.av_mw = 1.116 #+0.538#-0.385
    #Initial guess for Av. It will only be used if the model is BlackBody_Av or BlackBody2_Av.
    bb.av_host = 0 


def set_disk_phot(bb):


    #Source photometry
    bb.bands = np.array(["omegacam,u", "omegacam,g", "omegacam,r",   \
    "omegacam,i", "paranal,J", "paranal,H"])

    bb.mags = np.array([19.46, 20.13,  19.5, 19.18, 18.58, 18.41])
    bb.magerrs = np.array([0.05, 0.03,  0.03, 0.04, 0.06, 0.1])
    bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag"])

    #Set some initial parameters for the fit
    bb.Mstar = 0.5
    bb.Rstar = 0.1
    bb.logMacc = -10.0
    bb.R_out = 10.0
    
    #Distance in Mpc
    bb.distMpc = 2.3e-3
    bb.av_mw = 1.116#+0.538#-0.385##/
    bb.av_host = 0 
    


def set_vega_phot(bb):
    '''
    Test the code with Vega photometry.
    The star parameters are:
        Radius	      2.362 - 2.818 Rsun
        Luminosity	      40.12 +/- 0.45 Lsun
        Temperature	 (8,152 - 10,060 K)
        Distance        7.68 pc
    The fit should give something close (although not exactly the same due to blanketing
    and other effects not accouned in the simple BB model).
    '''
    #Vega test photometry
    bb.bands =np.array(["B", "V", "R", "I", "J", "K"]) 
    bb.mags=np.array([0.03, 0.03, 0.07, 0.1, 0.02,	0.02])
    bb.magerrs = np.ones(6) * 0.01
    bb.photsys = np.repeat("vegamag", 6)

    bb.mjd = 0
    #Set the distance to the source and MW reddening
    bb.distMpc = 7.68e-6
    bb.av_mw = 0

    #Set some initial parameters for the fit
    bb.initT1 = 9000
    bb.initR1 = 1


#Create an instance of the python class
bb = BBFit.BBFit()


#Set to the photometric measurements and some parameters (distance, reddening)
#And the initial guess values of the model.
#set_2bb_phot(bb)
#set_disk_ophot(bb)
set_vega_phot(bb)

#Set the MCMC parameters. 
#This values are just an quick check that the code works. 
#Set the iteration to a larger number (50,000) and birning (5,000) for more precise results.
bb.niterations = 3000
bb.burnin = 1500
bb.nwalkers = 20

#Set the plotting directory
bb.plotdir = "../../data/plots" 
bb.resdir = "../../data/modelfits" 

#Select from: "BlackBody", "BlackBody_Av", "BlackBody2_Av", "PowerLaw", "PowerLaw_BlackBody", "Disk"
bb.model = "BlackBody"


#Initialize . If plot=True, it will also plot the model with the initial guess parameters.    
bb.initialize(plot=True)

#Runs the MCMC and saves the best parameters to a results file within bb.resdir.
bb.run()

#Plot the best fit
bb.plot_fit()
#Plot the posterior distribution
bb.plot_corner_posteriors()
#Writes the best fit parameters to the standard output.
bb.write_fit_params()