# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:09:28 2018

@author: nadiablago
"""
from model import BBFit
import numpy as np
from matplotlib import pylab as plt

def set_nova_phot(bb):
    
    #Set the photometry
    #bb.bands = np.array(["B", "V", "R", "I", "Y", "J", "H", "K"])
    bb.bands = np.array(["B", "V", "I", "Y", "J", "H", "K"])

    #JD 2458203.6
    #bb.mags = np.array([6.995, 6.004, 5.290, 4.832, 4.906, 4.71, 4.33, 3.90])
    #bb.magerrs = np.array([0.13, 0.09, 0.1, 0.125, 0.04, 0.1, 0.1, 0.1, 0.1])
    
    #JD 2458211.6
    #
    bb.mags = np.array([7.212, 6.587, 5.285, 5.108, 4.827, 4.523, 4.083])
    bb.magerrs = np.array([0.13, 0.09, 0.125, 0.04, 0.1, 0.1, 0.1, 0.1])
    bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag"])

    #Day the photometry was taken
    bb.mjd = 58211.1
    bb.distMpc = 2.3e-3
    bb.av_mw = 0
    bb.av_host = 1
    
    #Set some initial parameters for the fit
    bb.initT1 = 10000
    bb.initR1 = 1
    
    bb.initT2 = 5000
    bb.initR2 = 1
    
def set_last_nova_phot():

    #aperture = 1.5 * FWHM

    bb.bands = np.array(["u,sdss", "g,sdss", "i,sdss"])  #"r,sdss"
    bb.mags = np.array([10.686, 10.610, 10.5075]) #9.8342
    bb.magerrs = np.array([0.08, 0.05, 0.022]) #0.000410
    bb.photsys = np.array(["abmag", "abmag", "abmag"])
    


    #Day the photometry was taken
    bb.mjd = 58550.1
    bb.distMpc = 2.3e-3
    bb.av_mw = 0.5
    bb.av_host = 0
    
    #Set some initial parameters for the fit
    bb.initT1 = 10000
    bb.initR1 = 1
    
    bb.initT2 = 5000
    bb.initR2 = 1
    
    bb.model = "BlackBody2"    
    
    
def set_progenitor_phot(bb):


    #Progenitor photometry
    bb.bands = np.array(["omegacam,u", "omegacam,g", "omegacam,Halpha", "omegacam,r", "omegacam,i", "paranal,J", "paranal,H", "paranal,Ks"])
    bb.mags = np.array([20.67, 19.75, 18.02, 18.39, 17.59, 16.25, 15.69, 15.53] )
    bb.magerrs = np.array([0.09, 0.02, 0.03, 0.02, 0.02, 0.05, 0.049, 0.061])
    bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag"])
        
    #Set some initial parameters for the fit
    bb.initT1 = 10000
    bb.initR1 = 5
    bb.initT2 = 5000
    bb.initR2 = 5
    
    bb.distMpc = 2.3e-3
    bb.av_mw = 0.5
    bb.av_host = 0 #0.5

def set_progenitor_powerlaw(bb):


    #Progenitor photometry
    bb.bands = np.array(["omegacam,u", "omegacam,g", "omegacam,Halpha", "omegacam,r", "omegacam,i", "paranal,J", "paranal,H", "paranal,Ks"])
    bb.mags = np.array([20.67, 19.75, 18.02, 18.39, 17.59, 16.25, 15.69, 15.53] )
    bb.magerrs = np.array([0.09, 0.02, 0.03, 0.02, 0.02, 0.05, 0.049, 0.061])
    bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag"])
    
    bb.distMpc = 2.3e-3
    bb.av_mw = 0
    bb.model = "PowerLaw"


    #Set some initial parameters for the fit
    bb.alpha = 3./4
    bb.initR1 = 1
    bb.av_host = 3
    

    
def set_vega_phot(bb):
    
    #Vega test photometry
    bb.bands =np.array(["B", "V", "R", "I"]) 
    bb.mags=np.zeros(4) + 0.03
    bb.magerrs = np.ones(4) * 0.01
    bb.photsys = np.repeat("vegamag", 4)

    bb.mjd = 0
    #Set the distance to the source and MW reddening
    bb.distMpc = 7.68e-6
    bb.av_mw = 0

    #Set some initial parameters for the fit
    bb.initT1 = 10000
    bb.initR1 = 1
    
#Create an instance of the python class
bb = BBFit.BBFit()


#Set to the right photometric measurements
#set_progenitor_powerlaw(bb)
#set_progenitor_phot(bb)
#set_nova_phot(bb)
set_last_nova_phot()

#Set the MCMC parameters
bb.niterations = 1000
bb.burnin = 10
bb.nwalkers = 20

#Set the plotting directory
bb.plotdir = "data/plots/" 

bb.model = "BlackBody2"
#bb.model = "PowerLaw"

#Initialize     
bb.initialize(plot=True)

#Run
bb.run()


#Plot the best fit
bb.plot_fit()

#Plot the posterirs with the labels that we assigned to each parameter.
if bb.model == "BlackBody2":
    bb.plot_corner_posteriors(labels=['T1', 'R1', "T2", "R2"])
elif bb.model.startswith("BlackBody"):
    bb.plot_corner_posteriors(labels=['T1', 'R1', "A_v", "T2", "R2"])
    
#Write the best fit parameters
if bb.model.startswith("BlackBody"):

    #Prints the best parameters
    print ('''
                Temperature:    %.1f -%.1f +%.1f K
                Radius:         %.1f -%.1f +%.1f R$_{\odot}$
                Luminosity       %.1e -%.1e +%.1e L$_{\odot}$'''%(\
    bb.T,bb.Terr1, bb.Terr2, \
    bb.R, bb.Rerr1, bb.Rerr2, \
    bb.L, bb.Lerr1, bb.Lerr2))

if bb.model == "BlackBody_Av":
    print ("            Av:    %.1f -%.1f +%.1f K"%(bb.Av, bb.Averr1, bb.Averr2))
    
if bb.model == "BlackBody2_Av":
    print ("                Av:    %.1f -%.1f +%.1f K"%(bb.Av, bb.Averr1, bb.Averr2))
    print ("                Temperature2:    %.1f -%.1f +%.1f K"%(bb.Tsec,bb.Tsecerr1, bb.Tsecerr2))
    print ("                Radius2:         %.1f -%.1f +%.1f R$_{\odot}$"%(bb.Rsec, bb.Rsecerr1, bb.Rsecerr2))

if bb.model == "BlackBody2":
    print ("                Temperature2:    %.1f -%.1f +%.1f K"%(bb.Tsec,bb.Tsecerr1, bb.Tsecerr2))
    print ("                Radius2:         %.1f -%.1f +%.1f R$_{\odot}$"%(bb.Rsec, bb.Rsecerr1, bb.Rsecerr2))


if (bb.model == "PowerLaw"):
    
    bb.plot_corner_posteriors(labels=['alpha', 'scale', "A_v"])

    print (    bb.alpha, bb.alphaerr1, bb.alphaerr2, \
    bb.R, bb.Rerr1, bb.Rerr2, \
    bb.Av, bb.Averr1, bb.Averr2)
    
    print ('''
                alpha:    %.1f -%.1f +%.1f
                Scale (R):  %.2f -%.2f +%.2f
                Av       %.1e -%.1e +%.1e L$_{\odot}$'''%(\
    bb.alpha, bb.alphaerr1, bb.alphaerr2, \
    bb.R, bb.Rerr1, bb.Rerr2, \
    bb.Av, bb.Averr1, bb.Averr2))