# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:09:28 2018

@author: nadiablago
"""
import BBFit
import numpy as np

#Create an instance of the python class
bb = BBFit.BBFit()

#Set some initial parameters for the fit
bb.initT1 = 3000
bb.initA1 = 25

#Set the photometry
bb.bands = np.array(["B", "V", "R", "I", "Y", "J", "H", "K"])
bb.mags = np.array([6.995, 6.004, 5.290, 4.832, 4.906, 4.71, 4.33, 3.90])
bb.magerrs = np.array([0.13, 0.09, 0.1, 0.125, 0.04, 0.1, 0.1, 0.1, 0.1])
bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag", "vegamag"])


#Day the photometry was taken
bb.mjd = 58205.64

#Set the distance to the source and MW reddening
bb.distMpc = 2.3e-3
bb.av_mw = 1.9

#Set the MCMC parameters
bb.niterations = 2000
bb.burnin = 500
bb.nwalkers = 8

#Set the plotting directory
bb.plotdir = "/Users/nadiablago/Desktop" 

#Initialize     
bb.initialize(plot=True)

#Run
bb.run()

#Plot the posterirs
bb.plot_corner_posteriors()

#Plot the best fit
bb.plot_fit()

#Prints the best parameters
print ('''
            Temperature:    %.1f -%.1f +%.1f K
            Radius:         %.1f -%.1f +%.1f R$_{\odot}$
            log(Area/cm2):  %.2f -%.2f +%.2f
            Luminosity       %.1e -%.1e +%.1e L$_{\odot}$'''%(\
bb.T,bb.Terr1, bb.Terr2, \
bb.R, bb.Rerr1, bb.Rerr2, \
bb.A, bb.Aerr1, bb.Aerr2, \
bb.L, bb.Lerr1, bb.Lerr2))
