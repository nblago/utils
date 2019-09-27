# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:19:04 2016

@author: nadiablago
"""

import os
import shutil
import subprocess
import numpy as np



SEXCONFIG = os.path.abspath(os.path.join('../config/daofind.sex'))

if os.path.isfile(SEXCONFIG):
    print ('Found sextractor configuration file: %s'%SEXCONFIG)
else:
    print ('Please, put the sextractor configuration file daofind.sex in: %s'%(os.path.dirname(SEXCONFIG)))
    print ('Otherwise, redefine the environmental variable SEXCONFIG to point to the location where daofind.sex is located.')

    
def run_sex(flist, overwrite=False):
    
    d = os.path.dirname(flist[0])
    if d == "":
        d = "."
    os.chdir(d)

    #Create the directory where the sextracted images are going to go.
    sexdir = os.path.join(d, "sextractor")    
    if (not os.path.isdir(sexdir)):
        os.makedirs(sexdir)
        
    newlist = []
    for f in flist:
        newimage = os.path.join(sexdir, os.path.basename(f).replace(".fits", ".sex")) 

        if (os.path.isfile(newimage) and not overwrite):
            newlist.append(newimage)
            print ("Sextracted image %s already exists."%newimage)
        else:
            try:
                out = os.path.abspath(f)
    
                cmd="sex -c %s %s"%(SEXCONFIG, out) 
                subprocess.call(cmd, shell=True)
                shutil.move("image.sex", newimage)
                newlist.append(newimage)
            except IOError:
                print ("IOError detected reading file",f)
                pass
        
    return newlist


    
def analyse_image(sexfile, arcsecpix):
    '''
    Analyses the sextractor filelist to determine the best focus.
    If FWHM in pixes is required, set arcsecpix=1
    
    returns: A tuple containing:
        1. - Number of extracted sources.
        2. - FWHM in arcsecs.
        3. - Ellipticity.
        4. - Background

	#   1 X_IMAGE                Object position along x                                    [pixel]
	#   2 Y_IMAGE                Object position along y                                    [pixel]
	#   3 ALPHA_J2000            Right ascension of barycenter (J2000)                      [deg]
	#   4 DELTA_J2000            Declination of barycenter (J2000)                          [deg]
	#   5 MAG_BEST               Best of MAG_AUTO and MAG_ISOCOR                            [mag]
	#   6 MAGERR_BEST            RMS error for MAG_BEST                                     [mag]
	#   7 FWHM_WORLD             FWHM assuming a gaussian core                              [deg]
	#   8 FWHM_IMAGE             FWHM assuming a gaussian core                              [pixel]
	#   9 ELLIPTICITY            1 - B_IMAGE/A_IMAGE                                       
	#  10 BACKGROUND             Background at centroid position                            [count]
	#  11 FLAGS                  Extraction flags   	
	#  12 A_IMAGE                Isophotal image mejor axis
	#  13 B_IMAGE                Isophotal image minor axis
   	#  14 THETA_IMAGE            Isophotal image position angle
   	#  15 PETRO_RADIUS           Petrosian radius
 
    '''
    


    s = np.genfromtxt(sexfile, comments="#", dtype=[("x", np.float), ("y", np.float), ("ra", np.float), ("dec", np.float), \
        ("mag", np.float), ("magerr",np.float), ("fwhm_world", np.float), ("fwhm_image", np.float), ("ellipticity",np.float), \
        ("background", np.float), ("flags", np.float), ("a_image", np.float), ("b_image", np.float),("theta_image", np.float), ("petro_radius", np.float)])

    if (s is None or s.ndim==0 or len(s)==0):
        print ("Empty content of the file for file %s. The length of the file is %d"%(sexfile, len(s)))
        return 0,0,0,0
        
    # Select with good flags only.
    s = s[s["flags"]==0]

    nsources = len(s) 
    if (nsources == 0):
        return 0,0,0,0
    #Select round sources (ellipticity is 1-axis_ratio)
    s = s[s["ellipticity"]<0.3]
    ellipticity = np.nanmedian(s["ellipticity"])

    #Select FWHM at least 3 pixels and lower than 15 arcsec
    s = s[ (s["fwhm_image"]>3)*(s["fwhm_image"]*arcsecpix<15)]
    
    nsources = len(s) 
    if (nsources == 0):
        return 0,0,0,0
        
    #Select bright magnitudes
    s = s[s["mag"]<np.percentile(s["mag"], 20)]
       
    fwhm = np.nanmedian(s["fwhm_image"]*arcsecpix)
    bkg = np.nanmedian(s["background"])
    
    return nsources, fwhm, ellipticity, bkg
        
    
def get_image_pars(image, arcsecpix):
    '''
    Returns a set of statistics for a given image.
    '''
    sexfiles = run_sex([image])
    pars = analyse_image(sexfiles[0], arcsecpix=arcsecpix)
    
    return pars[0], pars[1], pars[2], pars[3]

