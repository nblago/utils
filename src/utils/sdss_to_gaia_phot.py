'''
Created on June, 29 2013

@author: Nadejda Blagorodnova <nblago@ast.cam.ac.uk>

Util class to transform between Gaia and other Photometric systems.
Coefficients used from Jordi et. al. 2010.

This transformations are created for different values of extinction for stars of different spectral types.

'''
from __future__ import print_function


import numpy as np
import os

def G_g(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: G_g(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 0)

def G_Grvs(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: G_Grvs(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 1)

def G_Gbp(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: G_Gbp(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 2)

def G_Grp(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: G_Grp(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 3)

def g_Grvs(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: g_Grvs(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 4)

def g_Gbp(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: g_Gbp(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 5)

def g_Grp(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: g_Grp(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 6)

def Gbp_Grp(gr=None, gi=None, gz=None, ri=None):
	'''
		Usage: Gbp_Grp(gr=sdss_gr_color)
		
		at least one color has to be set up.
		
		Input:
		gr: SDSS g - r color
		gi: SDSS g - i color
		gz: SDSS g - z color
		ri: SDSS r - i color
	'''
	return getColorPos(gr, gi, gz, ri, 7)


def getColorPos(gr, gi, gz, ri, pos):
	'''
	Returns the Gaia coded color for the position pos.	
	'''
	vec = {"gr":gr, "gi":gi, "gz":gz, "ri":ri}
	veco = {}    
	for i in vec.keys():
		if not vec[i] is None:
			veco[i]= vec[i]

	ncols = len(veco)
	
	if ncols == 1:
		return give_coefs_1_color(veco, pos)
	elif ncols == 2:
		return give_coefs_2_color(veco, pos)
	else:
		print ("Provide 1 or 2 colors please")
		return None

def give_coefs_1_color(vec, pos):
	'''
		Retrieves color coefficients for 1 color.
	'''
	col_name = list(vec.keys())[0]
	col_vec = list(vec.values())[0]
	try:
		path = os.path.join(os.getcwd(),  "../../data/sdss_{:}.dat".format(col_name)) 
		dat = np.loadtxt(path, usecols=(0, 1, 2, 3))
	except:
		print ("File not found: {:}".format(path))
		return None
	coefs = np.array(dat[pos])
	
	return coefs[0] + coefs[1]*col_vec + coefs[2]*col_vec**2 + coefs[3]*col_vec**3


def give_coefs_2_color(vec, pos):
	'''
		Retrieves color coefficients for 2 colors.
	'''
	col_names = vec.keys()
	col_vec = np.array(vec.values(), ndmin=2)
	try:
		path = os.path.join(os.getcwd(),  "../../data/sdss_{:}_{:}.dat".format(col_names[1], col_names[0])) 
		dat = np.loadtxt(path, usecols=(0, 1, 2, 3))
	except:
		print ("File not found: {:}".format(path))
		return None
	coefs = np.array(dat[pos])

	return coefs[0] + coefs[1]*col_vec[1] + coefs[2]*col_vec[1]**2 + coefs[3]*col_vec[1]**3 + coefs[4]*col_vec[0] + coefs[5]*col_vec[0]**2 + coefs[6]*col_vec[0]**3 + coefs[7]*col_vec[0]*col_vec[1] 
