
# coding: utf-8

# # Photometry
# @author: Nadia Blagorodnova

# From astropy and [Photutils](http://photutils.readthedocs.io/en/stable/).


from astropy import units as u
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from photutils import SkyCircularAperture, SkyCircularAnnulus
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
import astropy.io.fits as fits
import astropy.wcs
from astropy.time import Time
import datetime
import logging
import os, math, sys
import numpy as np
import matplotlib
from matplotlib import pylab as plt
import scipy.optimize as opt
from scipy import stats
import warnings


try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    from urllib.request import urlretrieve
    from urllib import request
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
    from urllib import urlretrieve
    
#Personal code inputs
from photometry import QueryCatalogue
from photometry import sextractor
#import QueryCatalogue
#import sextractor

from utils import fitsutils


class Photometry:

            
    def __init__(self):

        # Define our logger. Change the path to log the messages to a different location.

        #Root directory        
        self._base_path = "."        
        #Directory where logs shall be stored
        self._logpath = os.path.join(self._base_path, "data/log")
        #Directory where the photometry will be stored
        self._photpath = os.path.join(self._base_path, "data/phot")
        #Directory where the plots will be stored
        self._plotpath = os.path.join(self._base_path, "data/plots")
        #Directory where temporary files will be stored
        self._tmppath =  os.path.join(self._base_path, "data/tmp")
                
        #Create the directories if they do not exist
        dirs = [self._logpath, self._photpath, self._plotpath, self._tmppath]
        for d in dirs:
            if not os.path.isdir(d):
                os.makedirs(d)
                
        self.logger = None
        self.initialize_logger()
                
        # Create a dictionary to set up equivalent filter names between the observed system 
        # and the catalogues to be queried.
        self.filter_dic = {'ip':'i', 'rp':'r', 'gp':'g', 'up':'u', 'zs':'z', \
            'raj2000':'ra', 'dej2000':'dec', 'RAJ2000':'ra', 'DEJ2000':'dec',\
            'u_psf':'u', 'g_psf':'g', 'r_psf':'r', 'i_psf':'i', 'z_psf':'z',\
            'e_u_psf':'du', 'e_g_psf':'dg', 'e_r_psf':'dr', 'e_i_psf':'di', 'e_z_psf':'dz',
            'Vmag':'V',   'e_Vmag':'dV', 'Bmag':'B',   'e_Bmag':'dB', \
            'Rmag' : 'R', 'e_Rmag':'dR', 'Imag' : 'I', 'e_Imag':'dI',\
            'g_mag': 'g',  'e_g_mag':'dg', 'r_mag': 'r',  'e_r_mag':'dr', 'i_mag': 'i',  'e_i_mag':'di',
            'umag':'u', 'gma':'g', 'rmag':'r', 'imag':'i', \
            'raMean':'ra', 'decMean':'dec',\
            'gMeanPSFMag':'g', 'gMeanPSFMagErr':'dg', 'rMeanPSFMag':'r', 'rMeanPSFMagErr':'dr',
            'iMeanPSFMag':'i', 'iMeanPSFMagErr':'di', 'zMeanPSFMag':'z', 'zMeanPSFMagErr':'dz',
            'yMeanPSFMag':'y', 'yMeanPSFMagErr':'dy',\
            'Y':'y', 'dY':'dy',\
            'Err_g':'dg', 'Err_r':'dr', 'Err_i':'di', 'Err_z':'dz', 'Err_y':'dy',
            'gmag':'g', 'rmag':'r', 'imag':'i', 'zmag':'z', 'ymag':'y',\
             'e_gmag':'dg', 'e_rmag':'dr', 'e_imag':'di', 'e_zmag':'dz', 'e_ymag':'dy',
             'j_m':'J',   'j_cmsig':'dJ', 'h_m':'H', 'h_cmsig':'dH', 'k_m':'K', 'k_cmsig':'dK', 'Ks':'K', 'Ks + clear':'K',
             'SDSS-G': 'g', 'SDSS-R':'r', 'SDSS-I':'i', 'SDSS-Z':'z', 'MK-H':'H'}

                        
        #Dictionary where we choose which other filter we require for zeropoint 
        # colour term correction.
        self.col_dic = {    
            "U" : "B",
            "B" : "V",
            "V" : "B",
            "R" : "B", 
            "I" : "R",
            "Y" : "I",
            "u" : "r",
            "g" : "r",
            "r" : "g",
            "i" : "r",
            "z" : "i",
            "y" : "z",
            "J" : "H",
            "H" : "K",
            "K" : "H"
        }
    
        #Set up some instrument specific keywords, so that the code knows how to find
        # the gain, the pixel scale, the read out noise or the extension where the science
        # data is.
        # The values provided here are standard for LCO
        self.gain_keyword = 'GAIN'
        self.pixscale_keyword = 'PIXSCALE'
        self.rdnoise_keyword = 'RDNOISE'
        self.filter_keyword = 'FILTER1'
        self.ext = 1
        self.minmag = 14
        self.maxmag = 20
        self.survey = None
        self.aperture = 1.5
        

        


    def _twoD_Gaussian(self, xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        '''
        Produces a 2D gaussian centered in xo, yo with the parameters specified.
        xdata_tuple: coordinates of the points where the 2D Gaussian is computed.
        
        '''
        (x, y) = xdata_tuple                                                        
        xo = float(xo)                                                              
        yo = float(yo)                                                              
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))                                   
        return g.ravel()
        
    # Function to fit Gaussians to X, Y position in the image to detect if the stars are present.
    def _find_fwhm(self, imfile, xpos, ypos, plot=True):
        '''
        Finds and returns the best parameters for the FWHM in arcsec for the stars marked with X, Y
        '''
        
        f = fits.open(imfile)
        img = f[self.ext].data
        pix2ang = fitsutils.get_par(imfile, self.pixscale_keyword , self.ext)

        # DEfault PSF or 2 arcsec translated to pixels.
        def_fwhm = 2./pix2ang
        # Radius to compute the PSF (10 arcsec in pixels)
        rad = math.ceil(10./pix2ang)
        
        out = np.zeros(len(xpos), dtype=[('detected', np.bool), ('fwhm', np.float), ('e', np.float)])
        
        for i, (x_i,y_i) in enumerate(zip(xpos, ypos)):
            x_i = int(x_i)
            y_i = int(y_i)
            hrad = int(math.ceil(rad/2.))
    
            try:
                #sub = img[x_i-hrad:x_i+hrad, y_i-hrad:y_i+hrad]
                sub = img[y_i-hrad:y_i+hrad, x_i-hrad:x_i+hrad]
    
                x = np.linspace(0, len(sub), len(sub))
                y = np.linspace(0, len(sub), len(sub))
                X, Y = np.meshgrid(x, y)
            
                #(xdata_tuple, amplitude, xo, yo, def_fwhm, def_fwhm, theta, offset):
                def_x = hrad#np.argmax(np.sum(sub, axis=0))
                def_y = hrad#np.argmax(np.sum(sub, axis=1))
        
                initial_guess = (100, def_x, def_y, def_fwhm, def_fwhm, 0, np.percentile(sub, 40))
                popt, pcov = opt.curve_fit(self._twoD_Gaussian, (X, Y), sub.flatten(), p0=initial_guess, maxfev = 10000)
                amplitude=popt[0]
                xpos = popt[1]
                ypos = popt[2]
                fwhm_x = np.abs(popt[3])*2*np.sqrt(2*np.log(2))*pix2ang
                fwhm_y = np.abs(popt[4])*2*np.sqrt(2*np.log(2))*pix2ang
                background=np.maximum(np.percentile(sub.flatten(), 1), popt[-1])
                detected = ~np.isnan(fwhm_x)*~np.isnan(fwhm_y)*(amplitude > 0)*(0.33<(fwhm_y/fwhm_x)<3) \
                    * (amplitude/background > 0.5) * (np.abs(xpos)<2*hrad) * (np.abs(ypos)<2*hrad) 
    
            #We exceeded the number of iterations, meaning the Gaussian is not there
            except RuntimeError:
                detected = False
                fwhm_x = 0
                fwhm_y = 0
                amplitude = 0
                background = 0.001
    
            
            if (detected):
                self.logger.debug("DETECTED: %d %d, %s %s Amplitude %.3f\t BG %.3f\t BG_stats %.3f\t  FWHM_x,FWHM_y=(%.3f, %.3f)"%\
                    (xpos, ypos, i, detected, amplitude, background, np.percentile(sub, 50), fwhm_x, fwhm_y))
            else:
                self.logger.debug("NOT DETECTED: %d %d %s %s Amplitude %.3f\t BG %.3f\t BG_stats %.3f\t  FWHM_x,FWHM_y=(%.3f, %.3f)"%\
                    (xpos, ypos, i, detected, amplitude, background, np.percentile(sub, 50), fwhm_x, fwhm_y))
                    
            #Fill the data with the best fit parameters of the star.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[i] = (detected, np.average([fwhm_x, fwhm_y]), np.minimum(fwhm_x, fwhm_y) / np.maximum(fwhm_x, fwhm_y))
            
            if (detected & plot):
                data_fitted = self._twoD_Gaussian((X, Y), *popt)
                plt.figure(figsize=(10,5))
                fig, (ax, ax2) = plt.subplots(1, 2)
                ax.imshow(sub, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
                ax.contour(X, Y, data_fitted.reshape(sub.shape[0], sub.shape[1]), 5, colors='w')
                plt.title("DETECTED X,Y = %d,%d\n S/N:%.2f %.2f %.2f"%(x_i,y_i, amplitude/background, fwhm_x, fwhm_y))
                ax2.imshow(sub-data_fitted.reshape(sub.shape[0], sub.shape[1]), cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
                ax2.contour(X, Y, data_fitted.reshape(sub.shape[0], sub.shape[1]), 5, colors='w')
                ax.scatter(def_x, def_y, marker="*", s=100, color="yellow")
                figname = os.path.join( self._plotpath, os.path.basename(imfile)+"_gauss_%d.png"%i)
                plt.savefig(figname)
                plt.close()
            if ((not detected) & plot):      
                plt.figure(figsize=(5,5))
                fig, ax = plt.subplots(1)
                ax.imshow(sub, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
                plt.title("NOT DETECTED X,Y = %d,%d\n S/N:%.2f %.2f %.2f"%(x_i,y_i, amplitude/background, fwhm_x, fwhm_y))
                figname = os.path.join( self._plotpath, os.path.basename(imfile)+"_ngauss_%d.png"%i)
                plt.tight_layout()
                plt.savefig(figname)
                plt.close()
                
        return out


    
    # Read the positions of our stars from SDSS.
    
    def _extract_star_sequence(self, imfile, survey='PS1V3OBJECTS', minmag=14.5, maxmag=20, plot=True, debug=False):
        '''
        Given a fits image: imfile and a the name of the band which we want to extract the sources from,
        it saves the extracted sources into  '/tmp/sdss_cat_det.txt' file.
        If the band does not match the bands in the survey, a change is performed to adapt to the new band.
        
        If plotting activated, plots the USNOB1 field of stars on top of the star field image.
        Red circles are stars identified from the catalogue in the correct magnitude range.
        Yellow circles are stars that are isolated.
        
        Parameters:
        -----------
        imfile: str
                Name of the file to be calibrated.
        survey: str
                Name of the survey to be used for zeropoint calibration. By default is PS1.
        minmag: int. Default 14.5 mag
                The brightest mangitude that will be retrieved from the catalogues.
        maxmag: int. Default 20.0 mag
                The faintest mangitude that will be retrieved from the catalogues.
        plot: boolean
                Boolean for plotting the zeropoint calibation plots in the plot directory.
        debug: boolean. Default False.
                Boolean to show debug additional plots in the plot directory.
        '''
        
        f = fits.open(imfile)
            
        survey = survey.upper()
            
        
        #Extract the filter
        fheader = f[self.ext].header[self.filter_keyword]
        band = self.filter_dic.get(fheader, fheader)
    
        #Extract the WCS
        wcs = astropy.wcs.WCS(f[self.ext].header)
    
        #Extract the pixel scale
        pix2ang = f[self.ext].header[self.pixscale_keyword]
            
        #Extract the data
        img = f[self.ext].data
        #Assume that negative values shall be corrected
        img[img<0] = 0
        
        
        
        #Compute the ra, dec of the centre of the filed and the edges
        ra, dec = wcs.wcs_pix2world(np.array([img.shape[0]/2, img.shape[1]/2], ndmin=2), 1)[0]
        ra0, dec0 = wcs.wcs_pix2world(np.array([img.shape[0], img.shape[1]], ndmin=2), 1)[0]
    
        #Calculate the size of the field --> search radius  . As maximum, it needs to be 0.25 deg.
        sr_ori = 2.1*np.abs(dec-dec0)
        sr = np.minimum(sr_ori, 0.25)
        self.logger.info("Field center: (%.4f %.4f) and FoV: %.4f  [arcmin] "%( ra, dec, sr_ori*60))
        
        #Creates the Query class
        qc = QueryCatalogue.QueryCatalogue(ra, dec, sr, minmag, maxmag, self.logger)

        
        cat_file = os.path.join(self._tmppath, 'query_result_%s_%.3f_%.3f_%.3f_%.2f_%.2f.txt'%(survey.split("/")[-1], ra, dec, sr, minmag, maxmag) )   
        detected_stars_file = os.path.join(self._tmppath, 'detected_result_%s_%.3f_%.3f_%.3f_%.2f_%.2f.txt'%(survey.split("/")[-1], ra, dec, sr, minmag, maxmag) )   
            
        #Check if the query already exists in our tmp directory,
        #so we do not need to query it again.
        if (os.path.isfile(cat_file)):
            self.logger.info("File %s already exists. Loading it."%cat_file)
            catalog = Table.read(cat_file, format="ascii")
        #If that is not the case, then we check if the catalogue is in one of the lists provided by VO portal
        else:
            self.logger.info("File %s does not exist. Querying it."%cat_file)            
            if (np.any( np.array(['GSC23', 'GSC11', 'GSC12', 'USNOB', 'SDSS', 'FIRST', '2MASS', 'IRAS', 'GALEX', 'GAIA', 'TGAS', 'WISE', \
                   'CAOM_OBSCORE', 'CAOM_OBSPOINTING', 'PS1V3OBJECTS', 'PS1V3DETECTIONS'])==survey)):
                catalog = qc.query_catalogue(catalog_name=survey, filtered=True)                              
            #But it can be a SkyMapper as well (in the south).
            elif (survey == 'SKYMAPPER'):
                catalog = qc.query_sky_mapper(filtered=True)
            #More services can be added here, but at the moment, if the survey is none of the
            # above, the we paunch an error.
            else:
                self.logger.warn("Survey %s not recognized. Trying to query Vizier for that."%survey)

                try:
                    catalog = qc.query_vizier(catalog=survey)

                except:
                    self.logger.error("Unknown survey %s"%survey)
                    return None
    
        if (np.ndim(catalog)==0 or catalog is None):
            return False
        else:
            catalog = Table(data=catalog)
            #if ( np.all(catalog[band].mask )):
            #    self.logger.error( "All magntiudes for filter %s are masked!"% band)
            #    return False
            #else:
            #    catalog = catalog[~catalog[band].mask]
    
        #Rename the columns, so that the filters match our standard.
        try:
            for n in catalog.colnames:
                if n in self.filter_dic.keys():
                    new_name = self.filter_dic[n]
                    if (new_name in catalog.colnames):
                        catalog.remove_column(new_name)
                    catalog.rename_column(n, new_name)
        except IOError:
            self.logger.error( "Problems with catalogue IO %s"% band)
            return False
        except ValueError:
            self.logger.error( "Problems with the catalogue for the image")
            return False

        print ("Length: %d"%len(catalog), "Renamed catalog", catalog[0])
        
        if 'd'+band in catalog.colnames:
            #Make sure that the filter we want to calibrate and the color filter are not limits
            try:
                mask = (~catalog[band].mask) * (~(catalog[self.col_dic[band]].mask)) * (~np.isnan(catalog['d'+band])) * (~np.isnan(catalog['d'+self.col_dic[band]]))
            except AttributeError:
                mask = (~np.isnan(catalog['d'+band])) * (~np.isnan(catalog['d'+self.col_dic[band]]))
            catalog = catalog[mask]
        else:
            col = Column(np.zeros(len(catalog)), name='d'+band)
            catalog.add_column(col)
            mask = (~catalog[band].mask) * (~(catalog[self.col_dic[band]].mask))
            catalog = catalog[mask]
            
        #Make sure we convert it into a table
        #catalog = catalog.to_table()

        # Determine the X and Y of all the stars in the query.
        catcoords = astropy.coordinates.SkyCoord( catalog['ra'], catalog['dec'], unit=u.deg)
    
        self.logger.info("Catalogue has %d entries"%len(catcoords))
        
        #Convert ra, dec position of all stars to pixels.
        pixcoord = wcs.all_world2pix( np.array([catalog['ra'],  catalog['dec']]).T, 1)
        x = pixcoord.T[0]
        y = pixcoord.T[1]
        
        #Select only the stars within the image (and within an offset of 15 arcsec (in pixels) from the border.)
        off = math.ceil(15/pix2ang)
        mask1 = (x>off) * (x<img.shape[1]-off)*(y>off) * (y<img.shape[0]-off)
           
        #Select only stars isolated in a radius of ~10 arcsec. Cross match against itself and select the second closest.
        indexes, separations, distances = catcoords.match_to_catalog_sky(catcoords, nthneighbor=2)
        mask2 = (separations >  10 * u.arcsec)
     
        #Select the right magnitude range
        if catalog['ra'].dtype == 'float':
            mask3 = (catalog[band] > minmag)*(catalog[band] < maxmag)
        else:
            mask3 = (catalog[band].data.data > minmag)*(catalog[band].data.data < maxmag)

        print ("Stars left: mask1 %d, mask2 %d, mask3 %d"%(np.count_nonzero(mask1), np.count_nonzero(mask2), np.count_nonzero(mask3)))
        #Combine all masks
        mask = mask1 * mask2 * mask3
        
        if (not np.any(mask)):
            self.logger.warn("No good stars left with current conditions.")
            return False
        
        #Otherwise, it means there are good stars left
        catalog = catalog[mask]
        
        self.logger.info("Catalog length after masking: %d"%len(catalog))
    
        self.logger.debug("Left %d stars."%(len(catalog)))
    
        z = np.zeros(len(catalog), dtype=[('xpos', np.float), ('ypos',np.float)])
        
        z['xpos'] = x[mask]
        z['ypos'] = y[mask]
    
        #Iteratively create a header to store catalogue data for selected stars
        #only the relevant fields.
        catalog_det = Table(data=z, names=['xpos', 'ypos'])
        
        for n  in catalog.colnames:
            if ((n in ["objid", "ra", "dec", "u", "g", "r", "i", "z", "y", "du", "dg", "dr", "di", "dz", "dy"]) or
                (n in ['id', 'ra', 'dec', 'U', 'B', 'V', 'R', 'I', 'dU', 'dB', 'dV', 'dR', 'dI'] ) or
                (n in ['id', 'ra', 'dec', 'J', 'H', 'K', 'dJ', 'dH', 'dK'])):
                    catalog_det.add_column(catalog[n])
    
    
        catalog_det.write(cat_file, format="ascii.csv", overwrite=True)
        self.logger.info( "Saved catalogue stars to %s"%cat_file )
        print( "Saved catalogue stars to %s"%os.path.abspath(cat_file))
    
        #Find FWHM for this image            
        out = self._find_fwhm(imfile, catalog_det['xpos'], catalog_det['ypos'], plot=debug)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_valid_fwhm = ( ~np.isnan(out['fwhm']) *  ~np.isnan(out['e']) * out['detected'] * (out['e']>0.5) * (out['fwhm'] < 10))            
    
        self.logger.info("Left %d stars with valid fwhm."%np.count_nonzero(mask_valid_fwhm))
        
        if (np.count_nonzero(mask_valid_fwhm) < 3):
            self.logger.error( "ERROR with FWHM!! Too few points for a valid estimation. %d"% np.count_nonzero(mask_valid_fwhm)+ ") points")
            return False
    
        #Add the FWHM of each detection to the detected stars list
        outd = out[mask_valid_fwhm]
        catalog_det = catalog_det[mask_valid_fwhm]
        fwhm = Column(data=outd['fwhm'], name='fwhm')
        catalog_det.add_column(fwhm)

        #Write it to the header as well        
        fitsutils.update_par(imfile, "FWHM", np.median(outd['fwhm']), ext=self.ext)

        
        catalog_det.write(detected_stars_file, format="ascii", overwrite=True)
        
        self.logger.info( 'Average FWHM %.1f pixels, %.3f arcsec'%(np.median(outd['fwhm'])/pix2ang,  np.median(outd['fwhm'])))
        
            
        self.logger.info( "Found %d stars in %s. "%(len(catalog), survey)+ \
            "%d of them are isolated."%np.count_nonzero(mask2)+\
            "%d of them within the FoV. "%np.count_nonzero(mask) +\
            "%d of them with detected stars."%np.count_nonzero(mask_valid_fwhm)) 
        
        
        if (plot):
            #Plot results
            img = img - np.nanmin(img)
            zmin = np.percentile(img, 5)
            zmax = np.percentile(img, 95)
            plt.figure(figsize=(12,12))
                
            im = plt.imshow(img, aspect="equal", origin="lower", cmap=matplotlib.cm.gray_r, vmin=zmin, vmax=zmax)
    
    
            selected_x = catalog_det['xpos']
            selected_y = catalog_det['ypos']
            if (len(selected_x) >0):
                plt.scatter(selected_x, selected_y, marker="o", s=400, \
                    edgecolor="blue", facecolor="none", label="detected")
            
            plt.legend(loc="best", frameon=False, framealpha=0.9)
            plt.title("Selected stars for filter %s"%band)
            extension = os.path.basename(imfile).split(".")[-1]
            figname = os.path.join( self._plotpath, os.path.basename(imfile).replace(extension, 'seqstars.png'))
            plt.savefig( figname)
            self.logger.info( "Saved stars to %s"%figname)        
            plt.clf()
    
        return detected_stars_file
                                                    
    def _compute_bmjd(self, imagefile, ras, decs):
        
        if (fitsutils.has_par(imagefile, 'MJD-OBS', ext=self.ext)):
            mjd = fitsutils.get_par(imagefile, 'MJD-OBS', ext=self.ext)
        else:
            mjd = Time(fitsutils.get_par(imagefile, 'DATE-OBS', ext=self.ext)).mjd
            
        try:
            longit = fitsutils.get_par(imagefile, 'LONGITUD', self.ext)
            latit = fitsutils.get_par(imagefile, 'LATITUDE', self.ext)
            height = fitsutils.get_par(imagefile, 'HEIGHT', self.ext)
            exptime = fitsutils.get_par(imagefile, 'EXPTIME', self.ext)
        
            #Just take the first coordinates, as the time between nearby stars is negligible
            coords = SkyCoord(ras[0], decs[0], frame='icrs', unit=(u.deg, u.deg))
        
            longitude = longit+360
            observatory = EarthLocation.from_geodetic(longitude,latit,height)
            times = Time(mjd+exptime/(2.*86400.), format='mjd',scale='utc', location=observatory)
                    
            ltt_bary = times.light_travel_time(coords)
            time_barycentre = times.tdb + ltt_bary
        
            #Barycentric MJD
            bmjd = time_barycentre.mjd 
    
        except TypeError:
            self.logger.error("""Error encountered in computing the BMJD. Does the
                fits file contain the needed keywords? e.g. LONGITUD, LATITUDE, HEIGHT, EXPTIME""")
            bmjd = 0
            
        return bmjd                                           

    def initialize_logger(self):
        '''
        Cretaes a new logger for the class to output the processing status.
        '''
        #Define the format of the logging
        FORMAT = '%(asctime)-15s %(levelname)s [%(name)s] %(message)s'
        now = datetime.datetime.utcnow()
        timestamp = datetime.datetime.isoformat(now)
        timestamp = timestamp.split("T")[0]
        
        if self.logger is None:
            try:
                #Log into a file
                logging.basicConfig(format=FORMAT, filename=os.path.join(self._logpath, "rcred_{0}.log".format(timestamp)), level=logging.INFO)
                self.logger = logging.getLogger('zeropoint')
                print ("Logger created as %s"%os.path.abspath(os.path.join(self._logpath, "rcred_{0}.log".format(timestamp))))
            except:
                logging.basicConfig(format=FORMAT, filename=os.path.join("/tmp", "rcred_{0}.log".format(timestamp)), level=logging.INFO)
                self.logger= logging.getLogger("zeropoint")
                print ("Logger created as %s"%os.path.join("/tmp", "rcred_{0}.log".format(timestamp)))
                
            #Add a handler to output the messages to STDOUT too
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        
    def app_phot(self, imagefile, ras, decs, fwhm, plot=False, save=False):
        '''
        Computes the aperture photometry on the image, for the coordinates given.
        
        Parameters
        ----------
        imagefile : str
            The name of the fits file with the image.
        ras : array
            Array of floats with the RA positions for which aperture photometry is needed.
        decs : array
            Array of floats with the DEC positions for which aperture photometry is needed.
        fwhm : float
            Average FWHM of the field used to compute the aperture.
        plot : boolean
            Shall the apertures be plotted in the plot directory?
        save : boolean
            Save the aperture measurement to a file.

        Returns
        -------
        
        phot : QTable
            A table of the photometry with the following columns:
    
            'id': The source ID.
            'xcenter', 'ycenter': The x and y pixel coordinates of the input aperture center(s).
            'celestial_center': 
             'aperture_sum': The sum of the values within the aperture.
             'aperture_sum_err': The corresponding uncertainty in the 'aperture_sum' values. Returned only if the input error is not None.

        '''
        
        if np.ndim(ras)==0 or np.ndim(decs)==0:
            ras = [ras]
            decs = [decs]
            
        data = fits.open(imagefile)[self.ext].data
        filt = fitsutils.get_par(imagefile, self.filter_keyword, self.ext)
        
        
        # Calculate BJDs
        if (fitsutils.has_par(imagefile, 'MJD-OBS', ext=self.ext)):
            mjd = fitsutils.get_par(imagefile, 'MJD-OBS', ext=self.ext)
        else:
            mjd = Time(fitsutils.get_par(imagefile, 'DATE-OBS', ext=self.ext)).mjd

        try:
            bmjd = self._compute_bmjd(imagefile, ras, decs)
        except ValueError:
            bmjd = 0
            print ('Error when computing BMJD.')
        
        zp = fitsutils.get_par(imagefile, 'ZP', self.ext)
        zperr = fitsutils.get_par(imagefile, 'ZPERR', self.ext)
        color = fitsutils.get_par(imagefile, 'COLOR', self.ext)
        kcoef = fitsutils.get_par(imagefile, 'KCOEF', self.ext)
        if zp is None:
            zp = 0
        if zperr is None:
            zperr = 0

        wcs = astropy.wcs.WCS(fits.open(imagefile)[self.ext].header)
        
        positions = SkyCoord(ras*u.deg, decs*u.deg, frame='icrs')
                        
        # Set aperture radius to "aperture" times the fwhm radius
        aperture_rad = np.median(fwhm)*self.aperture* u.arcsec    
        aperture = SkyCircularAperture(positions, r=aperture_rad)
        
        print ("APERTURE RAD", aperture_rad)
        annulus_apertures = SkyCircularAnnulus(positions, r_in=aperture_rad*2.5, r_out=aperture_rad*4)
    
        #Convert to pixels
        pix_aperture = aperture.to_pixel(wcs)
        pix_annulus = annulus_apertures.to_pixel(wcs)
        pix_annulus_masks = pix_annulus.to_mask(method='center')
        
        #Plot apertures
        from astropy.visualization import simple_norm
        
        try:
            if np.ndim(ras) == 0:
                c = wcs.wcs_world2pix(np.array([[ras, decs]]), 1)                
            else:
                c = wcs.wcs_world2pix(np.array([ras, decs]).T, 1)
        except ValueError:
            self.logger.error('The vectors of RAs, DECs could not be converted into pixels using the WCS!')
            self.logger.error(str(np.array([ras, decs]).T))
    
        if plot:
            
            dataT = data.T
            x = c[:,0]
            y = c[:,1]
            
            X, Y = dataT.shape
            
            
            x0 = np.maximum(1, int(x[0])-100)
            x1 = np.minimum(X, int(x[0])+100)
            y0 = np.maximum(1, int(y[0])-100)
            y1 = np.minimum(Y, int(y[0])+100)
            
            self.logger.info("Plotting aperture photometry for target in pixels %s. Data shape %d x %d. Positions %d, %d, %d, %d"%(c,X,Y, x0, x1, y0, y1))

            plt.figure(figsize=(10,10))
            data_cut = dataT[x0: x1, y0: y1]
            data_cut[np.isnan(data_cut)] = np.nanmin(data_cut)
            nx, ny = data_cut.shape
            if nx > 1 and ny > 1:
                self.logger.info("Plotting a cutout of %d x %d pixels."%data_cut.shape)
                norm = simple_norm(data_cut, 'sqrt', percent=98)
                plt.imshow(data, norm=norm, origin="lower")
                pix_aperture.plot(color='white', lw=2)
                pix_annulus.plot(color='red', lw=2)
                plt.xlim(x0, x1)
                plt.ylim(y0, y1)
                plt.title('Apertures for filter %s'%filt)
                plt.savefig(os.path.join(self._plotpath, "apertures_cutout_%s.png"%os.path.basename(imagefile)))
                plt.clf()
            else:
                self.logger.error("Not enough pixels within the object area. Object in pixel %d %d outside of the field!"%(x[0], y[0]))
                norm = simple_norm(data, 'sqrt', percent=99)
                plt.imshow(data, norm=norm, origin="lower")
                pix_aperture.plot(color='white', lw=2)
                pix_annulus.plot(color='red', lw=2)
                plt.title('Apertures for filter %s'%filt)
                plt.savefig(os.path.join(self._plotpath, "apertures_cutout_targetout_%s.png"%os.path.basename(imagefile)))
                plt.clf()
        #Divide each pixel in 5 subpixels to make apertures
        apers = [pix_aperture, pix_annulus]
        phot_table = aperture_photometry(data, apers, method='subpixel', subpixels=5)
        for col in phot_table.colnames:
            phot_table[col].info.format = '%.8g'  # for consistent table output
        
    
        bkg_median = []
        std_counts = []
        for mask in pix_annulus_masks:
            annulus_data = mask.multiply(data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, stdv_clip = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            std_counts.append(stdv_clip)
            
        bkg_median = np.array(bkg_median)
        std_counts = np.array(std_counts)
        
        phot = aperture_photometry(data, pix_aperture)
        phot['annulus_median'] = bkg_median
        phot['annulus_std'] = std_counts
        phot['aper_bkg'] = bkg_median * pix_aperture.area
        phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    
    
        # Flux = Gain * Counts / Exptime.
        exptime = fitsutils.get_par(imagefile, 'EXPTIME', self.ext)
        gain = fitsutils.get_par(imagefile, self.gain_keyword, self.ext)
        
        flux =  gain * phot['aper_sum_bkgsub'] / exptime
        inst_mag = -2.5*np.log10(flux)
    
        phot['flux'] = flux
        phot['inst_mag'] = inst_mag
        
        #Noise is the poisson noise of the source plus the background noise for the extracted area
        err = np.sqrt (flux + pix_aperture.area * std_counts**2)
    
        #Transform pixels to magnitudes
        flux2 = gain * (phot['aper_sum_bkgsub']+err) / exptime
        inst_mag2 = -2.5*np.log10(flux2)
        
        errmag = np.abs(inst_mag2 - inst_mag)
        
        phot['err_counts'] = err
        phot['err_mag'] = errmag
        
        for col in phot.colnames:
            phot[col].info.format = '%.8g'  # for consistent table output
        
        self.logger.info(phot[0])
            
        if save:
            appfile = os.path.join(self._photpath, fitsutils.get_par(imagefile, "OBJECT", self.ext)+".app.phot.txt")
            self.logger.info('Creating aperture photometry out file as %s'%appfile)
            #Save the photometry into a file
            if (not os.path.isfile(appfile)):
                with open(appfile, 'w') as f:
                    f.write("filename mjd bmjd filter instr_mag zp zperr color kcoef mag magerr magerrtot\n")
            
    
            with open(appfile, 'a') as f:
                self.logger.info('Adding aperture photometry to file %s'%appfile)

                print ((os.path.basename(imagefile), mjd, bmjd, self.filter_dic.get(filt, filt), phot['inst_mag'].data[0], \
                    zp, zperr, color, kcoef, phot['inst_mag'].data[0]+ zp, phot['err_mag'].data[0], np.sqrt(zperr**2 + phot['err_mag'].data[0]**2)))
                
                f.write("%s %.8f %.8f %s %.4f %.4f %.4f %s %.4f %.4f %.4f %.4f\n"%(os.path.basename(imagefile), mjd, bmjd, self.filter_dic.get(filt, filt), phot['inst_mag'].data[0], \
                    zp, zperr, color, kcoef, phot['inst_mag'].data[0]+ zp, phot['err_mag'].data[0], np.sqrt(zperr**2 + phot['err_mag'].data[0]**2)))


        return phot
    
        
    
    
    def get_zeropoint(self, imgfile, survey, filt, col_filt=None, minmag=None, maxmag=None, plot=False):
        '''
        Function that fits the zeropoint for the image through 
        fitting a polynomial to instrumental magnitudes
        ZP = m_cat – 2.5 log ( Flux ) + K * color
        
        Parameters
        ----------
        imgfile : str
            The name of the file that has the image to be calibrated.
        survey : str
            Name of the survey that shall be used to calibrated the zeropoint.
        filt : str
            The name of the filter that the image was taken.
        col_filt : str
            The name of the filter used to compute the color term.
        minmag : float
            The minimum (brightest) star mag to be used for zeropoint calibration.
        maxmag : float
            The minimum (faintest) star mag to be used for zeropoint calibration.
        Returns
        -------
        zp : float
            The zeropoint for the image
        zperr : float
            The stdev for the zeropoint from all the stars
        K : float
            The colour term for the field.
        '''
         
        if minmag is None:
            minmag = self.minmag
        if maxmag is None:
            maxmag = self.maxmag
            
        #Select the stars above
        detected_stars_file = self._extract_star_sequence(imgfile, survey=survey, minmag=minmag, maxmag=maxmag, debug=False)

        if not detected_stars_file:
            return 0, 0, 0 
        t = Table.read(detected_stars_file, format="ascii")

        #Run aperture photometry on the positions of the stars.
        phot = self.app_phot(imgfile, t['ra'], t['dec'], fwhm=np.median(t['fwhm']))
        
         #Retrieve the default colour to be used for calibrations.
        if col_filt is None:
             col_filt = self.col_dic[filt]
             
        #Color may not always exist
        mask_color_exists = np.abs(t[col_filt])<30
        phot = phot[mask_color_exists]
        t = t[mask_color_exists]
         
        #We need to find the zeropoint by fitting a line
        zp_vec = t[filt]-phot['inst_mag']
        slope, intercept, r_value, p_value, std_err = stats.linregress(t[filt],  zp_vec)
        line = slope*t[filt]+intercept
        
        
        _, median_sigclip, std_sigclip = sigma_clipped_stats(zp_vec)
        mask_good = np.abs(zp_vec - median_sigclip)<(3*std_sigclip)

        if (plot):
             plt.figure()
             plt.errorbar(t[filt][mask_good], zp_vec[mask_good], yerr=phot['err_mag'][mask_good], \
                 fmt="o", color="b", label="accepted", alpha=0.5, ms=5)
             plt.errorbar(t[filt], zp_vec, yerr=phot['err_mag'], fmt="o", color="r", label="rejected", alpha=0.5, ms=3)

             plt.xlabel('%s [mag]'%filt)
             plt.ylabel('ZP [mag]')
             plt.title("%d stars for ZP calibration in %s band"%(len(t[filt][mask_good]), filt))
             plt.tight_layout()
             plt.savefig(os.path.join(self._plotpath,"zp_cal_%s_%s_%s.png"%(os.path.basename(imgfile), filt, col_filt)))
             plt.clf()
         
         
         
        #First iteration
        color = t[filt]-t[col_filt]
         
        #Reject extreme colors... more than 0.8 mag
        mask_good = mask_good * (np.abs(color)<0.8)
        
        self.logger.info("First iteration for zeropoint calibration. %d good stars left."%np.count_nonzero(mask_good))
        #Only accept the good ones
        t = t[mask_good]
        zp_vec = zp_vec[mask_good]
        phot = phot[mask_good]
        color = color[mask_good]
           
        
        if 'd'+filt in t.keys():
            coefs, residuals, rank, singular_values, rcond = np.polyfit(color, zp_vec, w=1./np.sqrt(t['d'+filt]**2+phot['err_mag']**2), deg=1, full=True)
            p = np.poly1d(coefs)
        else:
            coefs, residuals, rank, singular_values, rcond = np.polyfit(color, zp_vec, w=1./phot['err_mag'], deg=1, full=True)
            p = np.poly1d(coefs)   
             
        slope, intercept, r_value, p_value, std_err = stats.linregress(color,  zp_vec)
        line = slope*color+intercept
        
        prediction_error_zp = zp_vec-p(color)
        _, median_sigclip, std_sigclip = sigma_clipped_stats(prediction_error_zp)
        mask_good = np.abs(prediction_error_zp - median_sigclip)<(3*std_sigclip)
        
        self.logger.info("Second iteration for zeropoint calibration. %d good stars left."%np.count_nonzero(mask_good))

        #Second itneration
        t = t[mask_good]
        zp_vec = zp_vec[mask_good]
        phot = phot[mask_good]
        color = color[mask_good]        
        
        if 'd'+filt in t.keys():
            coefs, residuals, rank, singular_values, rcond = np.polyfit(color, zp_vec, w=1./np.sqrt(t['d'+filt]**2+phot['err_mag']**2), deg=1, full=True)
            p = np.poly1d(coefs)
        else:
            coefs, residuals, rank, singular_values, rcond = np.polyfit(color, zp_vec, w=1./phot['err_mag'], deg=1, full=True)
            p = np.poly1d(coefs)   
             
        slope, intercept, r_value, p_value, std_err = stats.linregress(color,  zp_vec)
        line = slope*color+intercept
        
        prediction_error_zp = zp_vec-p(color)
        
        if (plot):
            plt.figure(figsize=(6,10))
            plt.subplot(2,1,1)
            plt.title("ZP: %.2f color-term: %.2f"%(coefs[1], coefs[0]))
            plt.errorbar(color, zp_vec, yerr=phot['err_mag'], fmt="o", alpha=0.5, ms=3)
            plt.plot(color, line, label="No Errors")
            plt.plot(color, p(color), label="With Errors")
            plt.xlabel('%s - %s [mag]'%(filt, col_filt))
            plt.ylabel('ZP [mag]')
            plt.legend()
            
            plt.subplot(2,1,2)
            plt.title("ZP STD %.3f"%(np.std(prediction_error_zp)))
            plt.errorbar(color, prediction_error_zp, yerr=phot['err_mag'], fmt="o", alpha=0.5, ms=3)
            plt.hlines(0, np.min(color), np.max(color))
            plt.xlabel('%s - %s [mag]'%(filt, col_filt))
            plt.ylabel('ZP$_{obs}$ - ZP$_{pred}$ [mag]')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self._plotpath,"zp_colorterm_%s_%s_%s.png"%(os.path.basename(imgfile), filt, col_filt)))
            plt.clf()
         
        self.logger.info("ZP median: %.4f STD: %.4f"%(np.median(zp_vec),np.std(zp_vec)))
        self.logger.info("ZP_err: %.4f color coef: %.4f"%(coefs[1], coefs[0]))
        self.logger.info("ZP_noerr: %.4f color coef: %.4f"%(intercept, slope))
        
        print ("ZP median: %.4f STD: %.4f"%(np.median(zp_vec),np.std(zp_vec)))
        print ("ZP_err: %.4f color coef: %.4f"%(coefs[1], coefs[0]))
        print ("ZP_noerr: %.4f color coef: %.4f"%(intercept, slope))
        
        #Save the image zeropoint in the header
        fitsutils.update_par(imgfile, "ZP", coefs[1], ext=self.ext)
        fitsutils.update_par(imgfile, "ZPERR", np.std(zp_vec-p(color)), ext=self.ext)
        fitsutils.update_par(imgfile, "KCOEF", coefs[0], ext=self.ext)
        fitsutils.update_par(imgfile, "COLOR", "%s-%s"%(filt, col_filt), ext=self.ext)
        
        return np.median(zp_vec), np.std(zp_vec-p(color)), coefs[0] 

      
             
    def measure_mag(self, imgfile, ra=None, dec=None, unify_headers=False, zp=None):
        '''
        Shows how to use the code to: 
        
        1) derive the zeropoint for the image. 
        2) Compute the aperture magnitude
        
        Parameters
        ----------
        imgfile : str
            Name of the fits file that contains the data.
        ra : float
            Right ascention of the object the aperture magnitude will be computed for.
            If the value is none, the coordinates will be extracted from the header keyword 'RA'
        dec : float
            Right ascention of the object the aperture magnitude will be computed for.   
            If the value is none, the coordinates will be extracted from the header keyword 'DEC'
        unify_headers : bool
            Boolean to say if we want to add to the header the unified names
            for the standard parameters. It will overwrite the old header with extra values. 
        '''
    
        #Define which catalogue we want to calibrate the zeropoint against.
        # Some choices are:
        # 'GSC23', 'GSC11', 'GSC12', 'USNOB', 'SDSS', 'FIRST', '2MASS', 'IRAS', 'GALEX', 'GAIA', 'TGAS', 'WISE', \
        #'CAOM_OBSCORE', 'CAOM_OBSPOINTING', 'PS1V3OBJECTS', 'PS1V3DETECTIONS','SKYMAPPER'
        
        if dec > -30:
            survey_dic = {
            "u" : "SDSS",
            "g" : "PS1V3OBJECTS",
            "r" : "PS1V3OBJECTS",
            "i" : "PS1V3OBJECTS",
            "z" : "PS1V3OBJECTS",
            "y" : "PS1V3OBJECTS",
            "Y" : "PS1V3OBJECTS",
            "J" : "2MASS",
            "H" : "2MASS",
            "K" : "2MASS",
            "U" : "APASS",
            "B" : "UCAC4",
            "V" : "UCAC4",
            "R" : "UCAC4",
            "I" : "APASS"
            }
        else:
            survey_dic = {
            "g" : "SKYMAPPER",
            "r" : "SKYMAPPER",
            "i" : "SKYMAPPER",
            "z" : "SKYMAPPER",
            "y" : "SKYMAPPER",
            "Y" : "SKYMAPPER",
            "J" : "2MASS",
            "H" : "2MASS",
            "K" : "2MASS",
            "U" : "APASS",
            "B" : "UCAC4",
            "V" : "UCAC4",
            "R" : "UCAC4",
            "I" : "APASS"
            }
            
        
        f = fits.open(imgfile)
        
        ext = 0
        while ext < len(f):
            if not 'TELESCOP' in f[ext].header:
                ext = ext + 1

            else:
                self.ext = ext
                self.logger.info("Found the data to be in extension %d"%ext)
                break
                
        if (ext == len(f)):
            self.logger.error("I searched through all the extensions in this files and could not find the one containing the TELESCOP keyword!")
            return
            
        #First Unify header words, which may be different fordifferent telescopes
        if (unify_headers):
            f = fits.open(imgfile)
            hdr = f[self.ext].header
            data = f[self.ext].data
            new_hdr = fitsutils.unify_header(hdr)
            f[self.ext].header = new_hdr
            new_imgfile = os.path.join(os.path.dirname(imgfile), "unified_"+os.path.basename(imgfile))
            f.writeto(new_imgfile, overwrite=True)
            self.logger.info("Wrote unified header into image %s"%new_imgfile)
            imgfile = new_imgfile

    
        #Extract the filter from extension 1
        print ( imgfile, self.filter_keyword, self.ext)
        filt_original = fitsutils.get_par(imgfile, self.filter_keyword, self.ext)
        filt = self.filter_dic.get(filt_original, filt_original)
        print ("Original FILTER: ", filt_original, ". New filter:", filt)
        if filt is None:
            print ("Could not find the keyword %s in the header in extension %d."%(self.filter_keyword, self.ext))
            print ("Please check that the extension you setected has data.")
            
    
        #Check which survey we should query provided the filter the data was taken. 
        if not self.survey is None:
            survey = self.survey
        elif filt in survey_dic.keys():
            survey = survey_dic[filt]
        else:
            print ("A survey name is required to calibrate your image against.")
            return
    
        if zp is None:
            try:
                #Compute the zeropoint
                #For that here we select stars between 15 and 19.5 mag.
                zp, zp_err, colorterm = self.get_zeropoint(imgfile, survey=survey, filt=filt, \
                    col_filt=self.col_dic[filt], minmag=self.minmag, maxmag=self.maxmag, plot=True)
            
                if zp == 0:
                    self.logger.error("There were not enough stars to compute a reliable zeropoint for image %s. Setting zeropoint to 0!"%imgfile)
                    fitsutils.update_par(imgfile, "ZP", 0, self.ext)
                    fitsutils.update_par(imgfile, "ZPERR", 0, self.ext)
                    fitsutils.update_par(imgfile, "COLOR", 0, self.ext)
                    fitsutils.update_par(imgfile, "KCOEF", 0, self.ext)
            except SystemExit:
                self.logger.error("Zeropoint could not be computed. Will set it to zero instead and compute instrumental mags only.")
                fitsutils.update_par(imgfile, "ZP", 0, self.ext)
                fitsutils.update_par(imgfile, "ZPERR", 0, self.ext)
                fitsutils.update_par(imgfile, "COLOR", 0, self.ext)
                fitsutils.update_par(imgfile, "KCOEF", 0, self.ext)
        elif zp == 0:
            self.logger.warn("You have selected zp=0! No further calibrations will be attempted!")
            fitsutils.update_par(imgfile, "ZP", 0, self.ext)
            fitsutils.update_par(imgfile, "ZPERR", 0, self.ext)
            fitsutils.update_par(imgfile, "COLOR", 0, self.ext)
            fitsutils.update_par(imgfile, "KCOEF", 0, self.ext)     

            
        #Now get the positions of the transient and run aperture photometry on it
        # with the FWHM computed in the previous step.
        fwhm = fitsutils.get_par(imgfile, "FWHM", self.ext)
        if fwhm is None or fwhm ==0:
            ang2pix = fitsutils.get_par(imgfile, self.pixscale_keyword, self.ext) 
            nsources, fwhm, ellipticity, bkg = sextractor.get_image_pars(imgfile, arcsecpix=ang2pix)
            print ("And2pix: ", ang2pix, "fwhm", fwhm)
    
        if ra is None or dec is None:    
            ra = fitsutils.get_par(imgfile, "RA", self.ext)
            dec = fitsutils.get_par(imgfile, "DEC", self.ext)
            #Set the values from the header if there is no better value
            self.logger.info("RA, DEC parameters not found. The following values were taken the from the header RA: %s DEC: %s."%(ra, dec))
        try:
            ra = float(ra)
            dec = float(dec)
            coords = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))
        except ValueError:
            coords = SkyCoord(ra, dec, frame='icrs', unit=(u.hour, u.deg))
            
            
        #Run aperture photometry on the positions of the stars.
        phot = self.app_phot(imgfile, coords.ra.deg, coords.dec.deg, fwhm=fwhm, save=True, plot=True)
        print (phot['inst_mag'].data[0], zp, phot['err_mag'].data[0])
        self.logger.info("RA: %.6f DEC: %.6f. App photometry for object: %.4f+/-%.4f"%( coords.ra.deg, coords.dec.deg, phot['inst_mag'].data[0]+ zp, phot['err_mag'].data[0]))



def usage_case(f, ra, dec, n_try=0, max_tries=1):
    '''
    
    Shows how to use this module to get the photometry.

    Parameters
    ----------
    f : String
        File name.
    ra : double
        Right ascention.
    dec : double
        Declination.
    n_try : int, optional
        Number of the try. The default is 0.
    max_tries : int, optional
        Maximum number of tries. The default is 1.

    Returns
    -------
    None.
    Writes the data with the photometric measurements in the directory phot

    '''    
    phot = Photometry()
    phot.minmag = 15
    phot.maxmag = 20
    phot.filter_keyword = 'FILTER1'
    
    #Abort if we tried already the maximum amount of times.
    if n_try >= max_tries:
        print ('''Error, number of tries reached the max allowed number of times: %d. 
               Aborting with this image.'''%max_tries)
        return
    
    try:
        phot.measure_mag(f, ra=ra, dec=dec, unify_headers=False)
    except TypeError:
        print ('ERROR detected with file %s. Probably exposure is weathered out.'%f)
    except KeyError:
        print ('ERROR, because this band is not contained in the base survey.')
    except IndexError:
        print ('Error, probably not enough sources on image.')
    except request.URLError:
        print ('Error, the web site Timed Out! Trying again.')
        usage_case(f, ra, dec, n_try=n_try+1, max_tries=1)
