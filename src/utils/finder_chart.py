from __future__ import print_function

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    from urllib.request import urlretrieve
    from urllib import request
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
    from urllib import urlretrieve
import sys
import numpy as np
import matplotlib.pyplot as plt
import pylab 
from matplotlib.ticker import MaxNLocator
import argparse
import os
import astropy.wcs
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.gaia import Gaia
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle


from scipy.ndimage.filters import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

def deg2hour(ra, dec, sep=":"):
    '''
    Transforms the coordinates in degrees into HH:MM:SS DD:MM:SS with the requested separator.
    '''
    
    if ( type(ra) is str and type(dec) is str ):
        return ra, dec
        
    c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    
    ra = c.ra.to_string(unit=u.hourangle, sep=sep, precision=2, pad=True)
    dec = c.dec.to_string(sep=sep, precision=2, alwayssign=True, pad=True)
    
    return str(ra), str(dec)
    
    
def hour2deg(ra, dec):
    '''
    Transforms string HH:MM:SS DD:MM:SS coordinates into degrees (floats).
    '''
    try:
        ra = np.double(ra)
        dec = np.double(dec)
        
    except:
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
        
        ra = c.ra.deg
        dec = c.dec.deg
    
    return ra, dec


def get_offset(ra1, dec1, ra2, dec2):
    '''
    Computes the offset in arcsec between two coordinates.
    The offset is from (ra1, dec1) - generally an offset star to (ra2, dec2) - the fainter target.
    '''
    
    from astropy.coordinates import SkyCoord
    bright_star = SkyCoord(ra1, dec1, frame='icrs', unit=(u.deg, u.deg))
    target = SkyCoord(ra2, dec2, frame='icrs', unit=(u.deg, u.deg))
    dra, ddec = bright_star.spherical_offsets_to(target)
    
    return dra.to(u.arcsec).value, ddec.to(u.arcsec).value

def query_sky_mapper_catalogue(ra, dec, radius_deg, minmag=15, maxmag=18.5):
    '''
    Sends a VO query to the SkyMapper catalogue.
    '''
    
    url = "http://skymapper.anu.edu.au/sm-cone/query?RA=%.6f&DEC=%.6f&SR=%.4f&RESPONSEFORMAT=CSV"%(ra, dec, radius_deg)

    f = open("/tmp/skymapper_cat.csv", "wb")
    page = urlopen(url)
    content = page.read()
    f.write(content)
    f.close()
    
    # Read RA, Dec and magnitude from CSV 
    catalog = Table.read("/tmp/skymapper_cat.csv", format="ascii.csv")

    mask = (catalog["class_star"]>0.7) * (catalog["ngood"] >5)  * (catalog['r_psf']>minmag) * (catalog['r_psf']<maxmag)
    catalog = catalog[mask]
    
    newcat = np.zeros(len(catalog), dtype=[("ra", np.double), ("dec", np.double), ("mag", np.double)])
    newcat["ra"] = catalog["raj2000"]
    newcat["dec"] = catalog["dej2000"]
    newcat["mag"] = catalog["r_psf"]
    
    return newcat
    
def query_ps1_catalogue(ra, dec, radius_deg, minmag=15, maxmag=18.5):
    '''
    Sends a VO query to the PS1 catalogue.
    Filters the result by mangitude (between 15 and 18.5)
    and by the PSF-like shape of the sources.
    '''

        
    url = "http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx?CAT=PS1V3OBJECTS&RA=%.5f&DEC=%.5f&SR=%.5f&FORMAT=csv"%(ra, dec, radius_deg)
    #urllib.urlretrieve(url, "/tmp/ps1_cat.xml") 
    
    f = open("/tmp/ps1_cat.csv", "wb")
    
    try:
        page = request.urlopen(url)
    except:
        page = urlopen(url)

    content = page.read()
    f.write(content)
    f.close()

    # Read RA, Dec and magnitude from CSV 
    catalog = Table.read("/tmp/ps1_cat.csv", format="ascii.csv", header_start=1)

    mask = (catalog["nDetections"]>3) * (catalog["rMeanPSFMag"] > minmag) * (catalog["rMeanPSFMag"] < maxmag) *\
    (catalog["iMeanPSFMag"] - catalog["iMeanKronMag"] < 0.1) #This last one to select stars.
    
    #*(catalog["rMeanPSFMag"] > minmag) * (catalog["rMeanPSFMag"] < maxmag) 
    catalog = catalog[mask]

    
    newcat = np.zeros(len(catalog), dtype=[("ra", np.double), ("dec", np.double), ("mag", np.double)])
    newcat["ra"] = catalog["RaMean"]
    newcat["dec"] = catalog["DecMean"]
    newcat["mag"] = catalog["rMeanPSFMag"]
    
    return newcat

def query_gaia_catalogue(ra, dec, radius_deg, minmag=15, maxmag=18.5):
    '''
    Sends a VO query to the Gaia catalogue.
    Filters the result by G mangitude (between minmag and maxmag).
    '''


    query = '''SELECT ra, dec, phot_g_mean_mag
                                    FROM gaiadr3.gaia_source
                                    WHERE 1=CONTAINS(
                                      POINT('ICRS', %.6f, %.6f),
                                      CIRCLE('ICRS',ra, dec, %.6f))
                                    AND phot_g_mean_mag>=%.2f AND phot_g_mean_mag<%.2f'''%(ra, dec, radius_deg, minmag, maxmag)
    print (query)
    job = Gaia.launch_job_async(query)
                                    #, dump_to_file=True, output_format='votable')

    catalog = job.get_results()

    print (catalog)
    
    newcat = np.zeros(len(catalog), dtype=[("ra", np.double), ("dec", np.double), ("mag", np.double)])
    newcat["ra"] = catalog["ra"]
    newcat["dec"] = catalog["dec"]
    newcat["mag"] = catalog["phot_g_mean_mag"]
    
    return newcat

def get_fits_image(ra, dec, rad, debug=True):
    '''
    Connects to the PS1 or SkyMapper image service to retrieve the fits file to be used as a bse for the finder chart.
    '''
    #If dec> -30, we have Pan-STARRS
    if dec > -30:
        # Construct URL to download Pan-STARRS image cutout, and save to tmp.fits
    
        # First find the index of images and retrieve the file of the image that we want to use.
        image_index_url = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=r'.format(ra, dec)
        urlretrieve(image_index_url, '/tmp/ps1_image_index.txt')
        ix = Table.read('/tmp/ps1_image_index.txt', format="ascii")
        f = ix['filename'].data[0]
        
        image_url = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={0}&format=fits&size={1}&ra={2}&dec={3}".format(f, int(np.round(rad*3600*8, 0)), ra, dec)
        if (debug):
            print ("URL:", image_url)
            print ("Downloading PS1 r-band image...")
            
        #Store the object to a fits file.
        urlretrieve(image_url, '/tmp/tmp.fits')
        
            
    #Otherwise, we have SkyMapper   
    else:    
        url="http://skymappersiap.asvo.nci.org.au/dr1_cutout/query?POS=%.6f,%.6f&SIZE=%.3f&FORMAT=image/fits&INTERSECT=center&RESPONSEFORMAT=CSV"%(ra, dec, rad*2)
        page = urlopen(url)
        content = page.read()
        f = open("/tmp/skymapper_image_index.csv", "wb")
        f.write(content)
        f.close()
        
        ix = Table.read('/tmp/skymapper_image_index.csv', format="ascii.csv")

        mask = ((ix['band']=='r')|(ix['band']=='g'))
        
        ix = ix[mask]
        ix.sort(keys='exptime')
    
        image_url = ix['get_image'][-1]
        urlretrieve(image_url, '/tmp/tmp.fits')



    #Finally, once we have Pan-STARRS or SkyMapper images, we try to open them.
    #If there has been any problem with that, we will just go to the DSS image service.
    try:
        image = fits.open("/tmp/tmp.fits")
        #If everything went well, it shall be a fits image and opening it shall cause no issue.
        return '/tmp/tmp.fits'

        #If there was an error with the fits, we shall go for the DSS image
    except IOError:
        #One of the services may fail, so we need to account for that and provide a backup DSS image service.
        try:
            image_url = 'http://archive.eso.org/dss/dss/image?ra=%.5f&dec=%.5f&x=%.2f&y=%.2f&Sky-Survey=DSS1&mime-type=download-fits' % \
                ((ra), (dec), (rad*60), (rad*60))
            if debug: print ("Downloading DSS image...")
            urlretrieve(image_url, '/tmp/tmp.fits')
        except:
            image_url = 'http://archive.stsci.edu/cgi-bin/dss_search?ra=%.6f&dec=%.6f&generation=DSS2r&equinox=J2000&height=%.4f&width=%.4f&format=FITS' % \
                (ra, dec, rad*60, rad*60)
            urlretrieve(image_url, '/tmp/tmp.fits')
            
        #We try one more time to open it. If not successful, we return None as the image filename.
        try:
            fits.open("/tmp/tmp.fits")
        except IOError:
            print ("Your fits image could not be retrieved.")
            return None

            
        
def get_cutout(ra, dec, name, rad, debug=True):
    '''
    Obtains the color composite cutout from the PS1 images.
    '''
    try:
        ra=np.double(ra)
        dec=np.double(dec)
    except:
        ra, dec = hour2deg(ra, dec) 
        
    catalog = query_ps1_catalogue(ra, dec, rad)
    
    if (debug):
        print (catalog)

    
    # Construct URL to download DSS image cutout, and save to tmp.fits
    image_index_url_red = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=y'.format(ra, dec)
    image_index_url_green = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=i'.format(ra, dec)
    image_index_url_blue = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=g'.format(ra, dec)

    urlretrieve(image_index_url_red, '/tmp/image_index_red.txt')
    urlretrieve(image_index_url_green, '/tmp/image_index_green.txt')
    urlretrieve(image_index_url_blue, '/tmp/image_index_blue.txt')

    ix_red = np.genfromtxt('/tmp/image_index_red.txt', names=True, dtype=None)
    ix_green = np.genfromtxt('/tmp/image_index_green.txt', names=True, dtype=None)
    ix_blue = np.genfromtxt('/tmp/image_index_blue.txt', names=True, dtype=None)
    
    image_url = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red=%s&green=%s&blue=%s&filetypes=stack&auxiliary=data&size=%d&ra=%.6f&dec=%.6f&output_size=256"%\
    (ix_red["filename"], ix_green["filename"], ix_blue["filename"], rad*3600*4, ra, dec)

    if (debug):
        print (image_url)
        print ("Downloading PS1 r-band image...")
    urlretrieve(image_url, '/tmp/tmp_%s.jpg'%name)
    
    
def get_finder(ra, dec, name, rad, debug=False, starlist=None, print_starlist=True, \
               telescope="P200", directory=".", minmag=15, maxmag=18.5, mag=np.nan, image_file=None,
               finder_ext="pdf"):
    '''
    Creates a PDF with the finder chart for the object with the specified name and coordinates.
    It queries the PS1 catalogue to obtain nearby offset stars and get an R-band image as background.
    
    Parameters
    ----------
    ra : double
        RA of our target in degrees.
    dec : double
        DEC of our target in degrees.
    name : str
        The name of your target
    rad : double
        Search radius for the finder in degrees.
    debug : bool (optional)
        Option to activate/ deactivate additional output.
    starlist : str (optional)
        Path/name of the file where the coordinates for the object are going to be saved. 
        If the file exists, the content will be appended at the end.
        If no value is provided, the output just writes to the standard output (in case print_starlist is True).
    print_starlist : boolean (optional)
        Indicates if the starlist shall be printed in the standard output.
    telescope : str (optional)
        The current accepted values are "Keck", "P200", "Calar".
    directory : str (optional)
        The directory where the PDF with the finder chart shall be stored. 
        If no value given, the file will be store in the current directory where the script is run.
    minmag : double (optional)
        The minimum magnitud (brightest in this case) star that we would like to use as an offset star.
    maxmag : double (optional)
        The maximum magnitude (faintest) star that we would like to use as an offset star.
    mag : double or `None` (optional)
        The magnitude of our target.
    image_file : str (optional)
        The name of the fits file that you want to use as a background to your finder chart. If none, provided,
        the script will automatically look for imaging catalogues: PS1 (North), SkyMapper (South), or DSS
    finder_ext : str (optional)
        Extension of the finder file. By default is a PDF, but can be a PNG or JPG for example.
    '''
        
    print ("Got it")
    try:
        ra=np.double(ra)
        dec=np.double(dec)
    except:
        ra, dec = hour2deg(ra, dec) 

    
    print ('Image file:', image_file)

    #If no custom fits image is provided, we query for one from PS1/DSS/SkyMapper
    if image_file is None:
        image_file = get_fits_image(ra, dec, rad, debug=debug)    
        image = fits.open(image_file)
        wcs = astropy.wcs.WCS(image[0].header)
    else:
        print ('Reading custom fits')
        image = fits.open(image_file)
        # Get pixel coordinates of SN, reference stars in DSS image
        wcs = astropy.wcs.WCS(fits.open(image_file)[0].header)
        
        try:
            image[0].data = np.rot90(np.rot90(image[0].data))
        except ValueError:
            print ('Rotation failed')


    if image_file is None or image is None:
        print ("FATAL ERROR! Your FITS image could not be retrieved.")
        return
        
    
    # Plot finder chart
    
    #Adjust some of the counts to make easier the plotting.
    image[0].data[image[0].data>20000] = 20000
    image[0].data[np.isnan(image[0].data)] = 0


    plt.figure(figsize=(8, 6))
    plt.set_cmap('gray_r')
    smoothedimage = gaussian_filter(image[0].data, 1.1)
    
    hdu = fits.open(image_file)[0]
    wcs = WCS(hdu.header)
    
    plt.subplot(projection=wcs)
    #plt.imshow(hdu.data, vmin=-2.e-5, vmax=2.e-4, origin='lower')
    plt.grid(color='white', ls='solid')
    plt.imshow(smoothedimage, origin='lower',vmin=np.percentile(smoothedimage.flatten(), 1), \
    vmax=np.percentile(smoothedimage.flatten(), 99))

    #Mark target in green
    ax = plt.gca()
    r = SphericalCircle((ra * u.deg, dec * u.deg), 2 * u.arcsec,
                         edgecolor='green', facecolor='none',
                         transform=ax.get_transform('fk5'))
    ax.add_patch(r)
    #Write the name of the target
    plt.annotate(name, xy=(ra, dec),  xycoords='data', xytext=(0.55, 0.5), textcoords='axes fraction', color="g")

    # Plot compass
    plt.plot([(image[0].data.shape[0])-10,(image[0].data.shape[0]-40)],[10,10], 'k-', lw=2)
    plt.plot([(image[0].data.shape[0])-10,(image[0].data.shape[0])-10],[10,40], 'k-', lw=2)
    plt.annotate("N", xy=((image[0].data.shape[0])-20, 40),  xycoords='data', xytext=(-4,5), textcoords='offset points')
    plt.annotate("E", xy=((image[0].data.shape[0])-40, 20),  xycoords='data', xytext=(-12,-5), textcoords='offset points')

    ax.set_xlabel('%.1f\''%(rad*60*2))
    ax.set_ylabel('%.1f\''%(rad*60*2))
    

    #Queries a catalogue to get the offset stars
    try:
        catalog = query_gaia_catalogue(ra, dec, rad, minmag=minmag, maxmag=maxmag)
    except Exception as e:
        print(e)
        catalog = []
        
    if len(catalog)==0:
        try:
            if dec < -30:
                catalog = query_sky_mapper_catalogue(ra, dec, (rad)*0.95, minmag=minmag, maxmag=maxmag)
            else:
                catalog = query_ps1_catalogue(ra, dec, (rad)*0.95, minmag=minmag, maxmag=maxmag)
        except Exception as e:
            print(e)
            catalog = []
        
    if (debug):
        print ("Catalog of %d stars retrieved"%len(catalog), catalog)

    
    #Possible ways to get more stars if we find nothing in the specified range.
    '''if (len(catalog)<3):
        if debug: print ("Looking for a bit fainter stars up to mag: %.2f"%(maxmag+0.25))
        catalog = query_ps1_catalogue(ra, dec, (rad/2.)*0.95, minmag=minmag, maxmag=maxmag+0.5)

    if (len(catalog)<3):
        print ("Restarting with larger radius %.2f arcmin"%(rad*60+0.5))
        get_finder(ra, dec, name, rad+0.5/60, directory=directory, minmag=minmag, \
                   maxmag=maxmag+0.5, mag=mag, starlist=starlist, telescope=telescope, image_file=image_file)
        return'''
        
    if (not catalog is None and len(catalog)>0):
        np.random.shuffle(catalog)

        #no_self_object = (np.abs(catalog["ra"]-ra)*np.cos(np.deg2rad(dec))>1./3600)*(np.abs(catalog["dec"]-dec)>1./3600)
        #catalog = catalog[no_self_object]
        #catalog.sort(order='mag')

    
    if (debug): print ("Once removed the object", catalog)

    
    if (len(catalog)>0):
        ref1_pix = wcs.wcs_world2pix(np.array([[catalog["ra"][0], catalog["dec"][0]]], np.double), 1)
    if (len(catalog)>1):
        ref2_pix = wcs.wcs_world2pix(np.array([[catalog["ra"][1], catalog["dec"][1]]], np.double), 1)

    # Mark and label reference stars
    #If we have 1, we mark it
    if (len(catalog)>0):
        s1ra = catalog[0]["ra"]
        s1dec = catalog[0]["dec"]
        s1= SphericalCircle(( s1ra* u.deg,  s1dec* u.deg), 2 * u.arcsec,
                             edgecolor='b', facecolor='none',
                             transform=ax.get_transform('fk5'))
        ax.add_patch(s1)
        plt.annotate("S1", xy=(ref1_pix[0][0]+3, ref1_pix[0][1]+3),  xycoords='data', color="b")

    #IFfwe have 2, we mark them as well
    if (len(catalog)>1):
        s2ra = catalog[1]["ra"]
        s2dec = catalog[1]["dec"]
        s2= SphericalCircle(( s2ra* u.deg,  s2dec* u.deg), 2 * u.arcsec,
                             edgecolor='r', facecolor='none',
                             transform=ax.get_transform('fk5'))
        
        ax.add_patch(s2)
        plt.annotate("S2", xy=(ref2_pix[0][0]+3, ref2_pix[0][1]+3),  xycoords='data', color="r")
        
    # Set size of window (leaving space to right for ref star coords)
    plt.subplots_adjust(right=0.65,left=0.1, top=0.99, bottom=0.01)
    
    #Write the starlist
        
    #If no magnitude was supplied, just do not put it on the chart.
    if not np.isnan(mag):
        target_mag = "mag=%.2f"%mag
    else:
        target_mag = ""
        
        
    # List name, coords, mag of references etc
    plt.text(1.02, 0.85, name, fontweight='bold', transform=ax.transAxes)

    plt.text(1.02, 0.85, name, transform=ax.transAxes, fontweight='bold')
    plt.text(1.02, 0.80, "%s"%target_mag, transform=ax.transAxes, fontweight='bold')
    plt.text(1.02, 0.75, "%.5f %.5f"%(ra, dec),transform=ax.transAxes)
    rah, dech = deg2hour(ra, dec)
    plt.text(1.02, 0.7,rah+"  "+dech, transform=ax.transAxes)
    
    
    #Put the text for the offset stars.
    if (len(catalog)>0):
        ofR1 = get_offset(catalog["ra"][0], catalog["dec"][0], ra, dec)
        S1 = deg2hour(catalog["ra"][0], catalog["dec"][0], sep=":")

        plt.text(1.02, 0.60,'S1, mag=%.2f'%catalog["mag"][0], transform=ax.transAxes, color="b")
        plt.text(1.02, 0.55,'%s %s'%(S1[0], S1[1]), transform=ax.transAxes, color="b")
        plt.text(1.02, 0.5,"E: %.2f N: %.2f"%(ofR1[0], ofR1[1]),transform=ax.transAxes, color="b")

    if (len(catalog)>1):
        ofR2 = get_offset(catalog["ra"][1], catalog["dec"][1], ra, dec)
        S2 = deg2hour(catalog["ra"][1], catalog["dec"][1], sep=":")

        plt.text(1.02, 0.4,'RS, mag=%.2f'%catalog["mag"][1], transform=ax.transAxes, color="r")
        plt.text(1.02, 0.35,'%s %s'%(S2[0], S2[1]), transform=ax.transAxes, color="r")
        plt.text(1.02, 0.3,"E: %.2f N: %.2f"%(ofR2[0], ofR2[1]),transform=ax.transAxes, color="r")
    


    
    #Print starlist in the right format for each telescope
    if telescope == "Keck":
        commentchar = "#"
        separator = ""
    if telescope == "P200":
        commentchar = "!"
        separator = "!"
    if telescope == "Calar":
        commentchar = "|"
        separator = "|"
        
    if (len(catalog)>0 and (print_starlist or not starlist is None)):
        ra_h, dec_h = deg2hour(ra, dec, sep=":")

        print ( "%s %s %s %s %s %s %s %s %s"%("# Object".ljust(20), separator, "alpha".ljust(11), separator, "delta".ljust(12), separator, "eq".ljust(6), separator, "camp de format lliure") )
        
        print ( "%s %s %s %s %s %s 2000.0 %s"%(name.ljust(20), separator, ra_h, separator, dec_h, separator, commentchar) )
        S1 = deg2hour(catalog["ra"][0], catalog["dec"][0], sep=":")
        print ( "{:s} {:s} {:s} {:s} {:s} {:s} 2000.0 {:s}  raoffset={:.2f} decoffset={:.2f} r={:.1f} ".format((name+"_S1").ljust(20), 
                separator, S1[0], separator, S1[1], separator, commentchar, ofR1[0], ofR1[1], catalog["mag"][0]) )
    
    if (len(catalog)>1 and (print_starlist or not starlist is None)):
        S2 = deg2hour(catalog["ra"][1], catalog["dec"][1], sep=":")
        print ( "{:s} {:s} {:s} {:s} {:s} {:s} 2000.0 {:s}  raoffset={:.2f} decoffset={:.2f} r={:.1f} ".format((name+"_S2").ljust(20), 
                separator, S2[0], separator, S2[1], separator, commentchar, ofR2[0], ofR2[1], catalog["mag"][1]) )
     
    r, d = deg2hour(ra, dec, sep=" ")
    
    #Write to the starlist if the name of the starlist was provided.
    if (not starlist is None) and (telescope =="Keck"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2}  2000.0 # {3} \n".format(name.ljust(17), r, d, target_mag) ) 
            if (len(catalog)>0):
                f.write ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"_S1").ljust(17), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"_S2").ljust(17), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')

    if (not starlist is None) and (telescope =="P200"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2}  2000.0 ! {3}\n".format(name.ljust(19), r, d, target_mag) )
            if (len(catalog)>0):
                f.write ( "{:s} {:s} {:s}  2000.0 ! raoffset={:.2f} decoffset={:.2f} r={:.1f}  \n".format( (name+"_S1").ljust(19), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s} {:s}  2000.0 ! raoffset={:.2f} decoffset={:.2f} r={:.1f}  \n".format( (name+"_S2").ljust(19), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')   

    if (not starlist is None) and (telescope =="Calar"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2} %s 2000.0 %s {3}\n".format(separator, separator, name.ljust(19), r, d, target_mag) )
            if (len(catalog)>0):
                f.write ( "{:s} {:s} {:s} %s 2000.0 %s raoffset={:.2f} decoffset={:.2f} r={:.1f}  \n".format( (name+"_S1").ljust(19), separator, S1[0], separator, S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s} {:s} %s 2000.0 %s raoffset={:.2f} decoffset={:.2f} r={:.1f}  \n".format( (name+"_S2").ljust(19), separator, S2[0], separator, S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')   
            
    # Save to pdf
    pylab.savefig(os.path.join(directory, str(name+'_finder.%s'%finder_ext)))
    if debug: print ("Saved to %s"%os.path.join(directory, str(name+'_finder.%s'%finder_ext)))
    pylab.close("all")

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description=\
        '''

        Creates the finder chart for the given RA, DEC and NAME.
        
        Usage: finder.py <RA> <Dec> <Name> <rad [deg]> <telescope [P200|Keck]>
            
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
        
    print ("Usage: finder_chart.py <RA> <Dec> <Name>  <rad [deg]> <telescope [P200|Keck]>")
    
    #Check if correct number of arguments are given
    if len(sys.argv) < 4:
    	print ("Not enough parameters given. Please, provide at least: finder_chart.py <RA> <Dec> <Name>")
    	sys.exit()
     
    ra = np.double(sys.argv[1])
    dec = np.double(sys.argv[2])
    name = str(sys.argv[3])
    if (len(sys.argv)>=5):
        rad = np.double(sys.argv[4])
        if (rad > 15./60):
            print ('Requested search radius of %.2f arcmin is larger than 15 arcmin. Not sure why you need such a large finder chart... reducing to 10 armin for smoother operations...'%(rad * 60))
            rad = 10./60
    else:
        rad = 2./60
        
    print ('Using search radius of %.1f arcsec.'%(rad*3600))

        
    if (len(sys.argv)>5):
        telescope = sys.argv[5]
    else:
        telescope = "P200"
        print ('Assuming that the telescope you observe will be P200. If it is "Keck", please specify otherwise.')

    
    get_finder(ra, dec, name, rad, telescope=telescope, debug=False, minmag=7, maxmag=15)


