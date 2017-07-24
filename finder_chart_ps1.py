from __future__ import print_function
import sys, urllib
import numpy as np
import matplotlib.pyplot as plt
import pylab 
from matplotlib.ticker import MaxNLocator
import argparse
from astropy.io import votable
import os
import astropy.wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage.filters import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

def deg2hour(ra, dec, sep=":"):
    
    if ( type(ra) is str and type(dec) is str ):
        return ra, dec
        
    c = SkyCoord(ra, dec, frame='icrs', unit='deg')
    
    ra = c.ra.to_string(unit=u.hourangle, sep=sep, precision=2, pad=True)
    dec = c.dec.to_string(sep=sep, precision=2, alwayssign=True, pad=True)
    
    return str(ra), str(dec)
    
    
def hour2deg(ra, dec):
    
    if ( type(ra) is float and type(dec) is float ):
        return ra, dec
        
    c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
    
    ra = c.ra.deg
    dec = c.dec.deg
    
    return ra, dec


def get_offset(ra1, dec1, ra2, dec2):


    '''
    Returns offset from ra1, dec1 position to ra2, dec2.
    
    returns ( East Offset [arcsec], 	North Offset [arcsec] )

    '''

    ra1, dec1 = hour2deg(ra1, dec1)
    ra2, dec2 = hour2deg(ra2, dec2)
    
    return np.round((ra2 - ra1) * np.cos(np.deg2rad(dec1))*3600,2), np.round((dec2-dec1)*3600, 2)
    
def query_ps1_catalogue(ra, dec, radius_deg, minmag=15, maxmag=18.5):
    '''
    Sends a VO query to the PS1 catalogue.
    Filters the result by mangitude.
    '''

        
    url = "http://gsss.stsci.edu/webservices/vo/CatalogSearch.aspx?CAT=PS1V3OBJECTS&RA=%.5f&DEC=%.5f&SR=%.5f"%(ra, dec, radius_deg)
    
    u = urllib.urlopen(url)
    f = open("/tmp/ps1_cat.xml", "w")
    f.writelines(u.readlines())
    f.close()
    
    # Read RA, Dec and magnitude from XML format USNO catalog
    catalog = votable.parse_single_table("/tmp/ps1_cat.xml").to_table()
    
    #print (catalog["gMeanPSFMag"], catalog["rMeanPSFMag"])
    
    #mask = (catalog["nDetections"]>4) * ((catalog["rMeanPSFMag"] > minmag) * (catalog["rMeanPSFMag"] < maxmag)) | ((catalog["gMeanPSFMag"] > minmag) * (catalog["gMeanPSFMag"] < maxmag))

    mask = (catalog["nDetections"]>4) * (catalog["rMeanPSFMag"] > minmag) * (catalog["rMeanPSFMag"] < maxmag) #*(catalog["rMeanPSFMag"] > minmag) * (catalog["rMeanPSFMag"] < maxmag) 
    catalog = catalog[mask]

    
    newcat = np.zeros(len(catalog), dtype=[("ra", np.double), ("dec", np.double), ("mag", np.float)])
    newcat["ra"] = catalog["RAmean"]
    newcat["dec"] = catalog["DECmean"]
    newcat["mag"] = catalog["rMeanPSFMag"]
    
    #print (newcat)

    return newcat

def get_cutout(ra, dec, name, rad, debug=True):
    
    try:
        ra=float(ra)
        dec=float(dec)
    except:
        ra, dec = hour2deg(ra, dec) 
        
    catalog = query_ps1_catalogue(ra, dec, rad)
    
    if (debug):
        print (catalog)

    
    # Construct URL to download DSS image cutout, and save to tmp.fits
    image_index_url_red = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=i'.format(ra, dec)
    image_index_url_green = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=r'.format(ra, dec)
    image_index_url_blue = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=g'.format(ra, dec)

    urllib.urlretrieve(image_index_url_red, '/tmp/image_index_red.txt')
    urllib.urlretrieve(image_index_url_green, '/tmp/image_index_green.txt')
    urllib.urlretrieve(image_index_url_blue, '/tmp/image_index_blue.txt')

    ix_red = np.genfromtxt('/tmp/image_index_red.txt', names=True, dtype=None)
    ix_green = np.genfromtxt('/tmp/image_index_green.txt', names=True, dtype=None)
    ix_blue = np.genfromtxt('/tmp/image_index_blue.txt', names=True, dtype=None)
    
    image_url = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red=%s&green=%s&blue=%s&filetypes=stack&auxiliary=data&size=%d&ra=%.6f&dec=%.6f"%\
    (ix_red["filename"], ix_green["filename"], ix_blue["filename"], rad*3600*4, ra, dec)

    if (debug):
        print (image_url)
        print ("Downloading PS1 r-band image...")
    urllib.urlretrieve(image_url, '/tmp/tmp_%s.jpg'%name)
    
    
def get_finder(ra, dec, name, rad, debug=False, starlist=None, telescope="P200", directory=".", minmag=15, maxmag=18.5, mag=np.nan):

        
    try:
        ra=float(ra)
        dec=float(dec)
    except:
        ra, dec = hour2deg(ra, dec) 


    catalog = query_ps1_catalogue(ra, dec, (rad/2.)*0.95, minmag=minmag, maxmag=maxmag)
    
    if (debug):
        print (catalog)

    
    if (len(catalog)<2):
        catalog = query_ps1_catalogue(ra, dec, (rad/2.)*0.95, minmag=minmag, maxmag=maxmag+0.5)

    if (len(catalog)<2):
        print ("Restarting with larger radius %.2f arcmin"%(rad*1.5*60))
        get_finder(ra, dec, name, rad*1.5, directory=directory, minmag=minmag, maxmag=maxmag+0.5, mag=mag, starlist=starlist, telescope=telescope)
        return
        
    if (not catalog is None and len(catalog)>0):
        np.random.shuffle(catalog)

    if (debug):
        print (catalog)

    no_self_object = (np.abs(catalog["ra"]-ra)*np.cos(np.deg2rad(dec))>2./3600)*(np.abs(catalog["dec"]-dec)>2./3600)
    catalog = catalog[no_self_object]
    
    
    # Construct URL to download DSS image cutout, and save to tmp.fits
    image_index_url = 'http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={0}&dec={1}&filters=r'.format(ra, dec)
    urllib.urlretrieve(image_index_url, '/tmp/image_index.txt')
    ix = np.genfromtxt('/tmp/image_index.txt', names=True, dtype=None)
    
    image_url = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red=%s&format=fits&size=%d&ra=%.6f&dec=%.6f"%(ix["filename"], rad*3600 *4, ra, dec)

    if (debug):
        print (image_url)
        print ("Downloading PS1 r-band image...")
    urllib.urlretrieve(image_url, '/tmp/tmp.fits')
    
    try:
        ps1_image = fits.open("/tmp/tmp.fits")
    #If there was an error with the fits, we shall go for the DSS image
    except IOError:
        try:
            image_url = 'http://archive.eso.org/dss/dss/image?ra=%.5f&dec=%.5f&x=%.2f&y=%.2f&Sky-Survey=DSS1&mime-type=download-fits' % ((ra), (dec), (rad*60), (rad*60))
            print ("Downloading DSS image...")
            urllib.urlretrieve(image_url, '/tmp/tmp.fits')
        except:
            image_url = 'http://archive.stsci.edu/cgi-bin/dss_search?ra=%.6f&dec=%.6f&generation=DSS2r&equinox=J2000&height=%.4f&width=%.4f&format=FITS' % (ra, dec, rad*60, rad*60)
            urllib.urlretrieve(image_url, '/tmp/tmp.fits')
        
        ps1_image = fits.open("/tmp/tmp.fits")
    
    # Get pixel coordinates of SN, reference stars in DSS image
    wcs = astropy.wcs.WCS(ps1_image[0].header)
    
    if (len(catalog)>0):
        #w = astropy.wcs.find_all_wcs(ps1_image[0].header, relax=True, keysel=None)[0]
        ref1_pix = wcs.wcs_world2pix(np.array([[catalog["ra"][0], catalog["dec"][0]]], np.float_), 1)
    if (len(catalog)>1):
        ref2_pix = wcs.wcs_world2pix(np.array([[catalog["ra"][1], catalog["dec"][1]]], np.float_), 1)
        #ref3_pix = wcs.wcs_world2pix(np.array([[catalog["ra"][2], catalog["dec"][2]]], np.float_), 1)
    target_pix = wcs.wcs_world2pix([(np.array([ra,dec], np.float_))], 1)
    
    
    # Plot finder chart
    
    ps1_image[0].data[ps1_image[0].data>30000] = 30000
    ps1_image[0].data[np.isnan(ps1_image[0].data)] = 0
    
    plt.figure(figsize=(8,6))
    plt.set_cmap('gray_r')
    smoothedimage = gaussian_filter(ps1_image[0].data, 1.5)
    plt.imshow(smoothedimage,origin='lower',vmin=np.percentile(ps1_image[0].data.flatten(), 10), \
    vmax=np.percentile(ps1_image[0].data.flatten(), 98.8))
    
    # Mark target
    plt.plot([target_pix[0,0]+15,(target_pix[0,0]+10)],[target_pix[0,1],(target_pix[0,1])], 'g-', lw=2)
    plt.plot([target_pix[0,0],(target_pix[0,0])],[target_pix[0,1]+10,(target_pix[0,1])+15], 'g-', lw=2)
    plt.annotate(name, xy=(target_pix[0,0], target_pix[0,1]),  xycoords='data',xytext=(22,-3), textcoords='offset points')
    
    # Mark and label reference stars
    if (len(catalog)>0):
        plt.plot([ref1_pix[0,0]+15,(ref1_pix[0,0]+10)],[ref1_pix[0,1],(ref1_pix[0,1])], 'b-', lw=2)
        plt.plot([ref1_pix[0,0],(ref1_pix[0,0])],[ref1_pix[0,1]+10,(ref1_pix[0,1])+15], 'b-', lw=2)
        plt.annotate("R1", xy=(ref1_pix[0,0], ref1_pix[0,1]),  xycoords='data',xytext=(22,-3), textcoords='offset points', color="b")
    if (len(catalog)>1):
        plt.plot([ref2_pix[0,0]+15,(ref2_pix[0,0]+10)],[ref2_pix[0,1],(ref2_pix[0,1])], 'r-', lw=2)
        plt.plot([ref2_pix[0,0],(ref2_pix[0,0])],[ref2_pix[0,1]+10,(ref2_pix[0,1])+15], 'r-', lw=2)
        plt.annotate("R2", xy=(ref2_pix[0,0], ref2_pix[0,1]),  xycoords='data',xytext=(22,-3), textcoords='offset points', color="r")

        
    # Set limits to size of DSS image
    pylab.xlim([0,(ps1_image[0].data.shape[0])])
    pylab.ylim([0,(ps1_image[0].data.shape[1])])
    
    # Plot compass
    plt.plot([(ps1_image[0].data.shape[0])-10,(ps1_image[0].data.shape[0]-40)],[10,10], 'k-', lw=2)
    plt.plot([(ps1_image[0].data.shape[0])-10,(ps1_image[0].data.shape[0])-10],[10,40], 'k-', lw=2)
    plt.annotate("N", xy=((ps1_image[0].data.shape[0])-20, 40),  xycoords='data',xytext=(-4,5), textcoords='offset points')
    plt.annotate("E", xy=((ps1_image[0].data.shape[0])-40, 20),  xycoords='data',xytext=(-12,-5), textcoords='offset points')
    
    # Set axis tics (not implemented correctly yet)
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')
    plt.axes().xaxis.set_major_locator(MaxNLocator(5))
    plt.axes().yaxis.set_major_locator(MaxNLocator(5))
    plt.axes().set_xlabel('%.1f\''%(rad*60))
    plt.axes().set_ylabel('%.1f\''%(rad*60))
    
    # Set size of window (leaving space to right for ref star coords)
    plt.subplots_adjust(right=0.65,left=0.05, top=0.99, bottom=0.05)
    #plt.tight_layout()
    
    if (len(catalog)>0):
        ofR1 = get_offset(catalog["ra"][0], catalog["dec"][0], ra, dec)
        plt.text(1.02, 0.60,'R1, mag=%.2f'%catalog["mag"][0], transform=plt.axes().transAxes, color="b")
        plt.text(1.02, 0.55,"E: %.2f N: %.2f"%(ofR1[0], ofR1[1]),transform=plt.axes().transAxes, color="b")
    if (len(catalog)>1):
        ofR2 = get_offset(catalog["ra"][1], catalog["dec"][1], ra, dec)
        plt.text(1.02, 0.45,'R2, mag=%.2f'%catalog["mag"][1], transform=plt.axes().transAxes, color="r")
        plt.text(1.02, 0.4,"E: %.2f N: %.2f"%(ofR2[0], ofR2[1]),transform=plt.axes().transAxes, color="r")
    
    # List coords, mag of references etc
    plt.text(1.02, 0.85, name, transform=plt.axes().transAxes, fontweight='bold')
    plt.text(1.02, 0.80, "mag=%.1f"%mag, transform=plt.axes().transAxes, fontweight='bold')
    plt.text(1.02, 0.75, "%.5f %.5f"%(ra, dec),transform=plt.axes().transAxes)
    rah, dech = deg2hour(ra, dec)
    plt.text(1.02, 0.7,rah+"  "+dech, transform=plt.axes().transAxes)

    # Save to pdf
    pylab.savefig(os.path.join(directory, str(name+'_finder.pdf')))
    print ("Saved to %s"%os.path.join(directory, str(name+'_finder.pdf')))
    pylab.close("all")
    
    #Print starlist
    if (len(catalog)>0):
        print ( "{0} {1} {2}  2000.0 # ".format(name.ljust(20), *deg2hour(ra, dec, sep=" ")) )
        S1 = deg2hour(catalog["ra"][0], catalog["dec"][0], sep=" ")
        print ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # ".format( (name+"_R1").ljust(20), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
    
    if (len(catalog)>1):
        S2 = deg2hour(catalog["ra"][1], catalog["dec"][1], sep=" ")
        print ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # ".format( (name+"_R2").ljust(20), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))


    if not np.isnan(mag):
        rmag = "r=%.2f"%mag
    else:
        rmag = ""
        
    if (not starlist is None) and (telescope =="Keck"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2}  2000.0 #".format(name.ljust(17), *deg2hour(ra, dec, sep=" ")) + "%s \n"%rmag ) 
            if (len(catalog)>0):
                f.write ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"R1").ljust(17), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s} {:s}  2000.0 raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"R2").ljust(17), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')
    else:
        print (starlist, telescope)

    if (not starlist is None) and (telescope =="P200"):
        with open(starlist, "a") as f:
            f.write( "{0} {1} {2}  2000.0 ! \n".format(name.ljust(17), *deg2hour(ra, dec, sep=" ")) )
            if (len(catalog)>0):
                f.write ( "{:s} {:s}  {:s}  2000.0 ! raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"_S1").ljust(19), S1[0], S1[1], ofR1[0], ofR1[1], catalog["mag"][0]))
            if (len(catalog)>1):
                f.write ( "{:s} {:s}  {:s}  2000.0 ! raoffset={:.2f} decoffset={:.2f} r={:.1f} # \n".format( (name+"_S2").ljust(19), S2[0], S2[1], ofR2[0], ofR2[1], catalog["mag"][1]))
            f.write('\n')        


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description=\
        '''

        Computes the zeropoints for all the images in the folder.
            
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
        
    #Check if correct number of arguments are given
    if len(sys.argv) < 4:
    	print ("Usage: finder.py <RA> <Dec> <Name>  <rad> <telescope>")
    	sys.exit()
     
    ra=sys.argv[1]
    dec=sys.argv[2]
    name=str(sys.argv[3])
    if (len(sys.argv)>=5):
        rad = float(sys.argv[4])
        if (rad > 10./60):
            print ('Search radius larger than 30 arcmin. Not sure why you need such a large finder chart... reducing to 10 armin for smoother operations...')
            rad = 10./60
    else:
        rad = 2./60
    if (len(sys.argv)>5):
        telescope = sys.argv[5]
    else:
        telescope = "P200"
        print ('Assuming that the telescope you observe will be P200. If it is "Keck", please specify otherwise.')

    
    get_finder(ra, dec, name, rad, telescope=telescope, debug=False)

