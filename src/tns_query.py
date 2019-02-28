# -*- coding: utf-8 -*-
"""
Created on Wed Deb 14 14:21:41 2018
Script to query TNS with different parameters:
either for transients discovered between two different dates,
or a cone search radius around a given RA, DEC

@author: nadiablago
"""
from __future__ import print_function

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

import sys, os
#reload(sys)
#sys.setdefaultencoding('utf8')

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.table import Column
import astropy.units as u
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO


def get_tns_date2date(date1, date2):
    '''
    Queries the TNS and obtains the targets reported between two dates.
    It parses the coordinates and transporms them into decimals.
    It writes a csv table with RA, DEC in degrees, which is directly ingestable into a postresql file.
    
    date1: in the format of: YYYY-MM-DD
    date2: in the format of: YYYY-MM-DD
    
    '''
    
    url = "https://wis-tns.weizmann.ac.il/search?&date_start%5Bdate%5D={0}&date_end%5Bdate%5D={1}&format=csv&num_page=1000".format(date1, date2)
    cont_url = urlopen(url)
    cont = cont_url.read() 
    
    t = Table.read(StringIO(cont), format='csv')
    coords = np.array([t["RA"], t["DEC"]]).T
    c = SkyCoord(coords, frame='icrs', unit=(u.hourangle, u.deg))
    
    radeg = Column(c.ra, name='RA')
    decdeg = Column(c.dec, name='DEC')
    
    t.remove_column("RA")
    t.remove_column("DEC")
    t.add_column(radeg, index=1)
    t.add_column(decdeg, index=2)
    
    t.write("tns_query_%s_%s.csv"%(date1, date2), format="csv")

    
    return t
    
def get_tns_ra_dec(ra, dec, rad=15):
    '''
    Queries the TNS and obtains the targets reported for the specified RA, DEC position.
    Provided that ASASSN targets are there, a 7 arcsec position error is expected.
    By default we will use 10 arcsec.
    
    ra: float
        position in degrees
    dec: float
        position in degrees
    rad: float, optional
        Search radius in arcseconds.
    
    '''
    
    url = "https://wis-tns.weizmann.ac.il/search?&name=&ra={0}&decl={1}&radius={2}&coords_unit=arcsec&format=csv".format(ra, dec, rad)
    cont_url = urlopen(url)
    cont = cont_url.read() 

    t = Table.read(StringIO(cont), format='ascii.csv')
      
    if len(t) > 0:
        coords = np.array([t["RA"], t["DEC"]]).T
        c = SkyCoord(coords, frame='icrs', unit=(u.hourangle, u.deg))
        basecoord = SkyCoord(ra, dec,  frame='icrs', unit=(u.deg, u.deg))
        
        #In case there are several objects in the match radius, we select the closest one
        dist = c.separation(basecoord)
        
        closest = t[np.argmin(dist)]
    else:
        closest = None
    
    return closest