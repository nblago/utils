# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:16:50 2015

@author: nadiablago
"""

from astropy.time import Time
import datetime

def date2mjd(datestr):
    t = Time("%s-%s-%s"%(datestr[0:4], datestr[4:6], datestr[6:]), scale='utc')
    return t.mjd
    
def utc2mjd(times):
    '''
    Returns the MJD corresponding to the parameter UTC.
    If no UTC parameter is provided, then the current UTC is assumed.
    '''
    if times is None:
        t = Time(datetime.datetime.utcnow().isoformat(), format='isot', scale='utc')
    else:
        t = Time(times, format='isot', scale='utc')
        
    return t.mjd

def utc2jd(times=None):
    '''
    Returns the JD corresponding to the parameter UTC.
    If no UTC parameter is provided, then the current UTC is assumed.
    '''
    if times is None:
        t = Time(datetime.datetime.utcnow().isoformat(), format='isot', scale='utc')
    else:
        if 'T' in times:
            t = Time(times, format='isot', scale='utc')
        else:
            t = Time(times, format='iso', scale='utc')
        
    return t.jd
    
def mjd2utc(mjd, string=False):
    t = Time(mjd+2400000.5, format='jd', scale="utc")
    
    if (string):
        return t.iso
    else:
        return datetime.datetime.strptime(t.iso, "%Y-%m-%d %H:%M:%S.%f")  
    
def jd2utc(jd, string=False):
    t = Time(jd, format='jd', scale="utc")
    
    if (string):
        return t.iso
    else:
        return datetime.datetime.strptime(t.iso, "%Y-%m-%d %H:%M:%S.%f")   
    
def get_rrlyrae_phase(P, phase_offset):
    '''
    Uses equation 2 from Branimir et. al. 2017 paper to compute the phase:
        
        phase = ( ((t - 2,400,000) mod P ) / P ) + phase_off
        
    Phase shall be in between 0 and 1.
    
    The maximum light is at phase = 0
    
    '''
    phase = ((utc2jd() - 2400000)%P)/P + phase_offset
    
    if phase > 1:
        phase = phase - 1
    elif phase < 0:
        phase = phase + 1
    
    return phase


def jd2hjd(jd, ra, dec):
    '''
    Uses the Palomar observatory location and the coordinates of the object to provide a correction from JD to Heliocentric julian Date.
    
    The ra, dec need to be provided in degrees.
    
    '''
    from astropy import time, coordinates as coord, units as u
    
    coords = coord.SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs')
    palomar_mountain = coord.EarthLocation(lon=243.1361*u.deg, lat=33.3558*u.deg, height=1712*u.m)
    times = time.Time(jd, format='jd', scale='utc', location=palomar_mountain)
    ltt_helio = times.light_travel_time(coords, 'heliocentric')
    hjd = times.utc + ltt_helio
    
    return hjd
    
def t2dt(atime):
    """
    Convert atime (a float) to DT.datetime
    This is the inverse of dt2t.
    assert dt2t(t2dt(atime)) == atime
    """
    year = int(atime)
    remainder = atime - year
    boy = datetime.datetime(year, 1, 1)
    eoy = datetime.datetime(year + 1, 1, 1)
    seconds = remainder * (eoy - boy).total_seconds()
    return boy + datetime.timedelta(seconds=seconds)

def dt2t(adatetime):
    """
    Convert adatetime into a float. The integer part of the float should
    represent the year.
    Order should be preserved. If adate<bdate, then d2t(adate)<d2t(bdate)
    time distances should be preserved: If bdate-adate=ddate-cdate then
    dt2t(bdate)-dt2t(adate) = dt2t(ddate)-dt2t(cdate)
    """
    year = adatetime.year
    boy = datetime.datetime(year, 1, 1)
    eoy = datetime.datetime(year + 1, 1, 1)
    return year + ((adatetime - boy).total_seconds() / ((eoy - boy).total_seconds()))
