utils
=====

# Nadia's Utils Python package

Util python scripts for data reduction, MCMC modeling and fits utilities.

## Getting Started

There are 3 main packages.

- [model](#model)
- [phot](#phot)
- [utils](#utils-1)


* **model**: offers the chance to fit black body emission spectrum to photometric datapoints. It converts the photometry into flux measurements. The MCMC code allows to bit: 1 black body, 1 black body + extinction, 2 black bodies, 2 black bodies + extinction, 1 power law + extinction.

* **phot**: module to provide photometric measurements and zeropoint calibration. It has been designed with LCO format in mind, so it may require some changes to adapt to other telescopes.
* **utils**: package with mixed utilities for management of fits files, creating finder charts from PS1 and SkyMapper images, queries to the Transient Name Server and time handling in astronomy.

### Prerequisites

The following modules are needed:
* astropy
* photutils
* extinction
* emcee
* corner
* pylab
* pysynphot

To install pysynphot, follow the instructions here: https://pysynphot.readthedocs.io/en/latest/

To use BBFit with non-hst filters, you need to download them from the FTP site specified in the [documentation of pysynphot](https://pysynphot.readthedocs.io/en/latest/index.html#pysynphot-installation-setup) and make them available by setting the PYSYN_CDBS variable to point to the directory where the files are.
For example, you can add to your .profile or .bashrc:

```
export PYSYN_CDBS=/Users/myuser/pysynphot_files
```

If the BBFit uses filters which are not in the pysynphot downloaded files, the utils/pysynphot_files directory in this github package has the most commonly used non-hst filters, downloaded from the [Filter Profile Service](http://svo2.cab.inta-csic.es/svo/theory/fps/).

The non-HST filters need to be placed (copied) into the directory:

```
$PYSYN_CDBS/comp/nonhst/.
```


### Installing

To install, clone the directory in your local filesystem.
From the utils directory, run:

```
python setup.py install
```

## Authors

* **Nadejda Blagorodnova** -  [nblago](https://github.com/nblago)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## Usage


### Model

One of the possible usages of BBFit (a part from fitting) is to transform magnitudes into fluxes.
The following example shows how to transform magnitudes specified in two different systems, into fluxes.
If you are unsure of what are the available filter names (as some belong to specific to instruments),
you can check the keys of the dictionary "banddic" inside BBFit.py.

Additionally, if you need to add a filter response that is not provided in `utils/pysynphot_files`,
then just download it from the [Filter Profile Service](http://svo2.cab.inta-csic.es/svo/theory/fps/), place the file into the `nonhst` folder and finally add a new entry to "banddic",
pointing the name of the filter than you want to use and point to the file where the filter response is.

Exmple:

```python
from model import BBFit
import numpy as np

#Create an instance of the class
bb = BBFit.BBFit()

#Set the filter names, magnitudes, errors and magnitude system (abmag or vegamag).
bb.bands = np.array(["B", "V", "I", "sdss,g"])
bb.mags = np.array([10.0, 11.0, 12.0, 11.5])
bb.magerrs = np.array([0.1, 0.1, 0.1, 0.05])
bb.photsys = np.array(["vegamag", "vegamag", "vegamag", "abmag"])

#Add additional parameters, such as redshift or distance (in Mpc), and the extinction to be 
#corrected for the magnitudes (assuming Rv=3.1).
bb.distMpc = 5.1
bb.av_mw = 0.3

#Next, invoke the routine to transform the fluxes. If you select the plotting, an SED will 
#be generated showing the fluxes and wavelengths for each magnitude.
bb.initialize(plot=True)

#Finally, collect the average wavelengths of your filters, the extinction corrected fluxes and errors from the class:
wavelengths = bb.wls
fluxes = bb.fluxes
fluxerrs = bb.fluxerrs
```

For fitting, the file `test_BBFit.py` shows several examples on how to use the MCMC fitter with different models.


### Utils

`finder_chart.py`

Script to make finder charts using the archival images of the PS1 survey (in the North) and SkyMapper (in the South). It also can provide blind offset stars within a given magntiude range (obtained from the Gaia catalogue) and print/store the starlist in Keck/P200 format.
By default it stores the finder chart in a pdf file in the same location where the script is.

Example from the basic usage from the bash console:

```
$python finder_chart.py <RA> <Dec> <Name>  <rad [deg]> <telescope [P200|Keck]>
```

Example within python (more functionality):
```python
import finder_chart

ra =100
dec = 10
rad = 2./60 #2 arcminutes
directory = "/Path/to/finder/pdf/output"
mag = 18.5 #Current mag of the transient
#If you don't want to use PS1/SkyMapper and have your own follow-up image, you can use it here.
image_file = "my_own_fits_file_as_background.fits" 

finder_chart.get_finder(ra=ra, dec=dec, name="MyTransient", rad=rad, \
  directory=".", minmag=15, maxmag=18.5, mag=mag, image_file=image_file)
```


### Phot

Performs aperture photometry on LCO (and few other) telescopes.
