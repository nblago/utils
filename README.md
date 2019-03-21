utils
=====


finder_chart.py

Queries PS1 survey (in the North) and SkyMapper (in the south), to obtain the positions of offset stars and provide the background image to locate the transient position.


# Nadia's Utils Python package

Util python scripts for data reduction, MCMC modeling and fits utilities.

## Getting Started

There are 3 main packages.
```
model
phot
utils
```

* model: offers the chance to fit black body emission spectrum to photometric datapoints. It converts the photometry into flux measurements. The MCMC code allows to bit: 1 black body, 1 black body + extinction, 2 black bodies, 2 black bodies plus extinction, 1 power law plus extinction.

* phot: module to provide photometri measurements and zeropoint calibration. It has been designed with LCO format in mind, so it may require some changes to adapt to other telescopes.
* utils: package with mixed utilities for management of fits files, creating finder charts from PS1 and SkyMapper images, queries to the Transient Name Server and time handling in astronomy.

### Prerequisites

The following modules are needed:
* astropy
* photutils
* extinction
* emcee
* corner
* pylab
* pysynphot

The pysynphot module requires all the non-hst filters to be available and the PYSYN_CDBS variable to point to the diretory where the files are.
If the BBFit uses filters which are not in the pysynphot file download, the utils/pysynphot_files directory in this github has the most commonly used non-hst filters, downloaded from http://svo2.cab.inta-csic.es/svo/theory/fps/


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

