
from setuptools import setup

setup(
   name='utils_git',
   version='0.31',
   description='Module with modeling and data reduction utilities.',
   author='Nadejda Blagorodnova',
   author_email='n.blagorodnova@astro.ru.nl',
   packages=['model', 'photometry', 'utils'],  #same as name
   package_dir={'model': 'src/model', 'photometry':'src/photometry', 'utils':'src/utils'},
   install_requires=['astroquery', 'astropy', 'scipy', 'photutils', 'extinction', 'emcee', 'pysynphot', 'corner'], #external packages as dependencies
   include_package_data = True,
   package_data = {
    # If any package contains *.ini files, include them
    '': ['*.ini'],
    }
  
)
