
from setuptools import setup

setup(
   name='utils_git',
   version='0.1',
   description='Module with modeling and data reduction utilities.',
   author='Nadejda Blagorodnova',
   author_email='n.blagorodnova@astro.ru.nl',
   packages=['model', 'phot', 'utils'],  #same as name
   package_dir={'model': 'src/model', 'phot':'src/phot', 'utils':'src/utils'},
   install_requires=['astropy', 'scipy', 'photutils', 'extinction', 'emcee', 'pysynphot', 'corner'], #external packages as dependencies
   
)
