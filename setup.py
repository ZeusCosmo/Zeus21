#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='zeus21',
          version='0.1dev',
          description='Zeus21: An analytic 21-cm code for cosmic dawn and EoR.',
          url='https://github.com/JulianBMunoz/Zeus21',
          author='Julian B. Muñoz',
          author_email='julianmunoz@austin.utexas.edu',
          license='MIT',
          packages=['zeus21'],
          long_description=open('README.md').read(),
          install_requires=[
           "numpy",
           "scipy",
           "mcfit",
           "classy",
           "numexpr",
           "astropy",
           "powerbox",
           "pyfftw",
           "tqdm"
       ],
)
