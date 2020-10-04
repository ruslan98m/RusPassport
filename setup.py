'''
PassportEye: Python tools for image processing of identification documents

Author: Alexand Dziuba
License: MIT
'''
import sys
from setuptools import setup, find_packages


setup(name='RusPasport',
      version=[ln for ln in open("passporteye/__init__.py") if ln.startswith("__version__")][0].split('"')[1],
      description="Extraction information from russian passports",
      long_description=open("README.rst").read(),
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 4 - Beta',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Image Recognition',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Legal Industry',
          'Intended Audience :: Financial and Insurance Industry'
      ],
      keywords='id-card passport image-processing mrz machine-readable-zone',
      author='Dziuba Alexandr',
      author_email='dziuba.ai@students.dvfu.ru',
      url='https://github.com/Ruslan2288/RusPasport',
      license='MIT',
      packages=find_packages(exclude=['tests']),
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy','cv2', 'PyQt5', 'pytesseract >= 0.2.0'],

     )
