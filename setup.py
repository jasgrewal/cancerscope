"""
Setup script for SCOPE (Python package: 'cancerscope')
"""
from setuptools import setup, find_packages
import os, sys
import re
from tests import *

"""Identify basic variables needd for setup"""
pckg_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(pckg_dir, 'cancerscope/__init__.py'), 'r') as f:
	VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

with open(os.path.join(pckg_dir, 'requirements.txt'), 'r') as f:
	requirements = f.readlines()

with open(os.path.join(pckg_dir, 'README.md'), 'r') as f:
	longdescription = f.read()

""" Post-setup script to download models 
As per answer posted at https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
""" 

""" Setup instructions """ 
setup(name='cancerscope', 
	version=VERSION, 
	setup_requires=['nose>=1.0'],
	description='An RNA-Seq based tool for Supervised Cancer Origin Prediction using Expression',
	long_description=longdescription, 
	long_description_content_type='text/markdown',
	author='Jasleen Grewal',
	author_email='grewalj23@gmail.com', 
	url='https://pypi.org/project/cancerscope/', 
	packages=find_packages(exclude=['tests*']), 
	license='MIT', 
	#dependency_links=['https://github.com/Lasagne/Lasagne/tarball/master#egg=lasagne-0.2.dev1', 'https://github.com/Theano/Theano/tarball/master#egg=theano-0.8'],
	python_requires='>=2.6,<3.0',
	install_requires=requirements, 
	include_package_data=True, zip_safe=False, 
	test_suite='nose.collector', tests_require=['nose'], 
	classifiers = ['Programming Language :: Python :: 2.7', 'Topic :: Scientific/Engineering :: Artificial Intelligence', 'Development Status :: 5 - Production/Stable', 'Intended Audience :: Healthcare Industry', 'Topic :: Scientific/Engineering :: Medical Science Apps.', 'Topic :: Scientific/Engineering :: Bio-Informatics'], 
	package_data = {'cancerscope': ['resources/*.txt', '*.rst']} 
)

