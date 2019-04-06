"""
Setup script for SCOPE (Python package: 'cancerscope')
"""
from setuptools import setup, find_packages
import os, sys
import re

"""Identify basic variables needd for setup"""
pckg_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(pckg_dir, 'cancerscope/__init__.py'), 'r') as f:
	VERSION = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

with open(os.path.join(pckg_dir, 'requirements.txt'), 'r') as f:
	requirements = f.readlines()

with open(os.path.join(pckg_dir, 'README.rst'), 'r') as f:
	longdescription = f.readlines()

""" Post-setup script to download models 
As per answer posted at https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
""" 
#class my_install(install_data):
#	def run(self):
#		install_data.run(self)
#		post_install()

""" Setup instructions """ 
def package_files(directory):
	paths = []
	for (path, dirs, fnames) in os.walk(directory):
		for filename in fnames:
			paths.append(os.path.join('.', path, filename))
	return paths

extra_files = package_files('cancerscope/bin/')
setup(name='cancerscope', 
	version=VERSION, 
	description='An RNA-Seq based tool for Supervised Cancer Origin Prediction using Expression',
	long_description=longdescription, 
	author='Jasleen Grewal', 
	author_email='grewalj23@gmail.com', 
	url='https://pypi.org/project/cancerscope/', 
	packages=find_packages(exclude=['tests*']), 
	license='MIT', 
	# NOT GOOD practise to use this variable: data_files = [('reference', ['cancerscope/resources/custom_ped_pog_diseasetypes_v8.txt', 'cancerscope/resources/gene_mapping_and_hugo_ids.txt'])],
	python_requires='>=2.6, !=3.*',
	install_requires=requirements, 
	include_package_data=True, zip_safe=False, 
	test_suite='nose.collector', tests_require=['nose'], 
	classifiers = ['Programming Language :: Python :: 2.7', 'Topic :: Scientific/Engineering :: Artificial Intelligence', 'Development Status :: 5 - Production/Stable', 'Intended Audience :: Healthcare Industry', 'Topic :: Scientific/Engineering :: Medical Science Apps.', 'Topic :: Scientific/Engineering :: Bio-Informatics'], 
	#scripts=['bin/'],
	#cmdclass={'install_data': my_install},
	package_data = {'cancerscope': ['cancerscope/resources/*.txt', '*.rst']}, #, '': extra_files},
	entry_points = {'console_scripts': ['cancerscope_setup=cancerscope.command_line:main']},
	project_urls={
		'Documentation': 'https://github.com/jasgrewal/cancerscope',
		'Source': 'https://github.com/jasgrewal/cancerscope/tree/master/cancerscope',
		'PyPi URL': 'https://pypi.org/project/cancerscope/'
	},
	
)


