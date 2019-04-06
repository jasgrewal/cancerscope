SCOPE - Supervised Cancer Origin Prediction using Expression
============================================================

|pypi| |build-status| || |docs| |license|

.. |pypi| image:: https://badge.fury.io/py/cancerscope.svg

   :target: https://pypi.python.org/pypi/cancerscope

   :alt: PyPI Release
   
.. |build-status| image:: https://travis-ci.org/jasgrewal/cancerscope.svg?branch=master

   :target: https://travis-ci.org/jasgrewal/cancerscope

   :alt: Travis CI status

.. |code-health| image:: 

   :target: 

   :alt: Landscape Code Health 
 
.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg

   :target: https://opensource.org/licenses/MIT

   :alt: MIT license


A python package that takes the whole transcriptome of a sample as input, and outputs a set of probabilities across 66 different categories (40 tumor types and 26 healthy tissues) that sum to 1.  

You can also train additional models and include them in the ensemble that SCOPE uses (Instructions forthcoming).  

# Currently Missing
===================
- [ ] Setup Tests    
- [ ] Tutorial   
- [ ] Licensing  
- [ ] Link to Model files  

# Installation
==============
## Automated Install
====================

SCOPE can be installed using the command `pip install cancerscope`    

## Installing Python Dependencies
=================================

If you have Anaconda installed, you can set up the environment using  

`>>> conda create --name cscope --file conda_specs-file.txt`  

# Setup and Usage
=================

To get started with SCOPE, launch a python instance and run:  

`>>> import cancerscope`  

`>>> cancerscope.test_setup()`  


# Folder descriptors
====================

All scripts required to run SCOPE are `included <cancerscope>`_.

Prior to running `the predictor <cancerscope/SCOPE*predict.py>`*, you will need to ensure you have the correct Python envirnoment set up. SCOPE requires Python 2.7.14.
- You can set up a custom Conda environment with the required packages from `here <cancerscope/cancerscope/conda*env.yml>`*.

# License
=========

# Issues
========

If you encounter any problems, please contact the developer and provide detailed error logs and description `here <https://github.com/jasgrewal/cancerscope/issues>`_.  



