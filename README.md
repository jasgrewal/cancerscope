# Cancerscope for SCOPE
[![pypi](https://badge.fury.io/py/cancerscope.svg)](https://pypi.python.org/pypi/cancerscope)
[![Coverage Status](https://coveralls.io/repos/github/jasgrewal/cancerscope/badge.svg?branch=master)](https://coveralls.io/github/jasgrewal/cancerscope?branch=master)
[![build_status](https://travis-ci.org/jasgrewal/cancerscope.svg?branch=master)](https://travis-ci.org/jasgrewal/cancerscope)
[![Documentation Status](https://readthedocs.org/projects/cancerscope/badge/?version=latest)](http://cancerscope.readthedocs.io/?badge=latest)
[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)    
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
 

SCOPE, Supervised Cancer Origin Prediction using Expression, is a method for predicting the tumor type (or matching normal) of an RNA-Seq sample.  
SCOPE's python package, **cancerscope**, allows users to pass the RPKM values with matching Gene IDs and receive a set of probabilities across 66 different categories (40 tumor types and 26 healthy tissues), that sum to 1. Users can optionally generate plots visualizing each sample's classification as well.  
 
Since SCOPE is an ensemble-based approach, it is possible to train additional models and include them in the ensemble that SCOPE uses (Instructions forthcoming).  

## This release contains   
- [x] Setup Tests    
- [ ] Tutorial   
- [x] License   
- [x] Model files setup   
- [ ] Landscape Code Health

## Installation   
Before installing **cancerscope**, you will need to install the correct version of the packages [lasagne](https://lasagne.readthedocs.io/en/latest/) and [theano](https://pypi.org/project/Theano/).  
`pip install --upgrade https://github.com/Theano/Theano/archive/master.zip`  
`pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`  

### Automated Install   
Once you have the latest lasagne and theano python packages installed, you can set up **cancerscope** using the command `pip install cancerscope`.  

At initial install, cancerscope will attempt to download the models needed for prediction. This may take a while depending on your internet connection (3-10 minutes). Please ensure you have a reliable internet connection and atleast 5 GB of space before proceeding with install.   

## Setup and Usage  
To get started with SCOPE, launch a python instance and run:  
`>>> import cancerscope`  

Incase the download was unsuccessful at the time of package install, the first time you import cancerscope, the package will attempt to set up a local download of the models needed for prediction. Please be patient as this will take a while (3-10 minutes).    

### Data import  
cancerscope reads in input from `.txt` files. Columns should be tab-separated, with unique sample IDs. The first column is always the Gene identifier (Official HUGO ID, Ensemble Gene ID, or Gencode). An example is shown with the first 3 rows of input.  

| Gene Name | Sample 1 | Sample 2 | ... |  
|---|---|---|---|
|ENSG000XXXXX| 0.2341 | 9451.2 | .... | 

### Prediction - Example  

### Visualizing or exporting results - Example  

## Folder descriptors  
All scripts required to run SCOPE are [included](cancerscope).

## Citing cancerscope  
If you have used this package for any academic research, it would be great if you could cite the associated paper.  
A bibtex citation is provided for your ease of use:  
`(paper currently embargoed)`

## License  
cancerscope is distributed under the terms of the [MIT](https://opensource.org/licenses/MIT) license.  

## Issues  
If you encounter any problems, please contact the developer and provide detailed error logs and description [here](https://github.com/jasgrewal/cancerscope/issues).  

## Common Errors  
Theano is a bit finicky when working with the cudnn backend, and may sometimes throw errors at you due to version conflicts. Here's a common one if you are setting up **cancerscope** in GPU-friendly environment.  
`RuntimeError: Mixed dnn version. The header is version 5110 while the library is version 7401.`  
- Please ensure that only 1 cudnn version exists on your system.  
- Cancerscope has been developed and tested with cudnn-7.0 (v3.0)  

