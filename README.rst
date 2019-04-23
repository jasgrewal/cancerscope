Cancerscope for SCOPE
=====================

[![pypi](https://badge.fury.io/py/cancerscope.svg)](https://pypi.python.org/pypi/cancerscope)

[![Coverage Status](https://coveralls.io/repos/github/jasgrewal/cancerscope/badge.svg?branch=master)](https://coveralls.io/github/jasgrewal/cancerscope?branch=master)

[![build_status](https://travis-ci.org/jasgrewal/cancerscope.svg?branch=master)](https://travis-ci.org/jasgrewal/cancerscope)

[![Documentation Status](https://readthedocs.org/projects/cancerscope/badge/?version=latest)](http://cancerscope.readthedocs.io/?badge=latest)

[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)    

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/cancerscope/)  

SCOPE, Supervised Cancer Origin Prediction using Expression, is a method for predicting the tumor type (or matching normal) of an RNA-Seq sample.  

SCOPE's python package, **cancerscope**, allows users to pass the RPKM values with matching Gene IDs and receive a set of probabilities across 66 different categories (40 tumor types and 26 healthy tissues), that sum to 1. Users can optionally generate plots visualizing each sample's classification as well.  
 
Since SCOPE is an ensemble-based approach, it is possible to train additional models and include them in the ensemble that SCOPE uses (Instructions forthcoming).  

# Installation
==============

Before installing **cancerscope**, you will need to install the correct version of the packages `lasagne](https://lasagne.readthedocs.io/en/latest/) and [theano <https://pypi.org/project/Theano/>`_ and `theano <https://pypi.org/project/Theano/>`_.  

`pip install --upgrade https://github.com/Theano/Theano/archive/master.zip`  

`pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip`  

## Automated Install
====================

Once you have the latest lasagne and theano python packages installed, you can set up **cancerscope** using the command `pip install cancerscope`.  

At initial install, cancerscope will attempt to download the models needed for prediction. This may take a while depending on your internet connection (3-10 minutes). Please ensure you have a reliable internet connection and atleast 5 GB of space before proceeding with install.   

# Setup and Usage
=================

To get started with SCOPE, launch a python instance and run:  

`>>> import cancerscope`  

Incase the download was unsuccessful at the time of package install, the first time you import cancerscope, the package will attempt to set up a local download of the models needed for prediction. Please be patient as this will take a while (3-10 minutes).    

## Data import
==============

cancerscope reads in input from `.txt` files. Columns should be tab-separated, with unique sample IDs. The first column is always the Gene identifier (Official HUGO ID, Ensemble Gene ID, or Gencode). An example is shown with the first 3 rows of input.  

| Gene Name | Sample 1 | Sample 2 | ... |  

|---|---|---|---|

|ENSG000XXXXX| 0.2341 | 9451.2 | .... | 

## Prediction - Example
=======================

Prediction can be performed from a pre-formatted input file, or by passing in the data matrix, list of sample names, list of feature names, and the type of gene names (ENSG, HUGO etc). Please refer to the `tutorial <tutorial/README.md>`_ for more information.  

The commands are as simple as follows:  

`>>> import cancerscope as cs`    

`>>> scope_obj = cs.scope()`   

This will set up the references to the requires SCOPE models.  

Next, you can process the predictions straight from the input file:  

`>>> predictions*from*file = scope*obj.get*predictions*from*file(filename) `    

...or you can pass in the data matrix, list of sample names, list of feature names, the type of gene names (ENSG, HUGO etc), and optionally, the list of sample names.  

`>>> predictions = scope_obj.predict(`  

`	X = numpy*array*X, `  

`	x*features = list*of_features, `

`	x*features*genecode = string_genecode, `

`	x*sample*names = list*of*sample_names)`  

The output will look like this:  

|'ix'|`sample*ix`|`label`|`pred`|`freq`|`models`|`rank*pred`|`sample_name`|

|---|---|---|---|---|---|---|---|

|0|0|BLCA\_TS|0.268193|2|v1\_none17kdropout,v1\_none17k|1|test1|

|1|0|LUSC\_TS|0.573807|1|v1\_smotenone17k|2|test1|

|2|0|PAAD\_TS|0.203504|1|v1\_rm500|3|test1|

|3|0|TFRI\_GBM\_NCL\_TS|0.552021|1|v1\_rm500dropout|4|test1|

|4|1|ESCA\_EAC\_TS|0.562124|2|v1\_smotenone17k,v1\_none17k|1|test2|

|5|1|HSNC\_TS|0.223115|1|v1\_rm500|2|test2|

|6|1|MB-Adult\_TS|0.743373|1|v1\_none17kdropout|3|test2|

|7|1|TFRI\_GBM\_NCL\_TS|0.777685|1|v1\_rm500dropout|4|test2|

Here, 2 samples, called *test1* and *test2*, were processed. The top prediction from each model in the ensemble was taken, and aggregated. 
- For instance, 2 models predicted that 'BLCA\_TS' was the most likely class for *test1*. The column **freq** gives you the count of contributing models for a prediction, and the column **models** lists these models. The other 3 models had a prediction of 'LUSC\_TS', 'PAAD\_TS', and 'TFRI\_GBM\_NCL\_TS' respectively.   
- You can use the rank of the predictions, shown in the column **rank\_pred**, to filter out the prediction you want to use for interpretation.  
- When SCOPE is highly confident in the prediction, you will see **freq** = 5, indicating all models have top-voted for the same class.  

## Visualizing or exporting results - Example
=============================================

**cancerscope** can also automatically generate plots for each sample, and save the prediction dataframe to file. This is done by passing the output directory to the prediction functions:  

`>>> predictions*from*file = scope*obj.get*predictions*from*file(filename, outdir = output_folder) `    

`>>> predictions = scope*obj.predict(X = numpy*array*X, x*features = list*of*features, x*features*genecode = string*genecode, x*sample*names = list*of*sample*names, **outdir = output_folder**)`  

This will automatically save the dataframe returned from the prediction functions as `output*folder + /SCOPE*topPredictions.txt`, and the predictions from all models across all classes as `output*folder + /SCOPE*allPredictions.txt`.  

Sample specific plots are also generated automatically in the same directory, and labelled `SCOPE*sample-SAMPLENAME*predictions.svg`.  

<p align="left">

  <img width="3000mm" height="900mm" src="https://github.com/jasgrewal/cancerscope/blob/master/tutorial/sample_output.svg">
</p>

# Citing cancerscope
====================

If you have used this package for any academic research, it would be great if you could cite the associated paper.  

A bibtex citation is provided for your ease of use:  

`(paper currently embargoed)`

# License
=========

cancerscope is distributed under the terms of the `MIT <https://opensource.org/licenses/MIT>`_ license.  

# Feature requests
==================

If you wished outputs were slightly (or significantly) easier to use, or want to see additional options for customizing the output, please open up a GitHub issue `here <https://github.com/jasgrewal/cancerscope/issues>`_.  

# Issues
========

If you encounter any problems, please contact the developer and provide detailed error logs and description `here <https://github.com/jasgrewal/cancerscope/issues>`_.  

# Common Errors
===============

Theano is a bit finicky when working with the cudnn backend, and may sometimes throw errors at you due to version conflicts. Here's a common one if you are setting up **cancerscope** in GPU-friendly environment.  

`RuntimeError: Mixed dnn version. The header is version 5110 while the library is version 7401.`  
- Please ensure that only 1 cudnn version exists on your system.  
- Cancerscope has been developed and tested with cudnn-7.0 (v3.0)  

pkg_resources.VersionConflict: (pandas xxxx (/path/to/sitepckgs/), Requirement.parse('pandas>=0.23.4'))  
- This error may arise because you have an older version of pandas installed, which conflicts with the plotting library we use (plotnine, this package needs pandas >=0.23.4)  
- You can either manually install plotnine ('pip install plotnine') or update your pandas library ('pip update pandas')  


