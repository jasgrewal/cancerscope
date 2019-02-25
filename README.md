# SCOPE - Supervised Cancer Origin Prediction using Expression  
A python package that takes the whole transcriptome of a sample as input, and outputs a set of probabilities across 66 different categories (40 tumor types and 26 healthy tissues) that sum to 1.  

You can also train additional models and include them in the ensemble that SCOPE uses (Instructions forthcoming).  

## Installation  
### Manual Setup  
[The files you need for launching SCOPE are at](cancerscope/bin)
Prior to running [the predictor](cancerscope/bin/lasagne_SCOPE_testsample.py), you will need to ensure you have the following Python environment set up (version agnostic update under development):  
Python 2.7.14  
- You can set up a custom Conda environment with the required packages from [here](cancerscope/cancerscope/bin/conda_env.yml).  

### Automated Install  
.. note:: Currently under development  



