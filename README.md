# SCOPE - Supervised Cancer Origin Prediction using Expression  
A python package that takes the whole transcriptome of a sample as input, and outputs a set of probabilities across 66 different categories (40 tumor types and 26 healthy tissues) that sum to 1.  

You can also train additional models and include them in the ensemble that SCOPE uses (Instructions forthcoming).  

## Installation  
### Manual Setup  
[All scripts required to run SCOPE are included](cancerscope/bin).  

Prior to running [the predictor](cancerscope/bin/lasagne_SCOPE_testsample.py), you will need to ensure you have the correct Python envirnoment set up. SCOPE requires Python 2.7.14.    
- You can set up a custom Conda environment with the required packages from [here](cancerscope/cancerscope/bin/conda_env.yml).  

### Automated Install  
<aside class="notice">
Currently under development  
</aside>

## License  

## Issues  
If you encounter any problems, please contact the developer and provide detailed error logs and description [here](https://github.com/jasgrewal/cancerscope/issues).  


