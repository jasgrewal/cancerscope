""" 
cancerscope provides cancer-type predictions for RNA-Seq samples  
"""
import os, sys
from tests import *

### Global variables 
SCOPEMODELS_DATADIR = os.path.abspath(os.path.dirname(__file__)) + "/data/"
SCOPEMODELS_LIST = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scope_files.txt')

### Dependent data
from cancerscope.scopemodel import scopemodel
from cancerscope.utils import *
### Data sources
from cancerscope.get_models import getmodel, findmodel

### General Functions  
#from cancerscope.predict import *
from cancerscope import scope

if sys.version_info < (2, 7):
	raise ImportError("Cancerscope module requires Python 2.7 or higher")

__version__ = '0.25'



