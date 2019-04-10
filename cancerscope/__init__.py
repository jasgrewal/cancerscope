""" 
cancerscope provides cancer-type predictions for RNA-Seq samples  
"""
import os, sys
from tests import *

### Global opts
from config import SCOPEMODELS_DATADIR, SCOPEMODELS_LIST, SCOPEMODELS_FILELIST_DIR

### Dependent data
from cancerscope.scopemodel import scopemodel
from cancerscope.utils import *
### Data sources
from cancerscope.get_models import getmodel, findmodel, downloadmodel
from cancerscope.get_models import *

### General Functions  
#from cancerscope.predict import *
from cancerscope import scope

if sys.version_info < (2, 7):
	raise ImportError("Cancerscope module requires Python 2.7 or higher")

__version__ = '0.25'



