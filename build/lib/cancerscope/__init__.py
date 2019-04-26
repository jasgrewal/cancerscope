""" 
cancerscope provides cancer-type predictions for RNA-Seq samples  
"""
import os, sys

### Global opts
from config import REF_LABELCOUNTS,SCOPEMODELS_DATADIR, SCOPEMODELS_LIST, SCOPEMODELS_FILELIST_DIR, REF_DISEASECODES, getmodelsdict
from cancerscope.utils import *

### Data sources
from cancerscope.get_models import getmodel, findmodel, downloadmodel
from cancerscope.get_models import *

### Dependent data
from cancerscope.scope_plots import *
from cancerscope.scope_ensemble import scopemodel, scope

if sys.version_info < (2, 7):
	raise ImportError("Cancerscope module requires Python 2.7*")
elif sys.version_info > (3, 0):
	raise ImportError("Cancerscope is not yet configured for Python 3.0 or higher")
__version__ = '0.30.4'



