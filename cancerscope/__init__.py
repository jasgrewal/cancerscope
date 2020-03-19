""" 
cancerscope provides cancer-type predictions for RNA-Seq samples  
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
### Global opts
from .config import REF_LABELCOUNTS,SCOPEMODELS_DATADIR, SCOPEMODELS_LIST, SCOPEMODELS_FILELIST_DIR, REF_DISEASECODES, getmodelsdict
from .utils import *
#from cancerscope.utils import *

### Data sources
#from cancerscope.get_models import getmodel, findmodel, downloadmodel
#from cancerscope.get_models import *
from .get_models import getmodel, findmodel, downloadmodel
#from .get_models import *

### Dependent data
#from cancerscope.scope_plots import *
#from cancerscope.scope_ensemble import scopemodel, scope
from .scope_plots import *
from .scope_ensemble import scopemodel, scope

if sys.version_info < (2, 7):
	raise ImportError("Cancerscope module requires Python 2.7*")
__version__ = '1.0'


