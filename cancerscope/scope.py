import cancerscope as cs
from cancerscope import SCOPEMODELS_DATADIR
import os, sys

### Check if models have been downloaded already, and if not, do it!
if os.path.isdir(SCOPEMODELS_DATADIR) is False:
	"""If not already downloaded to pckg site, retrieve the models"""
	print("Thankyou for using cancerscope. The initial run requires download of dependent model files. Proceeding with download now...\n\tModels will be downloaded to {0}".format(SCOPEMODELS_DATADIR))
	cs.downloadmodel(targetdir=SCOPEMODELS_DATADIR)

### Collate the directories of all the models
modeldirs_dict = cs.getmodel()

print("Models are downloaded at {0}".format(modeldirs_dict))


