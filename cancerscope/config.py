import os
import glob
### Global variables 
SCOPEMODELS_DATADIR = os.path.abspath(os.path.dirname(__file__)) + "/data/"
SCOPEMODELS_FILELIST_DIR = os.path.abspath(os.path.dirname(__file__))
SCOPEMODELS_LIST = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scope_files.txt')
REF_LABELCOUNTS=glob.glob("/".join([os.path.abspath(os.path.dirname(__file__)), "/data/rm500/dataset_train/"]) + "*/dict_labcount*txt")[0]
REF_DISEASECODES= "/".join([os.path.abspath(os.path.dirname(__file__)),'resources/diseasetypes_codes.txt'])

### Compile list of models used in training  
def getmodelsdict():
	modelOptions = {}
	with open(SCOPEMODELS_LIST, 'r') as f:
		for line in f:
			if line.strip()!='':
				modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
				modelOptions[modelname] = (url, expectedFile, expectedmd5)
	return modelOptions

