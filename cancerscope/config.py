import os
### Global variables 
SCOPEMODELS_DATADIR = os.path.abspath(os.path.dirname(__file__)) + "/data/"
SCOPEMODELS_FILELIST_DIR = os.path.abspath(os.path.dirname(__file__))
SCOPEMODELS_LIST = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scope_files.txt')
### Compile list of models used in training  
def getmodelsdict():
	modelOptions = {}
	with open(SCOPEMODELS_LIST, 'r') as f:
		for line in f:
			if line.strip()!='':
				modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
				modelOptions[modelname] = (url, expectedFile, expectedmd5)
	return modelOptions

