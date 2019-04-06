"""
Setup file for models required by SCOPE
"""
import cancerscope
from cancerscope import SCOPEMODELS_DATADIR, SCOPEMODELS_LIST
import os, sys
import tempfile
import shutil, six
## Get detailed list of models and source files
modelOptions = {}
with open(SCOPEMODELS_LIST, 'r') as f:
	for line in f:
		if line.strip()!= '':
			modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
			modelOptions[modelname] = (url, expectedFile, expectedmd5)

def findmodel(expectedDir, model_label):
	expectedFilename = "model_" + model_label + ".txt"
	modelOptions_local = {}
	if os.path.exists(expectedDir + "/" + expectedFilename):
		with open(expectedDir + "/" + expectedFilename) as f:
			for line in f:
				if line.strip()!= '':
					modelname_, url_, modeldir = line.strip().split('\t')
					modelOptions_local[modelname_] = modeldir
	
	if bool(modelOptions_local) is True:
		if os.path.exists(modelOptions_local[model_label] + "/" + model_label.split("_")[-1] + "/lasagne_bestparams.npz") is True:
			return modelOptions_local
		else:
			return None
	else:
		return None
	
def getmodel(model_label=None):
	"""Base function to retrieve models downloaded to package site directory"""
	model_dirs = {}
	if model_label is None:
		for m, _ in modelOptions.items():
			m_dirtemp = findmodel(expectedDir = SCOPEMODELS_DATADIR, model_label = m)
			if m_dirtemp is not None:
				model_dirs[m] = m_dirtemp
	else:
		m_dirtemp = findmodel(expectedDir = expectedDir, model_label = model_label)
		model_dirs[model_label] = m_dirtemp
	return model_dirs

def downloadmodel(model_label=None, targetdir=None):
	"""
	Query Zenodo and retrieve version-specific model parameter files and metadata
	"""
	global modelOptions
	
	if targetdir is not None:
		tempDir = targetdir
	else:
		tempDir = tempfile.mkdtemp()
	
	if model_label is None:
		for m, _ in modelOptions.items():
			downloadmodel(m, tempDir)
	else:
		assert model_label in modelOptions.keys(), "%s is not a valid option in %s" % (model_label, modelOptions.keys())
		print("Downloading model files for {0} \n\tData Downloaded at: {1}".format(model_label, tempDir))
		url, expectedFile, expectedmd5 = modelOptions[model_label]
		filesToDownload = [(url, expectedFile, expectedmd5)]
		expectedDir = expectedFile.replace(".tar.gz", "")
	
		try:
			cancerscope.utils._downloadFiles(filesToDownload, tempDir)
		except Exception as e:
			print(e)
			exc_info = sys.exc_info(); shutil.rmtree(tempDir); six.reraise(*exc_info)
	
		mainDir = os.path.abspath(tempDir)
		with open(os.path.dirname(cancerscope.__file__) + "/model_" + model_label + ".txt", "w") as f:
			f.write("\t".join([model_label, url, mainDir]))
		return(mainDir)

