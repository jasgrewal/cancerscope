import os
import cancerscope
import tempfile
import unittest

import numpy as np
from numbers import Number

class testEnsemble(unittest.TestCase):
	def test_listModel(self):
		"""Test if the models web-address file is read properly"""
		modelOptions = {}
		with open(os.path.join(os.path.dirname(cancerscope.get_models.__file__), 'scope_files.txt'), 'r') as f:
			for line in f:
				if line.strip()!= '':
					modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
					modelOptions[modelname] = (url, expectedFile, expectedmd5)
	
		assert len(modelOptions.keys()) == 5
	
	def test_downloadAllModels(self):
		"""Test if all models can be downloaded correctly"""
		modelOptions={}
		with open(os.path.join(os.path.dirname(cancerscope.get_models.__file__), 'scope_files.txt'), 'r') as f:
			for line in f:
				if line.strip()!= '':
					modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
		
	def test_singleModel(self):
		"""Test if all models can be downloaded correctly"""
		model_in = ""
		query_localdirs = cancerscope.get_models.findmodel(os.path.dirname(cancerscope.__file__), "v1_rm500")
		if query_localdirs is not None:
			model_in = query_localdirs["v1_rm500"]
		else:
			model_in = cancerscope.get_models.downloadmodel(model_label="v1_rm500")
		
		self.assertTrue(os.path.isdir(model_in))
		self.assertTrue(os.path.exists("".join([model_in, "/rm500/lasagne_bestparams.npz"])))
		
		"""Test if model can be setup correctly"""
		lmodel = cancerscope.scopemodel(model_in + "/rm500/")
		lmodel.fit()
	
		self.assertEqual(len(lmodel.features), 17688)

	def test_predict(self):
		"""Test if prediction works"""
		x_test = np.genfromtxt("tests/data/ensg_input.txt",delimiter="\t")
		query_localdirs = cancerscope.get_models.findmodel(os.path.dirname(cancerscope.__file__), "v1_rm500")
		if query_localdirs is not None:
			model_in = query_localdirs["v1_rm500"]
		else:
			model_in = cancerscope.get_models.downloadmodel(model_label="v1_rm500")
		lmodel = cancerscope.scopemodel(model_in + "/rm500/")
		lmodel.fit()
		#print(x_test[0:17688, 1].reshape(17688,1))
		random_sample = np.nan_to_num(x_test[0:17688, 1].reshape(1,17688))
		p1 = lmodel.predict(random_sample)[0]
		self.assertEqual(p1, "ESCA_TS")
		
		p2 = lmodel.predict(random_sample, get_all_predictions=True)[0]
		p3 = lmodel.predict(random_sample, get_all_predictions=True, get_numeric=True)[0]
		
		self.assertEqual(len(p2), 66)
		self.assertEqual(len(p3), 66)
		self.assertEqual(p2[0], "BRCA_TS")
		self.assertTrue(isinstance(p3[0], Number))
		
		"""Test if normalization works and is evaluated to correct floatpoint"""
		p4 = lmodel.get_normalized_input(random_sample)[0]
		self.assertEqual(p4[0],0.60640558591378269)	
		

if __name__ == '__main__':
	unittest.main()
#	x_test = np.genfromtxt("tests/data/ensg_input.txt", delimiter="\t")
	
