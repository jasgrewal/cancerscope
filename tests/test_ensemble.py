import os
import cancerscope
import tempfile
import unittest
import gc
import numpy as np
from numbers import Number

testsuite_dir = os.path.abspath(os.path.dirname(__file__))
SCOPEMODELS_LIST = os.path.join(testsuite_dir, "../cancerscope/scope_files.txt")

class testEnsemble(unittest.TestCase):
	def test_downloadAllModels(self):
		"""Test if all models are downloaded locally properly"""
		modelOptions = {}
		with open(SCOPEMODELS_LIST, 'r') as f:
			for line in f:
				if line.strip()!= '':
					modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
					modelOptions[modelname] = (url, expectedFile, expectedmd5)
	
		assert len(modelOptions.keys()) == 5
		
		my_downloaded_models = cancerscope.get_models.getmodel() ## This should retrieve all models
		print(my_downloaded_models)
		assert len(my_downloaded_models.keys()) == 5
		for k_model in my_downloaded_models.keys():
			modelname_address_pair = my_downloaded_models[k_model]
			"""For each model, test if model dir exists, then set up the model once"""
			self.assertTrue(os.path.isdir(modelname_address_pair[k_model]))
			self.assertTrue(os.path.exists("".join([modelname_address_pair[k_model], "/lasagne_bestparams.npz"])))
			"""TO BE FIXED: THEN SET UP MODEL (memory issues in travis (3 GB RAM there)"""
			#lmodel = cancerscope.scopemodel(modelname_address_pair[k_model])
			#lmodel.fit()
			#self.assertEqual(len(lmodel.features), 17688)
			#del lmodel; lmodel=None
			#for i in range(3):
			#	gc.collect()
	
	def test_singleModel(self):
		"""Test if all models can be downloaded correctly"""
		model_in = ""
		query_localdirs = cancerscope.get_models.findmodel(os.path.dirname(cancerscope.__file__), "v1_rm500")
		if query_localdirs is not None:
			model_in = query_localdirs["v1_rm500"]
		else:
			model_in = cancerscope.get_models.downloadmodel(model_label="v1_rm500")
		
		self.assertTrue(os.path.isdir(model_in))
		self.assertTrue(os.path.exists("".join([model_in, "/lasagne_bestparams.npz"])))
		
		"""Test if model can be setup correctly"""
		lmodel = cancerscope.scopemodel(model_in)
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
		lmodel = cancerscope.scopemodel(model_in)
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
	
