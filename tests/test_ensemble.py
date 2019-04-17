import os, sys
import cancerscope
import tempfile
import unittest
import gc
import numpy as np
from numbers import Number

pckg_dir = os.path.dirname(sys.modules["cancerscope"].__file__)
SCOPEMODELS_LIST = os.path.join(pckg_dir, "/scope_files.txt")

class testEnsemble(unittest.TestCase):
	def test_samplefilereading(self):
		my_test_file = "/".join([os.path.dirname(sys.modules["cancerscope"].__file__), "../tests/data/test_tcga.txt"]) #read_ensg_input.txt"])
		"""Test if a testX.txt file can be read in and mapped to the correct gene names"""
		scope_ensemble_obj = cancerscope.scope()
		test_X = scope_ensemble_obj.load_data(my_test_file) # X, samples, features_test, in_genecode
		self.assertEqual(test_X[-2][-1], "ZRANB1_ENSG00000019995")
		
		## Process input file and get predictions from all 5 models
		toppreds_dict = scope_ensemble_obj.get_predictions_from_file(my_test_file)
		print(toppreds_dict)
		toppreds_numeric_dict = scope_ensemble_obj.get_predictions_from_file(my_test_file, get_numeric=True, ensemble_score=False)
		allpreds_numeric_dict = scope_ensemble_obj.get_predictions_from_file(my_test_file, get_numeric=True, get_all_predictions=True, ensemble_score=False)
		allpreds_dict = scope_ensemble_obj.get_predictions_from_file(my_test_file, get_numeric=False, get_all_predictions=True, ensemble_score=False)
		print(toppreds_numeric_dict)
		print(allpreds_numeric_dict)
		print(allpreds_dict)
		self.assertEqual(toppreds_dict,0)
		
	def test_downloadAllModels(self):
		"""Test if all models are downloaded locally properly"""
		modelOptions = cancerscope.getmodelsdict()
		assert len(modelOptions.keys()) == 5
		
		scope_ensemble_obj = cancerscope.scope()
		#my_downloaded_models = cancerscope.get_models.getmodel() ## This should retrieve all models
		#print(my_downloaded_models)
		my_downloaded_models = scope_ensemble_obj.downloaded_models_dict
		assert len(my_downloaded_models.keys()) == 5
		for k_model in my_downloaded_models.keys():
			modelname_address = my_downloaded_models[k_model]
			"""For each model, test if model dir exists, then set up the model once"""
			self.assertTrue(os.path.isdir(modelname_address))
			self.assertTrue(os.path.exists("".join([modelname_address, "/lasagne_bestparams.npz"])))
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

if __name__ == '__main__':
	unittest.main()
#	x_test = np.genfromtxt("tests/data/ensg_input.txt", delimiter="\t")
	
