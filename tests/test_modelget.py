import os
import cancerscope
import tempfile
import unittest

import numpy as np
from numbers import Number

class testModel(unittest.TestCase):
	def test_listModel(self):
		"""Test if the models web-address file is read properly"""
		modelOptions = {}
		with open(os.path.join(os.path.dirname(cancerscope.get_models.__file__), 'scope_files.txt'), 'r') as f:
			for line in f:
				if line.strip()!= '':
					modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')
					modelOptions[modelname] = (url, expectedFile, expectedmd5)
	
		assert len(modelOptions.keys()) == 5
	
	def test_downloadModel(self):
		"""Test if models can be downloaded correctly"""
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
		random_sample = np.nan_to_num(x_test[0:17688, 1].reshape(1,17688))
		pred_testX = lmodel.predict(random_sample)[0]
		self.assertEqual(pred_testX, "ESCA_TS")
		
		allpreds_testX = lmodel.predict(random_sample, get_all_predictions=True)[0]
		allpredsNumeric_testX = lmodel.predict(random_sample, get_all_predictions=True, get_numeric=True)[0]
		
		self.assertEqual(len(allpreds_testX), 66)
		self.assertEqual(len(allpredsNumeric_testX), 66)
		self.assertEqual(allpreds_testX[0], "BRCA_TS")
		self.assertTrue(isinstance(allpredsNumeric_testX[0], Number))
		
		"""Test if normalization works and is evaluated to correct floatpoint"""
		normalized_testX = lmodel.get_normalized_input(random_sample)[0]
		self.assertEqual(normalized_testX[0],0.60640558591378269)	
		
		"""Test if Jacobian is evaluated correctly"""
		jacobian_test = lmodel.get_jacobian(random_sample)
		class0_highestjacobian = np.amax(jacobian_test[0,:])
		self.assertEqual(jacobian_test.shape[0], 66) ## Num rows = classes
		self.assertEqual(jacobian_test.shape[1], 17688) ## Num columns = genes
		self.assertEqual(class0_highestjacobian, 0.00012377805544765868)

if __name__ == '__main__':
	unittest.main()
	
