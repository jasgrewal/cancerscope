## This script will test whether all models have been downloaded  
## And that they can be initialized correctly  
from unittest import TestCase

import cancerscope  

class TestModel(TestCase):
	def __init__(self):
		self.modelname = TestCase#modelname
	def get_npz(self):
		self.model_npz = cancerscope.get_model_path(self.modelname)
	def test_complete_download(self):
		self.get_npz()
		m = cancerscope.get_model(self.model_npz)
		self.assertTrue(m[0].shape[0] == 17688)


