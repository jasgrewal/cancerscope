import os, sys
import cancerscope
import tempfile
import unittest
import gc
import numpy as np
import pandas as pd
from numbers import Number
import glob

pckg_dir = os.path.dirname(sys.modules["cancerscope"].__file__)
SCOPEMODELS_LIST = os.path.join(pckg_dir, "/resources/scope_files.txt")

class testEnsemble(unittest.TestCase):
	def test_ensemblewith2models(self):
		my_test_file = "/".join([os.path.dirname(sys.modules["cancerscope"].__file__), "../tests/data/test_tcga.txt"])
		my_models = ["v1_rm500", "v1_rm500dropout"]
		scope_ensemble_obj = cancerscope.scope()
		test_X = scope_ensemble_obj.load_data(my_test_file) # X, samples, features_test, in_genecode
		self.assertEqual(test_X[-2][-1], "ZRANB1_ENSG00000019995")
		## Process input file and get predictions from the selected 2 models  
		preds_df_from_file = scope_ensemble_obj.get_predictions_from_file(my_test_file, modelnames=my_models)
		### Compare to getting output from the x data object itself  
		test_x, test_samples, test_features, test_genecode = scope_ensemble_obj.load_data(my_test_file)
		preds_df_from_xdat = scope_ensemble_obj.predict(X = test_x, x_features = test_features, x_features_genecode = test_genecode, x_sample_names=test_samples, modelnames=my_models)
		print(preds_df_from_xdat); print(round(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[1],12))
		self.assertEqual(preds_df_from_file[preds_df_from_file["rank_pred"]==1].label.tolist(), ['TFRI_GBM_NCL_TS', 'TFRI_GBM_NCL_TS'])
		self.assertEqual(preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].label.tolist(), ['TFRI_GBM_NCL_TS', 'TFRI_GBM_NCL_TS'])
		self.assertEqual(round(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[0],12), 0.576824542222)
		self.assertEqual(round(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[1],12), 0.516448831323)
		self.assertEqual(round(preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].pred.tolist()[1],12), 0.516448831323)
		self.assertEqual(round(preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].pred.tolist()[0],12), 0.576824542222)
		self.assertEqual(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[0], preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].pred.tolist()[0])
		self.assertEqual(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[1], preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].pred.tolist()[1])
		### Plot models  
		plotdir = tempfile.mkdtemp()
		preds_df_plotted = scope_ensemble_obj.predict(X = test_x, x_features = test_features, x_features_genecode = test_genecode, x_sample_names=test_samples, modelnames=my_models, outdir = plotdir)
		self.assertEqual(glob.glob(plotdir + "/SCOPE_allPredictions.txt")[0], plotdir + "/SCOPE_allPredictions.txt")
		self.assertEqual(glob.glob(plotdir + "/SCOPE_topPredictions.txt")[0], plotdir + "/SCOPE_topPredictions.txt")
		preds_df_from_plots = pd.read_csv(glob.glob(plotdir + "/SCOPE_topPredictions.txt")[0], delimiter="\t")
		self.assertTrue(preds_df_from_xdat[['sample_ix', 'label', 'freq', 'models','rank_pred','sample_name']].equals(preds_df_from_plots[['sample_ix', 'label', 'freq', 'models','rank_pred','sample_name']]))
		self.assertTrue(preds_df_from_file[['sample_ix', 'label', 'freq', 'models','rank_pred','sample_name']].equals(preds_df_from_plots[['sample_ix', 'label', 'freq', 'models','rank_pred','sample_name']]))
		self.assertEqual(len(glob.glob(plotdir + "/SCOPE_sample*")), 2)
		
	def localtest_samplefilereading(self):
		my_test_file = "/".join([os.path.dirname(sys.modules["cancerscope"].__file__), "../tests/data/test_tcga.txt"]) #read_ensg_input.txt"])
		"""Test if a testX.txt file can be read in and mapped to the correct gene names"""
		scope_ensemble_obj = cancerscope.scope()
		test_X = scope_ensemble_obj.load_data(my_test_file) # X, samples, features_test, in_genecode
		self.assertEqual(test_X[-2][-1], "ZRANB1_ENSG00000019995")
		
		## Process input file and get predictions from all 5 models
		preds_df_from_file = scope_ensemble_obj.get_predictions_from_file(my_test_file)
		### Compare to getting output from the x data object itself  
		test_x, test_samples, test_features, test_genecode = scope_ensemble_obj.load_data(my_test_file) # 
		preds_df_from_xdat = scope_ensemble_obj.predict(X = test_x, x_features = test_features, x_features_genecode = test_genecode, x_sample_names=test_samples)
		self.assertEqual(preds_df_from_file[preds_df_from_file["rank_pred"]==1].label.tolist(), ['BLCA_TS', 'ESCA_EAC_TS'])
		self.assertEqual(preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].label.tolist(), ['BLCA_TS', 'ESCA_EAC_TS'])
		self.assertEqual(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[0], 0.26819298484099996)
		self.assertEqual(preds_df_from_file[preds_df_from_file["rank_pred"]==1].pred.tolist()[1], 0.562124497548)
		self.assertEqual(preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].pred.tolist()[1], 0.562124497548)
		self.assertEqual(preds_df_from_xdat[preds_df_from_xdat["rank_pred"]==1].pred.tolist()[0], 0.26819298484099996)
	
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
	
	def test_rm500(self):
		"""Test if all models can be downloaded correctly"""
		my_test_file = "/".join([os.path.dirname(sys.modules["cancerscope"].__file__), "../tests/data/test_tcga.txt"])
		scope_ensemble_obj = cancerscope.scope()
		test_X = scope_ensemble_obj.load_data(my_test_file) # X, samples, features_test, in_genecode
		## Get the model of interest
		model_name = "v1_rm500"
		model_in = ""
		query_localdirs = cancerscope.get_models.findmodel(os.path.dirname(cancerscope.__file__), model_name)
		if query_localdirs is not None:
			model_in = query_localdirs[model_name]
		else:
			model_in = cancerscope.get_models.downloadmodel(model_label=model_name)
		self.assertTrue(os.path.isdir(model_in))
		self.assertTrue(os.path.exists("".join([model_in, "/lasagne_bestparams.npz"])))
		"""Test if model can be setup correctly"""
		lmodel = cancerscope.scopemodel(model_in)
		lmodel.fit()
		self.assertEqual(len(lmodel.features), 17688)
		x_input = lmodel.prepare_input_featorders(X=test_X[0], x_features_genecode = test_X[3], x_features=test_X[2])
		"""Test if it predicts properly"""
		allpreds_names = lmodel.predict(x_input, get_all_predictions=True,get_numeric=False, get_predictions_dict=False)
		allpreds_values = lmodel.predict(x_input, get_all_predictions=True,get_numeric=True, get_predictions_dict=False)
		toppreds_names = lmodel.predict(x_input, get_all_predictions=False,get_numeric=False, get_predictions_dict=False)
		toppreds_values = lmodel.predict(x_input, get_all_predictions=False,get_numeric=True, get_predictions_dict=False)
		toppreds_df = lmodel.predict(x_input, get_all_predictions=True,get_numeric=False, get_predictions_dict=True)
		self.assertEqual(len(allpreds_names[0]), 66); self.assertEqual(len(allpreds_names[1]), 66); 
		self.assertEqual(allpreds_values.shape[1],66); 
		self.assertEqual(round(allpreds_values[0][1], 15), round(0.003065253372039,15))
		self.assertEqual(toppreds_names[0], "PAAD_TS"); self.assertEqual(toppreds_names[1], "HNSC_TS")
		self.assertEqual(round(toppreds_values[0],15), round(0.208874390780809,15)); self.assertEqual(round(toppreds_values[1],15), round(0.444162763077693,15))
		self.assertEqual(toppreds_df[0][0][0], toppreds_names[0]);  self.assertEqual(round(float(toppreds_df[0][0][1]), 12), round(toppreds_values[0], 12)); 
		self.assertEqual(toppreds_df[1][0][0], toppreds_names[1]); self.assertEqual(round(float(toppreds_df[1][0][1]), 12), round(toppreds_values[1], 12))
	
	def test_rm500dropout(self):
		"""Test if all models can be downloaded correctly"""
		my_test_file = "/".join([os.path.dirname(sys.modules["cancerscope"].__file__), "../tests/data/test_tcga.txt"])
		scope_ensemble_obj = cancerscope.scope()
		test_X = scope_ensemble_obj.load_data(my_test_file) # X, samples, features_test, in_genecode
		## Get the model of interest
		model_name = "v1_rm500dropout"
		model_in = ""
		query_localdirs = cancerscope.get_models.findmodel(os.path.dirname(cancerscope.__file__), model_name)
		if query_localdirs is not None:
			model_in = query_localdirs[model_name]
		else:
			model_in = cancerscope.get_models.downloadmodel(model_label=model_name)
		self.assertTrue(os.path.isdir(model_in))
		self.assertTrue(os.path.exists("".join([model_in, "/lasagne_bestparams.npz"])))
		"""Test if model can be setup correctly"""
		lmodel = cancerscope.scopemodel(model_in)
		lmodel.fit()
		self.assertEqual(len(lmodel.features), 17688)
		x_input = lmodel.prepare_input_featorders(X=test_X[0], x_features_genecode = test_X[3], x_features=test_X[2])
		"""Test if it predicts properly"""
		allpreds_names = lmodel.predict(x_input, get_all_predictions=True,get_numeric=False, get_predictions_dict=False)
		allpreds_values = lmodel.predict(x_input, get_all_predictions=True,get_numeric=True, get_predictions_dict=False)
		toppreds_names = lmodel.predict(x_input, get_all_predictions=False,get_numeric=False, get_predictions_dict=False)
		toppreds_values = lmodel.predict(x_input, get_all_predictions=False,get_numeric=True, get_predictions_dict=False)
		toppreds_df = lmodel.predict(x_input, get_all_predictions=True,get_numeric=False, get_predictions_dict=True)
		self.assertEqual(len(allpreds_names[0]), 66); self.assertEqual(len(allpreds_names[1]), 66);
		self.assertEqual(allpreds_values.shape[1],66);
		self.assertEqual(round(allpreds_values[0][1], 12),round(0.001936952939 , 12))
		self.assertEqual(toppreds_names[0], "TFRI_GBM_NCL_TS"); self.assertEqual(toppreds_names[1], "TFRI_GBM_NCL_TS")
		self.assertEqual(round(toppreds_values[0], 12), round(0.576824542222, 12)); self.assertEqual(round(toppreds_values[1],12), round(0.516448831323,12))
		self.assertEqual(toppreds_df[0][0][0], toppreds_names[0]);  self.assertEqual(round(float(toppreds_df[0][0][1]), 12), round(toppreds_values[0], 12));
		self.assertEqual(toppreds_df[1][0][0], toppreds_names[1]); self.assertEqual(round(float(toppreds_df[1][0][1]), 12), round(toppreds_values[1], 12))

if __name__ == '__main__':
	unittest.main()
	
