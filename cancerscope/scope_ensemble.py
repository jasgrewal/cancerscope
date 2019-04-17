import os, sys
import glob, yaml, gzip
import numpy as np
import pandas as pd
### Set theano flag to cpu to prevent cudnn issues with other users
os.environ['THEANO_FLAGS'] = "gpuarray.preallocate=0.2,device=cpu" 
###
import lasagne
import theano
import theano.tensor as T
#from scope_io_modules import *
from scope_io_modules import read_input, map_train_test_features_x, map_gene_names
from scope_normalization_functions import *
import cancerscope
from config import SCOPEMODELS_DATADIR, SCOPEMODELS_LIST, SCOPEMODELS_FILELIST_DIR
import heapq
import copy
#Change default theano GC for memory optimization
theano.config.allow_gc = True
theano.config.allow_pre_alloc = False
theano.config.scan.allow_gc = True

### Check if models have been downloaded already, and if not, do it!
if os.path.isdir(SCOPEMODELS_DATADIR) is False:
	"""If not already downloaded to pckg site, retrieve the models"""
	print("Thankyou for using cancerscope. The initial run requires download of dependent model files. Proceeding with download now...\n\tModels will be downloaded to {0}".format(SCOPEMODELS_DATADIR))
	cancerscope.downloadmodel(targetdir=SCOPEMODELS_DATADIR)

### Collate the directories of all the models
modeldirs_dict = cancerscope.getmodel()

print("Models are downloaded at {0}".format(modeldirs_dict))

def drop_sort_class(matrix4d_of_preds, label_to_drop=None, sortorder="alphabets"):
	### This function takes in a matrix form of the dictionary containing individual models (shape num_models, num_samples, num_features, feature_tuples)
	### It will return either a matrix form of the dictionary with the specified labels to drop removed, 
	### Or optionally can return a amtrix form of the dictionary where the tuples are ordered by either 'numbers' or 'alphabets'
	if sortorder == 'numbers':
		ix_in_tuple_for_order = 1
	else:
		ix_in_tuple_for_order = 0
	df_matrix = copy.deepcopy(matrix4d_of_preds)
	if label_to_drop is None:
		for i in range(0, df_matrix.shape[0]):
			for j in range(0, df_matrix.shape[1]):
				sample_list_of_lab_tuples = df_matrix[i][j]
				new_alphabet_order = sorted(sample_list_of_lab_tuples.tolist(), key=lambda x:-float(x[ix_in_tuple_for_order]))
				df_matrix[i][j] = new_alphabet_order
		return df_matrix
	else:
		if ix_in_tuple_for_order == 1:
			# We need to order the matrix alphabetically prior to deleting any labels
			df_matrix = drop_sort_class(matrix4d_of_preds, label_to_drop=None, sortorder='alphabets')
		for p in label_to_drop:
			## DO something to drop the column
			dropsite_labelix = np.where(df_matrix[0,0,:,0]==p)[0].tolist()
			df_matrix = np.delete(df_matrix, dropsite_labelix, axis=2)
		return drop_sort_class(df_matrix, label_to_drop=None, sortorder=sortorder)

def get_ensemble_score(dict_with_preds, ignorelabs_list=None):
	### This function assumes the label,value tuples for each sample, for each model, are ordered by highest value (ix 0) to lowest value (ix -1) 
	modelnames = dict_with_preds.keys() 
	df_temp_matrix = np.array([dict_with_preds[i] for i in modelnames])
	df_matrix = drop_sort_class(df_temp_matrix, label_to_drop=ignorelabs_list, sortorder='numbers')
	### First get a list of predicted labels and values
	num_models, num_samples, num_preds, num_tuples_preds = df_matrix.shape
	flat_top_preds_values = df_matrix[:, :,0, 1].transpose().flatten() # Across all models, across all samples, the 0th ordered (top level) prediction, numeric value
	flat_top_preds_labels = df_matrix[:, :,0, 0].transpose().flatten() # Across all models, across all samples, the 0th ordered (top level) prediction, label
	## Combine these into a sensible dataframe so as to separate each sample
	topPreds_bymodels_df = pd.DataFrame(np.column_stack([flat_top_preds_labels, flat_top_preds_values, modelnames * num_samples]), columns=['label', 'pred', 'modelname'])
	topPreds_bymodels_df['sample_ix'] = [np.mod(m/num_models, num_models) for m in topPreds_bymodels_df.index.tolist()]
	topPreds_bymodels_df[['pred']] = topPreds_bymodels_df[['pred']].astype(float) # dtype conversion for confidence scores
	## Aggregate based on the predicted labels
	avg_per_label = topPreds_bymodels_df.groupby(['sample_ix', 'label'], as_index=False).mean()
	modelnames_label = topPreds_bymodels_df.groupby(['sample_ix', 'label'], as_index=False)['modelname'].apply(lambda x: "%s" % ','.join(x)).reset_index().rename(columns={0:"models"})
	modelnames_count = topPreds_bymodels_df.groupby(['sample_ix', 'label'], as_index=False)['modelname'].count().rename(columns={"modelname":"freq"})
	joined_df_list = [avg_per_label, modelnames_count ,modelnames_label]
	df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['sample_ix','label'], how='outer'), joined_df_list)
	df_merged.sort_values(by=['sample_ix', 'freq', 'pred'], ascending=False).sort_values('sample_ix').reset_index(drop=True)
	df_merged["rank_pred"] = df_merged.index
	for s in range(0, num_samples):
		df_merged.loc[df_merged.sample_ix == s, "rank_pred"] =  range(1, df_merged[df_merged.sample_ix == s].shape[0]  +1)
	return df_merged

def get_plotting_df(dict_with_preds, x_sample_names=None):
	modelnames = dict_with_preds.keys()
	df_temp_matrix = np.array([dict_with_preds[i] for i in modelnames])
	num_models, num_samples, num_preds, num_tuples_preds = df_temp_matrix.shape
	flat_matrix = df_temp_matrix.flatten()
	for i in range(0, num_models):
		for j in range(0, num_samples):
			if i == 0 and j == 0:
				ret_df = pd.DataFrame(df_temp_matrix[i, j]); ret_df["model"] = modelnames[i]; ret_df["sample_ix"] = j
			else:
				new_data_line = pd.DataFrame(df_temp_matrix[i, j]); new_data_line["model"] = modelnames[i]; new_data_line["sample_ix"] = j
				ret_df = ret_df.append(new_data_line)
	if x_sample_names is not None:
		ret_df['sample_name'] = [x_sample_names[m] for m in ret_df['sample_ix'].tolist()]
	return ret_df

def build_custom_mlp(n_out, num_features, depth, width, drop_input, drop_hidden, input_var=None, is_image=False):
        if is_image:
                network = lasagne.layers.InputLayer(shape=(None, 1, num_features,1), input_var=input_var)
        else:
                network = lasagne.layers.InputLayer(shape=(None,num_features), input_var=input_var)
        if drop_input:
                network = lasagne.layers.dropout(network, p=drop_input)
        nonlin = lasagne.nonlinearities.tanh
        for _ in range(depth):
                network=lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
                if drop_hidden:
                        network = lasagne.layers.dropout(network, p=drop_hidden)
        #Final output layer is softmax, for multiclass output
        softmax = lasagne.nonlinearities.softmax
        network = lasagne.layers.DenseLayer(network, n_out, nonlinearity=softmax)
        return network

def getmodelsdict():
	modelOptions = {}
	with open(SCOPEMODELS_LIST, 'r') as f:
		for line in f:
			if line.strip()!='':
				modelname, url, expectedFile, expectedmd5 = line.strip().split('\t')

				modelOptions[modelname] = (url, expectedFile, expectedmd5)
	return modelOptions

class scope(object):
	def __init__(self):
		self.models_dict = cancerscope.getmodelsdict()
		self.downloaded_models_dict = {}
		[self.downloaded_models_dict.update(m) for m in cancerscope.get_models.getmodel().values()]
		self.model_names = self.downloaded_models_dict.keys()
	def load_data(self, X_file):
		x_dat, x_samples, x_features, x_features_genecode = read_input(X_file)
		sys.stdout.write("\nRead in sample file {0}, \n\tData shape {1}\n\tNumber of samples {2}\n\tNumber of genes in input {3}, with gene code {4}".format(X_file, x_dat.shape, len(x_samples), len(x_features), x_features_genecode))
		return [x_dat, x_samples, x_features, x_features_genecode]
	
	def predict(self, X, x_features, x_features_genecode, x_sample_names=None, outdir=None):
		self.predict_dict = {}
		## Iterating over each model in the ensemble,
		for k_model in self.model_names:
			## Set up each individual model (clas scopemodel)
			lmodel = cancerscope.scopemodel(self.downloaded_models_dict[k_model])
			lmodel.fit()
			## Map training features to the genecode in the input
			mapped_model_features = map_gene_names(lmodel.features, genecode_in = "SCOPE", genecode_out = x_features_genecode)
			feat_subset_x = map_train_test_features_x(X, mapped_model_features, x_features) ## This function will reorder the input based on the training features order, and if missing some genes, will set those to 0)
			self.predict_dict[k_model] = lmodel.predict(feat_subset_x, get_predictions_dict=True)
			if outdir is not None:
				sys.stdout.write("\nWriting predictions to output directory\n")
			### GC to free up memory
			lmodel = None
			for i in range(0,3):
				gc.collect()
			
		## Do something to process output for ensemble score  
		ens_df = get_ensemble_score(self.predict_dict)
		if x_sample_names is not None:
			ens_df['sample_name'] = [x_sample_names[m] for m in ens_df['sample_ix'].tolist()]
		return ens_df
	
	def get_predictions_from_file(self, X_file, outdir=None):
		x_input, x_samples, x_features, x_features_genecode = self.load_data(X_file)
		prediction_dict = self.predict(X=x_input, x_features = x_features, x_features_genecode = x_features_genecode, x_sample_names=x_samples, outdir=None)
		return(prediction_dict)
	
	def plot_samples(self, plot_outdir, X=None):
		if X is not None:
			self.predict_dict = self.predict(X)
		else:
			try:
				self.predict_dict
			except NameError:
				sys.stdout.write("Please use the predict(X) function first.\n")
			else:
				"""TO DO - PLOT each sample in an output directory"""


class scopemodel(object):
	def __init__(self, modeldir):
		self.in_modeldir = modeldir
		
	def load_model(self,indir):
		in_model_npz = glob.glob(indir + "/lasagne_bestparams.npz")[0]
		in_labdict = glob.glob(indir + "/dataset_train/"+ "*/dict_labels*"+".txt")[0]
		in_trainidx = glob.glob(indir + "/dataset_train/" + "*/trainingidxs.txt")[0]
		in_trainfeatures = glob.glob(indir + "/featurelist.txt")[0] 
		
		myargs = self.get_params()
		
		with np.load(in_model_npz) as f: model_params = [f['arr_%d' % i] for i in range(len(f.files))]
		self.trainingdat = myargs['input'][0]
		with open(in_trainfeatures) as f: self.features = f.read().splitlines()
		n_in = model_params[0].shape[0]
		n_class = model_params[-1].shape[0]
		n_hiddenlayers=(len(model_params)/2)-1
		n_hidden = model_params[0].shape[1]
		self.normalization_input = myargs['normtype'][0]
		try:
			with(open(in_labdict)) as f:
				self.numtolabel = dict((int(v.rstrip()), k.rstrip()) for k,v in (line.split('\t') for line in f))
		except ValueError:
			 with(open(in_labdict)) as f:
				self.numtolabel = dict((int(k.rstrip()), v.rstrip()) for k,v in (line.split('\t') for line in f))
		input_var= T.matrix('inputs')
		network = build_custom_mlp(input_var=input_var, is_image=False, n_out=n_class, num_features=n_in, depth=n_hiddenlayers, width=n_hidden, drop_hidden=0, drop_input=0)
		lasagne.layers.set_all_param_values(network, model_params)
		test_fn = theano.function([input_var], lasagne.layers.get_output(network, deterministic=True), allow_input_downcast=True)
		test_max_fn = theano.function([input_var], T.argmax(lasagne.layers.get_output(network, deterministic=True), axis=1), allow_input_downcast=True)
		return([network, test_fn, test_max_fn])	

	def predict(self,X, get_all_predictions=True, get_numeric=True, get_predictions_dict=True):
		X_normed = self.get_normalized_input(X)
		all_predicted = self.pred_fn(X_normed)
		max_predicted_classnum = self.pred_max_fn(X_normed)
		
		if get_predictions_dict is True:
			prediction_labs = [[self.numtolabel[i] for i,j in enumerate(m)] for m in all_predicted]
			sorted_list_of_lab_preds_sublists = [sorted(m, key=lambda x:x[1], reverse=True) for m in [zip(a,b) for a,b in zip(prediction_labs, all_predicted)]]
			#return [prediction_labs, all_predicted]
			return(np.asarray(sorted_list_of_lab_preds_sublists))
		else:
			if get_all_predictions is False:
				if get_numeric is False:
					return([self.numtolabel[i] for i in max_predicted_classnum])
				else:
					max_predicted_index_probability = np.amax(all_predicted, axis=1)
					return(max_predicted_index_probability)
			else:
				if get_numeric is False:
					return [[self.numtolabel[i] for i,j in enumerate(m)] for m in all_predicted]
				else:
					return(all_predicted)
	
	def fit(self):
		model_loaded = self.load_model(self.in_modeldir)
		self.network = model_loaded[0]
		self.pred_fn = model_loaded[1]
		self.pred_max_fn = model_loaded[2]

	def get_params(self):
		in_args = glob.glob(self.in_modeldir + "/argparse_args.txt")[0]
		myargs = yaml.load(open(in_args), Loader=yaml.BaseLoader)
		return(myargs)
	
	def get_normalized_input(self,X):
		print("...Normalization function being applied: {0}".format(self.normalization_input))
		x_test = apply_norm_func(norm_func=self.normalization_input, bysamples=1, xdat=(X))
		return(x_test)
		
	def get_jacobian(self,X_test):
		X_normed = self.get_normalized_input(X_test)
		input_var = T.matrix('inputs')
		layers = lasagne.layers.get_all_layers(self.network)
		outputs = lasagne.layers.get_output(layers, input_var)
		output_final = T.flatten(outputs[-1])
		output_hidden = outputs[-2]
		output_initial = outputs[0]
		
		jacobian = theano.gradient.jacobian(output_final, output_initial)
		run = theano.function([input_var], jacobian, allow_input_downcast=True)
		outjac=run(X_normed)
		#As this tensor is padded with zero matrices on the non-diagnols, convert it to a stacked set of per_sample_per_output x input Jacobians
		outjac=remove_extra_submats(outjac)
		return(outjac)

def remove_extra_submats(bigmat):
	#As the theano jacobian calculation appends null matrices at the non-diagonals, this function returns a stacked set of (output_nodes * number samples) x (input_nodes) as a 2d matrix, from a 3d matrix returned by the jacobian gradient function in theano
	in_samples = bigmat.shape[1]
	out_nodes = bigmat.shape[0] / in_samples
	mymat = bigmat[0:out_nodes,0,:]
	for i in range(1, in_samples):
		mymat =np.append(mymat, bigmat[(0 + out_nodes*i):(out_nodes + out_nodes*i), i, :], axis=0)
	return(mymat)

 
