import os
import glob, yaml, gzip
import numpy as np
import lasagne
import theano
import theano.tensor as T
#from scope_io_modules import *
from scope_normalization_functions import *
import cancerscope
from config import SCOPEMODELS_DATADIR, SCOPEMODELS_LIST, SCOPEMODELS_FILELIST_DIR
import heapq

#Change default theano GC for memory optimization
theano.config.allow_gc = True
theano.config.allow_pre_alloc = False

### Check if models have been downloaded already, and if not, do it!
if os.path.isdir(SCOPEMODELS_DATADIR) is False:
	"""If not already downloaded to pckg site, retrieve the models"""
	print("Thankyou for using cancerscope. The initial run requires download of dependent model files. Proceeding with download now...\n\tModels will be downloaded to {0}".format(SCOPEMODELS_DATADIR))
	cancerscope.downloadmodel(targetdir=SCOPEMODELS_DATADIR)

### Collate the directories of all the models
modeldirs_dict = cancerscope.get_models.getmodel()

print("Models are downloaded at {0}".format(modeldirs_dict))

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
		print(self.downloaded_models_dict)
		self.model_names = self.downloaded_models_dict.keys()
	def preprocess_X(self, X):
		X = "TO DO - THIS NEEDS TO BE MAPPED"	
	def predict(self, X):
		X = self.preprocess_X(X)
		self.predict_dict = {}
		for k_model in self.model_names:
			lmodel = cancerscope.scopemodel(self.downloaded_models_dict[k_model])
			lmodel.fit()
			self.predict_dict[k_model] = lmodel.predict(X)
		return self.predict_dict
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
		with open(in_trainfeatures) as f: features = f.read().splitlines()
		self.features = features
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

	def predict(self,X, get_all_predictions=None, get_numeric=False):
		X_normed = self.get_normalized_input(X)
		all_predicted = self.pred_fn(X_normed)
		max_predicted = self.pred_max_fn(X_normed)
		if get_all_predictions is None:
			if get_numeric is False:
				return([self.numtolabel[i] for i in max_predicted])
			else:
				return(max_predicted)
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

 
