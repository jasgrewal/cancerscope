import glob, yaml, gzip
import numpy as np
import lasagne
import theano
import theano.tensor as T
#from lasagna_mlp_definitions import build_custom_mlp
#from scope_io_modules import *
from scope_normalization_functions import *
import heapq


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

class scopemodel(object):
	def __init__(self, modeldir):
		self.in_modeldir = modeldir
		
	def load_model(self,indir):
		in_args = glob.glob(indir + "/argparse_args.txt")[0]
		in_model_npz = glob.glob(indir + "/lasagne_bestparams.npz")[0]
		in_labdict = glob.glob(indir + "/dataset_train/"+ "*/dict_labels*"+".txt")[0]
		in_trainidx = glob.glob(indir + "/dataset_train/" + "*/trainingidxs.txt")[0]
		in_trainfeatures = glob.glob(indir + "/featurelist.txt")[0] 
		myargs = yaml.load(open(in_args), Loader=yaml.BaseLoader)
			
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
		X_normed = apply_norm_func(norm_func=self.normalization_input, xdat = X, bysamples=1) # Should this be normalization_input[0]?
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

	def get_params(self, deep=False):
		in_args = glob.glob(self.in_modeldir + "/argparse_args.txt")[0]
		myargs = yaml.load(open(in_args))
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

	def get_matching_jacobian(self, outjac, y_test):
		#Only return jacobians for the corresponding output class versus all inputs
		outputsize=66
		for s_idx in range(0, outjac.shape(0) - outputsize + 1 , outputsize):
			excerpt = slice(start_idx, start_idx + batchsize)
			ojacob = outjac[excerpt][y_test[i]]
	
	def do_sensitivity_analysis(self, batchsize=40, outdir="/projects/jgrewal_prj/jgrewal_prj_results/tcga_v1/feature_sensitivity/"):
		training_dats = load_data(self.trainingdat)
		inputs = np.concatenate((training_dats[0][0], training_dats[1][0], training_dats[2][0]), axis=0)
		targets = np.concatenate((training_dats[0][1], training_dats[1][1], training_dats[2][1]), axis=0)
		model_base = "smote"
		with open(outdir + model_base + "_features.txt", 'w') as f:
			np.savetxt(f, self.features, fmt="%s", delimiter="\t", newline="\t")
		
		#np.savetxt(myfile, self.features, fmt="%s", delimiter="\t", newline="\t")
		#np.savetxt(myfile, ["\n"], newline="\n", fmt="%s")
		best_modeljacobs = open(outdir + model_base + "_OutputJacobians.txt",'w')
		target_outputs = self.numtolabel.values()
		target_outputs_repeated = np.tile(target_outputs, batchsize)
		counter_loops = 0
		for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
			if counter_loops % 5 == 0:
				if counter_loops != 0:
					print("SAVING FILE: Finished loop {0}".format((counter_loops/5)-1))
					np.save(myfile,batch_jacobs)
					myfile.close()
					print("\n\t{0}\n---".format(outdir + model_base + "_batch" + str((counter_loops/5)-1)))
				myfile = gzip.GzipFile(outdir + model_base + "_batch" + str(counter_loops/5) + ".npy.gz",'w')
			
			excerpt = slice(start_idx, start_idx + batchsize)
			excerpt_jacob = self.get_jacobian(inputs[excerpt])
			targets_repeated = np.repeat(targets[excerpt], 66)
			
			labelled_jacob = np.column_stack((np.asarray([self.numtolabel[i] for i in targets_repeated]), target_outputs_repeated, excerpt_jacob))
			labelled_output_jacob=labelled_jacob[np.where(labelled_jacob[:,1]==labelled_jacob[:,0])]
			
			if counter_loops % 5 == 0:
				batch_jacobs = labelled_jacob
			else:
				batch_jacobs = np.concatenate((batch_jacobs, labelled_jacob),axis=0)
			
			np.savetxt(best_modeljacobs, labelled_output_jacob, fmt="%s", delimiter="\t")	
			counter_loops = counter_loops +1
			#np.savetxt(myfile, labelled_jacob, fmt="%s",delimiter="\t")
		best_modeljacobs.close()
		#myfile2 = gzip.GzipFile(outdir + "smote.npy.gz",'r')
		#tdat = np.load(myfile2)
		return(outdir)	
		
			
def remove_extra_submats(bigmat):
	#As the theano jacobian calculation appends null matrices at the non-diagonals, this function returns a stacked set of (output_nodes * number samples) x (input_nodes) as a 2d matrix, from a 3d matrix returned by the jacobian gradient function in theano
	in_samples = bigmat.shape[1]
	out_nodes = bigmat.shape[0] / in_samples
	mymat = bigmat[0:out_nodes,0,:]
	for i in range(1, in_samples):
		mymat =np.append(mymat, bigmat[(0 + out_nodes*i):(out_nodes + out_nodes*i), i, :], axis=0)
	return(mymat)

if 0:

	#For each of the 40 samples fit, get the top gene and the maximum jacobian for it
	#Sample jacobians are along the 3D diagonal
	for i in range(0,40):
		label_num = dat[1][1][i]
		jac_slice = outjac[(0 + 66*i):(66 + 66*i), i, :]
		print("{0}: {1}: {2}, {3}".format(label_num,features[jac_slice[label_num, :].argmax()], jac_slice.max(), jac_slice[label_num,:].max()))
	
	collatedf = pd.DataFrame()
	collatedf['preds'] = dat[1][1][0:40]
	collatedf['genes'] = ["test"] * len(collatedf)
	for i in range(0,40):
		highestn=100
		label_num = dat[1][1][i]
		jac_slice = outjac[(0 + 66*i):(66 + 66*i), i, :]
		maxn= [np.where(jac_slice[label_num,:]==j)[0][0] for j in heapq.nlargest(highestn, jac_slice[label_num,:])]
		maxn_genes = [features[j] for j in maxn]
		#print("{0} : {1}".format(dat[1][1][i], maxn_genes))
		collatedf = collatedf.set_value(i , 'genes', maxn_genes)
	

	predclass=0
	predlist=[]
	tlist = zip(collatedf[collatedf['preds']==predclass]['genes'])
	[predlist.extend(j) for j in [(i[0]) for i in tlist]]
 
