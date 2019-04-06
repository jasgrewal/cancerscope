#This mlp has following implementations
# - Weight balanced class sizes (error updated accordingly)
# - SMOTE (oversample minority class with noising)
# - oversample minority with replication, synthetic expansion
# - expand training module to make it more customizable, using lasagne
# - Add support for backtracking which features are the most important for each class
# - Add option for getting top-n classes predicted, not just the maximum prediction class

from __future__ import print_function
import os; os.environ["CUDA_VISIBLE_DEVICES"]="-1"
__docformat__ = 'restructedtext en'
import sys, timeit, gzip, argparse, json, shutil, csv, operator
from collections import Counter
import datetime

from itertools import repeat
import numpy as np; import math
from os.path import basename
import pandas as pd
import pickle, cPickle
from sklearn.utils import shuffle
import theano
import theano.tensor as T  
import glob

## Get dependent modules
from io_modules import Unbuffered, load_data, load_txt_input
from argtypes import restricted_float, proper_file, proper_dir, Dictionary, check_balance_type, proper_genecol, proper_delim
from normalization_functions import *
from sampling_algorithms import *
from calculate_stats import *
from plot_confusion import *
from split_input_into_testtrainvalid import *

#Lasagne functions
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from calculate_nn_backprop import *
from lasagna_mlp_definitions import *
from sklearn.metrics import accuracy_score

# Has to be imported last, or segfault
import matplotlib
import matplotlib.pyplot as pyplot
matplotlib.use('Agg')

#genedict_file = os.path.abspath(sys.argv[0] + "/../../reference/gene_mapping_and_hugo_ids.txt")
def readtestinput(datfile, header_status, firstcol_ensg=True, transpose_dat=False, sep='\t'):
        "This function takes in a matrix of format samples (columns) x genes (rows) and generates an array of labels (Y) and an array of corresponding input vectors (X). The first column is the hugo ID, unless it is the ensemble ID followed by the hugo ID, in which case firstcol_ensg = True; the 3rd col onwards are the samples"
	print("Input has header? {0}, first two cols are gene names? {1}, setup is sample x gene? {2}, sep is: {3}".format(header_status, firstcol_ensg, transpose_dat, sep))
	if (datfile.endswith('.pkl')):
		with open(datfile, 'r') as infile:
			dataset=cPickle.load(infile)
			X=dataset[0]
			print(dataset[1])
			labels=dataset[1]
			Y=np.asarray(labels)
			samples=dataset[2]
	else:
		header_status = 0 if header_status == True else 'None'
		indat = pd.read_csv(datfile, sep=sep, header = header_status)
		if transpose_dat is True:
			indat2 = np.transpose(indat)
		else:
			indat2 = indat
		samples = list(indat2.columns)
		if firstcol_ensg is True:
			features_ensg = list(indat2.index); 
			features_hugo = list(indat2.iloc[:,0].values)
			sampleix = 1
		else:
			features_ensg = None
			features_hugo = list(indat2.index)
			sampleix = 0
		indat2 = indat2.replace("NA", float('nan'))
		X = indat2.iloc[:,sampleix:].values
		# Final X must be samples(rows) x genes(cols)
		if X.shape[0] == len(features_hugo):
			X = X.transpose()
	return [X, samples, features_ensg, features_hugo]


#Functions used in preprocessing (expanding dataset, or subsetting features based on an input list)
def make_blind(x, bfrac=0.0):
	blindarray = np.random.choice([0, 1], size=x.shape, p=[1.*bfrac, 1.0-bfrac])
	return(x*blindarray)

def adjust_trainset(x, y, adjust_type,verbose=False):
	print("Generating multiple iters using {0}...".format(adjust_type))
	duped_classx = []; duped_classy = []
	if adjust_type in ('smote', 'dup'):
		biggest_class_size = max(Counter(y).values())
		for ylab in set(y):
			class_indices = np.where(y == ylab)[0]
			subset_classx = np.take(x, class_indices,axis=0)
			classweight = float(biggest_class_size)/float(Counter(y)[ylab])
			#Extra samples required on top of duplicating by an integer factor
			spillover_samples = float(biggest_class_size) - (len(subset_classx)*np.floor(classweight))
			#expand the set
			duped_set = expand_this_set(minority_data_x = subset_classx, type=adjust_type, class_weight=classweight, additional_samples=spillover_samples)
			duped = (len(subset_classx)*np.floor(classweight))
			if verbose:
				print("CLASS {0}".format(ylab))
				print("....extending class {0} size {1} samples by factor of {2}, {3} samples".format(ylab, len(subset_classx), classweight, len(duped_set) ))
			duped_classx.extend(duped_set)
			duped_classy.extend(repeat(ylab, len(duped_set)))
	elif adjust_type == "imblearn":
		duped_classx, duped_classy = smote_this(x,y)
	elif adjust_type == "adasyn":
		duped_classx, duped_classy = adasyn_this(x,y)

	return(duped_classx, duped_classy)

def subset_features_x(x, fullfeatures_named, subsetfeatures_named):
	#Used to select a set of input features in a specific order
	#Format of input is samples x features
	subset_indices = []
	for select_this in subsetfeatures_named:
		subset_indices.append(fullfeatures_named.index(select_this))
	#returned object will have the features ordered in the manner of subsetfeatures_named
	return (np.apply_along_axis(lambda x: x[subset_indices], 1, x))

def padzero_features_x(x, fullfeatures_named, subsetfeatures_named, fillzero=True, default_value=0.0):
	reorder_subset = {subsetfeatures_named.index(i):fullfeatures_named.index(i) for i in subsetfeatures_named if i in fullfeatures_named}
        subset_keys_orderedix = [i[0] for i in sorted(reorder_subset.items(), key=operator.itemgetter(1))]
	subset_values = np.apply_along_axis(lambda x: x[subset_keys_orderedix], 1, x)
	set0_genes = []
        if fillzero:
                for i in range(0, len(fullfeatures_named), 1):
                        if fullfeatures_named[i] not in subsetfeatures_named:
                                subset_values=np.insert(subset_values, i, default_value, axis=1)
				set0_genes.append(fullfeatures_named[i])
        return(subset_values, set0_genes)

class basic_mlp_features(object):
	def __init__(self, learning_rate, L1_reg, L2_reg, n_epochs, dataset_file, batch_size, n_hidden, outdir, misclassflag, hiddenlayers=1, drop_input=0.0, drop_hidden=0.0,writeout=True,shuffle_train=False, normalize=None,normtype=None, keepsmall=False, test_dat = None, features_train=[]):
		self.inputfile = dataset_file
		self.outdir = outdir
		self.learning_rate=learning_rate; self.L1_reg = L1_reg; self.L2_reg = L2_reg
		self.n_epochs=n_epochs
		self.batch_size=batch_size
		self.n_hidden=n_hidden
		self.misclassflag=misclassflag
		self.writeout=writeout
		self.shuffle_train=shuffle_train
		self.rng = None
		#Params for data normalization
		self.normtype=normtype
		self.normalize=normalize
		#Param for data splitting
		self.keepsmall=keepsmall
		#Additional lasagna custom model inputs
		self.hiddenlayers=hiddenlayers
		self.drop_input=drop_input
		self.drop_hidden=drop_hidden
		self.separate_test=False
		if test_dat is not None:
			if(len(test_dat[0]) > 0):
				print("Testing input, assigning vars self.test_x after feat. selection and normalization.")
				self.X_test = np.asarray(test_dat[0])
				self.y_test = np.asarray(test_dat[1])
				self.separate_test=True
			
		#if features_train is not None:
		#if len(features_train) > 0:
		self.features_subset = features_train
		
	def load_mlp_data(self, fullset=False, verbose=True, shuffle_cv=False, save_idxs=False):
		if(self.inputfile.endswith(".txt")):
			self.load_txt_data(fullset=fullset, verbose=verbose, save_full_stats=save_idxs)
		else:
			self.load_pkl_data()
			print("Full set : {0}, saving idxs : {1}".format(fullset, save_idxs))
			if shuffle_cv:
				self.cv_shuffle_pkld_input(fullset=fullset,verbose=verbose, save_full_stats=save_idxs)
			elif fullset:
				self.cv_shuffle_pkld_input(fullset=fullset, verbose=verbose, save_full_stats=save_idxs)
		mlpobj.preprocess_data(verbose=True)

	def cv_shuffle_pkld_input(self, fullset, verbose=False, save_full_stats=False):
		print("Since input was pickled, pooling it back together and re-splitting. \n NO ADDITIONAL NORMALIZATION WILL BE DONE ON THE INPUT!")
		#Need to edit mydatstuff to return a pooled set, resplit, from input :(
		#Not sure if this dict is numeric_label:true_label key:value pair, or vice versa
		self.X_train, self.y_train = self.datasets[0]
		self.X_val, self.y_val = self.datasets[1]
		self.X_test_ds, self.y_test_ds = self.datasets[2]	
		self.x = np.concatenate((self.X_train, self.X_val, self.X_test_ds))
		self.y = np.concatenate((self.y_train, self.y_val, self.y_test_ds))
			
		self.labelnum = [self.labelnames.get(str(i)) for i in self.y]
		print(len(self.labelnum), len(self.y))
		dataseparator = mydatstuff(infile=None,x=self.x, labs=self.labelnum, ligs=self.features, outdir=self.outdir + "/dataset_train", trimflag=True, keepheader=True, drugbinderflag=False,keepsmall=self.keepsmall, fulltrainflag=fullset, classligands=None, testsize=0.2, outprefix="traincv", normtype=self.normtype, normby=self.normalize, makevalid=True,verbose=verbose)
		dataseparator.splitvalidtest()
		self.datasets, self.labelnames, self.labweighted, self.features = dataseparator.return_dataset()
		if save_full_stats:
			dataseparator.writeout() #Save idx's, not the actual dataset
			with open(os.path.join(self.outdir, 'featurelist.txt'), 'w') as f:
                                [ f.write('{0}\n'.format(x)) for x in self.features]					
		#Note that datasets[2], which is test, will be merged with validation if a test set has already been passed
		#This happens at training
			
	def load_pkl_data(self):
		self.datasets = load_data(self.inputfile, make_shared=False)
		indir = os.path.dirname(os.path.abspath(self.inputfile))
		self.labelweights = glob.glob(indir+"/dict_labweights_*")[0]
		self.labelset = glob.glob(indir+"/dict_labels_*")[0]
		self.featurelist = glob.glob(indir+"/featurelist.txt")
		self.labweighted = Dictionary(labelset=self.labelweights)
		self.labelnames = Dictionary(labelset=self.labelset)
		if self.featurelist:
			#print("...read in feature list")
			f = open(self.featurelist[0])
			with open(self.featurelist[0]) as f:
				self.features = [line.rstrip() for line in f]
		
		print("~~~~~LOADED PKL DATA FILE~~~~~~")	
	
	def load_txt_data(self,fullset=False,verbose=True, save_full_stats=False):
		dataseparator = mydatstuff(infile=self.inputfile, outdir=outdir, trimflag=True, keepheader=True, drugbinderflag=False,keepsmall=self.keepsmall, fulltrainflag=fullset, classligands=None, testsize=0.1, outprefix="testcv", normtype=self.normtype, normby=self.normalize, makevalid=True,verbose=verbose)
		dataseparator.splitvalidtest()
		dataseparator.normalize()
		self.datasets, self.labelnames, self.labweighted, self.features = dataseparator.return_dataset()
		if save_full_stats:
			dataseparator.writeout() #Save idx's, not the actual dataset
			
	def load_model(self, best_params_npz, label_dict_file, n_in, n_class):
		self.n_class = n_class; self.is_image=False
		print("Loading model from ... {0}".format(best_params_npz))
		with np.load(best_params_npz) as f:
			best_params = [f['arr_%d' % i] for i in range(len(f.files))]
		#set up network
		self.n_in = best_params[0].shape[0]; self.n_class = best_params[-1].shape[0]
		if self.n_in != n_in:
			print("Expect {0} and input {1}".format(self.n_in, n_in))
			print("WARNING: Passed input is not the same number of feats as trained model")
		self.target_var = T.ivector('targets') ; self.input_var = T.matrix('inputs') #Copied here from train during test run
			
		self.network = build_custom_mlp(input_var=self.input_var,n_out=self.n_class, num_features=self.n_in, depth=self.hiddenlayers, width=self.n_hidden, drop_input=self.drop_input, drop_hidden=self.drop_hidden, is_image=self.is_image)
		self.bestweights = [lasagne.layers.get_all_param_values(self.network)]	
		#Assign trained params to this network
		lasagne.layers.set_all_param_values(self.network, best_params)	
			
		#Update label dict mapping from integer to name
		self.labelnames = Dictionary(labelset=label_dict_file)
		if '0' in self.labelnames:
                        self.labelnames = dict([int(a),b] for a,b in self.labelnames.iteritems())
                else:
                        self.labelnames = dict([int(b),a] for a,b in self.labelnames.iteritems())
				
	def test_data(self, x_test=None, y_test=None):
		if x_test is None:
			x_test = self.X_test; y_test = self.y_test
		else:
			print("Input test set passed, different from runtime training-test sets")
		self.y_all_predicted = self.test_fn(x_test) #All label predictions for x
		self.y_predicted = np.argmax(self.y_all_predicted, axis=1) #Top label predicted for x
		self.y_predicted_probability = np.max(self.y_all_predicted, axis=1) #Probability score for top label 
		self.y_topn_predicted = map(lambda x: x[::-1][0:10], np.argsort(self.y_all_predicted, axis=1, kind='quicksort')) #top 10 labels predicted for x
		col_idx = np.arange(self.y_all_predicted.shape[0])[:,None]
		self.y_topn_predicted_probs=self.y_all_predicted[col_idx,self.y_topn_predicted]
		
	def write_test_output(self, samples=None):
		labelled_test_predict = ["c"] * len(self.y_predicted); labelled_test_predict_topn = ["c"] * len(self.y_predicted)
		for i in range(self.y_predicted.shape[0]):
			labelled_test_predict[i] = str(self.labelnames[self.y_predicted[i]])
			labelled_test_predict_topn[i] = map(lambda x:self.labelnames[x], self.y_topn_predicted[i])
		if not list(self.y_test):
			#If self.y_test is unassigned, re-assign it
			#Set it to a list of NAs
			self.y_test = ["NA"] * len(self.y_predicted)
		else:
			if isinstance(self.y_test[0], np.int):
                        	labelled_test_real = ["c"] * len(self.y_test);
				for i in range(self.y_test.shape[0]):
					labelled_test_real[i] = str(self.labelnames[self.y_test[i]])
			else:
				labelled_test_real = self.y_test
		if not not samples:
			#If a list of samples is provided, append that to self.y_test!
			self.y_test_labelled = zip(labelled_test_real, samples)
			header_all = list(["predicted","confidence","actual","sample"])
		else:
			header_all = list(["predicted","confidence","actual"])
					
		f_out=self.outdir + "/prediction_labels_ordered.txt"
		np.savetxt(f_out, np.column_stack((labelled_test_predict, self.y_predicted_probability, self.y_test_labelled)), fmt='%s',delimiter="\t")
		f_out2=self.outdir + "/prediction_labels_topn.txt"
		np.savetxt(f_out2, np.column_stack((self.y_test_labelled,  np.array(labelled_test_predict_topn))),delimiter="\t", fmt='%s')
		
		f_out3 = self.outdir+"/prediction_values_topn.txt"
		np.savetxt(f_out3, np.column_stack((self.y_test_labelled,np.array(self.y_topn_predicted_probs))),fmt='%s',delimiter="\t")
	
		fout=self.outdir+"/prediction_values_all.txt"
		np.savetxt(fout, np.column_stack(np.asarray(header_all+self.labelnames.values())), fmt='%s',delimiter="\t")
		with open(fout,'a') as f_out4:
			np.savetxt(f_out4, np.column_stack((labelled_test_predict, self.y_predicted_probability, self.y_test_labelled, np.array(self.y_all_predicted))), fmt='%s', delimiter="\t")
			
		print("TEST OUTPUT AT: {0}".format(self.outdir))
	
	def setup_test_data(self, is_train=False):
		#To asses testing, we don't need any dropout in the network
		#So we set deterministic=True (disables dropout)
		self.test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
		self.test_loss = lasagne.objectives.categorical_crossentropy(self.test_prediction, self.target_var)
		self.test_loss = self.test_loss.mean()
		#if len(self.y_test)>0:
		#	self.test_acc = T.mean(T.eq(T.argmax(self.test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)
	#		self.test_acc_fn = theano.function([self.input_var, self.target_var], [self.test_acc], allow_input_downcast=True)
		
		if is_train:
			if len(self.y_test)>0:
				self.test_acc = T.mean(T.eq(T.argmax(self.test_prediction, axis=1), self.target_var), dtype=theano.config.floatX)
				self.test_acc_fn = theano.function([self.input_var, self.target_var], [self.test_acc], allow_input_downcast=True)
				self.val_fn = theano.function([self.input_var, self.target_var], [self.test_loss ,self.test_acc], allow_input_downcast=True, on_unused_input='warn')
			else:
				print("WARNING: TRAINING data but no test set available for evaluation")
		
		self.test_fn = theano.function([self.input_var], lasagne.layers.get_output(self.network, deterministic=True), allow_input_downcast=True)
		
	def preprocess_data(self, verbose):
		if '0' in self.labelnames:
			self.labelnames = dict([int(a),b] for a,b in self.labelnames.iteritems())
		else:
			self.labelnames = dict([int(b),a] for a,b in self.labelnames.iteritems())

		self.sorted_keylabs=[]
		for i in sorted(self.labelnames.keys()):
                                self.sorted_keylabs.append(self.labelnames[i])
				
		self.X_train, self.y_train = self.datasets[0]
		self.X_val, self.y_val = self.datasets[1]
		
		#If test input if already passed, merge read-in dataset test component to validation
                if self.separate_test:
			print("TEST IS passed separately. Merging loaded test set with validation set.")
			self.X_val = np.concatenate((self.X_val, self.datasets[2][0]))
			self.y_val = np.concatenate((self.y_val, self.datasets[2][1]))
		else:
			print("TEST IS used from input")
			self.X_test, self.y_test = self.datasets[2] #NOT SURE IF THIS GOES HERE....
			
		print(self.X_train.shape, self.X_val.shape, self.X_test.shape)	
		if len(self.features_subset) > 0:
			self.X_train = subset_features_x(self.X_train, self.features, self.features_subset)
			self.X_val = subset_features_x(self.X_val, self.features, self.features_subset)
			self.X_test = subset_features_x(self.X_test, self.features, self.features_subset)
			self.features = self.features_subset
			print("___SUBSET INPUT, new training data shape {0}".format(self.X_train.shape))
			with open(os.path.join(self.outdir, 'featurelist.txt'), 'w') as f:
                                [ f.write('{0}\n'.format(x)) for x in self.features]
			
		self.n_class = len(set(self.y_train)) #Training set represents all the classes
		self.n_in = len(self.X_train[0]) #First input sample's length is the number of features (assumes it remains the same for all samples!)

		if isinstance(self.y_test[0], np.int):	
                        print("...--Verified test is correctly labelled--...")
		else:
			print("...--Test labels are alphabetical, mapping to numeric labels--...")
			#Temporarily invert labelnames dict so key:value is labelname:labelnum
			invert_dict = dict([b,a] for a,b in self.labelnames.iteritems())
			numeric_test_real = [0] * len(self.y_test)
			for i in range(self.y_test.shape[0]):
				if self.y_test[i] == "COAD_TS":
					self.y_test[i] = "COADREAD_TS"
				numeric_test_real[i] = invert_dict[self.y_test[i]]
			self.y_test = np.asarray(numeric_test_real) 
			
		if len(set(self.y_train)) != len(set(self.y_test)):
			print("WARNING: Test set does not have all the classes from the training set")
			print(set(self.y_train), set(self.y_test))
		if(verbose):
			print("Input is {0} \n Has {1} classes \n Learning params: \n {2} learning rate \n {3} l1 regression \n {4} l2 regression \n {5} hidden nodes \n {6} features per sample".format(self.inputfile, self.n_class, self.learning_rate, self.L1_reg, self.L2_reg, self.n_hidden, self.n_in))
		
        	if self.misclassflag == "none":
        	        print("No synthetic additions/modifications to the data")
        	elif self.misclassflag != "weight":
			self.X_train, self.y_train = adjust_trainset(self.X_train, self.y_train, self.misclassflag, verbose=verbose)

		if(self.shuffle_train):
                	print("...Shuffling training set")
			self.X_train, self.y_train  = shuffle(self.X_train, self.y_train, random_state=0)
		self.X_train = np.asarray(self.X_train)
		self.y_train = np.asarray(self.y_train)
		#print("Classes being trained:")
		#print(Counter(self.y_train))
		
	def get_classbased_features(self, writeout=False, outsuffix=""):
                if writeout:
                        #Create an output data directory
                        self.featoutdir=self.outdir + "/classbased_features" + outsuffix
                        try:
                                os.stat(self.featoutdir)
                        except:
                                os.mkdir(self.featoutdir)
                        print("...Writing class based features to output directory {0}".format(self.featoutdir))
                for i in range(self.y_predicted.shape[0]):
                        sample=self.y_test_labelled[i]
                        pred_class = self.y_predicted[i]                        
			correslist, featweights =get_weighted_features_per_class(pred_class, self.bestweights[0], feature_name_mapper=self.features)
                        #print("Class {0} influenced by features {1} ".format(self.labelnames[i],Counter(correslist)))
                        ##print("CLASS WEIGHINGS {0}".format(correslist))
                        if writeout:
                                with open(self.featoutdir + "/" + sample[1] + "_" + sample[0] + ".txt", 'w') as f:
                                        for labname, counts in Counter(correslist).most_common():
                                                f.write("{0}\t{1}\t{2}\n".format(sample,labname, counts))
                                with open(self.featoutdir + "/" + sample[1] + "_" + sample[0] + "weights.txt", 'w') as f:
                                        for i in range(len(featweights)):
                                                f.write("{0}\t{1}\t{2}\t{3}\n".format(sample[0],sample[1], featweights[i][0], featweights[i][1]))
		
	def get_performance_stats(self, y_test=None, y_pred=None, outdir=None, outsuffix=None, during_train=True):
		if y_test is None:
			y_test = self.y_test; 
		if y_pred is None:
			y_pred = self.y_predicted
		
		if not during_train:
			self.best_validation_loss=0; self.test_score=0
		
		if isinstance(y_test[0], np.int):
			labelled_test_real = ["c"] * len(y_test)
			for i in range(y_test.shape[0]):
				labelled_test_real[i] = str(self.labelnames[y_test[i]])
		else:
			labelled_test_real = y_test
			
		if isinstance(y_pred[0], np.int):
			labelled_test_predict = ["c"] * len(y_pred); 
			for i in range(y_pred.shape[0]):
				labelled_test_predict[i] = str(self.labelnames[y_pred[i]])
		else:
			labelled_test_predict = y_pred
		
		self.precision_test = calculate_precision(true_labs = labelled_test_real, pred_labs = labelled_test_predict)
		self.recall_test = calculate_recall(true_labs = labelled_test_real, pred_labs = labelled_test_predict)
		self.fscore_test = calculate_fbeta_score(true_labs = labelled_test_real, precision=self.precision_test, pred_labs = labelled_test_predict, recall=self.recall_test)
		self.mean_fscore = 1.00*sum(self.fscore_test.values()) / float(len(self.fscore_test.values()))
		if outdir is not None:
			try:
				self.sorted_keylabs
			except:
				self.sorted_keylabs = set(list(labelled_test_real) + list(labelled_test_predict))
			
			plot_confusion(pc=1, real_y=labelled_test_real,pred_y=labelled_test_predict, sorted_targetnames= self.sorted_keylabs,title="intermediate_test_output", outdir=outdir, outname="confusion"+outsuffix)
				
			pyplot.close()
		#Return accuracy on validation set, accuracy on test set, and fscore average on test set
		return(100.-(self.best_validation_loss * 100.), self.test_score * 100.,self.mean_fscore)
		#return(self.nn_best_val_acc, self.test_score * 100., self.mean_fscore)
			
def get_preds(myargs, modelname, modeldir):
	print("...TESTING input set")
	outdir=os.path.join(os.path.abspath(myargs.outdir[0]) + "/" + modelname + "_TESTPRED_" + myargs.pref[0] + "_" + datetime.datetime.now().strftime('%Y-%m-%d'));
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	
	logfile = open(outdir + '/logfile.txt','w')
	sys.stdout = Unbuffered(sys.stdout, logfile=logfile)
                
	#Load the initialization parameters from training
	in_args = glob.glob(modeldir + "/argparse_args.txt")[0]
	in_model = glob.glob(modeldir + "/lasagne_bestparams.npz")[0]
	in_labdict = glob.glob(modeldir + "/dataset_train/"+ "*/dict_labels*"+".txt")[0]
	in_trainidx = glob.glob(modeldir + "/dataset_train/" + "*/trainingidxs.txt")[0]
	in_trainfeats = glob.glob(modeldir + "/featurelist.txt")[0]
	in_genecode = myargs.genecol[0];
	firstcol_ensg = myargs.multicolgenes
	in_delim = myargs.delim[0]
	in_transpose = myargs.transpose
	myargs_test = myargs #Update current myargs
	myargs = json.load(open(in_args)) #Retrieve old myargs to re-initialize lasagne model
	
	with open(in_trainfeats) as f:
		features_train = [line.rstrip() for line in f]
	
	#Step 1 : Prep input dataset
	print("UPDATE: Reading input from {0}".format(myargs_test.infile[0]))
	x_test, samples_test, features_ensg, features_test = readtestinput(datfile=myargs_test.infile[0], header_status=True, firstcol_ensg=firstcol_ensg, sep=in_delim, transpose_dat = in_transpose)
	print("UPDATE: Successfully read data, input shape is {0}, with {1} samples and {2} genes".format(x_test.shape, len(samples_test), len(features_test)))
	## If input FEATURES are not HUGO, map to HUGO from whichever format 
	if in_genecode not in ['HUGO', 'HUGO_ENSG']:
		print("UPDATE: Switching feature mapping to HUGO id")
		genedict = pd.read_csv(genedict_file, sep="\t")
		genedict = genedict.loc[genedict['HUGO'].isin(features_train)]
		genedict = genedict.set_index(in_genecode) # GENCODE, GAF, HUGO
		features_test = list(genedict.ix[features_test]["HUGO"])
	if in_genecode == "HUGO_ENSG":
		features_test = [m.split("_ENSG", 1)[0] for m in features_test]
	
	##First subselect features, ordered in order of training set
	print("UPDATE: We will be updating the gene set order, input features {0}, \n \t\t\t with training features {1}".format(features_test[0:10], features_train[0:10]))
        x_test, null_in_test = padzero_features_x(x_test, features_train, features_test)
        print("UPDATE: \n\t Genes not found in input: {0}".format(null_in_test))
	### Based on sea otter, so not using: x_test = subset_features_x(x_test, features_test, features_train)
       	print("UPDATE: Testing samples, samples x genes setup {0}".format(x_test.shape)) 
	if(len(samples_test) != x_test.shape[0]):
		samples_test  = ["unknown"] * x_test.shape[0]
	##Then normalize features according to training set
	myargs = argparse.Namespace(**myargs)
	if(myargs.normalize[0] == 1):
		print("UPDATE: Normalization is per sample, so not considering training dat")
		print("UPDATE: Normalization: {0}".format(myargs.normtype[0]))
		
		x_test = apply_norm_func(norm_func=myargs.normtype[0], bysamples=myargs.normalize[0], xdat=(x_test))
	else:
		print("Training based normalization is not valid")
		with open(in_trainidx) as f:
			train_idx = f.read().splitlines()
		#Do more here, idk what
		## subset train input by indices, and pass it along with test dat (x) to normalization function
	
	#Lastly, check if the flag for random noising/blinding is on:
	if(myargs_test.makeblind[0] > 0):
		print("UPDATE: Randomly setting {0} fraction of features to 0".format(myargs_test.makeblind[0]))
		x_test = make_blind(x_test, myargs_test.makeblind[0])
		
	#Step 2 : Initialize mlp obj
	mlpobj = basic_mlp_features(learning_rate=myargs.learn[0], L1_reg=myargs.l1reg[0], L2_reg=myargs.l2reg[0], n_epochs=myargs.epochs[0], n_hidden=myargs.hidden[0], dataset_file=myargs.input[0], batch_size=myargs.batches[0], outdir=outdir,misclassflag=myargs.classbalance[0],writeout=False,shuffle_train=myargs.shuffle, normtype=myargs.normtype[0],normalize=myargs.normalize[0], keepsmall=myargs.smallset, hiddenlayers=myargs.hiddenlayers[0], drop_input=myargs.dropinput[0], drop_hidden=myargs.drophidden[0], test_dat = [x_test, samples_test])
	mlpobj.load_model(best_params_npz=in_model, label_dict_file=in_labdict, n_in=x_test.shape[1],n_class=sum(1 for line in open(in_labdict)))
	
	#Step 3 : test stuffffff
	#print([x_test,y_test])
	print(all(isinstance(n, (int,long,float)) for n in x_test))
	mlpobj.setup_test_data(is_train=False)
	mlpobj.features = features_train
	mlpobj.test_data()
	#mlpobj.get_performance_stats(outdir=outdir, outsuffix="_testset", during_train=False)	
	mlpobj.write_test_output(samples=samples_test)
	mlpobj.get_classbased_features(writeout=True)
			

if __name__ == '__main__':
	myargs = getargs()
	v1_models_names = [item for item in dir(models_nold) if not item.startswith("__")]
	for model_name in v1_models_names:
		myargs = getargs()
		model_dir = getattr(models_nold, model_name)
		get_preds(myargs, model_name, model_dir)


