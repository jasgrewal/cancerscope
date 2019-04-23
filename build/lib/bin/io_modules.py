#Contains commonly used classes and functions
#For I/O purposes in ML model runs

#Shared libs
import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit
import glob 
import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from argtypes import Dictionary
from sampling_algorithms import *

import datetime
from collections import Counter
import random
import itertools
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from split_input_into_testtrainvalid import *

#Definitions
import logging, sys

def subset_features_x(x, fullfeatures_named, subsetfeatures_named, fillzero=False):
        #Used to select a set of input features in a specific order
        #Format of input is samples x features
        #Order of features is fullfeatures_named; subsetting is subsetfeatures_named
        subset_indices = []; missing_values=[]
        for select_this in subsetfeatures_named:
                subset_indices.append(fullfeatures_named.index(select_this))
        #returned object will have the features ordered in the manner of subsetfeatures_named
        return (np.apply_along_axis(lambda x: x[subset_indices], 1, x))

class Unbuffered3:
	def __init__(self, stream):
		root = logging.getLogger()
		root.setLevel(logging.DEBUG)
		ch = logging.StreamHandler(sys.stdout)
		ch.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		root.addHandler(ch)


class Unbuffered:
        def __init__(self, stream, logfile=open("logfile.txt",'w')):
                self.stream = stream
		self.logfile = logfile
		print("writing to {0}".format(logfile))	
	def flush(self):
		pass        
        def write(self, data):
                self.stream.write(data)
		self.stream.flush()
                self.logfile.write(data)

def load_data(dataset, make_shared=False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path):
            dataset = new_path

    print('... loading data from {0}'.format(dataset))
    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if make_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]
    return rval

def load_cpickled_data(dataset):
        '''Load the data. Presume it is cPickled [(trainx,trainy),(validx,validy),(testx,testy)]
        	- Currently imported by SVM models
	'''
        print("...loading data")
        with gzip.open(dataset, 'rb') as f:
                try:
                        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
                except:
                        train_set, valid_set, test_set = pickle.load(f)

        train_set_x, train_set_y = train_set
        valid_set_x, valid_set_y = valid_set
        train_set_x = np.concatenate((train_set_x, valid_set_x), axis=0)
        train_set_y = np.asarray(list(train_set_y) + list(valid_set_y))
        test_set_x, test_set_y = test_set
        rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
        return rval

def load_sig_dat(infile, prefix, outdir, testpog=False):
        t1 = pd.read_table(infile)
        if prefix == "exposures":
                cols = ["id","tumour_type","signature","exposure_per_mb"]
                featname = "signature"; featvalue = "exposure_per_mb"
        elif prefix == "catalogs":
                cols = ["id", "tumour_type", "mutation_type", "mutations_per_mb"]
                featname = "mutation_type"; featvalue="mutations_per_mb"
	else:
		return(0)
        if testpog:
                t1=t1[t1.sample_prefix.str.contains("biop")==True]
                t1['id'] = t1[['id', 't','tumour_library','n','normal_library']].apply(lambda x: '_'.join(x), axis=1)
                t1['tumour_type'] = "unknown"
                prefix = prefix + "_pog"
        tdat=t1[cols]
        tdat_feats = tdat.pivot_table(index=["id","tumour_type"], columns=[featname], values=featvalue)
        tdat_feats = tdat_feats.reset_index().rename_axis(None,axis=1)
        intxt = outdir + "/data_" + prefix + ".txt"
        insamples = outdir + "/samples_" + prefix + ".txt"
        tdat_feats.ix[:,0:2].to_csv(insamples, index=None, sep="\t", mode="w")
        tdat_feats.ix[:,1:].to_csv(intxt, index=None, sep='\t', mode="w")
        return([tdat_feats, intxt, insamples])





def readtestinput(datfile, trimlabs, header_status, sep='\t'):
        "This function takes in a matrix of format genes(columns) x samples(rows) and generates an array of labels (Y) and an array of corresponding input vectors (X). The first column is the sample names/labels - Currently imported by SVM models"
        with open(datfile, 'r') as infile:
                reader = csv.reader(infile, delimiter=sep)
                X = []
                Y = []
                labels = []
                samples = []
                header = header_status
                for row in reader:
                        if header:
                                header = False
                                continue
                        row[2:] = [float('nan') if elem=="NA" else float(elem) for elem in row[2:]]
                        X.append(row[2:])
                        Y.append("0")
                        labels.append(row[0])
                        samples.append(row[1])
                X = np.asarray(X)
                #Y = np.asarray(Y)
                #Y = Y.reshape(Y.shape[0],1)
                #Update labels (from type_snum to type, and then a list of numeric labels for output types)  
                if trimlabs:
                        labels = [re.sub('[0-9]*','',x) for x in labels]
	return [X, Y, labels, samples]

def load_txt_input(datfile, trimlabs, header_status, drugbinderset, sep = '\t'):
    """This function takes in a matrix of format genes(columns) x samples(rows) and generates an array of labels (Y) and an array of corresponding input vectors (X). The first column is the sample names/labels """
    with open(datfile, 'r') as infile:
        reader = csv.reader(infile, delimiter=sep)
        X = []
        Y = []
        labels = []
        header = header_status
        if drugbinderset:
            ligands = []
            for row in reader:
                if header:
                    header = False
                    continue
                row[2:] = [ (float('nan') if elem == 'NA' else float(elem)) for elem in row[2:] ]
                X.append(row[2:])
                ligands.append(row[0])
                Y.append(row[1])
                labels.append(row[1])

            ligands = np.asarray(ligands)
            X = np.asarray(X)
            Y = np.asarray(Y)
            Y = Y.reshape(Y.shape[0], 1)
            if trimlabs:
                labels = [ re.sub('[0-9]*', '', x) for x in labels ]
            returnset = [X,
             Y,
             labels,
             ligands]
        else:
            for row in reader:
                if header:
                    header = False
                    continue
                row[1:] = [ (float('nan') if elem == 'NA' else float(elem)) for elem in row[1:] ]
                X.append(row[1:])
                Y.append(row[0])
                labels.append(row[0])

            X = np.asarray(X)
            Y = np.asarray(Y)
            Y = Y.reshape(Y.shape[0], 1)
            if trimlabs:
                labels = [ re.sub('[0-9]*', '', x) for x in labels ]
            returnset = [X,Y, labels,0]
        return returnset

class mldataset(object):
        def __init__(self, dataset_file, outdir, normtype="none",misclassflag="none", normby=str(1), keepsmall=True, deeplearn=False):
                #Provide input address
                self.inputfile = dataset_file
                self.outdir = outdir
                self.keepsmall = keepsmall
                self.normtype = normtype
                self.normby = normby
		self.shuffle_train = True
		self.misclassflag = misclassflag	
		self.deeplearn = deeplearn
		
        def load_mldata(self, fullset=False, verbose=True, save_full_stats=False, subsetfeatures=None):
		#subsetfeatures must be a txt file with list fo features to subset input with
                if(self.inputfile.endswith(".tsv")|self.inputfile.endswith(".txt")):
                        self.load_txtdata(fullset=fullset, verbose=verbose, save_full_stats=save_full_stats)
                else:
                        self.load_pkldata()
		self.X = np.concatenate((self.datasets[0][0], self.datasets[1][0], self.datasets[2][0]))
		self.y = np.concatenate((self.datasets[0][1], self.datasets[1][1], self.datasets[2][1]))
		
		if subsetfeatures is not None:
			if verbose:
				print("SUBSETTING TRAINING DATA FEATURES from {0}".format(subsetfeatures))
			features_to_subset = [line.rstrip().strip('"') for c, line in enumerate(open(subsetfeatures[0], 'r'))]
			self.X = subset_features_x(self.X, self.features, features_to_subset, fillzero=False)
			self.features = features_to_subset
		
		if save_full_stats:
			#Split data into train, validation, and test
			self.getCVfolds(self, k=5, validation_size = 0.1, makeupdate=True)
						
        def load_txtdata(self,fullset=False,verbose=True, save_full_stats=False):
                #This part splits and normalizes the data
                dataseparator = mydatstuff(infile=self.inputfile, outdir=self.outdir, trimflag=True, keepheader=True, drugbinderflag=False,keepsmall=self.keepsmall, fulltrainflag=fullset, classligands=None, testsize=0, outprefix="prepared_dat", normtype=self.normtype, normby=str(self.normby), makevalid=True,verbose=verbose, deeplearn=self.deeplearn)
                dataseparator.splitvalidtest()
                dataseparator.normalize()
                self.datasets, self.labelnames, self.labweighted, self.features = dataseparator.return_dataset()
                if save_full_stats:
                        dataseparator.writeout() #Save idx's, not the actual dataset
			self.save_preprocessed_files(verbose=verbose)
		
        def load_pkldata(self):
                #Assume pickled datasets are already normalized
                #Pickled datasets are also in 3 sets, train-validation-test (3 idxs)

                self.datasets = load_data(self.inputfile, make_shared=False)
                indir = os.path.dirname(os.path.abspath(self.inputfile))
		self.labelweights = glob.glob(indir+"/dict_labweights_*")[0]
                self.labelset = glob.glob(indir+"/dict_labels_*")[0]
                self.featurelist = glob.glob(indir+"/featurelist.txt")
                self.labweighted = Dictionary(labelset=self.labelweights)
                self.labelnames = Dictionary(labelset=self.labelset)
                
		if self.featurelist:
                        f = open(self.featurelist[0])
                        with open(self.featurelist[0]) as f:
                                self.features = [line.rstrip() for line in f]

        def getCVfolds(self, k=2, validation_size = 0, X = None, y=None, makeupdate=False):
		k = int(k)
		validation_size= float(validation_size)
		if X is None:
			X = self.X
			y = self.y
                if k==1:
			k = 2
			#Default CV folds are 2 (for a single run, select first return)
                
		#First remove the samples and classes that are represented at freq < k (disrupts folds)
                class_counts = Counter(y)
                dropclass = [lk for lk in dict(class_counts).keys() if class_counts[lk] < k]
                smallidx = [np.where(y==f)[0] for f in dropclass]
                ##print("-------\nDROPCLASSES: {0} ".format(dropclass))
		#Stratified K fold splitter. Returns k sets of [train, test] idx's
                skf = StratifiedKFold(n_splits=int(k))
                train_klist = []; test_klist = []
                
		#Get the k-fold set of train and test indices, and also append a random subset of samples from the small samples set
                if len(smallidx) > 0:
			print("--Appending small classes separately {0}".format(dropclass))
			largeclass = [lk for lk in dict(class_counts).keys() if class_counts[lk] >= k]
			largeidx = [np.where(y==f)[0] for f in largeclass]
			largeidx_list = list(itertools.chain.from_iterable(largeidx)) 
			Xl = np.take(X, largeidx_list, axis=0) 
			yl = np.take(y, largeidx_list, axis=0)
			
			skf.get_n_splits(X,y)
			
			for train_idx, test_idx in skf.split(X,y):
				#Remove any of the small indices
				train_idx = [mk for mk in train_idx if not mk in list(itertools.chain.from_iterable(smallidx))]
				test_idx = [mk for mk in test_idx if not mk in list(itertools.chain.from_iterable(smallidx))]
				
				#Randomly select the small indices from the separate list to backfill for the train and test segs
				smalltest_k = [random.choice(m) for m in smallidx]
                        	#print("{0} and contents {1} \n -----> ".format(len(smallidx),smalltest_k, np.take(y, smalltest_k,axis=0)))
				smalltrain_k = [mx for m in smallidx for mx in m if mx not in smalltest_k]
				
				#print("SUBSET HERE: \n{0} \n AND \n{1}\nAND TEST SUBSET{2}\nAND TEST {3}\n".format(train_idx,smalltrain_k, test_idx,smalltest_k))
				train_klist.extend([np.concatenate((train_idx, smalltrain_k))])
                        	test_klist.extend([np.concatenate((test_idx,smalltest_k))])
		else:
			skf.get_n_splits(X,y)
			for train_idx, test_idx in skf.split(X,y):
				train_klist.extend([train_idx])
				test_klist.extend([test_idx])
		
		if makeupdate:
			print("...Updated K fold sets for training and test.")
			self.trainidx_K = train_klist	
			self.testidx_K = test_klist
		
		if validation_size > 0.0 and validation_size <= 0.5 :
			self.trainvalididx_K = [] ##self.trainidx_K #train_klist
			for i in range(0,len(train_klist)):
				bigtrain_idx = train_klist[i].astype(int)
				X = np.take(self.X , bigtrain_idx, axis=0)
				train_kth, valid_kth = self.getCVfolds(k=int(1/validation_size), X = np.take(self.X , bigtrain_idx, axis=0), y=np.take(self.y, bigtrain_idx, axis=0))
				self.trainvalididx_K.extend([[train_kth[0], valid_kth[0]]])
		
		return(train_klist, test_klist)
		
	def save_preprocessed_files(self, verbose=True):
                if(self.inputfile.endswith(".txt")):
                        #If the input file was a txt, only then do you save preprocessed files (dont repickled pickled input, pointless)
                        #Create an output data directory
                        try:
                                self.datadir
                        except:
                                self.datadir=self.outdir + "/preprocessed_data"
                        try:
                                os.stat(self.datadir)
                        except:
                                os.mkdir(self.datadir)
			if verbose:
                        	print("...Pickling preprocessed, split data to output directory {0}".format(self.datadir))
                        filename = os.path.splitext(os.path.basename(self.inputfile.rsplit('.', 1)[0]))[0]
                        self.dataoutpref = filename + "_normed" + self.normtype + str(self.normby) + "_keptsmall" + str(self.keepsmall)
                        outall = gzip.open(os.path.join(self.datadir, ''.join([self.dataoutpref, '.pkl.gz'])), 'wb')
			
			
                        cPickle.dump([[self.X_train, self.y_train], [self.X_val, self.y_val], [self.X_test, self.y_test]], outall, protocol=cPickle.HIGHEST_PROTOCOL)
                        if verbose:
				print("   - pkl dump of split data at {0}".format(os.path.join(self.datadir, ''.join([self.dataoutpref, '.pkl.gz']))))
                        with open(os.path.join(self.datadir, ''.join(['dict_labels_', self.dataoutpref, '.txt'])), 'w') as f:
                                [ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labelnames.items() ]
                        with open(os.path.join(self.datadir, ''.join(['dict_labweights_', self.dataoutpref, '.txt'])), 'w') as f:
                                [ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labweighted.items() ]
                        with open(os.path.join(self.datadir, 'featurelist.txt'), 'w') as f:
                                [ f.write('{0}\n'.format(x)) for x in self.features]

                        if verbose:
				print("...Updated input file")

                        self.inputfile = os.path.join(self.datadir, ''.join([self.dataoutpref, '.pkl.gz']))
                else:
			if verbose:
                        	print("...Not pickling data because input was a pkl'd split dataset already")
	
	def get_twosets(self, set1idx, set2idx):
                X1 = np.take(self.X, set1idx, axis=0)
                y1 = np.take(self.y, set1idx, axis=0)
                X2 = np.take(self.X, set2idx, axis=0)
                y2 = np.take(self.y, set2idx, axis=0)
                return([[X1,y1],[X2,y2]])

	def save_dataset_details(self,outdir, write_idx=True):
		print("====OUTDIR = {0}".format(self.outdir))
		with open(''.join([self.outdir,"/featurelist.txt"]), 'w') as f:
			[ f.write('{0}\n'.format(m)) for m in self.features ]
		
		if(self.inputfile is None):
			filename="predefined_input"
		else:
			filename=os.path.splitext(os.path.basename(self.inputfile.rsplit('.', 1)[0]))[0]
		self.outpref = filename	
		self.datoutdir = os.path.join(os.path.abspath(self.outdir) + '/dataset_train/' + filename + '_' + datetime.datetime.now().strftime('%Y-%m-%d'))
		print("Saving Dataset details to {0}".format(self.datoutdir))
		if not os.path.exists(self.datoutdir):
			os.makedirs(self.datoutdir)
		 
		if write_idx:
			print("Writing idx's to output, NOTE if this is full training then test and train idxs are combined for train. Validation becomes test set too")
			with open(os.path.join(self.datoutdir, 'trainingidxs.txt'), 'w') as f:
				[ f.write('{0}\n'.format(x)) for x in self.sub_kfold_trainidx ]
			with open(os.path.join(self.datoutdir, 'testidxs.txt'), 'w') as f:
				[ f.write('{0}\n'.format(x)) for x in self.sub_kfold_testidx ]
			with open(os.path.join(self.datoutdir, 'valididxs.txt'), 'w') as f:
				[ f.write('{0}\n'.format(x)) for x in self.sub_kfold_valididx ]
		
		try:
			int(self.labelnames.keys()[0])
		except ValueError:
			self.labelnames = {k:v for v,k in self.labelnames.iteritems()}
				
		with open(os.path.join(self.datoutdir, ''.join(['dict_labels_', self.outpref, '.txt'])), 'w') as f:
            		[ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labelnames.items() ]
        	with open(os.path.join(self.datoutdir, ''.join(['dict_labcount_', self.outpref, '.txt'])), 'w') as f:
         		[ f.write('{0}\t{1}\n'.format(self.labelnames[key], value)) for key, value in dict(Counter(self.y)).items() ]   
	#		[ f.write('{0}\t{1}\n'.format(key, value)) for key, value in dict(Counter(np.concatenate((self.y_train, self.y_val, self.y_test),axis=0))).items() ]
		
	        with open(os.path.join(self.datoutdir, ''.join(['dict_labweights_', self.outpref, '.txt'])), 'w') as f:
        	    [ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labweighted.items() ]
		
	def preprocess_data(self, kfold, features_subset=[], verbose=True, fullset=False):
		if '0' in self.labelnames.keys() or 0 in self.labelnames.keys():
                        self.labelnames = dict([int(a),b] for a,b in self.labelnames.iteritems())
                else:
			if self.deeplearn is True:
				self.labelnames = dict([b,a] for a,b in self.labelnames.iteritems())
			else:
				self.labelnames = dict([int(b),a] for a,b in self.labelnames.iteritems())
		
                self.sorted_keylabs=[]
                for i in sorted(self.labelnames.keys()):
                                self.sorted_keylabs.append(self.labelnames[i])
		
		main_trainidx = self.trainidx_K[kfold-1]
		
		self.sub_kfold_trainidx, self.sub_kfold_valididx = self.trainvalididx_K[kfold-1]
		self.sub_kfold_testidx = self.testidx_K[kfold-1]
		
		kfold_trainidx = np.take(main_trainidx, self.sub_kfold_trainidx); 
		kfold_valididx = np.take(main_trainidx, self.sub_kfold_valididx)
		kfold_testidx = self.sub_kfold_testidx		
		
		kfold_trainidx = kfold_trainidx.astype(int)
		kfold_valididx = kfold_valididx.astype(int)
		kfold_testidx = kfold_testidx.astype(int)
		
		if fullset is True:
			if verbose:
				print("...Appending Test idxs to training prior to sample inflation \n\t Validation set is test for full model training")
			kfold_trainidx = np.concatenate((kfold_trainidx, kfold_testidx),axis=0)
		#	kfold_trainidx = list(set(kfold_trainidx + kfold_testidx))
			kfold_testidx = kfold_valididx
			
                [[self.X_train, self.y_train], [self.X_val, self.y_val]] = self.get_twosets(set1idx = kfold_trainidx, set2idx = kfold_valididx)
		[[self.X_traintemp, self.y_traintemp], [self.X_test, self.y_test]] = self.get_twosets(set1idx = kfold_trainidx, set2idx = kfold_testidx)
		
		if verbose:
			print("Train shape {0} \nValid shape {1} \nTest shape {2}".format(self.y_train.shape,self.y_val.shape, self.y_test.shape))
			print("X shape {0} \nY shape {1}".format(self.X.shape,self.y.shape))
			print("Train + valid uniq {0} \n{1}, overall uniq {2}".format(len(np.unique(self.trainidx_K[kfold-1])),np.unique(np.concatenate((kfold_valididx,kfold_trainidx),axis=0)).shape,np.unique(np.concatenate((kfold_testidx,kfold_valididx,kfold_trainidx),axis=0)).shape))	
			print("{0} -- {1} -- {2}".format(len(set(self.y_train)), len(set(self.y_val)), len(set(self.y_test))))	
			#if np.array_equal(set(kfold_trainidx + kfold_valididx + kfold_testidx), self.y.shape[0]):
			#	del self.X_traintemp; del self.y_traintemp
			#else:
			#	print("ERROR: There is something wrong with your K fold split, the Train fold is not consistent between validation and test pairs")
	
			print(set(self.y_val))	
                	print(self.X_train.shape, self.X_val.shape, self.X_test.shape)
                
		if len(features_subset) > 0:
                        self.X_train = subset_features_x(self.X_train, self.features, features_subset)
                        self.X_val = subset_features_x(self.X_val, self.features, features_subset)
                        self.X_test = subset_features_x(self.X_test, self.features, features_subset)
                        if verbose:
				print("___SUBSET INPUT, new training data shape {0}".format(self.X_train.shape))
                        with open(os.path.join(self.outdir, 'featurelist.txt'), 'w') as f:
                                [ f.write('{0}\n'.format(x)) for x in features_subset]
		
                self.n_class = len(set(self.y_train)) #Training set represents all the classes
                self.n_in = len(self.X_train[0]) #First input sample's length is the number of features (assumes it remains the same for all samples!)

                if isinstance(self.y_test[0], np.int):
                        if verbose:
				print("...--Verified test is correctly labelled--...")
                else:
			if not self.deeplearn:	
                        	if verbose:
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
                        if verbose:
				print("WARNING: Test set does not have all the classes from the training set")
                        	print(set(self.y_train), set(self.y_test))
                self.save_preprocessed_files(verbose=verbose)

                if self.misclassflag == "none":
                        if verbose:
				print("No synthetic additions/modifications to the data")
                elif self.misclassflag != "weight":
                        self.X_train, self.y_train = adjust_trainset(self.X_train, self.y_train, self.misclassflag, verbose=verbose)
                        self.X_train, self.y_train  = shuffle(self.X_train, self.y_train, random_state=0) #Must reshuffle after synthetic additions

                if(self.shuffle_train):
                        if verbose:
				print("...Shuffling training set")
                        self.X_train, self.y_train  = shuffle(self.X_train, self.y_train, random_state=0)
                self.X_train = np.asarray(self.X_train)
                self.y_train = np.asarray(self.y_train)	
	
