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

import datetime
from collections import Counter
import random
import itertools
import copy

#Definitions
import logging, sys

#Change default theano GC for memory optimization
theano.config.allow_gc = True

def subset_features_x(x, fullfeatures_named, subsetfeatures_named, fillzero=False):
        #Used to select a set of input features in a specific order
        #Format of input is samples x features
        #Order of features is fullfeatures_named; subsetting is subsetfeatures_named
        subset_indices = []; missing_values=[]
        for select_this in subsetfeatures_named:
                subset_indices.append(fullfeatures_named.index(select_this))
        #returned object will have the features ordered in the manner of subsetfeatures_named
        return (np.apply_along_axis(lambda x: x[subset_indices], 1, x))

def read_input(input_file, sep="\t"): 
	"""This function reads in the set of samples to predict. By default, the input is arranged as gene_name \t Sample 1 RPKM \t Sample 2 RPKM ....
	An alternative column separator can be passed to the function using the 'sep' attribute"""
	try:
		os.path.isfile(input_file):
	except:
		print "Error: {1}".format(sys.exc_info()[0])
	else:
		print("Reading file {0}".format(input_file))
		input_dat = np.loadtxt(input_file, delimiter=sep)
		genedict = pd.read_csv(genedict_file, sep="\t")
	
	"""Feature subsetting"""
	try:
		in_genecode in ['HUGO', 'HUGO_ENSG', 'GENCODE', 'GAF']:
	except TypeError, NameError,ValueError):
		print "Error: {1}".format(sys.exc_info()[0])
		print("First column's name should be one of {0}".format(genedict.columns.values.tolist()))
	else:
		if in_genecode == 'HUGO_ENSG':
			features_test = [m.split("_ENSG", 1)[0] for m in features_test]
			in_genecode = "HUGO"
		
		genedict = pd.read_csv(genedict_file, sep="\t")
		genedict = genedict.loc[genedict['HUGO'].isin(features_train)]
		genedict = genedict.set_index(in_genecode) # GENCODE, GAF, HUGO
		features_test = list(genedict.ix[features_test]["HUGO"])
	
	return X, features_test

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

