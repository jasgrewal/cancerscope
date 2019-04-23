#Contains commonly used classes and functions
#For I/O purposes in ML model runs

#Shared libs
import os, sys
import timeit
import pkgutil
import pandas as pd
import copy
import numpy as np
import operator

def map_train_test_features_x(x_in, training_features, test_features, fillzero=True, defaultvalue=0.0):
	x = copy.deepcopy(x_in)## Create another persistent copy of x as we are modifying it
	# Check that the matrix is ordered as samples x genes   
	assert(x.shape[1] == len(test_features))
	
	# For each training feature, get the corresponding index in the test features 
	reorder_test = {test_features.index(i):training_features.index(i) for i in test_features if i in training_features}
	### Get a list of indices indicating which index of the test features list corresponds to the corresponding ith' position in the training features list
	test_keys_orderedix = [i[0] for i in sorted(reorder_test.items(), key=operator.itemgetter(1))]
	
	num_testfeats_mapping_to_trainingfeats = len(test_keys_orderedix)
	assert(test_features[test_keys_orderedix[0]] == training_features[0])
	
	testX_values = np.apply_along_axis(lambda x: x[test_keys_orderedix], 1,x)
	if num_testfeats_mapping_to_trainingfeats != len(training_features):
		sys.stdout.write("\n...Only {0} of input features mapped to expected number of features. Setting the rest to {1}".format(num_testfeats_mapping_to_trainingfeats, defaultvalue))
	if fillzero:
		for i in range(0, len(training_features), 1):
			if training_features[i] not in test_features:
				testX_values = np.insert(testX_values, i, defaultvalue, axis=1)
	return(testX_values)

def map_gene_names(features_list, genecode_in, genecode_out):
	### Options for genecode_in and genecode_out are ENSG, HUGO, GENCODE, GAF
	genedict = pd.read_csv("/".join([os.path.dirname(sys.modules["cancerscope"].__file__), 'resources/scope_features_genenames.txt']), delimiter="\t")
	
	try:
		genecode_in in genedict.columns.tolist()
		genecode_out in genedict.columns.tolist()
	except (TypeError, NameError,ValueError):
		sys.stdout.write("Error: {1}\n".format(sys.exc_info()[0]))
		sys.stdout.write("Genecode should be one of {0}\n".format(genedict.columns.tolist()))
	else:
		genedict = genedict.set_index(genecode_in) 
		features_mapped = list(genedict.ix[features_list][genecode_out]) 
	
	try:
		len(features_mapped) == len(features_list)
	except:
		sys.stdout.write("\nThe mapping could not be completed accurately")
	else:
		return features_mapped
	
def read_input(input_file, sep="\t"): 
	"""This function reads in the set of samples to predict. By default, the input is arranged as gene_name \t Sample 1 RPKM \t Sample 2 RPKM ....
	An alternative column separator can be passed to the function using the 'sep' attribute"""
	try:
		os.path.isfile(input_file)
	except:
		sys.stdout.write("\nError: {1}\n".format(sys.exc_info()[0]))
	else:
		sys.stdout.write("\nReading file {0}\n".format(input_file))
		genedict = pd.read_csv("/".join([os.path.dirname(sys.modules["cancerscope"].__file__), 'resources/scope_features_genenames.txt']), delimiter=sep)
		input_dat = pd.read_csv(input_file, delimiter=sep)
		in_genecode = input_dat.columns.tolist()[0] # What is the gene naming convention for the input  - this is the 1st column
	"""Feature subsetting"""
	try:
		in_genecode in ["GENCODE", genedict.columns.tolist()]
	except (TypeError, NameError,ValueError):
		sys.stdout.write("Error: {1}\n".format(sys.exc_info()[0]))
		sys.stdout.write("First column's name should be one of {0}\n".format(genedict.columns.tolist()))
	else:
		"""Separate out the input data into features (col 1), X (numeric values), and samples"""
		features_test = input_dat[in_genecode].tolist()
		X_data = input_dat.loc[:, input_dat.columns != in_genecode]
		samples = X_data.columns.tolist()
		X = X_data.values #to_records(index=False)
		if in_genecode == "GENCODE":
			sys.stdout.write("\nYour genenames are of format ENSG.version. Truncating to ENSG ID only\n")
			features_test = [m.split(".")[0] for m in features_test]
		#sys.stdout.write("\nX shapei s {0} and features len is {1}".format(X.shape, len(features_test)))
	
	if(len(samples) == 1):
		## if there is only one sample, numpy array doesnt act nice, has to be reshaped
		X = X.reshape(1, len(X))
	else:
		## Otherwise, reshape as samples x genes, instead of genes x samples
		X = X.transpose()
	return X, samples, features_test, in_genecode

