#Embedded file name: split_input_into_testtrainvalid.py
import datetime, gzip
import csv, math
import numpy as np
import argparse
import re
import collections
import pandas
from argtypes import proper_file, makeintifpossible, int_or_float, check_norm_values
from normalization_functions import norm_minmax, norm_scale, norm_unitscale, norm_scaler_reapply, apply_norm_func
#This was from the older sklearn veresion. Update to version 0.18
#from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit 

import os, json, sys
from six.moves import cPickle
import pickle, csv

def readinputs(datfile, trimlabs, header_status, drugbinderset, sep = '\t', deeplearn=False, feat_startpos=1, sname_startpos=1):
    """This function takes in a matrix of format genes(columns) x samples(rows) and generates an array of labels (Y) and an array of corresponding input vectors (X). The first column is the sample names/labels """
    #If deeplearn, the first column is the samples, the next 4 are the ordered labels
    if deeplearn:
        feat_startpos = 5 ; sname_startpos = 0 # DELETE THIS AFTER DEEPLEARN 
        label_startpos=1 	
    else:
	label_startpos=0
        
    with open(datfile, 'r') as infile:
        reader = csv.reader(infile, delimiter=sep)
        X = []
        Y = []
        labels = []
        header = header_status
	featnames = []
        if drugbinderset:
            ligands = []
            for row in reader:
                if header:
                    header = False
		    featnames = [elem for elem in row[0:]]
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
		    featnames = [elem for elem in row[0:]]
                    header = False
                    continue
                row[feat_startpos:] = [ (float('nan') if elem == 'NA' else float(elem)) for elem in row[feat_startpos:] ]
                X.append(row[feat_startpos:])
                labels.append(row[label_startpos:feat_startpos])
            X = np.asarray(X)
            if (trimlabs) & (deeplearn is False):
		labels = [ re.sub('[0-9]*', '', x[0]) for x in labels ]
	    elif deeplearn is True:
		labels = [[re.sub('[0-9]*', '', i) for i in x] for x in labels ]
            returnset = [X,labels,featnames]
        return returnset

def getnumlabels(labellist):
    """This function takes in an ordered list of format [label1, label1, label2, label3, label3, ...] and generates a numeric label list """
    templist = dict([ (x, labellist.count(x)) for x in collections.OrderedDict.fromkeys(labellist) ])
    clist = [] #templist.values()
    retlist = []
    clabels = []
    retlist = labellist
    index_of_labelcount = 0; 
    #label_keys = collections.OrderedDict.fromkeys(labellist).keys(); 
    #Want an ordered labelling, with max class label = 0, next highest = 1, ....smallest has highest label num
    label_keys = sorted(templist, key=templist.get, reverse=True)
    for i in label_keys:
        if math.isnan(makeintifpossible(i)):
	    labname = index_of_labelcount
        else:
            labname = makeintifpossible(i)
        retlist = [ (labname if ylab == i else ylab) for ylab in retlist ]
        clabels += [labname]
	clist += [templist[i]]
        index_of_labelcount = index_of_labelcount + 1
    retlist = pandas.Series(data=retlist)
    return [templist,
     retlist,
     dict(zip(clabels, clist)),
     dict(zip(label_keys, clabels))]

def splitsets(labelnum, X, testsize, countlist, makevalid, whatsmall, verbose = True, validsize = None, keepsmall=False, deeplearn=False):
    if validsize is None:
        validsize = testsize/2
    print '..Splitting input into training and test, testsize {0} and validation {1}'.format(testsize, validsize)
    trainmat, testmat, trainidx, testidx = splittwogroups(labelnum, X, second_set=testsize, countdict=countlist, whatsmall=whatsmall, verbose=True, keepsmall=keepsmall, deeplearn=deeplearn)
    if makevalid == 1:
        print '...Now splitting training into train and validation'
        trainmat, validmat, trainidx, valididx = splittwogroups(pandas.Series(trainmat['y'][0]), trainmat['X'][0], second_set=validsize, countdict=countlist, whatsmall=whatsmall, verbose=True, keepsmall=keepsmall, deeplearn=deeplearn)
    else:
        testmat, validmat, temp1, valididx = splittwogroups(pandas.Series(testmat['y'][0]), testmat['X'][0], second_set=0, countdict=countlist, whatsmall=whatsmall, verbose=False, keepsmall=keepsmall, deeplearn=deeplearn)
    return [trainmat,validmat,testmat,trainidx,valididx,testidx]

def splittwogroups(inputlabels, inputvectors, second_set, countdict, whatsmall, verbose = True, keepsmall=False,deeplearn=False):
    """This function takes a list of labels and splits them into training and test (or training and validation, whatever). """
    #Inputs are panda Series objects
    if verbose:
        print '  ...Filter threshold is more than {0} samples per class'.format(2 * int(1 / whatsmall))
    inputlabels = pandas.Series(inputlabels)
    classes_high = [ i for i in countdict if countdict[i] > 2 * int(1 / whatsmall) ]
    classes_low = [ i for i in countdict if countdict[i] < 2 * int(1 / whatsmall) + 1 ]
    if 0 & deeplearn:
        lowclass_set = [x for x in inputlabels if x in classes_low]
        highclass_set = [x for x in inputlabels if x in classes_high]
    else:
        lowclass_set = inputlabels[inputlabels.isin(classes_low)]
        highclass_set = inputlabels[inputlabels.isin(classes_high)]
    train_y, test_y = train_test_split(highclass_set, test_size=second_set, random_state=0)
    if(second_set > 0):
	X_high = inputvectors[list(highclass_set.index),:]; y_high = inputlabels[list(highclass_set.index)]
	while((len(set(test_y)) < len(set(highclass_set))) or (len(set(train_y)) < len(set(highclass_set)))):
    		#Need to ensure that after the re-spliting the percentage is ALWAYS maintained
		#StratifiedSplit and other sklearn CV splitters can randomly maintain imbalance for very small classes
	    #	train_y, test_y = train_test_split(highclass_set, test_size = second_set, random_state=123)
    		sss = StratifiedShuffleSplit(n_splits=1, test_size=second_set, random_state=123)
		train_idx, test_idx = sss.split(X_high, y_high)
		train_y = inputlabels[y_high.index[train_idx[0]]]; 
		test_y = inputlabels[y_high.index[test_idx[0]]]
	
    train_x = inputvectors[list(train_y.index), :]
    test_x = inputvectors[list(test_y.index), :]
    small_trainy, small_testy = lowclass_set, lowclass_set
    if keepsmall:
    	train_y = pandas.concat([train_y, small_trainy])
	train_x = np.concatenate((train_x, inputvectors[list(small_trainy.index), :]))
    	if second_set > 0:
		test_y = pandas.concat([test_y, small_testy])
    		test_x = np.concatenate((test_x, inputvectors[list(small_testy.index), :]))
	
    if verbose:
        print '  ...{0} train large, {1} train small \n  ...{2} test large, {3} test small'.format(len(train_y), len(small_trainy), len(test_y), len(small_testy))

    trainmat = {'X': [np.asarray(train_x)],
     'y': [np.asarray(train_y)]}
    testmat = {'X': [np.asarray(test_x)],
     'y': [np.asarray(test_y)]}
    return [trainmat,testmat,train_y.index,test_y.index]


def takeawaytest(inputlabels, inputvectors, ligandlist, classligands = None):
    if classligands is None:
        classligs = list()
        for i in set(inputlabels):
            print 'Class %s' % i
            print set(ligandlist[np.where(inputlabels == i)])
            if len(set(ligandlist[np.where(inputlabels == i)])) < 2:
                print 'WARNING: This set only has 1 ligand, it will be split into train and valid, but not tested.'
            else:
                selectfortest = np.random.choice(ligandlist[np.where(inputlabels == i)])
                classligs.append(selectfortest)

    else:
        classligs = classligands
    test_idx = np.where(np.in1d(ligandlist, classligs))[0]
    train_idx = np.where(np.invert(np.in1d(ligandlist, classligs)))[0]
    test_y = inputlabels[test_idx]
    test_x = inputvectors[test_idx]
    train_y = inputlabels[train_idx]
    train_x = inputvectors[train_idx]
    trainmat = {'X': [np.asarray(train_x)],
     'y': [np.asarray(train_y)]}
    testmat = {'X': [np.asarray(test_x)],
     'y': [np.asarray(test_y)]}
    return [trainmat,
     testmat,
     train_y.index,
     test_y.index,
     classligs]


def labelweighing(labcountdict, labels_set):
    tlist = list(dict(labcountdict).values())
    maxcount = max(tlist)
    tlist2 = [ float(maxcount) / x for x in tlist ]
    tlist3 = [ math.log(x, 2) for x in tlist2 ]
    tlist = dict(zip(labels_set.values(), tlist3))
    return tlist

class mydatstuff(object):

    def __init__(self, infile, outdir, trimflag, keepheader, drugbinderflag, fulltrainflag, keepsmall, normtype=None, testsize = 0.1, normby = 1, outprefix = None , makevalid = 1, classligands = None,verbose=False, x=None, labs=None,ligs=None, deeplearn=False):
        self.input = infile
        self.outdir = outdir
	self.deeplearn = deeplearn
        if(x is not None):
		self.X, self.labels, self.ligands = x, labs, ligs
	else:
	    self.X, self.labels, self.ligands = readinputs(infile, trimflag, keepheader, drugbinderset=drugbinderflag, deeplearn=deeplearn)
	    
	    t = pandas.DataFrame(self.X); t = t.dropna(how='all', axis=1); self.X = np.asarray(t)
        
	if deeplearn:
		self.labels = np.asarray(self.labels)
		lc=[]; ln=[]; cl=[]; ls=[]; lw=[]
        	for i in range(np.shape(self.labels)[1]):
			catlist = getnumlabels(list(self.labels[:,i])) 
			lc.append(catlist[0]); ln.append(catlist[1]); cl.append(catlist[2]); ls.append(catlist[3])
			print("Total classes level {0} are {1}\n".format(str(i), str(max(catlist[1])+1)))
			lw.append(labelweighing(catlist[0], catlist[3]))
		
		merged_lab = [a + "-" + b + "-" + c + "-" + d for a,b,c,d in zip(self.labels[:,0], self.labels[:,1],self.labels[:,2],self.labels[:,3])]
		self.labelcount_unrolled = lc; self.labelnum_unrolled =ln; self.countlist_unrolled=cl; self.labels_set_unrolled = ls
		flip_labs = [{str(v).zfill(2): k for k, v in my_map.items()} for my_map in self.labels_set_unrolled]
			
		self.labelnum = [str(a).zfill(2) + str(b).zfill(2) + str(c).zfill(2) + str(d).zfill(2) for a,b,c,d in zip(self.labelnum_unrolled[0], self.labelnum_unrolled[1],self.labelnum_unrolled[2],self.labelnum_unrolled[3])]
		
		self.labels_set = [flip_labs[0][str(a).zfill(2)] + "-" + flip_labs[1][str(b).zfill(2)] + "-" + flip_labs[2][str(c).zfill(2)] + "-" + flip_labs[3][str(d).zfill(2)] for a,b,c,d in zip(self.labelnum_unrolled[0], self.labelnum_unrolled[1],self.labelnum_unrolled[2],self.labelnum_unrolled[3])]
		self.labels_set = dict(set(zip(self.labels_set,self.labelnum)))
		self.labelcount = dict(collections.Counter(merged_lab))
		self.countlist = dict(collections.Counter(self.labelnum))	
		self.merged_labels = merged_lab
		print("A sample list of y values, mapped to int levels {0}".format(self.labelnum[1:10]))
		
		self.labels_weighted = labelweighing(self.labelcount, self.labels_set)
		#self.labelcount_unrolled = lc; self.labelnum_unrolled =ln; self.countlist_unrolled=cl; self.labels_set_unrolled = ls
	else:
		self.labelcount, self.labelnum, self.countlist, self.labels_set = getnumlabels(self.labels)
		print 'Total classes are {0}'.format(str(max(self.labelnum) + 1))
        	self.labels_weighted = labelweighing(self.labelcount, self.labels_set)
       
	 
	self.outpref = outprefix
        self.normtype = normtype
        self.normby = int(normby)
        self.testsize = testsize
        self.makevalid = makevalid
	self.keep_small = keepsmall
	self.whatsmall=0.2
	self.verbose=verbose
	if self.normtype == 'none':
            normedby = '_normed' + self.normtype
        elif self.normby == 1:
            normedby = '_normedSamples_' + self.normtype
        else:
            normedby = '_normedFeatures_' + self.normtype
        self.outpref = self.outpref + normedby + '_makevalid' + str(self.makevalid) + '_keepsmall' + str(self.keep_small)
	self.drugbinderflag = drugbinderflag; self.fulltrainflag=fulltrainflag; 
	
    def splitvalidtest(self,returnsets=False):
        if self.drugbinderflag:
            if self.fulltrainflag:
                self.trainmat, self.validmat, self.testmat, self.trainidx, self.valididx, self.testidx = splitsets(labelnum=self.labelnum, X=self.X, testsize=0, countlist=self.countlist, makevalid=self.makevalid, validsize=self.testsize, keepsmall=self.keep_small, whatsmall=self.whatsmall, verbose=self.verbose)
                self.testlabs = [0, 0]
            else:
                self.trainmat, self.testmat, self.trainidx, self.testidx, self.testlabs = takeawaytest(self.labelnum, self.X, self.ligands, classligands)
                print '...Test set formed with ligands: {0}'.format(self.testlabs)
                if self.makevalid == 1:
                    print '...Generating validation set as well'
                    self.trainmat, self.validmat, self.trainidx, self.valididx = splittwogroups(pandas.Series(self.trainmat['y'][0]), self.trainmat['X'][0], second_set=self.testsize, countdict=self.countlist, whatsmall=self.whatsmall, keepsmall=self.keep_small)
                else:
                    self.trainmat, self.validmat, self.temp, self.valididx = splittwogroups(pandas.Series(self.trainmat['y'][0]), self.trainmat['X'][0], second_set=0, countdict=self.countlist, whatsmall=self.whatsmall, verbose=False, keepsmall=self.keep_small)
	else:
		if self.fulltrainflag:
			self.trainmat, self.validmat, self.testmat, self.trainidx, self.valididx, self.testidx = splitsets(labelnum=self.labelnum, X=self.X, testsize=0, countlist=self.countlist, makevalid=self.makevalid, validsize=self.testsize, keepsmall=self.keep_small, whatsmall=self.whatsmall, verbose=self.verbose, deeplearn=self.deeplearn)
            	else:
			self.trainmat, self.validmat, self.testmat, self.trainidx, self.valididx, self.testidx = splitsets(labelnum=self.labelnum, X=self.X, testsize=self.testsize, whatsmall=self.whatsmall, countlist=self.countlist, makevalid=self.makevalid, keepsmall=self.keep_small,verbose=self.verbose, deeplearn=self.deeplearn)
        
        print 'Training set has {0} classes \n .....{1} in test \n .....{2} in validation'.format(len(list(set(self.trainmat['y'][0]))), len(list(set(self.testmat['y'][0]))), len(list(set(self.validmat['y'][0]))))
        if self.fulltrainflag:
            self.testmat = self.validmat
            self.testidx = self.valididx
		
    def normalize(self):
        print("NORMALIZATION: {0}".format(self.normtype))
	if self.normtype == 'none':
            print 'Not normalizing data'
        elif self.normby == 1:
            print '...normalizing each sample. This is done agnostic of the type of dataset (train/test/valid).'
            self.trainmat['X'][0] = np.asarray(apply_norm_func(norm_func=self.normtype, xdat=self.trainmat['X'][0], bysamples=1))
            if (self.testmat['X'][0]).shape[0] > 0:
	        self.testmat['X'][0] = np.asarray(apply_norm_func(norm_func=self.normtype, xdat=self.testmat['X'][0], bysamples=1))
                self.validmat['X'][0] = np.asarray(apply_norm_func(norm_func=self.normtype, xdat=self.validmat['X'][0], bysamples=1))
        else:
            print '...normalizing each feature. This is done within the train/valid/test sets separately.'
            self.trainmat['X'][0], self.testmat['X'][0], self.validmat['X'][0] = apply_norm_func(norm_func=self.normtype, xdat=[self.trainmat['X'][0], self.testmat['X'][0], self.validmat['X'][0]], bysamples=0)

    def return_dataset(self):
        #Note that self.ligands is the set of features (if header provided for input txt), or an empty set, for general purposes.
	#It is the set of ligands when reading the drugbinder set
	if self.deeplearn:
		return([[self.trainmat['X'][0], self.trainmat['y'][0]], [self.validmat['X'][0], self.validmat['y'][0]], [self.testmat['X'][0], self.testmat['y'][0]]],self.labels_set, dict(self.labels_weighted), self.ligands)
	else:
		return([[self.trainmat['X'][0], self.trainmat['y'][0]], [self.validmat['X'][0], self.validmat['y'][0]], [self.testmat['X'][0], self.testmat['y'][0]]],self.labels_set, dict(self.labels_weighted), self.ligands)

    def writeout(self,save_dat=False, write_idx=True):
        if(self.input is None):
                filename = "predefined_input"
	else:
		filename = os.path.splitext(os.path.basename(self.input.rsplit('.', 1)[0]))[0]
        self.outpref = filename + '_' + self.outpref
        self.outdir = os.path.join(os.path.abspath(self.outdir) + '/' + self.outpref + '_' + datetime.datetime.now().strftime('%Y-%m-%d'))
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        print (self.trainmat['X'][0].shape, self.validmat['X'][0].shape, self.testmat['X'][0].shape)
	print("Writing processed data features to {0}".format(self.outdir))
        if save_dat:
		outall = gzip.open(os.path.join(self.outdir, ''.join([self.outpref, '.pkl.gz'])), 'wb')
		cPickle.dump([[self.trainmat['X'][0], self.trainmat['y'][0]], [self.validmat['X'][0], self.validmat['y'][0]], [self.testmat['X'][0], self.testmat['y'][0]]], outall, protocol=cPickle.HIGHEST_PROTOCOL)
		outall.close()
	
	if write_idx:
		with open(os.path.join(self.outdir, 'trainingidxs.txt'), 'w') as f:
			[ f.write('{0}\n'.format(x)) for x in self.trainidx ]
		with open(os.path.join(self.outdir, 'testidxs.txt'), 'w') as f:
			[ f.write('{0}\n'.format(x)) for x in self.testidx ]
		with open(os.path.join(self.outdir, 'valididxs.txt'), 'w') as f:
			[ f.write('{0}\n'.format(x)) for x in self.valididx ]
        
	with open(os.path.join(self.outdir, ''.join(['dict_labels_', self.outpref, '.txt'])), 'w') as f:
            [ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labels_set.items() ]
        with open(os.path.join(self.outdir, ''.join(['dict_labcount_', self.outpref, '.txt'])), 'w') as f:
            [ f.write('{0}\t{1}\n'.format(key, value)) for key, value in dict(self.labelcount).items() ]
        with open(os.path.join(self.outdir, ''.join(['dict_labweights_', self.outpref, '.txt'])), 'w') as f:
            [ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labels_weighted.items() ]

	if self.deeplearn:
		with open(os.path.join(self.outdir, ''.join(['dict_labels_', self.outpref, 'unrolled.txt'])), 'w') as f:
			[ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labels_set_unrolled.items() ]
		with open(os.path.join(self.outdir, ''.join(['dict_labcount_', self.outpref, 'unrolled.txt'])), 'w') as f:
			[ f.write('{0}\t{1}\n'.format(key, value)) for key, value in dict(self.labelcount_unrolled).items() ]
		with open(os.path.join(self.outdir, ''.join(['dict_labweights_', self.outpref, 'unrolled.txt'])), 'w') as f:
			[ f.write('{0}\t{1}\n'.format(key, value)) for key, value in self.labels_weighted_unrolled.items() ]

if 0:
    myargs = getargs()
    logfile = open(myargs.outdir[0] + '/logfile.txt', 'a')
    datfile = myargs.input[0]
    outpref = myargs.pref[0]
    makevalid = myargs.make_valid
    testsize = int_or_float(myargs.testsize[0])
    outdir = myargs.outdir[0]
    print 'Reading input from {0} \n Outprefix is {1} \n Creating validation set? {2} \n Size of test set (and validation set, if any) {3} \n'.format(datfile, outpref, makevalid, testsize)
    if myargs.normtype is None:
        normtopass = 'none'
        norm_sample = 0
    else:
        normtopass = myargs.normtype[0]
        norm_sample = myargs.normalize[0]
    
    if myargs.ligands is None:
        dataseparator = mydatstuff(infile=myargs.input[0], outdir=myargs.outdir[0], trimflag=myargs.trimlabels, keepheader=myargs.header_row, testsize=int_or_float(myargs.testsize[0]), normby=norm_sample, outprefix=myargs.pref[0], normtype=normtopass, makevalid=myargs.make_valid, drugbinderflag=myargs.drugbinder, fulltrainflag=myargs.fulltrainset, keepsmall=myargs.keep_small)
    else:
        print '...Using user defined classes as tests, {0}'.format(myargs.ligands)
        dataseparator = mydatstuff(infile=myargs.input[0], outdir=myargs.outdir[0], trimflag=myargs.trimlabels, keepheader=myargs.header_row, testsize=int_or_float(myargs.testsize[0]), normby=norm_sample, outprefix=myargs.pref[0], normtype=normtopass, makevalid=myargs.make_valid, drugbinderflag=myargs.drugbinder, classligands=myargs.ligands, fulltrainflag=myargs.fulltrainset, keepsmall=myargs.keep_small)
    dataseparator.normalize()
    dataseparator.writeout()
