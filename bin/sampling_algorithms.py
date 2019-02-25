from __future__ import division
from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
from itertools import repeat
#from imblearn import over_sampling
from collections import Counter

##Imblearn based functions
#def smote_this(all_x, all_y):
#	sm = over_sampling.SMOTE(kind='regular')
#	X_resampled, y_resampled = sm.fit_sample(all_x, all_y)
#	return X_resampled, y_resampled

def adjust_trainset(x, y, adjust_type, verbose=False):
	print("Generating synthetic samples using {0}...".format(adjust_type))
	duped_classx = []; duped_classy = []
	if adjust_type in ('smote', 'dup'):
		biggest_class_size = max(Counter(y).values())
		for ylab in set(y):
			class_indices = np.where(y == ylab)[0]
			subset_classx = np.take(x, class_indices, axis=0)
			classweight = float(biggest_class_size)/float(Counter(y)[ylab]) #Multiplicative term to expand smaller class by
			#Extra samples required, based on size of biggest class
			spillover_samples = float(biggest_class_size) - (len(subset_classx) * np.floor(classweight))
			#expand the set
			duped_set = expand_this_set(minority_data_x = subset_classx, type=adjust_type, class_weight=classweight, additional_samples=spillover_samples)
			duped = (len(subset_classx)*np.floor(classweight))
			if verbose:
				print("CLASS {0}".format(ylab))
				print("....extending class {0} size {1} samples by factor of {2}, {3} samples".format(ylab, len(subset_classx), classweight, len(duped_set) ))
			duped_classx.extend(duped_set)
			duped_classy.extend(repeat(ylab, len(duped_set)))
			
	elif adjust_type == "imblearn":
		duped_classx, duped_classy = probl_balance(x,y)
	elif adjust_type == "adasyn":
		duped_classx, duped_classy = adasyn_this(x,y)
	print("WARNING: Re-shuffle training set before learning, after set expansion")
	return(duped_classx, duped_classy)


##Shared functions
###probabilistic sampling
def probl_balance(x,y, dataset_size=None):
	classcounts = dict(Counter(y))
	largest_class = max(classcounts.values())
	classprobs = {k:1-(v/float(largest_class)) for k,v in classcounts.iteritems()}
	if dataset_size is None:
		dataset_size = x.shape[0]/len(set(y))
	xr = [] ; yr = []
	for i in range(0, x.shape[0]):
		rclass = random.randint(0,max(classcounts.keys())) #Randomly choose a class
		if random.random() < max(0.10,classprobs[rclass]):
			#If prob of selecting an element of this class is passed
			#Append an element from this  class to the list
			ridx = random.choice(np.where(y==rclass)[0])
			xr.append(x[ridx]); yr.append(y[ridx])
	
	return(np.asarray(xr), np.asarray(yr))
	#for i in range(0, x.shape[0]):
	#	if random.random() < max(0.10,classprobs[y[i]]):
	#		xr.append(x[i]); yr.append(y[i])

##ADASYN
#Implementation from pip install git+https://github.com/stavskal/ADASYN
#
def adasyn_this_test(all_x, all_y, tolerable_imbalance=0.9):
	from adasyn import ADASYN
	adsn = ADASYN(k=7, imb_threshold = tolerable_imbalance, ratio=0.75)
	new_X, new_y = adsn.fit_transform(all_x, all_y)
	return new_X, new_y

def adasyn_this(all_x, all_y, tolerable_imbalance=0.9):
	y_classes_count, sorted_y, cmax = getclasses(all_y)
	k=5
	return_np_array = all_x
	return_np_labs = all_y
	for class_small in sorted_y[1:]:
		csmall = y_classes_count[class_small]
		d_imb = csmall/cmax #Degree of imbalance
		if d_imb < tolerable_imbalance:
			#Set up the nearest neighbour model
			#NearestNeighbors will fit to the sample too, so we calculate k+1 neighbours and exclude current sample
			get_k_neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(all_x)
			G_factor = (cmax - csmall) * 1
			csmall_indices = [i for i,x in enumerate(all_y) if x==class_small]
			small_list_r, small_list_neighbours = build_small_list(csmall_to_test = csmall_indices,csmall_indices=csmall_indices, get_k_neighbors=get_k_neighbors, all_x=all_x)
			#OPTIONAL: Set ri to 0 if only 1 out of 5 neighbours is a non-self example
			#print("Setting small ri (=<0.2) to 0")
		##	print(Counter(small_list_r))
			#small_list_r = [0 if x < 0.21 else x for x in small_list_r]
			#Normalize ri so that each ri corresponds to a distribution	
			#Calculate the number of new examples needed for each minority example xi,
			#By converting the ratios of (# neighbours in majority class / total neighbours) for each xi
			#Into a density distribution, and then converting it to required number of new examples
			#So you don't need new examples when the nearest neighbours are all the same class,
			# for all examples in a minority class
			if(sum(small_list_r) == 0):
				print("....All samples for class {0} define the class quite well, not generating new samples".format(class_small))
			else:
				normalize_ri = map(lambda x: (x/sum(small_list_r)), small_list_r);
				##Select G_factor based not on the extra samples needed to match input size, but to balance
				##Samples with all closest neighbours (normalize_ri == 0), with samples confounding to other classes
				##G_factor = min(G_factor, len([x for x in small_list_r if x ==0.0]))
				normalize_ri = map(lambda x: (x/sum(small_list_r)), small_list_r);
				gi = map(lambda x: np.ceil(G_factor * x), normalize_ri)
		##		print("...Generating {0} samples for class {1}, needed {2}".format(sum(gi),class_small, G_factor))
				for i in range(0, len(csmall_indices)):
					ci = csmall_indices[i]
					cxi = small_list_neighbours[i]
					#If there are no nearest neighbours in the same class for a sample
					#Then assign this sample as its own nearest neighbour
					if(gi[i] > 0):
						if(len(cxi) == 0): 
							cxi=ci ; #print(ci, cxi, gi[i])
						random_k_idx = np.random.choice(cxi, gi[i], replace=True)
						for kix in random_k_idx:
							gap = random.betavariate(1,1)
							si = all_x[ci] + (all_x[kix] - all_x[ci])*gap
							return_np_array = np.append(return_np_array, si.reshape(1,-1), axis=0)
							return_np_labs = np.append(return_np_labs, class_small.reshape(1,-1))
		##	print("...CLASS {0} extended to size {1}".format(class_small, len(return_np_labs[return_np_labs==class_small])))
			
	for class_small in sorted_y[1:]:
			csmall_indices = [i for i,x in enumerate(all_y) if x==class_small]
			get_k_neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(all_x)
			small_list_r, small_list_neighbours = build_small_list(csmall_to_test = csmall_indices, csmall_indices=csmall_indices,get_k_neighbors=get_k_neighbors, all_x=all_x)
			print(Counter(small_list_r))
			get_k_neighbors = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(return_np_array)
			new_csmall_indices = [i for i,x in enumerate(return_np_labs) if x==class_small] 
			small_list_r, small_list_neighbours = build_small_list(csmall_to_test = csmall_indices, get_k_neighbors=get_k_neighbors, all_x=all_x, csmall_indices = new_csmall_indices)
		##	print(Counter(small_list_r))
		##	print(".....END OF CLASS{0}......".format(class_small))
	return return_np_array, return_np_labs

def build_small_list(csmall_to_test, get_k_neighbors, all_x, csmall_indices):
	#This function takes a list of indices indicating the location of the small class
	#And a pre-fit NearestNeighbours function
	#And generates a list of ratios indicating, for each sample,(number of non-self neighbours)/(total neighbours)
	#Also returns the actual list of neighbours for each sample
	small_list_neighbours = []; small_list_r = []
	for ci in csmall_to_test:
		xi = all_x[ci].reshape(1,-1)
		indices = getknn(get_k_neighbors, xi)
		ci_neighbours = [x for x in indices if x in csmall_indices]
		small_list_r.append( float(len(indices) - len(ci_neighbours))/len(indices) )
		small_list_neighbours.append(ci_neighbours)
	return(small_list_r, small_list_neighbours)

def getknn(fit_knn, sample, exclude_self=True):
	#Assumes fit_knn was fit including the sample, so we exclude it from the fit's NearNeighbor results
	dist, ind = fit_knn.kneighbors(sample)
	if exclude_self:
		return(ind[0][1:])
	else:
		return(ind[0])

def getclasses(y):
	y_classes_count = Counter(y)
	sorted_y = sorted(y_classes_count, key=y_classes_count.get, reverse=True)
	cmax = y_classes_count[sorted_y[0]]
	return [y_classes_count, sorted_y, cmax]
	
##SMOTE 
def expand_this_set(minority_data_x, type,  additional_samples=0, class_weight = 0):
        #minority_data_x : nparray of samples with set of features representing each sample
        #It is assumed that the passed data is all from the same (minority) class
        #For k, We will pick either the 5 nearest neighbours, or the size of the input class 
	#Whichever is lesser
	#inflate_factor is 2^class weight, calculated as (biggest_class_size/this_class_size)
	return_np_array = minority_data_x #Get the first set of values for this class as the default samples
	additional_samples = int(additional_samples)
	inflate_factor = int(np.floor(class_weight) - 1)
	if type=="smote":
		#WARNING: Currently not excluding self from set of nearest neighbours when you have a small class,
		# so it can be chosen as the nearest neighbour
		class_size = len(minority_data_x)
		min_neighbours = int(min(5, class_size))
		get_k_neighbors = NearestNeighbors(n_neighbors=min_neighbours, algorithm='auto').fit(minority_data_x)
		#Plus select random minority samples to fill up the additional < (len(minority_set)) samples that are needed
		if inflate_factor > 0:
		##	print("....Synthetically extending dataset {0} time".format(inflate_factor))
			for x in minority_data_x:
				x = x.reshape(1,-1)
				indices = getknn(get_k_neighbors,x, exclude_self=(class_size > min_neighbours))
				#Uniform random sampling to pick a random sample from the k nearest neighbours
				random_k_idx = np.random.choice(indices, inflate_factor, replace=(inflate_factor > len(indices)))
				diff = x - minority_data_x[random_k_idx]
				gap = random.betavariate(1,1); n = x + diff*gap
				return_np_array = np.append(return_np_array, n, axis=0)
		if additional_samples >0:
		##	print("....artifically generating {0} additional samples".format(additional_samples))
			addlist = np.asarray([random.choice(minority_data_x) for _ in range(additional_samples)])
			for x in addlist:
				x = x.reshape(1,-1)
				indices = getknn(get_k_neighbors, x, exclude_self = True)
				random_k_idx = np.random.choice(indices, 1, replace=False)
				diff = x - minority_data_x[random_k_idx]
				gap = random.betavariate(1,1); n = x + diff*gap
				return_np_array = np.append(return_np_array, n, axis=0)	
	if type=="dup":
		return_np_array = np.repeat(minority_data_x, inflate_factor, axis=0)
		#Lastly, add the additional samples from the original by random sampling
		if(additional_samples > 0):
		##	print("....randomly sampling {0} additional samples on top of extending samples (synthetically or otherwise) {1} times".format(additional_samples, inflate_factor-1))
			addlist = np.asarray([random.choice(minority_data_x) for _ in range(additional_samples)])
			return_np_array = np.concatenate((return_np_array, addlist),axis=0)	
	
	return return_np_array

