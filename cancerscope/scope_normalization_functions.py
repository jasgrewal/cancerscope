from sklearn import preprocessing
import numpy as np
from scipy.stats.mstats import rankdata as ranksample

#Normalization functions
def norm_minmax(x, min=0, max=1):
	#This function is useful for sparse data with lots of zeroes
	##print("....by minmax, min {0} and max {1}".format(min,max))
	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(min,max))
	x_minmax = min_max_scaler.fit_transform(x) #Does it across each feature (i.e. column)
	return [x_minmax, min_max_scaler]

#def norm_scaler_reapply(scaler_func, x):
#        x_transformed = scaler_func.transform(x)
#        return x_transformed

def norm_rasterize(x_sample):
	#x_sample is an ndarray, of dimension (#features x 1)
	#Remember that newlist=list(oldlist) will still be mutable for nested lists in oldlist
	rasterize_range = range(1,len(x_sample)+1)
	ycopy=x_sample.copy(); prev=0
	for ordered_value in sorted(set(x_sample)):
		used_range = rasterize_range[prev:prev+len(x_sample[x_sample==ordered_value])]
		ycopy[x_sample==ordered_value] = sum(used_range)/len(used_range)
		prev = prev + len(x_sample[x_sample==ordered_value])
	#print(len(ycopy))
	return ycopy

def apply_norm_func(norm_func, xdat, bysamples=1):
        #If bysamples is 0, xdat[0] must be train_x, and xdat[1] must be test_x
	#If bysamples is 1, xdat is #samples x #features (not feat x samples, #Edit 03 Feb, 2017 | jgrewal)
	result = []
	if norm_func not in ["none", "rastminmax"]:
		raise ValueError('Incorrect normalization function, {0} passed'.format(norm_func))
	if norm_func=="none":
		result=xdat

	if norm_func=="rastminmax":
		temprast=[]
		#First rasterize
		if(type(xdat)==list):
			#xdat is sample x features, then for each sample
			#xdat is features x samples, then for each feature (bysamples==0)
			result=[]
			for xtype in xdat:
				temp = (map(norm_rasterize,xtype))
				temprast.append(temp)
		else:
			##print("Rasterized {0}".format(xdat.shape))
			temprast = ranksample(xdat, axis=bysamples) #(map(norm_rasterize,xdat))
			
		#Then minmax per sample (if bysamples==1) or by features (if bysamples==0)
		#if bysamples==0:
		#	[train_x_norm, scalefunc] = norm_minmax(x=temprast[0])
		#	test_x_norm=norm_scaler_reapply(scalefunc, temprast[1])
		#	valid_x_norm=norm_scaler_reapply(scalefunc, temprast[2])
		#	result=[train_x_norm, test_x_norm, valid_x_norm]
		#else:
		if bysamples==1:
			temprast = np.asarray(temprast)
			temprast = np.transpose(temprast)
			##print("Minmax'ing across each column {0}".format(temprast.shape))
			##print(temprast);
			##print(temprast.shape)
			x_norm = norm_minmax(x=temprast)[0]
			result = x_norm
			##print(result.shape)
			##Edit 03 Feb, 2017 | jgrewal
			result = result.transpose()
	
	if type(result) != list:
		print("norm result shape {0}".format(result.shape))
	return result


