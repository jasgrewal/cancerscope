import numpy as np
from itertools import chain
from collections import Counter 

#Get predicted label
def get_weighted_features_per_class(class_numlabel, modelset, feature_name_mapper=None):
	#Class_numlabel ranges from 0 to max_num_classes-1; is an int
        #modelset is of format [W_1stlayerfrominput, bias_1stlayerfrominput, ..., W_layerbeforeoutput, bias_layerbeforeoutput]
        #feature_name_mapper is a list of length num_input_features with ordered with input names corresponding to input feature index. Izzz optional
        ypred = class_numlabel
        #Now, work backwards, assuming modelset = [W, b, W, b]
        class_bias=[]; class_weights_perlayer=[];
	is_nparray = isinstance(modelset[0], (np.ndarray,np.generic))	
        if is_nparray:
		for i in range(0,len(modelset)/2):
			class_bias.append(modelset[(i*2)+1])
			class_weights_perlayer.append(modelset[(i*2)])
	else:
		for i in range(0,len(modelset)/2):
			class_bias.append(np.array(modelset[(i*2)+1].eval()))
			class_weights_perlayer.append(np.array(modelset[(i*2)].eval()))
	
	#Forward multiplication guys!
	for i in range(1, len(modelset)/2):
		if i==1:
			carry_forward = np.dot((class_weights_perlayer[0] + class_bias[0]), (class_weights_perlayer[i] + class_bias[i]))
		else:
			carry_forward = np.dot(carry_forward, (class_weights_perlayer[i] + class_bias[i]))
	#Final matrix (carry_forward) is num_features x num_classes, so figure out class specific topn features
	top_features_queryclass = np.transpose(carry_forward)[ypred].argsort()[::-1]#[0:100]
	values_features_queryclass = np.transpose(carry_forward)[ypred]
	if feature_name_mapper is not None:
		feature_named = []
		featweights = []
		for feat_ix in top_features_queryclass:
			featweights.append([feature_name_mapper[feat_ix], values_features_queryclass[feat_ix]])
			
			feature_named.append(feature_name_mapper[feat_ix])
	else:
		feature_named = top_features_queryclass
		
	return (feature_named, featweights)
	
def get_weighted_features_per_class_eachstep(class_numlabel, modelset, feature_name_mapper=None):
	#Class_numlabel ranges from 0 to max_num_classes-1
	#modelset is of format [W_1stlayerfrominput, bias_1stlayerfrominput, ..., W_layerbeforeoutput, bias_layerbeforeoutput]
	#feature_name_mapper is a list of length num_input_features with ordered with input names corresponding to input feature index. Izzz optional
	ypred = class_numlabel

	#Now, work backwards, assuming modelset = [W, b, W, b]
	class_bias=[]; class_weights_perlayer=[]; 
	for i in range(0,len(modelset)/2):
		class_bias.append(np.array(modelset[(i*2)+1].eval()))
		class_weights_perlayer.append(np.array(modelset[(i*2)].eval()))
		#class_bias.append(modelset[(i*2)+1])
		#class_weights_perlayer.append(modelset[(i*2)])	
	#Now, presuming input is inputx, figure out which of the input features were weighted the highest
	topn=10
	top_weighted_iminus1thlayer=[]
	for i in range(0, len(class_bias))[::-1]:
		if i == len(class_bias) - 1 :
			#If we are at the instance of initial backprop, claculate top weighted nodes in the out-1 layer
			top_weighted_iminus1thlayer.append((np.transpose(class_weights_perlayer[i])[ypred]).argsort()[::-1][0:topn].tolist())
			top_weighted_currentminusone=list(chain.from_iterable(top_weighted_iminus1thlayer[-1:]))
		else:
			#print("Accessing {0} + 1th layer from initial input layer".format(i))
			#print("...its highest ranked weights are {0}".format(top_weighted_currentminusone))
			templist=[]
			for top_node in np.unique(top_weighted_currentminusone):
				templist.append((np.transpose(class_weights_perlayer[i])[top_node]).argsort()[::-1][0:topn].tolist())
			top_weighted_iminus1thlayer.append(templist)
			top_weighted_currentminusone=list(chain.from_iterable(top_weighted_iminus1thlayer[-1]))
		
	if feature_name_mapper is not None:
	#the last output from the loop was the actual most weighted features, so we use that
		feature_numeric=top_weighted_currentminusone
		feature_named = []
		for feat_ix in feature_numeric:
			feature_named.append(feature_name_mapper[feat_ix])
	else:
		feature_named = top_weighted_currentminusone
	
	return(top_weighted_iminus1thlayer, feature_named)

	
