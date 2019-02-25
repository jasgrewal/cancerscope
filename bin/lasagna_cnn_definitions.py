import lasagne
import numpy as np
#Define implicit matrix for input using a theano fuction
from theano import function
import theano.tensor as T
import theano

#Input is a 1 dimensional vector of all variabls
x = T.vector('data_vector', dtype='float32')
keep_idx = T.vector('norm_idx', dtype='int8')
norm_axis = x[keep_idx]
norm_axis_no0inv = T.switch(theano.tensor.eq(norm_axis,0), 1, 1/norm_axis)
opp_axis = x
return_mat = theano.tensor.dot(norm_axis_no0inv.reshape((-1,1)), opp_axis.reshape((1,-1)))
fn_make_mat_theano = theano.function(inputs = [x, keep_idx], outputs=return_mat)

#Set 0 values in normalizing vector to 1; flip/invert others to make them scaling factors
class vec_to_image(lasagne.layers.Layer):
	def __init__(self, incoming, num_x1, num_x2, norm_idx, **kwargs):
		super(vec_to_image, self).__init__(incoming,  **kwargs)
		self.num_x1 = num_x1
		self.num_x2 = num_x2
		self.norm_idx = norm_idx
	
	def get_output_for(self, input,**kwargs):
		#return fn_make_mat_theano(input, self.norm_idx)
		input_subset = input[self.norm_idx]
		norm_axis_no0inv = T.switch(theano.tensor.eq(input_subset,0),1,1/input_subset)
		return_mat = theano.tensor.dot(norm_axis_no0inv.reshape((-1,1)), input.reshape((1,-1))).reshape((1,1,self.num_x2, self.num_x1))
		#print(return_mat.shape)
		#getmem()	
		return(return_mat)
	
        def get_output_shape_for(self, input_shape):
		return(1, 1, self.num_x2, self.num_x1)

#Define a custom CNN (called during training). 
def build_custom_cnn(n_out, num_features, normidx, depth, drop_input, drop_hidden=0, width=None, input_var=None, is_image=False, num_responses=1, num_filters=1, filter_size=5, stride_conv=2, drop_out=False):
	if is_image:
		#make_testable_mapping = lapply(fn_make_mat_theano(x,
		#tvar = [fn_make_mat_theano(z, normidx) for z in input_var.eval()]
		#network = lasagne.layers.InputLayer(shape=(None,num_features), input_var=input_var)
		#network = vec_to_image(network, num_x1=num_features, num_x2=num_responses, norm_idx=normidx)
		network = lasagne.layers.InputLayer(shape=(None, 1, num_responses,num_features), input_var=input_var)
	else:
		network = lasagne.layers.InputLayer(shape=(None, 1,num_features), input_var=input_var)
	
	if width is None:
		width = num_features
		print("setting fully connected layer size to number of inputs, {1}".format(num_features))
	
	#The CNN will have a stride of 1, fully connected at pooling. There is usually no dropout in CNNs
	#First a convolution layer, then a max pooling layer
	nonlin = lasagne.nonlinearities.tanh
	conv_nonlin = lasagne.nonlinearities.rectify
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size = (1, 15*stride_conv))
	for _ in range(depth):
		if is_image:
			#If input is 2d image, add a 2D convolution layer, as signal is in 2 dims
			network = lasagne.layers.Conv2DLayer(
				network, num_filters = num_filters, filter_size=(filter_size,filter_size), nonlinearity=conv_nonlin, W=lasagne.init.GlorotUniform(),
				pad = 'full', stride = (stride_conv))
			#Two common variations are F=3, S=2; and more commonly F=2, S=2
			network = lasagne.layers.MaxPool2DLayer(network, pool_size = (stride_conv, stride_conv), stride = stride_conv, pad=0)
		
		else:
			network = lasagne.layers.Conv1DLayer(
				network, num_filters = num_filters, filter_size=filter_size, nonlinearity=conv_nonlin, W=lasagne.init.GlorotUniform(), pad = 'full', stride = stride_conv)
			network = lasagne.layers.MaxPool1DLayer(network, pool_size = 2*stride_conv)

	t2 = lasagne.layers.get_all_layers(network)
	print(map(lambda x: lasagne.layers.get_output_shape(x), t2))
	
	##Then add a fully connected layer, dropout as per input options
	network = lasagne.layers.DenseLayer(network, num_units = num_responses + num_features, nonlinearity = nonlin)
	if drop_hidden:
		network = lasagne.layers.dropout(network, p=drop_hidden)
	
	#Final output layer is softmax, for multiclass output
	softmax = lasagne.nonlinearities.softmax
	network = lasagne.layers.DenseLayer(network, n_out, nonlinearity=softmax)
	if drop_out:
		network = lasagne.layers.dropout(network, p=drop_hidden)
	
	print("---full network---")
	t2 = lasagne.layers.get_all_layers(network)
        print(map(lambda x: lasagne.layers.get_output_shape(x), t2))
	return network

#Define iterator over minibatches of inputs (x) and corresponding labels (y)
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
        	else:
			excerpt = slice(start_idx, start_idx + batchsize)
	yield inputs[excerpt], targets[excerpt]

#Define function to reshape 1 dimensional input to an image with 1 channel and width 1
def make_1d_image(x):
	return(x.reshape((x.shape[0],1,x.shape[1],1)))

import theano.sandbox.cuda.basic_ops as sbcuda
import numpy as np
import theano.tensor as T
T.config.floatX = 'float32'
import psutil
import os

def getmem():
	GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
	freeGPUMemInGBs = GPUFreeMemoryInBytes/1024./1024/1024
	print "Your GPU has %s GBs of free memory" % str(freeGPUMemInGBs)
	#An operation is to be executed below
	#testData = shared(np.random.random((5000, 256 * 256)).astype(T.config.floatX), borrow = True)
	#print "The tasks above used %s GBs of your GPU memory. The available memory is %s GBs" % (str(freeGPUMemInGBs - GPUFreeMemoryInBytes/1024./1024/1024), str(GPUFreeMemoryInBytes/1024./1024/1024))
	print("...CPU percent used {0}, physical mem available {1} GB, swap avail {2} GB".format(psutil.cpu_percent(), psutil.virtual_memory().available >> 30, psutil.swap_memory().free >> 30))
	py = psutil.Process(os.getpid())
	memoryUse = py.memory_info()[0]/2.**30
	print("....Process CPU memory use {0} GB".format(memoryUse))

#Define a function to convert a 1d vector into a 2d matrix normalized by an input axis
def normed_idx_vector(idx_norm, data_vector, exclude_normidx=1):
	norm_axis = np.take(data_vector, idx_norm)
	#Since using norm_axis values as scaling factor (dividing), cannot have any zeroes in it
	#Set 0 values in norm_axis vector to 1
	norm_axis[norm_axis==0] = 1
	if exclude_normidx:
		data_vector = np.delete(data_vector, idx_norm)
	##getmem()
	return_dat = np.repeat(data_vector, len(norm_axis)).reshape(len(data_vector), len(norm_axis)).transpose()
	return_dat = return_dat / norm_axis[:, np.newaxis]
	#Shape of returned object is (norming_axis, data_vector)
	#If exclude_normidx is False (or 0), returned object shape is (norming_axis, data_vector without norming elements)
	return(return_dat)

#fn_make_tensor_theano = theano.function(inputs=[xmat, xidx, keep_idx], outputs=[xidx,return_mat])
#tfat = theano.scan(fn=lambda prev_count, next_mat: fn_make_tensor_theano(xmat, prev_step, keep_idx), outputs_info=0, n_steps=xmaxn)

#tfat = theano.scan(fn=fn_make_tensor_theano(xmat, prev_step, keep_idx),outputs_info=[dict(initial=xmat, taps=-1)], sequences=xmat, n_steps=xmaxn) # n_steps=xmaxn)
#xmaxn = T.vector('max_samples', dtype='int8')
#results, updates = theano.scan(fn=fn_make_tensor_theano(fn=lambda prev_step, nmat = fn_make_tensor_theano(xmat, prev_step, keep_idx), outputs_info=[{'initial':xmat}], n_steps=xmaxn)

#tfat = theano.function(inputs=[xmaxn], outputs=xmaxr)

#A = T.matrix("A")
#idx_range = T.arange(A.shape[0])
#updated_img = T.switch(theano.tensor.eq(A, 0), 0, fn_make_mat_theano(A, keep_idx))
#result, updates = theano.scan(fn=lambda idx: fn_make_mat_theano(T.as_tensor_variable(A[idx]), keep_idx), sequences=idx_range)
#fn_mat_img = theano.function(inputs = [A, keep_idx], outputs=[results,updates])
#xs = T.matrix('inputs')

#snum = xs.shape[0]

