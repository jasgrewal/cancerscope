import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.utils import floatX
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

#Define a custom mlp (called during training)
def build_custom_mlp(n_out, num_features, depth, width, drop_input, drop_hidden, input_var=None, is_image=False):
	network={}
	if is_image:
		network['input'] = lasagne.layers.InputLayer(shape=(None, 1, num_features,1), input_var=input_var)
	else:
		network['input'] = lasagne.layers.InputLayer(shape=(None,num_features), input_var=input_var)
	if drop_input:
		network['input'] = lasagne.layers.dropout(network['input'], p=drop_input)
	
	nonlin = lasagne.nonlinearities.tanh
	for i in range(depth):
		if i == 0:
			network["hidden0"] = lasagne.layers.DenseLayer(network['input'],width, nonlinearity=nonlin) 
		else:
			network["hidden"+str(i)]=lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
		
		if drop_hidden:
			network["hidden"+str(i)] = lasagne.layers.dropout(network["hidden"+str(i)], p=drop_hidden)
	
	#Final output layer is softmax, for multiclass output
	softmax = lasagne.nonlinearities.softmax
	network['output'] = lasagne.layers.DenseLayer(network["hidden"+str(i)], n_out, nonlinearity=softmax)
	return(network['output'], network)

def load_network(network_params_npz):
	with np.load(network_params_npz) as f:
		network_params = [f['arr_%d' % i] for i in range(len(f.files))]
	n_in = network_params[0].shape[0]; n_class = network_params[-1].shape[0]
	num_hidden_layers = (len(network_params)/2) - 1
	num_hidden_nodes = network_params[0].shape[1]
	target_var = T.ivector('targets'); input_var=T.matrix('inputs')
	ntwk = build_custom_mlp(input_var=input_var, n_out=n_class, num_features=n_in, depth=num_hidden_layers, width=num_hidden_nodes, drop_input=0.0, drop_hidden=0.0)
	return(ntwk)

def build_transfer_mlp(old_network, drop_input, drop_hidden, input_var=None, is_image=False):
	if os.path.isfile(old_network):
		old_network = load_network(old_network)
	

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

class TransposedDenseLayer(lasagne.layers.DenseLayer):
	def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,**kwargs):
		super(TransposedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)
	
	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.num_units)
	
	def get_output_for(self, input, **kwargs):
		if input.ndim > 2:
			input = input.flatten(2)
		activation = T.dot(input, self.W.T)
		if self.b is not None:
			activation = activation + self.b.dimshuffle('x', 0)
		return self.nonlinearity(activation)
