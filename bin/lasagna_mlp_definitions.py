import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.utils import floatX
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

#Define a custom autoencoder (called during training)
def build_custom_autoencoder(n_encoded, num_features, target_var, learn_rate=0.001, input_var=None, is_image=False, keep_positive=False, l2_reg=0.001):
	if is_image:
		l_in = lasagne.layers.InputLayer(shape=(None, 1, num_features, 1), input_var=input_var)
	else:
		l_in = lasagne.layers.InputLayer(shape=(None, num_features), input_var=input_var)
	smallest_size = 2	
	nonlin_encode = lasagne.nonlinearities.sigmoid #linear
	nonlin_decode = lasagne.nonlinearities.rectify #sigmoid
	init_weights = lasagne.init.Constant(val=1.0) 
	init_weights = lasagne.init.Normal(std=1, mean = 1)
	#init_weights = lasagne.init.GlorotUniform()
	init_weights1 = theano.shared(floatX(init_weights((num_features, n_encoded))))
	init_weights2 = theano.shared(floatX(init_weights((n_encoded, n_encoded/2))))
	init_weights3 = theano.shared(floatX(init_weights((n_encoded/2, smallest_size))))
	print("Encode nonlin: {0}\nDecode nonlin: {1}\nInit weights type:{2}".format(nonlin_encode,nonlin_decode,init_weights))	
	if keep_positive:
		init_weights = theano.shared(floatX(init_weights((num_features, n_encoded))))
		init_weights = T.exp(init_weights)
		print(init_weights)	
	#Encoder step
	encoder_l_out1 = lasagne.layers.DenseLayer(l_in, num_units=n_encoded, W = init_weights1, nonlinearity=nonlin_encode, b=lasagne.init.Constant(0.))
	
	encoder_l_out2 = lasagne.layers.DenseLayer(encoder_l_out1, num_units=n_encoded/2, W = init_weights2, nonlinearity=nonlin_encode, b=lasagne.init.Constant(0.))
	encoder_l_out = lasagne.layers.DenseLayer(encoder_l_out2, num_units=smallest_size, W = init_weights3, nonlinearity=nonlin_encode, b=lasagne.init.Constant(0.))
		
	#Decoder step
	#	decoder_l_out = lasagne.layers.DenseLayer(encoder_l_out, num_units = num_features/2, W=encoder_l_out2.W.T, nonlinearity=nonlin_decode, b=lasagne.init.Constant(0.))
	decoder_l_out = TransposedDenseLayer(encoder_l_out, num_units = n_encoded/2, W=encoder_l_out.W, nonlinearity=nonlin_decode, b=lasagne.init.Constant(0.))
	decoder_l_out =  TransposedDenseLayer(decoder_l_out, num_units = n_encoded, W=encoder_l_out2.W, nonlinearity=nonlin_decode, b=lasagne.init.Constant(0.))
	decoder_l_out =  TransposedDenseLayer(decoder_l_out, num_units = num_features, W=encoder_l_out1.W, nonlinearity=nonlin_decode, b=lasagne.init.Constant(0.))	
	#	decoder_l_out = lasagne.layers.DenseLayer(dencoder_l_out, num_units = num_features, W=encoder_l_out1.W.T, nonlinearity=nonlin_decode, b=lasagne.init.Constant(0.))
	
	##Decoder with tied weights
	#First add inverting dense layer
	#decoder_l_out = lasagne.layers.InverseLayer(encoder_l_out, encoder_l_out)
	###decoder_l_out = TransposedDenseLayer(encoder_l_out, num_units = num_features, W=encoder_l_out.W, nonlinearity=nonlin_decode, b=lasagne.init.Constant(0.))
	#decoder_l_out = lasagne.layers.DenseLayer(encoder_l_out, num_units = num_features, W=encoder_l_out.W.T, nonlinearity=nonlin_decode)
	
	#Define some theano vars
	target_values = target_var
	encoded_output = lasagne.layers.get_output(encoder_l_out)
	network_output = lasagne.layers.get_output(decoder_l_out)
	
	cost = lasagne.objectives.squared_error(network_output, target_values).mean() + regularize_layer_params(decoder_l_out, l2) * l2_reg
	all_params = lasagne.layers.get_all_params(decoder_l_out, trainable=True)
	
	#AdaDelta updates for training
	updates = lasagne.updates.rmsprop(cost, all_params, learning_rate=learn_rate)
	##updates = lasagne.updates.adadelta(cost, all_params)
	##updates = lasagne.updates.sgd(cost, all_params, learning_rate=learn_rate)
	
	#Theano functions for key queries
	train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
	network = decoder_l_out
	predict = theano.function([l_in.input_var], network_output, allow_input_downcast=True)
	encode = theano.function([l_in.input_var], encoded_output, allow_input_downcast=True)
	return network, train, predict, encode

#Define a custom mlp (called during training)
def build_custom_mlp(n_out, num_features, depth, width, drop_input, drop_hidden, input_var=None, is_image=False):
	if is_image:
		network = lasagne.layers.InputLayer(shape=(None, 1, num_features,1), input_var=input_var)
	else:
		network = lasagne.layers.InputLayer(shape=(None,num_features), input_var=input_var)
	if drop_input:
		network = lasagne.layers.dropout(network, p=drop_input)
	nonlin = lasagne.nonlinearities.tanh
	for _ in range(depth):
		network=lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
		if drop_hidden:
			network = lasagne.layers.dropout(network, p=drop_hidden)
	#Final output layer is softmax, for multiclass output
	softmax = lasagne.nonlinearities.softmax
	network = lasagne.layers.DenseLayer(network, n_out, nonlinearity=softmax)
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
