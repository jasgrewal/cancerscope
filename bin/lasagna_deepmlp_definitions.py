import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.utils import floatX
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.objectives import squared_error as sqerror
from lasagne.layers import get_output as getout
import lasagne.layers 

#Define a custom autoencoder (called during training)
		
def cost_network(network, target_var, l1_reg, l2_reg, learn, train_layers=[], output_layer=[]):
	#for key in dont_train:
	#	network[key].params[network[key].W].remove("trainable")
	#	network[key].params[network[key].b].remove("trainable")
	#Basic loss is negative loss likelihood
	network_out = network[output_layer]
	prediction = lasagne.layers.get_output(network_out)
	loss = T.mean(T.nnet.categorical_crossentropy(prediction,target_var))
	#Shared costs
	l1_penalty = regularize_layer_params(network_out, l1) * l1_reg
	l2_penalty = regularize_layer_params(network_out, l2) * l2_reg
	cost = loss + l2_penalty + l1_penalty
	
	#params = lasagne.layers.get_all_params(network_out, trainable=True)
	#print(params)
	params=[]
	for p in train_layers:
		params.append(network[p].get_params(trainable=True))
	
	params = [item for sublist in params for item in sublist]
	
	print([i.eval().shape for i in params])
	print(params)	
	print(train_layers)
	print("----")
	updates = lasagne.updates.sgd(cost, params, learning_rate=learn)
	return([cost, updates, loss])
	
def build_deep_mlp(input_var=T.matrix('inputs'), target_var1=T.ivector('targets1'), target_var2=T.ivector('targets2'),target_var3=T.ivector('targets3'),target_var4=T.ivector('targets4'), num_features=17688, num_h1=12000, num_h2=10000, num_h3=8000, num_h4=4000, num_o1 = 2, num_o2 = 16, num_o3=40, num_o4=80, learn_rate=0.001, dropout_levels=[0,0,1,1,1],di=0.0, dh=0.5, is_image=False, l1_reg=0.0, l2_reg=0.0):
	#Assume last output layer is a multiclass output structure
	net = {}
	if is_image:
		net['l_in'] = lasagne.layers.InputLayer(shape=(None, 1, num_features, 1), input_var=input_var)
	else:
		net['l_in'] = lasagne.layers.InputLayer(shape=(None, num_features), input_var=input_var)
	
	nonlin = lasagne.nonlinearities.tanh #rectify
	nonlin_decode = lasagne.nonlinearities.softmax  #Multiclass output softmax
	init_weights = lasagne.init.GlorotUniform()
	bias = lasagne.init.Constant(0.)
	network_outputs = []; cost_outputs = []; updates_outputs = []
	train = []; predict=[]; val_fn=[]; test_acc_fn = []
	
	net['h1'] = lasagne.layers.DenseLayer(net['l_in'], num_units=num_h1, W=init_weights, nonlinearity=nonlin, b=bias)
	if(num_o1 != 0):
		net['o1'] = lasagne.layers.DenseLayer(net['h1'], num_units=num_o1, W=init_weights, nonlinearity=nonlin_decode, b=bias)
		cost1, updates1, base_loss = cost_network(net, target_var1, l1_reg, l2_reg, learn_rate, train_layers=["h1","o1"],output_layer="o1")
		network_outputs.append(net['o1']); cost_outputs.append(cost1); updates_outputs.append(updates1)
		train.append(theano.function([net['l_in'].input_var, target_var1], cost1, updates=updates1, allow_input_downcast=True))
		predict.append(theano.function([net['l_in'].input_var], lasagne.layers.get_output(net['o1'], deterministic=True), allow_input_downcast=True))
		predict_fn = theano.function([net['l_in'].input_var], lasagne.layers.get_output(net['o1'], deterministic=True), allow_input_downcast=True)
		test_acc = T.mean(T.eq(T.argmax(lasagne.layers.get_output(net['o1'], deterministic=True), axis=1), target_var1), dtype=theano.config.floatX)
		test_acc_fn.append(theano.function([net['l_in'].input_var, target_var1], [test_acc], allow_input_downcast=True))
		val_fn.append(theano.function([net['l_in'].input_var, target_var1], [base_loss, test_acc], allow_input_downcast=True, on_unused_input='warn'))
			
	####cost_outputs.append(sqerror(getout(o1), target_var_o1).mean() + regularize_layer_params(decoder_l_out, l2) * l2_reg)
	net['h2'] = lasagne.layers.DenseLayer(net['h1'], num_units=num_h2, W=init_weights, nonlinearity=nonlin, b=bias)
	
	net['o2']= lasagne.layers.DenseLayer(net['h2'], num_units=num_o2, W=init_weights, nonlinearity=nonlin_decode, b=bias)
	cost2, updates2, base_loss = cost_network(net, target_var2, l1_reg, l2_reg, learn_rate*1.0, train_layers=["h2","o2"], output_layer="o2")	
	network_outputs.append(net['o2']); cost_outputs.append(cost2); updates_outputs.append(updates2)
	train.append(theano.function([net['l_in'].input_var, target_var2], cost2, updates=updates2, allow_input_downcast=True))
	predict.append(theano.function([net['l_in'].input_var], lasagne.layers.get_output(net['o2'],deterministic=True), allow_input_downcast=True))
	test_acc = T.mean(T.eq(T.argmax(lasagne.layers.get_output(net['o2'], deterministic=True), axis=1), target_var2), dtype=theano.config.floatX)
	test_acc_fn.append(theano.function([net['l_in'].input_var, target_var2], [test_acc], allow_input_downcast=True))
	val_fn.append(theano.function([net['l_in'].input_var, target_var2], [base_loss, test_acc], allow_input_downcast=True, on_unused_input='warn'))
		
	net['h3'] = lasagne.layers.DenseLayer(net['h2'], num_units=num_h3, W=init_weights, nonlinearity=nonlin, b=bias)
	
	net['o3']= lasagne.layers.DenseLayer(net['h3'], num_units=num_o3, W=init_weights, nonlinearity=nonlin_decode, b=bias)
	cost3, updates3, base_loss = cost_network(net, target_var3, l1_reg, l2_reg, learn_rate*0.1, train_layers=["h2","h3","o3"], output_layer="o3")
	network_outputs.append(net['o3']); cost_outputs.append(cost3); updates_outputs.append(updates3)
	train.append(theano.function([net['l_in'].input_var, target_var3], cost3, updates=updates3, allow_input_downcast=True))
	predict.append(theano.function([net['l_in'].input_var], lasagne.layers.get_output(net['o3'],deterministic=True), allow_input_downcast=True))
	test_acc = T.mean(T.eq(T.argmax(lasagne.layers.get_output(net['o3'], deterministic=True), axis=1), target_var3), dtype=theano.config.floatX)
	test_acc_fn.append(theano.function([net['l_in'].input_var, target_var3], [test_acc], allow_input_downcast=True))
	val_fn.append(theano.function([net['l_in'].input_var, target_var3], [base_loss, test_acc], allow_input_downcast=True, on_unused_input='warn'))
	
	net['h4'] = lasagne.layers.DenseLayer(net['h3'], num_units=num_h4, W=init_weights, nonlinearity=nonlin, b=bias)
	
	net['o4'] = lasagne.layers.DenseLayer(net['h4'], num_units=num_o4, W=init_weights, nonlinearity=nonlin_decode, b=bias)
	cost4, updates4, base_loss = cost_network(net, target_var4, l1_reg, l2_reg, learn_rate*0.1, train_layers=["h1","h2","h3","h4","o4"], output_layer="o4")
	network_outputs.append(net['o4']); cost_outputs.append(cost4); updates_outputs.append(updates4)
	train.append(theano.function([net['l_in'].input_var, target_var4], cost4, updates=updates4, allow_input_downcast=True))
	predict.append(theano.function([net['l_in'].input_var], lasagne.layers.get_output(net['o4'],deterministic=True), allow_input_downcast=True))
	test_acc = T.mean(T.eq(T.argmax(lasagne.layers.get_output(net['o4'], deterministic=True), axis=1), target_var4), dtype=theano.config.floatX)
	test_acc_fn.append(theano.function([net['l_in'].input_var, target_var4], [test_acc], allow_input_downcast=True))
	val_fn.append(theano.function([net['l_in'].input_var, target_var4], [base_loss, test_acc], allow_input_downcast=True, on_unused_input='warn'))
	
	return network_outputs, train, predict, val_fn, test_acc_fn

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
