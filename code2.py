import theano, numpy
from theano import tensor as T
import lasagne

#fake data, same shapes
train_in = numpy.zeros((100,46,30)).astype('float32')
train_out = numpy.zeros((100,46)).astype('int32')
mask_train = numpy.zeros((100,46)).astype('int32')

input_var = T.tensor3('input_var')
mask_var = T.imatrix('mask_var')
target_var = T.imatrix('target_var')

num_epochs = 10
batch_size = 10

#################
## BUILD MODEL ##
#################
num_units, num_classes = 300, 127

l_inp = lasagne.layers.InputLayer((batch_size, 46, 30), input_var=input_var)
l_mask = lasagne.layers.InputLayer((batch_size, 46), input_var=mask_var)

l_lstm = lasagne.layers.LSTMLayer(l_inp, num_units=num_units,
ingate=lasagne.layers.Gate(),
forgetgate=lasagne.layers.Gate(),
cell=lasagne.layers.Gate(
W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
outgate=lasagne.layers.Gate(),
nonlinearity=lasagne.nonlinearities.tanh,
cell_init=lasagne.init.Constant(0.),
hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
precompute_input=True, mask_input=l_mask)

l_shp = lasagne.layers.ReshapeLayer(l_lstm, (-1, num_units))
l_den = lasagne.layers.DenseLayer(l_shp, num_classes, nonlinearity=lasagne.nonlinearities.softmax)
l_out = lasagne.layers.ReshapeLayer(l_den, (-1, 46, num_classes))

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.categorical_crossentropy(prediction.reshape((-1,num_classes)), target_var.flatten())
loss = lasagne.objectives.aggregate(loss, mask_var.flatten())

params = lasagne.layers.get_all_params(l_out, trainable=True)

print "Computing updates ..."
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

print "Compiling functions ..."
train_fn = theano.function([input_var, target_var, mask_var], loss, updates=updates)

idx = 0
for epoch in range(num_epochs):
	print "Training epoch", epoch+1
	s_in = train_in[idx:idx+batch_size]
	print s_in.shape
	s_out = train_out[idx:idx+batch_size]
	print s_out.shape
	s_mask = mask_train[idx:idx+batch_size]
	train_fn(s_in, s_out, s_mask)
	idx = idx + batch_size

