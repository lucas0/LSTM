import cPickle
import theano, numpy
from theano import tensor as T
import lasagne
import gensim

# nv :: size of our vocabulary
# de :: dimension of the embedding space
# cs :: context window size
nv, de, cs = 1000, 10, 3

embeddings = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, \
(nv+1, de)).astype(theano.config.floatX)) # add one for PADDING at the end
idxs = T.imatrix() 
x = embeddings[idxs].reshape((idxs.shape[0], de*cs))
#Function that creates the word embeddings
create_embeddings = theano.function(inputs=[idxs], outputs=x)

print "Loading data ..."
f = "atis.pkl"
train_set, test_set, dicts = cPickle.load(open(f,'rb'))


def contextwin(l, win):
# win :: int corresponding to the size of the window
# given a list of indexes composing a sentence
# l :: array containing the word indexes
# it will return a list of list of indexes corresponding
# to context windows surrounding each word in the sentence
	assert (win % 2) == 1
	assert win >= 1
	l = list(l)
	lpadded = win // 2 * [-1] + l + win // 2 * [-1]
	out = [lpadded[i:(i + win)] for i in range(len(l))]
	assert len(out) == len(l)
	return out

idx2label = dict((k,v) for v,k in dicts['labels2idx'].iteritems())
idx2ent  = dict((k,v) for v,k in dicts['tables2idx'].iteritems())
idx2word  = dict((k,v) for v,k in dicts['words2idx'].iteritems())

def word2vec(win, dim):
# win :: int corresponding to the size of the window
# dim :: dimensionality of the resulting points
    sentences = [];
    for n in train_set[0]:
        exampleWI = n
        sentences.append(map(lambda x: idx2word[x], exampleWI))
    for n in test_set[0]:
        exampleWI = n
        sentences.append(map(lambda x: idx2word[x], exampleWI))
    model = gensim.models.Word2Vec(sentences, size=dim, window=win, min_count=0, workers=4)
    out = []
    for sentenceA in sentences:
        sentenceContext = []
        for word in sentenceA:
            sentenceContext.append(model[word])
        pad_size = 46 - len(sentenceA)
        pad = numpy.zeros((pad_size,dim))
        sentenceContext = numpy.concatenate([sentenceContext,pad])
        print sentenceContext.shape
        out.append(sentenceContext)
    return out

def get_longest_sentence():
	maior = 0
	for n in train_set[0]:
		if (maior < len(n)):
			maior = len(n)
	return maior


#Printing one Example
def print_Example(index=0):
	if ((index  > len(train_set[0])) or (index < 0)):
		print "Choose a number between 0 and", len(train_set[0])+1, "to print as example"
	else:
		#Word Indexes
		exampleWI = train_set[0][index]
		print "Words:", map(lambda x: idx2word[x], exampleWI)	
		print "Word Idx:", exampleWI
		#Name Entities
		exampleNE = train_set[1][index]
		print "Name Ent:", map(lambda x: idx2ent[x], exampleNE) 
		print "Name Ent Idx:", train_set[1][index]
		#Label Indexes (target)
		exampleLI = train_set[2][index]
		print "IOB:", map(lambda x: idx2label[x], exampleLI)		
		print "IOB Idx:", exampleLI
		

def build_dataset():
	print "Creating Context Windows ..."
	#Creating Word Indexes Context Windows for the Training set
	#train_contextWin = []
	mask_train = numpy.zeros((len(train_set[0]), get_longest_sentence()))
	count = 0
	for tr in train_set[0]:
		pad_size = 46 - len(tr)
		pad = numpy.zeros((pad_size))

		padded_train = numpy.concatenate([tr, pad])
		padded_train_out = numpy.concatenate([train_set[2][count], pad])
		train_set[2][count] = padded_train_out
		#train_contextWin.append(contextwin(padded_train, cs))

		mask_train[count, :len(tr)] = 1
		count += 1

	#Creating Word Indexes Context Windows for the Test set
	#test_contextWin = []
	mask_test = numpy.zeros((len(test_set[0]), get_longest_sentence()))
	count = 0
	for te in test_set[0]:
		pad_size = 46 - len(te)
		pad = numpy.zeros((pad_size))

		padded_test = numpy.concatenate([te, pad])
		padded_test_out = numpy.concatenate([test_set[2][count], pad])
		test_set[2][count] = padded_test_out
		#test_contextWin.append(contextwin(padded_test, cs))

		mask_test[count, :len(te)] = 1
		count += 1
        
	#Creating the Word Embeddings from Word Indexes Context Windows for the Training set
	#print train_contextWin
	
	contextWin = word2vec(5,3) ## 5 word from window, 3 dimensions 
	train_contextWin = contextWin[:len(train_set[0])-1]
	test_contextWin = contextWin[len(train_set[0]):]
	
	raise SystemExit
    
    ##########################################
    
    
	print "Creating Word Embeddings ..."
	train_Emb = []
	for i in train_contextWin:
		train_Emb.append(create_embeddings(i))

	#Creating the Word Embeddings from Word Indexes Context Windows for the Test set
	test_Emb = []
	for i in test_contextWin:
		test_Emb.append(create_embeddings(i))

	train_in = numpy.array(train_Emb)
	test_in = numpy.array(test_Emb)

	train_out = numpy.array(train_set[2])
	test_out = numpy.array(test_set[2])
        
        

	return train_in, test_in, train_out.astype('int32'), test_out.astype('int32'), mask_train.astype('int32'), mask_test.astype('int32')

#print_Example(120)

input_var = T.tensor3('input_var')
mask_var = T.imatrix('mask_var')
target_var = T.imatrix('target_var')

#################
## BUILD MODEL ##
#################
batch_size = 19
num_epochs = 10
num_units = 300
num_classes = 127

l_inp = lasagne.layers.InputLayer((batch_size, 46, 30), input_var=input_var)
l_mask = lasagne.layers.InputLayer((batch_size, 46), mask_var)

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
l_den = lasagne.layers.DenseLayer(l_shp, 127, nonlinearity=lasagne.nonlinearities.softmax)
l_out = lasagne.layers.ReshapeLayer(l_den, (-1, 46, 127))

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.categorical_crossentropy(prediction.reshape((-1,127)), target_var.flatten())
loss = lasagne.objectives.aggregate(loss, mask_var.flatten())

params = lasagne.layers.get_all_params(l_out, trainable=True)
print "Computing updates ..."
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
#test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
#test_loss = lasagne.objectives.categorical_crossentropy(test_prediction.reshape((-1,127)), target_var.flatten(2))
#test_loss = test_loss.mean()

# As a bonus, also create an expression for the classification accuracy:
#test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

print "Compiling functions ..."
train_fn = theano.function([input_var, target_var, mask_var], loss, updates=updates)


train_in, test_in, train_out, test_out, mask_train, mask_test = build_dataset()

print train_in.shape
print test_in.shape
idx = 0
for epoch in range(num_epochs):
	print "Training epoch", epoch+1
	s_in = train_in[idx:idx+batch_size]
	s_out = train_out[idx:idx+batch_size]
	s_mask = mask_train[idx:idx+batch_size]
	print train_fn(s_in, s_out, s_mask)
	idx = idx + batch_size
