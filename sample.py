

import numpy as np
import nolearn.lasagne as nl
import lasagne as lsgn

X = np.random.standard_normal([256,10]).astype(np.float32)
y = X[:,0:2]

l = lsgn.layers.InputLayer(shape=(None, X.shape[1]))
l = lsgn.layers.DenseLayer(l, num_units=y.shape[1], nonlinearity=None)
net = nl.NeuralNet(l, 
	regression=True, 
	update_learning_rate=0.1,
	max_epochs = 10, 
	verbose = 1)

net.fit(X,y)

X_test = np.random.standard_normal([128,10]).astype(np.float32)
y_test = X_test[:,0:2]
print ("Score on test:" + str(net.score(X_test, y_test)))
