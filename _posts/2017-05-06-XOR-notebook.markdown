---
layout: post
comments: true
title:  "XOR Model"
excerpt: "XOR Model using Theano"
date:   2017-05-06 15:40:00
mathjax: false
---

In this blog post, I will discuss how can you train a simple classifier for a
XOR function using theano.

```{.python .input  n=1}
%matplotlib inline
%load_ext autoreload
%autoreload 2
```

```{.python .input  n=2}
from sklearn.externals import joblib
import pandas as pd
import redis
import numpy as np 
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
from operator import itemgetter
import itertools
import redis
from collections import Counter
from ast import literal_eval as ast_literal_eval
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
pd.options.display.max_colwidth = 250
```

```{.python .input  n=3}
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import matplotlib
import theano
import theano.tensor as T
from IPython.display import Image
from IPython.display import SVG
import timeit
```

```{.python .input  n=4}
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = train_X[:, 0].min() - .5, train_X[:, 0].max() + .5
    y_min, y_max = train_X[:, 1].min() - .5, train_X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.Spectral)
```

```{.python .input}

```

```{.python .input}

```

##### Define the network

First of all, we need to define two variables in theano. The X variable is a
matrix, which represents the input data with D number of features. The y vector
is out output label.

```{.python .input  n=6}
X = T.matrix('X')
y = T.lvector('y')
```

```{.python .input  n=73}
num_examples = 4
nn_input_dim = 2
nn_hdim = 4
nn_output_dim = 2

epsilon = 0.01
regularization = 0.01
```

```{.python .input  n=74}
# Shared variables with initial values. We need to learn these.
W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
b1 = theano.shared(np.zeros(nn_hdim), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
b2 = theano.shared(np.zeros(nn_output_dim), name='b2')
```

```{.python .input  n=75}
z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)

loss_reg = 1./num_examples * regularization/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2))) 

# the loss function we want to optimize
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

# Returns a class prediction
prediction = T.argmax(y_hat, axis=1)
```

```{.python .input  n=76}
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)
dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
```

```{.python .input  n=77}
gradient_step = theano.function(
    [X, y],
    updates=((W2, W2 - epsilon * dW2),
             (W1, W1 - epsilon * dW1),
             (b2, b2 - epsilon * db2),
             (b1, b1 - epsilon * db1)))
```

```{.python .input  n=78}
forward_prop = theano.function([X], y_hat)
calculate_loss = theano.function([X, y], loss)
prediction = theano.function([X], prediction)
```

```{.python .input  n=79}
# This function learns parameters for the neural network and returns the model.
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(num_passes=10000, print_loss=False):
    
    # Re-Initialize the parameters to random values. We need to learn these.
    # (Needed in case we call this function multiple times)
    np.random.seed(0)
    W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim))
    W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_output_dim))
    
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
        # This will update our parameters W2, b2, W1 and b1!
        gradient_step(train_X, train_y)
        
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print "Loss after iteration %i: %f" %(i, calculate_loss(train_X, train_y))
```

```{.python .input}

```

```{.python .input}

```

```{.python .input}

```

#### Xor function

```{.python .input  n=84}
train_X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
train_y = np.array([0, 1, 1, 0])
```

```{.python .input  n=85}
build_model(print_loss=True)
```

```{.json .output n=85}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Loss after iteration 0: 0.694005\nLoss after iteration 1000: 0.500917\nLoss after iteration 2000: 0.309874\nLoss after iteration 3000: 0.190666\nLoss after iteration 4000: 0.140858\nLoss after iteration 5000: 0.119209\nLoss after iteration 6000: 0.108479\nLoss after iteration 7000: 0.102523\nLoss after iteration 8000: 0.098904\nLoss after iteration 9000: 0.096533\n"
 }
]
```

```{.python .input  n=88}
plot_decision_boundary(lambda x: prediction(x))
```

```{.json .output n=88}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x10cdc3450>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

#### And gate

```{.python .input  n=89}
train_X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
train_y = np.array([1, 1, 1, 0])
```

```{.python .input  n=90}
build_model(print_loss=True)
```

```{.json .output n=90}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Loss after iteration 0: 0.769078\nLoss after iteration 1000: 0.103026\nLoss after iteration 2000: 0.057478\nLoss after iteration 3000: 0.045957\nLoss after iteration 4000: 0.041489\nLoss after iteration 5000: 0.039324\nLoss after iteration 6000: 0.038112\nLoss after iteration 7000: 0.037359\nLoss after iteration 8000: 0.036850\nLoss after iteration 9000: 0.036480\n"
 }
]
```

```{.python .input  n=91}
plot_decision_boundary(lambda x: prediction(x))
```

```{.json .output n=91}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<matplotlib.figure.Figure at 0x111dc7990>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```
