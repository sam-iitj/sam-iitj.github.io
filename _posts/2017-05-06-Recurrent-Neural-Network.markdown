---
layout: post
comments: true
title:  "Parity problem with RNN"
excerpt: "Parity problem with RNN"
date:   2017-05-05 15:40:00
mathjax: false
---

#### Parity problem with RNN

For last couple of days, I was working on understanding and implementing simple recurrent neural network and its variants. So, I thought that it would be a good idea to share what I learned while experimenting with these simple models and its variants. Here, I have tried solving the parity problem using RNN, GRU and LSTM. So, what is a parity problem first of all. Parity compuation is widely used in communications technology to check if the recieved message is intact or not. Basically, the sender and reciever can have an agreement between them via which they decide the parity of the incoming messages. So, if the agreement between the two parties is of even parity. Then, it is the responsibility of the sender to make sure that every message send on the network has even number of zeros. For example, if the message sender wants to send is 01101. But, the agreement between the two parties is of even parity. Then, the sender system can send a 1 in the end of the message. 


```python
%matplotlib inline
%load_ext autoreload
%autoreload 2
```


```python
import theano
import theano.tensor as T 
import numpy as np
import matplotlib.pyplot as plt 
from theano import pp
from IPython.display import display, Math, Latex
```

The idea behing a basic Recurrent Neural Network is to have a unit which can take into account the past while processing the new input. So, the idea is that each state of the RNN unit is a function of the current input $x_{t}$ as well as the previous state of the system $h_{t-1}$.
$$y(t) = f(x_{t-1}, h_{t-1})$$

Here's the implementation for a basic Recurrent Neural Network. 

$$h_{t} = x_{t}^TW_{x} + h_{t-1}^TW_{h} + b_{h}$$
$$y_{t} = softmax(h_{t}^TW_{o} + b_{o})$$

##### RNN


```python
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, all_parity_pairs_with_sequence_labels


class SimpleRNN:
    def __init__(self, M):
        self.M = M # hidden layer size

    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1] # X is of size N x T(n) x D
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f = activation

        # initial weights
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        # make them theano shared
        self.Wx = theano.shared(Wx, name="Wx")
        self.Wh = theano.shared(Wh, name="Wh")
        self.bh = theano.shared(bh, name="bh")
        self.h0 = theano.shared(h0, name="h0")
        self.Wo = theano.shared(Wo, name="Wo")
        self.bo = theano.shared(bo, name="bo")
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=prediction, name="predict")
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, y],
            updates=updates, 
            name='train'
        )

        costs = []
        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in xrange(N):
                c, p, rout = self.train_op(X[j], Y[j])
                cost += c
                if p[-1] == Y[j,-1]:
                    n_correct += 1
            print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
            costs.append(cost)
            if n_correct == N:
                break

        if show_fig:
            plt.plot(costs)
            plt.show()


def parity(B=12, learning_rate=10e-5, epochs=200):
    X, Y = all_parity_pairs_with_sequence_labels(B)

    rnn = SimpleRNN(4)
    rnn.fit(X, Y, learning_rate=learning_rate, epochs=epochs, activation=T.tanh, show_fig=True)  
    return rnn
```


```python
rnn = parity()
```

    i: 0 cost: 2712.45241092 classification rate: 0.491951219512
    i: 1 cost: 2493.5639616 classification rate: 0.506097560976
    i: 2 cost: 2408.72680792 classification rate: 0.500731707317
    i: 3 cost: 2371.86381031 classification rate: 0.50243902439
    i: 4 cost: 2319.63690975 classification rate: 0.503170731707
    i: 5 cost: 2251.80445496 classification rate: 0.497317073171
    i: 6 cost: 2197.44248021 classification rate: 0.492682926829
    i: 7 cost: 2159.41030698 classification rate: 0.499268292683
    i: 8 cost: 2134.48165162 classification rate: 0.506341463415
    i: 9 cost: 2116.233429 classification rate: 0.496341463415
    i: 10 cost: 2061.34719322 classification rate: 0.51243902439
    i: 11 cost: 1982.76503601 classification rate: 0.490243902439
    i: 12 cost: 1959.84536855 classification rate: 0.508536585366
    i: 13 cost: 1950.24146986 classification rate: 0.498048780488
    i: 14 cost: 1938.33274301 classification rate: 0.502682926829
    i: 15 cost: 1932.21508866 classification rate: 0.493414634146
    i: 16 cost: 1926.89688701 classification rate: 0.506829268293
    i: 17 cost: 1923.03518397 classification rate: 0.507317073171
    i: 18 cost: 1920.81061191 classification rate: 0.496097560976
    i: 19 cost: 1918.85585256 classification rate: 0.488048780488
    i: 20 cost: 1916.06850649 classification rate: 0.491463414634
    i: 21 cost: 1915.53184621 classification rate: 0.482682926829
    i: 22 cost: 1912.4453205 classification rate: 0.510243902439
    i: 23 cost: 1912.82618378 classification rate: 0.494390243902
    i: 24 cost: 1911.20794161 classification rate: 0.499512195122
    i: 25 cost: 1910.3184564 classification rate: 0.506097560976
    i: 26 cost: 1909.92869146 classification rate: 0.48756097561
    i: 27 cost: 1908.47652956 classification rate: 0.499512195122
    i: 28 cost: 1907.60184562 classification rate: 0.502195121951
    i: 29 cost: 1906.95563278 classification rate: 0.503170731707
    i: 30 cost: 1907.0365338 classification rate: 0.491707317073
    i: 31 cost: 1906.88682937 classification rate: 0.498536585366
    i: 32 cost: 1906.19779842 classification rate: 0.497073170732
    i: 33 cost: 1905.36657676 classification rate: 0.497804878049
    i: 34 cost: 1905.17870739 classification rate: 0.50756097561
    i: 35 cost: 1904.33455963 classification rate: 0.505609756098
    i: 36 cost: 1903.84301247 classification rate: 0.523414634146
    i: 37 cost: 1904.53957539 classification rate: 0.499756097561
    i: 38 cost: 1904.42271089 classification rate: 0.502926829268
    i: 39 cost: 1903.34831287 classification rate: 0.494634146341
    i: 40 cost: 1903.02703064 classification rate: 0.507804878049
    i: 41 cost: 1902.88483985 classification rate: 0.508536585366
    i: 42 cost: 1903.03647753 classification rate: 0.488536585366
    i: 43 cost: 1902.70323171 classification rate: 0.502926829268
    i: 44 cost: 1902.36993357 classification rate: 0.502926829268
    i: 45 cost: 1901.59730639 classification rate: 0.51512195122
    i: 46 cost: 1901.93303006 classification rate: 0.499268292683
    i: 47 cost: 1901.43963514 classification rate: 0.501463414634
    i: 48 cost: 1900.95558127 classification rate: 0.512195121951
    i: 49 cost: 1900.31232911 classification rate: 0.50756097561
    i: 50 cost: 1899.41456064 classification rate: 0.492926829268
    i: 51 cost: 1813.80081038 classification rate: 0.515365853659
    i: 52 cost: 1773.48565467 classification rate: 0.51756097561
    i: 53 cost: 1769.50573099 classification rate: 0.524146341463
    i: 54 cost: 1765.49557108 classification rate: 0.521707317073
    i: 55 cost: 1758.59610737 classification rate: 0.52487804878
    i: 56 cost: 688.580182232 classification rate: 0.811707317073
    i: 57 cost: 21.7266883587 classification rate: 1.0



<div class="imgcap">
<img src="./assets/rnn/output_7_1.png">
<div class="thecap">Flask Server up and running</div>
</div>
![png](../assets/rnn/output_7_1.png)


Once, the rnn unit has learned the function we are trying to learn. We try some arbitrary examples just to see what parity is assigns to the sequence. As we can see below, the rnn unit is able to correctly tell the parity as we go from left to right in the sequence. 


```python
rnn.predict_op(np.array([1, 0, 1, 0, 1], dtype=np.float32).reshape(-1, 1))
```




    array([1, 1, 0, 0, 1])




```python

```


```python

```

##### GRU


```python
class GRU:
    def __init__(self, M):
        self.M = M # hidden layer size

    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1] 
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f = activation

        Wxr = init_weight(D, M)
        Whr = init_weight(M, M)
        br  = np.zeros(M)
        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        bz  = np.zeros(M)
        Wxh = init_weight(D, M)
        Whh = init_weight(M, M)
        bh  = np.zeros(M)
        h0  = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        # theano vars
        self.Wxr = theano.shared(Wxr)
        self.Whr = theano.shared(Whr)
        self.br  = theano.shared(br)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz  = theano.shared(bz)
        self.Wxh = theano.shared(Wxh)
        self.Whh = theano.shared(Whh)
        self.bh  = theano.shared(bh)
        self.h0  = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wxr, self.Whr, self.br, self.Wxz, self.Whz, self.bz, self.Wxh, self.Whh, self.bh, 
                       self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            r_t = T.nnet.nnet.sigmoid(x_t.dot(self.Wxr) + h_t1.dot(self.Whr) + self.br)
            z_t = T.nnet.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
            hhat_t = self.f(x_t.dot(self.Wxh) + (r_t * h_t1).dot(self.Whh) + self.bh)
            h_t = ( 1 - z_t) * h_t1 + z_t * hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, y],
            updates=updates
        )

        costs = []
        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in xrange(N):
                c, p, rout = self.train_op(X[j], Y[j])
                cost += c
                if p[-1] == Y[j,-1]:
                    n_correct += 1
            print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
            costs.append(cost)
            if n_correct == N:
                break

        if show_fig:
            plt.plot(costs)
            plt.show()


def parity_gru(B=12, learning_rate=10e-5, epochs=200):
    X, Y = all_parity_pairs_with_sequence_labels(B)

    gru = GRU(4)
    gru.fit(X, Y, learning_rate=learning_rate, epochs=epochs, activation=T.tanh, show_fig=True)  
    return gru
```


```python
gru = parity_gru()
```

    i: 0 cost: 2838.82693421 classification rate: 0.494390243902
    i: 1 cost: 2829.40506202 classification rate: 0.500487804878
    i: 2 cost: 2823.16730698 classification rate: 0.501463414634
    i: 3 cost: 2813.97392561 classification rate: 0.502926829268
    i: 4 cost: 2793.02280711 classification rate: 0.499268292683
    i: 5 cost: 2738.52835083 classification rate: 0.497804878049
    i: 6 cost: 2629.55853745 classification rate: 0.504146341463
    i: 7 cost: 1984.42380188 classification rate: 0.614146341463
    i: 8 cost: 112.742301358 classification rate: 0.999024390244
    i: 9 cost: 35.1580380061 classification rate: 1.0



![png](../assets/rnn/output_14_1.png)



```python
gru.predict_op(np.array([1, 0, 1, 0, 1], dtype=np.float32).reshape(-1, 1))
```




    array([1, 1, 0, 0, 1])




```python

```


```python

```


```python

```

##### LSTM


```python
class LSTM:
    def __init__(self, M):
        self.M = M # hidden layer size

    def fit(self, X, Y, learning_rate=10e-1, mu=0.99, reg=1.0, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1] 
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f = activation

        # weights for i_t
        Wxi = init_weight(D, M)
        Whi = init_weight(M, M)
        Wci = init_weight(M, M)
        bi  = np.zeros(M)
        
        # weights for f_t 
        Wxf = init_weight(D, M)
        Whf = init_weight(M, M)
        Wcf = init_weight(M, M)
        bf = np.zeros(M)
        
        # weights for c_t
        Wxc = init_weight(D, M)
        Whc = init_weight(M, M)
        bc = np.zeros(M)
        
        # weights for o_t 
        Wxo = init_weight(D, M)
        Who = init_weight(M, M)
        Wco = init_weight(M, M)
        bo = np.zeros(M)
        
        # Initial values 
        c0 = np.zeros(M)
        h0 = np.zeros(M)
        
        Woo = init_weight(M, K)
        boo = np.zeros(K)
        
        # weights for i_t
        self.Wxi = theano.shared(Wxi)
        self.Whi = theano.shared(Whi)
        self.Wci = theano.shared(Wci)
        self.bi = theano.shared(bi)
        
        # weights for f_t
        self.Wxf = theano.shared(Wxf)
        self.Whf = theano.shared(Whf)
        self.Wcf = theano.shared(Wcf)
        self.bf = theano.shared(bf)
        
        # weights for c_t 
        self.Wxc = theano.shared(Wxc)
        self.Whc = theano.shared(Whc)
        self.bc = theano.shared(bc)
        
        # weights for o_t
        self.Wxo = theano.shared(Wxo)
        self.Who = theano.shared(Who)
        self.Wco = theano.shared(Wco)
        self.bo = theano.shared(bo)
        
        # Initial weights 
        self.c0 = theano.shared(c0)
        self.h0 = theano.shared(h0)
        
        self.Woo = theano.shared(Woo)
        self.boo = theano.shared(boo)
        
        self.params = [self.Wxi, self.Whi, self.Wci, self.bi, \
                       self.Wxf, self.Whf, self.Wcf, self.bf, \
                       self.Wxc, self.Whc, self.bc, \
                       self.Wxo, self.Who, self.Wco, self.bo, \
                       self.c0, self.h0, self.Woo, self.boo]
        
        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1, c_t1):
            i_t = T.nnet.sigmoid(x_t.dot(self.Wxi) + h_t1.dot(self.Whi) + c_t1.dot(self.Wci) + self.bi)
            f_t = T.nnet.sigmoid(x_t.dot(self.Wxf) + h_t1.dot(self.Whf) + c_t1.dot(self.Wcf) + self.bf)
            c_t = f_t * c_t1 + i_t * T.tanh(x_t.dot(self.Wxc) + h_t1.dot(self.Whc) + self.bc)
            o_t = T.nnet.sigmoid(x_t.dot(self.Wxo) + h_t1.dot(self.Who) + c_t.dot(self.Wco) + self.bo)
            h_t = o_t * T.tanh(c_t)
            y_t = T.nnet.softmax(h_t.dot(self.Woo) + self.boo)
            return h_t, c_t, y_t
    
        
        [h, c, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, self.c0, None],
            sequences=thX,
            n_steps=thX.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, y],
            updates=updates
        )

        costs = []
        for i in xrange(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in xrange(N):
                c, p, rout = self.train_op(X[j], Y[j])
                cost += c
                if p[-1] == Y[j,-1]:
                    n_correct += 1
            print "i:", i, "cost:", cost, "classification rate:", (float(n_correct)/N)
            costs.append(cost)
            if n_correct == N:
                break

        if show_fig:
            plt.plot(costs)
            plt.show()


def parity_lstm(B=12, learning_rate=10e-5, epochs=200):
    X, Y = all_parity_pairs_with_sequence_labels(B)

    lstm = LSTM(4)
    lstm.fit(X, Y, learning_rate=learning_rate, epochs=epochs, activation=T.tanh, show_fig=True)  
    return lstm
```


```python
lstm = parity_lstm()
```

    i: 0 cost: 2846.20684849 classification rate: 0.508048780488
    i: 1 cost: 2839.93866702 classification rate: 0.490975609756
    i: 2 cost: 2833.70456889 classification rate: 0.49512195122
    i: 3 cost: 2826.48753154 classification rate: 0.498292682927
    i: 4 cost: 2814.91195755 classification rate: 0.503170731707
    i: 5 cost: 2795.00510416 classification rate: 0.498780487805
    i: 6 cost: 2749.02066143 classification rate: 0.499024390244
    i: 7 cost: 2640.5532041 classification rate: 0.5
    i: 8 cost: 2511.1313085 classification rate: 0.494146341463
    i: 9 cost: 2433.50354199 classification rate: 0.495853658537
    i: 10 cost: 2378.09531217 classification rate: 0.506829268293
    i: 11 cost: 2303.49945155 classification rate: 0.5
    i: 12 cost: 2161.94503824 classification rate: 0.497804878049
    i: 13 cost: 1149.92452173 classification rate: 0.738536585366
    i: 14 cost: 129.421600924 classification rate: 1.0



![png](../assets/rnn/output_21_1.png)



```python
lstm.predict_op(np.array([1, 0, 1, 0, 1], dtype=np.float32).reshape(-1, 1))
```




    array([1, 1, 0, 0, 1])


