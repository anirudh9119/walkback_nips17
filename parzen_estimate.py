#!/usr/bin/env python
"""
This file was copyed from pyleran2.distributions.parzen.py
Their license is BSD clause-3: https://github.com/lisa-lab/pylearn2/
"""
import numpy
import ipdb
import theano
T = theano.tensor


def log_mean_exp(a):
    """
    We need the log-likelihood, this calculates the logarithm
    of a Parzen window
    """
    max_ = a.max(1)
    return max_ + T.log(T.exp(a - max_.dimshuffle(0, 'x')).mean(1))


def make_lpdf(mu, sigma):
    """
    Makes a Theano function that allows the evalution of a Parzen windows
    estimator (aka kernel density estimator) where the Kernel is a normal
    distribution with stddev sigma and with points at mu.
    Parameters
    -----------
    mu : numpy matrix
        Contains the data points over which this distribution is based.
    sigma : scalar
        The standard deviation of the normal distribution around each data \
        point.
    Returns
    -------
    lpdf : callable
        Estimator of the log of the probability density under a point.
    """
    x = T.matrix()
    mu = theano.shared(mu)
    a = (x.dimshuffle(0, 'x', 1) - mu.dimshuffle('x', 0, 1)) / sigma
    E = log_mean_exp(-0.5*(a**2).sum(2))
    Z = mu.shape[1] * T.log(sigma * numpy.sqrt(numpy.pi * 2))
    return theano.function([x], E - Z)


class ParzenWindows(object):
    """
    Parzen Window estimation and log-likelihood calculator.
    This is usually used to test generative models as follows:
    1 - Get 10k samples from the generative model
    2 - Contruct a ParzenWindows object with the samples from 1
    3 - Test the log-likelihood on the test set
    Parameters
    ----------
    samples : numpy matrix
        See description for make_lpdf
    sigma : scalar
        See description for make_lpdf
    """
    def __init__(self, samples, sigma):
        # just keeping these for debugging/examination, not needed
        self._samples = samples
        self._sigma = sigma

        self.lpdf = make_lpdf(samples, sigma)

    def get_ll(self, x, batch_size=10):
        """
        Evaluates the log likelihood of a set of datapoints with respect to the
        probability distribution.
        Parameters
        ----------
        x : numpy matrix
            The set of points for which you want to evaluate the log \
            likelihood.
        """
        inds = range(x.shape[0])
        n_batches = int(numpy.ceil(float(len(inds)) / batch_size))

        lls = []
        for i in range(n_batches):
            lls.extend(self.lpdf(x[inds[i::n_batches]]))

        return numpy.array(lls).mean()


from fuel.datasets import MNIST
dataset_train = MNIST(['train'], sources=('features',))
dataset_test = MNIST(['test'], sources=('features',))
n_colors = 1
spatial_width = 28
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten
import numpy as np
test_stream = Flatten(DataStream.default_stream(dataset_test,
                        iteration_scheme=ShuffledScheme(
                        examples=dataset_test.num_examples,
                        batch_size=10000)))

train_stream = Flatten(DataStream.default_stream(dataset_train,
                        iteration_scheme=ShuffledScheme(
                        examples=dataset_train.num_examples,
                        batch_size=100)))
'''
i =0
qw = np.load('gen_samples_170217T154923.npz')
qw = qw['arr_0']
'''

'''
qw = np.load('/data/lisatmp3/anirudhg/mnist_walk_back/walkback_-170217T154923/batch_index_1_inference_means_epoch_1_step_39.npz')
qw = qw['X']
qw = qw.reshape((100, 784))

for i in range(99):
    qw2 = np.load('/data/lisatmp3/anirudhg/mnist_walk_back/walkback_-170217T154923/batch_index_' + str(i+2) + '_inference_means_epoch_1_step_39.npz')
    qw2 = qw2['X']
    qw2 = qw2.reshape((100, 784))
    qw = np.concatenate([qw, qw2])


np.savez('gen_samples_means_170217T154923_model_3000.npz', qw)

qw = np.load('gen_samples_means_170217T154923_model_3000.npz')
qw = qw['arr_0']
qw = qw.reshape((10000, 784))
'''
for data in train_stream.get_epoch_iterator():
    test_data = data[0]
    ipdb.set_trace()

'''
var = 0.01
for i in range(98):
    pwin = ParzenWindows(qw, var)
    ll  = pwin.get_ll(test_data)
    print ll, var
    var += 0.01
'''
