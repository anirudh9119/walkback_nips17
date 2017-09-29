import sys

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)


import warnings
warnings.filterwarnings("ignore")

rng = np.random.RandomState(23455)
# set a fixed number for 2 purpose:
#  1. repeatable experiments; 2. for multiple-GPU, the same initial weights

def layers2params(layers):
    paramLst = []

    for layer in layers:
        paramLst += layer.params


    return paramLst


#precision = 'float16'
precision = 'float32'

print "USING CONVOLUTIONAL PRECISION", precision

def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
    img = gpu_contiguous(X)
    kerns = gpu_contiguous(w)
    desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode, precision=precision)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1]).shape, kerns.shape)
    out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
    d_img = GpuDnnConvGradI()(kerns, img, out, desc)
    return d_img



class Weight(object):

    def __init__(self, w_shape, mean=0, std=1.0):

        super(Weight, self).__init__()

        print "conv layer using std of", std, "and mean of", mean, "with shape", w_shape

        if std != 0:

            self.np_values = np.asarray(
               rng.normal(mean, std, w_shape), dtype=theano.config.floatX).astype('float32')

        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX)).astype('float32')

        self.val = self.np_values

def ConvLayer(in_channels, out_channels, kernel_len, stride, params, prefix,bn = False):

    if True:

        bias_init = 0.0

        std = 0.02

        if stride >= 1:
            filter_shape = np.asarray((in_channels, kernel_len, kernel_len, out_channels))
            W = Weight(filter_shape, std = std)
            b = Weight(filter_shape[3], bias_init, std=0)
        else:
            filter_shape = np.asarray((in_channels, out_channels, kernel_len, kernel_len))
            W = Weight(filter_shape, std = std)
            b = Weight(filter_shape[1], bias_init, std=0)

        if bn:
            bn_mean = np.zeros(shape = (1,out_channels,1,1)).astype('float32')
            bn_std = np.random.normal(1.0, 0.001, size = (1,out_channels,1,1)).astype('float32')

    params[prefix + "_W"] = W.val
    params[prefix + "_b"] = b.val


    if bn:
        params[prefix + "_mean"] = bn_mean
        params[prefix + "_std"] = bn_std

    return params

def ConvOutput(input, params, prefix, kernel_len, stride,activation,bn=False):

    W = params[prefix + "_W"]
    b = params[prefix + "_b"]

    if bn:
        bn_mean = params[prefix + "_mean"]
        bn_std = params[prefix + "_std"]

    if kernel_len == 1:
        padsize = 0
    elif kernel_len == 3:
        padsize = 1
    elif kernel_len == 5:
        padsize = 2
    elif kernel_len == 7:
        padsize = 3
    elif kernel_len == 11:
        padsize = 5
    else:
        raise Exception()

    if True:

        if stride >= 1:
            W_shuffled = W.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        else:
            W_shuffled = W

        if stride >= 1:
            conv_out = dnn.dnn_conv(img=input,
                                        kerns=W_shuffled,
                                        subsample=(stride, stride),
                                        border_mode=padsize, precision = precision)
        elif stride == -2:
            conv_out = deconv(input, W_shuffled, subsample=(2, 2), border_mode=(2,2))
        else:
            raise Exception("DONE")

        conv_out = conv_out + T.sum(b.dimshuffle('x', 0, 'x', 'x'))

        if bn:
            conv_out = (conv_out - T.mean(conv_out, axis = (0,2,3), keepdims = True)) / (0.01 + T.std(conv_out, axis=(0,2,3), keepdims = True))
            conv_out = conv_out * T.addbroadcast(bn_std,0,2,3) + T.addbroadcast(bn_mean, 0,2,3)

        if activation == "relu":
            out = T.maximum(0.0, conv_out)
        elif activation == 'lrelu':
            out = T.nnet.relu(conv_out, alpha = 0.02)
        elif activation == "tanh":
            out = T.tanh(conv_out)
        elif activation == 'sigmoid':
            out = T.nnet.sigmoid(conv_out)
        elif activation == None:
            out = conv_out


        return out



if __name__ == "__main__":

    x = T.tensor4()

    randData = np.random.normal(size = (32,3,32,32)).astype('float32')

    c1 = ConvLayer(3,64,5,2, prefix='conv_1')
    c2 = ConvLayer(64,128,5,2, prefix='conv_2')
    c3 = ConvLayer(128,256,5,2, prefix='conv_3')


    y = c1.output(x)

    f = theano.function(inputs = [x], outputs = {'y' : y})

    #print f(randData)['g']
    out = f(randData)


    print (randData**2).sum()
    print (out['c1']**2).sum()
    print (out['c2']**2).sum()
    print (out['c3']**2).sum()
    print (out['c4']**2).sum()
    print (out['c5']**2).sum()

    for element in sorted(out.keys()):
        print element, out[element].shape
