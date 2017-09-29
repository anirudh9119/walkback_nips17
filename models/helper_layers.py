import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from lasagne.layers import MergeLayer, Layer
from lasagne.layers.merge import autocrop, autocrop_array_shapes
from lasagne.layers.pool import Upscale2DLayer
import theano.tensor as T
from lasagne.utils import as_tuple
import lasagne
from lasagne.random import get_rng

from models.model_helpers import softmax4D

import numpy as np

class CroppingLayer(MergeLayer):
    """
    This layer performs an elementwise merge of its input layers.
    It requires all input layers to have the same output shape.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal
    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.
    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`
    See Also
    --------
    ElemwiseSumLayer : Shortcut for sum layer.
    """

    def __init__(self, incomings, merge_function, cropping=None, **kwargs):
        super(CroppingLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function
        self.cropping = cropping

    def get_output_shape_for(self, input_shapes):
        input_shapes = autocrop_array_shapes(input_shapes, self.cropping)
        # Infer the output shape by grabbing, for each axis, the first
        # input size that is not `None` (if there is any)
        output_shape = input_shapes[1]

        return output_shape

    def get_output_for(self, inputs, **kwargs):
        inputs = autocrop(inputs, self.cropping)
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output

class DePool2D(Upscale2DLayer):
    ''' Code taken from:
    https://github.com/nanopony/keras-convautoencoder/blob/master/autoencoder_layers.py
    Simplar to Upscale2DLayer, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
    '''

    def __init__(self, incoming, scale_factor, pool2d_layer,
            pool2d_layer_in, **kwargs):
        super(Upscale2DLayer, self).__init__(incoming, **kwargs)
        self.scale_factor = as_tuple(scale_factor, 2)
        self.pool2d_layer = pool2d_layer
        self.pool2d_layer_in = pool2d_layer_in
        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_for(self, upscaled, **kwargs):
        a, b = self.scale_factor
        # get output for pooling and pre-pooling layer
        inp, out =\
                lasagne.layers.get_output([self.pool2d_layer_in,
                                           self.pool2d_layer])
        # upscale the input feature map by scale_factor
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
        # get the shapes for pre-pooling layer and upscaled layer
        sh_pool2d_in = T.shape(inp)
        sh_upscaled = T.shape(upscaled)
        # in case the shape is different left-bottom-pad with zero
        tmp = T.zeros(sh_pool2d_in)
        indx = (slice(None),
                slice(None),
                slice(0, sh_upscaled[2]),
                slice(0, sh_upscaled[3]))
        upscaled = T.set_subtensor(tmp[indx], upscaled)
        # get max pool indices
        indices_pool = T.grad(None, wrt=inp,
                known_grads={out: T.ones_like(out)})
        # mask values using indices_pool
        f = indices_pool * upscaled

        return f


class GaussianNoiseLayerSoftmax(Layer):
    """
    Gaussian noise layer. Adapted from Lasagne.
    Add zero-mean Gaussian noise of given standard deviation to the input [1],
    and then normalize with softmax.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
    sigma : float or tensor scalar
            Standard deviation of added Gaussian noise
    Notes
    -----
    The Gaussian noise layer is a regularizer. During training you should set
    deterministic to false and during testing you should set deterministic to
    true.
    References
    ----------
    .. [1] K.-C. Jim, C. Giles, and B. Horne (1996):
           An analysis of noise in recurrent neural networks: convergence and
           generalization.
           IEEE Transactions on Neural Networks, 7(6):1424-1438.
    """
    def __init__(self, incoming, sigma=0.1, **kwargs):
        super(GaussianNoiseLayerSoftmax, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:
            return softmax4D(input + self._srng.normal(input.shape,
                                                       avg=0.0,
                                                       std=self.sigma))


class RGBtoBGRLayer(lasagne.layers.Layer):
    def __init__(self, l_in, bgr_mean=np.array([103.939, 116.779, 123.68]),
                 data_format='bc01', **kwargs):
        """A Layer to normalize and convert images from RGB to BGR
        This layer converts images from RGB to BGR to adapt to Caffe
        that uses OpenCV, which uses BGR. It also subtracts the
        per-pixel mean. From:
        https://github.com/fvisin/reseg/blob/variable_size_images/vgg16.py
        Parameters
        ----------
        l_in : :class:``lasagne.layers.Layer``
            The incoming layer, typically an
            :class:``lasagne.layers.InputLayer``
        bgr_mean : iterable of 3 ints
            The mean of each channel. By default, the ImageNet
            mean values are used.
        data_format : str
            The format of l_in, either `b01c` (batch, rows, cols,
            channels) or `bc01` (batch, channels, rows, cols)
        """
        super(RGBtoBGRLayer, self).__init__(l_in, **kwargs)
        assert data_format in ['bc01', 'b01c']
        self.l_in = l_in
        floatX = theano.config.floatX
        self.bgr_mean = bgr_mean.astype(floatX)
        self.data_format = data_format

    def get_output_for(self, input_im, **kwargs):
        if self.data_format == 'bc01':
            input_im = input_im[:, ::-1, :, :]
            input_im -= self.bgr_mean[:, np.newaxis, np.newaxis]
        else:
            input_im = input_im[:, :, :, ::-1]
            input_im -= self.bgr_mean
        return input_im
