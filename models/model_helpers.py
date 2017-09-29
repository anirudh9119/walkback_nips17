import theano
import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import ConcatLayer


def freezeParameters(net, single=True):
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].remove('trainable')
            except KeyError:
                pass


def unfreezeParameters(net, single=True):
    all_layers = lasagne.layers.get_all_layers(net)

    if single:
        all_layers = [all_layers[-1]]

    for layer in all_layers:
        layer_params = layer.get_params()
        for p in layer_params:
            try:
                layer.params[p].add('trainable')
            except KeyError:
                pass


def softmax4D(x):
    """
    Softmax activation function for a 4D tensor of shape (b, c, 0, 1)
    """
    # Compute softmax activation
    stable_x = x - theano.gradient.zero_grad(x.max(1, keepdims=True))
    exp_x = stable_x.exp()
    softmax_x = exp_x / exp_x.sum(1)[:, None, :, :]

    return softmax_x


def concatenate(net, in1, concat_layers, concat_vars, pos):

    if pos < len(concat_layers) and concat_layers[pos] == 'input':
        concat_layers[pos] = in1

    if in1 in concat_layers:
        net[in1 + '_h'] = InputLayer((None, net[in1].input_shape[1] if
                                      (concat_layers[pos] != 'noisy_input' and
                                      concat_layers[pos] != 'input')
                                      else 3, None, None), concat_vars[pos])
        net[in1 + '_concat'] = ConcatLayer((net[in1 + '_h'],
                                            net[in1]), axis=1, cropping=None)
        pos += 1
        out = in1 + '_concat'

        laySize = net[out].output_shape
        n_cl = laySize[1]
        print('Number of feature maps (concat):', n_cl)
    else:
        out = in1

    if concat_layers and pos <= len(concat_layers) and concat_layers[pos-1] == 'noisy_input':
        concat_layers[pos-1] = 'input'

    return pos, out
