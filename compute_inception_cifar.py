import argparse
import numpy as np
import os
import sys
import ipdb
import mimir
#import time
import theano
import theano.tensor as T
import optimizers
from collections import OrderedDict
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Flatten#, ScaleAndShift
from fuel.datasets.toy import Spiral
from util import norm_weight, _p, itemlist,  load_params, create_log_dir #unzip,  save_params
from viz import plot_images
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import (gpu_contiguous, gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI
from ConvLayer import ConvOutput, ConvLayer
rng = RandomStreams(12345)
sys.setrecursionlimit(10000000)

class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()
use_conv = True
INPUT_SIZE = 32*32*3
WIDTH=32
N_COLORS=3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=500, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--resume_file', default=None, type=str,
                        help='Name of saved model to continue training')
    parser.add_argument('--suffix', default='', type=str,
                        help='Optional descriptive suffix for model')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Output directory to store trained models')
    parser.add_argument('--ext-every-n', type=int, default=25,
                        help='Evaluate training extensions every N epochs')
    parser.add_argument('--model-args', type=str, default='',
                        help='Dictionary string to be eval()d containing model arguments.')
    parser.add_argument('--dropout_rate', type=float, default=0.,
                        help='Rate to use for dropout during training+testing.')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Name of dataset to use.')
    parser.add_argument('--plot_before_training', type=bool, default=False,
                        help='Save diagnostic plots at epoch 0, before any training.')
    parser.add_argument('--num_steps', type=int, default=2,
                        help='Number of transition steps.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Standard deviation of the diffusion process.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha factor')
    parser.add_argument('--dims', default=[4096], type=int,
                        nargs='+')
    parser.add_argument('--noise_prob', default=0.1, type=float,
                        help='probability for bernouli distribution of adding noise of 1 to each input')
    parser.add_argument('--avg', default=0, type=float)
    parser.add_argument('--std', default=1., type=float)
    parser.add_argument('--noise', default='gaussian', choices=['gaussian', 'binomial'])
    parser.add_argument('--reload_', type=bool, default = False,
                        help='Reloading the parameters')
    parser.add_argument('--saveto_filename', type = str, default = None,
                        help='directory where parameters are stored')
    parser.add_argument('--extra_steps', type = int, default = 0,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--meta_steps', type = int, default = 10,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use!!')
    parser.add_argument('--temperature_factor', type = float, default = 2.0,
                        help='How much temperature must be scaled')

    args = parser.parse_args()

    model_args = eval('dict(' + args.model_args + ')')
    print model_args


    if not os.path.exists(args.output_dir):
        raise IOError("Output directory '%s' does not exist. "%args.output_dir)
    return args, model_args



def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
        print kk
    return tparams


layers = {'ff': ('param_init_fflayer', 'fflayer')}

def get_layer(name):
        fns = layers[name]
        return (eval(fns[0]), eval(fns[1]))

def deconv(X, w, subsample=(1, 1), border_mode=(0, 0), conv_mode='conv'):
     img = gpu_contiguous(X)
     kerns = gpu_contiguous(w)
     desc = GpuDnnConvDesc(border_mode=border_mode, subsample=subsample, conv_mode=conv_mode)(gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0],    img.shape[3]*subsample[1]).shape, kerns.shape)
     out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*subsample[0], img.shape[3]*subsample[1])
     d_img = GpuDnnConvGradI()(kerns, img, out, desc)
     return d_img

import numpy.random as np_rng
def param_init_convlayer(options, params, prefix='ff', nin=None, nout=None, kernel_len=5, ortho=True, batch_norm=False):
    params[_p(prefix, 'W')] = 0.01 * np_rng.normal(size=(nout, nin, kernel_len, kernel_len)).astype('float32')
    params[_p(prefix, 'b')]= np.zeros(shape=(nout,)).astype('float32')
    return params

def convlayer(tparams, state_below, options, index,
             prefix='rconv',
             activ='lambda x: tensor.tanh(x)',
             stride=None,trans_weights=False):

     kernel_shape = tparams[prefix+"_W"].get_value().shape[2]
     if kernel_shape == 5:
         if stride == 2 or stride == -2:
             padsize = 2
         else:
             padsize = 2
     elif kernel_shape == 1:
         padsize = 0
     else:
         raise Exception(kernel_shape)
     weights = tparams[prefix+'_W']
     if trans_weights:
         weights = weights.transpose(1,0,2,3)
     if stride == -2:
         conv_out = deconv(state_below,weights.transpose(1,0,2,3),subsample=(2,2), border_mode=(2,2))
     else:
         conv_out = dnn.dnn_conv(img=state_below,kerns=weights,subsample=(stride, stride),border_mode=padsize,precision='float32')
     conv_out = conv_out + tparams[prefix+'_b'].dimshuffle('x', 0, 'x', 'x')
     if prefix+"_newmu" in tparams:
         batch_norm = True
         #print "using batch norm for prefix", prefix
     else:
         batch_norm = False
     if batch_norm:
         conv_out = (conv_out - T.mean(conv_out, axis=(0,2,3), keepdims=True)) / (0.01 + T.std(conv_out, axis=(0,2,3), keepdims=True))
         conv_out = conv_out*tparams[prefix+'_newsigma'][index].dimshuffle('x',0,'x','x') + tparams[prefix+'_newmu'][index].dimshuffle('x',0,'x','x')
     conv_out = eval(activ)(conv_out)
     return conv_out


def init_params(options):
    params = OrderedDict()
    if use_conv:
        bn=True
        params = ConvLayer(3,64,5,2,params=params, prefix='conv_1',bn=bn)
        params = ConvLayer(64,128,5,2,params=params, prefix='conv_2',bn=bn)
        params = ConvLayer(128,256,5,2,params=params,prefix='conv_3',bn=bn)

        '''
        params = get_layer('ff')[0](options, params, prefix='layer_1',nin=4*4*256, nout=2048,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_2',nin=2048, nout=2048,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_3',nin=2048, nout=2048,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_4',nin=2048, nout=2048,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_5',nin=2048, nout=4*4*256,ortho=False)
        '''

        '''

        params = param_init_convlayer(options, params, prefix='conv_1', nin=3, nout=64, kernel_len=5, batch_norm=bn)
        params[_p('conv_1', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 64)).astype('float32')
        params[_p('conv_1', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 64)).astype('float32')

        params = param_init_convlayer(options, params, prefix='conv_2', nin=64, nout=128, kernel_len=5, batch_norm=bn)
        params[_p('conv_2', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 128)).astype('float32')
        params[_p('conv_2', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 128)).astype('float32')

        params = param_init_convlayer(options, params, prefix='conv_3', nin=128, nout=256, kernel_len=5, batch_norm=bn)
        params[_p('conv_3', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 256)).astype('float32')
        params[_p('conv_3', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 256)).astype('float32')
        '''

        params = get_layer('ff')[0](options, params, prefix='layer_1', prefix_bnorm='layer_1_step_0', nin=4*4*256, nout=2048, ortho=False, batch_norm=True)
        params = get_layer('ff')[0](options, params, prefix='layer_2', prefix_bnorm='layer_2_step_0', nin=2048, nout=2048, ortho=False,batch_norm=True)
        params = get_layer('ff')[0](options, params, prefix='layer_3', prefix_bnorm='layer_3_step_0', nin=2048, nout=2048, ortho=False,batch_norm=True)
        params = get_layer('ff')[0](options, params, prefix='layer_4', prefix_bnorm='layer_4_step_0', nin=2048, nout=2048, ortho=False,batch_norm=True)
        params = get_layer('ff')[0](options, params, prefix='layer_5', prefix_bnorm='layer_5_step_0', nin=2048, nout=4*4*256, ortho=False,batch_norm=True)

        params[_p('layer1_bnorm', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps , 2048)).astype('float32')
        params[_p('layer1_bnorm', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 2048)).astype('float32')

        params[_p('layer2_bnorm', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps , 2048)).astype('float32')
        params[_p('layer2_bnorm', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 2048)).astype('float32')

        params[_p('layer3_bnorm', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps , 2048)).astype('float32')
        params[_p('layer3_bnorm', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 2048)).astype('float32')

        params[_p('layer4_bnorm', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps , 2048)).astype('float32')
        params[_p('layer4_bnorm', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 2048)).astype('float32')

        params[_p('layer5_bnorm', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps , 4*4*256)).astype('float32')
        params[_p('layer5_bnorm', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 4*4*256)).astype('float32')

        params = ConvLayer(256,128,5,-2,params=params,prefix='conv_4_mu',bn=bn)
        params = ConvLayer(128,64,5,-2,params=params,prefix='conv_5_mu',bn=bn)
        params = ConvLayer(64,3,5,-2,params=params,prefix='conv_6_mu')

        params = ConvLayer(256,128,5,-2,params=params,prefix='conv_4_s',bn=bn)
        params = ConvLayer(128,64,5,-2,params=params,prefix='conv_5_s',bn=bn)
        params = ConvLayer(64,3,5,-2,params=params,prefix='conv_6_s')

        '''
        params = param_init_convlayer(options, params, prefix='conv_4_mu', nin=256, nout=128, kernel_len=5, batch_norm=bn)
        params[_p('conv_4_mu', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 128)).astype('float32')
        params[_p('conv_4_mu', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 128)).astype('float32')

        params = param_init_convlayer(options, params, prefix='conv_5_mu', nin=128, nout=64, kernel_len=5, batch_norm=bn)
        params[_p('conv_5_mu', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 64)).astype('float32')
        params[_p('conv_5_mu', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 64)).astype('float32')

        params = param_init_convlayer(options, params, prefix='conv_6_mu', nin=64, nout=3, kernel_len=5, batch_norm =False)

        params = param_init_convlayer(options, params, prefix='conv_4_s', nin=256, nout=128, kernel_len=5, batch_norm=bn)
        params[_p('conv_4_s', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 128)).astype('float32')
        params[_p('conv_4_s', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 128)).astype('float32')

        params = param_init_convlayer(options, params, prefix='conv_5_s', nin=128, nout=64, kernel_len=5, batch_norm=bn)
        params[_p('conv_5_s', 'newmu')] = np.zeros(shape=(args.num_steps *args.meta_steps, 64)).astype('float32')
        params[_p('conv_5_s', 'newsigma')] = np.ones(shape=(args.num_steps *args.meta_steps, 64)).astype('float32')

        params = param_init_convlayer(options, params, prefix='conv_6_s', nin=64, nout=3, kernel_len=5, batch_norm = False)
        '''

    return params

# P(next s | previous s) as a gaussian with mean = (1-alpha)*previous_s + alpha * F(previous_s) + sigma(previous_s)*Gaussian_noise(0,1)
# where we learn the functions F and sigma (e.g. as MLPs), with sigma>0 by construction.

def join(a, b=None):
    if b==None:
        return a
    else:
        return T.concatenate([a,b],axis=1)

def ln(inp):
    return (inp - T.mean(inp,axis=1,keepdims=True)) / (0.001 + T.std(inp,axis=1,keepdims=True))

def param_init_fflayer(options, params, prefix='ff', prefix_bnorm='bnorm', nin=None, nout=None, ortho=True, batch_norm=False):

    if prefix in params:
        print 'this layer is already present'
    else:
        params[_p(prefix, 'W')] = norm_weight(nin, nout)
        params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams,
            state_below,
            options,
            index,
            prefix='rconv',
            prefix_bnorm='bnorm',
            activ='lambda x: tensor.tanh(x)',
            batch_norm = False,
            **kwargs):
    preactivation = T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]

    if batch_norm:
        preactivation = (preactivation - preactivation.mean(axis=0)) / (0.0001 + preactivation.std(axis=0))
        preactivation = (tparams[_p(prefix_bnorm, 'newmu')][index] + preactivation* tparams[_p(prefix_bnorm, 'newsigma')][index])

    return preactivation


#from lib.ops import batchnorm#, dropout
def transition_operator(tparams, options, x, temperature, num_step):

    xr = x.reshape((args.batch_size, 3, 32, 32))

    bn=True
    c1 = ConvOutput(xr, tparams, prefix="conv_1", kernel_len=5, stride=2, activation='relu',bn=bn)
    c2 = ConvOutput(c1, tparams, prefix="conv_2", kernel_len=5, stride=2, activation='relu',bn=bn)
    c3 = ConvOutput(c2, tparams, prefix="conv_3", kernel_len=5, stride=2, activation='relu',bn=bn)

    '''
    h1 = T.nnet.relu(batchnorm(fflayer(tparams, c3.flatten(2), options, prefix='layer_1')), alpha = 0.02)
    h2 = T.nnet.relu(batchnorm(fflayer(tparams, h1, options, prefix='layer_2')), alpha = 0.02)
    h3 = T.nnet.relu(batchnorm(fflayer(tparams, h2, options, prefix='layer_3')), alpha = 0.02)
    h4 = T.nnet.relu(batchnorm(fflayer(tparams, h3, options, prefix='layer_4')), alpha = 0.02)
    h5 = T.nnet.relu(batchnorm(fflayer(tparams, h4, options, prefix='layer_5')), alpha = 0.02)
    hL = h5.reshape((args.batch_size,256, 4, 4))

    '''
    h1 = T.nnet.relu((fflayer(tparams, c3.flatten(2),  options, index = num_step, prefix='layer_1', prefix_bnorm='layer1_bnorm', batch_norm=True)))
    h2 = T.nnet.relu((fflayer(tparams, h1,  options, index = num_step, prefix='layer_2', prefix_bnorm='layer2_bnorm', batch_norm=True)))
    h3 = T.nnet.relu((fflayer(tparams, h2,  options, index = num_step, prefix='layer_3', prefix_bnorm='layer3_bnorm', batch_norm=True)))
    h4 = T.nnet.relu((fflayer(tparams, h3,  options, index = num_step, prefix='layer_4', prefix_bnorm='layer4_bnorm', batch_norm=True)))
    h5 = T.nnet.relu((fflayer(tparams, h4,  options, index = num_step, prefix='layer_5', prefix_bnorm='layer5_bnorm', batch_norm=True)))
    hL = h5.reshape((args.batch_size,256, 4, 4))

    c4_mu = ConvOutput(join(hL), tparams, prefix="conv_4_mu", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c5_mu = ConvOutput(join(c4_mu), tparams, prefix="conv_5_mu", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c6_mu = ConvOutput(join(c5_mu), tparams, prefix="conv_6_mu", kernel_len=5, stride=-2, activation=None)

    c4_s = ConvOutput(join(hL), tparams, prefix="conv_4_s", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c5_s = ConvOutput(join(c4_s), tparams, prefix="conv_5_s", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c6_s = ConvOutput(join(c5_s), tparams, prefix="conv_6_s", kernel_len=5, stride=-2, activation=None)
    '''
    c1 = convlayer(tparams, xr, options, prefix='conv_1', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=2)
    c2 = convlayer(tparams, c1, options, prefix='conv_2', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=2)
    c3 = convlayer(tparams, c2, options, prefix='conv_3', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=2)
    '''


    '''
    c4_mu = convlayer(tparams, join(hL), options, prefix='conv_4_mu', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=-2)
    c5_mu = convlayer(tparams, join(c4_mu), options, prefix='conv_5_mu', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=-2)
    c6_mu = convlayer(tparams, join(c5_mu), options, prefix='conv_6_mu', index = num_step, activ='lambda x: x', stride=-2)

    c4_s = convlayer(tparams, join(hL), options, prefix='conv_4_s', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=-2)
    c5_s = convlayer(tparams, join(c4_s), options, prefix='conv_5_s', index = num_step, activ='lambda x: T.nnet.relu(x)', stride=-2)
    c6_s = convlayer(tparams, join(c5_s), options, prefix='conv_6_s', index = num_step, activ='lambda x: x', stride=-2)
    '''

    cL_mu = c6_mu
    cL_s = c6_s

    mu = cL_mu.flatten(2)
    sigma = T.nnet.softplus(cL_s.flatten(2))

    sigma *= temperature

    sigma = T.sqrt(sigma)

    epsilon = rng.normal(size=(args.batch_size, INPUT_SIZE), avg=args.avg, std=args.std, dtype=theano.config.floatX)
    epsilon = epsilon + 0.0 * num_step * epsilon

    x_hat = consider_constant(x*0.5 + 0.5 * (mu + 0.00001 * sigma * epsilon)).clip(0.0,1.0)
    log_p_reverse = -0.5 * T.sum(T.sqr(x - mu)  + 0.0 * num_step * sigma,[1])
    #log_p_reverse = -0.5 * T.sum(T.abs_(x - mu) + 0.0 * (T.log(2 * np.pi) + 2 * T.log(sigma) + (x - mu) ** 2 / (2 * sigma ** 2)),[1])


    return x_hat, log_p_reverse, T.mean(sigma), T.mean(sigma), T.mean(sigma)



def reverse_time(scl, shft, sample_drawn, name):
    new_image = np.asarray(sample_drawn).astype('float32').reshape(args.batch_size, N_COLORS, WIDTH, WIDTH)
    plot_images(new_image, name)

def sample(tparams, options):
    #batch_size = 32
    x_data = T.matrix('x_sample', dtype='float32')
    temperature = T.scalar('temperature_sample', dtype='float32')
    num_step = T.scalar('num_step', dtype='int32')
    x_tilde, _, sampled, sampled_activation, sampled_preactivation = transition_operator(tparams, options, x_data, temperature, num_step)
    f = theano.function([x_data, temperature, num_step], [x_tilde, sampled, sampled_activation, sampled_preactivation])
    return f

def compute_loss(x, options, tparams, start_temperature, num_step):
     temperature = start_temperature
     x_tilde, log_p_reverse, _, _, _ = transition_operator(tparams, options, x, temperature, num_step)
     log_p_reverse_list = [log_p_reverse]
     print args.num_steps
     loss = -T.mean(sum(log_p_reverse_list, 0.0))
     return loss

def one_step_diffusion(x, options, tparams, temperature, num_step):
    x_tilde, log_p_reverse, sampled, sampled_activation, sampled_preactivation = transition_operator(tparams, options, x, temperature, num_step)
    forward_diffusion =  theano.function([x, temperature, num_step], [x_tilde, sampled, sampled_activation, sampled_preactivation])
    return forward_diffusion

def build_model(tparams, model_options):
    x = T.matrix('x', dtype='float32')
    start_temperature = T.scalar('start_temperature', dtype='float32')
    num_step = T.scalar('num_step', dtype='int32')
    loss = compute_loss(x, model_options, tparams, start_temperature, num_step)
    return x, loss, start_temperature, num_step

def train(args,
          model_args):

    #model_id = '/data/lisatmp4/lambalex/lsun_walkback/walkback_'

    model_id = '/data/lisatmp4/anirudhg/cifar_walk_back/walkback_'
    model_dir = create_log_dir(args, model_id)
    model_id2 =  'logs/walkback_'
    model_dir2 = create_log_dir(args, model_id2)
    print model_dir
    print model_dir2 + '/' + 'log.jsonl.gz'
    logger = mimir.Logger(filename=model_dir2  + '/log.jsonl.gz', formatter=None)

    # TODO batches_per_epoch should not be hard coded
    lrate = args.lr
    import sys
    sys.setrecursionlimit(10000000)
    args, model_args = parse_args()

    #trng = RandomStreams(1234)

    if args.resume_file is not None:
        print "Resuming training from " + args.resume_file
        from blocks.scripts import continue_training
        continue_training(args.resume_file)

    ## load the training data
    if args.dataset == 'MNIST':
        print 'loading MNIST'
        from fuel.datasets import MNIST
        dataset_train = MNIST(['train'], sources=('features',))
        dataset_test = MNIST(['test'], sources=('features',))
        n_colors = 1
        spatial_width = 28

    elif args.dataset == 'CIFAR10':
        from fuel.datasets import CIFAR10
        dataset_train = CIFAR10(['train'], sources=('features',))
        dataset_test = CIFAR10(['test'], sources=('features',))
        n_colors = 3
        spatial_width = 32

    elif args.dataset == "lsun" or args.dataset == "lsunsmall":

        print "loading lsun class!"

        from load_lsun import load_lsun

        print "loading lsun data!"

        if args.dataset == "lsunsmall":
            dataset_train, dataset_test = load_lsun(args.batch_size, downsample=True)
            spatial_width=32
        else:
            dataset_train, dataset_test = load_lsun(args.batch_size, downsample=False)
            spatial_width=64

        n_colors = 3


    elif args.dataset == "celeba":

        print "loading celeba data"

        from fuel.datasets.celeba import CelebA

        dataset_train = CelebA(which_sets = ['train'], which_format="64", sources=('features',), load_in_memory=False)
        dataset_test = CelebA(which_sets = ['test'], which_format="64", sources=('features',), load_in_memory=False)

        spatial_width = 64
        n_colors = 3

        tr_scheme = SequentialScheme(examples=dataset_train.num_examples, batch_size=args.batch_size)
        ts_scheme = SequentialScheme(examples=dataset_test.num_examples, batch_size=args.batch_size)

        train_stream = DataStream.default_stream(dataset_train, iteration_scheme = tr_scheme)
        test_stream = DataStream.default_stream(dataset_test, iteration_scheme = ts_scheme)

        dataset_train = train_stream
        dataset_test = test_stream

        #epoch_it = train_stream.get_epoch_iterator()

    elif args.dataset == 'Spiral':
        print 'loading SPIRAL'
        train_set = Spiral(num_examples=100000, classes=1, cycles=2., noise=0.01,
                           sources=('features',))
        dataset_train = DataStream.default_stream(train_set,
                            iteration_scheme=ShuffledScheme(
                            train_set.num_examples, args.batch_size))

    else:
        raise ValueError("Unknown dataset %s."%args.dataset)

    model_options = locals().copy()

    if args.dataset != 'lsun' and args.dataset != 'celeba':
        train_stream = Flatten(DataStream.default_stream(dataset_train,
                              iteration_scheme=ShuffledScheme(
                                  examples=dataset_train.num_examples - (dataset_train.num_examples%args.batch_size),
                                  batch_size=args.batch_size)))
    else:
        train_stream = dataset_train
        test_stream = dataset_test

    print "Width", WIDTH, spatial_width

    shp = next(train_stream.get_epoch_iterator())[0].shape

    print "got epoch iterator"

    Xbatch = next(train_stream.get_epoch_iterator())[0]
    scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
    shft = -np.mean(Xbatch*scl)

    print 'Building model'
    params = init_params(model_options)
    if args.reload_:
        print "Trying to reload parameters"
        if os.path.exists(args.saveto_filename):
            print 'Reloading Parameters'
            print args.saveto_filename
            params = load_params(args.saveto_filename, params)
    tparams = init_tparams(params)
    print tparams
    x, cost, start_temperature, step_chain = build_model(tparams, model_options)
    inps = [x.astype('float32'), start_temperature, step_chain]

    x_Data = T.matrix('x_Data', dtype='float32')
    temperature  = T.scalar('temperature', dtype='float32')
    step_chain_part  = T.scalar('step_chain_part', dtype='int32')

    forward_diffusion = one_step_diffusion(x_Data, model_options, tparams, temperature, step_chain_part)

    print tparams
    grads = T.grad(cost, wrt=itemlist(tparams))

    #get_grads = theano.function(inps, grads)

    for j in range(0, len(grads)):
        grads[j] = T.switch(T.isnan(grads[j]), T.zeros_like(grads[j]), grads[j])


    # compile the optimizer, the actual computational graph is compiled here
    lr = T.scalar(name='lr')
    print 'Building optimizers...',
    optimizer = args.optimizer

    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    #for param in tparams:
    #    print param
    #    print tparams[param].get_value().shape

    print 'Buiding Sampler....'
    f_sample = sample(tparams, model_options)
    print 'Done'

    uidx = 0
    estop = False
    bad_counter = 0
    max_epochs = 4000
    batch_index = 1
    print  'Number of steps....'
    print args.num_steps
    print "Number of metasteps...."
    print args.meta_steps
    print 'Done'
    count_sample = 1
    save_data = []
    for eidx in xrange(1):
        print 'Starting Next Epoch ', eidx
        for data_ in range(500):#train_stream.get_epoch_iterator():
            if args.noise == "gaussian":
                x_sampled = np.random.normal(0.5, 2.0, size=(args.batch_size,INPUT_SIZE)).clip(0.0, 1.0)
            else:
                s = np.random.binomial(1, 0.5, INPUT_SIZE)

            temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps - 1))
            x_data = np.asarray(x_sampled).astype('float32')
            for i in range(args.num_steps*args.meta_steps + args.extra_steps):
                x_data,  sampled, sampled_activation, sampled_preactivation = f_sample(x_data.astype('float32'), temperature, args.num_steps*args.meta_steps -i - 1)
                print 'On step number, using temperature', i, temperature
                #reverse_time(scl, shft, x_data, model_dir + '/batch_index_' + str(batch_index) + '_inference_' + 'epoch_' + str(count_sample) + '_step_' + str(i))
                x_data = np.asarray(x_data).astype('float32')
                x_data = x_data.reshape(args.batch_size, INPUT_SIZE)
                if temperature == args.temperature:
                    temperature = temperature
                else:
                    temperature /= args.temperature_factor

            count_sample  = count_sample + 1
            save_data.append(x_data)
            fname = model_dir + '/batch_index_' + str(batch_index) + '_inference_' + 'epoch_' + str(count_sample) + '_step_' + str(i)
            np.savez(fname + '.npz', x_data)

        save2_data = np.asarray(save_data).astype('float32')
        fname = model_dir + '/generted_images_50000' #+ args.saveto_filename
        np.savez(fname + '.npz', save_data)


    ipdb.set_trace()

if __name__ == '__main__':
    args, model_args = parse_args()
    train(args, model_args)
    pass
