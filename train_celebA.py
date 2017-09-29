'''
iterate over examples
   clamp v=x
   (if there is an h, do feedforward inference)
   iterate over temperatures, starting at 1 and doubling each time
      sample s(t+1) from P(next s | previous s = s(t)) at the current temperatore
      update parameters to maximize P(next s = s(t) | previous s = s(t+1))
One thing I did not mention is that you should be able to measure the variational bound while you train,
both on the training set and test set. This is simply the average (over the training or test examples)
and over samples in the diffusion stage of P(previous state | next state), where the first in the series
has the chosen example, times the probability of the last state under the "global" Gaussian
(whose mean and variance can be estimated by measuring them on the last stage).
Thus you can track training quality along the way. The P( ) I mentioned in the previous e-mail is different
each time because of the temperature change.
There should also be a way to estimate the true NLL, using importance sampling, but it's more expensive.
Basically you use not just the P but also the Q, as follows, by sampling a large number (say K) diffusion
paths for EACH x (on which you want to estimate P(x)):
NLL_estimator(x) = log mean_{trajectories started at x ~ Q(states | x)} (P(x|states) P(states)) / Q(states | x)
where the numerator is like the one used for estimating the variational bound, and the numerator is
the probability of the trajectory path that was sampled.
This estimator is slightly conservative (in average it gives a slightly worse likelihood than the true one, but the bias goes
to 0 as K increases).
'''

import argparse
import numpy as np
import os
#import warnings

import mimir
import time


import theano
import theano.tensor as T
#from theano.tensor.shared_randomstreams import RandomStreams

from collections import OrderedDict
#from blocks.algorithms import (RMSProp, GradientDescent, CompositeRule, RemoveNotFinite)
#from blocks.extensions import FinishAfter, Timing, Printing
#from blocks.extensions.monitoring import (DataStreamMonitoring, TrainingDataMonitoring)
#from blocks.extensions.saveload import Checkpoint
#from blocks.extensions.training import SharedVariableModifier
#from blocks.filter import VariableFilter
#from blocks.graph import ComputationGraph, apply_dropout
#from blocks.main_loop import MainLoop
#import blocks.model
#from blocks.roles import INPUT#, PARAMETER

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.transformers import Flatten#, ScaleAndShift
from fuel.datasets.toy import Spiral
import optimizers
#import extensions
#import model
from util import  unzip, norm_weight, _p, itemlist,  load_params, create_log_dir,  save_params  #ortho_weight
#import ipdb
from viz import plot_images
import sys

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
rng = RandomStreams(12345)

from ConvLayer import ConvLayer, ConvOutput

sys.setrecursionlimit(10000000)



#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from theano.tensor.opt import register_canonicalize


class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()
#register_canonicalize(theano.gof.OpRemove(consider_constant), name='remove_consider_constant')

#INPUT_SIZE = 28*28
#WIDTH=28
#N_COLORS=1

#INPUT_SIZE = 784
#WIDTH=28
#N_COLORS=1

#NUM_HIDDEN = 4096

INPUT_SIZE = 64*64*3
WIDTH=64
N_COLORS=3
use_conv = True

#INPUT_SIZE = 32*32*3
#WIDTH=32
#N_COLORS=3

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
    parser.add_argument('--dataset', type=str, default='MNIST',
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
    parser.add_argument('--extra_steps', type = int, default = 10,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--meta_steps', type = int, default = 10,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use!!')
    parser.add_argument('--temperature_factor', type = float, default = 2.0,
                        help='How much temperature must be scaled')
    parser.add_argument('--sigma', type = float, default = 0.00001,
                        help='How much Noise should be added at step 1')

    args = parser.parse_args()

    model_args = eval('dict(' + args.model_args + ')')
    print model_args


    if not os.path.exists(args.output_dir):
        raise IOError("Output directory '%s' does not exist. "%args.output_dir)
    return args, model_args


def param_init_fflayer(options, params, prefix='ff',
                       nin=None, nout=None, ortho=True, flag=False):

    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


layers = {'ff': ('param_init_fflayer', 'fflayer')}

def get_layer(name):
        fns = layers[name]
        return (eval(fns[0]), eval(fns[1]))


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return T.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]


def init_params(options):

    params = OrderedDict()

    if not use_conv:

        params = get_layer('ff')[0](options, params, prefix='layer_1',
                                nin=INPUT_SIZE, nout=args.dims[0],
                                ortho=False)

        params = get_layer('ff')[0](options, params, prefix='layer_2',
                                nin=args.dims[0], nout=args.dims[0],
                                ortho=False)

    if use_conv and args.dataset == "celeba":

        bn=True
        params = ConvLayer(3,64,5,2,params=params, prefix='conv_1',bn=bn)
        params = ConvLayer(64,128,5,2,params=params, prefix='conv_2',bn=bn)
        params = ConvLayer(128,256,5,2,params=params,prefix='conv_3',bn=bn)
        '''
        params = get_layer('ff')[0](options, params, prefix='layer_1',nin=8*8*256, nout=1024,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_2_mu',nin=1024, nout=256,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_2_sigma',nin=1024, nout=256,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_3',nin=256, nout=8*8*256,ortho=False)
        '''

        params = get_layer('ff')[0](options, params, prefix='layer_1',nin=8*8*256, nout=1024,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_2',nin=1024, nout=1024,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_3',nin=1024, nout=1024,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_4',nin=1024, nout=1024,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_5',nin=1024, nout=8*8*256,ortho=False)

        params = ConvLayer(256,128,5,-2,params=params,prefix='conv_4_mu',bn=bn)
        params = ConvLayer(128,64,5,-2,params=params,prefix='conv_5_mu',bn=bn)
        params = ConvLayer(64,3,5,-2,params=params,prefix='conv_6_mu')

        params = ConvLayer(256,128,5,-2,params=params,prefix='conv_4_s',bn=bn)
        params = ConvLayer(128,64,5,-2,params=params,prefix='conv_5_s',bn=bn)
        params = ConvLayer(64,3,5,-2,params=params,prefix='conv_6_s')

    elif use_conv and args.dataset == "MNIST":

        bn=True
        params = ConvLayer(1,128,5,2,params=params, prefix='conv_1',bn=bn)
        params = ConvLayer(128,256,5,2,params=params, prefix='conv_2',bn=bn)
        params = get_layer('ff')[0](options, params, prefix='layer_1',nin=7*7*256, nout=1024,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_2',nin=1024, nout=256,ortho=False)
        params = get_layer('ff')[0](options, params, prefix='layer_3',nin=256, nout=7*7*256,ortho=False)

        params = ConvLayer(256,128,5,-2,params=params,prefix='conv_4_mu',bn=bn)
        params = ConvLayer(128,1,5,-2,params=params,prefix='conv_5_mu')

    else:


        #TODO: Ideally, only in the output layer, flag=True should be set.
        if len(args.dims) == 1:
            params = get_layer('ff')[0](options, params, prefix='mu_0',
                                nin=args.dims[0], nout=INPUT_SIZE,
                                ortho=False, flag=True)
            if args.noise == 'gaussian':
                params = get_layer('ff')[0](options, params, prefix='sigma_0',
                                        nin=args.dims[0], nout=INPUT_SIZE,
                                        ortho=False)


        for i in range(len(args.dims)-1):
                params = get_layer('ff')[0](options, params, prefix ='mu_'+str(i),
                                    nin=args.dims[i], nout=args.dims[i+1],
                                    ortho=False)
                if args.noise == 'gaussian':
                    params = get_layer('ff')[0](options, params, prefix='sigma_'+str(i),
                                    nin=args.dims[i], nout=args.dims[i+1],
                                    ortho=False, flag=True )


        if len(args.dims) > 1:
            params = get_layer('ff')[0](options, params, prefix='mu_'+str(i+1),
                                    nin=args.dims[i+1], nout=INPUT_SIZE,
                                    ortho=False, flag=True)

            if args.noise == 'gaussian':
                params = get_layer('ff')[0](options, params, prefix='sigma_'+str(i+1),
                                    nin=args.dims[i+1], nout=INPUT_SIZE,
                                    ortho=False)
    return params

# P(next s | previous s) as a gaussian with mean = (1-alpha)*previous_s + alpha * F(previous_s) + sigma(previous_s)*Gaussian_noise(0,1)
# where we learn the functions F and sigma (e.g. as MLPs), with sigma>0 by construction.

def join(a,b=None):
    if b==None:
        return a
    else:
        return T.concatenate([a,b],axis=1)


from lib.ops import batchnorm
def transition_operator(tparams, options, x, temperature):

    xr = x.reshape((args.batch_size,3,64,64))
    bn=True

    c1 = ConvOutput(xr, tparams, prefix="conv_1", kernel_len=5, stride=2, activation='relu',bn=bn)
    c2 = ConvOutput(c1, tparams, prefix="conv_2", kernel_len=5, stride=2, activation='relu',bn=bn)
    c3 = ConvOutput(c2, tparams, prefix="conv_3", kernel_len=5, stride=2, activation='relu',bn=bn)

    h1 = T.nnet.relu(batchnorm(fflayer(tparams, c3.flatten(2), options,prefix='layer_1')), alpha = 0.02)
    h2 = T.nnet.relu(batchnorm(fflayer(tparams, h1, options,prefix='layer_2')), alpha = 0.02)
    h3 = T.nnet.relu(batchnorm(fflayer(tparams, h2, options, prefix='layer_3')), alpha = 0.02)
    h4 = T.nnet.relu(batchnorm(fflayer(tparams, h3, options, prefix='layer_4')), alpha = 0.02)
    h5 = T.nnet.relu(batchnorm(fflayer(tparams, h4, options, prefix='layer_5')), alpha = 0.02)

    hL = h5.reshape((args.batch_size,256,8,8))

    c4_mu = ConvOutput(join(hL), tparams, prefix="conv_4_mu", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c5_mu = ConvOutput(join(c4_mu), tparams, prefix="conv_5_mu", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c6_mu = ConvOutput(join(c5_mu), tparams, prefix="conv_6_mu", kernel_len=5, stride=-2, activation=None)

    c4_s = ConvOutput(join(hL), tparams, prefix="conv_4_s", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c5_s = ConvOutput(join(c4_s), tparams, prefix="conv_5_s", kernel_len=5, stride=-2, activation='relu',bn=bn)
    c6_s = ConvOutput(join(c5_s), tparams, prefix="conv_6_s", kernel_len=5, stride=-2, activation=None)

    cL_mu = c6_mu
    cL_s = c6_s

    '''
    mu = cL_mu.flatten(2)
    sigma = T.nnet.softplus(cL_s.flatten(2))
    sigma = args.sigma * sigma * T.sqrt(temperature)
    epsilon = rng.normal(size=(args.batch_size, INPUT_SIZE), avg=args.avg, std=args.std, dtype=theano.config.floatX)
    #x_hat = consider_constant((args.alpha)*x + (1-args.alpha) * (mu) +  sigma * epsilon).clip(0.0,1.0)
    x_hat = consider_constant(x*0.5 + 0.5 * (mu + 0.00001 * sigma * epsilon)).clip(0.0,1.0)
    mean_ = ((0.5)*x + (0.5) * (mu)).clip(0.0,1.0)
    log_p_reverse = -0.5 * T.sum(1.0 * (T.log(2 * np.pi) + T.log(sigma) + (x - mu) ** 2 / (sigma)),[1])
    return x_hat, log_p_reverse, sigma, mu, mean_

    This part is working!
    '''
    mu = cL_mu.flatten(2)
    sigma = T.nnet.softplus(cL_s.flatten(2))
    sigma *= temperature
    sigma = T.sqrt(sigma)
    epsilon = rng.normal(size=(args.batch_size, INPUT_SIZE), avg=args.avg, std=args.std, dtype=theano.config.floatX)
    x_hat = consider_constant(x*0.5 + 0.5 * (mu + 0.00001 * sigma * epsilon)).clip(0.0,1.0)
    log_p_reverse = -0.5 * T.sum(T.sqr(x - mu) + 0.0 * (T.log(2 * np.pi) + 2 * T.log(sigma) + (x - mu) ** 2 / (2 * sigma ** 2)),[1])
    return x_hat, log_p_reverse, T.mean(sigma), T.mean(sigma), T.mean(sigma)

def reverse_time(scl, shft, sample_drawn, name):
    #new_image = ((sample_drawn-shft)/scl)
    #new_image = new_image.reshape(args.batch-size, 1, WIDTH, WIDTH)
    new_image = np.asarray(sample_drawn).astype('float32').reshape(args.batch_size, N_COLORS, WIDTH, WIDTH)
    plot_images(new_image, name)

def sample(tparams, options):
    #batch_size = 32
    x_data = T.matrix('x_sample', dtype='float32')
    temperature = T.scalar('temperature_sample', dtype='float32')
    x_tilde, _, sampled, sampled_activation, sampled_preactivation = transition_operator(tparams, options, x_data, temperature)
    f = theano.function([x_data, temperature], [x_tilde, sampled, sampled_activation, sampled_preactivation])
    return f

def compute_loss(x, options, tparams, start_temperature):
     temperature = start_temperature
     x_tilde, log_p_reverse, _, _, _ = transition_operator(tparams, options, x, temperature)
     states = [x_tilde]
     log_p_reverse_list = [log_p_reverse]
     print args.num_steps
     for _ in range(args.num_steps - 1):
         temperature *= args.temperature_factor
         x_tilde, log_p_reverse, _, _, _ = transition_operator(tparams, options, states[-1], temperature)
         states.append(x_tilde)
         log_p_reverse_list.append(log_p_reverse)
     loss = -T.mean(sum(log_p_reverse_list, 0.0))
     return loss

def one_step_diffusion(x, options, tparams, temperature):
    x_tilde, log_p_reverse, sampled, sampled_activation, sampled_preactivation = transition_operator(tparams, options, x, temperature)
    forward_diffusion =  theano.function([x, temperature], [x_tilde, sampled, sampled_activation, sampled_preactivation])
    return forward_diffusion

def build_model(tparams, model_options):
    x = T.matrix('x', dtype='float32')
    start_temperature = T.scalar('start_temperature', dtype='float32')
    loss = compute_loss(x, model_options, tparams, start_temperature)
    return x, loss, start_temperature

def train(args,
          model_args,
          lrate):

    model_id = '/data/lisatmp4/anirudhg/celebA_walkback/walkback_'
    model_dir = create_log_dir(args, model_id)
    model_id2 =  '../celebA_logs/walkback_'
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
        tr_scheme = SequentialScheme(examples=dataset_train.num_examples - (dataset_train.num_examples % args.batch_size), batch_size=args.batch_size)
        ts_scheme = SequentialScheme(examples=dataset_test.num_examples - (dataset_test.num_examples % args.batch_size), batch_size=args.batch_size)

        train_stream = DataStream.default_stream(dataset_train, iteration_scheme = tr_scheme)
        test_stream = DataStream.default_stream(dataset_test, iteration_scheme = ts_scheme)

        print "using this many train examples", dataset_train.num_examples - (dataset_train.num_examples % args.batch_size)

        dataset_train = train_stream
        dataset_test = test_stream


    elif args.dataset == "celeba":

        print "loading celeba data"

        from fuel.datasets.celeba import CelebA

        dataset_train = CelebA(which_sets = ['train'], which_format="64", sources=('features',), load_in_memory=False)
        dataset_test = CelebA(which_sets = ['test'], which_format="64", sources=('features',), load_in_memory=False)

        spatial_width = 64
        n_colors = 3

        tr_scheme = SequentialScheme(examples=dataset_train.num_examples - (dataset_train.num_examples % args.batch_size), batch_size=args.batch_size)
        ts_scheme = SequentialScheme(examples=dataset_test.num_examples - (dataset_test.num_examples % args.batch_size), batch_size=args.batch_size)

        train_stream = DataStream.default_stream(dataset_train, iteration_scheme = tr_scheme)
        test_stream = DataStream.default_stream(dataset_test, iteration_scheme = ts_scheme)

        print "using this many train examples", dataset_train.num_examples - (dataset_train.num_examples % args.batch_size)

        dataset_train = test_stream
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
                                  examples=dataset_train.num_examples - (dataset_train.num_examples % args.batch_size),
                                  batch_size=args.batch_size)))
    else:
        train_stream = dataset_train
        test_stream = dataset_test

    print "Width", WIDTH, spatial_width

    shp = next(train_stream.get_epoch_iterator())[0].shape

    print "got epoch iterator"

    # make the training data 0 mean and variance 1
    # TODO compute mean and variance on full dataset, not minibatch
    Xbatch = next(train_stream.get_epoch_iterator())[0]
    scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
    shft = -np.mean(Xbatch*scl)
    # scale is applied before shift
    #train_stream = ScaleAndShift(train_stream, scl, shft)
    #test_stream = ScaleAndShift(test_stream, scl, shft)

    print 'Building model'
    params = init_params(model_options)
    if args.reload_:
        print "Trying to reload parameters"
        if os.path.exists(args.saveto_filename):
            print 'Reloading Parameters'
            print args.saveto_filename
            params = load_params(args.saveto_filename, params)
    tparams = init_tparams(params)

    '''
    x = T.matrix('x', dtype='float32')
    f=transition_operator(tparams, model_options, x, 1)
    for data in train_stream.get_epoch_iterator():
        print data[0]
        a = f(data[0])
        print a
        ipdb.set_trace()
    '''
    x, cost, start_temperature = build_model(tparams, model_options)
    inps = [x,start_temperature]

    x_Data = T.matrix('x_Data', dtype='float32')
    temperature  = T.scalar('temperature', dtype='float32')
    forward_diffusion = one_step_diffusion(x_Data, model_options, tparams, temperature)

    print 'Building f_cost...',
    f_cost = theano.function(inps, cost)
    print 'Done'
    print tparams
    grads = T.grad(cost, wrt=itemlist(tparams))

    get_grads = theano.function(inps, grads)

    for j in range(0, len(grads)):
        grads[j] = T.switch(T.isnan(grads[j]), T.zeros_like(grads[j]), grads[j])


    # compile the optimizer, the actual computational graph is compiled here
    lr = T.scalar(name='lr')
    print 'Building optimizers...',
    optimizer = args.optimizer

    f_grad_shared, f_update = getattr(optimizers, optimizer)(lr, tparams, grads, inps, cost)
    print 'Done'

    for param in tparams:
        print param
        print tparams[param].get_value().shape

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
    for eidx in xrange(max_epochs):
        n_samples = 0
        print 'Starting Next Epoch ', eidx
        for data in train_stream.get_epoch_iterator():

            if args.dataset == 'celeba':
                data_use = (data[0].reshape(args.batch_size,3*64*64),)
            if args.dataset == "MNIST":
                data_use = (data[0].reshape(args.batch_size,1*28*28),)

            t0 = time.time()
            batch_index += 1
            n_samples += len(data_use[0])
            uidx += 1
            if batch_index%500==0:
                params = unzip(tparams)
                save_params(params, model_dir + '/' + 'params_batch_index' + str(batch_index) + '.npz')
            if data_use[0] is None:
                print 'No data '
                uidx -= 1
                continue

            data_run = data_use[0]

            temperature_forward = args.temperature
            meta_cost = []
            for meta_step in range(0, args.meta_steps):

                meta_cost.append(f_grad_shared(data_run, temperature_forward))

                f_update(lrate)

                if args.meta_steps > 1:
                    data_run, sigma, _, _ = forward_diffusion(data_run, temperature_forward)

                    temperature_forward *= args.temperature_factor

            cost = sum(meta_cost) / len(meta_cost)

            gradient_updates_ = get_grads(data_use[0],args.temperature)

            if np.isnan(cost) or np.isinf(cost):
                print 'NaN detected'
                return 1.
            logger.log({'epoch': eidx,
                        'batch_index': batch_index,
                        'uidx': uidx,
                        'training_error': cost,
                        'Norm_1': np.linalg.norm(gradient_updates_[0]),
                        'Norm_2': np.linalg.norm(gradient_updates_[1]),
                        'Norm_3': np.linalg.norm(gradient_updates_[2]),
                        'Norm_4': np.linalg.norm(gradient_updates_[3])})

            if batch_index%20==0:
                print batch_index, "cost", cost
            if batch_index%100==0:
                '''
                count_sample += 1
                temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps -1 ))
                temperature_forward = args.temperature

                for num_step in range(args.num_steps * args.meta_steps):
                    print "Forward temperature", temperature_forward
                    if num_step == 0:
                        x_data, sampled, sampled_activation, sampled_preactivation = forward_diffusion(data_use[0], temperature_forward)
                        x_data = np.asarray(x_data).astype('float32').reshape(args.batch_size, INPUT_SIZE)
                        x_temp = x_data.reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                        plot_images(x_temp, model_dir + '/' + "batch_" + str(batch_index) + '_corrupted_' + 'epoch_' + str(count_sample) + '_time_step_' + str(num_step))

                    else:
                        x_data, sampled, sampled_activation, sampled_preactivation = forward_diffusion(x_data, temperature_forward)
                        x_data = np.asarray(x_data).astype('float32').reshape(args.batch_size, INPUT_SIZE)
                        x_temp = x_data.reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                        plot_images(x_temp, model_dir + '/batch_' + str(batch_index) + '_corrupted_' + 'epoch_' + str(count_sample) + '_time_step_' + str(num_step))


                    temperature_forward = temperature_forward * args.temperature_factor;

                print "PLOTTING ORIGINAL IMAGE"
                x_temp2 = data_use[0].reshape(args.batch_size, n_colors, WIDTH, WIDTH)
                plot_images(x_temp2, model_dir + '/' + 'orig_' + 'epoch_' + str(count_sample) + '_batch_index_' +  str(batch_index))

                print "DONE PLOTTING ORIGINAL IMAGE"


                temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps - 1 ))

                for i in range(args.num_steps*args.meta_steps + args.extra_steps):
                    x_data, sampled, sampled_activation, sampled_preactivation  = f_sample(x_data, temperature)
                    print 'On backward step number, using temperature', i, temperature
                    reverse_time(scl, shft, x_data, model_dir + '/'+ "batch_" + str(batch_index) + '_samples_backward_' + 'epoch_' + str(count_sample) + '_time_step_' + str(i))
                    x_data = np.asarray(x_data).astype('float32')
                    x_data = x_data.reshape(args.batch_size, INPUT_SIZE)
                    if temperature == args.temperature:
                        temperature = temperature
                    else:
                        temperature /= args.temperature_factor

                '''
                if args.noise == "gaussian":
                    x_sampled = np.random.normal(0.5, 2.0, size=(args.batch_size,INPUT_SIZE)).clip(0.0, 1.0)
                else:
                    x_sampled = np.random.binomial(1, 0.5, size=(args.batch_size,INPUT_SIZE))

                temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps - 1))

                x_data = np.asarray(x_sampled).astype('float32')
                for i in range(args.num_steps*args.meta_steps + args.extra_steps):
                    x_data,  sampled, sampled_activation, sampled_preactivation = f_sample(x_data, temperature)
                    print 'On step number, using temperature', i, temperature
                    reverse_time(scl, shft, x_data, model_dir + '/batch_index_' + str(batch_index) + '_inference_' + 'epoch_' + str(count_sample) + '_step_' + str(i))
                    x_data = np.asarray(x_data).astype('float32')
                    x_data = x_data.reshape(args.batch_size, INPUT_SIZE)
                    if temperature == args.temperature:
                        temperature = temperature
                    else:
                        temperature /= args.temperature_factor
                '''
                qw = data_use[0]
                qw2 = qw.reshape((100, 3 * 64 * 64))
                x_sampled = np.random.normal(0.5, 2.0, size=(100,  64*64*3)).clip(0.0, 1.0)
                temperature = args.temperature * (args.temperature_factor ** (args.num_steps*args.meta_steps - 1))

                qw2[:,3*64*32 :] = x_sampled[:, 3*64*32:]
                #qw2 = qw2.reshape((100, 3 ,64 ,64))

                x_data = qw2
                for i in range(args.num_steps*args.meta_steps + args.extra_steps ):
                     x_data,  sampled, sampled_activation, sampled_preactivation = f_sample(x_data, temperature)
                     print 'Impainting using temperature', i, temperature
                     reverse_time(scl, shft, x_data, model_dir + '/batch_index_' + str(batch_index) + '_inpaiting_' + 'epoch_' + str(count_sample) + '_step_' + str(i))
                     x_data = np.asarray(x_data).astype('float32')
                     x_data = x_data.reshape(args.batch_size, INPUT_SIZE)
                     x_data[:, :3 * 64*32] = qw2[:, :3*64*32]

                     if temperature == args.temperature:
                         temperature = temperature
                     else:
                         temperature /= args.temperature_factor
                '''

            #print time.time() - t5, "time for not doing special calls, should be 0"

    import ipdb
    ipdb.set_trace()

if __name__ == '__main__':
    args, model_args = parse_args()
    train(args, model_args, lrate=0.000001)
    pass
