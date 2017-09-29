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
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten#, ScaleAndShift
from fuel.datasets.toy import Spiral
import optimizers
#import extensions
#import model
from util import norm_weight, _p, itemlist,  load_params, create_log_dir  #unzip, save_params  #ortho_weight
import ipdb
from viz import plot_images
import sys
sys.setrecursionlimit(10000000)



#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#from theano.tensor.opt import register_canonicalize


class ConsiderConstant(theano.compile.ViewOp):
    def grad(self, args, g_outs):
        return [T.zeros_like(g_out) for g_out in g_outs]

consider_constant = ConsiderConstant()
#register_canonicalize(theano.gof.OpRemove(consider_constant), name='remove_consider_constant')



INPUT_SIZE = 784
#NUM_HIDDEN = 4096

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=500, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-1, type=float,
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
    parser.add_argument('--temperature', type=float, default=1e-2,
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
    parser.add_argument('--reload_', type=bool, default = True,
                        help='Reloading the parameters')
    parser.add_argument('--saveto_filename', type = str, default = None,
                        help='directory where parameters are stored')
    parser.add_argument('--extra_steps', type = int, default = 0,
                        help='Number of extra steps to sample at temperature 1')
    parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use!!')
    parser.add_argument('--temperature_factor', type = float, default = 2,
                        help='How much temperature must be scaled')

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
    if flag:
        #params[_p(prefix, 'b')] = np.full(nout,-1).astype('float32')
        import gzip
        import pickle
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, _ , _ = pickle.load(f)
            train_x, train_y = train_set
            marginals = np.clip(train_x.mean(axis=0), 1e-7, 1- 1e-7)
            initial_baises = np.log(marginals/(1-marginals))
            params[_p(prefix, 'b')] = initial_baises.astype('float32')

    else:
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
    params = get_layer('ff')[0](options, params, prefix='layer_1',
                                nin=INPUT_SIZE, nout=args.dims[0],
                                ortho=False)

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
                                    ortho=False, flag=True)

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

#kepler3
#kepler3
#bart14
#bart5
#gru 5 kepler2
def transition_operator(tparams, options, x, temperature):
    pre_activation = fflayer(tparams, x, options,
                             prefix='layer_1', activ='linear')
    h = T.tanh(pre_activation)

    for i in range(len(args.dims)):
        if i == 0:
            mu = fflayer(tparams, h, options, prefix='mu_0')
            if args.noise == 'gaussian':
                sigma = fflayer(tparams, h, options, prefix='sigma_0')
        else:
            mu = fflayer(tparams, mu, options, prefix='mu_' + str(i))
            if args.noise == 'gaussian':
                sigma = fflayer(tparams, sigma, options, prefix='sigma_' + str(i))

    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    rng = RandomStreams(12345)

    if args.noise == 'gaussian':
        sigma = T.nnet.softplus(sigma)
        sigma = T.sqrt(temperature) * T.log(1 + T.exp(sigma))
        epsilon = rng.normal(size=(args.batch_size, INPUT_SIZE), avg=args.avg, std=args.std, dtype=theano.config.floatX)
        x_hat = consider_constant((1-args.alpha)*x + (args.alpha)*mu + sigma * epsilon)
        #reverse_transition
        # should take into account the mean of x
        log_p_reverse = -0.5 * T.sum(
                            T.log(2 * np.pi) + 2 * T.log(sigma) +
                            (x - mu) ** 2 / (2 * sigma ** 2),[1])
        # f = theano.function([x],[x_hat, log_p_reverse])
    else:
        # binomial mask is applied
        #mu = T.nnet.sigmoid(mu)
        pre_activat_sampled = (1 - args.alpha) * x + args.alpha * (mu)
        p = T.nnet.sigmoid(pre_activat_sampled/temperature)
        print 'something interesting is going on'
        x_hat = rng.binomial(n = 1, p = p, size = x.shape, dtype='float32')
        log_p_reverse = 1 * (x * T.log(p) + (1 -x) * T.log(1 - p))

    return x_hat, log_p_reverse, pre_activat_sampled, p, pre_activat_sampled/temperature

def viz_forward_trajectory(data, forward_diffusion, scl, shft):
    temperature = args.temperature
    for num_step in range(args.num_steps):
        x_recons, _ = forward_diffusion(data, temperature)
        temperature = temperature * args.temperature_factor
        x_recons = np.asarray(x_recons).astype('float32')
        x_recons = x_recons.reshape(args.batch_size, INPUT_SIZE)
        x_recons = x_recons.reshape(args.batch_size, 1, 28, 28)
        x_recons = ((x_recons-shft)/scl)
        plot_images(x_recons, 'forward_' + str(num_step))

def reverse_time(scl, shft, sample_drawn, name):
    #new_image = ((sample_drawn-shft)/scl)
    #new_image = new_image.reshape(args.batch-size, 1, 28, 28)
    new_image = np.asarray(sample_drawn).astype('float32').reshape(args.batch_size, 1, 28, 28)
    plot_images(new_image, name)

def sample(tparams, options):
    #batch_size = 32
    x_data = T.matrix('x_sample', dtype='float32')
    temperature = T.scalar('temperature_sample', dtype='float32')
    x_tilde, _, sampled, sampled_activation, sampled_preactivation = transition_operator(tparams, options, x_data, temperature)
    f = theano.function([x_data, temperature], [x_tilde, sampled, sampled_activation, sampled_preactivation])
    return f

def compute_loss(x, options, tparams):
     temperature = args.temperature
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
    loss = compute_loss(x, model_options, tparams)
    return x, loss

def inpainting_digit(digit, clamped_image):
    output = np.hstack([digit[:, 0:392], clamped_image[:, 392:784]])
    return output

def change_image(train_X, factor):
    for i in range(500):
        train_X[i, 0, :, :] = np.rot90(train_X[i, 0, :, :], factor)

def do_half_image(train_X, orig_digit):
    train_temp = train_X
    change_image(train_temp.reshape(500, 1,  28, 28), 3)
    train_temp = train_temp.reshape(500, 784)
    output = inpainting_digit(train_temp, orig_digit)
    change_image(output.reshape(500, 1, 28, 28), 1)
    return output

def train(args,
          model_args,
          lrate):

    model_id = '/data/lisatmp4/anirudhg/minst_walk_back/walkback_'
    model_dir = create_log_dir(args, model_id)
    model_id2 =  'walkback_'
    model_dir2 = create_log_dir(args, model_id2)
    print model_dir
    logger = mimir.Logger(filename=model_dir2 + '/' + model_id2 + 'log.jsonl.gz', formatter=None)

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



    train_stream = Flatten(DataStream.default_stream(dataset_train,
                              iteration_scheme=ShuffledScheme(
                                  examples=dataset_train.num_examples,
                                  batch_size=args.batch_size)))


    shp = next(train_stream.get_epoch_iterator())[0].shape
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
    if args.reload_ and os.path.exists(args.saveto_filename):
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
    x, cost = build_model(tparams, model_options)
    inps = [x]

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

    print 'Buiding Sampler....'
    f_sample = sample(tparams, model_options)
    print 'Done'

    uidx = 0
    estop = False
    bad_counter = 0
    max_epochs = 4000
    batch_index = 0
    print  'Number of steps....'
    print args.num_steps
    print 'Done'
    count_sample = 1
    for eidx in xrange(max_epochs):
        n_samples = 0
        print 'Starting Next Epoch ', eidx
        for data in train_stream.get_epoch_iterator():
            batch_index += 1
            n_samples += len(data[0])
            uidx += 1
            if data[0] is None:
                print 'No data '
                uidx -= 1
                continue
            ud_start = time.time()
            cost = f_grad_shared(data[0])
            f_update(lrate)
            ud = time.time() - ud_start


            if batch_index%1==0:
                print 'Cost is this', cost
                count_sample += 1

                from impainting import change_image, inpainting
                train_temp = data[0]
                print data[0].shape
                change_image(train_temp.reshape(args.batch_size, 1,28,28), 3)
                train_temp = train_temp.reshape(args.batch_size, 784)
                output = inpainting(train_temp)
                change_image(output.reshape(args.batch_size,1,28,28), 1)

                reverse_time(scl, shft, output, model_dir + '/' + 'impainting_orig_' + 'epoch_' + str(count_sample) + '_batch_index_' +  str(batch_index))
                x_data = np.asarray(output).astype('float32')
                temperature = args.temperature * (args.temperature_factor ** (args.num_steps -1 ))
                temperature = args.temperature #* (args.temperature_factor ** (args.num_steps -1 ))
                orig_impainted_data = np.asarray(data[0]).astype('float32')

                for i in range(args.num_steps + args.extra_steps + 5):
                    x_data,  sampled, sampled_activation, sampled_preactivation = f_sample(x_data, temperature)
                    print 'Impainting using temperature', i, temperature
                    x_data = do_half_image(x_data, orig_impainted_data)
                    reverse_time(scl, shft, x_data, model_dir + '/' + 'impainting_orig_' + 'epoch_' + str(count_sample) + '_batch_index_' +  str(batch_index) + 'step_' + str(i))
                    x_data = np.asarray(x_data).astype('float32')
                    x_data = x_data.reshape(args.batch_size, INPUT_SIZE)
                    if temperature == args.temperature:
                        temperature = temperature
                    else:
                        temperature = temperature
                        #temperature /= args.temperature_factor
    ipdb.set_trace()

if __name__ == '__main__':
    args, model_args = parse_args()
    train(args, model_args, lrate=0.000001)
    pass
