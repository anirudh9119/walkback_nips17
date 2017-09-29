import theano
from theano import tensor

import numpy as np

import six
from util import itemlist


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup)

    updates = []

    t_prev = theano.shared(np.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)


    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))


    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in six.iteritems(tparams)]
    running_up2 = [theano.shared(p.get_value() * np.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in six.iteritems(tparams)]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in six.iteritems(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp,
                                    cost,
                                    updates=zgup + rg2up,
                                    name='adadelta')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)
             ]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr],
                               [],
                               updates=ru2up + param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in six.iteritems(tparams)]
    running_grads = [theano.shared(p.get_value() * np.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in six.iteritems(tparams)]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in six.iteritems(tparams)]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp,
                                    cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop')

    updir = [theano.shared(p.get_value() * np.float32(0.),
                           name='%s_updir' % k)
             for k, p in six.iteritems(tparams)]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(
        itemlist(tparams), updir_new)]
    f_update = theano.function([lr],
                               [],
                               updates=updir_new + param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in six.iteritems(tparams)]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp,
        cost,
        updates=gsup,
        name='sgd')

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    f_update = theano.function([lr], [], updates=pup)

    return f_grad_shared, f_update
