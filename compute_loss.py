import numpy as np
import theano
import theano.tensor as T

x = T.matrix('x_sample', dtype='float32')
m = T.max(x, axis=1)
out = m + T.log(T.sum(T.exp(x-m.dimshuffle(0,'x')), axis=1))

f = theano.function([x], [out])

def compute_pdf():
    x = T.matrix('x_sample', dtype='float32')
    mu = T.matrix('mu', dtype='float32')
    sigma = T.matrix('sigma', dtype='float32')
    log_p_reverse = -0.5 * T.sum((T.log(2 * np.pi) + T.log(sigma) + (x - mu) ** 2 / (sigma)),[1])
    log_p_reverse_ = -0.5 * T.sum((T.log(2 * np.pi) + 2 * T.log(sigma) + (x - mu) ** 2 / (2 * sigma**2)),[1])
    #log_p_reverse_ = -0.5 * T.sum((T.abs_(x - mu)),[1])
    f = theano.function([x, mu, sigma], [log_p_reverse, log_p_reverse_])
    return f

f = compute_pdf()
#x_sampled = np.random.normal(0.0, 10.0, size=(100, 784)).clip(0.0, 1.0)
#x_sampled = x_sampled.reshape(100, 784)#n_colors * spatial_width  * spatial_width)

#x_sampled = np.load('batch_3000_corrupted_epoch_30_time_step_29.npz')
x_sampled = x_sampled['X']
x_data = np.asarray(x_sampled).astype('float32')



means_ = np.load('mnist_example.npz')
means_ = means_['arr_0']
means = means_.mean(axis = 0)
means = np.asarray(means).astype('float32')
means = np.array([means,]*100)

vars_ = means_.var(axis = 0)
vars2_ = np.ones([100, 784])
vars_ = np.asarray(vars_).astype('float32')
vars2_ = np.asarray(vars2_).astype('float32')



#vars_ = np.ones([100, 784])
#vars_ = vars_ * 2.0

vars_ = np.array([vars_,]*100)
x_data = x_data.reshape((100, 784))
vars_ = vars_.reshape((100, 784))
vars_ = vars_ + vars2_
vars_ = np.asarray(vars_).astype('float32')
means = means.reshape((100, 784))
log_, log__ = f(x_data, means_, vars_)

'''
experiment_id = '/data/lisatmp3/anirudhg/mnist_walk_back/walkback_-170216T205222'
for batch_index in range(200, 7600, 200):
    qpath = np.load(experiment_id + '/q_path_' + str(batch_index) + '.npz')
    ppath = np.load(experiment_id + '/q_path_' + str(batch_index) +  '_0.npz')

    qsum = np.sum(qpath['X'], axis = 0)
    psum = np.sum(ppath['X'], axis = 0)

    diff = np.sum(psum - qsum, axis = 0)
    diff_ = np.asarray(diff).astype('float32')
    diff_ = diff_.reshape((1,100))
    out = f(diff_)
    print batch_index, diff_.mean()
'''
import ipdb
ipdb.set_trace()
