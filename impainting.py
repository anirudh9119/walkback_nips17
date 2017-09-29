import numpy
import numpy as np
import gzip
INPUT_SIZE=784
import cPickle
cast32      = lambda x : numpy.cast['float32'](x)
from viz import plot_images

def inpainting(digit):
    noise_sampled = np.random.binomial(1, 0.5, [32, 784])
    output = np.hstack([noise_sampled[:, 0:392], digit[:, 392:784]])
    return output

def load_mnist_binary(path):
    f = gzip.open('impainting_mnist.pkl.gz','rb')
    data = cPickle.load(f)
    data = [list(d) for d in data]
    data[0][0] = (data[0][0] > 0.5).astype('float32')
    data[1][0] = (data[1][0] > 0.5).astype('float32')
    data[2][0] = (data[2][0] > 0.5).astype('float32')
    data = tuple([tuple(d) for d in data])
    return data

#(train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y) = load_mnist_binary('.')
#train_X = train_X[0:32, :]
def change_image(train_X, factor):
    for i in range(32):
        train_X[i, 0, :, :] = np.rot90(train_X[i, 0, :, :], factor)

'''
train_temp = train_X
change_image(train_temp.reshape(32,1,28,28), 3)
train_temp = train_temp.reshape(32, 784)
output = inpainting(train_temp)
change_image(output.reshape(32,1,28,28), 1)

print output.shape
plot_images(output.reshape(32,1,28,28),'anirudh')
'''
