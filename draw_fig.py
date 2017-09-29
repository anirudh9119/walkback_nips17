import numpy as np
extra_ = np.zeros((2000, 3, 64, 64))
#from viz import plot_images
i = 30
count = 0
while count < 200 :
    qw = np.load('/data/lisatmp3/anirudhg/celebA_walkback/walkback_-170513T225650/batch_index_100_inference_epoch_1_step_' + str(i) + '.npz')
    #qw['X'][0:10, :,:,:].shape
    extra_[count*10:count*10+10, :,:,:] = qw['X'][10:20,:,:,:]
    i = i + 50
    count = count + 1
#plot_images(extra_, 'orig')

print i
np.savez('qw.npz', extra_)
