# walkback_nips17

This repo contains code for running VW on Cifar, SVHN, CelebA, LSUN, Circle, Spiral , Mixture of Gaussian. Some of these datasets are not mentioned in the paper. 

In all the image experiments, we observed that by having different batchnorm papemeters for different steps, actually improves the result considerably. Having different batchnorm parameters was also necessery for making it work on mixture on gaussian. The authors were not able to make it work on MoG without different parameters. One possible way, could be to let optimizer know that we are on different step by giving the temperature information to the optimizer too. But there's a
tradeoff, in some cases having different batchnorm parameters, improves the results (visually as well as lower bound) and in some cases it decreases the bound considerably, keeping the visual quality of images same.

We observed better results while updating the parameters in online-mode, as compared to batch mode. (i.e instead of accumulating gradients across different steps, we update the parameters in an online fashion) 

