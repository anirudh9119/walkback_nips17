# walkback_nips17

Rosemary and Me would like to apologize for not pushing clean code, but you should be able to reproduce the experiments just by running the scripts.

Requirements - The code requires theano(version -0.9.0.dev-c697eeab84e5b8a74908da654b66ec9eca4f1291) Unfortunately, the code is not compatible with the latent version of theano, to make it compatible, we need to make some changes.

This repo contains code for running VW on Cifar, SVHN, CelebA, LSUN, Circle, Spiral , Mixture of Gaussian. Some of these datasets are not mentioned in the paper. 

In all the image experiments, we observed that by having different batchnorm papemeters for different steps, actually improves the result considerably. Having different batchnorm parameters was also necessery for making it work on mixture on gaussian. The authors were not able to make it work on MoG without different parameters. One possible way, could be to let optimizer know that we are on different step by giving the temperature information to the optimizer too. But there's a
tradeoff, in some cases having different batchnorm parameters, improves the results (visually as well as lower bound) and in some cases it decreases the bound considerably, keeping the visual quality of images same.

We observed better results while updating the parameters in online-mode, as compared to batch mode. (i.e instead of accumulating gradients across different steps, we update the parameters in an online fashion) 

The authors would also like to thank open-source contributors from all the different libraries, as the authors have used source code of other people too. (https://github.com/casperkaae/parmesan,  https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models)
