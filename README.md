# Deep_Learning_Project

Implementation of a Noise2Noise model: an image denoising network trained without a clean reference image. The original paper can be found at
https://arxiv.org/abs/1803.04189.

The project is divided in two parts: The first one is to build a network that denoises using the PyTorch framework, in particular the torch.nn modules and autograd. The development and analysis of an Autoencoder can be found in the Miniproject_1 folder. Investigation of different weight initialisations has been made, as well a parameter optimisation. 

The second project can be found under Miniproject_2. A simplified version of an Autoencoder has been built and includes a framework, its constituent modules, that are the standard building blocks of deep networks without PyTorch's autograd. All modules have been hardcoded by hand. 
