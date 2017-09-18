Tensorfow implementation of BEGAN with multi generators and multi-GPU support


## Purpose
Consider that for a distribution of functional requirements, there theoretically exists a corresponding set of conceptual designs (the ground truth) that fulfill therequirements, and some of them are revealed as the samples (existing designs). The learningobjective of the proposed system is to match the generated designs with the ground truth, for all functional requirements. The key challenge of this task is in extrapolation, i.e., whilemodels can fit data and interpolate well, they often do not have the capability to create newdesigns that fulfill new functions or even physically make sense.  This challenge can be observed from state-of-the-art generative models (e.g., for attribute-based image generation),that  perform  reasonably  well  within  the  vicinity  of  the  existing  data  points  (3D  models)while fail hard on tasks where the generations are meant to be far from the data

![generative3](/../subnets/assets/generative3.png)
> (a) A SOA model (3D GAN) that produces plausible component combinations forsimilar3D objects.  (b) Given images containing “squares” and “square+crosses”, existing models cannotgenerate  a  “cross”,  i.e.,  they  fail  at  extracting  components  from  an  assembly,  thus  will  not  beable to produce meaningfulnovelcombinations (see details).  The proposed method will addressfundamental limitations of the current SOA

## Requirements

This repo requires [tensorflow](https://tensorflow.org/) to run. It has been tested under version 1.1.0. Other required packages include:
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [h5py](http://www.h5py.org/)
- [PIL](http://www.pythonware.com/products/pil/)
- [datetime](https://stackoverflow.com/questions/20849517/no-datetime-module-in-python-how-to-install-via-easy-install)
(If not included in your python package)

## Usage
First download the dataset by:
```sh
$ wget -O crs.zip https://www.dropbox.com/sh/hxduua3dhghoreu/AADLbqiOOQjxHtEBoTlk1DZja?dl=1
$ unzip crs.zip -d data
```
Network settings can be specified in config.py. To train a model, simply run:
```sh
$ python main.py
```
Training is monitored under tensorboard, which can be visualized by running:
```sh
$ tensorboard --logdir=./logs --port=6006
```
The results are saved under ./logs and checkpoint is saved under ./models.

## Network structure

Since single generative network will have a hard time in doing this job, we made use of multiple generators.
Our network is based BEGAN but with multiple generators and decoders, as shown in figure below:

![model](/../subnets/assets/model.png)

The training schedule is composed of three stages.
Besides iteratively training of generator and discriminator we will also update a control variable k_t at every iteration.
As in vanilla BEGAN, k_t controls the learning speed of generator and discriminator.

![g_opt](http://latex.codecogs.com/gif.latex?\theta_G=\min{L_G})

![d_opt](http://latex.codecogs.com/gif.latex?\theta_E,\theta_D=\min(L_D-k_tL_G))

![kt](http://latex.codecogs.com/gif.latex?k_{t+1}=k_t+\lambda_k\Big({\gamma}L_D-L_G\Big))

where:

![gloss]( http://latex.codecogs.com/gif.latex?L_G=L_{rec}(x_f)+L_z(dz_f))

![dloss](http://latex.codecogs.com/gif.latex?L_D=L_{rec}(x_r)&plus;L_z(dz_r))

and:

![lx](http://latex.codecogs.com/gif.latex?L_{rec}=||x-x_{rec}||_2)

![lz](http://latex.codecogs.com/gif.latex?L_z=||z-dz||_2)

## Results

For a toy case, consider we have a collection of images of "square" and "cross+square" of different orientation and size.
Note that we are trying to recover the information of "cross" from data directly.
Instead, we wish the network to figure out the distribution for the "cross"  from the "cross+square" and "square" data by doing subtraction.

![model](/../subnets/assets/itr44500.png)

> Original images x_r, reconstruct real images x_r_rec, generated images x_f, reconstructed generated images x_f_rec
> output of the first decoder subnetwork when x_f as input, output of the first decoder subnetwork when x_r as input
> output of the second decoder subnetwork when x_f as input, output of the second decoder subnetwork when x_r as input
> output of the first generator subnetwork with random input, output of the second decoder subnetwork with random input

However, from current results, it seems that the network can infer the mean value of the "cross" distribution, but not the variance.



## Related works

This project is largely based on

| Github | Arxiv |
| ------ | ------ |
| [BEGAN](https://github.com/hope-yao/BEGAN-tensorflow) | https://arxiv.org/abs/1703.10717 |



## Todos

 - Mode collapse exists for unseen attributes
 - Compare with more networks


