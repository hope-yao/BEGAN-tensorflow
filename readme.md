Tensorfow implementation of BEGAN with multi generators


## Purpose
Every complex object or structure is assembled from several subsystem or components.
Our goal is to create a generative model, which takes the assembled complex structure and decompose it into its subsystems. By creatively combine the subsystems in some novel way, the generator should also be able to create new design configurations from these components.

## Requirements

This repo requires [tensorflow](https://tensorflow.org/) to run. It has been tested under version 1.1.0. Other required packages include:
- [Dataset]()
- [tqdm]()
- [h5py]()
- [PIL]()
- [datetime]()

## Usage
To train a model, simply run
```sh
$ python main.py
```
Training is monitored under tensorboard, which can be visualized by running:
```sh
$ tensorboard --logdir=./logs --port=6006
```
To test a model, simply run
```sh
$ python test.py
```

## Network structure

Since single generative network will have a hard time in doing this job, we made use of multiple generators.
Our network is based BEGAN but with multiple generators and decoders, as shown in figure below:

![model](/../subnets/assets/model.png)

The training schedule is composed of three stages.
Besides iteratively training of generator and discriminator we will also update a control variable k_t at every iteration.
As in vanilla BEGAN, k_t controls the learning speed of generator and discriminator.
![g_opt](http://latex.codecogs.com/gif.latex?\min_{\theta_G}L_g)
![d_opt](http://latex.codecogs.com/gif.latex?\min_{\theta_E,\theta_D}(L_D-k_tL_G))
![kt](http://latex.codecogs.com/gif.latex?k_{t+1}=k_t+\lambda_k\Big(\gamma L_D-L_G\Big))
where:
![gloss]( http://latex.codecogs.com/gif.latex?L_G=L_{rec}(x_f)L_z(x_f))
![dloss](http://latex.codecogs.com/gif.latex?L_D=L_{rec}(x_r)&plus;L_z(dz_r))


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

License
----

MIT

