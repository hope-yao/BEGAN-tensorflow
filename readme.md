Tensorfow implementation of BEGAN with multi generators


## Purpose
Every complex object or structure is assembled from several subsystem or components. Our goal is to create a generative model, which takes the assembled complex structure and decompose it into its subsystems. By creatively combine the subsystems in some novel way, the generator should also be able to create new design configurations from these components.

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

As shown in figure below, this network is very similar to BEGAN. 
but not all individual components are realized as images in a binary vector representing the choice of components
For a much simpler case, consider and output an image that realizes the choice. Yes, the training data we used does not have the information of "cross" data directly. Instead, we wish the network to figure out the distribution for the "cross"  from the "cross+square" and "square" data by doing subtraction. However, it seems that the network can infer the mean value of the "cross" distribution, but not the variance. I have tested network parameters a lot but it doesn't help much. 
A single generative network will have a hard time in doing job. 
 

![struc](https://www.dropbox.com/sh/7vrmmhtil6k7chi/AACjq5M2TN6AutPpVdcSUTf0a?dl=0&preview=struc.PNG)

![gloss]( http://latex.codecogs.com/gif.latex?L_G=L_{rec}(x_f)L_z(x_f))

![dloss](http://latex.codecogs.com/gif.latex?L_D=L_{rec}(x_r)&plus;L_z(dz_r)-k_t[L(x_f)&plus;L_z(dz_f)])

![kt](http://latex.codecogs.com/gif.latex?k_{t+1}=k_t+\lambda_k\Big(\gamma[L_{rec}(x_r)&plus;L_z(dz_r)]-[L(x_f)&plus;L_z(dz_f)]\Big))

![test](http://latex.codecogs.com/gif.latex?)


## Results


> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.


This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.


## Related works

This project is largely based on
| Github | Arxiv |
| ------ | ------ |
| [BEGAN](https://github.com/hope-yao/BEGAN-tensorflow) | [plugins/dropbox/README.md] [PlDb] |


## Todos

 - Mode collapse exists for unseen attributes
 - Compare with more networks

License
----

MIT





































































































































































































































