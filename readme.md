# Purpose
Every complex object or structure is assembled from several subsystem or components. Our goal is to create a generative model, which takes the assembled complex structure and decompose it into its subsystems. By creatively combine the subsystems in some novel way, the generator should also be able to create new design configurations from these components.


# Results

You can also:
  - Import and save files from GitHub, Dropbox, Google Drive and One Drive
  - Drag and drop markdown and HTML files into Dillinger
  - Export documents as Markdown, HTML and PDF

> The overriding design goal for Markdown's
> formatting syntax is to make it as readable
> as possible. The idea is that a
> Markdown-formatted document should be
> publishable as-is, as plain text, without
> looking like it's been marked up with tags
> or formatting instructions.
### Prerequisite

This repo requires [tensorflow](https://tensorflow.org/) to run. It has been tested under version 1.1.0. Other packages used include:

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md] [PlDb] |

To excecute the program, simply run
```sh
$ python main.py
```

![gloss]( http://latex.codecogs.com/gif.latex?L_G=L_{rec}(x_f)L_z(x_f))

![dloss](http://latex.codecogs.com/gif.latex?L_D=L_{rec}(x_r)&plus;L_z(dz_r)-k_t[L(x_f)&plus;L_z(dz_f)])

![kt](http://latex.codecogs.com/gif.latex?k_{t+1}=k_t+\lambda_k\Big(\gamma[L_{rec}(x_r)&plus;L_z(dz_r)]-[L(x_f)&plus;L_z(dz_f)]\Big))

![test](http://latex.codecogs.com/gif.latex?)

![struc](https://www.dropbox.com/sh/7vrmmhtil6k7chi/AACjq5M2TN6AutPpVdcSUTf0a?dl=0&preview=struc.PNG)
Training is monitored under tensorboard, which can be visualized by running:

```sh
$ tensorboard --logdir=./logs --port=6006
```

### Results


This will create the dillinger image and pull in the necessary dependencies. Be sure to swap out `${package.json.version}` with the actual version of Dillinger.

### Todos

 - Write MORE Tests
 - Add Night Mode

License
----

MIT



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [tensorflow]: <https://www.tensorflow.org>



































































































































































































































