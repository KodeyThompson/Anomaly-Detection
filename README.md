# Industrial Scale Data Poisoning (branch)

This is based on https://github.com/JonasGeiping/data-poisoning

This framework implements data poisoning strategies that reliably apply imperceptible adversarial patterns to training data. If this training data is later used to train an entirely new model, this new model will misclassify specific target images.

Implemented settings:
* From-scratch training
* Finetuning (Transfer learning a finetuned feature representation)
* Transfer learning (with a fixed feature representation)


Implemented poison recipes:
* Gradient Matching (Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching - https://openreview.net/forum?id=01olnfLIbD)
* Poison Frogs (Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks - https://arxiv.org/abs/1804.00792)
* Bullseye Polytope (Bullseye Polytope: A Scalable Clean-Label Poisoning Attack with Improved Transferability - https://arxiv.org/abs/2005.00191) [in the straight feature-collision version]
* Watermarking  [https://arxiv.org/abs/1804.00792]
* MetaPoison (MetaPoison: Practical General-purpose Clean-label Data Poisoning - https://arxiv.org/abs/2004.00225) [for small ensembles, without color perturbation, refer to https://github.com/wronnyhuang/metapoison for the full construction of a large staggered ensemble and color perturbations.]
* Hidden-Trigger Backdoor Attacks (https://arxiv.org/abs/1910.00033)
* Convex Polytope attack (https://arxiv.org/pdf/1905.05897.pdf) [However we do not implement the threat model discussed there]
* Patch Attacks (BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain - https://arxiv.org/abs/1708.06733)
### Dependencies:
* PyTorch => 1.6.*
* torchvision > 0.5.*
* higher [best to directly clone https://github.com/facebookresearch/higher and use ```pip install .```]
* efficientnet_pytorch [```pip install --upgrade efficientnet-pytorch``` only if EfficientNet is used]
* python-lmdb [only if datasets are supposed to be written to an LMDB]



## USAGE:

The cmd-line script ```brew_poison.py``` can be run with default parameters to get a first impression for a ResNet18 on CIFAR10 with some ``--recipe`` for a poisoning attack, such as ```--recipe gradient-matching```.

### Cmd-Line Interface Usage:

The cmd-script features a ton of arguments, most of which are very optional. Important parameters are:
* ```--net```: Choose the neural network that is being poisoned. Current options:
    - *CIFAR/MNIST*: ```ResNet-``` variants including WideResNets via ```ResNet-DEPTH-WIDTH```, a ```ConvNet```, ```Linear```, ```VGG``` variants
    - *ImageNet*: All torchvision models (https://pytorch.org/docs/stable/torchvision/models.html), ```Linear``` and EfficientNet (https://github.com/lukemelas/EfficientNet-PyTorch)
* ```--dataset```: Choose the dataset from (currently) ```CIFAR10```, ```CIFAR100```, ```MNIST``` and ```ImageNet```
* ```--threatmodel```: Poisoning can work in a variety of threat models. Options: ```single-class```, ```third-party``` and ```random-subset```
* ```--scenario```: Whether to attack in the from-scratch, finetuning or transfer scenario.

* ```--eps```, ```--budget``` and ```--targets``` control the strength of the attack

* ```--ensemble``` controls the number of networks used to brew the poison.

All possible arguments can be found under ```forest/options.py```.

## Attacks
Attacks are controlled by the ``--recipe`` argument, which can be one of
* ``gradient-matching``
* ``watermark``
* ``poison-frogs``
* ``metapoison``
* ``hidden-trigger``
* ``convex-polytope``                                                                         
* ``bullseye``
* ``patch``

or one of the variations mentioned in ``forest/options.py``, for example ``gradient-matching-private`` to adaptively attack DP-SGD defenses.

## Defenses
### Filtering Defenses
To apply filter defenses, supply the argument ``--filter_defense`` with one of ``spectral_signatures, deepknn, activation_clustering``.

### Data Augmentation Defenses
To apply defenses through strong data augmentations (https://arxiv.org/abs/2103.02079) that mix data, use ``--mixing_method`` and provide a data augmentation argument,
such as ``Cutout,Mixup,CutMix,Maxup-Cutout,8way-mixup`` or variations thereof. The mixing of label information can be disabled by ``--mixing_disable_correction``. This defense can be easily incorporated into a different codebase using only the file ``forest/data/mixing_data_augmentations.py``.

### Differential Privacy Defenses
Supply the arguments ``--gradient_noise`` and ``--gradient_clip``.


### Adversarial Defenses
To apply defenses based on adversarial training (https://arxiv.org/abs/2102.13624), call the same code with the additional arguments ``--optimization defensive --defense_type adversarial-wb-recombine``, i.e.
```
python brew_poison --net Resnet18 --recipe gradient-matching --restarts 1 --optimization defensive --defense_type adversarial-wb-recombine
```

The `--defense_type` argument can be broken up into components, to ablate parts of the defense. Replacing the middle part `-wb-` (standing for Witches' Brew, i.e. gradient matching) with another attack, e.g. `-fc-` (for feature collision), allows for a different attack to be used during the defense.

## Distributed Training

Using model ensembles to brew transferable poisons is *very* parallel. We make us of this by launching via
```
python -m torch.distributed.launch --nproc_per_node=GPUs_per_node --master_port=20704\
          dist_brew_poison.py --extra_args
```
on a single-node setup with multiple GPUs.

Running on a multi-node setup, the launch command is slightly different:
##### SLURM:
```
python -m torch.distributed.launch --nproc_per_node=$SLURM_GPUS_PER_NODE
          --nnodes=$SLURM_JOB_NUM_NODES --node_rank=$SLURM_NODEID\
          --master_addr="192.168.1.1" --master_port=11234 \
          dist_brew_poison.py --extra_args
```

#### PBS:
```
python -m torch.distributed.launch --nproc_per_node=4
          --nnodes=$PBS_NUM_NODES --node_rank=$PBS_O_NODENUM \
          --master_addr="192.168.1.1" --master_port=11234 \
          dist_brew_poison.py --extra_args
```
See https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py for details.
