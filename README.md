<p align="center">
<h1 align="center">
Studying How to Efficiently and Effectively Guide Models with Explanations
</h1>

<p align="center">
<a href="https://sukrutrao.github.io"><strong>Sukrut Rao<sup>*</sup></strong></a>
·
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/moritz-boehle/"><strong>Moritz Böhle<sup>*</sup></strong></a>
·
<a href="https://www.linkedin.com/in/amin-parchami"><strong>Amin Parchami-Araghi</strong></a>
·
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele"><strong>Bernt Schiele</strong></a>
</p>
  
<h3 align="center">
IEEE/CVF International Conference on Computer Vision (ICCV) 2023
</h3>
  
<h3 align="center">
<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Rao_Studying_How_to_Efficiently_and_Effectively_Guide_Models_with_Explanations_ICCV_2023_paper.html">Paper</a>
|
<a href="https://github.com/sukrutrao/Model-Guidance">Code</a>
|
<a href="https://youtu.be/g9tKVe3fEcQ?feature=shared">Video</a>
</h3>
</p>

## Setup

### Prerequisites

All the required packages can be installed using conda with the provided [environment.yml](environment.yml) file.

### Data

Scripts to download and preprocess the VOC2007 and COCO2014 datasets have been provided in the [datasets](datasets) directory. Please refer to the README file provided there.

### ImageNet Pre-trained Weights

A script to download the pre-trained ImageNet weights for B-cos and X-DNN backbones has been provided in the [weights](weights) directory. Please refer to the README file provided there.


## Training Models

To train a model, use:

```bash
python train.py [options]
```

The list of options and their descriptions can be found by using:

```bash
python train.py -h
```

### Training without Model Guidance

For example, to train a B-cos model on VOC2007, use:

```bash
python train.py --model_backbone bcos --dataset VOC2007 --learning_rate 1e-4 --train_batch_size 64 --total_epochs 300
```

### Fine-tuning with Model Guidance

For example, to optimize B-cos attributions using the Energy loss at the Input layer, use:

```bash
python train.py --model_backbone bcos --dataset VOC2007 --learning_rate 1e-4 --train_batch_size 64 --total_epochs 50 --optimize_explanations --model_path models/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-04_sll1.0_layerInput/model_checkpoint_f1_best.pt --localization_loss_lambda 1e-3 --layer Input --localization_loss_fn Energy --pareto
```
---
Code for training on the Waterbirds-100 dataset and scripts for visualizing explanations will be added soon.

## Acknowledgements

This repository uses and builds upon code from the following repositories:
* [B-cos/B-cos-v2](https://github.com/B-cos/B-cos-v2)
* [stevenstalder/NN-Explainer](https://github.com/stevenstalder/NN-Explainer)
* [visinf/fast-axiomatic-attribution](https://github.com/visinf/fast-axiomatic-attribution)

## Citation

Please cite our paper as follows:

```tex
@InProceedings{Rao_2023_ICCV,
    author    = {Rao, Sukrut and B\"ohle, Moritz and Parchami-Araghi, Amin and Schiele, Bernt},
    title     = {Studying How to Efficiently and Effectively Guide Models with Explanations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {1922-1933}
}
```



