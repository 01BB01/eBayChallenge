<div align="center">

# FGVC9 2022 eBay Challenge

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/CVPR-2022-4b44ce.svg)](https://sites.google.com/view/fgvc9/home?authuser=0)

</div>

## Description

Our codebase

## How to run
Organize dataset as following under ```./data/eBay/```
```
├── Images
│   ├── index
│   ├── query_part1
│   ├── train
│   └── val
└── metadata
    ├── index.csv
    ├── query_part1.csv
    ├── train.csv
    └── val.csv
```

Install dependencies

```bash
# clone project
git clone https://github.com/01BB01/eBayChallenge.git

# create conda environment
conda create -n ebay python=3.8
conda activate ebay

# install requirements
pip install -r requirements.txt

# install hooks
pre-commit install
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```

You can visualize running experiments here
https://wandb.ai/01bb01/fgvc9_ebay_challenge
