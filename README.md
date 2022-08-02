<div align="center">

# Large-Scale Product Retrieval with Weakly Supervised Representation Learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Conference](http://img.shields.io/badge/CVPR-2022-4b44ce.svg)](https://sites.google.com/view/fgvc9/home?authuser=0)
[![Paper](http://img.shields.io/badge/paper-arxiv.2208.00955-B31B1B.svg)](https://arxiv.org/abs/2208.00955)
[![Leaderboard](http://img.shields.io/badge/EvalAI-Leaderboard-4b44ce.svg)](https://eval.ai/web/challenges/challenge-page/1541/leaderboard/3831)
[![Certificate](http://img.shields.io/badge/2nd-Certificate-yellow.svg)](https://kamwoh.github.io/files/2nd-place-certificate-eproduct-fgvc9.pdf)


</div>

## Description

The second place solution (Involution King) for 2nd eBay eProduct Visual Search Challenge (FGVC9-CVPR2022).

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

# set eval.ai CLI
evalai set_token eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6MTY3Nzg0MDYxMCwianRpIjoiYjM5MjcyNmViZjQ4NDNlODgyZDE5M2I2MzJmMTE3NDgiLCJ1c2VyX2lkIjoxODkxNX0.kemV9j0kiX6is1h-Y1P2NT93_Sxl0CuYN3N_F7A1W2w
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on single GPU
python train.py trainer.gpus=1

# train on multiple GPUs
python train.py trainer.gpus=4
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

You can do inference like this
```bash
python test.py datamodule.batch_size=1024 datamodule.num_workers=4 ckpt_path=<path to ckpt>
```

You can submit result via eval.ai CLI like this
```bash
evalai challenge 1541 phase 3084 submit --file <submission_file_path>
```
