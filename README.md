# GiTE
## Introduction

Dear Reviewers, this is the cleaned code for submitting "GiTE: A Generic Vision Transformer Encoding Scheme for Efficient ViT Architecture Search".

The training scripts are put under `scripts/`, and the implementation of TA-GATES is put under `my_nas/`. Besides, we also provide the training data (under `data/`) and configurations (under `cfgs/`) to enable the reproducibility. In the following parts, we will guide you to reproduce the experiments in our paper.

To install all requirements, run
```sh
$ conda create -n mynas python==3.7.6 pip
$ conda activate mynas
$ pip install -r requirements.txt
```

## Single-Task Prediction

To evaluate the performance of GiTE with a training proportion 10% on AutoFormer-Tiny space, one should run the following instructions:
```sh
$ python scripts/train_vitasbench_101.py cfgs/vb101/vb101_tiny_gite.yaml --gpu 0 --seed [random seed] --train-dir results/vb101_tiny_gite_tr1e-1/ --save-every 200 --eval-only-last 5 --train-pkl data/ViTAS-Bench-101/vitasbench_101_tiny_train_1.pkl --valid-pkl data/ViTAS-Bench-101/vitasbench_101_tiny_valid.pkl --train-ratio 0.1
```

To evaluate the performance of GiTE on UFO_CPLFW task, one should run the following instructions:
```sh
$ python scripts/train_ufo.py cfgs/ufo/ufo_gite.yaml --gpu 0 --seed [random seed] --train-dir results/ufo_cplfw_gite/ --save-every 200 --eval-only-last 5 --train-pkl data/UFO/cplfw.pkl --valid-pkl data/UFO/cplfw.pkl
```

## Cross-Task Prediction

To evaluate the performance of GiTE with a training proportion 10% on AutoFormer-Tiny space for the aircraft task, one should run the following instructions:
```sh
$ python scripts/train_vitasbench_102.py cfgs/vb102/vb102_tiny_gite.yaml --gpu 0 --seed [random seed] --train-dir results/vb102_tiny_gite_aircraft_tr1e-1/ --save-every 200 --eval-only-last 5 --train-pkl data/ViTAS-Bench-102/vitasbench_102_tiny_train_1.pkl --valid-pkl data/ViTAS-Bench-102/vitasbench_102_tiny_valid.json --train-ratio 0.1 --valid-task aircraft
```
