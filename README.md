# MPJL: Multi-Period Joint Learning with Period-Aligned Relational Imaging

This repository provides the official PyTorch implementation of our paper:
**"Multi-Period Joint Learning with Period-Aligned Relational Imaging for Industrial Multivariate Time-Series Anomaly Detection"**

---

## Requirements

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

The code was tested on Ubuntu 20.04 with an NVIDIA RTX 3090 GPU.

---

## Getting Started

Please download the datasets from their official sources and place them into the folder:

```text
./dataset/
```

The expected file structure is:

```text
dataset/
  SKAB/
    SKAB_train.npy
    SKAB_test.npy
    SKAB_test_label.npy
```

On the first run, TC-GAF relational-image patches will be generated and cached automatically under:

```text
dataset/<DATASET_NAME>/patches/
```

---

## Reproducing Table 4 Results (SKAB)

A pre-trained checkpoint is provided under `weights/`. To evaluate directly without retraining:

```bash
python main.py \
  --dataset SKAB --data_path SKAB \
  --win_size 64 --input_size 16 --input_c 8 --output_c 8 \
  --mode test \
  --anomaly_ratio 0.3 \
  --thr_mode combined \
  --random_seed 42 \
  --ckpt_tag baseline \
  --ckpt_seed 42 \
  --save_artifacts 1
```

---

## Training from Scratch

To retrain MPJL on SKAB from scratch:

```bash
python main.py \
  --dataset SKAB --data_path SKAB \
  --win_size 64 --input_size 16 --input_c 8 --output_c 8 \
  --mode train \
  --anomaly_ratio 0.3 \
  --loss_fuc margin --margin 0.05 \
  --thr_mode combined \
  --random_seed 42 \
  --save_artifacts 1
```

---
