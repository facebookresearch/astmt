# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--active_tasks 1 1 1 1 1 \
--arch se_res26 \
--pretr imagenet \
-lr 0.001 \
--trBatch 8 \
--epochs 60  \
--cls atrous-v3 \
--stride 16 \
--trNorm True \
--overfit False  \
--dec_w 64 \
--dscr fconv \
--lr_dscr 10 \
--dscrk 1 \
--dscrd 2  \
--seenc True \
--sedec True \
--adapt True \
--dscr_w 0.01 \
--resume_epoch 60