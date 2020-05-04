#!/bin/bash

python3.6 get_joint_feats.py --mode "eval" \
               --hidden-size 512 \
               --batch-size 4096 \
               --vbatch-size 4096 \
               --epoch 40 \
               --data-root data \
               --save log_dgx/trash1 \
               --log-freq 50 \
	           --resume log_dgx/tr_te_bl_np_ans/best.pth.tar \
               --wemb-init data \
