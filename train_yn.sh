#!/bin/bash

python3.6 yn_net.py --mode "eval" \
               --hidden-size 512 \
               --batch-size 256 \
               --vbatch-size 4096 \
               --epoch 200 \
               --data-root data/ \
               --save log_dgx/trash1\
	       --resume log_dgx/tr_te_np_np_ans/best.pth.tar \
               --log-freq 2000 \
               --wemb-init data/glove_pretrained_300.npy \
