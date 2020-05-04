#!/bin/bash

python3.6 main.py --mode "eval" \
               --hidden-size 512 \
               --batch-size 512 \
               --vbatch-size 4096 \
               --epoch 80 \
               --data-root data/ \
               --save log_dgx/trash1 \
               --resume log_dgx/tr_te_wholeVQA/best.pth.tar \
	       --log-freq 50 \
               --wemb-init /b_test/jolly/VQA_Bottom-up/data_vqa2/glove_pretrained_300.npy \
