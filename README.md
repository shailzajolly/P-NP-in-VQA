# P ≈ NP, at least in Visual Question Answering 

This is a Pytorch implementation for our recent paper (https://arxiv.org/abs/2003.11844). The paper talks about polar (Yes/No) and non-polar (Non Yes/No) imbalances in Visual Question Answering dataset.

## Prerequisites
- python 3.6+
- numpy
- [pytorch](http://pytorch.org/) 0.4
- [tqdm](https://pypi.python.org/pypi/tqdm)
- [nltk](http://www.nltk.org/install.html)
- [pandas](https://pandas.pydata.org/)

## Implementation Details

Similar to [this repo](https://github.com/hengyuan-hu/bottom-up-attention-vqa), we also don't use extra data from [Visual Genome](http://visualgenome.org/), use only a fixed number of objects per image (K=36), use a simple and single stream classifier without pre-training and use ReLU activation instead of GatedTanh in our final implementation. 
The code for GatedTanh is also provided in the model.py file which can be used for further experimentations. 

These simplifications lead to faster training and help us to reach validation accuracy very near to the one reported in the paper [Best result on validation set: 63.15]. However, the results without using visual genome data [62.48], using 36 objects per image [62.82], result using ReLU [61.63] are close to the one we got i.e. 62.4% on Validation data of VQA. Please refer to table 1 of paper: "Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge", for these numbers. 


## Data
- [VQA 2.0](http://visualqa.org/download.html)
- [COCO 36 features pretrained resnet model](https://github.com/peteanderson80/bottom-up-attention#pretrained-features)
- [GloVe pretrained Wikipedia+Gigaword word embedding](https://nlp.stanford.edu/projects/glove/)

## Steps to use this repository
- First, download the data using:
```bash
  bash download_extract.sh
  ```
This will download and extract annotation files of VQA V2, glove embeddings and pretrained image features.  

- Secondly, the following command is used to prepare data for training. 
```bash
python prepro.py
  ```
-Once the data is ready for training, the training can be started using this command. 
```bash
bash train.sh
  ```
  This will produce results on validation split of VQA2.0 with 62.4% validation accuracy after 31 training epochs. Further, simply change file names in prepro.py to train model on polar or non-polar samples.

-Once the trained model is saved you can extract features of either polar or non-polar samples using this trained model using

```bash
bash get_j_feats.sh
  ```
  This will give features which can then be used by polar or non-polar classifier using 

```bash
bash train_yn.sh
  ```

## Notes
- Some of `preproc.py` and `utils.py` are based on [this repo](https://github.com/markdtw/vqa-winner-cvprw-2017) 

## Resources
- [The paper](https://arxiv.org/pdf/1708.02711.pdf).
- [Their CVPR Workshop slides](http://cs.adelaide.edu.au/~Damien/Research/VQA-Challenge-Slides-TeneyAnderson.pdf).

## Citation
If you plan to use this code as part of your published research, we'd appreciate it if you could cite our paper:
```
@misc{jolly2020p,
    title={P $\approx$ NP, at least in Visual Question Answering},
    author={Shailza Jolly and Sebastian Palacio and Joachim Folz and Federico Raue and Joern Hees and Andreas Dengel},
    year={2020},
    eprint={2003.11844},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```