# Rethinking Loss Functions for Fact Verification


This is the official implementation of the following paper:

>Yuta Mukobara, Yutaro Shigeto, Masashi Shimbo. 
>Rethinking Loss Functions for Fact Verification.
>EACL 2024

[ [arXiv](https://arxiv.org/abs/2403.08174) | [ACL Anthology](https://aclanthology.org/2024.eacl-short.38) ]




The following shows how to clone a repository including even submodules.
```
git clone --recursive git@github.com:yuta-mukobara/RLF-KGAT.git
```

A Docker environment for [thunlp/KernelGAT](https://github.com/thunlp/KernelGAT)


## Build a docker image
```
make docker-build
```


## Download data and checkpoint

BERT based models and checkpoints used for training and RoBERTa based models and checkpoints can be downloaded with the following command.

```
make download
```
If the above command did not download successfully, you can download from Ali Drive in [thunlp/KernelGAT](https://github.com/thunlp/KernelGAT).

All data and BERT based chechpoints: [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip)

RoBERTa based models and chechpoints: [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT_roberta_large.zip)


## Preprocess

Set up for train, test and eval.
```
make prepro
```


## Usage

### Train
```
make kgat
```

#### Hyperparameter
- **comp** (None, all, srn, sr) OVR refers to XE in the paper.
- **nl_coef** (float) nl_coef corresponds to $\lambda$ in the paper.
- **imb** (store_true) with/without imbalanced learning
- **beta** (float) beta refers to the Weighting $\beta$ in the paper.


### Test
```
make test
```


### Evaluate
```
make eval
```


## Citation

If you use this code, please cite our paper:
```
@inproceedings{mukobara-etal-2024-rethinking,
    title     = {Rethinking Loss Functions for Fact Verification},
    author    = {Mukobara, Yuta and Shigeto, Yutaro and Shimbo, Masashi},
    booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers)},
    month     = {03},
    year      = {2024},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2024.eacl-short.38},
    pages     = {432--442}
}

```


## Contact
If you have questions, suggestions and bug reports, please email:
```
yuta.mukobara@gmail.com
```
