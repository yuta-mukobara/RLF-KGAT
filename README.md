# Rethinking Loss Functions for Fact Verification

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

## Contact
If you have questions, suggestions and bug reports, please email:
```
yuta.mukobara@gmail.com
```
