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
```
make prepro
```


## Usage

### Train
```
make kgat
```

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
