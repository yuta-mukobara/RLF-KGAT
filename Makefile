VER='0.1'
DOCKER_IMAGE='rlf_kgat:'$(VER)

docker-build:
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

docker-run:
	docker run --gpus '"device=1"' -it --rm --shm-size 100G \
  -v $(PWD)/KernelGAT/KernelGAT_roberta_large/:/workspace/KernelGAT_roberta_large \
  -v /data2/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) \
    /bin/bash

################ 

download:
	cd KernelGAT && curl -O https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip && unzip KernelGAT.zip && curl -O https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT_roberta_large.zip && unzip KernelGAT_roberta_large.zip && cp -r KernelGAT/* . && cp KernelGAT_roberta_large/* . 

retrieval: 
	docker run --gpus 1 -it --rm --shm-size 100G -v $(PWD)/KernelGAT/kgat:/workspace/kgat -v /data1/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) python data/generate_pair.py --infile data/all_dev.json --outfile data/dev_pair
	docker run --gpus 1 -it --rm --shm-size 100G -v $(PWD)/KernelGAT/kgat:/workspace/kgat -v /data1/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) python data/generate_pair.py --infile data/all_train.json --outfile data/train_pair
	docker run --gpus 1 -it --rm --shm-size 100G -v $(PWD)/KernelGAT/kgat:/workspace/kgat -v /data1/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) python retrieval_model/train.py --outdir checkpoint/retrieval_model \
                                                                                                                          --train_path data/train_pair \
                                                                                                                          --valid_path data/dev_pair \
                                                                                                                          --bert_pretrain bert_base


kgat: 
	docker run --gpus '"device=1"' -it --rm --shm-size 100G \
  -v $(PWD)/KernelGAT/KernelGAT_roberta_large/:/workspace/KernelGAT_roberta_large \
  -v /data2/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python KernelGAT_roberta_large/kgat/train.py \
      --outdir checkpoint_$(CNT)/kgat \
      --train_path data/new.json \
      --valid_path data/bert_dev.json \
      --bert_pretrain KernelGAT_roberta_large/roberta_large \
      --comp $(NUM1) \
      --nl_coef $(NUM2) \
      --imb --beta $(NUM3)


eval_kgat:
	docker run --gpus '"device=1"' -it --rm --shm-size 100G \
  -v $(PWD)/KernelGAT/KernelGAT_roberta_large/:/workspace/KernelGAT_roberta_large \
  -v /data2/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python KernelGAT_roberta_large/kgat/fever_score_test.py \
      --predicted_labels KernelGAT_roberta_large/kgat/output/dev.json \
      --predicted_evidence data/bert_eval.json \
      --actual data/dev_eval.json \
      --idx $(NUM4) \
      --name eval


test_kgat:
	docker run --gpus '"device=1"' -it --rm --shm-size 100G \
  -v $(PWD)/KernelGAT/KernelGAT_roberta_large/:/workspace/KernelGAT_roberta_large \
  -v /data2/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python KernelGAT_roberta_large/kgat/test.py \
      --outdir KernelGAT_roberta_large/kgat/output \
      --test_path data/bert_eval.json \
      --bert_pretrain KernelGAT_roberta_large/roberta_large \
      --checkpoint checkpoint_$(CNT)/kgat \
      --name dev.json \
      --comp $(NUM1) \
      --nl_coef $(NUM2) \
      --imb --beta $(NUM3) \
      --idx $(NUM4)
	docker run --gpus '"device=1"' -it --rm --shm-size 100G \
  -v $(PWD)/KernelGAT/KernelGAT_roberta_large/:/workspace/KernelGAT_roberta_large \
  -v /data2/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python KernelGAT_roberta_large/kgat/test.py \
      --outdir KernelGAT_roberta_large/kgat/output \
      --test_path data/bert_test.json \
      --bert_pretrain KernelGAT_roberta_large/roberta_large \
      --checkpoint checkpoint_$(CNT)/kgat \
      --name test.json \
      --comp $(NUM1) \
      --nl_coef $(NUM2) \
      --imb --beta $(NUM3) \
      --idx $(NUM4)
