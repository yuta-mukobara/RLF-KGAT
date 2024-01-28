VER='0.1'
DOCKER_IMAGE='rlf_kgat:'$(VER)

docker-build:
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

docker-run:
	docker run --gpus all -it --rm --shm-size 100G \
  -v $(PWD)/KernelGAT/:/workspace/ $(DOCKER_IMAGE) \
    /bin/bash


download:
	cd KernelGAT && wget https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip && \
    unzip "KernelGAT.zip" && rm -rf "KernelGAT.zip"
	cd KernelGAT && wget https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT_roberta_large.zip && \
    unzip "KernelGAT_roberta_large.zip" && rm -rf "KernelGAT_roberta_large.zip"


prepro:
	cp -f new_files/* KernelGAT/kgat/
	cp -f new_files/* KernelGAT/KernelGAT_roberta_large/kgat/
	sed -e "s/768/1024/g" new_files/train.py > KernelGAT/KernelGAT_roberta_large/kgat/train.py
	sed -e "s/768/1024/g" new_files/test.py > KernelGAT/KernelGAT_roberta_large/kgat/test.py
	cp KernelGAT/KernelGAT/data/* KernelGAT/data/
	cp KernelGAT/KernelGAT/bert_base/* KernelGAT/bert_base/


kgat:
	docker run --gpus all -it --rm --shm-size 100G \
    -v $(PWD)/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python kgat/train.py \
      --outdir checkpoint \
      --train_path data/bert_train.json \
      --valid_path data/bert_dev.json \
      --bert_pretrain bert_base \
      --postpretrain /workspace/KernelGAT/checkpoint/pretrain/model.best.pt \
      --comp sr \
      --nl_coef 0.25 \
      --imb --beta 0.999999


test:
	docker run --gpus all -it --rm --shm-size 100G \
    -v $(PWD)/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python kgat/test.py \
      --outdir kgat/output/ \
      --test_path data/bert_eval.json \
      --bert_pretrain bert_base \
      --checkpoint checkpoint \
      --name dev.json \
      --comp sr \
      --nl_coef 0.25 \
      --imb --beta 0.999999
	docker run --gpus all -it --rm --shm-size 100G \
    -v $(PWD)/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python kgat/test.py \
      --outdir kgat/output/ \
      --test_path data/bert_test.json \
      --bert_pretrain bert_base \
      --checkpoint checkpoint \
      --name test.json \
      --comp sr \
      --nl_coef 0.25 \
      --imb --beta 0.999999


eval:
	docker run --gpus all -it --rm --shm-size 100G \
    -v $(PWD)/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python kgat/fever_score_test.py \
      --predicted_labels kgat/output/dev.json \
      --predicted_evidence data/bert_eval.json \
      --actual data/dev_eval.json \
      --name eval

