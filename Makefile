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
	cp KernelGAT/KernelGAT/data/* KernelGAT/data/


kgat:
	docker run --gpus all -it --rm --shm-size 100G \
    -v $(PWD)/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python kgat/train.py \
      --outdir checkpoint \
      --train_path data/bert_train.json \
      --valid_path data/bert_dev.json \
      --bert_pretrain bert_base \
      --postpretrain /workspace/KernelGAT/checkpoint/pretrain/model.best.pt \
      --comp $(NUM1) \
      --nl_coef $(NUM2) \
      --imb --beta $(NUM3)


eval_kgat:
	docker run --gpus all -it --rm --shm-size 100G \
    -v $(PWD)/KernelGAT/KernelGAT_roberta_large/:/workspace/KernelGAT_roberta_large \
    -v /data2/mukobara/KernelGAT:/workspace $(DOCKER_IMAGE) \
    python KernelGAT_roberta_large/kgat/fever_score_test.py \
      --predicted_labels KernelGAT_roberta_large/kgat/output/dev.json \
      --predicted_evidence data/bert_eval.json \
      --actual data/dev_eval.json \
      --idx $(NUM4) \
      --name eval


test_kgat:
	docker run --gpus all -it --rm --shm-size 100G \
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
	docker run --gpus all -it --rm --shm-size 100G \
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
