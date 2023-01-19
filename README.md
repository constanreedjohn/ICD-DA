# FOMILE: Unsupervised Sequence-to-Sequence Domain Adpatation with Focal On Minimizing Latent Entropy in Scene Text Recognition.
Author: Hung Tran Tien, Thanh Duc Ngo

Scene Text Recognition has always been a popular interest in Computer Vision field. However, due to the lack of real-world data and labelling process is time-consuming, training a recognition model is difficult. With the help of synthetic data, many proposals have been introduced with promissing results. 
On the other hand, recognizing scene text images still faces challenges due to domain shift between synthetic and real-world domains. Moreover, heavily imbalanced character-level distribution can damages the training process and affects on the model performance. To tackle this problem, we introduced an unsupervised domain adaptation method by minimizing latent representation entropy inspired from Chang et al's SMILE by using focal mechanism. Our proposal outperform SMILE and other UDA methods with SOTA results on official scene text recognition benchmarks.

## Overview
The proposed FOMILE (Unsupervised Sequence-to-Sequence Domain Adaptation with Focal On Minimizing Latent Entropy in Scene Text Recognition.) is an UDA method with minimizing latent representation entropy for scene text recognition inspired by Chang et al's SMILE with focal mechanism.

## Installation
- building environment: ```cuda==11.0, python==3.7.10```

- install requirements:```pip3 install torch==1.2.0 pillow==6.2.1 torchvision==0.4.0 opencv-python scipy lmdb nltk natsort```
## Training
```
CUDA_VISIBLE_DEVICES=1 python train_fomile.py 
      --Transformation TPS --FeatureExtraction ResNet \ --SequenceModeling BiLSTM --Prediction Attn \
	--src_train_data data_lmdb_release/training \
	--src_select_data MJ-ST \
	--tar_train_data data_lmdb_release/validation \
	--tar_select_data real_data \
	--tar_batch_ratio 1.0 \
	--valid_data data_lmdb_release/evaluation/IC15_1811 \
	--batch_size 64 --lr 1 \
	--workers 16 \
	--num_iter 30000 \
	--continue_model pretrained/pretrained.pth \
	--init_portion 0.0 --add_portion 0.00005 \
	--loss_type FocalLoss --entropy_type FE \
	--src_lambda 1.0 --tar_lambda 1.0 \
	--focal_alpha 0.2 --focal_gamma 2 --loss_focal_gamma 2
```
## Evaluation
```
CUDA_VISIBLE_DEVICES=1 python test.py 
      --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
      --eval_data data_lmdb_release/evaluation \
      --benchmark_all_eval \
      --batch_size 128 \
      --workers 16 \
      --saved_model pretrained/pretrained.pth
```
## Acknowledgement
The project is based on [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark), [Seq2SeqAdapt](https://github.com/AprilYapingZhang/Seq2SeqAdapt) and [SMILE](https://github.com/timtimchang/SMILE)
