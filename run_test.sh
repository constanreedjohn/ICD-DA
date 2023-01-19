CUDA_VISIBLE_DEVICES=1 python test.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
	--eval_data /data/hungtt/KLTN/SMILE/data_lmdb_release/evaluation \
	--benchmark_all_eval \
	--batch_size 128 \
	--workers 16 \
	--data_filtering_off \
	--saved_model /data/hungtt/FOMILE/best_accuracy.pth