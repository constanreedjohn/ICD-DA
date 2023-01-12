for name in IC03_fine IIIT5k_3000_fine SVT_fine
do
	for exp in `ls /data/hungtt/KLTN/SMILE/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/fomile_hyper_optim_fl/stage_3/${name}`
	do
		CUDA_VISIBLE_DEVICES=1 python test.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
			--eval_data /data/hungtt/KLTN/SMILE/data_lmdb_release/evaluation \
			--benchmark_all_eval \
			--batch_size 128 \
			--workers 16 \
			--data_filtering_off \
			--saved_model /data/hungtt/KLTN/SMILE/saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111/fomile_hyper_optim_fl/stage_3/${name}/${exp}/best_accuracy.pth
	done
done