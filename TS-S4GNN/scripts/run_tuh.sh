#!/bin/bash

RAW_DATA_DIR=<dir-to-resampled-signals>
PREPROC_DIR=<preproc-save-dir>
SAVE_DIR=<your-save-dir>

BATCH_SIZE=1
#output_dim 癫痫检测取1，分类取4
python train.py \
    --dataset 'tuh' \
    --raw_data_dir $RAW_DATA_DIR \
    --preproc_dir $PREPROC_DIR \
    --max_seq_len 60 \
    --num_nodes 19 \
    --input_dim 1 \
    --output_dim 4 \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --num_workers 8 \
    --adj_mat_dir 'data/eeg_electrode_graph/adj_mx_3d.pkl' \
    --model_name 'feature_fusion' \
    --graph_learn_metric "self_attention" \
    --dropout 0.1 \
    --g_conv 'gine' \
    --num_gcn_layers 2 \
    --hidden_dim 128 \
    --num_temporal_layers 4 \
    --state_dim 64 \
    --bidirectional False \
    --temporal_model 's4' \
    --temporal_pool 'mean' \
    --graph_pool 'max' \
    --activation_fn 'leaky_relu' \
    --prune_method 'thresh_abs' \
    --thresh 0.1 \
    --use_prior True \
    --metric_name 'auroc' \
    --save_dir $SAVE_DIR \
    --eval_metrics  'F1' 'precision' 'recall' 'acc' \
    --metric_avg 'weighted' \
    --lr_init 1e-4 \
    --l2_wd 1e-2 \
    --num_epochs 100 \
    --scheduler timm_cosine \
    --t_initial 100 \
    --warmup_t 10 \
    --optimizer adamw \
    --do_train True \
    --balanced_sampling True \
    --accumulate_grad_batches 2 \
    --gpus 1 \
    --regularizations 'feature_smoothing' 'degree' 'sparse' \
    --sparse_weight 0.05 \
    --feature_smoothing_weight 0.05 \
    --degree_weight 0.05 \
    --residual_weight 0.8 \
    --knn 2\
    --resolution 2000               #initial 2000
   