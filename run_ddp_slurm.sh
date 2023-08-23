#!/bin/bash

#SBATCH -e logs/2023-06-26--long-gpu-test-%J.err
#SBATCH -o logs/2023-06-26--long-gpu-test-%J.out

#SBATCH --partition=rush
#SBATCH --gres=gpu:a6000:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=672:00:00



###
export NCCL_P2P_LEVEL=NVL
/home/jxm3/.conda/envs/torch/bin/torchrun --master_port 0 --nproc_per_node 8 --rdzv-backend=c10d --rdzv-endpoint=localhost:0  run.py --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --max_seq_length 128 --model_name_or_path t5-base --embedder_model_name gtr_base --num_repeat_tokens 16 --embedder_no_grad True --freeze_strategy none --embedder_fake_with_zeros False --encoder_dropout_disabled False --decoder_dropout_disabled False --use_less_data -1 --num_train_epochs 200 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 100000 --bf16=1 --use_lora=0 --use_wandb=1 --embedder_model_api text-embedding-ada-002 --use_frozen_embeddings_as_input True --corrector_model_alias openai_msmarco__msl128__100epoch --experiment corrector_encoder --lr_scheduler_type constant_with_warmup --exp_group_name jul11-openai-msl128-corrector-ddp-full-8gpu-long-slurm-ddp-2 --learning_rate 0.0001 --resume_from_checkpoint saves/6702902bea591e5ac3268a61d48baa6a/checkpoint-1072000/ 
