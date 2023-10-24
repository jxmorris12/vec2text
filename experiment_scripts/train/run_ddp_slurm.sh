#!/bin/zsh
#SBATCH --partition=rush,gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=672:00:00
#SBATCH --requeue

export BASEDIR="/home/jxm3/research/retrieval/inversion/vec2text/"
export NCCL_P2P_LEVEL=NVL

echo "hostname:"
hostname
echo "nvidia-smi"
nvidia-smi
echo "running df -h TMPDIR ($TMPDIR).."
df -h $TMPDIR
echo "start:: running command with BASEDIR=$BASEDIR"
cd $BASEDIR
pwd
/home/jxm3/.conda/envs/torch/bin/torchrun --master_port 0 --nproc_per_node 4 --rdzv-backend=c10d --rdzv-endpoint=localhost:0  "$BASEDIR/run.py" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --max_seq_length 128 --model_name_or_path t5-base --dataset_name msmarco --embedder_model_name "sentence-transformers/all-MiniLM-L6-v2" --num_repeat_tokens 16 --embedder_no_grad True --num_train_epochs 100 --max_eval_samples 500 --eval_steps 20000 --warmup_steps 10000 --bf16=1 --use_wandb=1 --use_frozen_embeddings_as_input True --experiment inversion --lr_scheduler_type constant_with_warmup --exp_group_name oct-sbert --learning_rate 0.001 --output_dir "$BASEDIR/saves/gtr-1" --save_steps 2000

echo "finished:: running command with BASEDIR=$BASEDIR"
