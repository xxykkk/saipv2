#!/bin/bash
#SBATCH -J saipv2
#SBATCH -p L40
#SBATCH -N 1

#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:l40:8

#SBATCH --cpus-per-task=7

#SBATCH --mail-type=end
#SBATCH --mail-user=2500049201@qq.com
#SBATCH --output=slurm_logs/kd_sapiens-%j.out
#SBATCH --error=slurm_logs/kd_sapiens-%j.err


NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
NPROC_PROCESS=$SLURM_CPUS_PER_TASK
NUM_WORKERS=$(((NPROC_PROCESS - 1)*2))


module load cuda/11.8
source /share/home/u24011/software/miniconda3/etc/profile.d/conda.sh
conda activate pretrain

echo "The value of NPROC_PER_NODE: $NPROC_PER_NODE"
echo "The value of NPROC_PROCESS: $NPROC_PROCESS"
echo "The value of NUM_WORKERS: $NUM_WORKERS"

DEBUG_MODE=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1; shift ;;
        *) echo "Unknown parameter: $1"; shift ;;
    esac
done

output_dir="work_dirs/lup1m_sapiens-l_to_vit_tiny_hw256_cp128_cls"


if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
script_path=$(realpath "$0")
cp "$script_path" "$output_dir/"
cp "main_align_pretrain_LUP1M_sapiens.py" "$output_dir/"
echo "start training"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29601 main_align_pretrain.py \
  --batch_size=256 --accum_iter=1 \
  --model=saipv1_kd_vit_tiny_patch16_adapt_vit_l \
  --data_path=data/LUP1M \
  --norm_pix_loss \
  --mask_ratio=0.75 \
  --epochs=200 \
  --blr=2.5e-4 \
  --weight_decay=0.05 \
  --warmup_epochs=10 \
  --height=256 \
  --width=256 \
  --crop_height=128 \
  --crop_width=128 \
  --global_crops_scale 0.8 1. \
  --local_crops_scale 0.05 0.8 \
  --output_dir=$output_dir \
  --log_dir=$output_dir \
  --teacher_model=expert_sapiens_0_3b \
  --teacher_pretrained='pretrained_models/sapiens_0.3b_epoch_1600_clean.pth' \
  --local_crops_number=6 \
  --start_epoch=0 \
  --num_workers=$NUM_WORKERS \
  