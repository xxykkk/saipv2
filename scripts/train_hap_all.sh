#!/bin/bash




# module load cuda/11.8
# module load nccl
# source /share/home/u24011/software/miniconda3/etc/profile.d/conda.sh
# conda activate pretrain

## train config
NPROC_PER_NODE=8
NPROC_PROCESS=9
NUM_WORKERS=$(((NPROC_PROCESS - 1)))
export CUDA_VISIBLE_DEVICES='0,1,2,3,5,6,7,8'

# # test config
# NPROC_PER_NODE=1
# NPROC_PROCESS=3
# NUM_WORKERS=$(((NPROC_PROCESS - 1)))
# export CUDA_VISIBLE_DEVICES='4'


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

output_dir="work_dirs/hap-b_to_vit_tiny_cls_patch_atten"


if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi
script_path=$(realpath "$0")
cp "$script_path" "$output_dir/"
cp "main_align_pretrain.py" "$output_dir/"
echo "start training"

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=29999 main_align_pretrain.py \
  --batch_size=384 --accum_iter=1 \
  --model=saipv1_kd_vit_tiny_patch16_adapt_vit_b \
  --data_path=data/LUP1M \
  --norm_pix_loss \
  --mask_ratio=0.75 \
  --epochs=200 \
  --blr=2.5e-4 \
  --weight_decay=0.05 \
  --warmup_epochs=10 \
  --height=256 \
  --width=128 \
  --crop_height=128 \
  --crop_width=64 \
  --global_crops_scale 0.8 1. \
  --local_crops_scale 0.05 0.8 \
  --output_dir=$output_dir \
  --log_dir=$output_dir \
  --teacher_model=expert_vit_base_hap \
  --teacher_pretrained='pretrained_models/hap_official_state_dict_exclude_decoder.pth' \
  --local_crops_number=6 \
  --start_epoch=0 \
  --num_workers=$NUM_WORKERS \
  