exp_name="deit_base_pela"
output_dir=/path/to/output/${exp_name} 
dataset_dir=/path/to/dataset # path to imagenet dataset
teacher_ckpt=params/deit_base_patch16_224-b5f2ef4d.pth 
mkdir -p ${output_dir}

python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 23456 main.py \
    --model deit_base_patch16_224 \
    --batch-size 64 \
    --epochs 50 \
    --warmup-epoch 1 \
    --scale_lr 1 \
    --data-path ${dataset_dir} \
    --distillation 1 \
    --compression_ratio 2 \
    --kd_alpha 1.0 \
    --kd_beta 0.0 \
    --output_dir ${output_dir} \
    --resume_teacher ${teacher_ckpt}
  
