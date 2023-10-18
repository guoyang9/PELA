exp_name="deit3_large_pela"
output_dir=/path/to/output_dir/${exp_name}
dataset_dir=/temp/guangzhi/datasets/imagenet
teacher_ckpt="params/deit_3_large_224_1k.pth"
mkdir -p ${output_dir}

python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 34567 main.py \
    --model deit_large_patch16_LS \
    --data-path  ${dataset_dir} \
    --batch 16 \
    --epochs 20 \
    --weight-decay 0.1 \
    --sched cosine \
    --input-size 224 \
    --eval-crop-ratio 1.0 \
    --reprob 0.0 \
    --smoothing 0.1 \
    --warmup-epochs 1 \
    --drop 0.0 \
    --seed 0 \
    --opt adamw \
    --mixup .8 \
    --drop-path 0.45 \
    --cutmix 1.0 \
    --unscale-lr \
    --aa rand-m9-mstd0.5-inc1 \
    --no-repeated-aug \
    --compression_ratio 2.0 \
    --scale_lr 1 \
    --distillation 1 \
    --kd_alpha 1.0 \
    --resume_teacher ${teacher_ckpt} \
    --output_dir ${output_dir}