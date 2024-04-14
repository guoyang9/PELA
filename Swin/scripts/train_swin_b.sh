dataset_dir=/path/to/dataset
teacher_ckpt=checkpoints/swin_base_patch4_window7_224.pth

python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--cfg configs/swin/swin_base_patch4_window7_224.yaml \
--data-path ${dataset_dir} \
--batch-size 64 \
--distillation 1 \
--compression_ratio 1.5 \
--kd_alpha 1 \
--inf_norm_weight 1 \
--opts MODEL.NAME swin_base_pela_1.5 \
--resume_teacher ${teacher_ckpt}

