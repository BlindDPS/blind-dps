#!/bin/bash

python3 blind_deblur_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --reg_ord=1 \
    --reg_scale=1.0;
