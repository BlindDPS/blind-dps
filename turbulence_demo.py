import os
import argparse
import yaml
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from functools import partial

from guided_diffusion.blind_condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_operator, get_noise
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import Blurkernel, clear_color, generate_tilt_map
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    # Configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_model_config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--kernel_model_config', type=str, default='configs/kernel_model_config.yaml')
    parser.add_argument('--tilt_model_config', type=str, default='configs/tilt_model_config.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--task_config', type=str, default='configs/turbulence_config.yaml')
    # Training
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    # Regularization
    parser.add_argument('--reg_scale', type=float, default=0.0)
    parser.add_argument('--reg_ord', type=int, default=0, choices=[0, 1])

    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    img_model_config = load_yaml(args.img_model_config)
    kernel_model_config = load_yaml(args.kernel_model_config)
    tilt_model_config = load_yaml(args.tilt_model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # configs to namespace save space
    args.kernel = task_config["kernel"]
    args.kernel_size = task_config["kernel_size"]
    args.intensity = task_config["intensity"]
   
    # Load model
    img_model = create_model(**img_model_config)
    img_model = img_model.to(device)
    img_model.eval()
    kernel_model = create_model(**kernel_model_config)
    kernel_model = kernel_model.to(device)
    kernel_model.eval()
    tilt_model = create_model(**tilt_model_config)
    tilt_model = tilt_model.to(device)
    tilt_model.eval()
    model = {'img': img_model, 'kernel': kernel_model, 'tilt': tilt_model}

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
    measurement_cond_fn = cond_method.conditioning

    # We will not use regularization. So skip the part.
    
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    logger.info(f"work directory is created as {out_path}")
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # set seed for reproduce
    np.random.seed(123)
    
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        
        # blur
        conv = Blurkernel('gaussian', kernel_size=args.kernel_size, device=device)
        kernel = conv.get_kernel().type(torch.float32)
        kernel = kernel.to(device).view(1, 1, args.kernel_size, args.kernel_size)

        # tilt
        img_size = img_model_config["image_size"]
        # tile_map requires loop for generation that could be slow.
        tilt = generate_tilt_map(img_h=img_size, img_w=img_size, kernel_size=7, device=device)
        tilt = torch.clip(tilt, -2.5, 2.5)
        
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img, kernel, tilt)
        y_n = noiser(y)
        
        # Set initial sample 
        # !All values will be given to operator.forward(). Please be aware it.
        x_start = {'img': torch.randn(ref_img.shape, device=device).requires_grad_(),
                   'kernel': torch.randn(kernel.shape, device=device).requires_grad_(),
                   'tilt': torch.randn(tilt.shape, device=device).requires_grad_()}
        
        # !prior check: keys of model (line 74) must be the same as those of x_start to use diffusion prior.
        for k in x_start:
            if k in model.keys():
                logger.info(f"{k} will use diffusion prior")
            else:
                logger.info(f"{k} will use uniform prior.")
       
        # sample 
        sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', 'ker_'+fname), clear_color(kernel))
        plt.imsave(os.path.join(out_path, 'label', 'img_'+fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', 'img_'+fname), clear_color(sample['img']))
        plt.imsave(os.path.join(out_path, 'recon', 'ker_'+fname), clear_color(sample['kernel']))

if __name__ == '__main__':
    main()
