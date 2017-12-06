import os
import torchvision
import torch
import argparse
from preprocessing import get_category, get_imagepath
from data_loader import get_loader
from torch.backends import cudnn
from solver import Solver

def main(config):
    cudnn.benchmark = True

    data_loader = get_loader(image_path=config.image_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers)
    
    solver = Solver(config, data_loader)
    
    for x in data_loader:
        print(x)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=200)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    
    # training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam
    
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--image_path', type=str, default='./dataset/wiki_crop/00')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_step', type=int , default=500)
    config = parser.parse_args()
    print(config)
    main(config)
    