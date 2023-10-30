# -*- coding: utf-8 -*-
import argparse
import os 

def parse_opts():

### Init 
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', default = os.path.abspath(os.path.dirname(__file__)), type=str, help='root path')
    parser.add_argument('--data_root', default='env', type=str, help='file path')

### MAML
    parser.add_argument('--epochs', default=400, type=int, help='epochs')
    parser.add_argument('--fast_lr', default=0.05, type=float, help='fast_lr')
    parser.add_argument('--meta_batch_size', default=16, type=int, help='meta_batch_size')
    parser.add_argument('--meta_lr', default=0.01, type=float, help='meta_lr')
    parser.add_argument('--maml_save', default='./checkpoints/ood/', type=str, help='MAML model save path')
    parser.add_argument('--maml_model', default='./checkpoints/ood/new_ep300', type=str, help='MAML model path')
    parser.add_argument('--predict_output', default='./output/ood/', type=str, help='predict_output')
    parser.add_argument('--exact_solution', default='./env/bob_info.csv', type=str, help='exact_solution')
    parser.add_argument('--ways', default=4, type=int, help='n-way')


    parser.add_argument('--mlp_model', default='./checkpoints/grid_search_b1.pkl', type=str, help='MLP model path')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')


    args = parser.parse_args()
    return args
