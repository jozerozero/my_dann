import torch
import torch.optim as optim
import torch.nn as nn
import model
import transform as tran
import adversarial1 as ad
import numpy as np
from read_data import ImageList
import argparse
import os
import torch.nn.functional as F


torch.set_num_threads(1)
parser = argparse.ArgumentParser(description='PyTorch BSP Example')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--src', type=str, default='A', metavar='S',
                    help='source dataset')
parser.add_argument('--tgt', type=str, default='C', metavar='S',
                    help='target dataset')
parser.add_argument('--num_iter', type=int, default=50002,
                    help='max iter_num')
args = parser.parse_args()

def get_datasetname(args):
    noe = 0
    office = False
    A = './data/Art.txt'
    C = './data/Clipart.txt'
    P = './data/Product.txt'
    R = './data/Real_World.txt'
    if args.src == 'A':
        src = A
    elif args.src == 'C':
        src = C
    elif args.src == 'P':
        src = P
    elif args.src == 'R':
        src = R
    if args.tgt == 'A':
        tgt = A
    elif args.tgt == 'C':
        tgt = C
    elif args.tgt == 'P':
        tgt = P
    elif args.tgt == 'R':
        tgt = R
    a = './data/amazon.txt'
    w = './data/webcam.txt'
    d = './data/dslr.txt'
    if args.src == 'a':
        src = a
        office = True
    elif args.src == 'w':
        src = w
        office = True
    elif args.src == 'd':
        src = d
        noe = noe + 1
        office = True
    if args.tgt == 'a':
        tgt = a
        noe = noe + 1
    elif args.tgt == 'w':
        tgt = w
    elif args.tgt == 'd':
        tgt = d
    return src, tgt, office,noe


src, tgt, office, noe = get_datasetname(args)

batch_size = {"train": 36, "val": 36, "test": 4}














