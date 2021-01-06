
import argparse

import torch
import torch.optim as optim
from thop import profile
from thop import clever_format
from models import DPTNet_base
from others.optimizer_dptnet import TransformerOptimizer
from others.data import AudioDataset, AudioDataLoader, EvalDataset, EvalDataLoader

from utils import device

from solver import Solver

parser = argparse.ArgumentParser( "DPTNet")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default='/data/WSJ/wsj_data/min/tr',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default='/data/WSJ/wsj_data/min/cv',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_json', type=str, default='/data/WSJ/wsj_data/min/cv/mix.json',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4.0, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8.0, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=64, type=int,
                    help='Dim of feature to the DPRNN')
parser.add_argument('--W', default=2, type=int,
                    help='Filter lenght in encoder, or the length of window in samples')
parser.add_argument('--K', default=250, type=int,
                    help='Chunk size in frames')
parser.add_argument('--D', default=6, type=int,
                    help='Number of DPRNN blocks')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--E', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block, dim of feature to the DPRNN')
parser.add_argument('--H', default=128, type=int,
                    help='Number of hidden units in each direction of RNN')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=1, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--validate', dest='validate', default=1, type=int,
                    help='Enables validation iteration of model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')

def main(args):
    # data
    tr_dataset = AudioDataset(args.train_dir, batch_size=1,
                              sample_rate=args.sample_rate, segment=args.segment)
    cv_dataset = AudioDataset(args.valid_dir, batch_size=args.batch_size,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1, shuffle=False)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    
    model = DPTNet_base(enc_dim=256, feature_dim=64, hidden_dim=128, layer=6, segment_size=250, nspk=2, win_len=2)
    
    print(model)
    if args.use_cuda:
        # model = torch.nn.DataParallel(model)
        model.cuda()
        model.to(device)
    
    optimizier = TransformerOptimizer(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), k=0.2, d_model=64, warmup_steps=4000)

    solver = Solver(data, model, optimizier, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    # print(args)
    main(args)