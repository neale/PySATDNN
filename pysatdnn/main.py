import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
from utils import timed, to_torch_tensor, generate_iid_samples
from smt_methods import create_smt_formula, solve
from sgd_training import eval_smt_dnn2, train_dnn_gd

import os
import time

from torch.utils.tensorboard import SummaryWriter


def load_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--verbose', action='store_true', help='toggle verbosity')
    parser.add_argument('--epochs', type=int, default=100, help='num passes through data of GD')
    parser.add_argument('--exp', default='')
    parser.add_argument('--n_examples', type=int, default=100)
    args = parser.parse_args()
    return args


def app(args):
    # sample data from the corners of the unit HyperCube
    n_layers = 2
    n_data = [args.n_examples]
    n_weight = [[2, 2], [4, 2], [8, 2], [16, 2], [32, 2]]
    data_dimension = [2, 4, 8, 16, 32]
    n_iters = 0

    os.makedirs(args.exp, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp)

    for n_weights, dim in zip(n_weight, data_dimension):
        for n_samples in n_data:
            print ('New SMT problem with {} weights, data dim {}, and {} training examples'.format(
                n_weights, dim, n_samples))

            data, labels = generate_iid_samples(dim, n_samples)
            data_test, labels_test = generate_iid_samples(dim, n_samples)
            bin_x = np.array([np.binary_repr(sample, width=dim) for sample in data])
            bin_x_test = np.array([np.binary_repr(sample, width=dim) for sample in data])
    
            if args.verbose:
                print ('generated data for dim={}: {}'.format(dim, list(zip(data, labels))))
            
            solve_start_time = time.time()
            smt_formula, weights, biases = create_smt_formula(bin_x, labels, dim, n_weights)
            
            size =  len(str(smt_formula).encode('utf-8'))
            print ('Formula Size: {} Bytes'.format(size))

            if args.verbose:
                print ('DNN [Boolean expression]: ', smt_formula)
            
            ret = solve(smt_formula, weights, biases, args)
            solver_time = time.time()-solve_start_time
            
            if ret == 0:
                continue
            else:
                weights, biases = ret
            xt, yt, wt, bt = to_torch_tensor(bin_x, labels, weights, biases)
            xt_test, yt_test, wt, bt = to_torch_tensor(bin_x_test, labels_test, weights, biases)

            y, y_hat = eval_smt_dnn2(x=xt, w=wt, b=bt, y=yt, layers=2)
            
            y_test, y_hat_test = eval_smt_dnn2(x=xt_test, w=wt, b=bt, y=yt_test, layers=2)
            
            if args.verbose:
                print ('Labels: {}\nPredictions: {}'.format(y, y_hat.view(-1)))
            print ('[Train] PySAT Classifier Accuracy: [{}/{}]'.format(torch.eq(y, y_hat.view(-1)).sum(), len(y)))
            print ('[Test] PySAT Classifier Accuracy: [{}/{}]'.format(torch.eq(y_test, y_hat_test.view(-1)).sum(), len(y_test)))
            print ('Training with gradient descent...')
            
            gd_start_time = time.time()
            correct, loss, correct_test = train_dnn_gd(xt, yt, xt_test, yt_test, epochs=args.epochs)      
            gd_time = time.time() - gd_start_time
            print ('[Train] Epoch: {}, Correct: [{}/{}], Loss: {}'.format(args.epochs, correct, len(yt), loss))
            print ('[Test] Correct: [{}/{}]'.format(correct_test, len(yt_test)))
            acc = correct/(len(yt))
            
            print('\n\n')
            n_iters += 1
            
            writer.add_scalar('Time/solver_time', solver_time, n_iters)
            writer.add_scalar('Formula/size', size, n_iters)
            writer.add_scalar('Time/gd_time', gd_time, n_iters)
            writer.add_scalar('Acc/accuracy', acc, n_iters)

if __name__ == '__main__':
    args = load_args()
    app(args)
