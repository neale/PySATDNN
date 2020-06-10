#####################################
# one_layer_dnn.py
# Quick example for a simple one layer neural network
# standalone example, not meant to be run with the rest of the package
####################################
import torch
import torch.nn.functional as F
import torch.nn as nn
from pysmt.shortcuts import Symbol, LE, GE, And, Int, Real, Equals, Plus, Bool
from pysmt.shortcuts import Solver, Not, is_sat, get_model, Iff, Times
from pysmt.typing import BOOL, REAL, INT
import numpy as np
import argparse

def load_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--verbose', action='store_true', help='toggle verbosity')
    args = parser.parse_args()
    return args


class BoolDNN(nn.Module):
    def __init__(self, d_in, d_out):
        super(BoolDNN, self).__init__()
        self.linear1 = nn.Linear(d_in, 10)
        self.linear2 = nn.Linear(10, d_out)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

def train_dnn_gd(x, y):
    model = BoolDNN(x.shape[1], 2)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for epoch in range(1000):
        output = model(x)
        loss = F.cross_entropy(output, y.long())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.float().eq(y.float().view_as(pred)).sum().item()
        total = len(y)
    print ('Epoch: {}, Correct: [{}/{}], Loss: {}'.format(epoch, correct, total, loss))


def eval_smt_dnn(x, w, b, y):
    x = [[float(char) for char in item] for item in x]
    x = torch.tensor(x)
    w = torch.tensor(w).unsqueeze(0)
    b = torch.tensor(b).unsqueeze(0)
    y = torch.tensor(y).float()
    print ('Data shape: {}, {}'.format(x.shape, y.shape))
    print ('Parameters shape: {}, {}'.format(w.shape, b.shape))
    y_hat = F.linear(x, w, bias=b).view(-1)
    print ('PySAT Classifier Accuracy: [{}/{}]'.format(torch.eq(y, y_hat).sum(), len(y)))
    
    print ('Training with gradient descent...')
    train_dnn_gd(x, y)
    print()


def generate_iid_samples(dim, n_samples, unsat=False):
    upper_bound = 2**dim
    samples = np.random.randint(0, upper_bound, n_samples)
    samples = [sample.item() for sample in samples]
    # labels assigned to half of the cube
    if unsat:
        y = [1 if np.random.uniform() < .5 else 0 for sample in samples]
    else:
        y = [1 if sample < (upper_bound/2) else 0 for sample in samples]
    return samples, y 


def app(args):
    # sample data from the corners of the unit HyperCube
    n_layers = 1
    n_data = [10, 100, 1000, 10000, 100000]
    n_weight = [2, 4, 8, 16, 32]
    data_dimension = [2, 4, 8, 16, 32]#, 3, 4]#, 8, 16, 32, 64, 128, 256, 512, 1024]
    for n_weights, dim in zip(n_weight, data_dimension):
        for n_samples in n_data:
            data, labels = generate_iid_samples(dim, n_samples)
            if args.verbose:
                print ('generated data for dim={}: {}'.format(dim, list(zip(data, labels))))
            
            training_symbols = []
            label_symbols = []
            # Create symbolic representation of data
            training_symbols = [Symbol('input_{}'.format(i), BOOL) for i in range(len(data))]
            label_symbols = [Symbol('label_{}'.format(i), REAL) for i in range(len(labels))]
            
            bin_x = np.array([np.binary_repr(sample, width=dim) for sample in data])
            layers = []
            layer_input = bin_x
            weights = []
            for layer in range(n_layers):
                
                weight_symbols = [Symbol('weight_{}'.format(i), REAL) for i in range(n_weights)]
                weights.extend(weight_symbols)
                bias_symbol = Symbol('bias_{}'.format(layer), REAL)
                
                layer_domains = []
                for i in range(len(layer_input)):  # loop over data
                    # \sum <w_ji,  x_i> + b_i
                    weight_input_i = Plus([Times(w_j, Real(int(x_j))) for w_j, x_j in zip(weight_symbols, layer_input[i])])
                    prod_bias_i = Plus(weight_input_i, bias_symbol)

                    layer_domain = Equals(prod_bias_i, Real(labels[i]))

                    layer_domains.append(layer_domain)
                layer = And(x for x in layer_domains)
                layers.append(layer)
            
            network_domain = And([x for x in layers])  
            dnn_problem = network_domain
            if args.verbose:
                print ('DNN [Boolean expression]: ', dnn_problem)
            print ('DNN [Satisfiable?]: ', is_sat(dnn_problem))
            if is_sat(dnn_problem):
                model = get_model(dnn_problem)
                print (model)
            else:
                print ('[DNN] not satisfiable')
                break

            with Solver(name='msat') as solver:
                solver.add_assertion(dnn_problem)
                if solver.solve():
                    # extract_values
                    weights = [float(solver.get_py_value(w)) for w in weights]
                    bias = float(solver.get_py_value(bias_symbol))
                
                # evaluate model on generated weights
                eval_smt_dnn(x=bin_x, w=weights, b=bias, y=labels)
                
            
        

if __name__ == '__main__':
    args = load_args()
    app(args)
