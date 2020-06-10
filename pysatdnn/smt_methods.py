######################################
# smt_methods.py
# main core for evaluting the 2 node neural network
# given training data, finds a satisfying assignment to the formula
# uses the Z3 solver to get weights from the formula
######################################
from pysmt.shortcuts import Symbol, LE, GE, And, Int, Real, Equals, Plus, Bool
from pysmt.shortcuts import Solver, Not, is_sat, get_model, Iff, Times
from pysmt.typing import BOOL, REAL, INT
import numpy as np
import argparse
from utils import timed

def create_smt_formula(data, labels, dim, n_weights):
    weights = []
    biases = []
    weight_symbols = [Symbol('weight_{}'.format(i), REAL) for i in range(n_weights[0])]
    weight_symbols2 = Symbol('weight_out', REAL)
    weights.append(weight_symbols)
    weights.append([weight_symbols2])

    bias_symbol1 = Symbol('bias_{}'.format(1), REAL)
    bias_symbol2 = Symbol('bias_{}'.format(2), REAL)
    biases.append([bias_symbol1])
    biases.append([bias_symbol2])

    layer_domains = []
    layer_input = data
    for i in range(len(layer_input)):  # loop over data
        # \sum <w_ji,  x_i> + b_i
        g = len(weight_symbols)//2
        # Layer 1
        weight_input1_i = Plus([Times(w_i, Real(int(x_j)))
            for w_i, x_j in zip(weight_symbols, layer_input[i])])
        prod_bias1_i = Plus(weight_input1_i, bias_symbol1)
        # Layer 2
        weight_input2_i = Plus(Times(prod_bias1_i, weight_symbols2))
        prod_bias2_i = Plus(weight_input2_i, bias_symbol2)
        # output
        weight_output = prod_bias2_i
        layer_domain = Equals(weight_output, Real(labels[i]))
        layer_domains.append(layer_domain)

    network_domain = And(x for x in layer_domains)
    dnn_problem = network_domain
    return dnn_problem, weights, biases


def solve(formula, weight_symbols, bias_symbols, args):
    with Solver(name='z3') as solver:
        solver.add_assertion(formula)
        satisfiable = solver.is_sat(formula)
        if satisfiable:
            print ('DNN [Is Satisfiable]: ', satisfiable)
        else:
            print ('DNN is UNSAT')
            return 0
        if solver.solve():
            # extract_values
            if args.verbose:
                print (solver.get_model())
            weights = []
            biases = []
            for weight_set, bias_set in zip(weight_symbols, bias_symbols):
                weights.append([float(solver.get_py_value(w)) for w in weight_set])
                biases.append([float(solver.get_py_value(b)) for b in bias_set])

    return (weights, biases)

