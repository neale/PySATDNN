# PySATDNN
A tool for solving a 3 node neural network with PySMT
===================================

This repository contains the source code for a tool that finds optimal weights for a 3 node neural network architecture. 

The training data is drawn from the unit-hypercube, and the labels are assigned to be binary. 

We use PySMT with the z3 solver to find both a CNF formula and a satisfying assignment for the model. 

The tool records the following quantities using tensorboard:

* Formula size (in bytes)

* Time (s) to extract formula and assignment

* Accuracy of assigned weights on the training data

* Comparative accuracy of gradient descent weights on the same data

* Time (s) to find a solution on the same data with gradient descent

## Requirements
* pytorch
* pysmt -- with z3 solver
* numpy
* tensorboard
* python3 
* probably ubuntu

## Usage
First clone the repository to the home directory (or anywhere you prefer)

`git clone https://github.com/neale/pysatdnn . && cd pysatdnn`

Its helpful to instantiate a virtual environment, to avoid cluttering the system

`python 3 -m venv pysat_venv`

Activate the virtual environment to start from a clean slate

`source ./pysat_venv/bin/activate`

Install the required packages for PySATDNN

`pip3 install -r requirements.txt`

If there is an error, you might need to update pip `pip3 install --upgrade pip`

Now that you have the required packages, we depend on the z3 solver for pysmt

`pysmt-install --z3` [y]

Now you're free to check out the source code, or run the test suite with `./start_all.sh`, which will loop over data with 2 to 32 dimensions, on training set sizes 100 to 100k


## Issues
If you get the error `NoSolverAvailable: No Solver is Available`, then do the following:

Run `pysmt-install --check`

The output should show that z3 is installed as a solver. If not, then install as shown above. 

If it is shown as installed, then the PYTHONPATH is likely pointing somewhere else.

Run `pysmt-install --env` to obtain a string that you can run to temporarily update the PYTHONPATH to the solver's location




