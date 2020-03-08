This is software to train the RNN models on the J1-J2 heisenberg described in LINK PAPER

Train a model from scrach

Type into the command line:

python3 code.py [num_sites] [batch_size] [learning rate] [J2]

where code.py is the software (either 1D or 2D) num sites is the length of the side of a lattice, batch size is the number of samples per batch, and J2 represents J2/J1 the next nearest neighbor couplinng

Example:
python3 2d_heisenberg.py 4 100 1e-4 0.0
to train a 4x4 model from scratch

You can play with other hyperparameters within the code

This code will output 3 files labeled "loss", "info", "model". "loss" records observables averaged over 100 minibatches whereas "info" records observables over single minibatch. 'model' is the saved paraaamters of the model. 

Iterative Training:
When you train a model this code will save the model every 100 minibatches. If you want to generalize to a larger model set [num_sites] = larger_model_size and load the previous with the NN.load_state_dict command. 



