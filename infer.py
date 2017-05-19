import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
#import vae
from vae import VariationalAutoencoder

np.random.seed(66)
tf.set_random_seed(666)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_samples = mnist.train.num_examples

network_architecture = \
    dict(n_hidden_recog_1 = 500,
         n_hidden_recog_2 = 500,
         n_hidden_gener_1 = 500,
         n_hidden_gener_2 = 500,
         n_input = 784,
         n_z = 20)
learning_rate=0.001
batch_size=100

vae = VariationalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
vae.restore('./tmp/vae_saved.dat')

