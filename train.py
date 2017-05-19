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

def train(network_architecture, learning_rate=0.001, batch_size=100, training_epochs=10, display_step=5):
    vae = VariationalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
    #vae.restore('./tmp/vae_saved.dat')
    for epoch in range(training_epochs):
        print(epoch)
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = mnist.train.next_batch(batch_size)[0]
            cost = vae.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))

    return vae

network_architecture = \
    dict(n_hidden_recog_1 = 500,
         n_hidden_recog_2 = 500,
         n_hidden_gener_1 = 500,
         n_hidden_gener_2 = 500,
         n_input = 784,
         n_z = 20)

#vae = train(network_architecture, training_epochs=75)
vae = train(network_architecture, training_epochs=5)
print(vae.save('tmp/vae_saved.dat'))
