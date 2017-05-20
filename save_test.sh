import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import input_data
from vae import *
#%matplotlib inline

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

def train(vae, learning_rate=0.001, batch_size=100, training_epochs=10, display_step=1, save_interval = 1):
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
       # if epoch % save_interval == 0:
        #    print("saved: epoch" + str(epoch))
         #   vae.save('tmp/2212', global_step=epoch)
    return vae


vae = train(vae, training_epochs=5)
vae.save('tmp/05201542',0)
#vae.restore('tmp/1243')

x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)

plt.figure(figsize=(8, 12))

for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()


