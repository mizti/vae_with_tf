import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
%matplotlib inline

network_architecture = \
    dict(n_hidden_recog_1 = 500,
         n_hidden_recog_2 = 500,
         n_hidden_gener_1 = 500,
         n_hidden_gener_2 = 500,
         n_input = 784,
         n_z = 20)
learning_rate=0.001
batch_size=100
training_epochs=10
display_step=5

vae = VariationalAutoencoder(network_architecture, learning_rate=learning_rate, batch_size=batch_size)
vae.restore('./tmp/vae_saved.dat')

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
