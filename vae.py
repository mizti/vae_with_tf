import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
#%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


class VariationalAutoencoder(object):
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        self._create_network()
        self._create_loss_optimizer()
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        #self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(var_list={
            "weights_recog_h1":self.network_weights["weights_recog"]['h1'],
            "weights_recog_h2":self.network_weights["weights_recog"]['h2'],
            "weights_recog_out_mean":self.network_weights["weights_recog"]['out_mean'],
            "weights_recog_out_log_sigma":self.network_weights["weights_recog"]['out_log_sigma'],
            
            "biases_recog_b1":self.network_weights["biases_recog"]['b1'],
            "biases_recog_b2":self.network_weights["biases_recog"]['b2'],
            "biases_recog_out_mean":self.network_weights["biases_recog"]['out_mean'],
            "biases_recog_out_log_sigma":self.network_weights["biases_recog"]['out_log_sigma'],

            "weights_gener_h1":self.network_weights["weights_gener"]['h1'],
            "weights_gener_h2":self.network_weights["weights_gener"]['h2'],
            "weights_gener_out_mean":self.network_weights["weights_gener"]['out_mean'],
            "weights_gener_out_log_sigma":self.network_weights["weights_gener"]['out_log_sigma'],

            "biases_gener_b1":self.network_weights["biases_gener"]['b1'],
            "biases_gener_b2":self.network_weights["biases_gener"]['b2'],
            "biases_gener_out_mean":self.network_weights["biases_gener"]['out_mean'],
            "biases_gener_out_log_sigma":self.network_weights["biases_gener"]['out_log_sigma']
        })
        self.sess.run(init)

    def _create_network(self):
        self.network_weights = self._initialize_weights(**self.network_architecture)
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(self.network_weights["weights_recog"], self.network_weights["biases_recog"])
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, dtype=tf.float32)

        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        self.x_reconstr_mean = \
            self._generator_network(self.network_weights["weights_gener"], self.network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_gener_1, n_hidden_gener_2, n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}

        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}

        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}

        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}

        return all_weights

    def _recognition_network(self,weights, biases):
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']), biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        x_reconstr_mean = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean']))
        return x_reconstr_mean
    
    def _create_loss_optimizer(self):
        reconstr_loss = -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean), 1)
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
    def partial_fit(self, X):
        #print(X.__class__) # <type 'numpy.ndarray'>
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        return cost
        #cost = self.sess.run(self.cost, feed_dict={self.x: X})
        #return cost

    def save(self, filename, global_step):
        save_path = self.saver.save(self.sess, filename, global_step=global_step)
        return save_path

    def restore(self, save_path):
        self.saver.restore(self.sess, save_path)

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=self.netowrk_architecutre["n_z"])
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        return self.sess.run(self.x_reconstr_mean, feed_dict={self.x:X})
