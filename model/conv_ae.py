from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""
This file contains a variety of base model classes which contain functions for creating NN layers
such as a fully connected layer, convolutional layer, etc.
"""


class conv_ae(object):
    def __init__(self, hyp):
        self.data_format = hyp['train']['data_format']

    def forward_pass(self, x, is_training):
        raise NotImplementedError('forward_pass is implemented in conv_autoencoder subclasses')

    def conv2d(self, hyp_conv, layer_num, x):
        if layer_num == -1 or layer_num == len(hyp_conv['filters']):
            activation=None
            name = 'conv2d_out'
        else:
            activation=tf.nn.relu
            name = 'conv2d_'+str(layer_num)
    	conv = tf.layers.conv2d(x, hyp_conv['filters'][layer_num], hyp_conv['kernel_size'][layer_num],
    				strides=hyp_conv['strides'][layer_num],
    				padding=hyp_conv['padding'],
    				data_format=self.data_format,
    				dilation_rate=hyp_conv['dilation_rate'][layer_num],
    				activation=activation,
    				kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    				bias_initializer=tf.zeros_initializer(),
    				kernel_regularizer=tf.contrib.layers.l2_regularizer(hyp_conv['lambda']),
                    reuse=tf.AUTO_REUSE,
    				name=name)
    	return conv

    def conv1d(self, hyp_conv, layer_num, x):
        if layer_num == -1 or layer_num == len(hyp_conv['filters']):
            activation=None
            name = 'conv1d_out'
        else:
            activation=tf.nn.relu
            name = 'conv1d_'+str(layer_num)
    	conv = tf.layers.conv1d(x, hyp_conv['filters'][layer_num], hyp_conv['kernel_size'][layer_num],
    				strides=hyp_conv['strides'][layer_num],
    				padding=hyp_conv['padding'],
    				data_format=self.data_format,
    				dilation_rate=hyp_conv['dilation_rate'][layer_num],
    				activation=activation,
    				kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    				bias_initializer=tf.zeros_initializer(),
    				kernel_regularizer=tf.contrib.layers.l2_regularizer(hyp_conv['lambda']),
                    reuse=tf.AUTO_REUSE,
    				name=name)
    	return conv

    def conv2d_transpose(self, hyp_conv_t, layer_num, x):
        conv_t = tf.layers.conv2d_transpose(x, hyp_conv_t['filters'][layer_num], hyp_conv_t['kernel_size'][layer_num],
            strides=hyp_conv_t['strides'][layer_num],
            padding=hyp_conv_t['padding'],
            data_format=self.data_format,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(hyp_conv_t['lambda']),
            reuse=tf.AUTO_REUSE,
            name='conv2d_t'+str(layer_num))
        return conv_t

    def max_pooling2d(self, hyp_pool, layer_num, x):
    	pool = tf.layers.max_pooling2d(x, hyp_pool['pool_size'][layer_num], hyp_pool['strides'][layer_num],
    		padding=hyp_pool['padding'],
    		data_format=self.data_format,
    		name='max_pool_2d_'+str(layer_num))

    	return pool

    def max_pooling1d(self, hyp_pool, layer_num, x):
    	pool = tf.layers.max_pooling1d(x, hyp_pool['pool_size'][layer_num], hyp_pool['strides'][layer_num],
    		padding=hyp_pool['padding'],
    		data_format=self.data_format,
    		name='max_pool_1d_'+str(layer_num))

    	return pool

    def batch_normalization(self, hyp_norm, layer_num, x, train_flag, axis=None):
        if axis is None:
            if self.data_format == 'channels_first':
                axis = 1
            else:
                axis = -1

    	return tf.layers.batch_normalization(x,
    		axis=axis,
    		momentum=hyp_norm['momentum'],
    		epsilon=hyp_norm['epsilon'],
    		training=train_flag,
    		reuse=tf.AUTO_REUSE,
    		name='batch_norm_'+str(layer_num))

    def mlp(self, hyp_mlp, x, train_flag, out_act=False):
        if 'dropout' in hyp_mlp:
            hyp_drop = hyp_mlp['dropout']
            assert(hyp_drop['in_rate'] >= 0), "Dropout rate cannot be less than zero"
            assert(hyp_drop['in_rate'] <= 1), "Dropout rate cannot be greater than one"
            assert(hyp_drop['hid_rate'] >= 0), "Dropout rate cannot be less than zero"
            assert(hyp_drop['hid_rate'] <= 1), "Dropout rate cannot be greater than one"
        if 'norm' in hyp_mlp:
            hyp_norm = hyp_mlp['norm']

        assert(len(hyp_mlp['d_layers'])) > 0, "MLP must have at least one layer"
        assert(all([type(hyp_mlp['d_layers'][i]) == int for i in range(len(hyp_mlp['d_layers']))])), "Layer sizes must be integers"

        outputs = x

    	if 'norm' in hyp_mlp:
    			outputs = self.batch_normalization(hyp_norm, 0, outputs, train_flag, axis=-1)
    	# Input dropout layer
    	if 'dropout' in hyp_mlp:
    		if hyp_drop['in_rate'] > 0:
    			# assert(hyp_drop['rate'] > 0), "Dropout rate must be greater than zero when input_dropout is set to True"
    			outputs = tf.layers.dropout(outputs, rate=hyp_drop['in_rate'], training=train_flag, name='input_dropout')

    	# Hidden layers
    	for i in range(len(hyp_mlp['d_layers'])-1):
    		outputs = tf.layers.dense(outputs, hyp_mlp['d_layers'][i],
    			activation=tf.nn.relu,
    			kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    			bias_initializer=tf.zeros_initializer(),
    			kernel_regularizer=tf.contrib.layers.l2_regularizer(hyp_mlp['lambda']),
    			reuse=tf.AUTO_REUSE,
    			name='hid_'+str(i)
    			)
    		if 'norm' in hyp_mlp:
    			outputs = self.batch_normalization(hyp_norm, i+1, outputs, train_flag, axis=-1)
    		if 'dropout' in hyp_mlp:
    			if hyp_drop['hid_rate'] > 0:
    				outputs = tf.layers.dropout(outputs, rate=hyp_drop['hid_rate'], training=train_flag, name='hid_'+str(i)+'_dropout')

        if out_act:
            activation=tf.nn.relu
        else:
            activation=None
    	outputs = tf.layers.dense(outputs, hyp_mlp['d_layers'][-1],
    		activation=activation,
    		kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            bias_initializer=tf.zeros_initializer(),
    		kernel_regularizer=tf.contrib.layers.l2_regularizer(hyp_mlp['lambda']),
    		reuse=tf.AUTO_REUSE,
    		name='out')

        return outputs

    def vae_bottleneck(self, hyp_vae, x, is_training):
        mean = tf.layers.dense(x, hyp_vae['d_layers'],
    		activation=None,
            use_bias=False,
    		kernel_initializer=tf.zeros_initializer(),
    		reuse=tf.AUTO_REUSE,
    		name='mean')
        log_std = tf.layers.dense(x, hyp_vae['d_layers'],
    		activation=None,
            use_bias=False,
    		kernel_initializer=tf.zeros_initializer(),
    		reuse=tf.AUTO_REUSE,
    		name='log_std')

        return mean, log_std

    def ae_bottleneck(self, hyp_ae, x, is_training):
        lat_coord = tf.layers.dense(x, hyp_ae['d_layers'],
    		activation=None,
            use_bias=False,
    		kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
    		reuse=tf.AUTO_REUSE,
    		name='latent_sample')

        return lat_coord
