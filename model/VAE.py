from __future__ import division

import tensorflow as tf
import numpy as np
import sys

from src.models.base_models import conv_ae

"""
Class that builds the forward pass (variational) autoencoder model
"""

class conv_ae_model(conv_ae):

    def __init__(self, hyp, is_training):
        super(conv_ae_model, self).__init__(hyp)
        self.hyp_enc = hyp['encoder']
        self.hyp_dec = hyp['decoder']
        self.hyp_bot = hyp['bottleneck']
        self.data_format = hyp['train']['data_format']
        self.is_training = is_training

        self.shapes = None

    def forward_pass(self, x):

        if self.data_format == 'channels_first':
            # Data set format is N x H x W x D, transpose to N x D x H x W
            x = tf.transpose(x, perm=[0, 3, 1, 2])
        self.shapes = [x.get_shape().as_list()[1:]]

        # Encoder
        with tf.variable_scope(self.hyp_enc['scope']):
            x = tf.subtract(x, 0.5, name='standardize')
            for i in range(len(self.hyp_enc['conv']['filters'])):
                if self.hyp_enc['conv']['kernel_size'][i] is not None:
                    x = self.conv2d(self.hyp_enc['conv'], i, x)
                if 'pool' in self.hyp_enc:
                	x = self.max_pooling2d(self.hyp_enc['pool'], i, x)
                self.shapes.append(x.get_shape().as_list()[1:])
                if 'norm' in self.hyp_enc:
                	x = self.batch_normalization(self.hyp_enc['norm'], i, x, self.is_training)

            x = tf.layers.flatten(x, name='flatten')

            if len(self.hyp_enc['mlp']['d_layers']) > 0:
                x = self.mlp(self.hyp_enc['mlp'], x, self.is_training, out_act=True)

        # Bottleneck
        with tf.variable_scope(self.hyp_bot['scope']):
            if self.hyp_bot['variational']:
                mean, log_std = self.vae_bottleneck(self.hyp_bot, x, self.is_training)
                lat_sample = tf.add(mean,tf.multiply(tf.exp(log_std),tf.random_normal(tf.shape(mean))),name='latent_sample')
            else:
                mean = None
                log_std = None
                lat_sample = self.ae_bottleneck(self.hyp_bot, x, self.is_training)

        # Decoder
        with tf.variable_scope(self.hyp_dec['scope']):
            if len(self.hyp_dec['mlp']['d_layers']) == len(self.hyp_enc['mlp']['d_layers']):
                self.hyp_dec['mlp']['d_layers'].append(int(np.prod(self.shapes[-1])))
            x = self.mlp(self.hyp_dec['mlp'], lat_sample, self.is_training)

            x = tf.reshape(x, [-1] + self.shapes[-1])
            for i in range(len(self.hyp_dec['conv']['filters'])-1):
                if self.hyp_dec['conv']['kernel_size'][i] is not None:
                    x = self.conv2d(self.hyp_dec['conv'], i, x)
                    if 'norm' in self.hyp_dec:
                    	x = self.batch_normalization(self.hyp_enc['norm'], i, x, self.is_training)
                    if self.data_format == 'channels_first':
                        x = tf.transpose(x, perm=[0, 2, 3, 1])
                        new_size = self.shapes[len(self.shapes)-2-i][1:]
                    else:
                        new_size = self.shapes[len(self.shapes)-2-i][:2]
                    x = tf.image.resize_images(x, new_size, method=tf.image.ResizeMethod.BILINEAR)
                    if self.data_format == 'channels_first':
                        x = tf.transpose(x, perm=[0, 3, 1, 2])

            x = self.conv2d(self.hyp_dec['conv'], -1, x) # Last conv to compress to 1 channel
            if self.data_format == 'channels_first':
                # Transpose back to N x H x W x D to match true image
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            x = tf.add(x, 0.5)
            x = tf.sigmoid(x)

        return x, mean, log_std, lat_sample

    def encoder(self, x):

        if self.shapes is None:
            raise(ValueError('Shapes is not defined, run forward pass first to populate'))
            sys.exit()

        # Encoder
        with tf.variable_scope(self.hyp_enc['scope']):
            x = tf.subtract(x, 0.5, name='standardize')
            for i in range(len(self.hyp_enc['conv']['filters'])):
                if self.hyp_enc['conv']['kernel_size'][i] is not None:
                    x = self.conv2d(self.hyp_enc['conv'], i, x)
                if 'pool' in self.hyp_enc:
                    x = self.max_pooling2d(self.hyp_enc['pool'], i, x)
                self.shapes.append(x.get_shape().as_list()[1:])
                if 'norm' in self.hyp_enc:
                    x = self.batch_normalization(self.hyp_enc['norm'], i, x, self.is_training)

            x = tf.layers.flatten(x, name='flatten')

            if len(self.hyp_enc['mlp']['d_layers']) > 0:
                x = self.mlp(self.hyp_enc['mlp'], x, self.is_training, out_act=True)

        # Bottleneck
        with tf.variable_scope(self.hyp_bot['scope']):
            if self.hyp_bot['variational']:
                mean, log_std = self.vae_bottleneck(self.hyp_bot, x, self.is_training)
                lat_sample = tf.add(mean,tf.multiply(tf.exp(log_std),tf.random_normal(tf.shape(mean))),name='latent_sample')
            else:
                mean = None
                log_std = None
                lat_sample = self.ae_bottleneck(self.hyp_bot, x, self.is_training)

        return mean, log_std, lat_sample

    def decoder(self, x):

        if self.shapes is None:
            raise(ValueError('Shapes is not defined, run forward pass first to populate'))
            sys.exit()

        # Decoder
        with tf.variable_scope(self.hyp_dec['scope']):
            if len(self.hyp_dec['mlp']['d_layers']) == len(self.hyp_enc['mlp']['d_layers']):
                self.hyp_dec['mlp']['d_layers'].append(int(np.prod(self.shapes[-1])))
            x = self.mlp(self.hyp_dec['mlp'], x, self.is_training)

            x = tf.reshape(x, [-1] + self.shapes[-1])
            for i in range(len(self.hyp_dec['conv']['filters'])-1):
                if self.hyp_dec['conv']['kernel_size'][i] is not None:
                    x = self.conv2d(self.hyp_dec['conv'], i, x)
                    if 'norm' in self.hyp_dec:
                    	x = self.batch_normalization(self.hyp_dec['norm'], i, x, self.is_training)
                    if self.data_format == 'channels_first':
                        x = tf.transpose(x, perm=[0, 2, 3, 1])
                        new_size = self.shapes[len(self.shapes)-2-i][1:]
                    else:
                        new_size = self.shapes[len(self.shapes)-2-i][:2]
                    x = tf.image.resize_images(x, new_size, method=tf.image.ResizeMethod.BILINEAR)
                    if self.data_format == 'channels_first':
                        x = tf.transpose(x, perm=[0, 3, 1, 2])

            x = self.conv2d(self.hyp_dec['conv'], -1, x) # Last conv to compress to 1 channel
            if self.data_format == 'channels_first':
                # Transpose back to N x H x W x D to match true image
                x = tf.transpose(x, perm=[0, 2, 3, 1])
            x = tf.add(x, 0.5)
            x = tf.sigmoid(x)

        return x

    def tower_fn(self, features, labels, hyp, is_training):
        """
        Function for building a single computational tower (i.e. model, predictions, loss, gradients)

        Args:
            features: shard of features as input to a computational tower
            labels: corresponding shard of labels for loss calculation
            hyp: parsed dict of hyperparameters
            is_training: true if running training graph

        Returns:
            A tuple of the tower loss, tower gradients and trainable variables, and tower predictions
        """
        # model = mlp_ae_model(hyp, is_training)
        model = conv_ae_model(hyp, is_training)
        recon_image, latent_mean, latent_log_std, latent_sample = model.forward_pass(features) # Build the forward pass model
        tower_pred = {
            'reconstruction': recon_image,
            'latent_mean': latent_mean,
            'latent_log_std': latent_log_std,
            'latent_sample': latent_sample
        }

        # Get the total regularization loss based on what was passed to the kernel_regularizer argument
        # in the convolutional and dense layers. Equivalent to tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        reg_loss = tf.losses.get_regularization_loss()
        # tower_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)) + reg_loss

        flat_labels = tf.layers.flatten(labels, name='flat_labels')
        flat_recon_image = tf.layers.flatten(recon_image, name='flat_recon_image')

        bernoulli_nll = tf.reduce_mean(-tf.reduce_sum(tf.multiply(flat_labels, tf.log(tf.maximum(1e-9, flat_recon_image))) +
            tf.multiply(1-flat_labels, tf.log(tf.maximum(1e-9, 1-flat_recon_image))), axis=1), name='nll_loss')

        tower_loss = bernoulli_nll + reg_loss

        # tower_loss = tf.reduce_mean(tf.losses.mean_squared_error(flat_labels, flat_recon_image)) + reg_loss

        # Calculate KL divergence loss if latent mean and log std are returned
        if latent_mean is not None and latent_log_std is not None:
            d = latent_mean.get_shape().as_list()[1]
            targ_mean = tf.constant(np.zeros((1,d)),dtype=tf.float32)
            targ_log_std = tf.log(tf.constant(np.ones((1,d)),dtype=tf.float32))
            latent_var = tf.maximum(1e-9, tf.square(tf.exp(latent_log_std)))
            targ_var = tf.maximum(1e-9, tf.square(tf.exp(targ_log_std)))
            kl_loss = tf.reduce_mean((tf.reduce_sum(tf.log(targ_var), axis=1) - tf.reduce_sum(tf.log(latent_var), axis=1) - d
                + tf.reduce_sum(tf.divide(latent_var, targ_var), axis=1)
                + tf.reduce_sum(tf.divide(tf.square(targ_mean - latent_mean), targ_var), axis=1)) / 2., name='kl_loss')

            tower_loss = kl_loss + tower_loss

        model_params = tf.trainable_variables()
        tower_grads = tf.gradients(tower_loss, model_params)

        return tower_loss, zip(tower_grads, model_params), tower_pred
