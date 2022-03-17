import tensorflow as tf
import tensorflow_probability as tfpro
multi_gaussian = tfpro.python.distributions.MultivariateNormalDiag

from .attention import  Attention
from .ffnn import point_wise_feed_forward_network

class CVAE_attention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_modes, rate=0.1):
    super(CVAE_attention, self).__init__()

    #The number of modes to obtain from the CVAE
    self.num_modes = num_modes

    #This are the prior distribution parameters
    self.d_model = d_model
    self.prior_loc = tf.zeros(d_model)
    self.prior_scale_diag = tf.ones(d_model)
    self.prior = multi_gaussian(loc = self.prior_loc, scale_diag = self.prior_scale_diag)

    #Attention module and query vector to obtain loc and log_scale
    self.attention = Attention(d_model)
    self.average_query = tf.Variable(tf.ones(d_model), trainable = False)
    self.average_query = tf.expand_dims(tf.expand_dims(self.average_query, axis = 0), axis = 0)

    #Dense layers to obtain loc and log_scale
    self.loc_dense = tf.keras.layers.Dense(d_model)
    self.log_scale_dense = tf.keras.layers.Dense(d_model)

  def KL_Loss(self,mu,log_sigma):
    k = tf.shape(mu)[0]
    KL = tf.norm(mu)**2 
    KL += tf.math.reduce_sum(tf.math.exp(log_sigma))
    KL -= tf.cast(k, tf.float32)
    KL -= tf.math.reduce_sum(log_sigma)
    KL = KL/2
    return KL

  def call(self, x, training):
    # Input size: num_batch x sequence_length x d_model
    batch_size = tf.shape(x)[0]
    # sequence_size = tf.shape(x)[1]
    x , _ = self.attention(x, x, self.average_query, None)

    if training:
      #Obtain the location and the variance
      loc = self.loc_dense(x)
      log_sigma  = self.log_scale_dense(x)

      #Flatten to process simultaneously in a multi gaussian distribution
      loc_flat = tf.reshape(loc, [-1])
      log_sigma_flat = tf.reshape(log_sigma, [-1])

    else:
      #Obtain the flat location and variance
      loc_flat = tf.zeros(batch_size *self.d_model)
      log_sigma_flat = tf.ones(batch_size *self.d_model)

    #Obtain the posterior
    posterior = multi_gaussian(loc = loc_flat, scale_diag = log_sigma_flat)
    
    #Obtain the samples
    samples = posterior.sample(self.num_modes) # num_modes x (batch_size x d_model)

    #Reshape to obtain compatible tensor
    samples = tf.reshape(samples, [self.num_modes, batch_size, self.d_model])
    x = tf.transpose(samples, perm = [1,0,2])
    x = tf.reshape(x, [self.num_modes, batch_size, 1, self.d_model])

    #return output and value for the modified loss
    aux = self.KL_Loss(loc_flat, log_sigma_flat)
    return x, aux