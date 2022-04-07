import tensorflow as tf
import tensorflow_probability as tfp
multi_gaussian = tfp.distributions.MultivariateNormalDiag

from .attention import  Attention
from .ffnn import point_wise_feed_forward_network

# class CVAE_attention_1(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_modes, rate=0.1):
#     super(CVAE_attention, self).__init__()

#     #The number of modes to obtain from the CVAE
#     self.num_modes = num_modes

#     #This are the prior distribution parameters
#     self.d_model = d_model
#     self.prior_loc = tf.zeros(d_model)
#     self.prior_scale_diag = tf.ones(d_model)
#     self.prior = multi_gaussian(loc = self.prior_loc, scale_diag = self.prior_scale_diag)

#     #Attention module and query vector to obtain loc and log_scale
#     self.attention = Attention(d_model)
#     self.average_query = tf.Variable(tf.ones(d_model), trainable = False)
#     self.average_query = tf.expand_dims(tf.expand_dims(self.average_query, axis = 0), axis = 0)

#     #Dense layers to obtain loc and log_scale
#     self.loc_dense = tf.keras.layers.Dense(d_model)
#     self.log_scale_dense = tf.keras.layers.Dense(d_model)

#   def KL_Loss(self,mu,log_sigma):
#     k = tf.shape(mu)[0]
#     KL = tf.norm(mu)**2 
#     KL += tf.math.reduce_sum(tf.math.exp(log_sigma))
#     KL -= tf.cast(k, tf.float32)
#     KL -= tf.math.reduce_sum(log_sigma)
#     KL = KL/2
#     return KL

#   def call(self, x, training):
#     # Input size: num_batch x sequence_length x d_model
#     batch_size = tf.shape(x)[0]
#     # sequence_size = tf.shape(x)[1]
#     x , _ = self.attention(x, x, self.average_query, None)

#     if training:
#       #Obtain the location and the variance
#       loc = self.loc_dense(x)
#       log_sigma  = self.log_scale_dense(x)

#       #Flatten to process simultaneously in a multi gaussian distribution
#       loc_flat = tf.reshape(loc, [-1])
#       log_sigma_flat = tf.reshape(log_sigma, [-1])

#     else:
#       #Obtain the flat location and variance
#       loc_flat = tf.zeros(batch_size *self.d_model)
#       log_sigma_flat = tf.ones(batch_size *self.d_model)

#     #Obtain the posterior
#     posterior = multi_gaussian(loc = loc_flat, scale_diag = log_sigma_flat)
    
#     #Obtain the samples
#     samples = posterior.sample(self.num_modes) # num_modes x (batch_size x d_model)

#     #Reshape to obtain compatible tensor
#     samples = tf.reshape(samples, [self.num_modes, batch_size, self.d_model])
#     x = tf.transpose(samples, perm = [1,0,2])
#     x = tf.reshape(x, [self.num_modes, batch_size, 1, self.d_model])

#     #return output and value for the modified loss
#     aux = self.KL_Loss(loc_flat, log_sigma_flat)
#     return x, aux


#----- new try ----------------------------------------

class CVAE_attention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_modes, rate=0.1):
    super(CVAE_attention, self).__init__()

    self.d_model = d_model

    #The number of modes to obtain from the CVAE
    self.num_modes = num_modes

    #TODO: the value 10 should be a hyperparameter of the entire model
    self.z_size  = 10

    #Attention module and query vector to obtain loc and log_scale
    self.attention       = Attention(d_model)
    self.average_query   = tf.Variable(tf.ones([1,1,self.z_size]), trainable = True)

    #Dense layers to obtain loc and log_scale
    self.loc_dense       = tf.keras.layers.Dense(self.z_size)
    self.log_scale_dense = tf.keras.layers.Dense(self.z_size)

    #Dense layer to go back to dimension d_model after concatenation
    self.dense3 = tf.keras.layers.Dense(d_model)

  def KL_Loss(self,mu,log_sigma):
    # ------------------------ NOTE ------------------------------
    # Notice that the equation for the KL divergence of isotropic
    # gaussian distributions is additive for batched information
    # we then get the average value over the number of batchs
    # ------------------------------------------------------------
    k   = tf.math.reduce_prod(tf.shape(mu))
    KL  = tf.norm(mu)**2 
    KL += tf.math.reduce_sum(tf.math.exp(2*log_sigma))
    KL -= tf.cast(k, tf.float32)
    KL -= 2*tf.math.reduce_sum(log_sigma)
    KL  = KL/(2*mu.shape[0])
    return KL

  def call(self, x, training):
    # Input size: batch_size x sequence_length x d_model
    batch_size = tf.shape(x)[0]
    sequence_size = tf.shape(x)[1]
    avg , _    = self.attention(x, x, self.average_query, None) # batch_size x 1 x d_model

    if training:
      #Obtain the location and the variance
      loc       = self.loc_dense(avg)       # batch_size x 1 x z_size
      log_sigma = self.log_scale_dense(avg) # batch_size x 1 x z_size
      sigma     = tf.math.exp(log_sigma)    # batch_size x 1 x z_size

      #Flatten to process simultaneously in a multi gaussian distribution
      KL_value  = self.KL_Loss(loc, log_sigma)

    else:
      #Obtain the flat location and variance
      loc      = tf.zeros([batch_size,1,self.z_size])
      sigma    = tf.ones([batch_size,1,self.z_size])
      KL_value = 0

    #Obtain the posterior
    posterior = multi_gaussian(loc = loc, scale_diag = sigma)
    
    #Obtain the samples
    samples   = posterior.sample(self.num_modes) # num_modes x batch_size x 1 x z_size

    #Reshape to obtain compatible tensor
    samples   = tf.transpose(samples, perm = [1,0,2,3])
    # samples   = tf.reshape(samples, [self.num_modes, batch_size, 1, self.z_size])

    #repeat to make the concatenation
    samples   = tf.repeat(samples,sequence_size, axis = 2) # num_modes x batch_size x sequence_size x z_size

    #do concatenation
    x = tf.reshape(x,tf.concat([[1],x.shape],axis = 0)) # 1 x batch_size x sequence_size x d_model
    x = tf.repeat(x,self.num_modes, axis = 0)         # num_modes x batch_size x sequence_size x d_model
    x = tf.transpose(x,[1,0,2,3])                      # batch_size x num_modes x sequence_size x d_model
    x = tf.concat([x,samples], axis = -1)             # batch_size x num_modes x sequence_size x (d_model + z_size)

    #go back to size d_model
    x = self.dense3(x) # batch_size x num_modes x sequence_size x d_model

    #TODO: check if dropout is necessary

    return x, KL_value