import tensorflow as tf
import tensorflow_probability as tfpro
multi_gaussian = tfpro.python.distributions.MultivariateNormalDiag

from .attention import  Attention
from .ffnn import point_wise_feed_forward_network

class neighbor_attention(tf.keras.layers.Layer):
  def __init__(self, d_model, rate=0.1):
    super(neighbor_attention, self).__init__()

    #The number of modes to obtain from the CVAE
    self.att_average = Attention(d_model)
    self.average_query = tf.Variable(tf.ones(d_model), trainable = True)
    self.average_query = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.average_query, axis = 0), axis = 0), axis = 0)

    self.neighbor_att = Attention(d_model)


  def call(self, x, neighbors):

    neighbors, _ = self.att_average(neighbors,neighbors,self.average_query)
    x = self.neighbor_att(neighbors,neighbors,x)

    return x