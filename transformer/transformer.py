import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

from .encoder import Encoder
from .decoder import Decoder


class multi_modal(tf.keras.layers.Layer):
  def __init__(self, num_modes):
    super(multi_modal, self).__init__()

    self.dense = [tf.keras.layers.Dense(2) for _ in range(num_modes)]
    self.num_modes = num_modes
    
  def call(self, dec_output, training, evaluate = None):
    modes = []
    for i in range(self.num_modes):
        modes.append(self.dense[i](dec_output))
    modes = tf.stack(modes)

    return modes




class Transformer(tf.keras.Model):
  def __init__(self, d_model, num_layers, num_heads, dff, input_size, target_size, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(d_model, num_layers, num_heads, dff, input_size, 100, rate)

    self.decoder = Decoder(d_model, num_layers, num_heads, dff, 100, rate)

    self.modes = multi_modal(16)


  def call(self, inp, x, training, evaluate = None):
    enc_output = self.encoder(inp, training)
    dec_output, attention_weights = self.decoder(x, enc_output, training, evaluate)

    partition_output = self.modes(dec_output)

    return partition_output, attention_weights

# class Transformer(tf.keras.Model):
#   def __init__(self, d_model, num_layers, num_heads, dff, input_size, target_size, rate=0.1):
#     super(Transformer, self).__init__()
#     self.encoder = Encoder(d_model, num_layers, num_heads, dff, input_size, 100, rate)

#     self.decoder = Decoder(d_model, num_layers, num_heads, dff, 100, rate)

#     self.final_layer = tf.keras.layers.Dense(2)


#   def call(self, inp, x, training, evaluate = None):
#     enc_output = self.encoder(inp, training)
#     dec_output, attention_weights = self.decoder(x, enc_output, training, evaluate)

#     partition_output = tf.reshape(dec_output,[-1, 16, 8])
#     partition_output = tf.transpose(partition_output,[1,0,2])
#     final_output = self.final_layer(partition_output)

#     return final_output, attention_weights