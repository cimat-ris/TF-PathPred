import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

from .encoder import Encoder
from .decoder import Decoder
from .CVAE import CVAE_attention


class multi_modal(tf.keras.layers.Layer):
    def __init__(self, num_modes):
        super(multi_modal, self).__init__()

        self.dense = [tf.keras.layers.Dense(2) for _ in range(num_modes)]
        self.num_modes = num_modes

    def call(self, dec_output, training, evaluate=None):
        modes = []
        for i in range(self.num_modes):
            modes.append(self.dense[i](dec_output))
        # Stack in the 2nd dimension
        return tf.stack(modes, axis=1)


# Our Transformer model
class Transformer(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, dff, num_modes, rate=0.1):
        super(Transformer, self).__init__()
        # Encoder
        self.encoder = Encoder(d_model, num_layers, num_heads, dff, 100, rate)
        # Decoder
        self.decoder = Decoder(d_model, num_layers, num_heads, dff, 100, rate)
        #
        self.modes = multi_modal(num_modes)

    # Call to the transformer
    def call(self, input, x, training, evaluate=None):
        # Call the encoder on the inputs
        enc_output = self.encoder(input, training)  # [batch_size,sequence_size,d_model]
        # Decoder: takes as input the input encoding and the partial prediction
        dec_output, attention_weights = self.decoder(x, enc_output, training,
                                                     evaluate)  # [batch_size,sequence_size,d_model]
        # The output
        partition_output = self.modes(dec_output)  # [batch_size,num_modes,sequence_size,d_model]
        return partition_output, attention_weights, NULL


# Our Transformer model with CVAE
class Transformer_CVAE(tf.keras.Model):
    def __init__(self, d_model, num_layers, num_heads, dff, num_modes, rate=0.1):
        super(Transformer_CVAE, self).__init__()
        # Encoder
        self.encoder = Encoder(d_model, num_layers, num_heads, dff, 100, rate)
        # CVAE
        self.cvae = CVAE_attention(d_model, num_modes, rate)
        # Decoder
        self.decoder = Decoder(d_model, num_layers, num_heads, dff, 100, rate, num_modes=num_modes, cvae=True)

        # Go back to coordinates
        self.coor = tf.keras.layers.Dense(2)

    # Call to the transformer
    def call(self, input, x, training, evaluate=None):
        # Call the encoder on the inputs
        enc_output = self.encoder(input, training)  # [batch_size,sequence_size,d_model]
        # CVAE: Middle layer. Adds stochastic component
        cvae_output,KL_value = self.cvae(enc_output, training)  
        cvae_output = cvae_output + enc_output
        cvae_output = tf.transpose(cvae_output, [1,0,2,3]) # [batch_size,num_modes,sequence_size,d_model]
        # Decoder: takes as input the input encoding and the partial prediction
        dec_output, attention_weights = self.decoder(x, cvae_output, training,
                                                     evaluate)  # [batch_size,num_modes,sequence_size,d_model]

        # Go back to coordinates
        final_output = self.coor(dec_output)

        return final_output, attention_weights, KL_value
