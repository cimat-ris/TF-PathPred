import tensorflow as tf

from .position_enc import positional_encoding
from .attention import  Multi_headed_attention
from .ffnn import point_wise_feed_forward_network

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.att = Multi_headed_attention(d_model,num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    attn_output, _ = self.att(x, x, x, None)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x+attn_output)

    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1+ffn_output)

    return out2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, d_model, num_layers, num_heads, dff, input_size, maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()
    #Dimensions used for interchanging between attention and the FFNN
    self.d_model = d_model
    self.dff = dff
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Dense(d_model)

    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    batch_size = x.shape[0]

    #This only happens once
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:batch_size,:]
    x = self.dropout(x, training=training)


    #This sends the data through the encoder Layers

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)

    return x