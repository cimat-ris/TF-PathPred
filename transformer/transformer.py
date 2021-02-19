import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------Positional encoding-----------------------------

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  # pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(angle_rads, dtype=tf.float32)



#--------------------------------Attention----------------------------------

def scaled_dot_product_attention(q, k, v):

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class Attention(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super(Attention, self).__init__()

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def call(self, v, k, q):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

    output = scaled_attention

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])

#------------------------------Attention Decoder -----------------------------
def scaled_dot_product_attention_dec(q, k, v, mask):

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class Attention_dec(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super(Attention_dec, self).__init__()

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    scaled_attention, attention_weights = scaled_dot_product_attention_dec(q, k, v, mask)

    output = scaled_attention

    return output, attention_weights

#-------------------------------------Encoder------------------------------

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.att = Attention(d_model)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training):
    attn_output, _ = self.att(x, x, x)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x+attn_output)

    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1+ffn_output)

    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, input_size, maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    #Dimensions used for interchanging between attention and the FFNN
    self.d_model = d_model
    self.dff = dff

    self.embedding = tf.keras.layers.Dense(d_model)

    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.enc_layer = EncoderLayer(d_model, dff, rate)

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training):

    #This only happens once
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding
    x = self.dropout(x, training=training)


    #This sends the data through the encoder Layers

    x = self.enc_layer(x, training)

    return x



#-----------------------------------Decoder-----------------------------------


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.att1 = Attention_dec(d_model)
    self.att2 = Attention(d_model)

    self.ffn = point_wise_feed_forward_network(d_model,dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, mask):
    attn1, attn_weights_block1 = self.att1(x, x, x, mask)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(x + attn1)

    attn2, attn_weights_block2 = self.att2(enc_output, enc_output, x)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + x)

    ffn_output = self.ffn(out2)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)

    return out3


class Decoder(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model

    self.embedding = tf.keras.layers.Dense(d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layer = DecoderLayer(d_model, dff, rate)

    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, mask):
    batch_size = x.shape[0]
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:batch_size,:]

    x = self.dropout(x, training=training)

    x = self.dec_layer(x, enc_output, training, mask)

    # attention_weights['decoder_layer{}_block1'] = block1
    # attention_weights['decoder_layer{}_block2'] = block2


    return x #, attention_weights


#-------------------------Transformer--------------------------------------------


class Transformer(tf.keras.Model):
  def __init__(self, d_model, dff, input_size, target_size, rate=0.1):
    super(Transformer, self).__init__()
    self.encoder = Encoder(d_model, dff, input_size, input_size, rate)

    self.decoder = Decoder(d_model, dff, target_size, rate)

    self.final_layer = tf.keras.layers.Dense(2)


  def call(self, inp, x, training, mask):
    enc_output = self.encoder(inp, training)
    dec_output = self.decoder(x, enc_output, training, mask)
    final_output = self.final_layer(dec_output)

    return final_output #, attention_weights