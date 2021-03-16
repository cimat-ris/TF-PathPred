import tensorflow as tf

from .position_enc import positional_encoding
from .attention import  Multi_headed_attention
from .ffnn import point_wise_feed_forward_network
from transformer.masking import create_look_ahead_mask

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.att1 = Multi_headed_attention(d_model, num_heads)
    self.att2 = Multi_headed_attention(d_model, num_heads)

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

    attn2, attn_weights_block2 = self.att2(enc_output, enc_output, out1, None)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)

    ffn_output = self.ffn(out2)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
  def __init__(self, d_model, num_layers, num_heads, dff, maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Dense(d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, enc_output, training, evaluate = None):
    batch_size = x.shape[0]
    attention_weights = []

    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:batch_size,:]

    x = self.dropout(x, training=training)

    if evaluate == None:
        mask = create_look_ahead_mask(x.shape[0])
        for i in range(self.num_layers):
            x, _ , _ = self.dec_layers[i](x, enc_output, training, mask)
            return x, None
    else:
        for j in range(evaluate):
            predictions = x
            mask = create_look_ahead_mask(predictions.shape[0])
            for i in range(self.num_layers):
                predictions, block1, block2 = self.dec_layers[i](predictions, enc_output, training, mask)
                attention_weights.append(block1)
                attention_weights.append(block2)
            prediction = tf.expand_dims(predictions[-1,:],0)
            x = tf.concat([x,prediction], axis = 0)
        x = x[-evaluate:]


    return x, attention_weights