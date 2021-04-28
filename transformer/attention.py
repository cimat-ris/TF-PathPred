import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):

  matmul_qk = tf.matmul(q, k, transpose_b=True)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, v)

  return output, attention_weights

class Attention(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super(Attention, self).__init__()

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    output = scaled_attention

    return output, attention_weights

class Multi_headed_attention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(Multi_headed_attention, self).__init__()

    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def split_heads(self, x):
    x = tf.reshape(x,[-1,self.num_heads, self.depth])
    return tf.transpose(x, perm=[1,0,2])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    concat_attention = tf.reshape(scaled_attention,[-1,self.d_model])

    output = concat_attention

    # output = self.dense(concat_attention)

    return output, attention_weights