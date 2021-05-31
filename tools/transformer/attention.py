import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask):

  matmul_qk = tf.matmul(q, k, transpose_b=True)
  print("Matmul ",matmul_qk.shape)
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  print("Sca ",scaled_attention_logits.shape)
  # add the mask to the scaled tensor.
  if mask is not None:
    print("Mask ",mask.shape)
    scaled_attention_logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
  print("Attw ",attention_weights.shape)

  output = tf.matmul(attention_weights, v)
  print("Out ",output.shape)

  return output, attention_weights

class Attention(tf.keras.layers.Layer):
  def __init__(self, d_model):
    super(Attention, self).__init__()

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def call(self, v, k, q, mask):
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    output = scaled_attention

    return output, attention_weights

class Multi_headed_attention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(Multi_headed_attention, self).__init__()
    # Number of heads
    self.num_heads = num_heads
    #
    self.d_model   = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

  def split_heads(self, x):
    x = tf.reshape(x,[x.shape[0],-1,self.num_heads, self.depth])
    return tf.transpose(x, perm=[0,2,1,3])

  def call(self, v, k, q, mask):
    batch_size    = tf.shape(q)[0]
    sequence_size = tf.shape(q)[1]
    print(q.shape)
    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)
    print(q.shape)
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    print(q.shape)

    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    # TODO: Verify
    concat_attention = tf.reshape(scaled_attention,[scaled_attention.shape[0],-1,self.d_model])
    print(concat_attention.shape)

    output = concat_attention

    # output = self.dense(concat_attention)

    return output, attention_weights
