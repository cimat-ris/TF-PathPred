import numpy as np
import tensorflow as tf

def loss_function(real,pred):
    # Error for ade/fde
    diff = pred - real
    diff = diff**2
    diff = tf.sqrt(tf.reduce_sum(diff,[1,2]))
    return tf.math.reduce_min(diff)

def ADE_train(real,pred, max = False):
    diff = pred - real
    res = 0.
    for i in range(real.shape[0]):
        aux = tf.reduce_sum(diff[:,:i,:],1)
        aux = aux**2
        aux = tf.sqrt(tf.reduce_sum(aux,1))
        res = aux + res
    if max == False:
        return tf.reduce_min(res)/real.shape[0]
    else:
        return tf.reduce_max(res)/real.shape[0]


def ADE_FDE(real,pred):

    n = real.shape[0]
    diff = pred - real
    diff = diff**2
    FDE = diff[:,-1,:]
    FDE = tf.sqrt(tf.reduce_sum(FDE,axis = 1))
    FDE = tf.math.reduce_min(FDE)

    diff = tf.reduce_sum(diff,[1,2])/n
    ADE = tf.math.reduce_min(diff)

    return ADE.numpy(),FDE.numpy()


def accuracy_function(real,pred):
    # Error for ade/fde
    diff = real - pred
    diff = diff**2
    diff = -tf.sqrt(tf.reduce_sum(diff, axis=1))
    return tf.math.exp(diff)

@tf.function
def train_step(inp, tar, transformer, optimizer, train_loss, train_accuracy, burnout = False):
    # Target
    target_train = tar
    target_train = tar[:,:-1,:]
    aux          = tf.expand_dims(inp[:,-1,:],1)
    tar_train = tf.concat([aux,target_train], axis = 1)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_train, True)
        loss = ADE_train(tar, predictions, burnout)

    if loss < 10 or burnout == True:
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar, predictions))
