import numpy as np
import tensorflow as tf


def loss_function(real, pred):
    # Error for ade/fde
    diff = pred - real
    diff = diff ** 2
    diff = tf.sqrt(tf.reduce_sum(diff, [1, 2]))
    return tf.math.reduce_min(diff)


# real: n_batch x sequence_length x p
# prediction: n_batch x n_modes x sequence_length x p
def ADE_train(real, prediction, maxi=False):
    sequence_length= real.shape[1]
    n_batches      = prediction.shape[0]
    n_modes        = prediction.shape[1]
    real_expanded  = tf.expand_dims(real, 1)
    # diff: n_batch x n_modes x sequence_length x p
    diff   = (prediction - real_expanded)**2
    # diff = (tf.cumsum(prediction,axis=2) - tf.cumsum(real_expanded,axis=2))**2
    # Along time, Euclidean distance between predicted and real points
    losses = tf.sqrt(tf.reduce_sum(diff, 3))
    losses = tf.reduce_mean(losses, 2)
    # Over the samples: take the min or the max
    if not maxi:
        losses = tf.reduce_min(losses, axis=1)
    else:
        losses = tf.reduce_max(losses, axis=1)
    # Average over batch elements
    return tf.reduce_sum(losses) / n_batches


def KL_loss(real, prediction, mu, sigma, beta, maxi=False):
    k = len(mu)
    KL = np.dot(mu, mu) + np.sum(sigma) - k - np.sum(np.log(sigma))
    KL = KL / 2
    return ADE_train(real, prediction, maxi) + beta * KL


def ADE_train_CVAE(real, prediction, maximum=False):
    sequence_length = real.shape[1]
    n_batches = prediction.shape[0]
    n_modes = prediction.shape[1]
    real_expanded = tf.expand_dims(real, 1)
    # diff: n_batch x n_modes x sequence_length x p
    diff = prediction - real_expanded
    # Sum over time to get absolute positions and take the squares
    losses = tf.reduce_sum(diff, 2) ** 2
    losses = tf.sqrt(tf.reduce_sum(losses, 3))
    # Average over time
    losses = tf.reduce_sum(losses, 2) / sequence_length
    # Over the samples: take the min or the max
    if not maximum:
        losses = tf.reduce_min(losses, axis=1)
    else:
        losses = tf.reduce_max(losses, axis=1)
    # Average over batch elements
    return tf.reduce_sum(losses) / n_batches


def min_ADE_FDE(ground_truth, prediction):
    sequence_length = ground_truth.shape[1]
    diff = prediction - tf.expand_dims(ground_truth, 1)
    diff = diff ** 2
    # Evaluate FDE
    FDE = diff[:, :, -1, :]
    FDE = tf.norm(FDE, axis=2)
    FDE = tf.math.reduce_min(FDE, axis=1, keepdims=True)
    # Evaluate ADE
    ADE = tf.norm(diff, axis=3)
    ADE = tf.math.reduce_min(tf.math.reduce_mean(ADE, axis=2), axis=1, keepdims=True)
    return ADE.numpy(), FDE.numpy()


def accuracy_function(real, pred):
    # Error for ade/fde
    diff = real - pred
    diff = diff ** 2
    diff = -tf.sqrt(tf.reduce_sum(diff, axis=1))
    return tf.math.exp(diff)



#Beta is always positive
@tf.function
def train_step(input, target, transformer, optimizer, beta = 0, burnout=False):
    # Target
    target_train = target[:, :-1, :]
    # This is to hold one position only
    aux = input[:, -1:]
    # target_train will hold the last input data + the T_pred-1 first positions of the future
    # size: n_batch x sequence_size x p
    target_train = tf.concat([aux, target_train], axis=1)
    with tf.GradientTape() as tape:
        # Apply the transformer network to the input
        predictions, _, KL_value = transformer(input, aux, training = True, evaluate = 12)
        loss = ADE_train(target, predictions, burnout) + beta*KL_value
    if loss < 1000 or burnout == True:
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    return loss, KL_value
