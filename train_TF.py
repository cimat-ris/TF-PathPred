#Imports for the code of Javad
import argparse
import os

from crowdscan.loader.loader_eth import loadETH
from opentraj_benchmark.trajlet import split_trajectories

#Imports for handling trajectories
from tools.trajectories import obs_pred_trajectories

#Imports from the code of the TF tutorial
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

from tools.optimizer import CustomSchedule
from transformer.transformer import Transformer
# from tools.training import loss_function, accuracy_function


#Extra imports
from random import randint
import matplotlib.pyplot as plt

#----------------------------------Masking-----------------------------------------

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  

#------------------------------Training functions----------------------------------

def loss_function(real,pred):
    # Error for ade/fde
    diff = real - pred
    diff = diff**2
    diff = tf.sqrt(tf.reduce_sum(diff))
    return diff

def accuracy_function(real,pred):
    # Error for ade/fde
    diff = real - pred
    diff = diff**2
    diff = -tf.sqrt(tf.reduce_sum(diff, axis=1))
    return tf.math.exp(diff)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

@tf.function
def train_step(inp, tar):
  tar_train = tar
  tar_train = tar[:-1,:]
  aux = tf.expand_dims(inp[-1,:],0)
  tar_train = tf.concat([aux,tar_train], axis = 0)

  mask = create_look_ahead_mask(tar_train.shape[0])

  with tf.GradientTape() as tape:
    predictions = transformer(inp, tar_train, True, mask)
    loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(accuracy_function(tar, predictions))

#-------------------------------Evaluate transformer-------------------------------
def evaluate(inp, Tpred):
    output = tf.expand_dims(inp[-1,:], 0)

    for i in range(Tpred):
        mask = create_look_ahead_mask(output.shape[0])
        predictions = transformer(inp, output, False, mask)
        prediction = tf.expand_dims(predictions[-1,:],0)
        output = tf.concat([output,prediction], axis = 0)

    return output

#--------------------Here starts the code for training------------------------------

# Parser arguments
parser = argparse.ArgumentParser(description='Measure multimodality'
                                             'on trajectories.')
parser.add_argument('--root-path', '--root',
                    default='./datasets/',
                    help='path to foldet that contain dataset')
args = parser.parse_args()

if __name__=='__main__':

    dataset_name = 'eth'

    # Load dataset
    dataset = []
    if dataset_name == 'eth':
        # ETH
        root = os.path.join(args.root_path,
                            'ETH/seq_eth/obsmat.txt')

        dataset = loadETH(root, title='ETH')

    trajectories = dataset.get_trajectories()

    N_t = len(trajectories)
    
    #Take out the ones that have a total length less than 8s
    short_trajs = []
    
    for i in range(N_t):
        n = len(trajectories[i])
        # times = trajectories[i]["timestamp"].to_numpy()
        # length = times[n-1]-times[0]
        # if length < 8:
        #     short_trajs.append(i)
        if n < 20:
            short_trajs.append(i)
    
    n = len(short_trajs)
    for i in range(n):
        del trajectories[short_trajs[n-i-1]]

    #Obtain trajlets from big trajectories
    # trajectories = split_trajectories(trajectories, 8, 1)

    
    #Number of frames in observed and predicted trajlets
    Tobs = 8 #detect_separator(trajectories,3.2)
    Tpred = 12 #len(trajectories[0])-Tobs


    #Leave just the position information
    for i in range(len(trajectories)):
        trajectories[i] = trajectories[i][['pos_x','pos_y']]

    #Update the number of trajectories
    N_t = len(trajectories)
    
    #Obtain observed and predicted trajlets
    Xm, Xp = obs_pred_trajectories(trajectories,Tobs,Tpred+Tobs)

    Xm = tf.constant(Xm)
    Xp = tf.constant(Xp)

    Xm_ = []
    Xp_ = []
    for x in Xm:
        Xm_.append(x)
    for x in Xp:
        Xp_.append(x)

    Xm = Xm_
    Xp = Xp_

    d_model = 128
    dff = 512

    dropout_rate = 0.1

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(d_model)

    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

    transformer = Transformer(d_model, dff, Tobs, Tpred, dropout_rate)

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print ('Latest checkpoint restored!!')


    train_dataset = []
    for i in range(len(Xp)):
        train_dataset.append((Xm[i],Xp[i]))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    EPOCHS = 0

    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      train_accuracy.reset_states()

      # inp -> portuguese, tar -> english
      for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
          print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
              epoch + 1, batch, train_loss.result(), train_accuracy.result()))

      if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))

      print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

      print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    s = randint(0,(len(Xm)-1))
    inp = Xm[s].numpy()
    tar = Xp[s].numpy()
    tar = np.concatenate([np.array([inp[-1]]),tar], axis = 0)
    pred = evaluate(inp, Tpred).numpy()
    plt.cla()
    plt.plot(inp[:,0],inp[:,1], color = "red")
    plt.scatter(inp[:,0],inp[:,1], color = "red")
    plt.plot(tar[:,0],tar[:,1], color = "blue")
    plt.scatter(tar[:,0],tar[:,1], color = "blue")
    plt.plot(pred[:,0],pred[:,1], color = "green")
    plt.scatter(pred[:,0],pred[:,1], color = "green")
    plt.show()