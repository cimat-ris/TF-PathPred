import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import tensorflow as tf
import time


from opentraj_benchmark.all_datasets import get_trajlets


from tools.trajectories import obs_pred_trajectories, convert_to_traj
from tools.optimizer import CustomSchedule
from tools.parameters import *


from transformer.transformer import Transformer
from transformer.masking import create_look_ahead_mask
from transformer.training import loss_function, accuracy_function, train_step

from transformer.training import ADE_FDE

def test_model(test_name,path):

    trajectories = get_trajlets(path,test_name)[test_name[0]][:,:,:2]

    Starts_train , Xm_test, Xp_test = obs_pred_trajectories(trajectories,Tobs,Tpred+Tobs)

    Xm_test = tf.constant(Xm_test)
    Xp_test = tf.constant(Xp_test)

    #-------------------- Visualize solution ----------------------

    transformer = Transformer(d_model, num_layers, num_heads, dff, Tobs, Tpred, dropout_rate)

    checkpoint_path = f"./checkpoints/train/{test_name[0]}"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print('Latest checkpoint restored!!')
    else:
      print('No model trained for this particular dataset')
      return None

    ade,fde,weights,inps,tars,preds = [],[],[],[],[],[]
    print("calculating predictions")
    for s in range(len(Xm_test)):
        print(s, end = ", ")
        start = Starts_train[s]
        inp = Xm_test[s].numpy()
        tar = Xp_test[s].numpy()
        pred, w = transformer(inp,inp,False,12)

        for i in range(len(w)):
            w[i] = w[i].numpy()

        inp = convert_to_traj(start,inp)
        tar = convert_to_traj(inp[-1],tar)
        pred = convert_to_traj(inp[-1],pred.numpy())

        a,f = ADE_FDE(tar,pred)
        ade.append(a)
        fde.append(f)
        inps.append(inp)
        tars.append(tar)
        preds.append(pred)

        weights.append(w)
    trajs=[np.array(inps),np.array(tars),np.array(preds)]
    print("ADE:", np.mean(ade),"FDE:", np.mean(fde))

    return ade,fde,weights,trajs,transformer

#------------------------------ plot solution ----------------------------

def print_sol(inp, tar, pred):

    plt.plot(inp[:,0],inp[:,1], color = "red")
    plt.scatter(inp[:,0],inp[:,1], color = "red")

    for i in range(len(pred)):
        plt.plot(pred[i,:,0],pred[i,:,1], color = "green")
        plt.scatter(pred[i,:,0],pred[i,:,1], color = "green")


    plt.plot(tar[:,0],tar[:,1], color = "blue")
    plt.scatter(tar[:,0],tar[:,1], color = "blue")

if __name__ == '__main__':

    #------------------------ Parser ---------------------------

    # Parser arguments
    parser = argparse.ArgumentParser(description='Train transformer')
    parser.add_argument('--root-path', '--root',
                        default='./',
                        help='path to folder that contain dataset')
    args = parser.parse_args()

    #-------------------- Info for testing ----------------------
    # test_name = ['ETH-univ']
    test_name = ['ETH-hotel']
    # test_name = ['UCY-zara1']
    # test_name = ['UCY-zara2']
    # test_name = ['UCY-univ3']

    test_model(test_name,args.root_path)

    