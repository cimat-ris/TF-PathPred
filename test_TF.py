import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import tensorflow as tf
import time


from opentraj_benchmark.all_datasets import get_trajlets


from tools.trajectories import obs_pred_trajectories, convert_to_traj, obs_pred_rotated_velocities, convert_to_traj_with_rotations
from tools.parameters import *


from transformer.transformer import Transformer
from transformer.training import ADE_FDE

def test_model(test_name,path, n_trajs = None):

    trajectories = get_trajlets(path,test_name)[test_name[0]][:,:,:2]

    Starts_train , Xm_test, Xp_test, dists, mtcs = obs_pred_rotated_velocities(trajectories,Tobs,Tpred+Tobs)

    Xm_test = tf.constant(Xm_test)
    
    Xp_test = tf.constant(Xp_test)

    #-------------------- Visualize solution ----------------------

    transformer = Transformer(d_model, num_layers, num_heads, dff, Tobs, Tpred, num_modes, dropout_rate)

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
    if not type(n_trajs) == int:
        A = range(len(Xm_test))
    else:
        A = range(n_trajs)
    for s in A:
        print(s, end = ", ")
        start = Starts_train[s]
        distance = dists[s]
        mtc = mtcs[s]
        inp = Xm_test[s].numpy()
        tar = Xp_test[s].numpy()
        aux = Xm_test[0].numpy()[-1:]
        pred, w = transformer(inp,inp[-1:],False,12)
        pred = pred.numpy()
        # print(inp[-1])
        # print(pred)

        inp_tar = np.concatenate([inp,tar],axis = 0)
        inp_pred = np.zeros([pred.shape[0],(inp.shape[0]+pred.shape[1]),2])
        for i in range(pred.shape[0]):
        	inp_pred[i] = np.concatenate([inp,pred[i]],axis = 0)

        inp_tar = convert_to_traj_with_rotations(start,inp_tar,distance,mtc)
        inp_pred = convert_to_traj_with_rotations(start,inp_pred,distance,mtc)
        inp = inp_tar[:8]
        tar = inp_tar[7:20]
        pred = inp_pred[:,7:20,:]

        a,f = ADE_FDE(tar,pred)
        ade.append(a)
        fde.append(f)
        inps.append(inp)
        tars.append(tar)
        preds.append(pred)
        # print(Xm_test[s].numpy()-(inp[:-1]-inp[1:]),Xp_test[s].numpy()-tar[:-1])

    trajs=[np.array(inps),np.array(tars),np.array(preds)]
    print("ADE:", np.mean(ade),"FDE:", np.mean(fde))

    return ade,fde,None,trajs,transformer

#------------------------------ plot solution ----------------------------

def print_sol(inp, tar, pred, img = None):
    plt.figure(figsize = (10,10))
    if type(img) == np.ndarray:
        plt.imshow(img)

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
    # test_name = ['ETH-hotel']
    test_name = ['UCY-zara1']
    # test_name = ['UCY-zara2']
    # test_name = ['UCY-univ3']

    test_model(test_name,args.root_path)

    