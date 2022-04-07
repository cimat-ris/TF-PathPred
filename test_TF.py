import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import tensorflow as tf
import time


from tools.opentraj_benchmark.all_datasets import get_trajlets


from tools.trajectories import obs_pred_trajectories,obs_pred_rotated_trajectories, convert_to_traj_with_rotations
from tools.parameters import *


from tools.transformer.transformer import Transformer_CVAE, Transformer
from tools.transformer.training import min_ADE_FDE

def test_model(test_name,path, n_trajs = None):

    trajectories = get_trajlets(path,test_name)[test_name[0]][:,:,:2]

    Starts_train , Xm_test, Xp_test, dists, mtcs = obs_pred_rotated_trajectories(trajectories,Tobs,Tpred+Tobs)
    Xm_test = tf.constant(Xm_test)
    Xp_test = tf.constant(Xp_test)
    print(Xm_test.shape)
    print(Xp_test.shape)

    #-------------------- Visualize solution ----------------------

    transformer = Transformer_CVAE(d_model, num_layers, num_heads, dff, num_modes, dropout_rate)
    # transformer = Transformer_CVAE(d_model, num_layers, num_heads, dff, num_modes, dropout_rate)
    test_dataset = {"observations":[],"predictions":[],"starts":[],"distances":[],"mtcs":[]}
    # Form the training dataset
    for i in range(len(Xp_test)):
        test_dataset["observations"].append(Xm_test[i])
        test_dataset["predictions"].append(Xp_test[i])
        test_dataset["starts"].append(Starts_train[i])
        test_dataset["distances"].append(dists[i])
        test_dataset["mtcs"].append(mtcs[i])
    # Get the necessary data into a tf Dataset
    test_data = tf.data.Dataset.from_tensor_slices(test_dataset)
    # Form batches
    batched_test_data    = test_data.batch(len(Xp_test))

    checkpoint_path = f"./generated_data/checkpoints/train/{test_name[0]}"
    print(checkpoint_path)
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!',ckpt_manager.latest_checkpoint)
    else:
        print('No model trained for this particular dataset')
        return None


    weights = [],[],[],[]
    print("Calculating predictions")
    for batch in batched_test_data:
        start    = batch["starts"].numpy()
        distance = batch["distances"].numpy()
        mtc      = batch["mtcs"].numpy()
        input    = batch["observations"]
        target   = batch["predictions"]
        pred, w, KL_value = transformer(input,input[:,-1:],training=False,evaluate=12)

        # Reconstruct full trajectory: n_batch x sequence_lenth x p
        input  = input.numpy()
        target = target.numpy()
        pred   = pred.numpy()

        inp_tar = np.concatenate([input,target],axis = 1)
        inp_pred= np.zeros([pred.shape[0],pred.shape[1],(input.shape[1]+pred.shape[2]),2])
        # Stack the modes
        for i in range(pred.shape[1]):
        	inp_pred[:,i] = np.concatenate([input,pred[:,i]],axis = 1)

        for i in range(len(inp_tar)):
            inp_tar[i]  = distance[i] * inp_tar[i] @ mtc[i].T + start[i]
            inp_pred[i] = distance[i] * inp_pred[i] @ mtc[i].T +start[i]

        # Observations
        observations  = inp_tar[:,:7]
        ground_truth  = inp_tar[:,6:]
        predictions   = inp_pred[:,:,6:]
        
        ade,fde = min_ADE_FDE(ground_truth,predictions)
    trajs=[observations,ground_truth,predictions]
    print("ADE:", np.mean(ade),"FDE:", np.mean(fde))
    return ade,fde,None,trajs,transformer

#------------------------------ plot solution ----------------------------

def print_sol(inp, tar, pred, img = None):
    plt.figure(figsize = (10,10))
    if type(img) == np.ndarray:
        plt.imshow(img)

    if type(inp) == np.ndarray:
        plt.plot(inp[:,0],inp[:,1], color = "red")
        plt.scatter(inp[:,0],inp[:,1], color = "red")

    if type(pred) == np.ndarray:
        for i in range(len(pred)):
            plt.plot(pred[i,:,0],pred[i,:,1], color = "green")
            plt.scatter(pred[i,:,0],pred[i,:,1], color = "green")

    if type(tar) == np.ndarray:
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
    test_name = ['ETH-univ']
    # test_name = ['ETH-hotel']
    # test_name = ['UCY-zara1']
    # test_name = ['UCY-zara2']
    # test_name = ['UCY-univ3']

    test_model(test_name,args.root_path)
