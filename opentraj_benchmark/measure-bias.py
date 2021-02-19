# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import sys
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from crowdscan.loader.loader import loadETH, loadSDD, loadTrajNet
# matplotlib.use('PS')

opentraj_root = os.path.join(os.path.expanduser("~"), 'workspace2/OpenTraj')
output_dir = os.path.join(os.path.expanduser("~"), 'Dropbox/OpenTraj-paper/exp/ver-0.2')
# opentraj_root = sys.argv[1]
# output_dir = sys.argv[2]

## ETH
eth_root = os.path.join(opentraj_root, 'datasets/ETH/seq_eth/obsmat.txt')
# eth_dataset = ParserETH(eth_root)
eth_dataset = loadETH(eth_root, title='ETH')

hotel_root = os.path.join(opentraj_root, 'datasets/ETH/seq_hotel/obsmat.txt')
# hotel_dataset = ParserETH(hotel_root)
hotel_dataset = loadETH(hotel_root, title='Hotel')
# ******************************

## UCY
zara01_root = os.path.join(opentraj_root, 'datasets/UCY/zara01/obsmat.txt')
# zara01_dataset = ParserETH(zara01_root)
zara01_dataset = loadETH(zara01_root, title='zara01')

zara02_root = os.path.join(opentraj_root, 'datasets/UCY/zara02/obsmat.txt')
# zara02_dataset = ParserETH(zara02_root)
zara02_dataset = loadETH(zara02_root, title='zara02')

students03_root = os.path.join(opentraj_root, 'datasets/UCY/students03/obsmat.txt')
# students03_dataset = ParserETH(students03_root)
students03_dataset = loadETH(students03_root, title='students03')

# GC
# TODO: ParserGC is not inherited from TrajectoryDataset
# gc_root = os.path.join(opentraj_root, 'datasets/GC/Annotation')
# gc_dataset = ParserGC(gc_root, world_coord=True)

# SDD
# TODO: we dont have the homography matrix. Do we?
#  TrajNet have already computed this...

# FIXME: ParserSDD does not accept directory
bookstore_root = os.path.join(opentraj_root, 'datasets/SDD/bookstore')
# bookstore_dataset = ParserSDD(bookstore_root)
# ...

datasets = [
    # eth_dataset,
    # hotel_dataset,
    # zara01_dataset,
    # zara02_dataset,
    # students03_dataset,
]


## TRAJNET
trajnet_train_root = os.path.join(opentraj_root, 'datasets/TrajNet/Train')
# trajnet_dataset = loadTrajNet(trajnet_train_root)
trajnet_files = glob.glob(trajnet_train_root + "/**/*.txt", recursive=True)

for trajnet_file in trajnet_files:
    name = 'Trajnet - ' + trajnet_file.split('/')[-1][:-4]
    datasets_i = loadTrajNet(trajnet_file, title=name)
    datasets.append(datasets_i)
    trajs_i = datasets_i.get_trajectories(columns=["pos_x", "pos_y", "vel_x", "vel_y"], to_numpy=True)
    trajs_i = np.stack(trajs_i)
    trajs_zero_based = trajs_i[:, :, :2] - trajs_i[:, 7, :2].reshape((-1, 1, 2))
    start_thetas = np.arctan2(trajs_i[:, 7, 2], trajs_i[:, 7, 3])  # calculated from first velocity vector
    rot_matrices = np.stack([np.array([[np.cos(theta), -np.sin(theta)],
                                       [np.sin(theta), np.cos(theta)]]) for theta in start_thetas])
    trajs_i_rotated = np.matmul(rot_matrices, trajs_zero_based.transpose((0, 2, 1))).transpose((0, 2, 1))

    # ======== PLOT ============
    fig1, ax1 = plt.subplots()
    obsvs_plt = ax1.plot(trajs_i_rotated[:, :8, 1].T, trajs_i_rotated[:, :8, 0].T, '--', alpha=0.3)
    preds_plt = ax1.plot(trajs_i_rotated[:, 8:, 1].T, trajs_i_rotated[:, 8:, 0].T, alpha=0.3)
    avg_plt = ax1.plot(np.mean(trajs_i_rotated[:, 9::2, 1], axis=0),
                       np.mean(trajs_i_rotated[:, 9::2, 0], axis=0), '*', color='red')
    ax1.grid()
    ax1.set_aspect('equal')
    plt.title(datasets_i.title)
    plt.xlim([-4, 7])
    plt.ylim([-4, 4])
    origin_plt, = plt.plot(0, 0, 'o', color='green', label='origin')
    plt.legend(handles=[obsvs_plt[0], preds_plt[0], avg_plt[0], origin_plt],
               labels=["obsv", "pred", "mean(preds)", "origin"], loc="lower left")

    plt.savefig(os.path.join(output_dir, 'bias', name + '.png'))
    plt.show()


for dataset in datasets:
    all_trajs = dataset.get_trajectories()
    x = 1

