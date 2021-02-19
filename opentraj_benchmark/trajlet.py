# Author: Javad Amirian
# Email: amiryan.j@gmail.com


import numpy as np


def split_trajectories(traj_list, length=4.79, overlap=1.99, to_numpy=False):
    trajlets = []
    ts = traj_list[0]["timestamp"]
    dt = ts.iloc[1] - ts.iloc[0]
    f_per_traj = int(np.ceil((length - 1E-3) / dt))
    f_step = int(np.ceil((length - overlap) / dt))
    for tr in traj_list:
        n_frames = len(tr)
        for start_f in range(0, n_frames - f_per_traj, f_step):
            trajlets.append(tr.iloc[start_f:start_f + f_per_traj])

    trl_np_list = []
    if to_numpy:
        for trl in trajlets:
            trl_np = trl[["pos_x", "pos_y", "vel_x", "vel_y"]].to_numpy()
            trl_np_list.append(trl_np)
        trajlets = np.stack(trl_np_list)
    return trajlets


def to_numpy(trajlet_pd_list):
    pass


# test
if __name__ == "__main__":
    from crowdscan.loader.loader_eth import loadETH
    eth_dataset = loadETH("/home/cyrus/workspace2/OpenTraj/datasets/ETH/seq_eth/obsmat.txt")
    trajs = eth_dataset.get_trajectories()
    trajlets = split_trajectories(trajs)
    print(len(trajlets))

