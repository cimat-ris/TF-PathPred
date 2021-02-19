import numpy as np

def obs_pred_trajectories(trajectories, separator = 8, f_per_traj = 20):
    N_t = len(trajectories)
    Trajm = []
    Trajp = []
    for tr in trajectories:
        tr = tr.to_numpy()
        Trajm.append(np.array(tr[range(separator),:],dtype = 'f'))
        Trajp.append(np.array(tr[range(separator,f_per_traj),:], dtype = 'f'))
    return Trajm,Trajp

def detect_separator(trajectories,secs):
    traj = trajectories[0]
    for i in range(len(traj)):
        if traj["timestamp"].iloc[i]-traj["timestamp"].iloc[0]>secs:
            break
    return i-1