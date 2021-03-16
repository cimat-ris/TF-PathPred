import numpy as np
import tensorflow as tf
def convert_to_changes(tr):
    start = tr[0]
    changes = tr[:-1]-tr[1:]
    return start, changes

def convert_to_traj(s,changes):
    if len(changes.shape) == 2:
        n = changes.shape[0]
        res = np.zeros([n+1,2])
        res[0] = s
        for i in range(n):
            res[i+1] = res[i]+changes[i]
    else:
        l = changes.shape[0]
        n = changes.shape[1]
        res = np.zeros([l,n+1,2])
        for i in range(l):
            res[i,0] = s
            for j in range(n):
                res[i,j+1] = res[i,j]+changes[i,j]
    
    return res

def obs_pred_trajectories(trajectories, separator = 8, f_per_traj = 20):
    N_t = len(trajectories)
    Trajm = []
    Trajp = []
    starts = []
    for tr in trajectories:
        s, tr = convert_to_changes(tr)
        Trajm.append(np.array(tr[range(separator-1),:],dtype = 'f'))
        Trajp.append(np.array(tr[range(separator-1,f_per_traj-1),:], dtype = 'f'))
        starts.append(s)
    return np.array(starts),np.array(Trajm),np.array(Trajp)

def detect_separator(trajectories,secs):
    traj = trajectories[0]
    for i in range(len(traj)):
        if traj["timestamp"].iloc[i]-traj["timestamp"].iloc[0]>secs:
            break
    return i-1