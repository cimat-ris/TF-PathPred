import numpy as np
import tensorflow as tf
from math import atan,cos,sin
import matplotlib.pyplot as plt
def convert_to_displacements(tr):
    start = tr[0]
    changes = tr[1:]-tr[:-1]
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

def convert_to_traj_with_rotations(s, changes, d=1, mtc=np.array([[1.,0],[0,1]]) ):

    if len(changes.shape) == 2:
        n = changes.shape[0]
        res = np.zeros([n+1,2])
        for i in range(n):
            res[i+1] = res[i]+changes[i]
        res = res*d
        res = res.dot(mtc.T)
        res = res + s

    else:
        l = changes.shape[0]
        n = changes.shape[1]
        res = np.zeros([l,n+1,2])
        for i in range(l):
            for j in range(n):
                res[i,j+1] = res[i,j]+changes[i,j]
            res[i] = res[i]*d
            res[i] = res[i].dot(mtc.T)
            res[i] = res[i] + s

    return res

def obs_pred_trajectories(trajectories, separator = 8, f_per_traj = 20):
    N_t = len(trajectories)
    Trajm = []
    Trajp = []
    starts = []
    for tr in trajectories:
        s, tr = displacements(tr)
        Trajm.append(np.array(tr[range(separator-1),:],dtype = 'float32'))
        Trajp.append(np.array(tr[range(separator-1,f_per_traj-1),:], dtype = 'float32'))
        starts.append(s)
    return np.array(starts),np.array(Trajm),np.array(Trajp)

def obs_pred_rotated_velocities(trajectories, separator = 8, f_per_traj = 20):
    # Number of trajectories in the dataset
    N_t = len(trajectories)
    Trajm  = []
    Trajp  = []
    starts = []
    rot_mtcs = []
    dists = []
    fig,ax = plt.subplots(1)
    plt.margins(0, 0)
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Scan for all the trajctories (they come )
    for tr in trajectories:
        # First position
        # TODO: in most other codes the reference is taken at the last observation
        s  = tr[0]
        tr = tr - s
        # Goes to the first non-zero position
        i  = 0
        while not np.linalg.norm(tr[i]) > 0 and i<tr.shape[0]: i+=1
        # In case we have not seen one single significant displacement
        if i==tr.shape[0]:
            return
        # Get the x,y
        b,a = tr[i]
        d = np.linalg.norm(tr[i])
        # When the displacemnt is very small, we scale by a fixed quanttity
        if d<0.2:
            d=0.2
        rot_matrix = np.array([[b/d,a/d],[-a/d,b/d]])
        # Scaling the trajetory with respect to the length of the first displacement
        tr = tr/d
        # Rotate tr with the inverse
        tr = tr.dot(rot_matrix.T)
        ax.plot(tr[:,0],tr[:,1])
        # Get displacements from differences in absolute positions
        _ , tr = convert_to_displacements(tr)
        # Keep the rotation inverse
        rot_matrix = rot_matrix.T
        # Keep the observations
        Trajm.append(np.array(tr[range(separator-1),:],dtype = 'f'))
        # Keep the positions to predict
        Trajp.append(np.array(tr[range(separator-1,f_per_traj-1),:], dtype = 'f'))
        # Keep absolute starting point
        starts.append(s)
        # Keep rotation matrix
        rot_mtcs.append(rot_matrix)
        # The normalizing distances
        dists.append(d)
    plt.show()
    return np.array(starts), np.array(Trajm),np.array(Trajp), np.array(dists), np.array(rot_mtcs)

def detect_separator(trajectories,secs):
    traj = trajectories[0]
    for i in range(len(traj)):
        if traj["timestamp"].iloc[i]-traj["timestamp"].iloc[0]>secs:
            break
    return i-1

def traj_to_real_coordinates(traj,H):
    if len(traj.shape) == 2:
        cat = np.vstack([traj[:, 0], traj[:, 1], np.ones_like(traj[:, 0])]).T
        tCat = (H @ cat.T).T

        x = tCat[:, 1] / tCat[:, 2]
        y = tCat[:, 0] / tCat[:, 2]
        return np.vstack([x,y]).T
    else:
        res = []
        for i in range(len(traj)):
            cat = np.vstack([traj[i,:, 0], traj[i,:, 1], np.ones_like(traj[i,:, 0])]).T
            tCat = (H @ cat.T).T

            x = tCat[:, 1] / tCat[:, 2]
            y = tCat[:, 0] / tCat[:, 2]
            res.append(np.vstack([x,y]).T)
        return np.array(res)
