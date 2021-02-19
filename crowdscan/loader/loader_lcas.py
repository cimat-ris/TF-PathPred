# Author: Pat
# Email: bingqing.zhang.18@ucl.ac.uk

import numpy as np
import pandas as pd
import glob
import ast
from crowdscan.crowd.trajdataset import TrajDataset


#load LCAS data (two scenes in the rawdata: minerva & strands2, but only minerva is included here as strands2 data has some issues with repeated time)
#checked with the data provided by TrajNet++, they actually only used part of the data from minerva. Here I included all data from minerva
def loadLCAS(path, **kwargs):
    traj_dataset = TrajDataset()
    traj_dataset.title = "LCAS"
    minerva_files_list = glob.glob(path + "/minerva/**/data.csv")
    minerva_columns = ['frame_id','person_id','pos_x','pos_y','rot_z','rot_w','scene_id']
   
    # read from minerva data.csv
    minerva_raw_dataset = []
    # This load data from all files
    for file in minerva_files_list:
        print(file)
        data = pd.read_csv(file, sep=",", header=None,names=minerva_columns)
        minerva_raw_dataset.append(data)
    minerva_raw_dataset = pd.concat(minerva_raw_dataset)
    minerva_raw_dataset['scene_id'] = 'minerva'
    
    minerva_raw_dataset.reset_index(inplace=True, drop=True)

    traj_dataset.title = kwargs.get('title', "LCAS")
    traj_dataset.data[["frame_id", "agent_id","pos_x", "pos_y","scene_id"]] = \
        minerva_raw_dataset[["frame_id", "person_id","pos_x","pos_y","scene_id"]]

    traj_dataset.data["label"] = "pedestrian"

    # post-process. For LCAS, raw data do not include velocity, velocity info is postprocessed
    fps = kwargs.get('fps', 2.5)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)
    return traj_dataset

