# Author: Javad Amirian
# Email: amiryan.j@gmail.com

import os
import json
import glob
import numpy as np
import pandas as pd
from crowdscan.crowd.trajdataset import TrajDataset


def loadWildTrack(path: str, **kwargs):
    """
    :param path: path to annotations dir
    :param kwargs:
    :return:
    """
    traj_dataset = TrajDataset()

    files_list = sorted(glob.glob(path + "/*.json"))
    raw_data = []
    for file_name in files_list:
        frame_id = int(os.path.basename(file_name).replace('.json', ''))

        with open(file_name, 'r') as json_file:
            json_content = json_file.read()
            annots_list = json.loads(json_content)
            for annot in annots_list:
                person_id = annot["personID"]
                position_id = annot["positionID"]

                X = -3.0 + 0.025 * (position_id % 480)
                Y = -9.0 + 0.025 * (position_id / 480)
                raw_data.append([frame_id, person_id, X, Y])

    csv_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
    raw_dataset = pd.DataFrame(np.array(raw_data), columns=csv_columns)

    traj_dataset.title = kwargs.get('title', "Grand Central")

    # copy columns
    traj_dataset.data[["frame_id", "agent_id", "pos_x", "pos_y"]] = \
        raw_dataset[["frame_id", "agent_id", "pos_x", "pos_y"]]

    traj_dataset.data["scene_id"] = kwargs.get('scene_id', 0)
    traj_dataset.data["label"] = "pedestrian"

    # post-process
    fps = kwargs.get('fps', 10)
    sampling_rate = kwargs.get('sampling_rate', 1)
    use_kalman = kwargs.get('use_kalman', False)
    traj_dataset.postprocess(fps=fps, sampling_rate=sampling_rate, use_kalman=use_kalman)

    return traj_dataset
