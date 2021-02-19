# Author: Javad Amirian
# Email: amiryan.j@gmail.com


from crowdscan.loader.utils.kalman_smoother import KalmanModel
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class TrajDataset:
    def __init__(self):
        """
        data might include the following columns:
        "scene_id", "frame_id", "agent_id",
         "pos_x", "pos_y"
          "vel_x", "vel_y",
        """
        self.critical_columns = ["frame_id", "agent_id", "pos_x", "pos_y"]
        self.data = pd.DataFrame(columns=self.critical_columns)

        # a map from agent_id to a list of [agent_ids] that are annotated as her groupmate
        # if such informatoin is not available the map should be filled with an empty list
        # for each agent_id
        self.groupmates = {}

        # fps is necessary to calc any data related to time (e.g. velocity, acceleration)
        self.fps = -1

        self.title = ''

        # bounding box of trajectories
        self.bbox = pd.DataFrame({'x': [np.nan, np.nan],
                                  'y': [np.nan, np.nan]},
                                 index=['min', 'max'])

        # FixMe ?
        #  self.trajectories_lazy = []

    def postprocess(self, use_kalman_smoother=False):
        """
        This function should be called after loading the data by loader
        It performs the following steps:
        -: check fps value, should be set and bigger than 0
        -: check critical columns should exist in the table
        -: update data types
        -: fill 'groumates' if they are not set
        -: checks if velocity do not exist, compute it for each agent
        -: compute bounding box of trajectories
        """

        # check 1
        if self.fps < 0:
            raise ValueError("Error! fps of dataset is not set!")

        # check 2
        for critical_column in self.critical_columns:
            if critical_column not in self.data:
                raise ValueError("Error! critical columns are missing!")

        # modify data types
        self.data["frame_id"] = self.data["frame_id"].astype(int)
        if str(self.data["agent_id"].iloc[0]).replace('.', '', 1).isdigit():
            self.data["agent_id"] = self.data["agent_id"].astype(int)
        self.data["pos_x"] = self.data["pos_x"].astype(float)
        self.data["pos_y"] = self.data["pos_y"].astype(float)

        agent_ids = pd.unique(self.data["agent_id"])

        # fill scene_id
        # FIXME:
        if "scene_id" not in self.data:
            self.data["scene_id"] = 0

        # fill timestamps based on frame_id and video_fps
        if "timestamp" not in self.data:
            self.data["timestamp"] = self.data["frame_id"] / self.fps

        # fill groupmates
        for agent_id in agent_ids:
            if agent_id not in self.groupmates:
                self.groupmates[agent_id] = []

        # fill velocities
        if "vel_x" not in self.data:
            self.data["vel_x"] = None
            self.data["vel_y"] = None
            self.data["vel_x"] = self.data["vel_x"].astype(float)
            self.data["vel_y"] = self.data["vel_y"].astype(float)

            # ============================================
            for agent_id in agent_ids:
                print('calc velocity for', agent_id)
                indices_for_agent_id = np.nonzero((self.data["agent_id"] == agent_id).to_numpy())
                if len(indices_for_agent_id[0]) < 2: continue

                traj_df = self.data.iloc[indices_for_agent_id]
                dt = traj_df["timestamp"].diff()
                dt.iloc[0] = dt.iloc[1]

                if use_kalman_smoother:
                    # print('Yes! Smoothing the trajectories in train_set ...')
                    kf = KalmanModel(dt.iloc[1], n_dim=2, n_iter=7)
                    smoothed_pos, smoothed_vel = kf.smooth(traj_df[["pos_x", "pos_y"]].to_numpy())
                    traj_df[["vel_x", "vel_y"]] = smoothed_vel
                else:
                    traj_df["vel_x"] = traj_df["pos_x"].diff() / dt
                    traj_df["vel_y"] = traj_df["pos_y"].diff() / dt
                    traj_df["vel_x"].iloc[0] = traj_df["vel_x"].iloc[1]
                    traj_df["vel_y"].iloc[0] = traj_df["vel_y"].iloc[1]
                self.data["vel_x"].iloc[indices_for_agent_id] = traj_df["vel_x"]
                self.data["vel_y"].iloc[indices_for_agent_id] = traj_df["vel_y"]
                self.data["vel_x"].loc[indices_for_agent_id] = traj_df["vel_x"]
                self.data["vel_y"].loc[indices_for_agent_id] = traj_df["vel_y"]
        # compute bounding box
        # FixMe: what if there are multiple scenes?
        self.bbox['x']['min'] = min(self.data["pos_x"])
        self.bbox['x']['max'] = max(self.data["pos_x"])
        self.bbox['y']['min'] = min(self.data["pos_y"])
        self.bbox['y']['max'] = max(self.data["pos_y"])

    def interpolate_frames(self, inplace=True):
        all_frame_ids = sorted(pd.unique(self.data["frame_id"]))
        if len(all_frame_ids) < 2:
            # FixMe: print warning
            return

        frame_id_A = all_frame_ids[0]
        frame_A = self.data.loc[self.data["frame_id"] == frame_id_A]
        agent_ids_A = frame_A["agent_id"].to_list()
        interp_data = self.data  # "agent_id", "pos_x", "pos_y", "vel_x", "vel_y"
        # df.append([df_try] * 5, ignore_index=True
        for frame_id_B in all_frame_ids[1:]:
            frame_B = self.data.loc[self.data["frame_id"] == frame_id_B]
            agent_ids_B = frame_B["agent_id"].to_list()

            common_agent_ids = list(set(agent_ids_A) & set(agent_ids_B))
            frame_A_fil = frame_A.loc[frame_A["agent_id"].isin(common_agent_ids)]
            frame_B_fil = frame_B.loc[frame_B["agent_id"].isin(common_agent_ids)]
            for new_frame_id in range(frame_id_A+1, frame_id_B):
                alpha = (new_frame_id - frame_id_A) / (frame_id_B - frame_id_A)
                new_frame = frame_A_fil.copy()
                new_frame["frame_id"] = new_frame_id
                new_frame["pos_x"] = frame_A_fil["pos_x"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["pos_x"].to_numpy() * alpha
                new_frame["pos_y"] = frame_A_fil["pos_y"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["pos_y"].to_numpy() * alpha
                new_frame["vel_x"] = frame_A_fil["vel_x"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["vel_x"].to_numpy() * alpha
                new_frame["vel_y"] = frame_A_fil["vel_y"].to_numpy() * (1 - alpha) +\
                                     frame_B_fil["vel_y"].to_numpy() * alpha
                if inplace:
                    self.data = self.data.append(new_frame)
                else:
                    pass   # TODO
            frame_id_A = frame_id_B
            frame_A = frame_B
            agent_ids_A = agent_ids_B

    # FixMe: rename to add_row()/add_entry()
    def add_agent(self, agent_id, frame_id, pos_x, pos_y):
        """Add one single data at a specific frame to dataset"""
        new_df = pd.DataFrame(columns=self.critical_columns)
        new_df["frame_id"] = [int(frame_id)]
        new_df["agent_id"] = [int(agent_id)]
        new_df["pos_x"] = [float(pos_x)]
        new_df["pos_y"] = [float(pos_y)]
        self.data = self.data.append(new_df)

    def get_agent_ids(self):
        """:return all agent_id in data table"""
        return pd.unique(self.data["agent_id"])

    def get_trajectories(self, agent_ids=[], frame_ids=[], scene_ids="", label="", columns="", to_numpy=False) -> list:
        """
        Returns a list of trajectories
        :param agent_ids: select specific ids, ignore if empty
        :param frame_ids: select a time interval, ignore if empty  # TODO:
        :param scene_ids:
        :param label: select agents from a specific label (e.g. car), ignore if empty # TODO:
        :param columns:
        :param to_numpy:
        :return list of trajectories
        """
        if not agent_ids:
            agent_ids = pd.unique(self.data["agent_id"])
        if not scene_ids:
            scene_ids = pd.unique(self.data["scene_id"])

        trajectories = []
        for scene_id in scene_ids:
            for agent_id in agent_ids:
                if columns:
                    traj_df = self.data[columns].loc[(self.data["agent_id"] == agent_id) &
                                                     (self.data["scene_id"] == scene_id)]
                else:
                    traj_df = self.data.loc[(self.data["agent_id"] == agent_id) &
                                            (self.data["scene_id"] == scene_id)]
                if to_numpy:
                    trajectories.append(traj_df.to_numpy())
                else:
                    trajectories.append(traj_df)

        return trajectories

    # TODO:
    def get_entries(self, agent_ids=[], frame_ids=[], label=""):
        """
        Returns a list of data entries
        :param agent_ids: select specific agent ids, ignore if empty
        :param frame_ids: select a time interval, ignore if empty  # TODO:
        :param label: select agents from a specific label (e.g. car), ignore if empty # TODO:
        :return list of data entries
        """
        output_table = self.data  # no filter
        if agent_ids:
            output_table = output_table[output_table["agent_id"].isin(agent_ids)]
        if frame_ids:
            output_table = output_table[output_table["frame_id"].isin(frame_ids)]
        return output_table

    def get_frames(self, frame_ids: list = [], scene_ids=""):
        if not frame_ids:
            frame_ids = pd.unique(self.data["frame_id"])
        if not scene_ids:
            scene_ids = pd.unique(self.data["scene_id"])

        frames = []
        for scene_id in scene_ids:
            for frame_id in frame_ids:
                frame_df = self.data.loc[(self.data["frame_id"] == frame_id) &
                                         (self.data["scene_id"] == scene_id)]
                # traj_df = self.data.filter()
                frames.append(frame_df)
        return frames

    def apply_transformation(self, tf: np.ndarray, inplace=False):
        """
        :param tf: np.ndarray
            Homogeneous Transformation Matrix,
            3x3 for 2D data
        :param inplace: bool, default False
            If True, do operation inplace
        :return: transformed data table
        """
        if inplace:
            target_data = self.data
        else:
            target_data = self.data.copy()

        # data is 2D
        assert tf.shape == (3, 3)
        tf = tf[:2, :]  # remove the last row
        poss = target_data[["pos_x", "pos_y"]].to_numpy(dtype=np.float)
        poss = np.concatenate([poss, np.ones((len(poss), 1))], axis=1)
        target_data[["pos_x", "pos_y"]] = np.matmul(tf, poss.T).T

        # apply on velocities
        tf[:, -1] = 0  # do not apply the translation element on velocities!
        vels = target_data[["vel_x", "vel_y"]].to_numpy(dtype=np.float)
        vels = np.concatenate([vels, np.ones((len(vels), 1))], axis=1)
        target_data[["vel_x", "vel_y"]] = np.matmul(tf, vels.T).T

        return target_data
