import sys
sys.path.append("../")
import matplotlib as mpl
import matplotlib.pyplot as plt
from environments.core import Environment
import numpy as np
from environments.experiments.merge_room_data import WernleData
from simple2d import Simple2D

class Merging2D(Simple2D):
    def __init__(self, data_path="C:\Users\Coursework\Documents\MSc Machine Learning\Project\nn_Data+Code\data", environment_name="MergingRoom2D", **env_kwargs):
        self.data_path = data_path
        self.environment_name = environment_name
        self.data = WernleData(data_path=self.data_path, experiment_name=self.environment_name)
        self.AB_limits = np.array([[0, 200], [-100, 100]])
        self.A_limits = np.array([[0, 200], [0, 100]])
        self.B_limits = np.array([[0, 200], [0, 100]])
        env_kwargs["agent_step_size"] = 1/50  # In seconds
        super().__init__(environment_name, **env_kwargs)
        self.metadata["doi"] = "https://doi.org/10.11582/2017.00023"
        self.total_number_of_steps = self.data.position.shape[0]
        self.state_dims_labels = ["x_pos", "y_pos"]

        def get_room(room):
            if room == "AB":
                limits = np.abs(np.diff(self.AB_limits, axis=1))
                self.room_width = limits[0, 0]
                self.room_depth = limits[1, 0]

            if room == "A":
                limits = np.abs(np.diff(self.A_limits, axis=1))
                self.room_width = limits[0, 0]
                self.room_depth = limits[1, 0]

            if room == "B":
                limits = np.abs(np.diff(self.B_limits, axis=1))
                self.room_width = limits[0, 0]
                self.room_depth = limits[1, 0]

            env_kwargs["room_width"] = self.room_width
            env_kwargs["room_depth"] = self.room_depth

            return self.room_width, self.room_depth