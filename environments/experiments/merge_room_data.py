import os.path

import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio


class WernleData(object):
    """ Data wrapper for https://www.nature.com/articles/s41593-017-0036-6
    The data can be obtained from https://doi.org/10.11582/2017.00023
    """

    def __init__(self, data_path, experiment_name="MergingRoomData", verbose=False):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.inner_path = "nn_Data+Code/data/"
        self._load_data()

    def _load_data(self):
        # Ratemap is a ndarray (128 cells from 10 rats x 2) where each element is a 100x100 ratemap.
        # The first column are the reatmaps before merging the room, second column after merging the room.
        self.ratemap = sio.loadmat(os.path.join(self.data_path, self.inner_path, "Figures_1_2_3/ratemaps.mat"))["ratemaps"]

        # The following is the recording progression of grid cell patterns for 19 different cells
        # over 10 different rats

        #
        self.ratemap_dev = sio.loadmat(os.path.join(self.data_path, self.inner_path, r"Figure 4/ratemapsDevelopment.mat"))
        self.ratemap_dev = self.ratemap_dev["ratemapsDevelopment"]


if __name__ == "__main__":
    data_path = "Wernle2018/"
    data = WernleData(data_path=data_path, verbose=True)
    print("debug")