import os.path
import numpy as np
import scipy.io as sio
import glob
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter
import sys
import sehec
import matplotlib as mpl

class FullHaftingData(object):

    def __init__(self, data_path=None, recording_nbr= None, experiment_name="FullHaftingData", verbose=False,):
        self.experiment_name = experiment_name
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "envs/experiments/Hafting2008/")
        else:
            self.data_path = data_path
        self._load_data()
        self.list = []
        l = 0
        for i, rat_id in enumerate(self.data_per_animal):
            rat_id = list(self.data_per_animal.keys())
            for j, sess in enumerate(self.data_per_animal[rat_id[i]]):
                sess = list(self.data_per_animal[rat_id[i]])
                for k, cell in enumerate(self.data_per_animal[rat_id[i]][sess[j]]):
                    if cell != 'position':
                        cells = list(self.data_per_animal[rat_id[i]][sess[j]])
                        self.list.append(
                            {"recording_nbr": l, "rat_id": rat_id[i], "sess": sess[j], "record_type": cells[k]})
                        l = l + 1
        if recording_nbr is None:
            self.rat_id, self.sess, key = self.best_session["rat_id"], self.best_session["sess"], self.best_session[
                    "record_type"]
        else:
            self.rat_id, self.sess, self.key = self.get_recorded_session(recording_nbr)
        if verbose:
            self.show_readme()
            self.show_keys()
        self.set_behavioral_data()

    def _load_data(self):
        self.best_session ={'recording_nbr': 4, "rat_id": "11015", "sess": "13120410", "record_type": "t5c1"}
        self.arena_limits = np.array([[-200, 200], [-20, 20]])
        data_path_list = glob.glob(self.data_path + "*.mat")
        mice_ids = np.unique([dp.split("/")[-1][:5] for dp in data_path_list])
        self.data_per_animal = {}
        for m_id in mice_ids:
            m_paths_list = glob.glob(self.data_path + m_id + "*.mat")
            sessions = np.unique([dp.split("/")[-1].split("-")[1][:8] for dp in m_paths_list]).astype(str)
            self.data_per_animal[m_id] = {}
            for sess in sessions:
                s_paths_list = glob.glob(self.data_path + m_id + "-" + sess + "*.mat")
                cell_ids = np.unique([dp.split("/")[-1].split(".")[-2][-4:] for dp in s_paths_list]).astype(str)
                self.data_per_animal[m_id][sess] = {}
                for cell_id in cell_ids:
                    if cell_id == "_POS":
                        session_info = "position"
                    elif "EG" in cell_id:
                        session_info = cell_id[1:]
                    else:
                        session_info = cell_id

                    r_path = glob.glob(self.data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def show_keys(self):
        print("List of recordings: Recording number - Rat ID - Seesion number - Recording type ")
        for i,recording in enumerate(self.list):
            print(self.list[i])

    def show_readme(self):
        readme_path = glob.glob(self.data_path + "readme" + "*.txt")[0]
        with open(readme_path, 'r') as fin:
            print(fin.read())

    def plot_trajectory(self,recording_nbr=None, save_path=None, ax=None, ):
        if  recording_nbr is None:
            self.rat_id, self.sess, self.key  = self.best_session["rat_id"], self.best_session["sess"], self.best_session["record_type"]
        else:
            self.rat_id, self.sess, self.key = self.get_recorded_session(recording_nbr)
        cell_data = self.data_per_animal[self.rat_id][self.sess]
        position_data = cell_data["position"]
        x1, y1 = position_data["posx"][:], position_data["posy"][:]
        x = np.clip(x1, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y1, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        cmap = mpl.cm.get_cmap("plasma")
        time_array = position_data["post"][:]
        norm = plt.Normalize(0, np.size(x))
        arena_width = self.arena_limits[0, 0] - self.arena_limits[0, 1]
        arena_depth = self.arena_limits[1, 0] - self.arena_limits[1, 1]
        if arena_width == arena_depth:
            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(10, 8))
        if arena_width > arena_depth:
            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(np.round((arena_width / arena_depth)) * 8, 8))
        if arena_depth > arena_width:
            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(8, np.round((arena_depth / arena_width)) * 8))

        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 0]], [self.arena_limits[1, 0], self.arena_limits[1, 1]],
                "C3", lw=3)
        ax.plot([self.arena_limits[0, 1], self.arena_limits[0, 1]],
                [self.arena_limits[1, 0], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 0]],
                [self.arena_limits[1, 0], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 1]],
                [self.arena_limits[1, 1], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 1]],
                [self.arena_limits[1, 0], self.arena_limits[1, 0]], "C3", lw=3)

        i = 0
        len_x = np.ones(len(x) - 1)
        aux_x = []
        aux_y = []
        for k in len_x:
            x_ = [x[i], x[i + 1]]
            y_ = [y[i], y[i + 1]]
            aux_x.append(x[i])
            aux_y.append(y[i])
            i = i + 1
            sc = ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6, linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('width')
        ax.set_xlabel('depth')
        ax.set_title("position")
        sc = ax.scatter(aux_x, aux_y, c=np.arange(len(x) - 1), vmin=0, vmax=len(x), cmap="plasma", alpha=0.6, s=0.1)

        cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(x)])
        cbar.ax.set_ylabel('N steps', rotation=270, fontsize=12)
        cbar.ax.set_yticklabels([0, len(x)], fontsize=12)

    def get_recorded_session(self, recording_nbr):
        list_item = self.list[recording_nbr]
        rat_id, sess, cell = list_item["rat_id"], list_item["sess"], list_item["record_type"]
        return rat_id, sess, cell

    def plot_recording_tetr(self, recording_nbr=None, save_path=None, ax=None, ):
        if  recording_nbr is None:
            self.rat_id, self.sess, self.key  = self.best_session["rat_id"], self.best_session["sess"], self.best_session["record_type"]
        else:
            self.rat_id, self.sess, self.key = self.get_recorded_session(recording_nbr)
        if self.key == 'EEG' or self.key == 'EG2' or self.key== 'EG3' or self.key == 'EG4' or self.key== 'c1 2' or self.key == 'G 2' or self.key == 'OS 2' or self.key =='G2 2' or self.key=='c2 2' or self.key=='c4 2':
            print(' You have selected a '+ self.key + ' recording. You need to select a tetrode recording')
        else:
            cell_data = self.data_per_animal[self.rat_id][self.sess]
            position_data = cell_data["position"]
            x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
            x2, y2 = x1, y1
            # Selecting positional data
            x = np.clip(x2, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
            y = np.clip(y2, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
            time_array = position_data["post"][:]
            arena_width = self.arena_limits[0, 0] - self.arena_limits[0, 1]
            arena_depth = self.arena_limits[1, 0] - self.arena_limits[1, 1]

            if arena_width == arena_depth:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(10, 8))
            if arena_width > arena_depth:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(np.round((arena_width / arena_depth)) * 8, 8))
            if arena_depth > arena_width:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(8, np.round((arena_depth / arena_width)) * 8))

            tetrode_data = cell_data[self.key]
            test_spikes = tetrode_data["ts"][:]
            test_spikes = test_spikes[:, 0]
            time_array = time_array[:, 0]


            h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y, filter_result=True)
            sc = ax.imshow(h, cmap='jet')
            cbar = plt.colorbar(sc, ax=ax, ticks=[np.min(h), np.max(h)])
            cbar.ax.set_ylabel('Firing rate', rotation=270, fontsize=16)
            cbar.ax.set_yticklabels([np.round(np.min(h)), np.round(np.max(h))], fontsize=16)
            ax.set_title(self.key)
            ax.set_ylabel('width')
            ax.set_xlabel('depth')
            ax.set_xticks([])
            ax.set_yticks([])

            if not save_path is None:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close("all")
            else:
                return ax

    def set_behavioral_data(self, recording_nbr=None,tolerance=1e-10):
        arena_limits = np.array([[-50, 50], [-50, 50]])
        if recording_nbr is None:
            self.rat_id, self.sess, self.key = self.best_session["rat_id"], self.best_session["sess"], self.best_session[
                "record_type"]
        else:
            self.rat_id, self.sess, self.key = self.get_recorded_session(recording_nbr)
        rat_id  =self.rat_id
        session=self.sess
        cell_data = self.data_per_animal[rat_id][session]
        position_data = cell_data["position"]
    
        # Selecting positional data
        x1, y1 = position_data["posx"][:], position_data["posy"][:]
        # Selecting positional data
        x = np.clip(x1, a_min=-200, a_max=200)
        y = np.clip(y1, a_min=-20, a_max=20)
        time_array = position_data["post"][:]

        position = np.stack([x, y], axis=1) # Convert to cm
        head_direction = np.diff(position, axis=0)
        head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1) + tolerance)[..., np.newaxis]
        self.arena_limits = arena_limits
        self.position = position
        self.head_direction = head_direction
        self.time = time_array
               

class SargoliniData(object):

    def __init__(self, experiment_name='Sargolini_2006_Data', data_path=None):
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "envs/experiments/Sargolini2006")
        else:
            self.data_path = data_path
        self.arena_limits, self.position, self.head_direction = self.get_sargolini_data()

    def get_sargolini_data(self, tolerance=1e-10):
        arena_limits = np.array([[-50, 50], [-50, 50]])
        filenames_x = os.path.join(self.data_path, "sargolini_x_pos_")
        filenames_y = os.path.join(self.data_path, "sargolini_y_pos_")

        x_position = np.array([])
        y_position = np.array([])
        for i in range(61):
            aux_x = np.load(filenames_x + str(i) + ".npy")
            aux_y = np.load(filenames_y + str(i) + ".npy")
            x_position = np.concatenate([x_position, aux_x])
            y_position = np.concatenate([y_position, aux_y])

        position = np.stack([x_position, y_position], axis=1)*100 # Convert to cm
        head_direction = np.diff(position, axis=0)
        head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1) + tolerance)[..., np.newaxis]
        return arena_limits, position, head_direction


class FullSargoliniData(object):

    def __init__(self, data_path=None, recroding_nbr=None, experiment_name="FullSargoliniData", verbose=False,):
        if data_path is None:
            self.data_path = os.path.join(sehec.__path__[0], "envs/experiments/Sargolini2006/raw_data_sample/")
        else:
            self.data_path = data_path
        print("Data path ", self.data_path)
        self.experiment_name = experiment_name
        self.show_readme()
        self._load_data()
        self.list = []
        l = 0
        for i, rat_id in enumerate(self.data_per_animal):
            rat_id = list(self.data_per_animal.keys())
            for j, sess in enumerate(self.data_per_animal[rat_id[i]]):
                sess = list(self.data_per_animal[rat_id[i]])
                for k, cell in enumerate(self.data_per_animal[rat_id[i]][sess[j]]):
                    if cell != 'position':
                        cells=list(self.data_per_animal[rat_id[i]][sess[j]])
                        self.list.append({"recording_nbr": l, "rat_id": rat_id[i], "sess": sess[j], "record_type": cells[k]})
                        l = l + 1
        if recroding_nbr is None:
            self.rat_id, self.sess, key = self.best_session["rat_id"], self.best_session["sess"], self.best_session[
                    "record_type"]
        else:
            self.rat_id, self.sess, self.key = self.get_recorded_session(recording_nbr)
        if verbose:
            self.show_readme()
            self.show_keys()
        self.set_behavioral_data()

    def _load_data(self):
        self.best_session = self.best_session ={'recording_nbr': 20, "rat_id": "11016", "sess": "31010502", "record_type": "T8C2"}
        # self.best_session = {"rat_id": "10704", "sess": "20060402"}
        self.arena_limits = np.array([[-50.0, 50.0], [-50.0, 50.0]])

        data_path_list = glob.glob(self.data_path + "*.mat")
        mice_ids = np.unique([dp.split("/")[-1][:5] for dp in data_path_list])
        self.data_per_animal = {}
        for m_id in mice_ids:
            m_paths_list = glob.glob(self.data_path + m_id + "*.mat")
            sessions = np.unique([dp.split("/")[-1].split("-")[1][:8] for dp in m_paths_list]).astype(str)
            self.data_per_animal[m_id] = {}
            for sess in sessions:
                s_paths_list = glob.glob(self.data_path + m_id + "-" + sess + "*.mat")
                cell_ids = np.unique([dp.split("/")[-1].split(".")[-2][-4:] for dp in s_paths_list]).astype(str)
                self.data_per_animal[m_id][sess] = {}
                for cell_id in cell_ids:
                    if cell_id == "_POS":
                        session_info = "position"
                    elif cell_id in ["_EEG", "_EGF"]:
                        session_info = cell_id[1:]
                    else:
                        session_info = cell_id
                    r_path = glob.glob(self.data_path + m_id + "-" + sess + "*" + cell_id + "*.mat")
                    cleaned_data = clean_data(sio.loadmat(r_path[0]))
                    if cell_id != "_POS" and not cell_id in ["_EEG", "_EGF"]:
                        try:
                            self.data_per_animal[m_id][sess][session_info] = cleaned_data["cellTS"]
                        except:
                            pass
                    else:
                        self.data_per_animal[m_id][sess][session_info] = cleaned_data

    def show_keys(self):
        print("List of recordings: Recording number - Rat ID - Seesion number - Recording type ")
        for i,recording in enumerate(self.list):
            print(self.list[i])


    def show_readme(self):
        readme_path = glob.glob(self.data_path + "readme" + "*.txt")[0]
        with open(readme_path, 'r') as fin:
            print(fin.read())

    def plot_trajectory(self, recording_nbr=None, save_path=None, ax=None,):
        if recording_nbr is None:
            self.rat_id, self.sess, self.key = self.best_session["rat_id"], self.best_session["sess"], self.best_session["record_type"]
        else:
            self.rat_id, self.sess,self.key = self.get_recorded_session(recording_nbr)
        cell_data = self.data_per_animal[self.rat_id][self.sess]
        position_data = cell_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        if len(position_data["posy2"]) != 0:
            x2, y2 = position_data["posy2"][:, 0], position_data["posy2"][:, 0]
        else:
            x2, y2 = x1, y1
        # Selecting positional data
        x = np.mean(np.stack([x1, x2], axis=1), axis=1)
        y = np.mean(np.stack([y1, y2], axis=1), axis=1)
        x = np.clip(x, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        cmap = mpl.cm.get_cmap("plasma")
        time_array = position_data["post"][:, 0]
        norm = plt.Normalize(0, np.size(x))
        arena_width=self.arena_limits[0, 0]-self.arena_limits[0, 1]
        arena_depth = self.arena_limits[1, 0] - self.arena_limits[1, 1]
        if arena_width==arena_depth:
            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(10, 8))
        if arena_width> arena_depth:
            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(np.round((arena_width/arena_depth))*8, 8))
        if arena_depth> arena_width:
            if ax is None:
                f, ax = plt.subplots(1, 1, figsize=(8,np.round((arena_depth/arena_width))*8))

        ax.plot([self.arena_limits[0, 0],self.arena_limits[0, 0]],[self.arena_limits[1, 0],self.arena_limits[1, 1]] ,"C3", lw=3)
        ax.plot([self.arena_limits[0, 1], self.arena_limits[0, 1]],
                   [self.arena_limits[1, 0], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 0]],
                   [self.arena_limits[1, 0], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 1]],
                   [self.arena_limits[1, 1], self.arena_limits[1, 1]], "C3", lw=3)
        ax.plot([self.arena_limits[0, 0], self.arena_limits[0, 1]],
                   [self.arena_limits[1, 0], self.arena_limits[1, 0]], "C3", lw=3)

        i=0
        len_x=np.ones(len(x)-1)
        aux_x = []
        aux_y = []
        for k in len_x:
            x_ = [x[i],x[i+1]]
            y_ = [y[i],y[i+1]]
            aux_x.append(x[i])
            aux_y.append(y[i])
            i = i + 1
            sc=ax.plot(x_, y_, "-", color=cmap(norm(i)), alpha=0.6,linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('width')
        ax.set_xlabel('depth')
        ax.set_title("position")
        sc = ax.scatter(aux_x, aux_y, c=np.arange(len(x)-1),vmin=0, vmax=len(x), cmap="plasma", alpha=0.6,s=0.1 )

        cbar = plt.colorbar(sc, ax=ax, ticks=[0, len(x)])
        cbar.ax.set_ylabel('N steps', rotation=270, fontsize=12)
        cbar.ax.set_yticklabels([0, len(x)], fontsize=12)

    def get_recorded_session(self,recording_nbr):
        list_item = self.list[recording_nbr]
        rat_id, sess ,cell= list_item["rat_id"],  list_item["sess"], list_item["record_type"]
        return  rat_id, sess,cell

    def plot_recording_tetr(self, recording_nbr=None, save_path=None, ax=None, ):
        if recording_nbr is None:
            self.rat_id, self.sess, self.key = self.best_session["rat_id"], self.best_session["sess"], self.best_session["record_type"]
        else:
            self.rat_id, self.sess,self.key = self.get_recorded_session(recording_nbr)
        if self.key == 'EEG' or self.key == 'EGF':
            print(' You have selected a ' + self.key + ' recording. You need to select a tetrode recording')
        else:
            cell_data = self.data_per_animal[self.rat_id][self.sess]
            position_data = cell_data["position"]
            x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
            if len(position_data["posy2"]) != 0:
                x2, y2 = position_data["posy2"][:, 0], position_data["posy2"][:, 0]
            else:
                x2, y2 = x1, y1
            # Selecting positional data
            x = np.mean(np.stack([x1, x2], axis=1), axis=1)
            y = np.mean(np.stack([y1, y2], axis=1), axis=1)
            x = np.clip(x, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
            y = np.clip(y, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
            time_array = position_data["post"][:, 0]
            arena_width = self.arena_limits[0, 0] - self.arena_limits[0, 1]
            arena_depth = self.arena_limits[1, 0] - self.arena_limits[1, 1]

            if arena_width==arena_depth:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(10, 8))
            if arena_width> arena_depth:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(np.round((arena_width/arena_depth))*8, 8))
            if arena_depth> arena_width:
                if ax is None:
                    f, ax = plt.subplots(1, 1, figsize=(8,np.round((arena_depth/arena_width))*8))

            test_spikes = cell_data[self.key][:, 0]
            h, binx, biny = get_2D_ratemap(time_array, test_spikes, x, y, filter_result=True)
            sc=ax.imshow(h,cmap='jet')
            cbar = plt.colorbar(sc, ax=ax,ticks = [np.min(h), np.max(h)])
            cbar.ax.set_ylabel('Firing rate', rotation=270, fontsize=16)
            cbar.ax.set_yticklabels([np.round(np.min(h)), np.round(np.max(h))], fontsize=16)
            ax.set_title(self.key)
            ax.set_ylabel('width')
            ax.set_xlabel('depth')
            ax.set_xticks([])
            ax.set_yticks([])

            if not save_path is None:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close("all")
            else:
                return ax


    def set_behavioral_data(self, rat_id=None, recording_nbr=None):
        arena_limits = np.array([[-50, 50], [-50, 50]])
        if recording_nbr is None:
            self.rat_id, self.sess, self.key = self.best_session["rat_id"], self.best_session["sess"], self.best_session[
                "record_type"]
        else:
            self.rat_id, self.sess, self.key = self.get_recorded_session(recording_nbr)
        rat_id = self.rat_id
        session = self.sess
        cell_data = self.data_per_animal[rat_id][session]
        position_data = cell_data["position"]
        x1, y1 = position_data["posx"][:, 0], position_data["posy"][:, 0]
        if len(position_data["posy2"]) != 0:
            x2, y2 = position_data["posy2"][:, 0], position_data["posy2"][:, 0]
        else:
            x2, y2 = x1, y1

        # Selecting positional data
        x = np.mean(np.stack([x1, x2], axis=1), axis=1)
        y = np.mean(np.stack([y1, y2], axis=1), axis=1)
        x = np.clip(x, a_min=self.arena_limits[0, 0], a_max=self.arena_limits[0, 1])
        y = np.clip(y, a_min=self.arena_limits[1, 0], a_max=self.arena_limits[1, 1])
        time_array = position_data["post"][:, 0]
        position = np.stack([x, y], axis=1) # Convert to cm
        head_direction = np.diff(position, axis=0)
        head_direction = head_direction/np.sqrt(np.sum(head_direction**2, axis=1))[..., np.newaxis]
        self.arena_limits = arena_limits
        self.position = position
        self.head_direction = head_direction
        self.time = time_array


def clean_data(data, keep_headers=False):
    aux_dict = {}
    for key, val in data.items():
        if isinstance(val, bytes) or isinstance(val, str) or key == "__globals__":
            if keep_headers:
                aux_dict[key] = val
            continue
        else:

            if not np.isnan(val).any():
                aux_dict[key] = val
            else:
                # Interpolate nans
                x_range = np.linspace(0, 1, num=len(val))
                nan_indexes = np.logical_not(np.isnan(val))[:, 0]
                clean_x = x_range[nan_indexes]
                clean_val = np.array(val)[nan_indexes, 0]
                # print(clean_x.shape, clean_val.shape)
                f = interp1d(clean_x, clean_val, kind='cubic', fill_value="extrapolate")
                aux_dict[key] = f(x_range)[..., np.newaxis]
    return aux_dict


def get_2D_ratemap(time_array, spikes, x, y, x_size=50, y_size=50, filter_result=False):
    x_spikes, y_spikes = [], []
    for s in spikes:
        array_pos = np.argmin(np.abs(time_array-s))
        x_spikes.append(x[array_pos])
        y_spikes.append(y[array_pos])
    x_spikes = np.array(x_spikes)
    y_spikes = np.array(y_spikes)
    h, binx, biny = np.histogram2d(x_spikes, y_spikes, bins=(x_size, y_size))
    if filter_result:
        h = gaussian_filter(h, sigma=2)

    return h.T, binx, biny


if __name__ == "__main__":
    data = FullHaftingData(verbose=True)
    data.plot_trajectory(22)
    data.plot_recording_tetr(22)








