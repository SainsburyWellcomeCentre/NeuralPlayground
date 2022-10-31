from sehec.experiments import FullSargoliniData, SargoliniData
from sehec.arenas import Sargolini2006, BasicSargolini2006

if __name__ == "__main__":

    fullsarg_data = FullSargoliniData(verbose=True)
    fullsarg_data.plot_recording_tetr()
    fullsarg_data.plot_trajectory()

    data_path = None#"../envs/experiments/Sargolini2006/raw_data_sample/"

    simple_sarg = SargoliniData()

    env = BasicSargolini2006()

    env = FullSargoliniData(data_path=data_path,
                            verbose=True)

    env = Sargolini2006()

