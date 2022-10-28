from sehec.experiments import FullSargoliniData
from sehec.arenas import Sargolini2006

if __name__ == "__main__":

    fullsarg_data = FullSargoliniData(verbose=True)
    fullsarg_data.plot_recording_tetr()
    fullsarg_data.plot_trajectory()

    data_path = None#"../envs/experiments/Sargolini2006/raw_data_sample/"

    env = Sargolini2006(data_path=data_path,
                        verbose=True,
                        time_step_size=None,
                        agent_step_size=None)

