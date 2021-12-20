import sys

sys.path.append("../")
import numpy as np
import random
from model.core import NeuralResponseModel as NeurResponseModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import multivariate_normal
from environments.env_list.simple2d import Simple2D, Sargolini2006, BasicSargolini2006
from tqdm import tqdm


class ExcInhPlasticity(NeurResponseModel):

    def __init__(self, model_name="ExcitInhibitoryplastic", **mod_kwargs):
        super().__init__(model_name, **mod_kwargs)
        self.agent_step_size = mod_kwargs["agent_step_size"]
        self.metadata = {"mod_kwargs": mod_kwargs}
        self.etaexc = mod_kwargs["exc_eta"]  # Learning rate.
        self.etainh = mod_kwargs["inh_eta"]
        self.Ne = mod_kwargs["Ne"]
        self.Ni = mod_kwargs["Ni"]
        self.Nef = mod_kwargs["Nef"]
        self.Nif = mod_kwargs["Nif"]
        self.alpha_i = mod_kwargs["alpha_i"]
        self.alpha_e = mod_kwargs["alpha_e"]
        self.we_init = mod_kwargs["we_init"]
        self.wi_init = mod_kwargs["wi_init"]

        self.sigma_exc = mod_kwargs["sigma_exc"]
        self.sigma_inh = mod_kwargs["sigma_inh"]

        self.room_width, self.room_depth = mod_kwargs["room_width"], mod_kwargs["room_depth"]
        self.ro = mod_kwargs["ro"]
        self.obs_history = []

        self.resolution = 50
        self.x_array = np.linspace(-self.room_width/2, self.room_width/2, num=self.resolution)
        self.y_array = np.linspace(self.room_depth/2, -self.room_depth/2, num=self.resolution)
        self.mesh = np.array(np.meshgrid(self.x_array, self.y_array))
        self.xy_combinations = self.mesh.T.reshape(-1, 2)

        self.reset()

    def reset(self):
        self.global_steps = 0
        self.history = []
        # TODO : add this as default or input to the function
        self.wi = np.random.uniform(low=self.wi_init-0.05*self.wi_init, high=self.wi_init+0.05*self.wi_init, size=(self.Ni))
        self.we = np.random.uniform(low=self.we_init-0.05*self.we_init, high=self.we_init+0.05*self.we_init, size=(self.Ne))
        # self.wi = np.random.normal(loc=1.5, scale=1.5*0.05/3, size=(self.Ni))  # what is the mu and why do we have the 1 and not2
        # self.we = np.random.normal(loc=1.0, scale=1.0*0.05/3, size=(self.Ne))

        self.inh_rates_functions, self.inh_cell_list = self.generate_tuning_curves(n_curves=self.Ni,
                                                                                   cov_scale=self.sigma_inh,
                                                                                   Nf=self.Nif,
                                                                                   alpha=self.alpha_i)
        self.exc_rates_functions, self.exc_cell_list = self.generate_tuning_curves(n_curves=self.Ne,
                                                                                   cov_scale=self.sigma_exc,
                                                                                   Nf=self.Nef,
                                                                                   alpha=self.alpha_e)
        self.init_we_sum = np.sqrt(np.sum(self.we**2))

    def generate_tuning_curves(self, n_curves, cov_scale, Nf, alpha):
        width_limit = self.room_width / 2.0
        depth_limit = self.room_depth / 2.0
        cell_list = []
        function_list = []
        for i in tqdm(range(n_curves)):
            gauss_list = []
            cell_i = 0
            for j in range(Nf):
                mean1 = np.random.uniform(-width_limit*(1+0.2), width_limit*(1+0.2))
                mean2 = np.random.uniform(-depth_limit*(1+0.2), depth_limit*(1+0.2))
                cov = np.diag(np.multiply(cov_scale, np.array([self.room_width, self.room_depth]))**2)
                # cov = np.diag([(self.room_width * cov_scale)**2, (self.room_depth * cov_scale)**2])
                mean = np.array([mean1, mean2])
                rv = multivariate_normal(mean, cov)
                gauss_list.append([mean, cov])
                normalization_constant = 2*np.pi*np.sqrt(np.linalg.det(cov))
                cell_i += rv.pdf(self.xy_combinations)*normalization_constant*alpha
            function_list.append(gauss_list)
            cell_list.append(cell_i)
        return function_list, np.array(cell_list)

    def act(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) >= 1000:
            self.obs_history = [obs, ]
        action = np.random.normal(scale=0.1, size=(2,))
        return action

    def get_output_rates(self, pos):
        exc_rates = self.get_rates(self.exc_cell_list, pos)
        inh_rates = self.get_rates(self.inh_cell_list, pos)

        r_out = self.we.T @ exc_rates - self.wi.T @ inh_rates
        return np.clip(r_out, a_min=0, a_max=np.amax(r_out))

    def get_rates(self, cell_list, pos, get_n_cells=None):
        diff = self.xy_combinations - pos[np.newaxis, ...]
        dist = np.sum(diff**2, axis=1)
        index = np.argmin(dist)
        rout = []
        for i in range(cell_list.shape[0]):
            rout.append(cell_list[i, index])
        rout = np.array(rout)
        return np.clip(rout, a_min=0, a_max=np.amax(rout))

    def get_full_output_rate(self):
        r_out = self.we.T @ self.exc_cell_list - self.wi.T @ self.inh_cell_list
        return np.clip(r_out, a_min=0, a_max=np.amax(r_out))

    def update(self, exc_normalization=True, pos=None):
        if pos is None:
            pos = self.obs_history[-1]
        r_out = self.get_output_rates(pos)

        delta_we = self.etaexc*self.get_rates(self.exc_cell_list, pos=pos)*r_out
        delta_wi = self.etainh*self.get_rates(self.inh_cell_list, pos=pos)*(r_out - self.ro)

        self.we = self.we + delta_we
        self.wi = self.wi + delta_wi

        if exc_normalization:
            self.we = self.init_we_sum/np.sqrt(np.sum(self.we**2))*self.we

        self.we = np.clip(self.we, a_min=0, a_max=np.amax(self.we))
        self.wi = np.clip(self.wi, a_min=0, a_max=np.amax(self.wi))

    def full_average_update(self, exc_normalization=True):
        r_out = self.get_full_output_rate()
        # r_out = r_out.reshape(-1, 1)
        r_out = r_out[..., np.newaxis]
        delta_we = self.etaexc*(self.exc_cell_list @ r_out)/self.resolution**2
        delta_wi = self.etainh*(self.inh_cell_list @ (r_out-self.ro))/self.resolution**2

        self.we = self.we + delta_we[:, 0]
        self.wi = self.wi + delta_wi[:, 0]

        if exc_normalization:
            self.we = self.init_we_sum/np.sqrt(np.sum(self.we**2))*self.we

        self.we = np.clip(self.we, a_min=0, a_max=np.amax(self.we))
        self.wi = np.clip(self.wi, a_min=0, a_max=np.amax(self.wi))

    def full_update(self, exc_normalization=True):
        random_permutation = np.arange(self.xy_combinations.shape[0])
        xy_array = self.xy_combinations[random_permutation, :]
        for i in range(self.xy_combinations.shape[0]):
            self.update(exc_normalization=exc_normalization, pos=xy_array[i, :])

    def plot_rates(self, save_path=None):
        f, ax = plt.subplots(1, 3, figsize=(14, 5))

        r_out_im = self.get_full_output_rate()
        r_out_im = r_out_im.reshape((self.resolution, self.resolution))

        exc_im = self.exc_cell_list[0, ...].reshape((self.resolution, self.resolution))
        inh_im = self.inh_cell_list[0, ...].reshape((self.resolution, self.resolution))

        ax[0].imshow(exc_im, cmap="Reds")
        ax[0].set_title("Exc rates", fontsize=14)
        ax[1].imshow(inh_im, cmap="Blues")
        ax[1].set_title("Inh rates", fontsize=14)
        im = ax[2].imshow(r_out_im)
        ax[2].set_title("Out rate", fontsize=14)
        cbar = plt.colorbar(im, ax=ax[2])

        if not save_path is None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close("all")
        else:
            plt.show()


if __name__ == "__main__":

    run_raw_data = False

    if run_raw_data:
        data_path = "/home/rodrigo/HDisk/8F6BE356-3277-475C-87B1-C7A977632DA7_1/all_data/"

        session = {"rat_id": "11016", "sess": "31010502"}

        env = Sargolini2006(data_path=data_path,
                            verbose=False,
                            session=session,
                            time_step_size=None,
                            agent_step_size=None)

        exc_eta = 6.7e-5
        inh_eta = 2.7e-4
        model_name = "model_example"
        sigma_exc = 0.05
        sigma_inh = 0.1
        Ne = 4900
        Ni = 1225
        Nef = 1
        Nif = 1
        agent_step_size = 0.1
        alpha_i = 1
        alpha_e = 1

        print("init cells")
        agent = ExcInhPlasticity(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                                 sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                                 Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                                 alpha_i=alpha_i, alpha_e=alpha_e)

        print("Plotting rate")
        agent.plot_rates("figures/init_rates.pdf")

        print("running updates")
        n_steps = 30000
        # Initialize environment

        total_iters = 0

        all_sessions = {"11016": ['02020502', '25010501', '28010501', '29010503', '31010502'],  # 5/6
                        "10884": ['01080402', '02080405', '03080402', '03080405', '03080409', '04080402', '05080401',
                                   '08070402', '08070405', '09080404', '13070402', '14070405', '16070401', '19070401',
                                   '21070405', '24070401', '31070404'],  # 22/6
                        "10704": ['06070402', '07070402', '07070407', '08070402', '19070402', '20060402',
                                  '20070402', '23060402', '25060402', '26060402'],  # 32/6
                        "11084": ['01030503', '02030502', '03020501', '08030506', '09030501', '09030503', '10030502',
                                  '23020502', '24020502', '28020501'],  # 42/6
                        "11265": ['01020602', '02020601', '03020601', '06020601', '07020602', '09020601', '13020601',
                                  '16030601', '16030604', '31010601'],  # 52/6
                        "11207": ['03060501', '04070501', '05070501', '06070501', '07070501', '08060501', '08070504',
                                  '09060501']}  # 60/6

        for rat_id, session_list in all_sessions.items():
            for j, sess in enumerate(session_list):
                session = {"rat_id": rat_id, "sess": sess}
                obs, state = env.reset(sess=session)
                print("Running sess", session)
                for i in tqdm(range(n_steps)):
                    # Observe to choose an action
                    obs = obs[:2]
                    action = agent.act(obs)
                    rate = agent.update()
                    # Run environment for given action
                    obs, state, reward = env.step(action)
                    total_iters += 1
                agent.plot_rates(save_path="figures/iter_"+str(total_iters)+".pdf")

        print("plotting results")
        agent.plot_rates()

    else:
        data_path = "../environments/experiment_data/sargolini2006/"
        env = BasicSargolini2006(data_path=data_path,
                                 time_step_size=0.1,
                                 agent_step_size=None)
        exc_eta = 2e-4
        inh_eta = 8e-4
        model_name = "model_example"
        sigma_exc = np.array([0.05, 0.05])
        sigma_inh = np.array([0.1, 0.1])
        Ne = 4900
        Ni = 1225
        Nef = 1
        Nif = 1
        alpha_i = 1
        alpha_e = 1
        we_init = 1.0
        wi_init = 1.5

        agent_step_size = 0.1

        agent = ExcInhPlasticity(model_name=model_name, exc_eta=exc_eta, inh_eta=inh_eta, sigma_exc=sigma_exc,
                                 sigma_inh=sigma_inh, Ne=Ne, Ni=Ni, agent_step_size=agent_step_size, ro=1,
                                 Nef=Nef, Nif=Nif, room_width=env.room_width, room_depth=env.room_depth,
                                 alpha_i=alpha_i, alpha_e=alpha_e, we_init=we_init, wi_init=wi_init)

        agent.plot_rates()

        print("debug")

        plot_every = 10
        total_iters = 0

        obs, state = env.reset()
        #for i in tqdm(range(env.total_number_of_steps)):
        for i in tqdm(range(5000)):
            # Observe to choose an action
            obs = obs[:2]
            action = agent.act(obs)
            # rate = agent.update()
            agent.full_update()
            # Run environment for given action
            obs, state, reward = env.step(action)
            total_iters += 1
            if i % plot_every == 0:
                agent.plot_rates(save_path="figures/pre_processed_iter_"+str(i)+".pdf")
