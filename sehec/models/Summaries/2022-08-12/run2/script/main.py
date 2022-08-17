import matplotlib.pyplot as plt

from sehec.envs.arenas.TEMenv import TEMenv
from sehec.models.TEM.model import *
from sehec.models.TEM.parameters import *

gen_path, train_path, model_path, save_path, script_path = make_directories()

pars = default_params()
save_params(pars, save_path, script_path)
pars_orig = pars.copy()

# Initialise Graph
tf.reset_default_graph()

seq_index = tf.placeholder(tf.float32, shape=(), name='seq_index')
x1 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size'], pars['t_episode']), name='x')
d0 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['n_actions'], pars['t_episode']), name='d')
x = tf.unstack(x1, axis=2)
d = tf.unstack(d0, axis=2)

# Initialise Environment(s) and Agent (variables, weights etc.)
env_name = "TEMenv"
mod_name = "TEM"

envs = TEMenv(environment_name=env_name, **pars)
agent = TEM(x=x, d=d, model_name=mod_name, **pars)

fetches_all, fetches_all_ = [], []
fetches_all.extend([agent.g, agent.p, agent.p_g, agent.x_gt, agent.x_, agent.A, agent.A_inv,
                    agent.accuracy_gt, agent.train_op_all])
fetches_all_.extend([agent.g, agent.p, agent.p_g, agent.x_gt, agent.x_, agent.A, agent.A_inv,
                     agent.accuracy_gt, agent.temp])

# Create Session
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep = 1)
train_writer = tf.summary.FileWriter(train_path, sess.graph)
tf.global_variable_initializer().run()
tf.get_default_graph().finalize()

# Run Model
for i in range(pars['n_iters']):
    for j in range(pars['n_episode']):
        # RL Loop
        obs, states, rewards, actions, direcs = envs.step(agent.act)

        # feed_dict = {x: obs, d: direcs}
        # results = sess.run(fetches_all, feed_dict)
        x_, p, g = agent.update(obs, direcs, j)
        print("finished episode ", j)
    print("finished iteration ", i)

envs.plot_trajectory()
plt.show()
