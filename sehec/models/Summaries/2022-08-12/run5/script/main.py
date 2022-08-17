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
x_ = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['s_size_comp'] * pars['n_freq']), name='x_')
d0 = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['n_actions'], pars['t_episode']), name='d')
rnn = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['p_size'], pars['p_size']), name='rnn')
rnn_inv = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['p_size'],pars['p_size']), name='rnn_')
g_ = tf.placeholder(tf.float32, shape=(pars['batch_size'], pars['g_size']), name='g_')
x = tf.unstack(x1, axis=2)
d = tf.unstack(d0, axis=2)

# Initialise Environment(s) and Agent (variables, weights etc.)
env_name = "TEMenv"
mod_name = "TEM"

envs = TEMenv(environment_name=env_name, **pars)
agent = TEM(x, x_, d, seq_index, rnn, rnn_inv, g_, model_name=mod_name, **pars)

fetches_all, fetches_all_ = [], []
fetches_all.extend([agent.g, agent.p, agent.p_g, agent.x_gt, agent.x_, agent.A, agent.A_inv,
                    agent.accuracy_gt, agent.train_op_all])
fetches_all_.extend([agent.g, agent.p, agent.p_g, agent.x_gt, agent.x_, agent.A, agent.A_inv,
                     agent.accuracy_gt, agent.temp])

# Create Session
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=1)
train_writer = tf.summary.FileWriter(train_path, sess.graph)
tf.global_variable_initializer().run()
tf.get_default_graph().finalize()

# Run Model
for i in range(pars['n_iters']):
    # Information on direction
    pars, rn, n_restart, no_direc_batch = direction_pars(pars_orig, pars, pars['n_restart'])

    # Initialise Hebbian matrices each batch
    a_rnn, a_rnn_inv = TEM.initialise_hebbian()

    # Initialise Environment and Variables (same each batch)
    gs, x_s, visited = TEM.initialise_variables()

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
