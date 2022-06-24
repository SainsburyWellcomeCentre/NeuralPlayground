import pickle
from datetime import datetime
import os
from sehec.available_classes import import_classes
import_list = import_classes()
for imp in import_list:
    exec(imp)


def save_run(model, env, save_path):
    pass


def default_train_loop(agent, env, n_steps):
    total_iters = 0
    obs, state = env.reset()
    for i in range(n_steps):
        # Observe to choose an action
        obs = obs[:2]
        action = agent.act(obs)
        # rate = agent.update()
        agent.update()
        # Run environment for given action
        obs, state, reward = env.step(action)
        total_iters += 1
    return agent, env


def default_run(sub_exp_config, save_path=None):

    model_config = sub_exp_config.model_params
    env_config = sub_exp_config.env_params

    init_message = "Running "+model_config.config_id+" model \n"
    init_message += "in "+env_config.config_id+" environment"

    init_model_cmd = model_config.class_name+"(**model_config.__dict__)"
    model = eval(init_model_cmd)

    init_env_cmd = env_config.class_name+"(**env_config.__dict__)"
    env = eval(init_env_cmd)

    model, env = default_train_loop(agent=model, env=env, n_steps=model_config.n_iters)
    save_run(model, env, save_path=save_path)


if __name__ == "__main__":
    from sehec.experimentconfig import cfg
    print(cfg.config_tree())

    default_run(sub_exp_config=cfg.model1.exp_1.sub_exp_1)
