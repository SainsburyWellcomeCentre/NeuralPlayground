from sehec.available_classes import import_classes
import_list = import_classes()
for imp in import_list:
    exec(imp)


def default_run(model_config, env_config):
    init_message = "Running "+model_config.config_id+" model \n"
    init_message += "in "+env_config.config_id+" environment"

    init_model_cmd = model_config.class_name+"(**model_config.__dict__)"
    model = eval(init_model_cmd)

    init_env_cmd = env_config.class_name+"(**env_config.__dict__)"
    env = eval(init_env_cmd)

    print(model, env)


if __name__ == "__main__":
    from sehec.experimentconfig import cfg
    print(cfg.config_tree())

    env_params = cfg.model1.exp_1.sub_exp_1.env_params
    model_params = cfg.model1.exp_1.sub_exp_1.model_params
    default_run(model_config=model_params, env_config=env_params)