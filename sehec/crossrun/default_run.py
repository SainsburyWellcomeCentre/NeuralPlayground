from sehec.available_classes import import_classes
import_classes()


def default_run(model_config, env_config):
    init_message = "Running "+model_config.config_id+" model \n"
    init_message += "in "+env_config.config_id+" environment"

    init_model_cmd = "make str"
    model = eval()