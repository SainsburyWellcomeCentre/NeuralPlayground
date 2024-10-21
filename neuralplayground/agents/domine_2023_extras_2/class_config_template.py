from config_manager import config_field, config_template


class ConfigTemplate:
    base_config_template = config_template.Template(
        fields=[
            config_field.Field(
                name="experiment_name",
                types=[str, type(None)],
            ),
            config_field.Field(
                name="resample",
                types=[bool],
            ),
            config_field.Field(
                name="wandb_on",
                types=[bool],
            ),
            config_field.Field(
                name="batch_size",
                types=[int],
            ),
            config_field.Field(
                name="num_nodes_min",
                types=[int],
            ),
            config_field.Field(
                name="num_features",
                types=[int],
            ),
            config_field.Field(
                name="num_nodes_max",
                types=[int],
            ),
            config_field.Field(
                name="batch_size_test",
                types=[int],
            ),
            config_field.Field(
                name="num_nodes_min_test",
                types=[int],
            ),
            config_field.Field(
                name="num_nodes_max_test",
                types=[int],
            ),
            config_field.Field(
                name="num_hidden",
                types=[int],
            ),
            config_field.Field(
                name="num_layers",
                types=[int],
            ),
            # @param
            config_field.Field(
                name="num_message_passing_steps",
                types=[int],
            ),
            config_field.Field(
                name="seed",
                types=[int],
            ),
            # @param
            config_field.Field(
                name="learning_rate",
                types=[float],
            ),
            config_field.Field(
                name="dataset",
                types=[str],
            ),
            config_field.Field(
                name="weighted",
                types=[bool],
            ),
            config_field.Field(
                name="num_training_steps",
                types=[float, int],
            ),
            config_field.Field(
                name="residual",
                types=[bool],
            ),
            config_field.Field(
                name="layer_norm",
                types=[bool],
            ),
            config_field.Field(
                name="plot",
                types=[bool],
            ),
        ],
    )

