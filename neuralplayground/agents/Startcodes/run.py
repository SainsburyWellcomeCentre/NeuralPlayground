import os
import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pathlib import Path
from datetime import datetime
import wandb
import shutil
import yaml
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utils import set_device
from Graph_generation import sample_padded_grid_batch_shortest_path
from plotting_utils import plot_graph_grid_activations, plot_loss, plot_roc, plot_input_target_output, plot_message_passing_layers, plot_message_passing_layers_units
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from models import get_forward_function
from utils import rng_sequence_from_rng
from grid_run_config import GridConfig

# @title Graph net functions
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    metavar="-C",
    default="config.yaml",
    help="path to base configuration file.",
)

def run(config_path,config):
    train_on_shortest_path = config.train_on_shortest_path
  # @param
    log_every = config.num_training_steps // 10
    if config.weighted:
        edege_lables=True
    else:
        edege_lables = False
    if config.wandb_on:
        dateTimeObj = datetime.now()
        wandb.init(project="graph-brain", entity="graph-brain",
                   name=f"Grid_shortest_path" + dateTimeObj.strftime("%d%b_%H_%M"))
        wandb_logs = {}
        save_path= wandb.run.dir
        os.mkdir(os.path.join(save_path, 'results'))
        save_path = os.path.join(save_path, 'results')
    else:
        dateTimeObj = datetime.now()
        save_path = os.path.join(Path(os.getcwd()).resolve(), 'results')
        os.mkdir(os.path.join(save_path,  f"Grid_shortest_path" + dateTimeObj.strftime("%d%b_%H_%M")))
        save_path = os.path.join(os.path.join(save_path,  f"Grid_shortest_path" + dateTimeObj.strftime("%d%b_%H_%M")))
    # SAVING Trainning Files
    path = os.path.join(save_path, 'run.py')
    HERE = os.path.join(Path(os.getcwd()).resolve(), 'run.py')
    shutil.copyfile(HERE, path)

    path = os.path.join(save_path, 'Graph_generation.py')
    HERE = os.path.join(Path(os.getcwd()).resolve(), 'Graph_generation.py')
    shutil.copyfile(HERE, path)

    path = os.path.join(save_path, 'utils.py')
    HERE = os.path.join(Path(os.getcwd()).resolve(), 'utils.py')
    shutil.copyfile(HERE, path)

    path = os.path.join(save_path, 'plotting_utils.py')
    HERE = os.path.join(Path(os.getcwd()).resolve(), 'plotting_utils.py')
    shutil.copyfile(HERE, path)

    path = os.path.join(save_path, 'config_run.yaml')
    HERE = os.path.join(Path(os.getcwd()).resolve(), 'config.yaml')
    shutil.copyfile(HERE, path)

    # This is the function that does the forward pass of the model
    forward = get_forward_function(config.num_hidden, config.num_layers,config.num_message_passing_steps)
    net_hk = hk.without_apply_rng(hk.transform(forward))
    rng = jax.random.PRNGKey(config.seed)
    rng_seq = rng_sequence_from_rng(rng)
    if config.train_on_shortest_path:
        graph, targets = sample_padded_grid_batch_shortest_path(rng, config.batch_size,config.feature_position,config.weighted, config.nx_min, config.nx_max)
    else:
        graph, targets = sample_padded_grid_batch_shortest_path(rng, config.batch_size, config.feature_position,config.weighted,config.nx_min, config.nx_max)
        targets = graph.nodes
    params = net_hk.init(rng, graph)

    def compute_loss(params, model, inputs, targets):
        # not jitted because it will get jitted in jax.value_and_grad
        outputs = model.apply(params, inputs)
        return jnp.mean((outputs[0].nodes - targets) ** 2)  # using MSE

    # Set up optimizer.
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(grads, opt_state, params):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params

    def evaluate(model, params, inputs,target):
        outputs=model.apply(params, inputs)
        roc_auc = roc_auc_score(np.squeeze(target), np.squeeze(outputs[0].nodes))
        MCC= matthews_corrcoef(np.squeeze(target), round(np.squeeze(outputs[0].nodes)))

        return outputs, roc_auc, MCC

    losses = []
    losses_test = []
    roc_aucs_train= []
    MCCs_train = []
    MCCs_test = []
    roc_aucs_test = []
    rng = next(rng_seq)
    graph_test, target_test = sample_padded_grid_batch_shortest_path(rng, config.batch_size_test, config.feature_position,config.weighted,config.nx_min_test,config.nx_max_test)
    for n in range(config.num_training_steps):
        rng= next(rng_seq)
        # Sample a new batch of graph every itterations
        if config.resample:
            if train_on_shortest_path:
                graph, targets = sample_padded_grid_batch_shortest_path(rng, config.batch_size, config.feature_position,config.weighted,config.nx_min, config.nx_max)
            else:
                graph, targets = sample_padded_grid_batch_shortest_path(rng, config.batch_size,config.feature_position,config.weighted, config.nx_min, config.nx_max)
                targets = graph.nodes
        # Train
        loss, grads = jax.value_and_grad(compute_loss)(params, net_hk, graph, targets)  # jits inside of value_and_grad
        params = update_step(grads, opt_state, params)
        losses.append(loss)
        outputs_train, roc_auc_train,  MCC_train = evaluate(net_hk, params, graph, targets)
        roc_aucs_train.append(roc_auc_train)
        MCCs_train.append(MCC_train) # Matthews correlation coefficient
        # Test # model should basically learn to do nothing from this
        loss_test = compute_loss(params, net_hk, graph_test, target_test)
        losses_test.append(loss_test)
        outputs_test, roc_auc_test, MCC_test = evaluate(net_hk, params, graph_test, target_test)
        roc_aucs_test.append(roc_auc_test)
        MCCs_test.append(MCC_test)

        # Log
        wandb_logs = {'loss': loss, 'losses_test': loss_test ,'roc_auc_test':roc_auc_test, 'roc_auc':roc_auc_train }
        if config.wandb_on:
            wandb.log(wandb_logs)
        if n % log_every == 0:
            print(f'Training step {n}: loss = {loss}')

    # EVALUATE
    outputs, roc_auc, MCC = evaluate(net_hk, params, graph_test,target_test)
    print('roc_auc_score')
    print(roc_auc)
    print('MCC')
    print(MCC)

    # SAVE PARAMETER (NOT WE SAVE THE FILES SO IT SHOULD BE THERE AS WELL )
    if config.wandb_on:
        with open('readme.txt', 'w') as f:
            f.write('readme')
        with open(os.path.join(save_path, 'Constant.txt'), 'w') as outfile:
            outfile.write('num_message_passing_steps' + str( config.num_message_passing_steps) + '\n')
            outfile.write('Learning_rate:' + str(config.learning_rate) + '\n')
            outfile.write('num_training_steps:' + str(config.num_training_steps))
            outfile.write('roc_auc' + str(roc_auc))
            outfile.write('MCC' + str(MCC))

    # PLOTTING THE LOSS and AUC ROC
    plot_loss(losses, os.path.join(save_path, 'Losses.pdf'), 'Losses')
    plot_loss(losses_test, os.path.join(save_path, 'Losses_test.pdf'), 'Losses_test')
    plot_roc(roc_aucs_test, os.path.join(save_path, 'auc_roc_test.pdf'), 'auc_roc_test')
    plot_roc(roc_aucs_train, os.path.join(save_path, 'auc_roc_train.pdf'), 'auc_roc_train')
    plot_roc(MCCs_train, os.path.join(save_path, 'MCC_train.pdf'), 'MCC_train')
    plot_roc(MCCs_test, os.path.join(save_path, 'MCC_test.pdf'), 'MCC_test')

    # PLOTTING ACTIVATION OF THE FIRST 2 GRAPH OF THE BATCH
    plot_input_target_output(list(graph_test.nodes.sum(-1)), target_test.sum(-1), outputs[0].nodes.tolist(), graph_test, 4,edege_lables, os.path.join(save_path, 'in_out_targ.pdf'))
    plot_message_passing_layers(list(graph_test.nodes.sum(-1)), outputs[1],target_test.sum(-1),outputs[0].nodes.tolist(),graph_test,3,config.num_message_passing_steps,edege_lables, os.path.join(save_path, 'message_passing_graph.pdf'))
    # plot_message_passing_layers_units(outputs[1], target_test.sum(-1), outputs[0].nodes.tolist(),graph_test,config.num_hidden,config.num_message_passing_steps,edege_lables,os.path.join(save_path, 'message_passing_hidden_unit.pdf'))

    # Plot each seperatly
    plot_graph_grid_activations(outputs[0].nodes.tolist(), graph_test, os.path.join(save_path, 'outputs.pdf'),
             'Predicted Node Assignments with GCN',edege_lables)
    plot_graph_grid_activations(list(graph_test.nodes.sum(-1)), graph_test, os.path.join(save_path, 'Inputs.pdf'),
                              'Inputs node assigments',edege_lables)
    plot_graph_grid_activations(target_test.sum(-1), graph_test,  os.path.join(save_path, 'Target.pdf'), 'Target',edege_lables)

    plot_graph_grid_activations(outputs[0].nodes.tolist(), graph_test,
                                os.path.join(save_path, 'outputs_2.pdf'),
                                'Predicted Node Assignments with GCN',edege_lables, 2)
    plot_graph_grid_activations(list(graph_test.nodes.sum(-1)), graph_test,
                                os.path.join(save_path, 'Inputs_2.pdf'),
                                'Inputs node assigments',edege_lables, 2)
    plot_graph_grid_activations(target_test.sum(-1), graph_test,
                                os.path.join(save_path, 'Target_2.pdf'), 'Target',edege_lables, 2)
    return losses,roc_auc

if __name__ == "__main__":
    args = parser.parse_args()
    print("Hello, World!")
    set_device()
    config_class = GridConfig
    config = config_class(args.config_path)
    losses,auroc= run(config_path=args.config_path,config=config)


