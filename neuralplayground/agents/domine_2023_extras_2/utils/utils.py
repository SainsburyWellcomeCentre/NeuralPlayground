import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import torch

def set_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print(
            "WARNING: For this notebook to perform best, "
            "if possible, in the menu under `Runtime` -> "
            "`Change runtime type.`  select `GPU` "
        )
    else:
        print("GPU is enabled in this notebook.")


def update_outputs_test(outputs, indices):
    outputs_wse = outputs[0].nodes
    for ind in indices:
        outputs_wse = outputs_wse.at[ind].set(0)
    return outputs_wse

