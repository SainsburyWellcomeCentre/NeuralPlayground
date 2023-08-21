name = "models"
from importlib.metadata import PackageNotFoundError, version
from .agent_core import AgentCore, RandomAgent, LevyFlightAgent
from .stachenfeld_2018 import Stachenfeld2018
from .weber_2018 import Weber2018

# Check if torch is installed
try:
    TORCH_VERSION = version("torch")
except PackageNotFoundError as e:
    # raise warning about Whittington2020 requiring torch
    raise Warning(
        "The Whittington2020 agent (Tolman-Eichenbaum Machine) "
        "requires pytorch, which could not be found in your environment. "
        "If you want to use this agent, see https://pytorch.org/get-started/locally/ "
        "for installation instructions."
        "You can still use all other agents in neuralplayground.agents."
    )
else:
    from .whittington_2020 import Whittington2020
