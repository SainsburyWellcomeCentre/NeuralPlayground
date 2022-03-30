import numpy as np
from simple2d import Simple2D


class ConnectedRooms(Simple2D):

    def __init__(self, environment_name="ConnectedRooms", corridor_width=40.0, **env_kwargs):
        super().__init__(environment_name, **env_kwargs)

