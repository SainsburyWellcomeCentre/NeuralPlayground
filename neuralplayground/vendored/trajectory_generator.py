# -*- coding: utf-8 -*-
""" This code is from another repository, please visit their repository for more information:
https://github.com/ganguli-lab/grid-pattern-formation/blob/master/trajectory_generator.py

It is under Apache Licence 2.0
https://github.com/ganguli-lab/grid-pattern-formation/blob/master/LICENSE

This code has some changes from the source code cited above, later this code should be documented
following neuralplaground standards, to then add a NOTICE of changes in the code according to the licence.
This disclaimer is a placeholder for the proper acknowledgement of the original code for when merging it with
the neuralplayground main codebase.
"""


import numpy as np
import torch


class TrajectoryGenerator(object):
    def __init__(self, sequence_length, batch_size, room_width, room_depth, device, place_cells=None, periodic=False):
        # options will be removed to replace it with explicit arguments
        # self.options = options
        self.sequence_length = sequence_length
        self.periodic = periodic
        self.batch_size = batch_size
        self.room_width = room_width
        self.room_depth = room_depth
        self.device = device
        self.place_cells = place_cells  # Place cells not implemented yet

    def avoid_wall(self, position, hd, room_width, room_depth):
        """
        Compute distance and angle to nearest wall
        """
        x = position[:, 0]
        y = position[:, 1]
        dists = [room_width / 2 - x, room_depth / 2 - y, room_width / 2 + x, room_depth / 2 + y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2 * np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(self, room_width, room_depth, batch_size):
        """Generate a random walk in a rectangular box"""
        samples = self.sequence_length
        # TODO: Convert this number to arguments of the generator
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 0.13 * 2 * np.pi  # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples + 2, 2])
        head_dir = np.zeros([batch_size, samples + 2])
        position[:, 0, 0] = np.random.uniform(-room_width / 2, room_width / 2, batch_size)
        position[:, 0, 1] = np.random.uniform(-room_depth / 2, room_depth / 2, batch_size)
        head_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        velocity = np.zeros([batch_size, samples + 2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])
        random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
        v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

        for t in range(samples + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            if not self.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:, t], head_dir[:, t], room_width, room_depth)
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack([np.cos(head_dir[:, t]), np.sin(head_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            # Rotate head direction
            head_dir[:, t + 1] = head_dir[:, t] + turn_angle

        # Periodic boundaries
        if self.periodic:
            position[:, :, 0] = np.mod(position[:, :, 0] + room_width / 2, room_width) - room_width / 2
            position[:, :, 1] = np.mod(position[:, :, 1] + room_depth / 2, room_depth) - room_depth / 2

        head_dir = np.mod(head_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        traj = {}
        # Input variables
        traj["init_hd"] = head_dir[:, 0, None]
        traj["init_x"] = position[:, 1, 0, None]
        traj["init_y"] = position[:, 1, 1, None]

        traj["ego_v"] = velocity[:, 1:-1]
        ang_v = np.diff(head_dir, axis=-1)
        traj["phi_x"], traj["phi_y"] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

        # Target variables
        traj["target_hd"] = head_dir[:, 1:-1]
        traj["target_x"] = position[:, 2:, 0]
        traj["target_y"] = position[:, 2:, 1]

        return traj

    def get_batch_generator(self, batch_size=None, room_width=None, room_depth=None):
        """
        Returns a generator that yields batches of trajectories
        """
        if not batch_size:
            batch_size = self.batch_size
        if not room_width:
            room_width = self.room_width
        if not room_depth:
            room_depth = self.room_depth

        while True:
            traj = self.generate_trajectory(room_width, room_depth, batch_size)

            # Velocity vector
            v = np.stack([traj["ego_v"] * np.cos(traj["target_hd"]), traj["ego_v"] * np.sin(traj["target_hd"])], axis=-1)
            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

            # Target position
            pos = np.stack([traj["target_x"], traj["target_y"]], axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
            # Put on GPU if GPU is available
            pos = pos.to(self.device)

            # Place cell activity at target position
            place_outputs = self.place_cells.get_activation(pos)

            # initial position in place cell space
            init_pos = np.stack([traj["init_x"], traj["init_y"]], axis=-1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to(self.device)
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            v = v.to(self.device)
            inputs = (v, init_actv)

            yield (inputs, place_outputs, pos)

    def get_test_batch(self, traj=None, batch_size=None, room_width=None, room_depth=None):
        """For testing performance, returns a batch of smample trajectories"""
        if not batch_size:
            batch_size = self.batch_size
        if not room_width:
            room_width = self.room_width
        if not room_depth:
            room_depth = self.room_depth
        if not traj:
            traj = self.generate_trajectory(room_width, room_depth, batch_size)

        v = np.stack([traj["ego_v"] * np.cos(traj["target_hd"]), traj["ego_v"] * np.sin(traj["target_hd"])], axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

        pos = np.stack([traj["target_x"], traj["target_y"]], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.device)
        place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj["init_x"], traj["init_y"]], axis=-1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.device)
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        v = v.to(self.device)
        inputs = (v, init_actv)

        return (inputs, pos, place_outputs)
