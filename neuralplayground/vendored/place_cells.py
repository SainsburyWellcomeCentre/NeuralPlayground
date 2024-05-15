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
import scipy


class PlaceCells(object):

    # Here we replace the options object with explicit arguments
    def __init__(self, Np, place_cell_rf, surround_scale, room_width, room_depth, periodic, DoG, device):
    #def __init__(self, options, us=None):
        self.Np = Np
        self.sigma = place_cell_rf
        self.surround_scale = surround_scale
        self.room_width = room_width
        self.room_depth = room_depth
        self.is_periodic = periodic
        self.DoG = DoG
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)

        # Randomly tile place cell centers across environment
        np.random.seed(0)
        usx = np.random.uniform(-self.room_width / 2, self.room_width / 2, (self.Np,))
        usy = np.random.uniform(-self.room_width / 2, self.room_width / 2, (self.Np,))
        self.us = torch.tensor(np.vstack([usx, usy]).T)
        # If using a GPU, put on GPU
        self.us = self.us.to(self.device)
        # self.us = torch.tensor(np.load('models/example_pc_centers.npy')).cuda()

    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        d = torch.abs(pos[:, :, None, :] - self.us[None, None, ...]).float()

        if self.is_periodic:
            dx = d[:, :, :, 0]
            dy = d[:, :, :, 1]
            dx = torch.minimum(dx, self.room_width - dx)
            dy = torch.minimum(dy, self.room_depth - dy)
            d = torch.stack([dx, dy], axis=-1)

        norm2 = (d ** 2).sum(-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on
        # average and seems to speed up training.
        outputs = self.softmax(-norm2 / (2 * self.sigma ** 2))

        if self.DoG:
            # Again, normalize with prefactor
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= self.softmax(-norm2 / (2 * self.surround_scale * self.sigma ** 2))

            # Shift and scale outputs so that they lie in [0,1].
            min_output, _ = outputs.min(-1, keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)
        return outputs

    def get_nearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.

        Args:
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''
        _, idxs = torch.topk(activation, k=k)
        pred_pos = self.us[idxs].mean(-2)
        return pred_pos

    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid'''
        coordsx = np.linspace(-self.room_width / 2, self.room_width / 2, res)
        coordsy = np.linspace(-self.room_depth / 2, self.room_depth / 2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        pc_outputs = pc_outputs.reshape(-1, self.Np)

        T = pc_outputs.shape[0]  # T vs transpose? What is T? (dim's?)
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            gridval = scipy.interpolate.griddata(self.us.cpu(), pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])

        return pc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.room_width / 2, self.room_width / 2, res),
                                   np.linspace(-self.room_depth / 2, self.room_depth / 2, res))).T

        pos = torch.tensor(pos)

        # Put on GPU if available
        pos = pos.to(self.device)

        # Maybe specify dimensions here again?
        pc_outputs = self.get_activation(pos).reshape(-1, self.Np).cpu()

        C = pc_outputs @ pc_outputs.T
        Csquare = C.reshape(res, res, res, res)

        Cmean = np.zeros([res, res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i, j], -i, axis=0), -j, axis=1)

        Cmean = np.roll(np.roll(Cmean, res // 2, axis=0), res // 2, axis=1)

        return Cmean
