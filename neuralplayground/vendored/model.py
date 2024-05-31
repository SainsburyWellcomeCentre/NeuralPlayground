# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm


class Sorscher2022(torch.nn.Module):
    # Here we replace the options object with explicit arguments
    def __init__(
        self, Ng, Np, sequence_length, weight_decay, place_cells, activation="tanh", learning_rate=5e-3, device="cuda"
    ):
        super().__init__()
        self.Ng = Ng
        self.Np = Np
        self.sequence_length = sequence_length
        self.weight_decay = weight_decay
        self.place_cells = place_cells
        self.activation = activation
        self.learning_rate = learning_rate
        self.built_network()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def built_network(self):
        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2, hidden_size=self.Ng, nonlinearity=self.activation, bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        """
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        """
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN(v, init_state)
        return g

    def predict(self, inputs):
        """
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns:
            place_preds: Predicted place cell activations with shape
                [batch_size, sequence_length, Np].
        """
        place_preds = self.decoder(self.g(inputs))

        return place_preds

    def compute_loss(self, inputs, pc_outputs, pos):
        """
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        """
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        loss = -(y * torch.log(yhat)).sum(-1).mean()

        # Weight regularization
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()

        return loss, err

    def bptt_update(self, inputs, place_cells_activity, position):
        """
        Perform backpropagation through time and update weights.
        """
        self.zero_grad()
        loss, err = self.compute_loss(inputs, place_cells_activity, position)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy(), err.detach().cpu().numpy()

    def train_RNN(self, data_generator, training_steps):
        """
        Perform backpropagation through time and update weights.
        """
        loss_hist = []
        pos_err_hist = []

        for i in tqdm(range(training_steps)):
            # Inputs below is a tuple with velocity vector and initial place cell activity
            # pc outputs is the place cell activity for the whole trajectory, these are the target of the bptt step
            inputs, pc_outputs, positions = next(data_generator)
            loss, pos_err = self.bptt_update(inputs=inputs, place_cells_activity=pc_outputs, position=positions)
            loss_hist.append(loss)
            pos_err_hist.append(pos_err)

        # Save training results for later
        self.loss_hist = np.array(loss_hist)
        self.pos_err_hist = np.array(pos_err_hist)

        return self.loss_hist, self.pos_err_hist
