import numpy as np
import torch
import pickle
from tqdm import tqdm
import pandas as pd


class Sorscher2022exercise(torch.nn.Module):
    def __init__(
        self, Ng, Np, sequence_length, weight_decay, place_cells, device,
            activation=torch.nn.ReLU, learning_rate=5e-3, learning_rule="adam"
    ):
        super().__init__()
        self.Ng = Ng
        self.Np = Np
        self.sequence_length = sequence_length
        self.weight_decay = weight_decay
        self.place_cells = place_cells
        self.activation = activation
        self.learning_rate = learning_rate
        self.device = device
        self.dtype = torch.float32
        if activation == "tanh":
            self.non_linearity = torch.tanh
        elif activation == "relu":
            self.non_linearity = torch.nn.ReLU()
        else:
            self.non_linearity = torch.nn.Identity()
        self._initialize_weights()
        if learning_rule == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif learning_rule == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        self.loss_hist = []
        self.pos_err_hist = []
        self.to(self.device)


    def _initialize_weights(self):
        # Input weights
        # Ideally we would use uniform initialization, between -np.sqrt(in_features), +np.sqrt(in_features)
        k_g = 1/self.Ng
        k_p = 1/self.Np
        np_encoder_W = np.random.uniform(-np.sqrt(k_g), np.sqrt(k_g), size=(self.Ng, self.Np))
        recurrent_W = np.random.uniform(-np.sqrt(k_g), np.sqrt(k_g), size=(self.Ng, self.Ng))
        np_velocity_W = np.random.uniform(-np.sqrt(k_g), np.sqrt(k_g), size=(self.Ng, 2))
        np_decoder_W = np.random.uniform(-np.sqrt(k_p), np.sqrt(k_p), size=(self.Ng, self.Np))

        self.encoder_W = torch.nn.Parameter(torch.tensor(np_encoder_W, dtype=self.dtype, device=self.device),
                                            requires_grad=True)
        self.recurrent_W = torch.nn.Parameter(torch.tensor(recurrent_W, dtype=self.dtype, device=self.device),
                                              requires_grad=True)
        self.velocity_W = torch.nn.Parameter(torch.tensor(np_velocity_W, dtype=self.dtype, device=self.device),
                                             requires_grad=True)
        self.decoder_W = torch.nn.Parameter(torch.tensor(np_decoder_W, dtype=self.dtype, device=self.device),
                                            requires_grad=True)

        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        """
        Args:
            inputs: tuple with velocity and initial place cell activity with shapes [seq_len, batch, 2] and [batch, Np].

        Returns:
            g: Batch of grid cell activations with shape [sequence_length, batch, Ng].
        """
        velocity, init_place_cell = inputs
        initial_states = init_place_cell @ self.encoder_W.T
        # g, _ = self.RNN(velocity, initial_states)

        batch_size = velocity.shape[1]
        h_t_minus_1 = initial_states
        h_t = initial_states
        g_cell_activity = []
        for t in range(self.sequence_length):
            # No bias
            linear_input = velocity[t] @ self.velocity_W.T + h_t_minus_1 @ self.recurrent_W.T
            h_t = self.non_linearity(linear_input)
            g_cell_activity.append(h_t)
            h_t_minus_1 = h_t
        g_cell_activity = torch.stack(g_cell_activity)
        return g_cell_activity

    def predict(self, inputs):
        g_cell_activity = self.g(inputs)
        pred_place_cells = g_cell_activity @ self.decoder_W
        return pred_place_cells

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
        loss += self.weight_decay * (self.recurrent_W**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos) ** 2).sum(-1)).mean()
        return loss, err

    def bptt_update(self, inputs, place_cells_activity, position):
        """
        Perform backpropagation through time and update weights.
        """
        self.optimizer.zero_grad()
        loss, err = self.compute_loss(inputs, place_cells_activity, position)
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy(), err.detach().cpu().numpy()

    def train_RNN(self, data_generator, training_steps):
        """
        Perform backpropagation through time and update weights.
        """
        for i in tqdm(range(training_steps)):
            # Inputs below is a tuple with velocity vector and initial place cell activity
            # pc outputs is the place cell activity for the whole trajectory, these are the target of the bptt step
            inputs, pc_outputs, positions = next(data_generator)
            loss, pos_err = self.bptt_update(inputs=inputs, place_cells_activity=pc_outputs, position=positions)
            self.loss_hist.append(loss)
            self.pos_err_hist.append(pos_err)

        return self.loss_hist, self.pos_err_hist

    def save_model(self, path):
        torch.save({
            'loss_hist': self.loss_hist,
            'pos_err_hist': self.pos_err_hist,
            'Ng': self.Ng,
            'Np': self.Np,
            'sequence_length': self.sequence_length,
            'weight_decay': self.weight_decay,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'non_linearity': self.non_linearity,
            'softmax': self.softmax,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.loss_hist = checkpoint['loss_hist']
        self.pos_err_hist = checkpoint['pos_err_hist']
        self.Ng = checkpoint['Ng']
        self.Np = checkpoint['Np']
        self.sequence_length = checkpoint['sequence_length']
        self.weight_decay = checkpoint['weight_decay']
        self.activation = checkpoint['activation']
        self.learning_rate = checkpoint['learning_rate']
        self.device = checkpoint['device']
        self.non_linearity = checkpoint['non_linearity']
        self.softmax = checkpoint['softmax']
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class SorscherIdealRNN(object):

    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self._initialize_explicit_weights()

    def _initialize_explicit_weights(self):
        """ This is from Sorscher equations 35, 36 abd 37 """
        self.k0 = np.array([1, 0])
        self.k60 = np.array([0.5, np.sqrt(3) / 2])
        self.k120 = np.array([-0.5, np.sqrt(3) / 2])
        self.k_vec = np.stack([self.k0, self.k60, self.k120], axis=0)
        self.L = np.sqrt(self.n_neurons).astype(int)

        Jij = np.zeros((self.n_neurons, self.n_neurons))
        offset_Jij = np.zeros((self.n_neurons, self.n_neurons))
        grid_location = np.arange(self.L)

        print("Building recurrent matrix")
        sheet_locations = []
        for i in range(self.L):
            for j in range(self.L):
                sheet_locations.append(np.array([grid_location[i], grid_location[j]]))
        sheet_locations = np.stack(sheet_locations, axis=0)
        # Correct version of equation velocity modulation
        Mx = np.mod(sheet_locations[:, 1], 2)*((-1)**(sheet_locations[:, 0]))
        My = np.mod(sheet_locations[:, 1]+1, 2)*((-1)**(sheet_locations[:, 0]))
        Mixy = np.stack([Mx, My], axis=1)
        for i in range(self.n_neurons):
            si = sheet_locations[i, :]
            for j in range(self.n_neurons):
                sj = sheet_locations[j, :]
                s_diff = si - sj - Mixy[j, :]
                offset_Jij[i, j] = self.weight_function(s_diff[:, np.newaxis])
                s_diff = si - sj
                Jij[i, j] = self.weight_function(s_diff[:, np.newaxis])
        self.Jij = Jij
        self.offset_Jij = offset_Jij
        self.Mixy = Mixy
        self.bi = np.ones((self.n_neurons, 1))*0.1
        self.sheet_locations = sheet_locations

    def weight_function(self, x):
        # Make sure x.shape = (dim, 1), eq 37 in Sorscher
        inner = 2 * np.pi / self.L * (self.k_vec @ x)
        element_wise_cos = np.cos(inner)
        return np.sum(element_wise_cos)

    def rate_update(self, rates, velocity):
        # Equation 35 in Sorscher
        matrix_product = self.Jij @ rates
        velocity_product = self.Mixy @ velocity
        new_rates = matrix_product + velocity_product + self.bi
        return npRelu(new_rates)


def npRelu(x):
    return np.maximum(0, x)

def npsigmoid(z):
    return 1/(1 + np.exp(-z))



