'''
Hand-written recurrent encoders shared by the two Stage 4 methods.

PyTorch's `nn.RNN` / `nn.GRU` / `nn.LSTM` are fused kernels.  On the ROCm build used for
this project every new (batch, length) shape triggers a MIOpen kernel search of several
seconds, and the search result is not reused across processes, which makes variable length
mini-batches unusably slow.  Building the recurrence out of `nn.RNNCell` / `nn.GRUCell` /
`nn.LSTMCell` avoids that problem completely (a cell is one matrix multiplication, which the
BLAS library caches), and it also makes the "write your own RNN" part of Stage 4 explicit:

    h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)                  (vanilla RNN)
    r_t, z_t = sigmoid(...),  h_t = (1 - z_t) * n_t + z_t * h_{t-1}    (GRU)
    i_t, f_t, o_t, g_t,  c_t = f_t * c_{t-1} + i_t * g_t,              (LSTM)
                         h_t = o_t * tanh(c_t)

The encoder supports several stacked layers and a bidirectional pass; the state is frozen
whenever the current position is padding, so the padding of a mini batch never influences
the representation of a review or a joke.
'''

import torch
import torch.nn as nn

SUPPORTED_RNN_TYPES = ('rnn', 'gru', 'lstm')


def check_rnn_type(rnn_type):
    name = str(rnn_type).lower()
    if name not in SUPPORTED_RNN_TYPES:
        raise ValueError('unsupported recurrent unit: ' + str(rnn_type)
                         + ' (expected one of ' + ', '.join(SUPPORTED_RNN_TYPES) + ')')
    return name


def _build_cell(rnn_type, input_size, hidden_size):
    name = check_rnn_type(rnn_type)
    if name == 'rnn':
        # check here for the nn.RNNCell doc:
        # https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
        return nn.RNNCell(input_size, hidden_size, nonlinearity='tanh')
    if name == 'gru':
        # check here for the nn.GRUCell doc:
        # https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
        return nn.GRUCell(input_size, hidden_size)
    # check here for the nn.LSTMCell doc:
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
    return nn.LSTMCell(input_size, hidden_size)


class Recurrent_Encoder(nn.Module):
    '''
    A stack of recurrent layers, optionally bidirectional.

    forward(x, lengths=None, state=None) returns (output, state) where
        x      : [batch, time, input_size]
        output : [batch, time, hidden_size * directions]
        state  : list (one entry per layer) of per-direction states, used by the incremental
                 decoding of the language model.
    '''

    def __init__(self, rnn_type, input_size, hidden_size, num_layers=1, bidirectional=False,
                 dropout=0.0):
        super().__init__()
        self.rnn_type = check_rnn_type(rnn_type)
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)
        self.directions = 2 if self.bidirectional else 1

        self.cells = nn.ModuleList()
        for layer in range(self.num_layers):
            layer_input = self.input_size if layer == 0 else self.hidden_size * self.directions
            layer_cells = nn.ModuleList([_build_cell(self.rnn_type, layer_input,
                                                    self.hidden_size)
                                         for _ in range(self.directions)])
            self.cells.append(layer_cells)

    # ------------------------------------------------------------------ states
    def initial_state(self, batch_size, device=None):
        '''Zero states, one (h, c) pair per layer and direction (c is None for RNN / GRU)'''
        device = device or next(self.parameters()).device
        states = []
        for _ in range(self.num_layers):
            directions = []
            for _ in range(self.directions):
                hidden = torch.zeros(batch_size, self.hidden_size, device=device)
                memory = (torch.zeros(batch_size, self.hidden_size, device=device)
                          if self.rnn_type == 'lstm' else None)
                directions.append((hidden, memory))
            states.append(directions)
        return states

    @staticmethod
    def _step(cell, input_t, state):
        '''One cell step; returns the new (h, c) pair'''
        hidden, memory = state
        if isinstance(cell, nn.LSTMCell):
            hidden, memory = cell(input_t, (hidden, memory))
        else:
            hidden = cell(input_t, hidden)
            memory = None
        return hidden, memory

    # ------------------------------------------------------------------ forward
    def forward(self, x, lengths=None, state=None):
        batch_size, n_step = x.shape[0], x.shape[1]
        if state is None:
            state = self.initial_state(batch_size, x.device)

        if lengths is None:
            mask = None
        else:
            positions = torch.arange(n_step, device=x.device).unsqueeze(0)
            mask = positions < lengths.unsqueeze(1)

        layer_input = x
        new_state = []
        for layer in range(self.num_layers):
            direction_outputs = []
            layer_state = []
            for direction in range(self.directions):
                cell = self.cells[layer][direction]
                hidden, memory = state[layer][direction]
                steps = range(n_step) if direction == 0 else range(n_step - 1, -1, -1)
                outputs = [None] * n_step
                for step in steps:
                    input_t = layer_input[:, step, :]
                    new_hidden, new_memory = self._step(cell, input_t, (hidden, memory))
                    if mask is not None:
                        keep = mask[:, step].unsqueeze(1)
                        new_hidden = torch.where(keep, new_hidden, hidden)
                        if new_memory is not None:
                            new_memory = torch.where(keep, new_memory, memory)
                    hidden, memory = new_hidden, new_memory
                    outputs[step] = hidden
                direction_outputs.append(torch.stack(outputs, dim=1))
                layer_state.append((hidden, memory))
            layer_output = (torch.cat(direction_outputs, dim=2) if self.directions == 2
                            else direction_outputs[0])
            if self.dropout > 0 and layer < self.num_layers - 1:
                layer_output = nn.functional.dropout(layer_output, p=self.dropout,
                                                     training=self.training)
            layer_input = layer_output
            new_state.append(layer_state)

        self.last_state = new_state
        return layer_input, new_state

    # ------------------------------------------------------------------ helpers
    def output_size(self):
        return self.hidden_size * self.directions
