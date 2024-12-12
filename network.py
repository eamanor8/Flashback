# OK 

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Rnn(Enum):
    ''' The available RNN units '''

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))

class RnnFactory():
    ''' Creates the desired RNN unit. '''

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        """
        Creates an RNN unit based on the specified type with the given hidden size.

        Args:
            hidden_size (int): The number of features in the hidden state of the RNN.

        Returns:
            nn.Module: An RNN, GRU, or LSTM module initialized with the specified hidden size.
        """
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)

class Flashback(nn.Module):
    ''' Flashback RNN: Applies weighted average using spatial and tempoarl data in combination
    of user embeddings to the output of a generic RNN unit (RNN, GRU, LSTM).
    '''

    #* input_size is loc_count, the number of locations in total across all users
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory):
        super().__init__()
        self.input_size = input_size # location count
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size) # location embedding
        self.user_encoder = nn.Embedding(user_count, hidden_size) # user embedding
        self.rnn = rnn_factory.create(hidden_size)  #* seq2seq  (many-to-many RNN)
        self.fc = nn.Linear(2*hidden_size, input_size) # create outputs in lenght of locations  #* many-to-one RNN
                                                                                                #! but actually by the way forward() works, it is many-to-many (see shape of the returned y_linear object: one prediction is made for each leading substring of sequence_length==20)
        #* self.fc turns a (2*hidden_size==20) vector into a (loc_count==total number of locations) vector (can be large!)
        #TODO NB: Can never predict never-before-seen locations in the train dataset.

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        #* for shapes of x, t, s, y, y_t, y_s, active_users, see bottom of dataset.py
        #* note that squeeze has been called, so there isn't a prepended batch dimension. The
        #* dimensions stated in the bottom of dataset.py apply here exactly.
        #*! NB: if this is called from trainer.evaluate(), then active_user has NOT been squeezed.
        #*!     if this is called from trainer.loss(), then active_user has been squeezed.
        #*!     It doesn't matter.

        # print(f"active_user={active_user}; x={x}")

        # Ensure active_user is on the same device as the model
        # active_user = active_user.to(x.device)  # Move to the same device as input `x`

        seq_len, user_len = x.size()  #* user_len is the batch_size so yes user_len (see near bottom of dataset.py)
                                      #* x is of shape (sequence_length==20, batch_size). Loc IDs.
        x_emb = self.encoder(x)       #* shape is (sequence_length==20, batch_size, hidden_dim==10)
        out, h = self.rnn(x_emb, h)   #* h is of shape (1, batch_size, hidden_dim==10)
                                      #* out is of shape (sequence_length==20, batch_size, hidden_dim==10)  [the upwards arrows out of an RNN unrolled-view]
                                      #* h is of shape (1, batch_size, hidden_dim==10)

        # comopute weights per user  #* these are the spatio-temporal weights MULTIPLIED BY the hidden states, hence the self.hidden_size dimension
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)  #* shape is (batch_size, 1)
            for j in range(i+1):
                dist_t = t[i] - t[j]  #* shape is (batch_size,)
                dist_s = torch.norm(s[i] - s[j], dim=-1)  #* shape is (batch_size,)
                a_j = self.f_t(dist_t, user_len)  #* second arg not used; shape is (batch_size,)
                b_j = self.f_s(dist_s, user_len)  #* second arg not used; shape is (batch_size,)
                a_j = a_j.unsqueeze(1)  #* shape is (batch_size, 1)
                b_j = b_j.unsqueeze(1)  #* shape is (batch_size, 1)
                w_j = a_j*b_j + 1e-10 # small epsilon to avoid 0 division  #* shape is (batch_size, 1)
                sum_w += w_j
                out_w[i] += w_j*out[j]  #* out[j] shape is (batch_size, hidden_dim==10)
            # normliaze according to weights
            out_w[i] /= sum_w  #* sum across ALL O(n^2) weights associated with deltas (time and space)

        # add user embedding:
        p_u = self.user_encoder(active_user)  #* shape is (1, batch_size, hidden_size==10); active_user shape is (1, batch_size)
        p_u = p_u.view(user_len, self.hidden_size)  #* shape is (batch_size, hidden_size==10)
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
        y_linear = self.fc(out_pu)  #* (sequence_length==20, batch_size, 2*hidden_size==20) |-> (sequence_length==20, batch_size, loc_count==total number of locations)
        return y_linear, h

'''
~~~ h_0 strategies ~~~
Initialize RNNs hidden states
'''

def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)

class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)

