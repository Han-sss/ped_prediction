import torch
import torch.nn as nn
import torch.nn.functional as FF
from torch.autograd import Variable


class Frame_Encoder(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super().__init__()
        self.l1 = nn.Linear(dim_in, dim_hid)
        self.l2 = nn.Linear(dim_hid, dim_out)

    def forward(self, inputs):
        # out: [batch_size x frames x dim_out]
        hid = FF.relu(self.l1(inputs))
        out = self.l2(hid)

        return out

class Encoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, in_size, frames):
        super(Encoder, self).__init__()

        self.frames = frames

        # Initialize source embedding
        layers = []
        for layer_idx in range(n_layers):
            input_size = in_size if layer_idx == 0 else hidden_size
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, False))
        self.layers = nn.ModuleList(layers)
                                          
    def forward(self, inputs):
        # h: [batch_size x length x out_size]
        h = inputs

        cell_states, hidden_states = [], []
        for layer in self.layers:
            c, h = layer(h)  # c, h: [batch_size x length x hidden_size]            

            # c_last, h_last: [batch_size, hidden_size]           
            c_last = c[list(range(len(inputs))), self.frames-1,:]
            h_last = h[list(range(len(inputs))), self.frames-1,:]
            cell_states.append(c_last)
            hidden_states.append((h_last, h))

        # return lists of cell states and hidden states of each layer
        return cell_states, hidden_states


class Decoder(nn.Module):
    def __init__(self, qrnn_layer, n_layers, kernel_size,
                 hidden_size, batch_size, frames, in_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.frames = frames
        self.in_size = in_size
        
        layers = []
        for layer_idx in range(n_layers):
            input_size = in_size if layer_idx == 0 else hidden_size
            use_attn = True if layer_idx == n_layers-1 else False
            layers.append(qrnn_layer(input_size, hidden_size, kernel_size, use_attn))
        self.layers = nn.ModuleList(layers)
                                          
    def forward(self, init_states, memories, device):
        assert len(self.layers) == len(memories)

        cell_states, hidden_states = [], []
        # h: [batch_size x frames x in_size]
        h = torch.zeros(self.batch_size, self.frames , self.in_size).to(device)
        for layer_idx, layer in enumerate(self.layers):
            state = None if init_states is None else init_states[layer_idx]
            memory = memories[layer_idx]

            c, h = layer(h, state, memory)

            cell_states.append(c); hidden_states.append(h)

        # The shape of the each state: [batch_size x frames x hidden_size]
        # return lists of cell states and hidden_states
        return cell_states, hidden_states


class QRNNModel(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, qrnn_layer, n_layers, kernel_size, hidden_size,
                 batch_size, frames, dec_size, out_size, device):
        super(QRNNModel, self).__init__()

        self.device = device

        self.frame = Frame_Encoder(dim_in, dim_hid, dim_out)
        self.encoder = Encoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               dim_out, frames)
        self.decoder = Decoder(qrnn_layer, n_layers, kernel_size, hidden_size,
                               batch_size, frames, dec_size)
        self.proj_linear = nn.Linear(hidden_size, out_size)

    
    def frame_encode(self, inputs):
        return self.frame(inputs)

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, init_states, memories):
        cell_states, hidden_states = self.decoder(init_states, memories, self.device)
        # return:
        # projected hidden_state of the last layer: logit
        #   first reshape it to [batch_size x frame x hidden_size]
        #   after projection: [batch_size x frame x out_size]
        h_last = hidden_states[-1]

        return cell_states, self.proj_linear(h_last)

    def forward(self, frame_inputs):
        enc_inputs = self.frame_encode(frame_inputs)
        init_states, memories = self.encode(enc_inputs)
        # logits: [batch_size x frames x hidden_size]
        _, logits = self.decode(init_states, memories)
        return logits
        