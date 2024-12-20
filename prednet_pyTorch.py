from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell

'''' Code adapted from # code adapted from https://github.com/eijwat/prednet_in_pytorch'''

class SatLU(nn.Module):
    def __init__(self, lower=0, upper=1, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)


    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
	        + ', max_val=' + str(self.upper) \
	        + inplace_str + ')'


class PredNet(nn.Module):
    def __init__(self, A_channels, loss_weight=[1, 0, 0, 0], R_channels=None,
                 output_mode='error',
                 round_mode="down_up_down",
                 device=torch.device("cpu")):
        super(PredNet, self).__init__()
        self.device = device
        if R_channels is None:
            R_channels = A_channels
        self.r_channels = R_channels + [0,]  # for convenience
        self.a_channels = A_channels
        self.n_layers = len(R_channels)
        self.output_mode = output_mode
        self.round_mode = round_mode
        self.loss = nn.MSELoss(reduction="none")
        self.loss_weight = loss_weight
        self.outputs = defaultdict(list)

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            cell = ConvLSTMCell(2 * self.a_channels[i] + self.r_channels[i + 1], self.r_channels[i],
                                (3, 3))
            setattr(self, 'cell{}'.format(i), cell)
            self.register_hooks(cell, "ConvLSTMCell_layer{}".format(i))

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)
            self.register_hooks(conv, "Conv_sequential_layer{}".format(i))


        self.upsample = nn.Upsample(scale_factor=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for l in range(self.n_layers - 1):
            update_A = nn.Sequential(nn.Conv2d(2 * self.a_channels[l], self.a_channels[l + 1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)
            self.register_hooks(update_A, "UpdateA_layer{}".format(l))

        self.reset_parameters()

    def register_hooks(self, mod, name):
        def hook_fn(m, input, output):
            if isinstance(output[0], torch.Tensor):
                self.outputs[name].append(output[0].detach().cpu())
            elif isinstance(output[0], tuple):
                self.outputs[name].append([o.detach().cpu() for o in output[0]])
        mod.register_forward_hook(hook_fn)

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def down_to_up(self, A, Ahat_seq, E_seq, total_loss_per_layer):
        ''' Down to up '''

        for l in range(self.n_layers):

            # positive and negative error units
            pos = F.relu(Ahat_seq[l] - A)
            neg = F.relu(A - Ahat_seq[l])
            # print(neg.shape)

            # concatenate error units
            E = torch.cat([pos, neg], 1)
            E_seq[l] = E
            # print('shape', E.shape)

            # add loss
            if l > 0:
                total_loss_per_layer[l, :] += self.loss(Ahat_seq[l], A).mean(axis=(1, 2, 3))
            # total_loss_per_layer[l, :] += E.mean(axis=(1, 2, 3))
            # total_loss_per_layer[l, :] += E.mean()

            # update A (output from error units), except for the top layer
            if l < self.n_layers - 1:
                update_A = getattr(self, 'update_A{}'.format(l))
                A = update_A(E)

    def up_to_down(self, t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, total_loss_per_layer, gt):
        ''' Up to down '''

        for l in reversed(range(self.n_layers)):

            cell = getattr(self, 'cell{}'.format(l))

            # get error, representation and representation of previous layer
            if t == 0:
                E = E_seq[l]
                R = R_seq[l]
                hx = (R, R)
            else:
                E = E_seq[l]
                R = R_seq[l]
                hx = H_seq[l]

            # update representation
            if l == self.n_layers - 1:
                R, hx = cell(E, hx)
            else:
                tmp = torch.cat((E, self.upsample(R_seq[l + 1])), 1)
                R, hx = cell(tmp, hx)

            R_seq[l] = R
            H_seq[l] = hx

            # compute target
            conv = getattr(self, 'conv{}'.format(l))
            Ahat_seq[l] = conv(R_seq[l])

            # compute loss on next-frame prediction
            if l == 0:
                frame_prediction = Ahat_seq[l]
                total_loss += self.loss(frame_prediction, gt).mean(axis=(1, 2, 3))
                total_loss_per_layer[l, :] += self.loss(frame_prediction, gt).mean(axis=(1, 2, 3))
        
        return frame_prediction

    def forward(self, input):

        # initiate lists
        self.outputs = defaultdict(list)
        R_seq       = [None] * self.n_layers
        H_seq       = [None] * self.n_layers # hidden state (i.e. representation of previous timestep)
        E_seq       = [None] * self.n_layers
        Ahat_seq    = [None] * self.n_layers

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        # initiate tensors
        for l in range(self.n_layers):
            E_seq[l]            = torch.zeros(batch_size, 2 * self.a_channels[l], w, h).to(self.device)
            R_seq[l]            = torch.zeros(batch_size, self.r_channels[l], w, h).to(self.device)
            Ahat_seq[l]         = torch.zeros(batch_size, self.a_channels[l], w, h).to(self.device)
            w                   = torch.div(w, 2, rounding_mode='trunc')
            h                   = torch.div(h, 2, rounding_mode='trunc')

        # iterate over network
        pred = []
        time_steps = input.size(1)
        total_loss = torch.zeros(batch_size).to(self.device)  
        total_loss_per_layer = torch.zeros(self.n_layers, batch_size).to(self.device)                                   
        
        # error for FIRST layer (frame prediction)
        eval_index = [torch.zeros(batch_size).to(self.device) for _ in range(self.n_layers)]    # error for ALL layers
        for t in range(time_steps - 1):

            # select target and grount truth
            A   = input[:, t]
            gt  = input[:, t + 1]

            if self.round_mode == "down_up_down":

                # bott-up projection
                self.down_to_up(A, Ahat_seq, E_seq, total_loss_per_layer)

                # top-down projection
                frame_prediction = self.up_to_down(t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, total_loss_per_layer, gt)

            else:

                # top-down projection
                frame_prediction = self.up_to_down(t, Ahat_seq, E_seq, R_seq, H_seq, total_loss, total_loss_per_layer, gt)
                
                # bott-up projection
                self.down_to_up(A, Ahat_seq, E_seq, total_loss_per_layer)

            # add frame to list
            pred.append(frame_prediction)

        # calculate eval index (error for ALL layers)
        A = input[:, t + 1]
        for l in range(self.n_layers):
            eval_index[l] = self.loss(Ahat_seq[l], A).mean(axis=(1, 2, 3))
            pos = F.relu(Ahat_seq[l] - A)
            neg = F.relu(A - Ahat_seq[l])
            E = torch.cat([pos, neg], 1)
            E_seq[l] = E
            if l < self.n_layers - 1:
                update_A = getattr(self, 'update_A{}'.format(l))
                A = update_A(E)

        if self.output_mode == 'error':
            return pred, total_loss, total_loss_per_layer, eval_index, E_seq
        else:
            return frame_prediction

