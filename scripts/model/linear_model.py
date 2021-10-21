import torch
import torch.nn as nn


class PoswiseFeedForward(nn.Module):
    def __init__(self, d_dim, mid_dim, n_seq, bias=True):
        super(PoswiseFeedForward,self).__init__()
        # self.L1 = torch.nn.utils.weight_norm(nn.Linear(d_dim, mid_dim, bias=bias))
        # self.L2 = torch.nn.utils.weight_norm(nn.Linear(mid_dim, d_dim, bias=bias))
        self.L1 = nn.Linear(d_dim, mid_dim, bias=bias)
        self.L2 = nn.Linear(mid_dim, d_dim, bias=bias)
        #self.LN = nn.LayerNorm(d_dim, elementwise_affine=False)
        self.LN  = nn.LayerNorm([n_seq, d_dim], elementwise_affine=False)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs
        output = self.L1(inputs)
        output = self.relu(output)
        output = self.L2(output)
        return self.LN(output + residual)

class Mean(nn.Module):
    def __init__(self, *args):
        super(Mean, self).__init__()
        self.index = args
    def forward(self, input):
        return torch.mean(input, dim=-2)

class LinearModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_seq, out_dim, layer_nums=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layer_nums):
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))
        self.layers.append(PoswiseFeedForward(hidden_dim, hidden_dim, n_seq))
        self.layers.append(Mean())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

if __name__ == '__main__':
    model = LinearModel(768, 768, 32, 10)
    input = torch.zeros(size=[10, 32, 768])
    res = model(input)
    print(res)
