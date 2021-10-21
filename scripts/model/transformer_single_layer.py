import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# import torch.nn.modules.transformer
class Manual_Mult_Attention(nn.Module):
    def __init__(self, d_dim, k_dim, nheads, n_seq, bias=False):
        super(Manual_Mult_Attention,self).__init__()
        self.d_dim = d_dim
        self.k_dim = k_dim
        self.nheads = nheads
        self.n_seq = n_seq
        self.head_dim = k_dim // nheads
        assert self.head_dim * self.nheads == self.k_dim, "k_dim must be divisible by nheads"
        self.scaling = self.head_dim ** -0.5

        self.wq = nn.Linear(self.d_dim, self.k_dim, bias=False)
        self.wk = nn.Linear(self.d_dim, self.k_dim, bias=False)
        self.wv = nn.Linear(self.d_dim, self.k_dim, bias=False)

        self.softpro = nn.Softmax(dim=-1)

        self.wo = nn.Linear(self.k_dim, self.d_dim, bias=False)
        #self.norm_layer = nn.LayerNorm(self.d_dim, elementwise_affine=False)
        self.norm_layer = nn.LayerNorm([self.n_seq, self.d_dim], elementwise_affine=False)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.wq)
    #     nn.init.xavier_uniform_(self.wk)
    #     nn.init.xavier_uniform_(self.wv)
    #     nn.init.xavier_uniform_(self.wo)

    def forward(self, Q, K, V):
        residual, batch_size = Q, Q.size(0)
        qs = self.wq(Q).view(batch_size, -1, self.nheads, self.head_dim).transpose(1,2)
        ks = self.wk(K).view(batch_size, -1, self.nheads, self.head_dim).transpose(1,2)
        vs = self.wv(V).view(batch_size, -1, self.nheads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(qs, ks.transpose(-1,-2)) * self.scaling
        attn = self.softpro(scores)
        context = torch.matmul(attn, vs)

        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.nheads * self.head_dim)
        output = self.wo(context)
        # output + residual
        return self.norm_layer(output), attn

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
        return self.LN(output + residual)#
    #weight_norm(torch.nn.Conv2d(3, 10, 5),name='weight')

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        #self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0.0, d_model, 2) *
        #                      -(math.log(1000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term[0: (d_model)//2])
        # pe = pe.unsqueeze(0)
        pe = positionalencoding1d(d_model, length=max_len)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = torch.autograd.Variable(self.pe[:x.size(1), :], requires_grad=False)
        x = x + pe
        return x#self.dropout(x)

class my_transformer(nn.Module):
    '''
    Parameters:
        d_dim: dimensions of input Q
        k_dim: dimensions of input K. In self attention, d_dim = k_dim
        n_seq: sequence length
        nheads: multi head parameter
        mid_dim: map dimensions of feed forward layer
        output_dim: literal meaning
    '''
    def __init__(self, d_dim, k_dim, n_seq, nheads, mid_dim, output_dim):
        super(my_transformer,self).__init__()
        self.positionAdd = PositionalEncoding(d_dim)
        self.attnModel = Manual_Mult_Attention(d_dim, k_dim, nheads, n_seq=n_seq)
        self.ffnModel = PoswiseFeedForward(d_dim, mid_dim, n_seq=n_seq)
        self.ouputLayer = nn.Linear(d_dim, output_dim, bias=False)

    def forward(self, input):
        '''
        input dimension: batch_size, d_model, sequence_length
        '''
        x = self.positionAdd(input)
        context, attn = self.attnModel(Q = x, K = x, V = x)
        ffn = self.ffnModel(context)
        pre_data = torch.mean(ffn, dim=-2)
        output = self.ouputLayer(pre_data)
        return output

if __name__ == '__main__':
    x = torch.zeros(size=[10, 32, 16])
    model = my_transformer(16, 16, 32, 4, 16, 10)
    out = model(x)
    print(out)


