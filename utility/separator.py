import torch
import torch.nn as nn
import math
import copy
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


# Separator
def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Separator(nn.Module):
    "Separator is a stack of N layers"

    def __init__(self, layer, N=6):
        super(Separator, self).__init__()
        self.layers = clones(layer, N)
        self.N = N
    
    def forward(self, x):
        "Pass the input through each layer in turn"
        for layer in self.layers:
            x = layer(x)
        return x


# Multi-path architecture
class SeparatorLayer(nn.Module):
    "Separator is made up of spectral module, temporal module and self-attention module(defined below)"

    def __init__(self, RNN_module_s, RNN_module_t, self_attn):
        super(SeparatorLayer, self).__init__()
        self.spectral = RNN_module_s
        self.temporal = RNN_module_t
        self.self_attn = self_attn

    def forward(self, x):
        """
        x: batch, D, T, F 
        """
        output = self.spectral(x) # batch, D, T, F

        output = output.transpose(2,3).contiguous()
        output = self.temporal(output) # batch, D, F, T

        output = output.transpose(2,3).contiguous()
        output = self.self_attn(output) # batch, D, T, F

        return output


# Spectral & temporal module
class RNNModule(nn.Module):
    "Unfold -> LN -> BLSTM -> Deconv1D -> residual"

    def __init__(self, hidden_size=256, kernel_size=8, stride=1, embed_dim=32, dropout=0, bidirectional=True):
        super(RNNModule, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size

        self.layernorm = nn.LayerNorm(embed_dim*kernel_size)
        self.rnn = nn.LSTM(embed_dim*kernel_size, hidden_size, 1, dropout=dropout, batch_first=True, bidirectional=bidirectional) # N,L,F -> N,L,2H
        self.deconv1d = nn.ConvTranspose1d(2*hidden_size, embed_dim, kernel_size, stride=stride) 
    

    def forward(self, x):
        "x: batch, D, dim1, dim2"
        batch, embed_dim, dim1, dim2 = x.shape

        output = x.unfold(-1, self.kernel_size, self.stride) # batch, D, dim1, dim2/stride, kernel_size

        output = output.permute(0,2,3,4,1).contiguous().view(batch, dim1, -1, self.kernel_size*embed_dim)
        output = self.layernorm(output) # batch, dim1, dim2/stride, D*kernel_size

        output = output.view(batch*dim1, -1, self.kernel_size*embed_dim)
        output, _ = self.rnn(output) # batch*dim1, dim2/stride, 2*hidden_size

        output = output.contiguous().transpose(1,2).contiguous() # batch*dim1, 2*hidden_size, dim2/stride
        output = self.deconv1d(output) # batch*dim1, D, dim2

        output = output.view(batch, dim1, embed_dim, dim2).permute(0,2,1,3).contiguous()
        output = output + x

        return output


# Pre-processing method to generate final result in Attention module
class Generator(nn.Module):
    "1X1Conv2d -> PReLU -> cfLN"

    def __init__(self, input_dim=32, output_dim=4, F=129):
        super(Generator, self).__init__()
        self.conv2d = nn.Conv2d(input_dim, output_dim, 1)
        self.prelu = nn.PReLU()
        self.cfLN = nn.LayerNorm([output_dim, F])
    
    def forward(self, x):
        "x: batch, embed_dim, T, F"
        output = self.conv2d(x) # batch, output_dim, T, F
        output = self.prelu(output) # batch, output_dim, T, F

        output = output.transpose(1,2).contiguous() # batch, T, output_dim, F
        output = self.cfLN(output)
        output = output.transpose(1,2).contiguous()

        return output


# Pre-processing method to generate batched qkv
class MultiHeadGenerator(nn.Module):
    "1X1Conv2d -> PReLU -> cfLN"

    def __init__(self, input_dim=32, output_dim=4, F=129, h=4):
        super(MultiHeadGenerator, self).__init__()
        self.h = h
        self.output_dim = output_dim
        self.conv2d = nn.Conv2d(input_dim, h*output_dim, 1)
        self.prelu = nn.PReLU()
        self.cfLN = nn.LayerNorm([output_dim, F])
    
    def forward(self, x):
        "x: batch, embed_dim, T, F"
        batch, _, T, F = x.shape
        output = self.conv2d(x) # batch, h*output_dim, T, F
        output = self.prelu(output) # batch, h*output_dim, T, F

        output = output.view(batch, self.h, self.output_dim, T, F).transpose(2,3).contiguous() # batch, h, T, ouput_dim, F
        output = self.cfLN(output)
        output = output.transpose(3,4).contiguous() # batch, h, T, F, ouput_dim
        output = output.view(batch, self.h, T, F*self.output_dim)

        return output


# Dot-product Attention
def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    Q: batch, T, FxE
    K: batch, T, FxE
    V: batch, T, FxD/L
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# Multi-head Attention with 2D positional embedding (full-band self-attention module)
class PositionalMultiHeadAttention(nn.Module):
    """
    Multi-head attention with 2D positional encoding. 
    This concept (2D PE) was proposed in the "Translating 
    Math Formula Images to LaTeX Sequences Using 
    Deep Neural Networks with Sequence-level Training".
    """

    def __init__(self, h=4, d_model=32, d_q=4, F=129, dropout=0.1):
        super(PositionalMultiHeadAttention, self).__init__()

        assert d_model % h == 0
        self.h = h
        self.d_q = d_q
        self.d_k = d_q
        self.d_v = d_model // h
        self.RPE_size = F*d_q # FXE in paper

        self.q_proj = MultiHeadGenerator(d_model, self.d_q, F, h)
        self.k_proj = MultiHeadGenerator(d_model, self.d_k, F, h)
        self.v_proj = MultiHeadGenerator(d_model, self.d_v, F, h)
        self.out_proj = Generator(d_model, d_model, F)

        self.p_enc_2d = Summer(PositionalEncoding2D(F))

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask=None):
        "x: batch, D, T, F"
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        batch, _, T, F = x.shape

        # do all the projections in batch
        query = self.q_proj(self.p_enc_2d(x)) # B, N, T, FXE
        key = self.k_proj(self.p_enc_2d(x)) # B, N, T, FXE
        value = self.v_proj(x) # B, N, T, FX(D/L)

        output, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        output = output.view(batch, self.h, T, F, self.d_v).permute(0,1,4,2,3).contiguous().view(batch, self.h * self.d_v, T, F)
        output = self.out_proj(output)
        output = output + x

        del query
        del key
        del value

        return output
    
if __name__ == "__main__":
    # Test RNNModule
    # rnn_module = RNNModule(hidden_size=256)
    # x = torch.rand(3, 32, 1001, 129)
    # y = rnn_module(x)
    # print(y.shape)

    # Test MultiheadAttention
    B, D, T, F = 3, 32, 1001, 129
    x = torch.randn(B,D,T,F)
    model = PositionalMultiHeadAttention(h=4, d_model=D, d_q=4, F=F)
    y = model(x)
    print(y.shape)
