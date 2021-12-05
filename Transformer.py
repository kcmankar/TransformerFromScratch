import torch
from torch import nn
from PositionalEmbeddings import PositionalEncodings
from TransformerBlock import TransformerBlock
class Transformer(nn.Module):
    def __init__(self,input_dim,max_seq_length,output_dims,n_heads=8,mask=False,depth=1):
        self.outdims = output_dims
        super(Transformer,self).__init__()
        blocks = [
        TransformerBlock(input_dim,n_heads=8,mask=False, ff=4) for i in range(depth)
        ]
        self.TransformerBlocks = nn.Sequential(*blocks)
        self.outputLayer = nn.Linear(input_dim,output_dims)
        self.positional_encodings = PositionalEncodings(max_seq_length,input_dim)

    def forward(self,x):
        b,t,e = x.shape
        pos = self.positional_encodings(x)
        out = pos+x
        out = self.TransformerBlocks(out)
        out = self.outputLayer(out.view(b*t,e)).view(b,t,self.outdims)
        out = out.max(dim=1)
        return out

#tformers = Transformer(16,64,10)
#res = tformers(torch.rand(2,8,16))
#res.values.shape
