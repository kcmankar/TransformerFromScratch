import torch
from torch import nn
class PositionalEncodings(nn.Module):
    def __init__(self,seq_length,emb_size,device='cpu'):
        super(PositionalEncodings,self).__init__()
        self.encodings = torch.zeros(seq_length,emb_size,device=device)
        self.encodings.requires_grad = False  # we don't need  gradient

        position = torch.arange(0,seq_length,device=device)
        position = position.float().unsqueeze(dim=1)

        _2i = torch.arange(0, emb_size, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        #print(f'positions: {position}\n2i: {_2i}\nsine: {torch.sin(position / (10000 ** (_2i / emb_size)))}\n cos: {torch.cos(position / (10000 ** (_2i / emb_size)))}')

        self.encodings[:, 0::2] = torch.sin(position / (10000 ** (_2i / emb_size)))

        self.encodings[:, 1::2] = torch.cos(position / (10000 ** (_2i / emb_size)))

    def forward(self,x):
        b,seq,emb = x.size()
        return self.encodings[:seq,:]
