from SelfAttentionScratch import MultiHeadAttention
import torch
from torch import nn
class TransformerBlock(nn.Module):
    def __init__(self, emb_size,n_heads=8,mask=False, ff=4):
        super(TransformerBlock,self).__init__()
        self.attention = MultiHeadAttention(emb_size,n_heads,mask)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.fc = nn.Sequential(
            nn.Linear(emb_size,ff*emb_size),
            nn.ReLU(),
            nn.Linear(ff*emb_size,emb_size)
        )

    def forward(self,x):
        attn = self.attention(x)
        x = self.norm1(attn+x)
        out = self.fc(x)
        out = self.norm2(out+x)
        return out
