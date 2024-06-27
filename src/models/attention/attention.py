import torch
import torch.nn as nn


class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout, add_bias_kv, batch_first) -> None:
        assert batch_first and add_bias_kv
        super().__init__()
        self.embed_dim_per_head = embed_dim // num_heads
        assert self.embed_dim_per_head * num_heads == embed_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.k = nn.Linear(embed_dim, embed_dim)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.attention_softmax = nn.Softmax(dim=-1)
        self.scaler = (embed_dim // num_heads) ** 0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bs, length, *_ = x.size()
        view_shape = (bs, length, self.num_heads, self.embed_dim_per_head)
        k = self.k(x).view(*view_shape).permute(0, 2, 3, 1)
        q = self.q(x).view(*view_shape).permute(0, 2, 1, 3)
        v = self.v(x).view(*view_shape).permute(0, 2, 1, 3)
        x = torch.matmul(q, k) / self.scaler
        x = x + mask * -10_000
        x = self.dropout(self.attention_softmax(x))
        x = torch.matmul(x, v).permute(0, 2, 1, 3).contiguous()
        x = x.view(bs, length, self.embed_dim)
        return self.out(x)
