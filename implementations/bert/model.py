import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, attention_drop_p) -> None:
        super().__init__()
        self.hidden_size_per_head = hidden_size // heads
        assert self.hidden_size_per_head * heads == hidden_size
        self.heads = heads
        self.hidden_size = hidden_size

        self.k = nn.Linear(hidden_size, hidden_size)
        self.q = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=attention_drop_p)

        self.attention_softmax = nn.Softmax(dim=-1)
        self.scaler = (hidden_size // heads) ** 0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bs, length, *_ = x.shape
        view_shape = (bs, length, self.heads, self.hidden_size_per_head)
        k = self.k(x).view(*view_shape).permute(0, 2, 3, 1)
        q = self.q(x).view(*view_shape).permute(0, 2, 1, 3)
        v = self.v(x).view(*view_shape).permute(0, 2, 1, 3)
        x = torch.matmul(q, k)
        x = x / self.scaler
        x = x + mask * -10_000
        x = self.attention_softmax(x)
        x = self.dropout(x)  # not mentioned in the paper
        x = torch.matmul(x, v).permute(0, 2, 1, 3).contiguous()
        x = x.view(bs, length, self.hidden_size)
        return self.out(x)


class AttentionBlock(nn.Module):
    def __init__(self, heads, hidden_size, attention_drop_p=0.1, multi_head_attention_drop_p=0.1, out_drop_p=0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(heads=heads, hidden_size=hidden_size, attention_drop_p=attention_drop_p)
        self.attention_drop = nn.Dropout(p=multi_head_attention_drop_p)
        self.attention_norm = nn.LayerNorm((hidden_size,), eps=1e-12)
        self.out = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.out_drop = nn.Dropout(p=out_drop_p)
        self.out_norm = nn.LayerNorm((hidden_size,), eps=1e-12)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attention_drop(self.attention(x, mask=mask))
        x = self.attention_norm(x)
        x = x + self.out_drop(self.out(x))
        x = self.out_norm(x)
        return x


class WordEmbedding(nn.Module):
    def __init__(self, dict_size, max_sequence_length, hidden_size, drop_p=0.1) -> None:
        super().__init__()
        self.word = nn.Embedding(num_embeddings=dict_size, embedding_dim=hidden_size, padding_idx=0)
        self.position = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=hidden_size)
        self.segment = nn.Embedding(num_embeddings=2, embedding_dim=hidden_size)
        self.layernorm = nn.LayerNorm((hidden_size,), eps=1e-12)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: torch.Tensor) -> None:
        _, max_length, *_ = x.shape
        word_encoding = self.word(x)
        positional_encoding = self.position(torch.arange(max_length).unsqueeze(0))
        segment_encoding = self.segment(torch.zeros_like(x))

        x = word_encoding + segment_encoding + positional_encoding
        x = self.layernorm(x)  # not mentioned in the paper
        x = self.dropout(x)
        return x


class Bert(nn.Module):
    def __init__(self, L, H, A, dict_size, max_sequence_length=512) -> None:
        super().__init__()
        self.embedding = WordEmbedding(dict_size=dict_size, max_sequence_length=max_sequence_length, hidden_size=H)
        self.layers = nn.ModuleList([AttentionBlock(heads=A, hidden_size=H) for _ in range(L)])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        bs, max_length, *_ = x.shape
        x = self.embedding(x)
        mask = (torch.arange(max_length).expand(bs, -1) >= lengths.expand(max_length, -1).transpose(0, 1))

        # alternative: attention_mask = torch.logical_or(mask.unsqueeze(1), mask.unsqueeze(2)).unsqueeze(1)
        attention_mask = mask.unsqueeze(1).unsqueeze(1)

        for layer in self.layers:
            x = layer(x, mask=attention_mask)
        return x

    @classmethod
    def large(cls, **kwargs):
        return cls(L=24, H=1024, A=16, **kwargs)

    @classmethod
    def base(cls, **kwargs):
        return cls(L=12, H=768, A=12, **kwargs)
