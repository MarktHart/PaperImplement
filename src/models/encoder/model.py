import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, L: int, H: int, A: int, dict_size: int, max_sequence_length=512) -> None:
        super().__init__()
        self.embedding: WordEmbedding = WordEmbedding(dict_size=dict_size, max_sequence_length=max_sequence_length, hidden_size=H)
        self.layers: nn.Sequential = nn.Sequential(*[Layer(heads=A, hidden_size=H) for _ in range(L)])

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        mask = (torch.arange(x.size(1)).expand(x.size(0), -1) >= lengths.expand(x.size(1), -1).transpose(0, 1))
        return self.layers((x, mask))


class WordEmbedding(nn.Module):
    def __init__(self, dict_size, max_sequence_length, hidden_size, drop_p=0.1) -> None:
        super().__init__()
        self.word = nn.Embedding(num_embeddings=dict_size, embedding_dim=hidden_size, padding_idx=0)
        self.position = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        word_encoding = self.word(x)
        positional_encoding = self.position(torch.arange(x.size(1)).unsqueeze(0))
        segment_encoding = self.segment(torch.zeros_like(x))
        return self.dropout(self.layernorm(word_encoding + positional_encoding))


class Layer(nn.Module):
    def __init__(self, heads, hidden_size, attention_drop_p=0.1, multi_head_attention_drop_p=0.1, out_drop_p=0.1) -> None:
        super().__init__()
        self.attention = ResidualBlock(
            block=SelfAttention(num_heads=heads, embed_dim=hidden_size, dropout=attention_drop_p, add_bias_kv=True, batch_first=True),
            hidden_size=hidden_size,
            drop_p=multi_head_attention_drop_p,
        )
        self.out = ResidualBlock(
            block=nn.Sequential(
                nn.Linear(hidden_size, 4 * hidden_size),
                nn.GELU(),
                nn.Linear(4 * hidden_size, hidden_size),
            ),
            hidden_size=hidden_size,
            drop_p=out_drop_p,
        )

    def forward(self, attention_in: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, attention_mask = attention_in
        x = self.out(self.attention(attention_in))
        return x, attention_mask


class ResidualBlock(nn.Module):
    def __init__(self, block, hidden_size, drop_p) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.drop = nn.Dropout(p=drop_p)
        self.block = block

    def forward(self, x)-> torch.Tensor:
        return x + self.norm(self.drop(self.block(x)))


class SelfAttention(nn.MultiheadAttention):
    def forward(self, attention_in):
        x, mask = attention_in
        return super().forward(key=x, value=x, query=x, key_padding_mask=mask, need_weights=False)[0]
