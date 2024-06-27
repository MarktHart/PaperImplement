import torch
from torch import nn


class Layer(nn.Module):
    def __init__(self, heads, hidden_size, attention_drop_p=0.1, multi_head_attention_drop_p=0.1, out_drop_p=0.1) -> None:
        super().__init__()
        self.attention_block = nn.MultiheadAttention(num_heads=heads, embed_dim=hidden_size, dropout=attention_drop_p, add_bias_kv=True, batch_first=True)
        self.attention_norm = nn.LayerNorm((hidden_size,), eps=1e-12)
        self.attention_drop = nn.Dropout(p=multi_head_attention_drop_p)
        self.out_block = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.out_norm = nn.LayerNorm((hidden_size,), eps=1e-12)
        self.out_drop = nn.Dropout(p=out_drop_p)

    def forward(self, attention_in: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, attention_mask = attention_in
        x = self.attention_norm(x + self.attention_drop(self.attention_block(key=x, value=x, query=x, key_padding_mask=attention_mask, need_weights=False)[0]))
        x = self.out_norm(x + self.out_drop(self.out_block(x)))
        return x, attention_mask


class WordEmbedding(nn.Module):
    def __init__(self, dict_size, max_sequence_length, hidden_size, drop_p=0.1) -> None:
        super().__init__()
        self.word = nn.Embedding(num_embeddings=dict_size, embedding_dim=hidden_size, padding_idx=0)
        self.position = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=hidden_size)
        self.layernorm = nn.LayerNorm((hidden_size,), eps=1e-12)
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        word_encoding = self.word(x)
        positional_encoding = self.position(torch.arange(x.size(1)).unsqueeze(0))
        return self.dropout(self.layernorm(word_encoding + positional_encoding))


class GPT2(nn.Module):
    def __init__(self, L: int, H: int, A: int, dict_size: int, max_sequence_length=1024) -> None:
        super().__init__()
        self.embedding = WordEmbedding(dict_size=dict_size, max_sequence_length=max_sequence_length, hidden_size=H)
        self.layers = nn.Sequential(*[Layer(heads=A, hidden_size=H) for _ in range(L)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(self.embedding(x))

    @classmethod
    def large(cls, **kwargs):
        return cls(L=24, H=1024, A=16, **kwargs)

    @classmethod
    def base(cls, **kwargs):
        return cls(L=12, H=768, A=12, **kwargs)
