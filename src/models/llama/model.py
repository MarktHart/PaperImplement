import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class LlamaConfig:
    num_layers: int = 32
    embed_dim: int = 4096
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embed_dim_proj: int = 14_336
    vocab_size: int = 128_256
    bias: bool = False
    max_seq_len: int = 8_192
    rope_theta: float = 500_000.
    rms_norm_eps: float = 1.e-5

    @classmethod
    def llama3_8b(cls):
        return cls()

    @classmethod
    def llama3_70b(cls):
        return cls(
            embed_dim=8192,
            embed_dim_proj=28_672,
            num_q_heads=64,
            num_kv_heads=8,
            num_layers=80,
        )


class Llama(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim)
        positional_embedding = RotaryPositionalEmbedding(
            embed_dim=config.embed_dim,
            num_heads=config.num_q_heads,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.layers = nn.Sequential(*[LLamaLayer(config=config, positional_embedding=positional_embedding) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)
        self.norm = RMSNorm(embed_dim=config.embed_dim, eps=config.rms_norm_eps)
    
    def forward(self, tokens):
        return self.lm_head(self.norm(self.layers(self.embedding(tokens))))


class LLamaLayer(nn.Module):
    def __init__(self, config: LlamaConfig, positional_embedding: nn.Module) -> None:
        super().__init__()
        self.attention = LlamaBlock(
            block=GroupedQueryAttention(embed_dim=config.embed_dim, num_q_heads=config.num_q_heads, num_kv_heads=config.num_kv_heads, bias=config.bias, positional_embedding=positional_embedding),
            config=config,
        )
        self.proj: LlamaBlock = LlamaBlock(
            block=SwiGLU(embed_dim=config.embed_dim, embed_dim_proj=config.embed_dim_proj, bias=config.bias),
            config=config,
        )
    
    def forward(self, x):
        return self.proj(self.attention(x))


class LlamaBlock(nn.Module):
    def __init__(self, block: nn.Module, config: LlamaConfig) -> None:
        super().__init__()
        self.block = block
        self.norm = RMSNorm(embed_dim=config.embed_dim, eps=config.rms_norm_eps)
    
    def forward(self, x):
        return x + self.block(self.norm(x))


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, embed_dim_proj: int, bias: bool) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, embed_dim_proj * 2, bias=bias)
        self.out_proj = nn.Linear(embed_dim_proj, embed_dim, bias=bias)

    def forward(self, x):
        gate, up = torch.tensor_split(self.in_proj(x), 2, dim=-1)
        return self.out_proj(F.silu(gate) * up)


class RMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
    
    def forward(self, x):
        return self.weight * self.casted_forward(x.float()).type_as(x)

    def casted_forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, bias: bool, positional_embedding: nn.Module) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        q_to_kv_ratio = num_q_heads // num_kv_heads
        self.split_size = (q_to_kv_ratio + 2)
        self.head_size = embed_dim // num_q_heads
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.in_proj = nn.Linear(embed_dim, (embed_dim // q_to_kv_ratio) * self.split_size, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.positional_embedding = positional_embedding

    def forward(self, x):
        *q, k, v = torch.tensor_split(self.in_proj(x), self.split_size, dim=-1)
        q = torch.cat(q, dim=-1)
        q = q.view(x.size(0), x.size(1), self.num_q_heads, self.head_size)
        k = k.view(x.size(0), x.size(1), self.num_kv_heads, self.head_size)
        v = v.view(x.size(0), x.size(1), self.num_kv_heads, self.head_size)

        q, k = self.positional_embedding(q=q, k=k)

        v = torch.repeat_interleave(v, dim=2, repeats=4)
        k = torch.repeat_interleave(k, dim=2, repeats=4)

        attention_out = F.scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
        return self.out_proj(attention_out.flatten(2))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, theta):
        super().__init__()
        freqs = self.precompute_freqs(
            embed_dim // num_heads,
            max_seq_len * 2,
            theta,
        )
        freqs.requires_grad = False
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, q, k):
        xq_ = torch.view_as_complex(q.float().reshape(q.size(0), q.size(1), q.size(2), -1, 2))
        xk_ = torch.view_as_complex(k.float().reshape(k.size(0), k.size(1), k.size(2), -1, 2))
        freqs = self.freqs[None, :q.size(1), None, :]
        xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
        xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)
        return xq_out.type_as(q), xk_out.type_as(k)

    @torch.no_grad
    def precompute_freqs(self, dim: int, end: int, theta: float):
        with torch.device(torch.get_default_device() if torch.get_default_device().type != "meta" else "cpu"):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(end, device=freqs.device, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
            return freqs_cis

