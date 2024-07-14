import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import NamedTuple


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

    max_batch_size: int = 4

    @classmethod
    def llama3_8b(cls):
        return cls(
            embed_dim=4096,
            embed_dim_proj=14_336,
            num_q_heads=32,
            num_kv_heads=8,
            num_layers=32,
        )

    @classmethod
    def llama3_70b(cls):
        return cls(
            embed_dim=8192,
            embed_dim_proj=28_672,
            num_q_heads=64,
            num_kv_heads=8,
            num_layers=80,
        )


class GenerationConfig(NamedTuple):
    max_length: int = 4096
    max_new_tokens: int = 4096
    stop_token: int = 128009
    pad_token: int = 128009
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 128


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
        self.layers = Sequential(*[LlamaLayer(config=config, positional_embedding=positional_embedding) for _ in range(config.num_layers)])
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)
        self.norm = RMSNorm(embed_dim=config.embed_dim, eps=config.rms_norm_eps)
        self.config = config
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(self.layers(self.embedding(tokens), offset=0)))

    @torch.jit.export
    def generate(self, tokens: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        input_tokens = tokens
        output_tokens = tokens[:, :0]
        completed = torch.zeros_like(input_tokens[:, -1], dtype=torch.bool)

        offset: int = 0
        for _ in range(config.max_new_tokens):
            if completed.all():
                break

            new_token = self.generate_sample(hidden=self.layers(self.embedding(input_tokens), offset=offset), config=config)
            offset += input_tokens.size(-1)
            new_token[completed] = config.pad_token
            input_tokens = new_token[:, None]
            output_tokens = torch.cat([output_tokens, input_tokens], dim=-1)
            completed[new_token == config.stop_token] = True
        return output_tokens

    def generate_sample(self, hidden: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        new_logit = self.lm_head(self.norm(hidden[:, -1:, :]))[:, -1, :]
        probs = F.softmax(new_logit / config.temperature, dim=-1)
        probs_order_values, probs_order_indices = probs.sort(dim=-1, descending=True)

        top_k_probs = probs_order_values[:, :config.top_k]
        top_p_mask = top_k_probs.cumsum(dim=-1) > config.top_p
        top_p_mask[:, top_p_mask.max(dim=-1).indices] = False
        top_k_probs[top_p_mask] = 0
        return probs_order_indices.gather(dim=-1, index=torch.multinomial(top_k_probs, num_samples=1))[:, 0]


class LlamaLayer(nn.Module):
    def __init__(self, config: LlamaConfig, positional_embedding: nn.Module) -> None:
        super().__init__()
        self.attention = LlamaBlock(
            block=GroupedQueryAttention(
                embed_dim=config.embed_dim,
                num_q_heads=config.num_q_heads,
                num_kv_heads=config.num_kv_heads,
                bias=config.bias,
                positional_embedding=positional_embedding,
                max_batch_size=config.max_batch_size,
                max_seq_len=config.max_seq_len,
            ),
            config=config,
        )
        self.proj: LlamaBlock = LlamaBlock(
            block=SwiGLU(embed_dim=config.embed_dim, embed_dim_proj=config.embed_dim_proj, bias=config.bias),
            config=config,
        )

    def forward(self, x, offset: int):
        return self.proj(self.attention(x, offset=offset), offset=offset)


class LlamaBlock(nn.Module):
    def __init__(self, block: nn.Module, config: LlamaConfig) -> None:
        super().__init__()
        self.block = block
        self.norm = RMSNorm(embed_dim=config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x, offset: int):
        return x + self.block(self.norm(x), offset=offset)


class Sequential(nn.Sequential):
    def forward(self, x, offset: int):
        for layer in self:
            x = layer(x, offset=offset)
        return x


class SwiGLU(nn.Module):
    def __init__(self, embed_dim: int, embed_dim_proj: int, bias: bool) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, embed_dim_proj * 2, bias=bias)
        self.out_proj = nn.Linear(embed_dim_proj, embed_dim, bias=bias)

    def forward(self, x, offset: int):
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
    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int, bias: bool, positional_embedding: nn.Module, max_batch_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_to_kv_ratio = num_q_heads // num_kv_heads
        self.head_size = embed_dim // num_q_heads
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.in_proj = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.num_kv_heads * self.head_size, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.qkv_handler = GroupedQueryCachedAttention(
            positional_embedding=positional_embedding,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=num_kv_heads,
            head_size=self.head_size,
        )

    def forward(self, x, offset: int) -> torch.Tensor:
        q, k, v = torch.tensor_split(self.in_proj(x), [self.embed_dim, self.embed_dim + self.num_kv_heads * self.head_size], dim=-1)
        q = q.view(x.size(0), x.size(1), self.num_q_heads, self.head_size)
        k = k.view(x.size(0), x.size(1), self.num_kv_heads, self.head_size)
        v = v.view(x.size(0), x.size(1), self.num_kv_heads, self.head_size)

        q, k, v = self.qkv_handler(q=q, k=k, v=v, offset=offset)

        k = torch.repeat_interleave(k, dim=2, repeats=self.q_to_kv_ratio)
        v = torch.repeat_interleave(v, dim=2, repeats=self.q_to_kv_ratio)

        attention_out = F.scaled_dot_product_attention(
            query=q.transpose(1, 2),
            key=k.transpose(1, 2),
            value=v.transpose(1, 2),
            attn_mask=torch.ones(q.size(1), k.size(1), dtype=torch.bool, device=q.device).tril(diagonal=k.size(1) - q.size(1)),
        ).transpose(1, 2)
        return self.out_proj(attention_out.flatten(2))


class GroupedQueryCachedAttention(nn.Module):
    def __init__(self, positional_embedding, max_batch_size, max_seq_len, num_kv_heads, head_size):
        super().__init__()
        self.positional_embedding = positional_embedding
        self.register_buffer("k_cache", torch.zeros((max_batch_size, max_seq_len, num_kv_heads, head_size), dtype=torch.bfloat16, device=torch.device("cuda")), persistent=False)
        self.register_buffer("v_cache", torch.zeros((max_batch_size, max_seq_len, num_kv_heads, head_size), dtype=torch.bfloat16, device=torch.device("cuda")), persistent=False)

    def forward(self, q, k, v, offset: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.positional_embedding(t=q, offset=offset)
        k = self.positional_embedding(t=k, offset=offset)

        self.k_cache[:k.size(0), offset:offset + k.size(1), :, :] = k
        self.v_cache[:v.size(0), offset:offset + v.size(1), :, :] = v
        return q, self.k_cache[:k.size(0), :offset + k.size(1), :, :], self.v_cache[:v.size(0), :offset + v.size(1), :, :]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, theta):
        super().__init__()
        self.register_buffer(
            "freqs",
            self.precompute_freqs(
                embed_dim // num_heads,
                max_seq_len * 2,
                theta,
            ),
            persistent=False,
        )

    def forward(self, t, offset: int) -> torch.Tensor:
        ct = torch.view_as_complex(t.float().reshape(t.size(0), t.size(1), t.size(2), -1, 2))
        t_out = torch.view_as_real(ct * self.freqs[None, offset:offset+t.size(1), None, :]).flatten(-2)
        return t_out.type_as(t)

    @torch.no_grad
    def precompute_freqs(self, dim: int, end: int, theta: float):
        with torch.device(torch.get_default_device() if torch.get_default_device().type != "meta" else "cpu"):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(end, device=freqs.device, dtype=torch.float32)
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs) # complex64
            return freqs_cis

