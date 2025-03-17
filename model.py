from typing import Optional

import torch
from torch import nn
import numpy as np


class FFN(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self._linear1 = nn.Linear(embedding_dim, embedding_dim)
        self._linear2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, inputs):
        outputs = self._linear1(inputs)
        outputs = nn.functional.relu(outputs)
        outputs = self._linear2(outputs)
        return outputs

class Attention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        self._qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self._attention_mask = torch.tril(torch.ones(n_heads, n_heads, dtype=torch.bool))
        self._ffn = FFN(embedding_dim)
        self._n_heads = n_heads

    def forward(self,
                inputs: torch.Tensor,
                kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
        # inputs.shape: BxTxD if kv_cache is None else Bx1xD
        # kv_cache.shape: (BxHxT'xmH, BxHxT'xmH)
        B, T, _ = inputs.size()
        seq_len = T if kv_cache is None else kv_cache[0].size(-2) + 1
        # att_mask.shape: seq_lenxseq_len if kv_cache is None else 1xseq_len
        att_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)) if kv_cache is None else  torch.ones(1, seq_len, dtype=torch.bool)
        outputs = self._qkv(inputs).view(B, T, self._n_heads, -1) # BxTxD -> BxTxHx3mH
        outputs = outputs.transpose(1, 2)  # BxTxHx3mH -> BxHxTx3mH
        q, k, v = torch.split(outputs, outputs.size(-1) // 3, dim=-1)  # BxHxTx3mH -> (BxHxTxmH, BxHxTxmH, BxHxTxmH)
        # add to kv cache
        kv_cache = (k, v) if kv_cache is None else (torch.cat([kv_cache[0], k], dim=-2), torch.cat([kv_cache[1], v], dim=-2))
        new_k, new_v = kv_cache
        # q.shape: BxHxTxmH if kv_cache is None else BxHx1xmH
        # new_k.shape: BxHxseq_lenxmH, new_v.shape: BxHxseq_lenxmH
        # q @ new_k.transpose(-1, -2).shape: BxHxseq_lenxseq_len if kv_cache is None else BxHx1xseq_len
        # outputs.shape: BxHxseq_lenxmH if kv_cache is None else BxHx1xmH
        outputs = nn.functional.softmax(torch.where(att_mask, q @ new_k.transpose(-1, -2), -torch.inf) / np.sqrt(new_k.size(-1)), dim=-1) @ new_v
        outputs = outputs.transpose(1, 2)  # outputs.shape: BxTxHxmH
        outputs = outputs.reshape(B, T, -1)  # outputs.shape: BxTxD
        outputs = self._ffn(outputs)
        return outputs, kv_cache

class Block(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self._attention = Attention(embedding_dim, n_heads)
        self._ln1 = nn.LayerNorm(embedding_dim)
        self._ffn = FFN(embedding_dim)
        self._ln2 = nn.LayerNorm(embedding_dim)
        self._dropout = dropout

    def forward(self, inputs: torch.Tensor, kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
        # prenorm -> attention -> mlp
        out = self._ln1(inputs)
        att, kv_cache = self._attention(inputs=out, kv_cache=kv_cache)
        out = inputs + att
        out = out + self._ffn(self._ln2(out))
        return out, kv_cache

class Transformer(nn.Module):
    """
    This implementation follows the GPT-2 paper:
    1) normalization happens before attention in each attention block
    2) additional normalization after final attention block
    """

    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self._blocks = nn.ModuleList([Block(embedding_dim, n_heads, dropout) for _ in range(n_layers)])
        self._ln = nn.LayerNorm(embedding_dim)

    def forward(self, inputs: torch.Tensor, kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None):
        output = inputs
        kv_cache = [None for _ in range(len(self._blocks))] if kv_cache is None else kv_cache
        for i, block in enumerate(self._blocks):
            output, kv_cache[i] = block(inputs=output, kv_cache=kv_cache[i])
        output = self._ln(output)  # additional normalization after final attention block
        return output, kv_cache

class LM(nn.Module):
    def __init__(self, vocab_size: int = 128, embedding_dim: int = 128, n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        self._token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self._transformer = Transformer(embedding_dim, n_layers, n_heads, dropout)
        self._lm_head = nn.Linear(embedding_dim, vocab_size)
        self._label_smoothing = label_smoothing

    def forward(self,
                inputs: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None):
        # TODO support padding mask
        loss = None
        outputs = self._token_embedding(inputs)
        outputs, kv_cache = self._transformer(outputs, kv_cache=kv_cache)
        logits = self._lm_head(outputs)
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=self._label_smoothing)
        return logits, loss, kv_cache

    @torch.no_grad()
    def encode(self, inputs: list[str], max_len: int = None) -> torch.Tensor:
        if max_len is None:
            max_len = max([len(input) for input in inputs])
        # Assume zero left padding (chr(0) -> '𘚠') and output shape BxT
        output = torch.zeros(len(inputs), max_len, dtype=torch.int64)
        for i, input in enumerate(inputs):
            token_ids = [ord(c) for c in input]
            if max(token_ids) > 127:
                raise ValueError("only accept ascii characters")
            if len(token_ids) > max_len:
                token_ids = token_ids[:max_len]
            output[i, -len(token_ids):] = torch.Tensor(token_ids)
        return output

    @torch.no_grad()
    def decode(self, inputs: torch.Tensor) -> list[str]:
        # Assume inputs shape BxT
        output = []
        for v in inputs.detach().numpy():
            if max(v) > 127:
                raise ValueError("only accept ascii characters")
            output.append("".join([chr(i) for i in v]))
        return output

    @torch.no_grad()
    def top_k_sample(self, logits: torch.Tensor, top_k: int, temperature: float) -> torch.Tensor:
        logits = logits[:,-1,:]  # BxTxN -> BxN
        values, indices = torch.topk(logits, top_k, dim=-1)  # BxK
        masked_logits = torch.full(logits.size(), -torch.inf)
        masked_logits = masked_logits.scatter(-1, indices, values)
        m = torch.distributions.Categorical(logits=masked_logits / temperature)
        return m.sample().unsqueeze(dim=-1)

    @torch.no_grad()
    def predict(self, inputs: list[str], max_new_tokens: int, top_k: int = 5, temperature: float = 0.9) -> list[str]:
        token_ids = self.encode(inputs)  # BxT
        generated_ids = torch.empty(size=(token_ids.size(0), 0), dtype=torch.int64)
        kv_cache = None
        next_token_ids = None
        for _ in range(max_new_tokens):
            logits, _, kv_cache = self.forward(inputs=token_ids if next_token_ids is None else next_token_ids, kv_cache=kv_cache)
            next_token_ids = self.top_k_sample(logits, top_k, temperature)  # Bx1
            #  Bx(T+1)
            generated_ids = torch.concat([generated_ids, next_token_ids], dim=-1)
        return self.decode(torch.concat([token_ids, generated_ids], dim=-1))
