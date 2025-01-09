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

    def forward(self, inputs):
        B, T, _ = inputs.size()
        att_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        outputs = self._qkv(inputs).view(B, T, self._n_heads, -1) # BxTxD -> BxTxHx3mH
        outputs = outputs.transpose(1, 2)  # BxTxHx3mH -> BxHxTx3mH
        q, k, v = torch.split(outputs, outputs.size(-1) // 3, dim=-1)  # BxHxTx3mH -> (BxHxTxmH, BxHxTxmH, BxHxTxmH)
        outputs = nn.functional.softmax((q @ k.transpose(-1, -2)) * att_mask / np.sqrt(k.size(-1)), dim=-1) @ v
        outputs = outputs.transpose(1, 2)
        outputs = outputs.reshape(B, T, -1)
        outputs = self._ffn(outputs)
        return outputs

class Block(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int):
        super().__init__()
        self._attention = Attention(embedding_dim, n_heads)
        self._ln1 = nn.LayerNorm(embedding_dim)
        self._ffn = FFN(embedding_dim)
        self._ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):
        att = self._attention(inputs)
        out1 = self._ln1(att + inputs)
        out2 = self._ffn(out1)
        return self._ln2(out1 + out2)

class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int):
        super().__init__()
        self._blocks = nn.ModuleList([Block(embedding_dim, n_heads) for _ in range(n_layers)])

    def forward(self, inputs):
        output = inputs
        for block in self._blocks:
            output = block(output)
        return output

class LM(nn.Module):
    def __init__(self, vocab_size: int = 128, embedding_dim: int = 256, n_layers: int = 8, n_heads: int = 8):
        super().__init__()
        self._token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self._transformer = Transformer(embedding_dim, n_layers, n_heads)
        self._lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        loss = None
        outputs = self._token_embedding(inputs)
        outputs = self._transformer(outputs)
        logits = self._lm_head(outputs)
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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
