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
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self._attention = Attention(embedding_dim, n_heads)
        self._ln1 = nn.LayerNorm(embedding_dim)
        self._ffn = FFN(embedding_dim)
        self._ln2 = nn.LayerNorm(embedding_dim)
        self._dropout = dropout

    def forward(self, inputs):
        att = self._attention(inputs)
        att = torch.nn.functional.dropout(att, p=self._dropout)
        out1 = self._ln1(att + inputs)
        out2 = self._ffn(out1)
        out2 = torch.nn.functional.dropout(out2, p=self._dropout)
        return self._ln2(out1 + out2)

class Transformer(nn.Module):
    def __init__(self, embedding_dim: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self._blocks = nn.ModuleList([Block(embedding_dim, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, inputs):
        output = inputs
        for block in self._blocks:
            output = block(output)
        return output

class LM(nn.Module):
    def __init__(self, vocab_size: int = 128, embedding_dim: int = 256, n_layers: int = 8, n_heads: int = 8, dropout: float = 0.1, label_smoothing: float = 0.1):
        super().__init__()
        self._token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self._transformer = Transformer(embedding_dim, n_layers, n_heads, dropout)
        self._lm_head = nn.Linear(embedding_dim, vocab_size)
        self._label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor = None):
        # TODO support padding mask
        loss = None
        outputs = self._token_embedding(inputs)
        outputs = self._transformer(outputs)
        logits = self._lm_head(outputs)
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), label_smoothing=self._label_smoothing)
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

    @torch.no_grad()
    def top_k_sample(self, logits: torch.Tensor, top_k: int = 5, temperature: float = 0.9) -> torch.Tensor:
        logits = logits[:,-1,:]  # BxTxN -> BxN
        values, indices = torch.topk(logits, top_k, dim=-1)  # BxK
        masked_logits = torch.full(logits.size(), -torch.inf)
        masked_logits = masked_logits.scatter(-1, indices, values)
        m = torch.distributions.Categorical(logits=masked_logits / temperature)
        return m.sample().unsqueeze(dim=-1)

    @torch.no_grad()
    def predict(self, inputs: list[str], max_new_tokens: int) -> list[str]:
        encoded_batch = self.encode(inputs)  # BxT
        for _ in range(max_new_tokens):
            logits, _ = self.forward(encoded_batch)
            next_token_ids = self.top_k_sample(logits)  # Bx1
            #  Bx(T+1)
            encoded_batch = torch.concat([encoded_batch, next_token_ids], dim=-1)
        return self.decode(encoded_batch)
