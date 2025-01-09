from typing import Iterator
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils import tensorboard
from dataset import TextDataset
from model import LM


def grad_norm(parameters: Iterator[torch.nn.Parameter]) -> float:
   total_norm = 0.0
   for p in parameters:
      if p.grad is not None:
         param_norm = p.grad.data.norm(2)
         total_norm += param_norm.item() ** 2
   return total_norm ** 0.5

def main():
    lm = LM()
    dataset = TextDataset(file_path="tinyshakespeare.txt", context_size=256, encode_fn=lm.encode)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = AdamW(lm.parameters(), lr=1e-3)
    writer = tensorboard.SummaryWriter()
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        _, loss = lm(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), max_norm=1.0)
        optimizer.step()
        writer.add_scalar("train/loss", loss.item(), i)
        writer.add_scalar("train/grad_norm", grad_norm(lm.parameters()), i)

if __name__ == "__main__":
    main()
