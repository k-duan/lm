from datetime import datetime
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
    torch.manual_seed(123)
    log_name = f"tinyshakespeare-{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}"
    writer = tensorboard.SummaryWriter(log_dir=f"runs/{log_name}")
    lm = LM()
    dataset = TextDataset(file_path="tinyshakespeare.txt", context_size=256, encode_fn=lm.encode)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = AdamW(lm.parameters(), lr=1e-3)
    n_epochs = 2
    i = 0
    for _ in range(n_epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            _, loss = lm(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), max_norm=1.0)
            optimizer.step()
            writer.add_scalar("train/loss", loss.item(), i)
            writer.add_scalar("train/grad_norm", grad_norm(lm.parameters()), i)
            if i % 100 == 0:
                lm.eval()
                print(lm.predict(["\n"], max_new_tokens=500, top_k=20, temperature=0.9))
                lm.train()
            if i % 5000 == 0:
                torch.save(lm.state_dict(), f"checkpoints/{i:06}.ckpt")
            i += 1

if __name__ == "__main__":
    main()
