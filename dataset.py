from typing import Callable
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, file_path: str, context_size: int, encode_fn: Callable):
        super().__init__()
        with open(file_path) as f:
            self._data = f.read()
        self._encoded_data = encode_fn(self._data).view(-1)
        self._context_size = context_size

    def __len__(self):
        return self._encoded_data.size(-1) - self._context_size - 1

    def __getitem__(self, i):
        x = self._encoded_data[i:i+self._context_size]
        y = self._encoded_data[i+1:i+self._context_size+1]
        return x, y
