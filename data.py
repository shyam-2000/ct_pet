from typing import Iterable, SupportsFloat
import torch


class CTData(torch.utils.data.Dataset):
    def __init__(self, images: Iterable[SupportsFloat], masks: Iterable[SupportsFloat]):
        super().__init__()
        self.data = list(zip(images, masks))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.data[idx]
        return (
            torch.tensor(image, dtype=torch.float).unsqueeze(0),
            torch.tensor(mask, dtype=torch.float).unsqueeze(0),
        )
