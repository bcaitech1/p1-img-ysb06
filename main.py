from typing import Any
from torch.utils.data.dataloader import DataLoader
from mask_detector.dataset import MaskedFaceDataset, AgeGroup
from torch import LongTensor

class Test:
    def __init__(self, name) -> None:
        self.text = name

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print(self.text)
        print(args[0])

d = {"A": Test("A"), "B": Test("B")}

d("Test OK")["A"]