import pandas as pd
from mask_detector.dataset import Gender, generate_train_datasets
from torch.utils.data import DataLoader

train_set, valid_set = generate_train_datasets("/opt/ml/input/data")

train_set_loader = DataLoader(train_set, 4)
valid_set_loader = DataLoader(valid_set, 4)

for source, target, path_text in train_set_loader:
    print(source)
    print(target)
    print(path_text)
    break