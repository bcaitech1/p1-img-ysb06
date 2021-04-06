import torch
import torch.nn as nn
from torch import optim
import mask_detector.trainer as trainer
from mask_detector.trainer import Trainee
from mask_detector.dataset import DatasetType, generate_train_datasets, generate_test_datasets
from mask_detector.models import BaseModel
from mask_detector.combined_predictor import Predictor_M1, submission_label_recalc
import numpy as np
import random

def train_model():
    seed = 37764
    mask_model_trainee = Trainee("mask-classifier")
    train_set, valid_set = generate_train_datasets("/opt/ml/input/data", )
    mask_model_trainee.batch_size = 256
    mask_model_trainee.log_interval = calc_interval(train_set, mask_model_trainee.batch_size, 3)
    mask_model_trainee.epochs = 64
    mask_model_trainee.prepare_dataset(train_set, valid_set, DatasetType.General)

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def calc_interval(raw_set, batch_size, show_count):
    return int(len(raw_set) / (batch_size * show_count))

if __name__ == "__main__":
    train_model()