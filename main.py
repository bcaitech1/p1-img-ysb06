import random

import numpy as np
import torch
from torch import optim

from mask_detector.combined_predictor import Predictor_G1, Predictor_M2, Predictor_M3, Predictor_M4
from mask_detector.dataset import (DatasetType, generate_test_datasets,
                                   generate_train_datasets)
from mask_detector.loss import FocalLoss
from mask_detector.models import BaseModel
from mask_detector.trainer import Trainee


def train_model():
    print(f"PyTorch version: {torch.__version__}.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    seed = 37764
    seed_everything(seed)

    train_set, valid_set = generate_train_datasets("/opt/ml/input/data", random_seed=seed)

    # Combined 모델 2
    # train_mask_classifier(device, seed, train_set, valid_set)
    # train_gender_classifier(device, seed, train_set, valid_set)
    # train_u30_classifier(device, seed, train_set, valid_set)
    # train_o59_classifier(device, seed, train_set, valid_set)

    # 단일 모델 1 (efficientnet), 2(resnext)
    # train_general_classifier(device, seed, train_set, valid_set)

    # Combined 모델 3
    # 58세까지 60세로 그룹으로 편성 수정
    # train_mask_classifier(device, seed, train_set, valid_set)
    # train_o59_classifier(device, seed, train_set, valid_set)
    # train_gender_u30_combined_classifier(device, seed, train_set, valid_set)

    # Enssemble 모델
    train_general_classifier(device, seed, train_set, valid_set)
    # train_mask_classifier(device, seed, train_set, valid_set)
    # train_gender_classifier(device, seed, train_set, valid_set)
    # train_u30_classifier(device, seed, train_set, valid_set)
    # train_o59_classifier(device, seed, train_set, valid_set)

    
def train_mask_classifier(device, seed, train_set, valid_set):
    mask_trainee = Trainee("mask-classifier", device=device)
    mask_trainee.batch_size = 512
    mask_trainee.epochs = 10
    mask_trainee.prepare_dataset(train_set, valid_set, DatasetType.Mask_Combined, random_seed=seed)
    mask_trainee.log_interval = int(len(mask_trainee.train_set_loader) / 3)

    mask_trainee.model = BaseModel(num_classes=3).to(device)
    mask_trainee.criterion = FocalLoss()
    mask_trainee.optimizer = optim.Adam(
        mask_trainee.model.parameters(), lr=0.0001
    )
    mask_trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        mask_trainee.optimizer, 
        T_max=50, 
        eta_min=0
    )

    mask_trainee.train()


def train_gender_classifier(device, seed, train_set, valid_set):
    mask_trainee = Trainee("gender-classifier", device=device)
    mask_trainee.batch_size = 512
    mask_trainee.epochs = 10
    mask_trainee.prepare_dataset(train_set, valid_set, DatasetType.Gender, random_seed=seed)
    mask_trainee.log_interval = int(len(mask_trainee.train_set_loader) / 3)

    mask_trainee.model = BaseModel(num_classes=2).to(device)
    mask_trainee.criterion = FocalLoss()
    mask_trainee.optimizer = optim.Adam(
        mask_trainee.model.parameters(), lr=0.0001
    )
    mask_trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        mask_trainee.optimizer, 
        T_max=50, 
        eta_min=0
    )

    mask_trainee.train()


def train_u30_classifier(device, seed, train_set, valid_set):
    mask_trainee = Trainee("u30-classifier", device=device)
    mask_trainee.batch_size = 512
    mask_trainee.epochs = 10
    mask_trainee.prepare_dataset(train_set, valid_set, DatasetType.Under30Age, random_seed=seed)
    mask_trainee.log_interval = int(len(mask_trainee.train_set_loader) / 3)

    mask_trainee.model = BaseModel(num_classes=2).to(device)
    mask_trainee.criterion = FocalLoss()
    mask_trainee.optimizer = optim.Adam(
        mask_trainee.model.parameters(), lr=0.0001
    )
    mask_trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        mask_trainee.optimizer, 
        T_max=50, 
        eta_min=0
    )

    mask_trainee.train()


def train_o59_classifier(device, seed, train_set, valid_set):
    mask_trainee = Trainee("o59-classifier", device=device)
    mask_trainee.batch_size = 512
    mask_trainee.epochs = 10
    mask_trainee.prepare_dataset(train_set, valid_set, DatasetType.Over59Age, random_seed=seed)
    mask_trainee.log_interval = int(len(mask_trainee.train_set_loader) / 3)

    mask_trainee.model = BaseModel(num_classes=2).to(device)
    mask_trainee.criterion = FocalLoss()
    mask_trainee.optimizer = optim.Adam(
        mask_trainee.model.parameters(), lr=0.0001
    )
    mask_trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        mask_trainee.optimizer, 
        T_max=50, 
        eta_min=0
    )

    mask_trainee.train()


def train_gender_u30_combined_classifier(device, seed, train_set, valid_set):
    mask_trainee = Trainee("gender-u30-classifier", device=device)
    mask_trainee.batch_size = 64
    mask_trainee.epochs = 10
    mask_trainee.prepare_dataset(train_set, valid_set, DatasetType.Gender_U30_Combined, random_seed=seed)
    mask_trainee.log_interval = int(len(mask_trainee.train_set_loader) / 3) # 1 epoch당 3번씩 기록

    mask_trainee.model = BaseModel(num_classes=4).to(device)
    mask_trainee.criterion = FocalLoss()
    mask_trainee.optimizer = optim.Adam(
        mask_trainee.model.parameters(), lr=0.0001
    )
    mask_trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        mask_trainee.optimizer, 
        T_max=50, 
        eta_min=0
    )

    mask_trainee.train()


def train_general_classifier(device, seed, train_set, valid_set):
    mask_trainee = Trainee("gen-classifier", device=device)
    mask_trainee.batch_size = 512
    mask_trainee.epochs = 16
    mask_trainee.prepare_dataset(train_set, valid_set, DatasetType.General, random_seed=0)
    mask_trainee.log_interval = int(len(mask_trainee.train_set_loader) / 3)

    mask_trainee.model = BaseModel(num_classes=18).to(device)
    mask_trainee.criterion = FocalLoss()
    mask_trainee.optimizer = optim.Adam(
        mask_trainee.model.parameters(), lr=0.0001
    )   
    mask_trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        mask_trainee.optimizer, 
        T_max=50, 
        eta_min=0
    )

    mask_trainee.train()


def predict_from_models():
    print(f"PyTorch version: {torch.__version__}.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    dataset, answer_board = generate_test_datasets("/opt/ml/input/data")
    predictor = Predictor_M4(16, dataset, answer_board, device)
    predictor.predict()
    # 해야할 일 3개 모델 합한 모델 predictor를 작성할 것


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    # train_model()
    predict_from_models()
