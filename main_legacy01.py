import torch
import torch.nn as nn
from torch import optim
import mask_detector.trainer as trainer
from mask_detector.dataset import DatasetType, generate_train_datasets, generate_test_datasets
from mask_detector.models import BaseModel
from mask_detector.combined_predictor import Predictor_M1, submission_label_recalc
import numpy as np
import random

def train_predictor():
    print(f"PyTorch version: {torch.__version__}.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    seed = 92834
    seed_everything(seed)

    # 데이터셋 생성
    dataset_root = "/opt/ml/input/data"
    train_set, valid_set = generate_train_datasets(dataset_root, random_seed=seed, validation_ratio=0.225)

    # training_model("gender-classifier", DatasetType.Gender, train_set, valid_set, device, seed)
    # training_model("no-mask-classifier", DatasetType.Mask_Weared, train_set, valid_set, device, seed)
    # training_model("good-mask-classifier", DatasetType.Correct_Mask, train_set, valid_set, device, seed)
    # training_model("o60-classifier", DatasetType.Over59Age, train_set, valid_set, device, seed)
    training_model("u30-classifier", DatasetType.Under30Age, train_set, valid_set, device, seed)

def predict_label():
    print(f"PyTorch version: {torch.__version__}.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    dataset_root = "/opt/ml/input/data"
    dataset, answer_board = generate_test_datasets(dataset_root)
    predictor = Predictor_M1(batch_size=128, dataset=dataset, answer_board=answer_board, device=device)
    predictor.predict()

def training_model(model_name, dataset_type, train_set, valid_set, device, random_seed, load_prev = False, custom_epoch = None):
    epochs = 32
    if custom_epoch is not None:
        epochs = custom_epoch
    batch_size = 256
    logging_interval = int(len(train_set) / (batch_size * 3))
    lr = 0.0001

    # 모델 및 메트릭
    model = BaseModel(num_classes=2).to(device)
    if load_prev:
        model.load_state_dict(torch.load(f"result/checkpoint/{model_name}/gender_last_model.pth"))
    # 그래픽카드가 2개 이상인 경우, 고려
    # model = torch.nn.DataParallel(model)    # GPU는 하나 밖에 없는데....? ㅠㅠ

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Trainee 생성
    gender_classifier_trainee = trainer.generate_trainee(
        model_name,
        model,
        criterion,
        optimizer,
        device
    )
    # 체크포인트는 읽지 않는다 (학습을 중간에 그만두는 경우는 없어서...)
    # gender_classifier_trainee.load_last_checkpoint()
    gender_classifier_trainee.batch_size = batch_size
    gender_classifier_trainee.log_interval = logging_interval
    gender_classifier_trainee.epochs = epochs

    gender_classifier_trainee.prepare_dataset(train_set, valid_set, dataset_type, random_seed=random_seed)
    gender_classifier_trainee.train()

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    # predict_label()
    submission_label_recalc()