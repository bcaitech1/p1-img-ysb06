
import os
import pickle
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch
import yaml
from torch import device, nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import DatasetType, MaskedFaceDataset
from .models import BaseModel

save_path = "./result/"
trainee_save_path = f"{save_path}trainee/"
checkpoint_save_path = f"{save_path}checkpoint/"
tensorboard_log_path = f"{save_path}tensorboard/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)

if not os.path.isdir(trainee_save_path):
    os.mkdir(trainee_save_path)

if not os.path.isdir(checkpoint_save_path):
    os.mkdir(checkpoint_save_path)

if not os.path.isdir(trainee_save_path):
    os.mkdir(trainee_save_path)


class Trainee():
    def __init__(
            self,
            name: str,
            device: device
        ) -> None:
        self.name = name

        self.device = device
        
        self.batch_size = 1
        self.epochs = 128
        self.train_set_loader: DataLoader = None
        self.valid_set_loader: DataLoader = None

        self.model: BaseModel = None
        self.criterion: nn.Module = None
        self.optimizer = None
        self.scheduler = None

        self.log_interval = 16
    
    def prepare_dataset(self, train_set: MaskedFaceDataset, valid_set: MaskedFaceDataset, train_type: DatasetType, random_seed: int=None):
        print(f"---- Switch training dataset to {train_type.name} dataset")
        train_set.generate_serve_list(train_type, shuffle=True, random_seed=random_seed)
        print(f"---- Switch to validatation dataset as {train_type.name} dataset")
        valid_set.generate_serve_list(train_type, shuffle=True)

        print()
        print(f"Train Set Size: {len(train_set)}")
        print(f"Valid Set Size: {len(valid_set)}")

        log_trained_content(self, train_type)

        pin_memory = "cuda" in str(self.device)
        self.train_set_loader = DataLoader(
            train_set, 
            batch_size=self.batch_size, 
            num_workers=4, 
            shuffle=True, 
            pin_memory=pin_memory
        )
        self.valid_set_loader = DataLoader(
            valid_set, 
            batch_size=self.batch_size, 
            num_workers=4, 
            shuffle=True, 
            pin_memory=pin_memory
        )

    def load_last_checkpoint(self):
        self.model.load_state_dict(torch.load(f"{save_path}/gender_last_model.pth"))

    def train(self):
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
        logger = SummaryWriter(
            log_dir=tensorboard_log_path + current_time
        )
        log_trained_time(self, current_time)

        best_valid_accuracy = 0
        best_valid_loss = np.inf

        for epoch in range(self.epochs):
            # Train loop
            print(f"\n----- Start Epoch: {datetime.now()}-----\n")

            self.model.train()
            loss_value = 0
            matches = 0
            for idx, (sources, labels, path_texts) in enumerate(self.train_set_loader):
                # Load Data
                sources = sources.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                # Initialize Gradient
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(sources)
                predicts = torch.argmax(outputs, dim=-1)
                loss = self.criterion(outputs, labels)
                
                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # Examination
                loss_value += loss.item()
                matches += (predicts == labels).sum().item()
                if (idx + 1) % self.log_interval == 0:
                    # Calc
                    train_loss = loss_value / self.log_interval
                    train_accuracy = matches / (self.batch_size * self.log_interval)
                    current_lr = _get_lr(self.optimizer)

                    # Print examination result
                    print(f"Epoch: [{epoch + 1}/{self.epochs}] ({idx + 1}/{len(self.train_set_loader)})")
                    print(f"Training loss: {train_loss:4.4}")
                    print(f"Training accuracy: {train_accuracy:4.2%}")
                    print(f"Learning Rate: {current_lr}")
                    print()

                    # Write at Tensorboard
                    logger.add_scalar("Train/loss", train_loss, epoch * len(self.train_set_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_accuracy, epoch * len(self.train_set_loader) + idx)

                    # Initialize examination
                    loss_value = 0
                    matches = 0

            self.scheduler.step()

            # Validation
            with torch.no_grad():    
                print("Calculating validation results...")
                self.model.eval()

                loss_value = 0
                matches = 0
                valid_sample = None
                for sources, labels, path_texts in self.valid_set_loader:
                    sources = sources.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self.model(sources)
                    predicts = torch.argmax(outputs, dim=-1)
                    loss = self.criterion(outputs, labels)
                    loss_value += loss.item()
                    matches += (predicts == labels).sum().item()

                    # 추후 Label Smotheness 적용 후 가장 안 좋은 결과에 대해 저장
                    if valid_sample is None:
                        source_new = torch.clone(sources).detach().cpu().permute(0, 2, 3, 1).numpy()
                        valid_sample = make_sample(source_new, labels, predicts)
                
                # Validation 결과 계산
                valid_loss = loss_value / len(self.valid_set_loader)
                valid_accuracy = matches / len(self.valid_set_loader.dataset)
                best_valid_loss = min(best_valid_loss, valid_loss)

                # 모델 저장
                # 추후 f1-score로 바꾸기
                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    print(f"New best model for val accuracy : {valid_accuracy:4.2%}! saving the best model..")
                    if not os.path.isdir(f"{checkpoint_save_path}{self.name}/"):
                        os.mkdir(f"{checkpoint_save_path}{self.name}/")
                    torch.save(self.model.state_dict(), f"{checkpoint_save_path}{self.name}/best.pth")
                torch.save(self.model.state_dict(), f"{checkpoint_save_path}{self.name}/last.pth")
                
                # Validation 결과 출력
                print(f"Best loss: {best_valid_loss:4.4}")
                print(f"Current loss: {valid_loss:4.4}")
                print(f"Best accuracy: {best_valid_accuracy:4.2%}")
                print(f"Current accuracy: {valid_accuracy:4.2%}")

                logger.add_scalar("Val/loss", valid_loss, epoch)
                logger.add_scalar("Val/accuracy", valid_accuracy, epoch)
                logger.add_figure("Results", valid_sample, epoch)

                print(f"\n----- End Epoch: {datetime.now()}-----\n")

def generate_trainee(
        name: str,
        model: BaseModel,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: device
    ) -> Trainee:
    
    trainee = Trainee(name, device)
    trainee.model = model
    trainee.criterion = criterion
    trainee.optimizer = optimizer
    # 이상하게 이미 만들어진 scheduler 변수를 매개변수로 받아 넣으면 에러가 발생한다.
    # 뭔가 Scheduler에서 optimizer의 변수 참조가 넘겨 받는 과정에서 
    # optimizer step을 추적하는 함수 참조가 유실되어 schedular가 업데이트 되지 않는 것 같지만 정확한 것은 모름.
    trainee.scheduler = optim.lr_scheduler.CosineAnnealingLR(trainee.optimizer, T_max=50, eta_min=0)
    # trainee.scheduler = scheduler

    with open(trainee_save_path + "/" + trainee.name, 'wb') as fw:
        pickle.dump(trainee, fw)
    
    data = {
        "Trainee Info": {
            "Trained Time": [],
            "Trained Content": None,
            "Trained Env" : {
                "Model Name" : trainee.model._get_name(),
                "Model Backbone": trainee.model.backbone._get_name(),
                "Criterion": trainee.criterion._get_name(),
                "Optimizer": _get_class_name(trainee.optimizer),
                "LR Scheduler": _get_class_name(trainee.scheduler)
            },
        }
    }
    with open(trainee_save_path + "/" + trainee.name + "_summary", 'w') as fw:
        yaml.dump(data, fw)

    return trainee


def load_trainee(name: str) -> Trainee:
    with open(trainee_save_path + "/" + name, 'rb') as fr:
        trainee: Trainee = pickle.load(fr)

    # 필요하다면 체크포인트 부르는 코드도 작성
    return trainee


def log_trained_time(trainee: Trainee, time: str):
    if os.path.isfile(trainee_save_path + "/" + trainee.name + "_summary"):
        with open(trainee_save_path + "/" + trainee.name + "_summary", 'r') as f:
            data = yaml.load(f, yaml.FullLoader)
        
        data["Trainee Info"]["Trained Time"].append(time)

        with open(trainee_save_path + "/" + trainee.name + "_summary", 'w') as f:
            yaml.dump(data, f)
    else:
        print(f"Warning: No summary file for trainee {trainee.name}")


def log_trained_content(trainee: Trainee, content: DatasetType):
    if os.path.isfile(trainee_save_path + "/" + trainee.name + "_summary"):
        with open(trainee_save_path + "/" + trainee.name + "_summary", 'r') as f:
            data = yaml.load(f, yaml.FullLoader)
        
        data["Trainee Info"]["Trained Content"] = content.name

        with open(trainee_save_path + "/" + trainee.name + "_summary", 'w') as f:
            yaml.dump(data, f)
    else:
        print(f"Warning: No summary file for trainee {trainee.name}")
    

def _get_lr(optimizer: Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _get_class_name(target: Any):
    return str(target.__class__).replace('\'', '').replace('>', '').split('.')[-1]


def make_sample(images, labels, predicts):
    # 주의 : Batch가 아닌 경우 오류 발생
    if images.shape[0] > 8:
        images = images[0:8]
        labels = labels[0:8]
        predicts = predicts[0:8]

    sample_figure = plt.figure(figsize=(20, 10))
    label_dict = ["Male", "Female"]

    for idx, (image, label, predict) in enumerate(zip(images, labels.squeeze(), predicts)):
        label = label.item()
        predict = predict.item()

        plt.subplot(2, 4, idx + 1, title=f"Pred: {label_dict[predict]} / Label: {label_dict[label]}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)

    return sample_figure
