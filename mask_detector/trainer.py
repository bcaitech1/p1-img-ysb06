
import os
import pickle
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytz
import sklearn.metrics as skm
import torch
import yaml
from sklearn.metrics import f1_score
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
        print(f"[{train_set.grown_size}] data increased")
        print(f"---- Switch to validatation dataset as {train_type.name} dataset")
        valid_set.generate_serve_list(train_type, shuffle=True, random_seed=random_seed, oversampling=False)
        print(f"[{valid_set.grown_size}] data increased")

        print()
        print(f"Train Set Size: {len(train_set)}")
        print(f"Valid Set Size: {len(valid_set)}")

        pin_memory = "cuda" in str(self.device)
        self.train_set_loader = DataLoader(
            train_set, 
            batch_size=self.batch_size, 
            num_workers=2, 
            shuffle=True,
            pin_memory=pin_memory
        )
        self.valid_set_loader = DataLoader(
            valid_set, 
            batch_size=self.batch_size,
            num_workers=2, 
            shuffle=True,
            pin_memory=pin_memory
        )

    def load_last_checkpoint(self):
        self.model.load_state_dict(torch.load(f"{save_path}/gender_last_model.pth"))

    def train(self):
        kst = pytz.timezone('Asia/Seoul')
        current_time = datetime.now(kst).strftime("%Y-%m-%d %H.%M.%S")
        logger = SummaryWriter(
            log_dir=f"{tensorboard_log_path}{current_time} {self.name[:6]}"
        )

        best_valid_accuracy = 0
        best_valid_accuracy_epoch = 0
        best_valid_loss = np.inf
        best_f1 = 0
        best_f1_epoch = 0

        for epoch in range(self.epochs):
            # Train loop
            print(f"\n----- Start Epoch: {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')}-----\n")

            self.model.train()
            loss_value = 0
            tp_matches = 0
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
                tp_matches += (predicts == labels).sum().item()
                if (idx + 1) % self.log_interval == 0:
                    # Calc
                    train_loss = loss_value / self.log_interval
                    train_accuracy = tp_matches / (self.batch_size * self.log_interval)
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
                    tp_matches = 0

            self.scheduler.step()

            # Validation
            with torch.no_grad():    
                print("Calculating validation results...")
                self.model.eval()

                loss_value = 0
                tp_matches = 0
                valid_sample = None
                pred_list = []
                label_list = []
                for sources, labels, path_texts in self.valid_set_loader:
                    sources = sources.to(self.device)
                    labels = labels.to(self.device, dtype=torch.long)

                    outputs = self.model(sources)
                    predicts = torch.argmax(outputs, dim=-1)
                    loss = self.criterion(outputs, labels)
                    loss_value += loss.item()
                    tp_matches += (predicts == labels).sum().item()

                    pred_list.extend(predicts.cpu().tolist())
                    label_list.extend(labels.cpu().tolist())
                    # ?????? Label Smotheness ?????? ??? ?????? ??? ?????? ????????? ?????? ??????
                    if valid_sample is None:
                        source_new = torch.clone(sources).detach().cpu().permute(0, 2, 3, 1).numpy()
                        valid_sample = make_sample(source_new, labels, predicts)
                
                # Validation ?????? ??????
                valid_loss = loss_value / len(self.valid_set_loader)
                valid_accuracy = tp_matches / len(self.valid_set_loader.dataset)
                best_valid_loss = min(best_valid_loss, valid_loss)
                f1 = f1_score(label_list, pred_list, average='macro')
                if f1 > best_f1:
                    best_f1 = f1
                    best_f1_epoch = epoch

                # ?????? ??????
                # ?????? f1-score??? ?????????
                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    best_valid_accuracy_epoch = epoch
                    print(f"New best model for val accuracy : {valid_accuracy:4.2%}! saving the best model..")
                    if not os.path.isdir(f"{checkpoint_save_path}{self.name}/"):
                        os.mkdir(f"{checkpoint_save_path}{self.name}/")
                    torch.save(self.model.state_dict(), f"{checkpoint_save_path}{self.name}/best.pth")
                    logger.add_figure("Results", valid_sample, epoch)

                torch.save(self.model.state_dict(), f"{checkpoint_save_path}{self.name}/last.pth")
                
                # Validation ?????? ??????
                print(f"Best loss: {best_valid_loss:4.4}")
                print(f"Current loss: {valid_loss:4.4}")
                print(f"Best accuracy: {best_valid_accuracy:4.2%}")
                print(f"Current accuracy: {valid_accuracy:4.2%}")
                print(f"Best f1: {best_f1:4.2%}")
                print(f"Current f1: {f1:4.2%}")

                logger.add_scalar("Val/loss", valid_loss, epoch)
                logger.add_scalar("Val/accuracy", valid_accuracy, epoch)
                logger.add_scalar("Val/f1", f1, epoch)

                print(f"\n----- End Epoch: {datetime.now(kst).strftime('%Y-%m-%d %H:%M:%S')}-----\n")
        
        print(f"End Training for {self.name}\n\n")
        logger.close()
        # ????????? ????????? ?????? ?????????????????? ???????????? ????????? ????????? (????????? ????????????)
        training_summary = {
            "Training Info": {
                "Trainee": self.name,
                "Training start Time": current_time,
                "Training end Time": datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S"),
                "Trained Environment" : {
                    "Model Name" : self.model._get_name(),
                    "Model Backbone": self.model.backbone._get_name(),
                    "Criterion": self.criterion._get_name(),
                    "Optimizer": _get_class_name(self.optimizer),
                    "LR Scheduler": _get_class_name(self.scheduler)
                },
                "Best Loss": best_valid_loss,
                "Best Accuracy": best_valid_accuracy,
                "Best Accuracy Epoch": best_valid_accuracy_epoch,
                "Best F1-Score": float(best_f1),
                "Best F1-Score Epoch": best_f1_epoch,
            }
        }
        with open(f"{trainee_save_path}/{self.name}_summary.yaml", 'w') as fw:
            yaml.dump(training_summary, fw)

# Deprecated
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
    # ???????????? ?????? ???????????? scheduler ????????? ??????????????? ?????? ????????? ????????? ????????????.
    # ?????? Scheduler?????? optimizer??? ?????? ????????? ?????? ?????? ???????????? 
    # optimizer step??? ???????????? ?????? ????????? ???????????? schedular??? ???????????? ?????? ?????? ??? ????????? ????????? ?????? ??????.
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

    # ??????????????? ??????????????? ????????? ????????? ??????
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
    # ?????? : Batch??? ?????? ?????? ?????? ??????
    if images.shape[0] > 6:
        images = images[0:6]
        labels = labels[0:6]
        predicts = predicts[0:6]

    sample_figure = plt.figure(figsize=(11, 6))

    for idx, (image, label, predict) in enumerate(zip(images, labels.squeeze(), predicts)):
        label = label.item()
        predict = predict.item()

        plt.subplot(2, 3, idx + 1, title=f"Label: {label} / Pred: {predict}")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image)

    return sample_figure
