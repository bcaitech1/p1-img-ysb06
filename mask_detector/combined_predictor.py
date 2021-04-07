import pandas as pd
import torch
from torch import device
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import MaskedFaceDataset
from .models import BaseModel


class Predictor_M1():
    def __init__(self, batch_size: int, dataset: MaskedFaceDataset, answer_board: pd.DataFrame, device) -> None:
        self.data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
        self.answer_board: pd.DataFrame = answer_board
        self.device = device

        self.no_mask_model = BaseModel(num_classes=2).to(device)
        self.no_mask_model.load_state_dict(torch.load(f"./result/checkpoint/no-mask-classifier/best.pth"))
        self.no_mask_model.eval()
        self.good_mask_model = BaseModel(num_classes=2).to(device)
        self.good_mask_model.load_state_dict(torch.load(f"./result/checkpoint/good-mask-classifier/best.pth"))
        self.good_mask_model.eval()
        self.gender_model = BaseModel(num_classes=2).to(device)
        self.gender_model.load_state_dict(torch.load(f"./result/checkpoint/gender-classifier/best.pth"))
        self.gender_model.eval()
        self.u30_model = BaseModel(num_classes=2).to(device)
        self.u30_model.load_state_dict(torch.load(f"./result/checkpoint/u30-classifier/best.pth"))
        self.u30_model.eval()
        self.o60_model = BaseModel(num_classes=2).to(device)
        self.o60_model.load_state_dict(torch.load(f"./result/checkpoint/o60-classifier/best.pth"))
        self.o60_model.eval()
    
    def predict(self):
        no_mask_predictions = []
        good_mask_predictions = []
        gender_predictions = []
        u30_predictions = []
        o60_predictions = []
        print("Classifying...")
        for sources, _, _ in tqdm(self.data_loader):
            sources = sources.to(self.device)
            no_mask_output = self.gender_model(sources)
            no_mask_output = no_mask_output.argmax(dim=-1)
            no_mask_predictions.extend(no_mask_output.cpu().numpy())

            good_mask_output = self.good_mask_model(sources)
            good_mask_output = good_mask_output.argmax(dim=-1)
            good_mask_predictions.extend(good_mask_output.cpu().numpy())

            gender_output = self.gender_model(sources)
            gender_output = gender_output.argmax(dim=-1)
            gender_predictions.extend(gender_output.cpu().numpy())

            u30_output = self.u30_model(sources)
            u30_output = u30_output.argmax(dim=-1)
            u30_predictions.extend(u30_output.cpu().numpy())

            o60_output = self.o60_model(sources)
            o60_output = o60_output.argmax(dim=-1)
            o60_predictions.extend(o60_output.cpu().numpy())
    
        self.answer_board["mask"] = no_mask_predictions
        self.answer_board["good_mask"] = good_mask_predictions
        self.answer_board["gender"] = gender_predictions
        self.answer_board["u30"] = u30_predictions
        self.answer_board["o59"] = o60_predictions

        self.answer_board.to_csv("./result/submission.csv")
        submission_label_recalc()
        print("Predict Complete")


class Predictor_M2():
    def __init__(self, batch_size: int, dataset: MaskedFaceDataset, answer_board: pd.DataFrame, t_device: device) -> None:
        self.data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        self.answer_board: pd.DataFrame = answer_board
        self.device: device = t_device

        self.mask_model = BaseModel(num_classes=3).to(self.device)
        self.mask_model.load_state_dict(torch.load(f"./result/checkpoint/mask-classifier/best.pth"))
        self.mask_model.eval()
        self.gender_model = BaseModel(num_classes=2).to(self.device)
        self.gender_model.load_state_dict(torch.load(f"./result/checkpoint/gender-classifier/best.pth"))
        self.gender_model.eval()
        self.u30_model = BaseModel(num_classes=2).to(self.device)
        self.u30_model.load_state_dict(torch.load(f"./result/checkpoint/u30-classifier/best.pth"))
        self.u30_model.eval()
        self.o60_model = BaseModel(num_classes=2).to(self.device)
        self.o60_model.load_state_dict(torch.load(f"./result/checkpoint/o59-classifier/best.pth"))
        self.o60_model.eval()
    
    def predict(self):
        mask_predictions = []
        gender_predictions = []
        u30_predictions = []
        o60_predictions = []

        print("Classifying...")
        for sources, _, _ in tqdm(self.data_loader):
            sources = sources.to(self.device)

            mask_output = self.mask_model(sources)
            mask_output = mask_output.argmax(dim=-1)
            mask_predictions.extend(mask_output.cpu().numpy())

            gender_output = self.gender_model(sources)
            gender_output = gender_output.argmax(dim=-1)
            gender_predictions.extend(gender_output.cpu().numpy())

            u30_output = self.u30_model(sources)
            u30_output = u30_output.argmax(dim=-1)
            u30_predictions.extend(u30_output.cpu().numpy())

            o60_output = self.o60_model(sources)
            o60_output = o60_output.argmax(dim=-1)
            o60_predictions.extend(o60_output.cpu().numpy())

        self.answer_board["mask"] = mask_predictions
        self.answer_board["gender"] = gender_predictions
        self.answer_board["u30"] = u30_predictions
        self.answer_board["o59"] = o60_predictions

        self.answer_board.loc[self.answer_board["mask"] == 1, "ans"] += 6
        self.answer_board.loc[self.answer_board["mask"] == 2, "ans"] += 12
        self.answer_board.loc[self.answer_board["gender"] == 1, "ans"] += 3
        self.answer_board.loc[(self.answer_board["u30"] == 0) & (self.answer_board["o59"] == 0), "ans"] += 1
        self.answer_board.loc[self.answer_board["o59"] == 1, "ans"] += 2
        raw_f = self.answer_board[["ImageID", "ans"]]

        print("Printing...")
        self.answer_board.to_csv("./result/result_raw.csv")
        raw_f.to_csv("./result/submission.csv", index=False)
        
        print("Prediction Complete!!")


class Predictor_G1():
    def __init__(self, batch_size: int, dataset: MaskedFaceDataset, answer_board: pd.DataFrame, t_device: device) -> None:
        self.data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)
        self.answer_board: pd.DataFrame = answer_board
        self.device: device = t_device

        self.model = BaseModel(num_classes=18).to(self.device)
        self.model.load_state_dict(torch.load(f"./result/checkpoint/gen-classifier/best.pth"))
        self.model.eval()
    
    def predict(self):
        predictions = []

        print("Classifying...")
        for sources, _, _ in tqdm(self.data_loader):
            sources = sources.to(self.device)

            output = self.model(sources)
            output = output.argmax(dim=-1)
            predictions.extend(output.cpu().numpy())
        
        self.answer_board["ans"] = predictions

        print("Printing...")
        self.answer_board.to_csv("./result/result_raw.csv")
        self.answer_board.to_csv("./result/submission.csv", index=False)

        print("Prediction Complete!!")



def submission_label_recalc():
    raw = pd.read_csv("./result/submission.csv")
    raw.loc[raw["mask"] == 1, "ans"] += 6
    raw.loc[(raw["mask"] == 1) & (raw["good_mask"] == 1), "ans"] += 6
    raw.loc[(raw["gender"] == 1), "ans"] += 3
    raw.loc[(raw["u30"] == 0) & (raw["o59"] == 0), "ans"] += 1
    raw.loc[raw["o59"] == 1, "ans"] += 2
    raw_f = raw[["ImageID", "ans"]]
    print("Printing...")
    raw_f.to_csv("./result/result.csv", index=False)
    print("Finished!")
