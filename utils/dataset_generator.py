import cv2
import pandas as pd
from glob import glob
from pandas.core.series import Series
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class MaskedFaceTrainDataset(Dataset):
    def __init__(self, transform: transforms.Compose) -> None:
        self.images = []
        self.gender = []
        self.age_group = []
        self.mask_weared = []
        self.mask_in_good_condition = []

        self.transform = transform

    def __len__(self):
        if len(self.images) != len(self.gender) or \
                len(self.images) != len(self.age_group) or \
                len(self.images) != len(self.mask_weared) or \
                len(self.images) != len(self.mask_in_good_condition):
            raise Exception("Dataset is broken")
        return len(self.images)

    def __getitem__(self, index) -> Tensor:
        image = self.transform(self.images[index])
        label = Tensor([
            self.gender[index],
            self.age_group[index],
            self.mask_weared[index],
            self.mask_in_good_condition[index]
        ])
        return image, label


class MaskedFaceTestDataset(Dataset):
    def __init__(self, transform: transforms.Compose) -> None:
        self.images = []
        self.path = []
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Tensor:
        image = self.transform(self.images[index])
        return image, self.path[index]

def generate_train_datasets(
    data_root_path: str,
    random_seed: int = None,
    validation_ratio: int = 0.1,   # 추후 Validation 구현할 것
):
    image_path = f"{data_root_path}/train/images"
    label_raw = pd.read_csv(f"{data_root_path}/train/train.csv")
    if random_seed is not None:
        label_raw = label_raw.sample(frac=1, random_state=random_seed)
        label_raw = label_raw.reset_index(drop=True)

    dataset = MaskedFaceTrainDataset(transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    for label_data in tqdm(label_raw.iloc, total=len(label_raw)):
        target_image_dir = image_path + '/' + label_data["path"] + '/'

        inc_mask_image_path = glob(target_image_dir + "incorrect_mask.*")[0]
        mask_images_path = glob(target_image_dir + "mask*")
        normal_image_path = glob(target_image_dir + "normal.*")[0]

        __add_data_to_dataset(dataset, label_data,inc_mask_image_path, True, False)
        for mask_image_path in mask_images_path:
            __add_data_to_dataset(dataset, label_data, mask_image_path, True, True)
        __add_data_to_dataset(dataset, label_data, normal_image_path, False, False)

    return dataset

def generate_test_dataset(data_root_path: str):
    eval_images_path = glob(f"{data_root_path}/eval/images/*")

    dataset = MaskedFaceTestDataset(transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    for image_path in tqdm(eval_images_path):
        dataset.images.append(cv2.imread(image_path, cv2.IMREAD_COLOR))
        dataset.path.append(image_path.split('/')[-1])

    return dataset
    

def __add_data_to_dataset(
    dataset: MaskedFaceTrainDataset,
    data: Series,
    image_path: str,
    mask_weared: bool,
    mask_in_good_condition: bool
):
    # Alpha Chennel을 살리는 것이 좋은가?
    dataset.images.append(cv2.imread(image_path, cv2.IMREAD_COLOR))
    dataset.gender.append(0 if data["gender"] == "male" else 1)
    if data["age"] < 30:
        dataset.age_group.append(0)
    elif data["age"] >= 60:
        dataset.age_group.append(2)
    else:
        dataset.age_group.append(1)
    dataset.mask_weared.append(1 if mask_weared else 0)
    dataset.mask_in_good_condition.append(1 if mask_in_good_condition else 0)
