from enum import Enum
from typing import Any, List, Tuple

import albumentations as A
import cv2 as cv
import pandas as pd
from albumentations.augmentations import SmallestMaxSize
from albumentations.pytorch import ToTensorV2
from glob import glob
from pandas.core.series import Series
from torch import LongTensor, Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class Gender(Enum):
    Male = 0
    Female = 1


class AgeGroup(Enum):
    Under30 = 0
    Middle30to59 = 1
    Over59 = 2


class PersonLabel():
    def __init__(self) -> None:
        self.mask_exist: bool = False
        self.correct_mask: bool = False
        self.gender: Gender = Gender.Male
        self.age_group: AgeGroup = AgeGroup.Under30

    def get_combined_label(self) -> int:
        label = 0
        if not self.correct_mask:
            label = 6

        if not self.mask_exist:
            label = 12

        label += int(self.gender) * 3 + int(self.age_group)

        return LongTensor([label])

    def get_mask_exist_label(self) -> int:
        return LongTensor([self.mask_exist])

    def get_correct_mask_label(self) -> int:
        return LongTensor([self.correct_mask])

    def get_gender_label(self) -> int:
        return LongTensor([int(self.gender)])

    def get_age_label(self) -> int:
        return LongTensor([int(self.age_group)])

    def get_under30_label(self) -> int:
        return LongTensor([self.age_group == AgeGroup.Under30])

    def get_over59_label(self) -> int:
        return LongTensor([self.age_group == AgeGroup.Over59])

# 레이블 제공을 유연하게 할 수 있는 방법이 있을까?
# Dict로 구성하고 Key를 받는 방식이면 가능했을 듯.
# 다만, string key로 받는 방식을 좋아하지 않아(애매해짐 Vague) 그렇게 하지는 않음


class Person():
    def __init__(self) -> None:
        self.image_path: str = ""
        self.image_raw: Any = None
        self.label: PersonLabel = None


class DatasetType(Enum):
    Unknown = 0
    Mask_Weared = 1
    Correct_Mask = 2
    Gender = 3
    Under30Age = 4
    Over59Age = 5
    Testset = 6


class MaskedFaceDataset(Dataset):
    def __init__(self) -> None:
        self.data: List[Person] = []
        self.transform: transforms.Compose = None
        self.type: DatasetType = DatasetType.Unknown

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tensor:
        target: Person = self.data[index]
        target_image = self.transform(target.image_raw)

        # 혹시 Data를 Load했을 때 3번째 부분이 문제가 될 수 있으므로 테스트 필요
        return target_image, target.label, target.image_path


def get_basic_train_transforms(image_size: Tuple[int, int], mean: Tuple[float, float, float] = (0.548, 0.504, 0.479), std: Tuple[float, float, float] = (0.237, 0.247, 0.246)):
    min_length = min(image_size[0], image_size[1])
    train_transforms = A.Compose([
        SmallestMaxSize(max_size=min_length, always_apply=True),
        A.CenterCrop(min_length, min_length, always_apply=True),
        A.Resize(image_size[0], image_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.2,
                             sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return train_transforms


def get_valid_transforms(image_size: Tuple[int, int], mean: Tuple[float, float, float] = (0.548, 0.504, 0.479), std: Tuple[float, float, float] = (0.237, 0.247, 0.246)):
    val_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1], p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return val_transforms


def generate_train_datasets(
    data_root_path: str,
    random_seed: int = None,
    validation_ratio: int = 0.25,   # 추후 Validation 구현할 것
):
    """학습 데이터셋 생성

    Args:
        data_root_path (str): eval과 train폴더가 들어있는 폴더
        random_seed (int, optional): 학습 데이터셋과 검증 데이터셋으로 나누기 전 랜덤으로 섞을 때 사용하는 시드 값. 기본은 None이며 이 경우 랜덤으로 섞지 않음.
        validation_ratio (int, optional): 검증 데이터 셋 비율. 기본은 일반적으로 사용되는 6:2:2에 해당하는 값
    """
    image_path = f"{data_root_path}/train/images"
    label_raw = pd.read_csv(f"{data_root_path}/train/train.csv")
    if random_seed is not None:
        label_raw = label_raw.sample(frac=1, random_state=random_seed)
        label_raw = label_raw.reset_index(drop=True)

    dataset = MaskedFaceDataset()
    dataset.transform = get_basic_train_transforms((128, 128))

    for label_data in tqdm(label_raw.iloc, total=len(label_raw)):
        target_image_dir = image_path + '/' + label_data["path"] + '/'
        gender = label_data["gender"]
        age = label_data["age"]

        inc_mask_image_path = glob(target_image_dir + "incorrect_mask.*")[0]
        mask_images_path = glob(target_image_dir + "mask*")
        normal_image_path = glob(target_image_dir + "normal.*")[0]

        __add_data_to_dataset(dataset,  inc_mask_image_path, gender, age, True, False)
        for mask_image_path in mask_images_path:
            __add_data_to_dataset(dataset,  mask_image_path, gender, age, True, True)
        __add_data_to_dataset(dataset,  normal_image_path, gender, age, False, False)


def __add_data_to_dataset(
    dataset: MaskedFaceDataset,
    image_path: str,
    gender: int,
    age: int,
    mask_weared: bool,
    mask_in_good_condition: bool
):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # 데이터셋에 데이터를 넣는 코드


generate_train_datasets("/opt/ml/input/data")
