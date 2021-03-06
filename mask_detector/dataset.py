from os import sep
import random
from enum import Enum
from glob import glob
from typing import Any, List, Tuple

import albumentations as A
import cv2 as cv
import pandas as pd
from albumentations.augmentations import SmallestMaxSize
from albumentations.pytorch import ToTensorV2
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class Gender(Enum):
    Male = 0
    Female = 1

class DatasetType(Enum):
    General = 0
    Mask_Weared = 1
    Correct_Mask = 2
    Gender = 3
    Under30Age = 4
    Over59Age = 5
    Mask_Combined = 6
    Gender_U30_Combined = 7


class PersonLabel():
    def __init__(self) -> None:
        self.mask_exist: bool = False
        self.correct_mask: bool = False
        self.gender: Gender = Gender.Male
        self.age: int = 0
    
    def __repr__(self) -> str:
        mask = "Correct Mask" if self.correct_mask else "Incorrect Mask"
        mask = mask if self.mask_exist else "No mask"
        return str({"Mask": mask, "Gender": self.gender.name, "Age" : self.age})

    def get_combined_label(self) -> int:
        label = 0
        if not self.correct_mask:
            label = 6

        if not self.mask_exist:
            label = 12

        label += self.gender.value * 3 + self.get_age_label()

        return label

    def get_mask_exist_label(self) -> int:
        return self.mask_exist

    def get_correct_mask_label(self) -> int:
        return self.correct_mask

    def get_gender_label(self) -> int:
        return self.gender.value

    def get_age_label(self) -> int:
        if self.age < 30:
            return 0
        elif self.age > 59:
            return 2
        else:
            return 1

    def get_under30_label(self) -> int:
        return int(self.age < 30)

    def get_over59_label(self) -> int:
        return int(self.age > 58)
        # 기준을 58세 이상으로 바꾼거
        # 이거 좀 꼼수인데....일단은 ㄱㄱ
        # 1세만 높임
    
    def get_label(self, label_type: DatasetType) -> int:
        label: int = None
        if label_type == DatasetType.Mask_Weared:
            label = self.get_mask_exist_label()
        elif label_type == DatasetType.Correct_Mask:
            label = self.get_correct_mask_label()
        elif label_type == DatasetType.Gender:
            label = self.get_gender_label()
        elif label_type == DatasetType.Under30Age:
            label = self.get_under30_label()
        elif label_type == DatasetType.Over59Age:
            label = self.get_over59_label()
        elif label_type == DatasetType.Mask_Combined:
            if self.get_mask_exist_label():
                if self.get_correct_mask_label():
                    label = 0
                else:
                    label = 1
            else:
                label = 2
        elif label_type == DatasetType.Gender_U30_Combined:
            # 모델을 가능한 한 적게 하는 것이 좋다는 것이 결론
            # 마스크는 너무 확실해서 분리해도 좋지만
            # 남녀의 경우 성능이 잘 나오지 않으므로 성능이 안 나오는 것 끼리 뭉쳐서 Tuning
            # 60이상은 데이터가 없어서 원래 낮으므로 포기하는 수준
            # 남자 <30 0, 여자 <30 1
            # 남자 >=30 2, 여자 >=30 3
            label = self.get_under30_label() * 2
            label += self.get_gender_label()
        else:
            label = self.get_combined_label()
        
        return label


def get_label_count(label_type: DatasetType) -> int:
    if label_type.value >= 1 and label_type.value <= 5:
        return 2
    elif label_type == DatasetType.Mask_Combined:
        return 3
    elif label_type == DatasetType.Gender_U30_Combined:
        return 4
    else:
        return 18

# 레이블 제공을 유연하게 할 수 있는 방법이 있을까?
# Baseline도 내 코드처럼 복잡하지 않을 뿐이지 별도의 함수를 만듦
# 저 보기 싫은 get_label_count를 합치고 유연하게 구조를 짤 수 있을 것 같은데 생각할 시간이 없음...


class Person():
    def __init__(self) -> None:
        self.image_path: str = ""
        self.image_raw: Any = None
        self.label: PersonLabel = PersonLabel()

# 클래스 명이 Person인 것은 원래는 사진 하나마다 다른 사람으로 처리할 생각이었음
# 지금은 한 사람이 아닌 Picture 하나라고 보면 됨


class MaskedFaceDataset(Dataset):
    def __init__(self) -> None:
        self.data: List[Person] = []
        self.transform: transforms.Compose = None
        self.serve_type = DatasetType.General
        self.serve_list = []
        # 증가된 데이터셋 크기 (원래 데이터셋 크기는 len(self.data))
        # 원래는 증가된 만큼 Epoch를 줄이는 기준으로 사용하려고 했는데
        # 그냥 tensorboard보고 수동으로 꺼버리는 것이 편함. 
        self.grown_size: int = 0    

    def __len__(self):
        return len(self.serve_list)

    def __getitem__(self, index) -> Tensor:
        target: Person = self.data[self.serve_list[index]]

        source = self.transform(image=target.image_raw)["image"]
        label = target.label.get_label(self.serve_type)

        return source, label, target.image_path

    def generate_serve_list(self, serve_type: DatasetType, shuffle: bool = False, random_seed: int = None, oversampling: bool = True):
        # 레이블 데이터를 Class List가 아닌 그냥 Pandas로 저장했으면 훨씬 더 좋았을 것 같음
        # 그러면 pandas가 제공하는 기능을 그대로 쓸 수 있었을 텐데...
        # 갑자기 든 생각이지만 성별, 나이 구분도 마스크 여부에 따라 달라지지 않을까?
        # 하지만 이것저것 다 고려하면 머리 터질 것 같아 더 이상 생각하지 않음
        # 하위 수준 모델(마스크 제대로 썼는지, 60이상인지)에서 들어오는 데이터는 정확할 것이라고 판단.

        # 마스크가 없는 데이터는 아예 넣지 않음
        # 그런데 넣어도 될 것 같음...여유가 되면 넣은 것과 안 넣은 것 비교해 볼 것

        self.serve_type = serve_type
        rand = random.Random(random_seed)

        data_by_target_class = [
            [] for _ in range(get_label_count(serve_type))
        ]
        
        for index, data in enumerate(self.data):
            data_by_target_class[data.label.get_label(serve_type)].append(index)
        
        # 제일 많은 데이터를 가진 클래스 판별
        greatest_class_index = 0
        greatest_class_size = -1
        for index, group in enumerate(data_by_target_class):
            if len(group) > greatest_class_size:
                greatest_class_size = len(group)
                greatest_class_index = index

        # Oversampling
        # 근데 Dataloader에 Sampler라는 관련 함수가 있다고...?
        self.grown_size = 0
        for index, group in enumerate(data_by_target_class):
            # 셔플을 아래에서 하는데 여기서도 하는 이유는 
            # 루프 마지막에서 사이즈 맞춰주기 위해 리스트를 자르는데
            # 이 부분에서 편중된 데이터롤 자르지 않게 하기 위해
            if shuffle:
                rand.shuffle(group)

            if index == greatest_class_index:
                continue
            prev_size = len(group)

            if oversampling:
                if len(group) > 0:
                    grow_size = int(greatest_class_size / len(group))
                    group *= grow_size
                group += group[:(greatest_class_size - len(group))]

            self.grown_size += len(group) - prev_size
        
        self.serve_list = []
        for group in data_by_target_class:
            self.serve_list += group
        
        if shuffle:
            rand.shuffle(self.serve_list)

        

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

    # train, valid Dataframe으로 나누는 작업
    valid_size = int(len(label_raw) * validation_ratio)

    # 이미 여기서 사람들을 기준으로 Train과 Valid가 분리되어 있음 
    # (처음 짤 때부터 이렇게 짬...나도 몰랐음 ㅋㅋ)
    valid_label_raw = label_raw.iloc[:valid_size + 1]
    train_label_raw = label_raw.iloc[valid_size + 1:]
    # 원래는 60세 이상의 데이터 갯수가 부족하면 채워넣도록 코드를 짜려고 했지만 
    # 생각보다 원하는 비율로 잘 맞춰지고 있어서 일단 유보

    # 60 이상 분류가 잘 되지 않을 경우 아래 코드를 주석 해제하여 수정할 것
    # count_60 = len(valid_label_raw[valid_label_raw["age"] > 59])
    # min_count_60 = int(192 * validation_ratio)

    train_label_raw.reset_index(drop=True)
    valid_label_raw.reset_index(drop=True)

    # Train Dataset 생성
    train_dataset = MaskedFaceDataset()
    # 추후 Transform 커스텀으로 받을 수 있도록 수정
    train_dataset.transform = get_basic_train_transforms((256, 256))

    for label_data in tqdm(train_label_raw.iloc, total=len(train_label_raw)):
        target_image_dir = image_path + '/' + label_data["path"] + '/'
        gender = Gender.Male if label_data["gender"] == "male" else Gender.Female
        age = int(label_data["age"])

        inc_mask_image_path = glob(target_image_dir + "incorrect_mask.*")[0]
        mask_images_path = glob(target_image_dir + "mask*")
        normal_image_path = glob(target_image_dir + "normal.*")[0]

        target_id: str = label_data["id"]
        __add_data_to_dataset(train_dataset,  inc_mask_image_path, gender, age, True, False, target_id)
        for mask_image_path in mask_images_path:
            __add_data_to_dataset(train_dataset,  mask_image_path, gender, age, True, True, target_id)
        __add_data_to_dataset(train_dataset,  normal_image_path, gender, age, False, False, target_id)

    # Validation Dataset 생성
    valid_dataset = MaskedFaceDataset()
    valid_dataset.transform = get_valid_transforms((256, 256))

    for label_data in tqdm(valid_label_raw.iloc, total=len(valid_label_raw)):
        target_image_dir = image_path + '/' + label_data["path"] + '/'
        gender = Gender.Male if label_data["gender"] == "male" else Gender.Female
        age = int(label_data["age"])

        inc_mask_image_path = glob(target_image_dir + "incorrect_mask.*")[0]
        mask_images_path = glob(target_image_dir + "mask*")
        normal_image_path = glob(target_image_dir + "normal.*")[0]

        target_id: str = label_data["id"]
        __add_data_to_dataset(valid_dataset,  inc_mask_image_path, gender, age, True, False, target_id)
        for mask_image_path in mask_images_path:
            __add_data_to_dataset(valid_dataset,  mask_image_path, gender, age, True, True, target_id)
        __add_data_to_dataset(valid_dataset,  normal_image_path, gender, age, False, False, target_id)

    train_dataset.generate_serve_list(DatasetType.General)
    valid_dataset.generate_serve_list(DatasetType.General)
    
    return train_dataset, valid_dataset


def generate_test_datasets(
        data_root_path: str,
    ):
    image_path = f"{data_root_path}/eval/images"
    answer_board = pd.read_csv(f"{data_root_path}/eval/info.csv")
    target_images_paths = image_path + "/" + answer_board["ImageID"]
    print(target_images_paths)

    dataset = MaskedFaceDataset()
    dataset.transform = get_valid_transforms((256, 256))
    for image_path in tqdm(target_images_paths):
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        data = Person()
        data.image_path = image_path
        data.image_raw = image

        dataset.data.append(data)

    dataset.generate_serve_list(DatasetType.General)
    return dataset, answer_board

    


def __add_data_to_dataset(
    dataset: MaskedFaceDataset,
    image_path: str,
    gender: int,
    age: int,
    mask_weared: bool,
    mask_in_good_condition: bool,
    person_id : str
):
    # Baseline에서는 레이블만 먼저 생성하고 getitem 시점에서 불러오지만
    # 내 코드는 이미지까지 미리 메모리에 다 옮겨 놓고 시작
    # 메모리 낭비가 심하지만 메모리를 최대한 써버리는게 내 코딩 스타일이라...
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    data = Person()
    data.image_path = image_path
    data.image_raw = image
    data.label.gender = gender
    data.label.age = age
    data.label.mask_exist = mask_weared
    data.label.correct_mask = mask_in_good_condition

    data = fix_label(data, person_id)
    dataset.data.append(data)
    

    # 데이터셋에 데이터를 넣는 코드

# 처음에는 상속 필요없이 매개변수 몇 개로 커버가 가능할 줄 알았는데 구조가 너무 복잡해졌음
# 데이터셋을 상속 구조로 제대로 만들었으면 좋았을 터이지만 일단 이정도에서 마무리
# 추후 시간이 되면 코드를 수정해보자. (시간이 없을 것 같지만)

def get_basic_train_transforms(
        image_size: Tuple[int, int], 
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):  
    # https://github.com/lukemelas/EfficientNet-PyTorch
    # Mean, Std는 위 링크를 참조했으며 Efficientnet Pretrained Model의 설정 값으로 추정
    min_length = min(image_size[0], image_size[1])
    train_transforms = A.Compose([
        SmallestMaxSize(max_size=min_length, always_apply=True),
        A.CenterCrop(min_length, min_length, always_apply=True),
        A.Resize(image_size[0], image_size[1], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, rotate_limit=15),
        A.HueSaturationValue(hue_shift_limit=0.2,
                             sat_shift_limit=0.2, 
                             val_shift_limit=0.2, 
                             p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return train_transforms


def get_valid_transforms(
        image_size: Tuple[int, int], 
        mean: Tuple[float, float, float] = (0.548, 0.504, 0.479), 
        std: Tuple[float, float, float] = (0.237, 0.247, 0.246)
    ):
    val_transforms = A.Compose([
        A.Resize(image_size[0], image_size[1], p=1.0),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ])

    return val_transforms


def fix_label(target: Person, target_id: str):
    # 점점 땜빵 식 코드가 되어가지만...
    # 이런 식의 코드가 아니면 수정할 방법이 없음 (코드를 다시 짜든가...ㅠ)
    # Train 레이블을 제대로 주지 않은 운영진을 탓하자
    if "000020" == target_id:
        if not target.label.correct_mask and target.label.mask_exist:
            target.label.mask_exist = False
        elif not target.label.correct_mask and not target.label.mask_exist:
            target.label.mask_exist = True
    elif "000070" == target_id:
        if not target.label.correct_mask and target.label.mask_exist:
            target.label.correct_mask = True
    elif "000645" == target_id:
        if not target.label.correct_mask and target.label.mask_exist:
            target.label.correct_mask = True
    elif "001498-1" == target_id:
        target.label.gender = Gender.Female
    elif "003514" == target_id:
        if not target.label.correct_mask and target.label.mask_exist:
            target.label.correct_mask = True
    elif "004418" == target_id:
        if not target.label.correct_mask and target.label.mask_exist:
            target.label.mask_exist = False
        elif not target.label.correct_mask and not target.label.mask_exist:
            target.label.mask_exist = True
    elif "004432" == target_id:
        target.label.gender = Gender.Female
    elif "005227" == target_id:
        if not target.label.correct_mask and target.label.mask_exist:
            target.label.mask_exist = False
        elif not target.label.correct_mask and not target.label.mask_exist:
            target.label.mask_exist = True
    elif "006359" == target_id:
        target.label.gender = Gender.Male
    elif "006360" == target_id:
        target.label.gender = Gender.Male
    elif "006361" == target_id:
        target.label.gender = Gender.Male
    elif "006362" == target_id:
        target.label.gender = Gender.Male
    elif "006363" == target_id:
        target.label.gender = Gender.Male
    elif "006364" == target_id:
        target.label.gender = Gender.Male
        
    return target