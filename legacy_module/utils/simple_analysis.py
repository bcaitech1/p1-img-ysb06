import os
import pandas as pd
import matplotlib.pyplot as plt

def run(root_path: str):
    plt.figure(figsize=(8, 6))

    # Train Set 이미지가 있는 폴더 체크
    train_img_list = os.listdir(f"{root_path}/train/images")
    train_img_list = [folder for folder in train_img_list if folder[:2] != "._"]
    train_img_list.sort()

    # Test Set 이미지가 있는 폴더 체크
    test_img_list = os.listdir(f"{root_path}/eval/images")
    test_img_list = [folder for folder in test_img_list if folder[:2] != "._"]

    # 1명 당 7개의 이미지가 존재
    # Train Set에서는 1명 당 폴더로 구분되어 있음
    print("\n===== File analysis =====")
    print(f"Train Set Images: {len(train_img_list)}")
    # Test Set에서는 image 폴더에 몰아서 있음
    print(f"Test Set Images: {int(len(test_img_list) / 7)}")

    print("\n===== CSV Label =====")
    # Label CSV 읽기
    train_label_raw = pd.read_csv(f"{root_path}/train/train.csv")
    print(train_label_raw)
    # print(*train_img_list, sep='\n')
    count = 0
    for row in train_label_raw.iloc:
        if row["path"] in train_img_list:
            count += 1
    print()

    if count != len(train_img_list):
        print(f"Label is corrupted: {count} / {len(train_img_list)}")
    else:
        print(f"Label is perfect!: {count} / {len(train_img_list)}")

    print("\n===== Label Data Analysis")
    # Gender Count
    gender_data = train_label_raw["gender"].value_counts()
    print(gender_data)
    plt.subplot(2, 2, 1)
    plt.xlabel("Gender")
    plt.bar(gender_data.index, gender_data.tolist())
    print()

    # Race Count
    race_data = train_label_raw["race"].value_counts()
    print(race_data)
    plt.subplot(2, 2, 2)
    plt.xlabel("Race")
    plt.bar(race_data.index, race_data.tolist())
    print()

    # Age Count
    age_data = train_label_raw["age"].value_counts()
    groupA = age_data[age_data.index < 30]
    groupB = age_data[(age_data.index > 30) & (age_data.index <= 60)]
    groupC = age_data[age_data.index >= 60]

    print(f"<30          : {groupA.sum()}")
    print(f">=30 and <60 : {groupB.sum()}")
    print(f">=60         : {groupC.sum()}")
    plt.subplot(2, 2, 3)
    plt.xlabel("Age")
    plt.bar(age_data.index, age_data.tolist())
    plt.subplot(2, 2, 4)
    plt.xlabel("Age Group")
    plt.bar(["~29", "30~59", "60~"], [groupA.sum(), groupB.sum(), groupC.sum()])
    print()

    plt.subplots_adjust(hspace=0.5)
    if os.path.isdir("./sresult/"):
        os.mkdir("./result/")
    plt.savefig("./result/train_set_label.png")
    plt.show()
