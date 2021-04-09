# Mask Detector 주요 모듈

본 폴더는 main에서 핵심적으로 사용되는 모듈들이 포함되어 있다.

## dataset

데이터셋을 관리하는 모듈로 필요한 데이터셋을 생성하는 역할을 한다. 아래 1, 2 그리고 5 외에 외부에서 Import하여 사용할 것은 거의 없다고 볼 수 있다.

1. generate_train_datasets

    학습용 데이터셋을 얻기 위해서는 해당 함수를 호출한다. 해당 함수는 train 폴더로부터 데이터를 읽고 Train set과 Validation set으로 나누어 Pytorch의 Dataset을 상속받은 두 개의 객체를 반환한다.

2. generate_test_datasets

    테스트용 데이터셋을 얻기 위한 함수. 해당 함수는 eval 폴더로부터 이미지를 읽고 데이터셋을 제공한다. 또한 정답을 기록할 pandas.DataFrame 객체도 같이 제공한다.

3. MakeFaceDataset

    데이터셋 클래스, Pytorch에서 사용되는 Dataset이 가져야 할 기본적인 기능을 포함하고 있다. 여기에 추가로 generate_serve_list를 통해 데이터셋이 제공할 레이블 종류를 변경할 수 있고 Oversampling도 가능하다.

4. Person, PersonLabel

    데이터셋에 저장되는 데이터 원형 클래스. PersonLabel은 Person에 속한다. 주의할 점은 Person은 이미지 하나에 대한 것을 나타내지 실제 사람 개별의 프로필을 나타내지 않는다. (원래부터 사람마다 따로 관리한 계획은 없었음)

    PersonLabel에서는 레이블 정보를 저장하고 있는 것 외에 요청에 따라 다른 레이블을 제공하는 함수들을 포함함. 좀 복잡하게 되어 있는데 원래는 DatasetType enum 클래스를 사용할 생각은 없었음. 수정 필요.

5. DatasetType

    데이터셋의 특징을 나타내는 enum 클래스. 이 값에 따라 데이터셋에서 제공되는 레이블이 달라진다.

## loss

    제공된 Baseline 코드에서 그대로 복사한 코드이다. 현재 Focal Loss만 가져와서 사용하고 있다.

## models

프로젝트에서 사용되는 모델을 포함하는 모듈. 쓰다보디 다른 모듈을 사용할 필요가 없어서 BaseModel 하나만 사용된다.

1. BaseModel

    기본으로 사용되는 모델. 현재는 ResNeXt-101을 백본으로 사용하는 모델이다. 전 프로젝트에서 공통으로 사용하는 모델

2. GenderClassifierModel

    더 이상 사용되지 않는 성별 분류 모델

## trainer

모델 학습에 관련된 코드를 포함하는 모듈. 몇 가지 함수들이 있지만 최종적으로는 외부에서 사용되는 함수들은 없고 Trainee 클래스 하나만 외부에서 사용된다.

1. Trainee

    하이퍼파라미터와 모델 학습을 위한 코드를 가지고 있는 클래스. Trainee 정의 시 하이퍼파라미터들을 같이 정의해주고 train함수를 통해 학습을 수행한다. train 함수에는 학습 수행 및 Validation 기능을 기본으로 여기에 TensorBoard 기록 및 yaml로 결과 요약하는 기능까지 포함되어 있다.

    Trainee를 사용하는 예시는 main 함수에 잘 나와 있다.

## combined_predictor

    이 모듈에는 정답을 출력하기 위한 코드들이 포함되어 있다. 기본적으로 학습한 모델들을 조합하고 추론하는 클래스가 포함되어 있다. 자세한 내용을 적기에는 여백이 부족하여 작성하지 않는다.