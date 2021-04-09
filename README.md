# boostcamp-p1-image
Image classification project for Naver AI Tech Boostcamp

기본적으로 아래의 함수들을 사용하여 작업을 수행한다

- train_model

    모델을 학습하는 코드들이 들어있다. 필요한 모델을 주석 해제하여 실행할 수 있다. 현재는 18개 클래스로 나누는 1개 모델에 다른 성별/마스크를 분류하는 모델들이 Advisor로서 역할하는 모델을 학습하는 코드가 주석 해제되어 있다.

    함수 내부에는 모델 학습을 수행하는 함수들이 있는 데 해당 함수는 trainer.Trainee 클래스로부터 학습할 모델을 정의하고 trainer.Trainee.train 함수를 실행하는 코드를 포함한다.
    
    자세한 내용은 trainer 모듈을 참조한다.

- predict_model
    추론 모델을 실행한다. 대상 클래스를 수정하여 실행한다.

    자세한 내용은 combined_predictor 모듈을 참조한다.