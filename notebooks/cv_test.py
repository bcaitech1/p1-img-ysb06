import cv2 as cv

cascade_filename = './haarcascade_eye.xml'
cascade = cv.CascadeClassifier(cascade_filename)


def recognite(path: str):
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    grey_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    results = cascade.detectMultiScale(
        grey_image,         # 입력 이미지
        scaleFactor=1.5,    # 이미지 피라미드 스케일 factor
        minNeighbors=5,     # 인접 객체 최소 거리 픽셀
        minSize=(20, 20)    # 탐지 객체 최소 크기
    )

    centers = 2
    for box in results:
        # 좌표 추출
        x, y, w, h = box
        # 경계 상자 그리기
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
        cv.circle(image, (x, y), 10, (255, 0, 0))
    
    return image
