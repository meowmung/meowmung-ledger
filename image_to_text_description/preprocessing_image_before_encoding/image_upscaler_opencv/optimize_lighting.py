import cv2


def optimize_lighting(image_path):
    """
    이미지를 최적화하여 텍스트를 강조하는 함수.

    주어진 이미지에서 조명의 영향을 줄이고 텍스트를 더 잘 보이게 하기 위해
    대비를 조정하고 이진화 처리를 수행합니다.

    Args:
        image_path (str): 처리할 이미지 파일의 경로.

    Returns:
        numpy.ndarray: 이진화된 최종 이미지 (흑백 처리 결과).
    """

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary


# 사용 예시
optimized_image = optimize_lighting("FILE_PATH")
cv2.imwrite("optimized_receipt.jpg", optimized_image)
