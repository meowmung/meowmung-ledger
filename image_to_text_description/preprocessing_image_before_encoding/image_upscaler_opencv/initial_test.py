import os
import cv2
from cv2 import dnn_superres

img_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "images", "test_pet_3.jpg"
))

target_image = cv2.imread(img_path)

if target_image is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

sr = dnn_superres.DnnSuperResImpl_create()

model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "models", "EDSR_x3.pb"
))

sr.readModel(model_path)
sr.setModel("edsr", 3)

upscaled = sr.upsample(target_image)

result_path = os.path.join(
    os.path.dirname(__file__),
    "업스케일링_결과.png"
)

cv2.imwrite(result_path, upscaled)
