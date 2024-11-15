# image_upscaler_opencv/initial_test.py

import cv2
from cv2 import dnn_superres


img_path = "/Users/khb43/Desktop/GIT_SHOWVINI/meowmung-ledger/image_to_text_description/images/test_pet_3.jpg"

target_image = cv2.imread(img_path)

sr = dnn_superres.DnnSuperResImpl_create()

model_path = "/Users/khb43/Desktop/GIT_SHOWVINI/meowmung-ledger/image_to_text_description/upscaling_opencv/EDSR_x3.pb"
sr.readModel(model_path)
sr.setModel("edsr", 3)

upscaled = sr.upsample(target_image)

cv2.imwrite("./업스케일링_결과.png", upscaled)
