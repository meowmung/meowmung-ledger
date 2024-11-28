# image_upscaler_opencv/image_upscaler.py

import os
import cv2
from cv2 import dnn_superres

class ImageUpscaler:
    """
    OpenCV의 DNN 모듈과 사전 학습된 EDSR 모델을 사용하여 이미지를 업스케일링하는 클래스.
    
    속성:
    - model_path (str): 사전 학습된 모델 파일의 경로 (예: 'EDSR_x3.pb').
    - scale_factor (int): 업스케일링 배율 (예: 3은 3배 확대).
    - backend (str): 이미지 처리에 사용할 백엔드 ('cpu', 'cuda', 'mps' 중 선택).
    """
    
    def __init__(self, model_path, scale_factor=3, backend='cpu'):
        """
        지정된 모델, 배율, 처리 백엔드로 ImageUpscaler를 초기화합니다.
        
        매개변수:
        - model_path (str): EDSR 모델 파일의 경로.
        - scale_factor (int): 업스케일링 배율 (예: 3은 3배 확대).
        - backend (str): 처리 백엔드 ('cpu', 'cuda', 'mps' 중 선택).
        """
        self.model_path = model_path
        self.scale_factor = scale_factor

        # DNN 초해상도 모델 초기화
        self.sr = dnn_superres.DnnSuperResImpl_create()
        self.sr.readModel(model_path)
        self.sr.setModel('edsr', scale_factor)

        # 백엔드 설정
        if backend == 'cuda':
            self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif backend == 'mps':
            self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_MPS)
        else:
            self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def upscale_image(self, input_image_path, output_image_path):
        """
        단일 이미지를 업스케일링하여 지정된 경로에 저장합니다.
        
        매개변수:
        - input_image_path (str): 원본 이미지 경로.
        - output_image_path (str): 업스케일링된 이미지를 저장할 경로.
        """
        target_image = cv2.imread(input_image_path)
        if target_image is None:
            print(f"이미지 로드 오류: {input_image_path}")
            return

        upscaled_image = self.sr.upsample(target_image)
        cv2.imwrite(output_image_path, upscaled_image)
        print(f"업스케일링된 이미지를 저장했습니다: {output_image_path}")

    def upscale_images_in_folder(self, input_folder, output_folder):
        """
        지정된 입력 폴더의 모든 이미지를 업스케일링하여 출력 폴더에 저장합니다.
        
        매개변수:
        - input_folder (str): 업스케일링할 이미지가 있는 폴더 경로.
        - output_folder (str): 업스케일링된 이미지를 저장할 폴더 경로.
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in os.listdir(input_folder):
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                input_image_path = os.path.join(input_folder, file_name)
                output_image_path = os.path.join(output_folder, f'Upscaled_{file_name}')
                self.upscale_image(input_image_path, output_image_path)
