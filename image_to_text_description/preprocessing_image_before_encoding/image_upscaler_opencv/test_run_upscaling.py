# image_upscaler_opencv/test_run_upscaling.py

import os
from .image_upscaler import ImageUpscaler

def main():
    """
    ImageUpscaler 클래스를 사용하여 단일 이미지를 업스케일링하는 예제입니다.
    """

    # 절대 경로를 사용하여 모델 경로와 이미지 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 절대 경로
    model_path = os.path.join(base_dir, '../image_upscaler_opencv/models/EDSR_x3.pb')  # 모델의 절대 경로
    input_image_path = os.path.join(base_dir, '../../images/test_pet_1.jpg')  # 업스케일링할 이미지 절대 경로
    output_image_path = os.path.join(base_dir, '../../images/output_image.png')  # 결과 이미지 저장 절대 경로

    # ImageUpscaler 객체 생성 (Mac 사용자의 경우 MPS 백엔드 사용 가능)
    upscaler = ImageUpscaler(model_path=model_path, scale_factor=3, backend='cpu')

    # 단일 이미지 업스케일링
    upscaler.upscale_image(input_image_path, output_image_path)

if __name__ == "__main__":
    main()
