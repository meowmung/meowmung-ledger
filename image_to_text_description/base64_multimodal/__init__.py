# base64_multimodal/__init__.py

"""
base64_multimodal 패키지 초기화 모듈.

이 패키지는 이미지 파일 또는 URL을 base64로 인코딩하는 기능과
인코딩된 이미지를 바탕으로 모델과 상호작용하여 이미지 설명을 생성하는 기능을 제공합니다.
"""

from .multimodal import MultiModal
from .encoder import ImageEncoder
