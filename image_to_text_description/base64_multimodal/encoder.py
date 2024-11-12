# base64_multimodal/encoder.py

"""
ImageEncoder 클래스 모듈.

이 모듈은 이미지 파일 또는 URL을 base64로 인코딩하는 기능을 제공합니다.
"""

import base64
import requests
import os


class ImageEncoder:
    """
    이미지 파일 또는 URL을 base64로 인코딩하는 클래스.

    정적 메서드를 통해 이미지 경로를 받아 base64 인코딩된 문자열을 반환합니다.
    """

    @staticmethod
    def encode_image(image_path):
        """
        주어진 이미지 경로(URL 또는 파일 경로)에 따라 적절한 인코딩 메서드를 호출합니다.

        Args:
            image_path (str): 인코딩할 이미지의 URL 또는 파일 경로.

        Returns:
            str: base64로 인코딩된 이미지 데이터 URI.

        Raises:
            Exception: 이미지 다운로드에 실패한 경우.
        """
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return ImageEncoder.encode_image_from_url(image_path)
        else:
            return ImageEncoder.encode_image_from_file(image_path)

    @staticmethod
    def encode_image_from_url(url):
        """
        URL에서 이미지를 다운로드하고 base64로 인코딩합니다.

        Args:
            url (str): 이미지의 URL.

        Returns:
            str: base64로 인코딩된 이미지 데이터 URI.

        Raises:
            Exception: 이미지 다운로드에 실패한 경우.
        """
        response = requests.get(url)
        if response.status_code == 200:
            image_content = response.content
            if url.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            elif url.lower().endswith(".png"):
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
        else:
            raise Exception("이미지 다운로드 실패")

    @staticmethod
    def encode_image_from_file(file_path):
        """
        파일 시스템에서 이미지를 읽고 base64로 인코딩합니다.

        Args:
            file_path (str): 이미지 파일의 경로.

        Returns:
            str: base64로 인코딩된 이미지 데이터 URI.
        """
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in [".jpg", ".jpeg"]:
                mime_type = "image/jpeg"
            elif file_ext == ".png":
                mime_type = "image/png"
            else:
                mime_type = "image/unknown"
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
