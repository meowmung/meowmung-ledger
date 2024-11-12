# multimodal/encoder.py

import base64
import requests
import os


class ImageEncoder:

    @staticmethod
    def encode_image(image_path):
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return ImageEncoder.encode_image_from_url(image_path)
        else:
            return ImageEncoder.encode_image_from_file(image_path)

    @staticmethod
    def encode_image_from_url(url):
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
