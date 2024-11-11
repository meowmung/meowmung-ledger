import base64
import requests
from IPython.display import Image, display
import os


class MultiModal:
    """
    "다중 모달 클래스"는 텍스트, 이미지, 비디오 등 다양한 형태의 데이터를 함께 처리하거나 상호작용하는 기능을 제공하는 클래스를 의미합니다.
    이 클래스는 텍스트와 이미지 메시지를 모델에 전달하고 처리하기 위한 다중 모달 유틸리티 클래스입니다.
    URL 또는 파일 경로에서 이미지를 base64로 인코딩하고, 이미지를 표시하며, 이미지에 있는 텍스트 혹은 이미지 프롬프트를 함께 포함하는 메시지를 제공합니다. 
    또한 단일 메시지로 모델을 호출하거나, 여러 이미지 메시지를 일괄 처리하며, 스트림으로 모델 응답을 받을 수 있습니다.

    속성:
        model: 상호작용할 언어 모델.
        system_prompt (str): 시스템의 기본 행동을 설정하는 초기 프롬프트 (예: 한국어로 도움을 제공하는 조수).
        user_prompt (str): 사용자가 이미지에 대해 설명을 요청하는 프롬프트 (기본: 한국어 대체 텍스트).

    메서드:
        init_prompt(): 프롬프트가 제공되지 않은 경우 기본 프롬프트를 설정합니다.
        encode_image_from_url(url): URL에서 이미지를 base64 문자열로 인코딩합니다.
        encode_image_from_file(file_path): 로컬 파일에서 이미지를 base64 문자열로 인코딩합니다.
        encode_image(image_path): 이미지 경로가 URL인지 파일인지 확인 후 적절히 인코딩합니다.
        display_image(encoded_image): Jupyter에서 base64로 인코딩된 이미지를 표시합니다.
        create_messages(image_url, system_prompt, user_prompt, display_image): 텍스트 및 이미지 프롬프트가 포함된 메시지를 구성합니다.
        invoke(image_url, system_prompt, user_prompt, display_image): 단일 메시지로 모델을 호출합니다.
        batch(image_urls, system_prompts, user_prompts, display_image): 여러 이미지 메시지를 일괄 모드로 전송합니다.
        stream(image_url, system_prompt, user_prompt, display_image): 단일 이미지 메시지에 대해 모델 응답을 스트리밍합니다.
    """

    def __init__(self, model, system_prompt=None, user_prompt=None):
        """
        모델과 선택적인 시스템 및 사용자 프롬프트로 MultiModal 인스턴스를 초기화합니다.

        Args:
            model: 메시지 호출과 처리를 위한 모델 인스턴스.
            system_prompt (str): 한국어로 이미지를 설명하는 기본 시스템 프롬프트.
            user_prompt (str): 이미지를 설명하도록 모델에 요청하는 사용자 프롬프트 (기본 한국어).
        """
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        """
        프롬프트가 제공되지 않은 경우 기본 프롬프트를 설정합니다. 시스템 프롬프트는 조수의 행동을 안내하며,
        사용자 프롬프트는 이미지 설명에 대한 지시를 포함합니다.
        """
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant who helps users to write a report related to images in Korean."
        if self.user_prompt is None:
            self.user_prompt = "Explain the images as an alternative text in Korean."

    def encode_image_from_url(self, url):
        """
        URL에서 이미지를 base64 문자열로 인코딩하며 JPEG와 PNG 형식을 지원합니다.

        Args:
            url (str): 이미지의 URL.

        Returns:
            str: MIME 타입 접두사가 포함된 base64 인코딩된 이미지 데이터.

        Raises:
            Exception: URL에서 이미지를 다운로드할 수 없는 경우.
        """
        response = requests.get(url)
        if response.status_code == 200:
            image_content = response.content
            mime_type = (
                "image/jpeg" if url.lower().endswith((".jpg", ".jpeg")) else "image/png"
            )
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"
        else:
            raise Exception("Failed to download image")

    def encode_image_from_file(self, file_path):
        """
        로컬 파일에서 이미지를 base64 문자열로 인코딩하며 JPEG와 PNG 형식을 지원합니다.

        Args:
            file_path (str): 이미지 파일의 경로.

        Returns:
            str: MIME 타입 접두사가 포함된 base64 인코딩된 이미지 데이터.
        """
        with open(file_path, "rb") as image_file:
            image_content = image_file.read()
            mime_type = (
                "image/jpeg"
                if file_path.lower().endswith((".jpg", ".jpeg"))
                else "image/png"
            )
            return f"data:{mime_type};base64,{base64.b64encode(image_content).decode('utf-8')}"

    def encode_image(self, image_path):
        """
        이미지 경로가 URL인지 파일 경로인지 확인하고 적절히 인코딩하여 base64 문자열을 반환합니다.

        Args:
            image_path (str): 이미지 경로 또는 URL.

        Returns:
            str: MIME 타입 접두사가 포함된 base64 인코딩된 이미지 데이터.
        """
        return (
            self.encode_image_from_url(image_path)
            if image_path.startswith(("http://", "https://"))
            else self.encode_image_from_file(image_path)
        )

    def display_image(self, encoded_image):
        """
        Jupyter 환경에서 base64로 인코딩된 이미지를 표시합니다.

        Args:
            encoded_image (str): MIME 타입 접두사가 포함된 base64 인코딩된 이미지 데이터.
        """
        display(Image(url=encoded_image))

    def create_messages(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        """
        텍스트와 이미지 프롬프트를 포함한 구조화된 메시지를 생성하고, 필요시 이미지를 표시합니다.

        Args:
            image_url (str): 이미지의 경로 또는 URL.
            system_prompt (str): 사용자 지정 시스템 프롬프트 텍스트.
            user_prompt (str): 사용자 지정 사용자 프롬프트 텍스트.
            display_image (bool): True인 경우 노트북에서 이미지를 표시합니다.

        Returns:
            list: 시스템 및 사용자 역할을 포함한 텍스트 및 이미지 데이터를 포함한 구조화된 메시지.
        """
        encoded_image = self.encode_image(image_url)
        if display_image:
            self.display_image(encoded_image)

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt or self.user_prompt},
                    {"type": "image_url", "image_url": {"url": f"{encoded_image}"}},
                ],
            },
        ]
        return messages

    def invoke(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        """
        단일 이미지 메시지로 모델을 호출합니다.

        Args:
            image_url (str): 이미지의 경로 또는 URL.
            system_prompt (str): 사용자 지정 시스템 프롬프트 텍스트.
            user_prompt (str): 사용자 지정 사용자 프롬프트 텍스트.
            display_image (bool): True인 경우 노트북에서 이미지를 표시합니다.

        Returns:
            str: 모델의 응답 내용.
        """
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        response = self.model.invoke(messages)
        return response.content

    def batch(
        self, image_urls, system_prompts=[], user_prompts=[], display_image=False
    ):
        """
        여러 이미지 메시지를 모델에 일괄 모드로 전송합니다.

        Args:
            image_urls (list[str]): 이미지 경로 또는 URL 리스트.
            system_prompts (list[str]): 각 이미지에 대한 시스템 프롬프트 리스트.
            user_prompts (list[str]): 각 이미지에 대한 사용자 프롬프트 리스트.
            display_image (bool): True인 경우 각 이미지를 노트북에서 표시합니다.

        Returns:
            list[str]: 모델의 응답 내용 리스트.
        """
        messages = [
            self.create_messages(image_url, system_prompt, user_prompt, display_image)
            for image_url, system_prompt, user_prompt in zip(
                image_urls, system_prompts, user_prompts
            )
        ]
        response = self.model.batch(messages)
        return [r.content for r in response]

    def stream(
        self, image_url, system_prompt=None, user_prompt=None, display_image=True
    ):
        """
        단일 이미지 메시지에 대한 모델 응답을 스트리밍합니다.

        Args:
            image_url (str): 이미지의 경로 또는 URL.
            system_prompt (str): 사용자 지정 시스템 프롬프트 텍스트.
            user_prompt (str): 사용자 지정 사용자 프롬프트 텍스트.
            display_image (bool): True인 경우 노트북에서 이미지를 표시합니다.

        Returns:
            stream: 모델의 스트림 응답.
        """
        messages = self.create_messages(
            image_url, system_prompt, user_prompt, display_image
        )
        return self.model.stream(messages)
