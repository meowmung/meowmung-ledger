# base64_multimodal/multimodal.py

"""
MultiModal 클래스 모듈.

이 모듈은 인코딩된 이미지를 바탕으로 OpenAI 모델과 상호작용하여 이미지 설명을 생성하는 기능을 제공합니다.
"""

from .encoder import ImageEncoder
from dotenv import load_dotenv


class MultiModal:
    """
    멀티모달 모델 인터페이스 클래스.

    이미지 데이터를 인코딩하고, 모델에 메시지를 전달하여 이미지 설명을 생성합니다.
    """

    def __init__(self, model, system_prompt=None, user_prompt=None):
        """
        MultiModal 클래스 초기화.

        Args:
            model: 상호작용할 LLM 모델 객체.
            system_prompt (str, optional): 시스템 프롬프트. 기본값은 사전에 정의된 프롬프트.
            user_prompt (str, optional): 사용자 프롬프트. 기본값은 사전에 정의된 프롬프트.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        """
        시스템 및 사용자 프롬프트를 초기화합니다.
        기본 프롬프트가 제공되지 않은 경우, 사전에 정의된 기본값을 사용합니다.
        """
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant who helps users to write a report related to images in Korean."
        if self.user_prompt is None:
            self.user_prompt = "Explain the images as an alternative text in Korean."

    def create_messages(self, image_url, system_prompt=None, user_prompt=None):
        """
        모델에 전달할 메시지 구조를 생성합니다.

        Args:
            image_url (str): 인코딩된 이미지의 URL 또는 파일 경로.
            system_prompt (str, optional): 시스템 프롬프트. 기본값은 초기화된 시스템 프롬프트.
            user_prompt (str, optional): 사용자 프롬프트. 기본값은 초기화된 사용자 프롬프트.

        Returns:
            list: 모델에 전달할 메시지 리스트.
        """
        encoded_image = ImageEncoder.encode_image(image_url)

        system_prompt = (
            system_prompt if system_prompt is not None else self.system_prompt
        )

        user_prompt = user_prompt if user_prompt is not None else self.user_prompt

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"{encoded_image}"},
                    },
                ],
            },
        ]
        return messages

    def invoke(self, image_url, system_prompt=None, user_prompt=None):
        """
        단일 이미지에 대해 모델을 호출하여 설명을 생성합니다.

        Args:
            image_url (str): 인코딩된 이미지의 URL 또는 파일 경로.
            system_prompt (str, optional): 시스템 프롬프트.
            user_prompt (str, optional): 사용자 프롬프트.

        Returns:
            str: 모델의 응답 내용.
        """
        messages = self.create_messages(image_url, system_prompt, user_prompt)
        response = self.model.invoke(messages)
        return response.content

    def batch(
        self,
        image_urls: list[str],
        system_prompts: list[str] = [],
        user_prompts: list[str] = [],
    ):
        """
        여러 이미지에 대해 모델을 일괄 호출하여 설명을 생성합니다.

        Args:
            image_urls (list[str]): 인코딩된 이미지들의 URL 또는 파일 경로 리스트.
            system_prompts (list[str], optional): 각 이미지에 대한 시스템 프롬프트 리스트.
            user_prompts (list[str], optional): 각 이미지에 대한 사용자 프롬프트 리스트.

        Returns:
            list[str]: 각 모델 응답의 내용 리스트.
        """
        messages = []
        for image_url, system_prompt, user_prompt in zip(
            image_urls, system_prompts, user_prompts
        ):
            message = self.create_messages(image_url, system_prompt, user_prompt)
            messages.append(message)
        response = self.model.batch(messages)
        return [r.content for r in response]

    def stream(self, image_url, system_prompt=None, user_prompt=None):
        """
        단일 이미지에 대해 모델을 스트리밍 방식으로 호출하여 설명을 생성합니다.

        Args:
            image_url (str): 인코딩된 이미지의 URL 또는 파일 경로.
            system_prompt (str, optional): 시스템 프롬프트.
            user_prompt (str, optional): 사용자 프롬프트.

        Returns:
            스트리밍 응답 객체.
        """
        messages = self.create_messages(image_url, system_prompt, user_prompt)
        response = self.model.stream(messages)
        return response
