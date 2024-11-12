# base64_multimodal/multimodal.py

from .encoder import ImageEncoder
from dotenv import load_dotenv


class MultiModal:
    def __init__(self, model, system_prompt=None, user_prompt=None):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.init_prompt()

    def init_prompt(self):
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant who helps users to write a report related to images in Korean."
        if self.user_prompt is None:
            self.user_prompt = "Explain the images as an alternative text in Korean."

    def create_messages(self, image_url, system_prompt=None, user_prompt=None):
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
        messages = self.create_messages(image_url, system_prompt, user_prompt)
        response = self.model.invoke(messages)
        return response.content

    def batch(
        self,
        image_urls: list[str],
        system_prompts: list[str] = [],
        user_prompts: list[str] = [],
    ):
        messages = []
        for image_url, system_prompt, user_prompt in zip(
            image_urls, system_prompts, user_prompts
        ):
            message = self.create_messages(image_url, system_prompt, user_prompt)
            messages.append(message)
        response = self.model.batch(messages)
        return [r.content for r in response]

    def stream(self, image_url, system_prompt=None, user_prompt=None):
        messages = self.create_messages(image_url, system_prompt, user_prompt)
        response = self.model.stream(messages)
        return response
