# test_application_core_feature.py

import os
import json
import pytest
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from image_to_text_description.base64_multimodal import MultiModal


def load_environment():
    """
    환경 변수를 로드하고 OpenAI API 키를 반환합니다.
    """
    load_dotenv()  # .env 파일 로드
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
    return openai_api_key


def load_prompt_config(config_path):
    """
    YAML 파일에서 프롬프트 구성을 로드합니다.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML 파일을 찾을 수 없습니다: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML 파일 구문 오류: {e}")


def initialize_llm():
    """
    LLM 객체를 초기화합니다.
    """
    return ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o",
    )


@pytest.fixture
def prompt_config_path():
    """프롬프트 구성 파일 경로를 반환합니다."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "../image_to_text_description/prompt_config.yaml")


@pytest.fixture
def image_path():
    """테스트 이미지 경로를 반환합니다."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "image_to_text_description/images/get_pet_6.jpg")


def test_application_core_features(image_path):
    """
    initial_application.py의 주요 기능을 테스트합니다.
    """
    # 환경 변수 로드
    try:
        openai_api_key = load_environment()
        assert openai_api_key is not None
    except ValueError as e:
        pytest.fail(f"환경 변수 오류: {e}")

    # 프롬프트 구성 파일 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_CONFIG_PATH = os.path.join(BASE_DIR, "image_to_text_description", "prompt_config.yaml")

    # 프롬프트 구성 로드
    try:
        prompt_config = load_prompt_config(PROMPT_CONFIG_PATH)
        system_prompt = prompt_config["prompts"]["system_prompt"]
        user_prompt = prompt_config["prompts"]["user_prompt"]
    except (FileNotFoundError, ValueError) as e:
        pytest.fail(f"프롬프트 구성 로드 오류: {e}")

    # LLM 초기화
    llm = initialize_llm()
    assert llm is not None

    # MultiModal 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 실제 이미지 파일을 사용하여 invoke 호출
    try:
        result = multimodal.invoke(image_path)
        assert isinstance(result, str)
        result_json = json.loads(result)
        assert isinstance(result_json, dict)

        # 테스트 출력
        print(json.dumps(result_json, ensure_ascii=False, indent=4))
    except Exception as e:
        pytest.fail(f"이미지 처리 중 오류 발생: {e}")

