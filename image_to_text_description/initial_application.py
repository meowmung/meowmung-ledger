# image_to_text_description/application.py

"""
메인 애플리케이션 스크립트.

이 스크립트는 이미지 파일을 입력으로 받아 MultiModal 클래스를 통해
이미지 설명을 생성하고, 이를 JSON 형식으로 출력합니다.
"""

import os
import json
import yaml
import logging
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from base64_multimodal import MultiModal


def setup_logging():
    """
    로깅 설정을 초기화합니다.

    Returns:
        logging.Logger: 설정된 로거 객체.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def load_environment(logger):
    """
    환경 변수를 로드하고 OpenAI API 키를 반환합니다.

    Args:
        logger (logging.Logger): 로깅 객체.

    Returns:
        str: OpenAI API 키.

    Raises:
        SystemExit: OPENAI_API_KEY가 설정되지 않은 경우.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        exit(1)
    return openai_api_key


def load_prompt_config(config_path, logger):
    """
    YAML 파일에서 프롬프트 구성을 로드합니다.

    Args:
        config_path (str): YAML 파일 경로.
        logger (logging.Logger): 로깅 객체.

    Returns:
        dict: 프롬프트 구성 딕셔너리.

    Raises:
        SystemExit: YAML 파일을 찾을 수 없거나 파싱에 실패한 경우.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            prompt_config = yaml.safe_load(file)
        logger.info("프롬프트 구성 로드 성공.")
        return prompt_config
    except FileNotFoundError:
        logger.error(f"YAML 파일을 찾을 수 없습니다: {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"YAML 파싱 오류: {e}")
        exit(1)


def initialize_llm():
    """
    LLM 객체를 초기화합니다.

    Returns:
        ChatOpenAI: 초기화된 LLM 객체.
    """
    return ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o",
    )


def main():
    """
    메인 함수.

    로깅을 설정하고, 환경 변수와 프롬프트 구성을 로드한 후,
    MultiModal 객체를 통해 이미지를 설명하고 결과를 출력합니다.

    Workflow:
    1. 로깅 설정
    2. 환경 변수 및 API 키 로드
    3. 프롬프트 구성 로드
    4. LLM 및 MultiModal 객체 생성
    5. 이미지 설명 생성 및 JSON 출력
    """
    # 로깅 설정
    logger = setup_logging()

    # 환경 변수 로드
    openai_api_key = load_environment(logger)
    os.environ["OPENAI_API_KEY"] = openai_api_key  # 필요 시 추가 설정

    # 프롬프트 구성 로드
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_CONFIG_PATH = os.path.join(BASE_DIR, "prompt_config.yaml")
    prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, logger)

    system_prompt = prompt_config["prompts"]["system_prompt"]
    user_prompt = prompt_config["prompts"]["user_prompt"]

    # LLM 객체 생성
    llm = initialize_llm()

    # MultiModal 객체 생성
    multimodal_llm_with_prompt = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    # 이미지 경로 설정
    IMAGE_PATH_FROM_FILE = os.path.join(BASE_DIR, "images", "get_pet_4.jpg")

    logger.info("이미지 설명 생성 시작.")

    # 이미지 설명 생성
    try:
        answer = multimodal_llm_with_prompt.invoke(IMAGE_PATH_FROM_FILE)
    except Exception as e:
        logger.error(f"이미지 설명 생성 중 오류 발생: {e}")
        exit(1)

    # 문자열을 JSON 객체으로 변환 및 출력
    try:
        # 응답에서 코드 블록 제거
        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[len("```json") : -len("```")].strip()
        answer_json = json.loads(answer)
        print(json.dumps(answer_json, ensure_ascii=False, indent=4))
        logger.info(f"응답 타입: {type(answer_json)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류: {e}")
        print("응답 내용:", answer)


if __name__ == "__main__":
    main()
