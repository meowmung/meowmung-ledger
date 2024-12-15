# application_connect_RESTful_API.py

import os
import logging
import uuid
import json
import yaml
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from image_to_text_description.base64_multimodal import MultiModal
import uvicorn

import asyncio  # 비동기 처리를 위한 모듈
import requests
from requests_aws4auth import AWS4Auth

app = FastAPI(root_path="/ml")


# 로깅 설정 초기화 (모듈 초기화 시점에 한 번만 설정)
def setup_logging() -> logging.Logger:
    """
    애플리케이션을 위한 로깅을 설정합니다.

    이 함수는 애플리케이션 내의 이벤트와 오류를 추적하기 위해 로깅 구성을 초기화합니다.
    로깅 레벨을 INFO로 설정하고 현재 모듈에 대한 로거 객체를 반환합니다.

    Returns:
        logging.Logger: 설정된 로거 객체.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


logger = setup_logging()


def load_environment(logger: logging.Logger) -> Dict[str, str]:
    """
    .env 파일에서 환경 변수를 로드합니다.

    이 함수는 dotenv 패키지를 사용하여 환경 변수를 로드하고,
    OpenAI와 AWS 서비스에 필요한 자격 증명 및 구성 값을 가져옵니다.
    필요한 환경 변수가 누락된 경우, 오류를 로깅하고 애플리케이션을 종료합니다.

    Args:
        logger (logging.Logger): 오류를 로깅하는 데 사용되는 로거 객체.

    Returns:
        Dict[str, str]: 로드된 환경 변수들을 포함하는 사전.

    Raises:
        SystemExit: 필요한 환경 변수가 누락된 경우.
    """
    load_dotenv()

    required_vars = [
        "OPENAI_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_SERVICE",
    ]

    env_vars = {}
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value

    if missing_vars:
        logger.error(f".env 파일에서 누락된 환경 변수: {', '.join(missing_vars)}")
        exit(1)

    return env_vars


def load_prompt_config(config_path: str, logger: logging.Logger) -> dict:
    """
    YAML 파일에서 프롬프트 구성을 로드합니다.

    이 함수는 프롬프트 구성을 포함하는 YAML 파일을 읽고,
    이를 사전 형태로 반환합니다. 파일이 존재하지 않거나 YAML 파싱에 실패할 경우,
    오류를 로깅하고 애플리케이션을 종료합니다.

    Args:
        config_path (str): YAML 구성 파일의 경로.
        logger (logging.Logger): 오류를 로깅하는 데 사용되는 로거 객체.

    Returns:
        dict: 파싱된 프롬프트 구성.

    Raises:
        SystemExit: 파일이 존재하지 않거나 YAML 파싱에 실패한 경우.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            prompt_config = yaml.safe_load(file)
        logger.info("프롬프트 구성이 성공적으로 로드되었습니다.")
        return prompt_config
    except FileNotFoundError:
        logger.error(f"YAML 파일을 찾을 수 없습니다: {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"YAML 파싱 오류: {e}")
        exit(1)


def initialize_llm() -> ChatOpenAI:
    """
    언어 모델(LLM) 객체를 초기화합니다.

    이 함수는 ChatOpenAI 클래스를 사용하여 LLM 객체를 생성하고,
    이를 반환합니다. 생성된 LLM 객체는 이미지로부터 텍스트를 추출하는 데 사용됩니다.

    Returns:
        ChatOpenAI: 초기화된 LLM 객체.
    """
    return ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o",
    )


async def process_image(
    image_data: bytes, multimodal_llm_with_prompt: MultiModal
) -> dict:
    """
    업로드된 이미지를 처리하여 구조화된 데이터를 추출합니다.

    이 비동기 함수는 업로드된 이미지 데이터를 임시 파일로 저장한 후,
    MultiModal 객체를 사용하여 이미지를 처리하고, 결과로 얻은 JSON 데이터를 파싱하여 반환합니다.
    처리 중 예외가 발생하면 적절히 처리하고 임시 파일을 삭제합니다.

    Args:
        image_data (bytes): 업로드된 이미지의 바이트 데이터.
        multimodal_llm_with_prompt (MultiModal): 이미지 처리를 위한 MultiModal 객체.

    Returns:
        dict: 추출된 JSON 데이터.

    Raises:
        HTTPException: 이미지 처리 중 오류가 발생한 경우.
    """
    temp_file_path = None
    try:
        # 임시 파일 생성 및 이미지 데이터 저장
        temp_file_path = f"temp_{uuid.uuid4().hex}.jpeg"
        with open(temp_file_path, "wb") as f:
            f.write(image_data)

        logger.info(f"임시 파일 저장 경로: {temp_file_path}")

        # MultiModal LLM을 사용하여 이미지 처리 (비동기적으로 실행)
        answer = await asyncio.to_thread(
            multimodal_llm_with_prompt.invoke, temp_file_path
        )

        # JSON 블록 제거
        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[len("```json") : -len("```")].strip()

        # JSON 문자열을 파이썬 객체로 변환
        answer_json = json.loads(answer)
        return answer_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def combine_json_outputs(json_outputs: List[dict], logger: logging.Logger) -> dict:
    """
    여러 JSON 출력을 하나의 JSON으로 병합합니다.

    이 함수는 여러 이미지에서 추출된 JSON 데이터를 하나의 통합된 JSON 객체로 합칩니다.
    날짜와 위치 정보는 처음 등장하는 값을 사용하며, 아이템 목록은 모두 합칩니다.
    총액은 마지막으로 등장한 값을 사용합니다.

    Args:
        json_outputs (List[dict]): 각 이미지에서 추출된 JSON 데이터의 리스트.
        logger (logging.Logger): 정보를 로깅하는 데 사용되는 로거 객체.

    Returns:
        dict: 병합된 JSON 데이터.
    """
    combined_data = {"date": None, "location": None, "items": [], "total_amount": None}
    for data in json_outputs:
        if not combined_data["date"] and data.get("date"):
            combined_data["date"] = data["date"]
        if not combined_data["location"] and data.get("location"):
            combined_data["location"] = data["location"]
        combined_data["items"].extend(data.get("items", []))
        if data.get("total_amount"):
            combined_data["total_amount"] = data["total_amount"]
    return combined_data


def get_file_extension(url: str) -> str:
    """
    주어진 URL에서 파일 확장자를 추출합니다.

    Args:
        url (str): 파일이 위치한 URL.

    Returns:
        str: 소문자로 변환된 파일 확장자.
    """
    parsed = requests.utils.urlparse(url)
    root, ext = os.path.splitext(parsed.path)
    return ext.lower()


def get_extension_from_content_type(content_type: str) -> str:
    """
    Content-Type 헤더로부터 파일 확장자를 매핑합니다.

    Args:
        content_type (str): HTTP 응답의 Content-Type 헤더 값.

    Returns:
        str: 매핑된 파일 확장자. 알 수 없는 경우 기본값으로 .jpeg를 반환합니다.
    """
    mapping = {
        "image/jpeg": ".jpeg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    return mapping.get(content_type.lower(), ".jpeg")  # 기본값으로 .jpeg 사용


class ImageURLs(BaseModel):
    """
    이미지 URL 리스트를 담는 Pydantic 모델.

    Attributes:
        image_data (List[str]): 처리할 이미지의 URL 리스트.
    """

    image_data: List[str]


# 엔드포인트 정의
@app.post("/ledger_receipt")
async def ledger_receipt(urls: ImageURLs):
    """
    영수증 이미지를 처리하여 구조화된 JSON 데이터를 반환하는 엔드포인트입니다.

    이 엔드포인트는 사용자가 제공한 하나 이상의 영수증 이미지 URL을 받아서,
    각 이미지를 다운로드하고 처리한 후, 추출된 데이터를 병합하여 JSON 형식으로 반환합니다.

    Args:
        urls (ImageURLs): 이미지 URL 리스트를 포함하는 요청 본문.

    Returns:
        dict: 병합된 JSON 데이터.

    Raises:
        HTTPException: 이미지 다운로드 실패, 지원되지 않는 파일 형식,
                        또는 이미지 처리 중 오류가 발생한 경우.
    """
    logger.info(f"수신된 요청: {urls}")
    logger.info(f"이미지 데이터: {urls.image_data}")
    logger.info(f"타입: {type(urls.image_data)}")


    required_vars = [
            "OPENAI_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_REGION",
            "AWS_SERVICE",
        ]


    # 환경 변수 로드
    env_vars = load_environment(logger)
    openai_api_key = env_vars["OPENAI_API_KEY"]
    aws_access_key = env_vars["AWS_ACCESS_KEY_ID"]
    aws_secret_key = env_vars["AWS_SECRET_ACCESS_KEY"]
    region = env_vars["AWS_REGION"]
    service = env_vars["AWS_SERVICE"]

    os.environ["OPENAI_API_KEY"] = openai_api_key  # OpenAI API 키 설정

    # 프롬프트 구성 파일의 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_CONFIG_PATH = os.path.join(
        BASE_DIR, "image_to_text_description", "prompt_config.yaml"
    )
    # 프롬프트 구성 로드
    prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, logger)
    system_prompt = prompt_config["prompts"]["system_prompt"]
    user_prompt = prompt_config["prompts"]["user_prompt"]

    # LLM 객체 초기화
    llm = initialize_llm()
    # MultiModal 객체 초기화
    multimodal_llm_with_prompt = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    json_outputs = []

    # AWS 인증 설정
    awsauth = AWS4Auth(aws_access_key, aws_secret_key, region, service)

    for image_url in urls.image_data:
        logger.info(f"처리 중인 이미지 URL: {image_url}")
        response = requests.get(image_url, auth=awsauth)
        logger.info(f"이미지 형식: {type(response.content)}")
        logger.info(f"HTTP 응답 상태: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"이미지 다운로드 실패: {image_url}")
            raise HTTPException(
                status_code=400, detail=f"이미지 다운로드 실패: {image_url}"
            )

        ext = get_file_extension(image_url)

        if ext not in [".jpeg", ".jpg", ".png", ".gif", ".webp"]:
            logger.error(f"지원되지 않는 파일 형식: {ext}")
            raise HTTPException(
                status_code=400, detail=f"지원되지 않는 파일 형식: {ext}"
            )

        json_output = await process_image(response.content, multimodal_llm_with_prompt)
        json_outputs.append(json_output)

    combined_json = combine_json_outputs(json_outputs, logger)
    logger.info(f"결합된 JSON: {combined_json}")
    return combined_json


# 애플리케이션 실행 (직접 실행 시)
if __name__ == "__main__":
    uvicorn.run(
        "application_connect_RESTful_API:app", reload=True, host="0.0.0.0", port=8085
    )


# application_connect_RESTful_API.py
# 처음에 서버와 통신 시 이미지를 직접 받을 때 사용한 코드

# import os
# import logging
# import uuid
# import json
# import yaml
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from typing import List
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from image_to_text_description.base64_multimodal import MultiModal
# import uvicorn

# app = FastAPI()


# def setup_logging():
#     """
#     로깅을 설정하는 함수입니다.

#     이 함수는 애플리케이션에서 발생하는 이벤트나 오류를 추적하기 위해
#     로깅 설정을 초기화합니다. 기본적으로 INFO 레벨로 설정되며,
#     현재 모듈(__name__)에 대한 로거 객체를 반환합니다.

#     Returns:
#         logging.Logger: 설정된 로거 객체를 반환합니다.
#     """
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     return logger


# def load_environment(logger):
#     """
#     환경 변수를 로드하고, OpenAI API 키를 가져오는 함수입니다.

#     이 함수는 .env 파일에서 환경 변수를 로드하고, 그 중에서
#     OPENAI_API_KEY를 가져옵니다. 만약 API 키가 설정되어 있지 않다면,
#     오류를 로깅하고 프로그램을 종료합니다.

#     Args:
#         logger (logging.Logger): 로깅에 사용될 로거 객체입니다.

#     Returns:
#         str: 로드된 OpenAI API 키를 반환합니다.

#     Raises:
#         SystemExit: OPENAI_API_KEY가 설정되지 않은 경우 프로그램을 종료합니다.
#     """
#     load_dotenv()
#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     if not openai_api_key:
#         logger.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
#         exit(1)
#     return openai_api_key


# def load_prompt_config(config_path, logger):
#     """
#     프롬프트 구성을 담은 YAML 파일을 로드하는 함수입니다.

#     이 함수는 주어진 경로의 YAML 파일을 읽어 프롬프트 구성을
#     딕셔너리 형태로 반환합니다. 파일이 존재하지 않거나 YAML 파싱에
#     실패하면 오류를 로깅하고 프로그램을 종료합니다.

#     Args:
#         config_path (str): 프롬프트 구성 파일의 경로입니다.
#         logger (logging.Logger): 로깅에 사용될 로거 객체입니다.

#     Returns:
#         dict: 로드된 프롬프트 구성 딕셔너리를 반환합니다.

#     Raises:
#         SystemExit: 파일이 없거나 파싱에 실패한 경우 프로그램을 종료합니다.
#     """
#     try:
#         with open(config_path, "r", encoding="utf-8") as file:
#             prompt_config = yaml.safe_load(file)
#         logger.info("프롬프트 구성 로드 성공.")
#         return prompt_config
#     except FileNotFoundError:
#         logger.error(f"YAML 파일을 찾을 수 없습니다: {config_path}")
#         exit(1)
#     except yaml.YAMLError as e:
#         logger.error(f"YAML 파싱 오류: {e}")
#         exit(1)


# def initialize_llm():
#     """
#     LLM(Language Model) 객체를 초기화하는 함수입니다.

#     이 함수는 ChatOpenAI 클래스를 사용하여 LLM 객체를 생성하고
#     반환합니다. 생성된 LLM 객체는 이미지로부터 텍스트를 추출하는데
#     사용됩니다.

#     Returns:
#         ChatOpenAI: 초기화된 LLM 객체를 반환합니다.
#     """
#     return ChatOpenAI(
#         temperature=0.2,
#         model_name="gpt-4o",
#     )


# async def process_image(image_file: UploadFile, multimodal_llm_with_prompt):
#     """
#     업로드된 이미지 파일을 처리하여 구조화된 데이터를 추출하는 함수입니다.

#     이 함수는 업로드된 이미지 파일을 고유한 이름의 임시 파일로 저장한 후,
#     MultiModal 객체를 사용하여 이미지를 처리합니다. 처리 결과로 얻은 텍스트를
#     JSON 형식으로 파싱하여 반환합니다. 처리 중 예외가 발생할 경우, 적절히 처리하고
#     임시 파일을 삭제합니다.

#     Args:
#         image_file (UploadFile): 업로드된 이미지 파일 객체입니다.
#         multimodal_llm_with_prompt (MultiModal): 이미지 처리를 위한 MultiModal 객체입니다.

#     Returns:
#         dict: 추출된 JSON 데이터를 반환합니다.

#     Raises:
#         HTTPException: 이미지 처리 중 오류가 발생한 경우 예외를 발생시킵니다.
#     """
#     temp_file_path = None
#     try:
#         contents = await image_file.read()
#         temp_file_path = f"temp_{uuid.uuid4().hex}_{image_file.filename}"

#         with open(temp_file_path, "wb") as f:
#             f.write(contents)

#         answer = multimodal_llm_with_prompt.invoke(temp_file_path)
#         if answer.startswith("```json") and answer.endswith("```"):
#             answer = answer[len("```json") : -len("```")].strip()

#         answer_json = json.loads(answer)
#         return answer_json

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             os.remove(temp_file_path)


# def combine_json_outputs(json_outputs: List[dict], logger):
#     """
#     여러 개의 JSON 출력을 하나의 JSON으로 병합하는 함수입니다.

#     이 함수는 각 이미지로부터 추출된 JSON 데이터를 받아서,
#     하나의 통합된 JSON 데이터로 병합합니다.
#     날짜와 위치 정보는 첫 번째로 등장하는 값을 사용하며,
#     아이템 목록은 모두 합쳐집니다.
#     총액은 마지막으로 등장한 값을 사용합니다.

#     Args:
#         json_outputs (List[dict]): 각 이미지로부터 추출된 JSON 데이터의 리스트입니다.
#         logger (logging.Logger): 로깅에 사용될 로거 객체입니다.

#     Returns:
#         dict: 병합된 JSON 데이터를 반환합니다.
#     """
#     combined_data = {"date": None, "location": None, "items": [], "total_amount": None}
#     for data in json_outputs:
#         if not combined_data["date"] and data.get("date"):
#             combined_data["date"] = data["date"]
#         if not combined_data["location"] and data.get("location"):
#             combined_data["location"] = data["location"]
#         combined_data["items"].extend(data.get("items", []))
#         if data.get("total_amount"):
#             combined_data["total_amount"] = data["total_amount"]
#     return combined_data


# # 엔드포인트 정의
# @app.post("/ledger_receipt")
# async def ledger_receipt(files: List[UploadFile] = File(...)):
#     """
#     영수증 이미지를 처리하여 구조화된 JSON 데이터를 반환하는 엔드포인트입니다.

#     이 엔드포인트는 사용자가 업로드한 하나 이상의 영수증 이미지 파일을 받아서,
#     각 이미지를 처리하고 추출된 데이터를 병합하여 JSON 형식으로 반환합니다.

#     Args:
#         files (List[UploadFile]): 업로드된 이미지 파일들의 리스트입니다.

#     Returns:
#         dict: 병합된 JSON 데이터를 반환합니다.

#     Raises:
#         HTTPException: 처리 과정에서 오류가 발생한 경우 예외를 발생시킵니다.
#     """

#     import math
#     import time

#     start = time.time()
#     math.factorial(100000)

#     logger = setup_logging()
#     # 환경 변수에서 OpenAI API 키를 로드
#     openai_api_key = load_environment(logger)
#     os.environ["OPENAI_API_KEY"] = openai_api_key  # OpenAI API 키 설정

#     # 프롬프트 구성 파일의 경로를 설정
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROMPT_CONFIG_PATH = os.path.join(
#         BASE_DIR, "image_to_text_description", "prompt_config.yaml"
#     )
#     # 프롬프트 구성을 로드
#     prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, logger)
#     system_prompt = prompt_config["prompts"]["system_prompt"]
#     user_prompt = prompt_config["prompts"]["user_prompt"]

#     # LLM 객체를 초기화
#     llm = initialize_llm()
#     # MultiModal 객체를 초기화
#     multimodal_llm_with_prompt = MultiModal(
#         llm, system_prompt=system_prompt, user_prompt=user_prompt
#     )

#     json_outputs = []
#     # 각 이미지 파일을 순회하며 처리
#     for image_file in files:
#         # 이미지 파일을 처리하여 JSON 데이터를 추출
#         json_output = await process_image(image_file, multimodal_llm_with_prompt)
#         json_outputs.append(json_output)

#     # 추출된 JSON 데이터를 병합
#     combined_json = combine_json_outputs(json_outputs, logger)

#     # 결과 출력
#     print(combined_json)
#     print(type(combined_json))
#     end = time.time()

#     print()
#     print()
#     print()
#     print("RESTful API 시간 측정")
#     print(f"{end - start:.5f} sec")
#     print()
#     print()
#     print()

#     # 병합된 JSON 데이터를 반환
#     return combined_json


# # 애플리케이션 실행 (직접 실행 시)
# if __name__ == "__main__":
#     uvicorn.run(
#         "application_connect_RESTful_API:app", reload=True, host="0.0.0.0", port=8085
#     )
