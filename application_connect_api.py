"""
FastAPI 애플리케이션 스크립트.

이 스크립트는 "/ledger_receipt" 엔드포인트를 제공하며, 사용자가 업로드한 영수증 이미지를 처리하여
구조화된 JSON 데이터를 생성하고, 이를 데이터베이스에 저장한 후 사용자에게 반환합니다.
"""

import os
import logging
import json
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from image_to_text_description.base64_multimodal import MultiModal
import uvicorn
from pydantic import BaseModel
from tortoise import Tortoise, fields
from tortoise.models import Model


# 데이터베이스 모델 정의
# (Tortoise ORM 사용: 데이터베이스와의 상호 작용을 위해, 즉 데이터베이스 테이블 구조를 정의하고, 데이터를 저장하거나 조회할 때 사용)
class ReceiptItem(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)
    price = fields.IntField()
    category = fields.CharField(max_length=50)
    receipt = fields.ForeignKeyField("models.Receipt", related_name="items")


class Receipt(Model):
    id = fields.IntField(pk=True)
    date = fields.CharField(max_length=20)
    location = fields.CharField(max_length=255)
    total_amount = fields.IntField()
    items: fields.ReverseRelation["ReceiptItem"]


# Pydantic 모델 정의
# (요청 및 응답 검증용: 들어오는 데이터의 유효성을 검사하고, 응답 데이터를 일정한 형식으로 반환하기 위해 Pydantic 사용)
class Item(BaseModel):
    name: str
    price: int
    category: str


class ReceiptData(BaseModel):
    date: str
    location: str
    items: List[Item]
    total_amount: int


# FastAPI 앱 초기화
app = FastAPI()


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


async def process_image(image_file: UploadFile, multimodal_llm_with_prompt):
    """
    단일 이미지 파일을 처리하여 구조화된 데이터를 추출합니다.

    Args:
        image_file (UploadFile): 사용자가 업로드한 이미지 파일.
        multimodal_llm_with_prompt (MultiModal): 이미지 처리를 위한 MultiModal 객체.

    Returns:
        dict: 추출된 JSON 데이터.

    Raises:
        HTTPException: 처리 실패 시 예외 발생.
    """
    try:
        # 이미지 파일 읽기
        contents = await image_file.read()
        # 임시 파일로 저장
        temp_file_path = f"temp_{image_file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        # 이미지 처리
        answer = multimodal_llm_with_prompt.invoke(temp_file_path)
        # 임시 파일 삭제
        os.remove(temp_file_path)
        # 응답을 JSON으로 파싱
        # 코드 블록 제거
        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[len("```json") : -len("```")].strip()
        answer_json = json.loads(answer)
        return answer_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")


def combine_json_outputs(json_outputs: List[dict], logger):
    """
    여러 개의 JSON 출력을 하나의 JSON으로 병합합니다.

    Args:
        json_outputs (List[dict]): 각 이미지로부터 생성된 JSON 출력 리스트.

    Returns:
        dict: 병합된 JSON 출력.
    """
    combined_data = {"date": None, "location": None, "items": [], "total_amount": None}

    for data in json_outputs:
        # date와 location이 설정되지 않았다면 업데이트
        if not combined_data["date"] and data.get("date"):
            combined_data["date"] = data["date"]
        if not combined_data["location"] and data.get("location"):
            combined_data["location"] = data["location"]
        # items 추가
        combined_data["items"].extend(data.get("items", []))
        # total_amount는 마지막 이미지의 값을 사용
        if data.get("total_amount"):
            combined_data["total_amount"] = data["total_amount"]

    return combined_data


# 엔드포인트 정의
@app.post("/ledger_receipt")
async def ledger_receipt(files: List[UploadFile] = File(...)):
    """
    영수증 이미지를 처리하여 JSON 데이터를 반환하는 엔드포인트.

    Args:
        files (List[UploadFile]): 업로드된 이미지 파일 목록.

    Returns:
        dict: 병합된 JSON 데이터.

    Raises:
        HTTPException: 처리 실패 시 예외 발생.
    """

    logger = setup_logging()
    # 환경 변수 로드
    openai_api_key = load_environment(logger)
    os.environ["OPENAI_API_KEY"] = openai_api_key  # OpenAI API 키 설정

    # 프롬프트 구성 로드
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_CONFIG_PATH = os.path.join(
        BASE_DIR, "image_to_text_description", "prompt_config.yaml"
    )  ################################추후 상황에 맞게 다시 경로 설정################################
    prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, logger)

    system_prompt = prompt_config["prompts"]["system_prompt"]
    user_prompt = prompt_config["prompts"]["user_prompt"]

    # LLM 초기화
    llm = initialize_llm()

    # MultiModal 객체 초기화
    multimodal_llm_with_prompt = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    json_outputs = []

    for image_file in files:
        # 각 이미지를 처리
        json_output = await process_image(image_file, multimodal_llm_with_prompt)
        json_outputs.append(json_output)

    # JSON 출력 병합
    combined_json = combine_json_outputs(json_outputs, logger)

    print(combined_json)
    print(type(combined_json))

    return combined_json


# 애플리케이션 실행 (직접 실행 시)
if __name__ == "__main__":
    uvicorn.run("application_connect_api:app", host="0.0.0.0", port=8000)
