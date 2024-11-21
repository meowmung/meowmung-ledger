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

# FastAPI 애플리케이션 초기화
app = FastAPI()


def setup_logging():
    """
    로깅을 설정하는 함수입니다.

    이 함수는 애플리케이션에서 발생하는 이벤트나 오류를 추적하기 위해
    로깅 설정을 초기화합니다. 기본적으로 INFO 레벨로 설정되며,
    현재 모듈(__name__)에 대한 로거 객체를 반환합니다.

    Returns:
        logging.Logger: 설정된 로거 객체를 반환합니다.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def load_environment(logger):
    """
    환경 변수를 로드하고, OpenAI API 키를 가져오는 함수입니다.

    이 함수는 .env 파일에서 환경 변수를 로드하고, 그 중에서
    OPENAI_API_KEY를 가져옵니다. 만약 API 키가 설정되어 있지 않다면,
    오류를 로깅하고 프로그램을 종료합니다.

    Args:
        logger (logging.Logger): 로깅에 사용될 로거 객체입니다.

    Returns:
        str: 로드된 OpenAI API 키를 반환합니다.

    Raises:
        SystemExit: OPENAI_API_KEY가 설정되지 않은 경우 프로그램을 종료합니다.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        exit(1)
    return openai_api_key


def load_prompt_config(config_path, logger):
    """
    프롬프트 구성을 담은 YAML 파일을 로드하는 함수입니다.

    이 함수는 주어진 경로의 YAML 파일을 읽어 프롬프트 구성을
    딕셔너리 형태로 반환합니다. 파일이 존재하지 않거나 YAML 파싱에
    실패하면 오류를 로깅하고 프로그램을 종료합니다.

    Args:
        config_path (str): 프롬프트 구성 파일의 경로입니다.
        logger (logging.Logger): 로깅에 사용될 로거 객체입니다.

    Returns:
        dict: 로드된 프롬프트 구성 딕셔너리를 반환합니다.

    Raises:
        SystemExit: 파일이 없거나 파싱에 실패한 경우 프로그램을 종료합니다.
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
    LLM(Language Model) 객체를 초기화하는 함수입니다.

    이 함수는 ChatOpenAI 클래스를 사용하여 LLM 객체를 생성하고
    반환합니다. 생성된 LLM 객체는 이미지로부터 텍스트를 추출하는데
    사용됩니다.

    Returns:
        ChatOpenAI: 초기화된 LLM 객체를 반환합니다.
    """
    return ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o",
    )


async def process_image(image_file: UploadFile, multimodal_llm_with_prompt):
    """
    업로드된 이미지 파일을 처리하여 구조화된 데이터를 추출하는 함수입니다.

    이 함수는 업로드된 이미지 파일을 임시로 저장한 후,
    MultiModal 객체를 사용하여 이미지를 처리합니다.
    처리 결과로 얻은 텍스트를 JSON 형식으로 파싱하여 반환합니다.
    처리 과정에서 발생하는 오류는 HTTPException으로 처리됩니다.

    Args:
        image_file (UploadFile): 업로드된 이미지 파일 객체입니다.
        multimodal_llm_with_prompt (MultiModal): 이미지 처리를 위한 MultiModal 객체입니다.

    Returns:
        dict: 추출된 JSON 데이터를 반환합니다.

    Raises:
        HTTPException: 이미지 처리 중 오류가 발생한 경우 예외를 발생시킵니다.
    """
    try:
        # 이미지 파일의 내용을 비동기로 읽어옵니다.
        contents = await image_file.read()
        # 임시 파일 경로를 설정합니다.
        temp_file_path = f"temp_{image_file.filename}"
        # 이미지를 임시 파일로 저장합니다.
        with open(temp_file_path, "wb") as f:
            f.write(contents)
        # 이미지 파일을 처리하여 답변을 얻습니다.
        answer = multimodal_llm_with_prompt.invoke(temp_file_path)
        # 임시 파일을 삭제합니다.
        os.remove(temp_file_path)
        # 답변에서 코드 블록 표시를 제거합니다.
        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[len("```json") : -len("```")].strip()
        # 답변을 JSON 형식으로 파싱합니다.
        answer_json = json.loads(answer)
        return answer_json
    except Exception as e:
        # 오류가 발생하면 HTTPException을 발생시킵니다.
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류 발생: {e}")


def combine_json_outputs(json_outputs: List[dict], logger):
    """
    여러 개의 JSON 출력을 하나의 JSON으로 병합하는 함수입니다.

    이 함수는 각 이미지로부터 추출된 JSON 데이터를 받아서,
    하나의 통합된 JSON 데이터로 병합합니다.
    날짜와 위치 정보는 첫 번째로 등장하는 값을 사용하며,
    아이템 목록은 모두 합쳐집니다.
    총액은 마지막으로 등장한 값을 사용합니다.

    Args:
        json_outputs (List[dict]): 각 이미지로부터 추출된 JSON 데이터의 리스트입니다.
        logger (logging.Logger): 로깅에 사용될 로거 객체입니다.

    Returns:
        dict: 병합된 JSON 데이터를 반환합니다.
    """
    combined_data = {"date": None, "location": None, "items": [], "total_amount": None}
    for data in json_outputs:
        # 날짜와 위치 정보 설정
        if not combined_data["date"] and data.get("date"):
            combined_data["date"] = data["date"]
        if not combined_data["location"] and data.get("location"):
            combined_data["location"] = data["location"]
        # 아이템 목록 합치기
        combined_data["items"].extend(data.get("items", []))
        # 총액 업데이트
        if data.get("total_amount"):
            combined_data["total_amount"] = data["total_amount"]
    return combined_data


# 엔드포인트 정의
@app.post("/ledger_receipt")
async def ledger_receipt(files: List[UploadFile] = File(...)):
    """
    영수증 이미지를 처리하여 구조화된 JSON 데이터를 반환하는 엔드포인트입니다.

    이 엔드포인트는 사용자가 업로드한 하나 이상의 영수증 이미지 파일을 받아서,
    각 이미지를 처리하고 추출된 데이터를 병합하여 JSON 형식으로 반환합니다.

    Args:
        files (List[UploadFile]): 업로드된 이미지 파일들의 리스트입니다.

    Returns:
        dict: 병합된 JSON 데이터를 반환합니다.

    Raises:
        HTTPException: 처리 과정에서 오류가 발생한 경우 예외를 발생시킵니다.
    """
    logger = setup_logging()
    # 환경 변수에서 OpenAI API 키를 로드합니다.
    openai_api_key = load_environment(logger)
    os.environ["OPENAI_API_KEY"] = openai_api_key  # OpenAI API 키 설정

    # 프롬프트 구성 파일의 경로를 설정합니다.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROMPT_CONFIG_PATH = os.path.join(
        BASE_DIR, "image_to_text_description", "prompt_config.yaml"
    )
    # 프롬프트 구성을 로드합니다.
    prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, logger)
    system_prompt = prompt_config["prompts"]["system_prompt"]
    user_prompt = prompt_config["prompts"]["user_prompt"]

    # LLM 객체를 초기화합니다.
    llm = initialize_llm()
    # MultiModal 객체를 초기화합니다.
    multimodal_llm_with_prompt = MultiModal(
        llm, system_prompt=system_prompt, user_prompt=user_prompt
    )

    json_outputs = []
    # 각 이미지 파일을 순회하며 처리합니다.
    for image_file in files:
        # 이미지 파일을 처리하여 JSON 데이터를 추출합니다.
        json_output = await process_image(image_file, multimodal_llm_with_prompt)
        json_outputs.append(json_output)

    # 추출된 JSON 데이터를 병합합니다.
    combined_json = combine_json_outputs(json_outputs, logger)

    # 결과를 출력합니다.
    print(combined_json)
    print(type(combined_json))

    # 병합된 JSON 데이터를 반환합니다.
    return combined_json


# 애플리케이션 실행 (직접 실행 시)
if __name__ == "__main__":
    uvicorn.run("application_connect_api:app", host="0.0.0.0", port=8000)
