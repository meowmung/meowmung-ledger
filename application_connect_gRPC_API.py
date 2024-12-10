# application_connect_gRPC_API.py

import os
import logging
import uuid
import json
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from image_to_text_description.base64_multimodal import MultiModal

import grpc
from concurrent.futures import ThreadPoolExecutor  # 올바른 스레드 풀 임포트
import asyncio  # 비동기 처리를 위한 모듈

# proto 패키지에서 생성된 모듈 임포트
from proto import ledger_pb2
from proto import ledger_pb2_grpc


def setup_logging():
    """
    로깅을 설정하는 함수입니다.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


def load_environment(logger):
    """
    환경 변수를 로드하고, OpenAI API 키를 가져오는 함수입니다.
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
    """
    return ChatOpenAI(
        temperature=0.2,
        model_name="gpt-4o",
    )


async def process_image(image_data: bytes, multimodal_llm_with_prompt):
    """
    이미지 데이터를 처리하여 구조화된 JSON 데이터를 추출하는 비동기 함수입니다.
    """
    temp_file_path = None
    try:
        temp_file_path = f"temp_{uuid.uuid4().hex}.jpg"
        with open(temp_file_path, "wb") as f:
            f.write(image_data)

        # 멀티모달 LLM을 사용하여 이미지 처리 (동기 함수인 경우 비동기로 실행)
        answer = await asyncio.to_thread(multimodal_llm_with_prompt.invoke, temp_file_path)

        if answer.startswith("```json") and answer.endswith("```"):
            answer = answer[len("```json") : -len("```")].strip()

        # JSON 문자열을 파이썬 객체로 변환
        answer_json = json.loads(answer)
        return answer_json

    except Exception as e:
        raise e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def combine_json_outputs(json_outputs: list, logger):
    """
    여러 개의 JSON 출력을 하나의 JSON으로 병합하는 함수입니다.
    """
    combined_data = {
        "date": None,
        "location": None,
        "items": [],
        "total_amount": None,
    }
    for data in json_outputs:
        if not combined_data["date"] and data.get("date"):
            combined_data["date"] = data["date"]
        if not combined_data["location"] and data.get("location"):
            combined_data["location"] = data["location"]
        combined_data["items"].extend(data.get("items", []))
        if data.get("total_amount"):
            combined_data["total_amount"] = data["total_amount"]
    return combined_data


class LedgerServiceServicer(ledger_pb2_grpc.LedgerServiceServicer):
    def __init__(self):
        self.logger = setup_logging()
        self.openai_api_key = load_environment(self.logger)
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # 프롬프트 구성 파일의 경로 설정
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROMPT_CONFIG_PATH = os.path.join(
            BASE_DIR, "image_to_text_description", "prompt_config.yaml"
        )
        # 프롬프트 로드
        prompt_config = load_prompt_config(
            PROMPT_CONFIG_PATH, self.logger
        )
        system_prompt = prompt_config["prompts"]["system_prompt"]
        user_prompt = prompt_config["prompts"]["user_prompt"]

        # LLM 객체 초기화
        llm = initialize_llm()
        # MultiModal 객체 초기화
        self.multimodal_llm_with_prompt = MultiModal(
            llm, system_prompt=system_prompt, user_prompt=user_prompt
        )

    async def ProcessReceipt(self, request, context):
        """
        단일 영수증 이미지를 처리하여 JSON 데이터를 반환하는 비동기 메서드입니다.
        """
        try:
            image_data = request.image_data  # 요청에서 이미지 데이터 추출
            json_output = await process_image(
                image_data, self.multimodal_llm_with_prompt
            )  # 이미지 처리
            combined_json = json_output  # 단일 처리이므로 바로 반환
            return ledger_pb2.ReceiptResponse(
                json_output=json.dumps(combined_json)
            )  # JSON 응답 반환
        except Exception as e:
            context.set_details(f"이미지 처리 중 오류 발생: {e}")  # 에러 메시지 설정
            context.set_code(grpc.StatusCode.INTERNAL)  # gRPC 상태 코드 설정
            return ledger_pb2.ReceiptResponse()  # 빈 응답 반환

    async def ProcessReceipts(self, request, context):
        """
        다중 영수증 이미지를 처리하여 병합된 JSON 데이터를 반환하는 비동기 메서드입니다.
        """
        try:
            json_outputs = []  # JSON 출력 리스트 초기화
            for image_data in request.image_data:
                json_output = await process_image(
                    image_data, self.multimodal_llm_with_prompt
                )  # 각 이미지 처리
                json_outputs.append(json_output)  # 결과 추가

            combined_json = combine_json_outputs(json_outputs, self.logger)  # 결과 병합
            return ledger_pb2.ReceiptsResponse(
                json_output=json.dumps(combined_json)
            )  # 병합된 JSON 응답 반환
        except Exception as e:
            context.set_details(f"이미지 처리 중 오류 발생: {e}")  # 에러 메시지 설정
            context.set_code(grpc.StatusCode.INTERNAL)  # gRPC 상태 코드 설정
            return ledger_pb2.ReceiptsResponse()  # 빈 응답 반환


async def serve():
    # ThreadPoolExecutor를 grpc.aio.server에 전달
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )  # 비동기 gRPC 서버 생성 및 스레드 풀 설정
    ledger_pb2_grpc.add_LedgerServiceServicer_to_server(
        LedgerServiceServicer(), server
    )  # 서비스 등록
    server.add_insecure_port("[::]:8085")  # 서버 포트 설정 (암호화되지 않음)
    await server.start()  # 서버 시작
    print("gRPC 서버가 시작되었습니다.")  # 서버 시작 메시지 출력
    await server.wait_for_termination()  # 서버 종료 대기


if __name__ == "__main__":
    asyncio.run(serve())  # 서버 실행
