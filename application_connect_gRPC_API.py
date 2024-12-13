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
        answer = await asyncio.to_thread(
            multimodal_llm_with_prompt.invoke, temp_file_path
        )

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
        prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, self.logger)
        system_prompt = prompt_config["prompts"]["system_prompt"]
        user_prompt = prompt_config["prompts"]["user_prompt"]

        # LLM 객체 초기화
        llm = initialize_llm()
        # MultiModal 객체 초기화
        self.multimodal_llm_with_prompt = MultiModal(
            llm, system_prompt=system_prompt, user_prompt=user_prompt
        )

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
    # ThreadPoolExecutor를 grpc.aio.server에 전달 및 메시지 크기 옵션 설정
    server = grpc.aio.server(
        ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
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


# # application_connect_gRPC_API.py
# 백엔드 서버와 통신 코드

# import os
# import logging
# import uuid
# import json
# import yaml
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from image_to_text_description.base64_multimodal import MultiModal

# import grpc
# from concurrent.futures import ThreadPoolExecutor  # 올바른 스레드 풀 임포트
# import asyncio  # 비동기 처리를 위한 모듈

# # proto 패키지에서 생성된 모듈 임포트 (ledger_pb2_grpc.py 파일 경로 수정 요망 from . import ledger_pb2 as ledger__pb2)
# from proto import ledger_pb2
# from proto import ledger_pb2_grpc


# import requests
# from requests_aws4auth import AWS4Auth
# import boto3


# def setup_logging():
#     """
#     로깅을 설정하는 함수입니다.
#     """
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)
#     return logger


# def load_environment(logger):
#     """
#     환경 변수를 로드하고, OpenAI API 키를 가져오는 함수입니다.
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
#     """
#     return ChatOpenAI(
#         temperature=0.2,
#         model_name="gpt-4o",
#     )


# async def process_image(image_data: bytes, multimodal_llm_with_prompt):
#     """
#     이미지 데이터를 처리하여 구조화된 JSON 데이터를 추출하는 비동기 함수입니다.
#     """

#     # print(f"받은 이미지 데이터: {image_data}")

#     temp_file_path = None
#     try:
#         temp_file_path = f"temp_{uuid.uuid4().hex}.jpeg"
#         with open(temp_file_path, "wb") as f:
#             f.write(image_data)

#         print(f"임시 저장된 파일 경로: {temp_file_path}")

#         # 멀티모달 LLM을 사용하여 이미지 처리 (동기 함수인 경우 비동기로 실행)
#         answer = await asyncio.to_thread(
#             multimodal_llm_with_prompt.invoke, temp_file_path
#         )

#         if answer.startswith("```json") and answer.endswith("```"):
#             answer = answer[len("```json") : -len("```")].strip()

#         # JSON 문자열을 파이썬 객체로 변환
#         answer_json = json.loads(answer)
#         return answer_json

#     except Exception as e:
#         raise e
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             os.remove(temp_file_path)


# # async def process_image(image_data: any, multimodal_llm_with_prompt):
# #     """
# #     이미지 데이터를 처리하여 구조화된 JSON 데이터를 추출하는 비동기 함수입니다.
# #     """

# #     print(f"받은 이미지 데이터: {image_data}")

# #     # temp_file_path = None
# #     # try:
# #     #     temp_file_path = f"temp_{uuid.uuid4().hex}.jpg"
# #     #     with open(temp_file_path, "wb") as f:
# #     #         f.write(image_data)

# #     #     print(f"임시 저장된 파일 경로: {temp_file_path}")

# #     # 멀티모달 LLM을 사용하여 이미지 처리 (동기 함수인 경우 비동기로 실행)
# #     answer = await asyncio.to_thread(
# #         multimodal_llm_with_prompt.invoke, image_data
# #     )

# #     if answer.startswith("```json") and answer.endswith("```"):
# #         answer = answer[len("```json") : -len("```")].strip()

# #     # JSON 문자열을 파이썬 객체로 변환
# #     answer_json = json.loads(answer)
# #     return answer_json

# #     # except Exception as e:
# #     #     raise e
# #     # finally:
# #     #     if temp_file_path and os.path.exists(temp_file_path):
# #     #         os.remove(temp_file_path)


# def combine_json_outputs(json_outputs: list, logger):
#     combined_data = {
#         "date": None,
#         "location": None,
#         "items": [],
#         "total_amount": None,
#     }
#     for data in json_outputs:
#         if not combined_data["date"] and data.get("date"):
#             combined_data["date"] = data["date"]
#         if not combined_data["location"] and data.get("location"):
#             combined_data["location"] = data["location"]
#         combined_data["items"].extend(data.get("items", []))
#         if data.get("total_amount"):
#             combined_data["total_amount"] = data["total_amount"]
#     return combined_data


# # class LedgerServiceServicer(ledger_pb2_grpc.LedgerServiceServicer):
# #     def __init__(self):
# #         self.logger = setup_logging()
# #         self.openai_api_key = load_environment(self.logger)
# #         os.environ["OPENAI_API_KEY"] = self.openai_api_key

# #         # 프롬프트 구성 파일의 경로 설정
# #         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# #         PROMPT_CONFIG_PATH = os.path.join(
# #             BASE_DIR, "image_to_text_description", "prompt_config.yaml"
# #         )
# #         # 프롬프트 로드
# #         prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, self.logger)
# #         system_prompt = prompt_config["prompts"]["system_prompt"]
# #         user_prompt = prompt_config["prompts"]["user_prompt"]

# #         # LLM 객체 초기화
# #         llm = initialize_llm()
# #         # MultiModal 객체 초기화
# #         self.multimodal_llm_with_prompt = MultiModal(
# #             llm, system_prompt=system_prompt, user_prompt=user_prompt
# #         )

# #     # async def ProcessReceipts(self, request, context):

# #     #     print(f"나오노{request}")

# #     #     try:
# #     #         json_outputs = []
# #     #         for image_data in request.image_data:
# #     #             json_output = await process_image(
# #     #                 image_data, self.multimodal_llm_with_prompt
# #     #             )
# #     #             json_outputs.append(json_output)

# #     #         combined_json = combine_json_outputs(json_outputs, self.logger)
# #     #         print(combined_json)  # 로그로 확인

# #     #         return ledger_pb2.ReceiptsResponse(
# #     #             json_output=json.dumps(combined_json)
# #     #         )

# #     #     except Exception as e:
# #     #         context.set_details(f"이미지 처리 중 오류 발생: {e}")
# #     #         context.set_code(grpc.StatusCode.INTERNAL)
# #     #         return ledger_pb2.ReceiptsResponse()

# #     async def ProcessReceipts(self, request, context):

# #         print(f"나오노{request}")
# #         print(f"나오노{request.image_data}")
# #         print(f"{type(request.image_data)}")

# #         try:
# #             json_outputs = []
# #             for image_url in request.image_data:
# #                 json_output = await process_image(
# #                     image_url, self.multimodal_llm_with_prompt
# #                 )
# #                 json_outputs.append(json_output)

# #             combined_json = combine_json_outputs(json_outputs, self.logger)
# #             print(combined_json)  # 로그로 확인

# #             return ledger_pb2.ReceiptsResponse(json_output=json.dumps(combined_json))

# #         except Exception as e:
# #             context.set_details(f"이미지 처리 중 오류 발생: {e}")
# #             context.set_code(grpc.StatusCode.INTERNAL)
# #             return ledger_pb2.ReceiptsResponse()

# def get_file_extension(url):
#     parsed = requests.utils.urlparse(url)
#     root, ext = os.path.splitext(parsed.path)
#     return ext.lower()

# def get_extension_from_content_type(content_type):
#     mapping = {
#         'image/jpeg': '.jpeg',
#         'image/jpg': '.jpg',
#         'image/png': '.png',
#         'image/gif': '.gif',
#         'image/webp': '.webp',
#     }
#     return mapping.get(content_type.lower(), '.jpeg')  # 기본값으로 .jpeg 사용


# class LedgerServiceServicer(ledger_pb2_grpc.LedgerServiceServicer):


#     def __init__(self):
#         self.logger = setup_logging()
#         self.openai_api_key = load_environment(self.logger)
#         os.environ["OPENAI_API_KEY"] = self.openai_api_key

#         BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#         PROMPT_CONFIG_PATH = os.path.join(
#             BASE_DIR, "image_to_text_description", "prompt_config.yaml"
#         )
#         prompt_config = load_prompt_config(PROMPT_CONFIG_PATH, self.logger)
#         system_prompt = prompt_config["prompts"]["system_prompt"]
#         user_prompt = prompt_config["prompts"]["user_prompt"]

#         llm = initialize_llm()
#         self.multimodal_llm_with_prompt = MultiModal(
#             llm, system_prompt=system_prompt, user_prompt=user_prompt
#         )

#     async def ProcessReceipts(self, request, context):
#         self.logger.info(f"요청 수신: {request}")
#         self.logger.info(f"이미지 데이터: {request.image_data}")
#         self.logger.info(f"타입: {type(request.image_data)}")

#         # AWS 자격 증명 가져오기
#         region = '~~~~~~~~~~~`'  # 사용할 AWS 리전
#         service = '~~~~~~~~~~~`'  # 사용할 AWS 서비스 이름 (예: 's3', 'dynamodb' 등)

#         # AWS4Auth 객체 생성
#         awsauth = AWS4Auth('~~~~~~~~~~~`', '~~~~~~~~~~~`', region, service)

#         try:
#             json_outputs = []
#             for image_url in request.image_data:
#                 self.logger.info(f"처리 중인 이미지 URL: {image_url}")
#                 response = requests.get(image_url, auth=awsauth)
#                 self.logger.info(f"HTTP 응답 상태: {response.status_code}")

#                 if response.status_code != 200:
#                     self.logger.error(f"이미지 다운로드 실패: {image_url}")
#                     raise Exception(f"Failed to download image: {image_url}")

#                 ext = get_file_extension(image_url)

#                 if ext not in ['.jpeg', '.jpg', '.png', '.gif', '.webp']:
#                     self.logger.error(f"지원되지 않는 파일 형식: {ext}")
#                     raise Exception(f"Unsupported file format: {ext}")

#                 temp_file_path = f"temp_{uuid.uuid4().hex}{ext}"
#                 self.logger.info(f"임시 저장된 파일 경로: {temp_file_path}")

#                 # 이미지 데이터를 임시 파일로 저장
#                 with open(temp_file_path, "wb") as f:
#                     f.write(response.content)

#                 json_output = await process_image(
#                     response.content, self.multimodal_llm_with_prompt
#                 )
#                 json_outputs.append(json_output)

#             combined_json = combine_json_outputs(json_outputs, self.logger)
#             self.logger.info(f"결합된 JSON: {combined_json}")

#             return ledger_pb2.ReceiptsResponse(
#                 json_output=json.dumps(combined_json)
#             )

#         except Exception as e:
#             self.logger.error(f"이미지 처리 중 오류 발생: {e}")
#             context.set_details(f"이미지 처리 중 오류 발생: {e}")
#             context.set_code(grpc.StatusCode.INTERNAL)
#             return ledger_pb2.ReceiptsResponse()


# async def serve():
#     # ThreadPoolExecutor를 grpc.aio.server에 전달 및 메시지 크기 옵션 설정
#     server = grpc.aio.server(
#         ThreadPoolExecutor(max_workers=10),
#         options=[
#             ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
#             ("grpc.max_receive_message_length", 50 * 1024 * 1024),
#         ],
#     )  # 비동기 gRPC 서버 생성 및 스레드 풀 설정
#     ledger_pb2_grpc.add_LedgerServiceServicer_to_server(
#         LedgerServiceServicer(), server
#     )  # 서비스 등록
#     server.add_insecure_port("[::]:8085")  # 서버 포트 설정 (암호화되지 않음)
#     await server.start()  # 서버 시작
#     print("gRPC 서버가 시작되었습니다.")  # 서버 시작 메시지 출력
#     await server.wait_for_termination()  # 서버 종료 대기


# if __name__ == "__main__":
#     asyncio.run(serve())  # 서버 실행
