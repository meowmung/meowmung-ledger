import pytest
import os
from fastapi.testclient import TestClient
from application_connect_api import app

# FastAPI TestClient 초기화
client = TestClient(app)


# 테스트 환경 준비: .env 파일의 존재 확인
def test_environment_variables():
    assert os.path.exists(".env"), "The .env file does not exist."
    with open(".env", "r") as f:
        env_content = f.read()
    assert "OPENAI_API_KEY" in env_content, "OPENAI_API_KEY is missing in .env file."


# 테스트 환경 준비: 프롬프트 파일의 존재 확인
def test_prompt_config_file():
    with open("image_to_text_description/prompt_config.yaml", "r") as f:
        assert "prompts" in f.read()


# 테스트: /ledger_receipt 엔드포인트
def test_ledger_receipt_endpoint():
    # 상대 경로로 테스트에 사용할 파일 지정
    test_file_path = "image_to_text_description/images/test_pet_5.jpg"

    # 테스트 파일이 존재하는지 확인
    assert os.path.exists(test_file_path), f"Test file not found: {test_file_path}"

    # 파일을 열고 요청에 첨부
    with open(test_file_path, "rb") as test_file:
        response = client.post("/ledger_receipt", files={"files": test_file})

    # 응답 확인
    assert response.status_code == 200  # 성공 여부 확인
    assert "date" in response.json()  # JSON 응답 필드 확인
    assert "location" in response.json()


# 클린업: 임시 파일 삭제
def test_cleanup():
    import os

    if os.path.exists("test_receipt.jpg"):
        os.remove("test_receipt.jpg")
