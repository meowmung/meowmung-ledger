from fastapi.testclient import TestClient
from application_connect_api import app  # FastAPI 애플리케이션 가져오기

client = TestClient(app)


def test_ledger_receipt():
    # Dummy 이미지 데이터 생성
    dummy_file = {"files": ("dummy.jpg", b"dummy_image_data", "image/jpeg")}

    # API 호출
    response = client.post("/ledger_receipt", files=dummy_file)

    # 상태 코드 검증
    assert response.status_code == 200

    # JSON 응답 형식 검증
    json_response = response.json()
    assert "date" in json_response
    assert "location" in json_response
    assert "items" in json_response
    assert "total_amount" in json_response
