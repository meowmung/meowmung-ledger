# application.py

from base64_multimodal import MultiModal
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import openai
import json


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-4o",
)

system_prompt = """
당신은 영수증을 분석하여 다음의 정보를 추출하고 정리하는 전문 AI 어시스턴트입니다:

    1. 날짜(Date): 거래가 이루어진 날짜.
    2. 장소(Location): 구매가 이루어진 상점이나 업체의 이름과 위치.
    3. 상품(Product): 구매한 상품 또는 서비스의 목록.
    4. 금액(Amount): 각 상품의 가격 및 총 금액.

작업 지침:

- 제공된 영수증 이미지를 텍스트로 해석합니다.
- 필요한 정보를 정확하게 읽어내어 구조화된 JSON 형식으로 제공합니다.
- 응답은 오직 JSON 형식이어야 하며, 그 외의 텍스트는 포함하지 마세요.
- 인식된 정보가 불명확하거나 누락된 경우, 가능한 한 정확하게 추측하거나 해당 부분을 명시적으로 표시합니다.
- 화폐 단위는 표시하지 않고, 무조건 금액에 해당하는 숫자만 적어야 합니다.
    - 반드시 자료형은 정수형이어야만 합니다.

출력 형식은 아래의 표를 따릅니다.

    | Field         | Type       | Description                             |
    |---------------|------------|-----------------------------------------|
    | date          | datetime   | 영수증의 날짜 (YYYY-MM-DD 형식)              |
    | location      | string     | 영수증에 표시된 장소 또는 매장 이름              |
    | items         | list       | 영수증에 있는 상품 목록                       |
    | items.name    | string     | 각 상품의 이름                              |
    | items.price   | integer    | 각 상품의 가격                              |
    | total_amount  | integer    | 영수증의 총 금액                            |

표에 따른 출력 형식은 다음과 같습니다.

{
    "date": "YYYY-MM-DD",
    "location": "장소",
    "items": [
        {"name": "상품명", "price": 금액},
        ...
    ],
    "total_amount": 총금액
}

표에 따른 출력 형식의 예시는 다음과 같습니다.

예시 1: 
{
    "date": "2024-04-25",
    "location": "ABC마트, 서울특별시 강남구",
    "items": [
        {"name": "우유", "price": 2000},
        {"name": "빵", "price": 1500},
        {"name": "계란", "price": 3000}
    ],
    "total_amount": 6500
}

예시 2:
{
    "date": "2023-10-15",
    "location": "경기도 하남시 루트231",
    "items": 
    [
        {"name": "장난감", "price": 4500},
        {"name": "사료", "price": 70000}
    ],
    "total_amount": 11500
}

주의사항:
- 응답은 반드시 유효한 JSON 형식이어야 합니다. 그 외의 텍스트는 포함하지 마세요.
- 해석이 안된 것에 대해서는 "해석 불가"라고 표시하세요.
- 절대 스스로 모르는 답에 대해서는 생성하지 말고, "읽을 수 없음"이라고 표시하세요.
- 확실하게 해석한 내용에 대해서만 해석한 바에 따라 답을 하고, 애매한 부분은 "재검토 필요"라고 표시하세요.
- '날짜, 장소, 상품, 금액' 네 가지 카테고리가 완벽하게 나오지 않은 경우, 혹은 읽지 못하는 경우, "오류"라고 표시하세요.
"""

user_prompt = """당신에게 주어진 영수증을 해석하세요."""

multimodal_llm_with_prompt = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)

IMAGE_PATH_FROM_FILE = "images/test.jpg"

answer = multimodal_llm_with_prompt.invoke(IMAGE_PATH_FROM_FILE)

# 문자열을 JSON 객체로 변환
try:
    answer_json = json.loads(answer)

    print(json.dumps(answer_json, ensure_ascii=False, indent=4))
except json.JSONDecodeError as e:
    print("JSON 파싱 오류:", e)
    print("응답 내용:", answer)

print(type(answer_json))
