# application_gRPC_client_test.py

import grpc  # gRPC 모듈
from proto import ledger_pb2  # Protocol Buffers로 생성된 메시지 클래스
from proto import ledger_pb2_grpc  # Protocol Buffers로 생성된 gRPC 클래스
import json  # JSON 처리 모듈
import asyncio  # 비동기 처리를 위한 모듈
import aiofiles  # 비동기 파일 처리를 위한 aiofiles 모듈


async def run():
    # 비동기 gRPC 채널 생성 (서버와의 연결 설정)
    channel = grpc.aio.insecure_channel(
        'localhost:8085',
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )
    stub = ledger_pb2_grpc.LedgerServiceStub(channel)  # LedgerServiceStub 생성

    # 단일 이미지 처리 요청
    async with aiofiles.open('image_to_text_description/images/get_pet_6.jpg', 'rb') as f:
        image_data = await f.read()  # 이미지 파일 비동기 읽기

    receipt_request = ledger_pb2.ReceiptRequest(image_data=image_data)  # 단일 이미지 요청 메시지 생성
    receipt_response = await stub.ProcessReceipt(receipt_request)  # 서버에 단일 이미지 처리 요청 전송

    print("단일 영수증 처리 응답:")  # 응답 출력 메시지
    print(json.loads(receipt_response.json_output))  # 응답 JSON 출력

    # 다중 이미지 처리 요청
    image_paths = [
        'image_to_text_description/images/test_pet_2.jpg',
        'image_to_text_description/images/test_pet_1.jpg'
    ]  # 실제 이미지 경로로 변경
    images = []
    for path in image_paths:
        async with aiofiles.open(path, 'rb') as f:
            images.append(await f.read())  # 각 이미지 파일 비동기 읽기 및 리스트에 추가

    receipts_request = ledger_pb2.ReceiptsRequest(image_data=images)  # 다중 이미지 요청 메시지 생성
    receipts_response = await stub.ProcessReceipts(receipts_request)  # 서버에 다중 이미지 처리 요청 전송

    print("다중 영수증 처리 응답:")  # 응답 출력 메시지
    print(json.loads(receipts_response.json_output))  # 응답 JSON 출력

    await channel.close()  # 채널 닫기

if __name__ == "__main__":
    asyncio.run(run())  # 클라이언트 실행
