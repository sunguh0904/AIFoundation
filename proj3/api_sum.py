# 텍스트 요약하기 최대 200자

# STEP 1, 필요한 패키지 설치
from transformers import pipeline

from fastapi import FastAPI, Form

# STEP 2, 추론기 만들기
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

app = FastAPI()


@app.post("/summarization/")
async def summarization(text: str = Form()):

    # STEP 3, 데이터 가져오기
    # text

    # STEP 4, 데이터 추론시키기
    result = summarizer(text)

    # STEP 5, 결과 반환
    return {"result": result}