# STEP 1
from transformers import pipeline

from fastapi import FastAPI, Form

# STEP 2
question_answerer = pipeline("question-answering", model="stevhliu/my_awesome_qa_model")

app = FastAPI()


@app.post("/qna/")
async def qna(q: str = Form(), a: str = Form()):

    # STEP 3, 데이터 가져오기
    # text

    # STEP 4
    result = question_answerer(question=q, context=a)

    # STEP 5
    return {"result": result}