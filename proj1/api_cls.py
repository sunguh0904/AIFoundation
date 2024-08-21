# STEP 1: Import the necessary modules. 추론할 패키지
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

from fastapi import FastAPI, File, UploadFile

# STEP 2: Create an ImageClassifier object. 추론기 객체 만드는 법
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=4)
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


from PIL import Image
import io
import numpy as np
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image. 추론시킬 데이터
    # 1. create pil image from http file
    # 1-1. convert http file to file
    pil_img = Image.open(io.BytesIO(contents))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image. 추론 시키기
    classification_result = classifier.classify(image)

    # STEP 5: Process the classification result. In this case, visualize it. 추론이랑은 상관 없음, 딥러닝 끝
    # 보기쉽게? 보여주는 용도
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"
    
    return {"result": result}