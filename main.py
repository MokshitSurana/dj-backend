from deepface import DeepFace
import numpy as np
from PIL import Image
from fastapi import FastAPI,  File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost",
"http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/upload_images/")
async def upload_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    uploaded_file = Image.open(image1.file)
    uploaded_file_array = np.array(uploaded_file)

    picture = Image.open(image2.file)
    picture_array = np.array(picture)

    result = DeepFace.verify(picture_array,uploaded_file_array)
    return 1 if result['verified'] else 0
