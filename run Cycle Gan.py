from fastapi import FastAPI,UploadFile,File,HTTPException
import numpy as np
import cv2
import tensorflow as tf
import uvicorn
from fastapi.responses import StreamingResponse
from io import BytesIO
app = FastAPI()

model = tf.keras.models.load_model('Weights.h5',compile=False)

def normalize(img):
    img = tf.cast(img, dtype=tf.float32) / 127.5 - 1.0  # Normalize images to [-1,1]
    return img

def denormalize(img):
        return (img + 1) / 2


@app.post("/CycleGAN")
async def generate_img(file: UploadFile = File(...)):
    #read Img
    contents = await file.read()
    # npbuffer
    nparr = np.frombuffer(contents,np.uint8)
    # cv2---> Image
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    if image is None:
        HTTPException(status_code=400,detail="image not found")
    image = cv2.resize(image,(272,272))
    #preprocess Image [-1,1]
    image = normalize(image)
    image = np.expand_dims(image,axis=0)
    # Run model
    Generated = model(image,training=False)
    # Denormalize
    Generated = denormalize(Generated.numpy().squeeze())  # [0, 1]
    Generated = (Generated * 255).astype(np.uint8)         # [0, 255] uint8
    # save
    output_img = cv2.cvtColor(Generated,cv2.COLOR_RGB2BGR)
    success,encoded_image = cv2.imencode(".jpg",output_img)
    if success is None:
        HTTPException(status_code=500,detail="error encoding image")
    
    return StreamingResponse(BytesIO(encoded_image.tobytes()), media_type="image/jpeg")     
if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8001)
    