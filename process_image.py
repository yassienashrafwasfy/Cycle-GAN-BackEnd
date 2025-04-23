from fastapi import FastAPI,HTTPException,File, UploadFile
import numpy as np
import tensorflow as tf
import cv2
import uvicorn
app = FastAPI()

@app.post("/preprocess Image")
async def preprocess_Image(file: UploadFile = File(...)):
    # read image
    contents = await file.read()
    #Image Size
    size = len(contents)
    # decode as np
    nparr = np.frombuffer(contents,np.uint8)
    #decode as img
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    # raise error
    if image is None:
        HTTPException(status_code=400,detail="Image Invalid")
    #preprocess Image between [-1,1]
    image = tf.cast(image,tf.float32)/127.5 - 1.0
    
    return {
        "image_Name": file.filename,
        "size": size,
        "highest value": float(image.numpy().max()),
        "lowest value": float(image.numpy().min()),
    }

if __name__ == "__main__":
    uvicorn.run(app,host="127.0.0.1",port=8001)
