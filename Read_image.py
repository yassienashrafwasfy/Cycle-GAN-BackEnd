from fastapi import FastAPI,HTTPException,File, UploadFile
import numpy as np
import cv2
import uvicorn
app = FastAPI()

@app.post("/image-info")
async def image_info(file: UploadFile = File(...)):
    
#1) read data
    contents = await file.read()
    size = len(contents)
    content_type = file.content_type
    filename = file.filename
    
#2) convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
#3) get image shape
    height, width, channels = image.shape
    
    return {
        "filename": filename,
        "content_type": content_type,
        "size_in_bytes": size,
        "height": height,
        "width": width,
        "channels": channels
    }
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
    