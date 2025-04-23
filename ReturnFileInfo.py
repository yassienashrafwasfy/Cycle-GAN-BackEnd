from fastapi import FastAPI ,File, UploadFile
import uvicorn
app = FastAPI()

@app.post("/echo-file")
async def echo_file(file: UploadFile= File(...)):
    contents = await file.read()
    size = len(contents)
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_in_bytes": size,
    }
    # Declare a parameter `file` of type UploadFile. 
    # File(...) tells FastAPI: “This comes from the request’s multipart/form-data body.”


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)