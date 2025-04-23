from fastapi import FastAPI
import uvicorn
app = FastAPI()

@app.get("/ping")
async def ping():
    return {"ping": "pong!"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)