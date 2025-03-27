from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from Python backend!"}

@app.get("/data")
def get_data():
    sample_data = {"name": "ChatGPT", "type": "AI", "version": "4.0"}
    return sample_data
