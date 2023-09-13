from fastapi import FastAPI, UploadFile

api = FastAPI()

# define a root `/` endpoint
@api.get("/")
def index():
    return {"ok": "API connected"}


@api.post("/predict")
async def predict(uploaded_file: UploadFile):
    return {'prediction': (uploaded_file.size % 2) == 0}
