from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid

from transform_photo import transform_photo
from gold_layer import EmotionRecognition

# Initialize FastAPI app
app = FastAPI()

# templates
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")

BRONZE_DIR = "resources/bronze/"
SILVER_DIR = "resources/silver/"
GOLDEN_DIR = "resources/golden/"
CSV = "file.csv"
MODEL = "model_emotion_v4.pth"

# create directories if not exist
if not os.path.exists(BRONZE_DIR): os.makedirs(BRONZE_DIR)
if not os.path.exists(SILVER_DIR): os.makedirs(SILVER_DIR)
if not os.path.exists(GOLDEN_DIR): os.makedirs(GOLDEN_DIR)

@app.post("/emotion-recognition/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):

    # generate new file name and save it BRONZE folder
    new_filename = str(uuid.uuid4()) + "." + file.filename.split(".")[-1]
    bronze_path = os.path.join(BRONZE_DIR, new_filename)

    with open(bronze_path, "wb") as f:
        f.write(await file.read())

    # process image fom BRONZE and save it to SILVER
    silver_image_name = transform_photo(image_path=bronze_path, destination_folder=SILVER_DIR)

    # pass to image from SILVER emotion recognition script
    recognizer = EmotionRecognition(model_path=MODEL, csv_path=GOLDEN_DIR + CSV, gold_layer=GOLDEN_DIR)
    golden_image_name = recognizer.process_image(SILVER_DIR + silver_image_name)

    # Provide the URL for the processed image
    processed_image_url = f"/{golden_image_name}"

    return templates.TemplateResponse(
        "processed_page.html", 
        {"request": request, "processed_image_url": processed_image_url}
    )

@app.get("/emotion-recognition/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="emotion_recognition.html"
    )

@app.get("/resources/golden/{filename}", response_class=HTMLResponse)
async def read_item(filename: str):
    file_path = GOLDEN_DIR + filename
    
    # Return the file as a response
    return FileResponse(path=file_path, media_type="image/jpeg")
