from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid

from transform_photo import transform_photo
from gold_layer import update_csv

# Initialize FastAPI app
app = FastAPI()

# templates
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")

BRONZE_DIR = "resources/bronze/"
SILVER_DIR = "resources/silver/"
GOLDEN_DIR = "resources/golden/"

@app.post("/emotion-recognition/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # generate new file name and save it BRONZE folder
    new_filename = str(uuid.uuid4()) + "." + file.filename.split(".")[-1]
    bronze_path = os.path.join(BRONZE_DIR, new_filename)
    with open(bronze_path, "wb") as f:
        f.write(await file.read())
        
    print(bronze_path)

    # process image fom BRONZE and save it to SILVER
    silver_image_name = transform_photo(image_path=bronze_path, destination_folder=SILVER_DIR)

    # pass to image from SILVER emotion recognition script
    golden_image_name = update_csv(image_path=silver_image_name, predicted_probability=0.07, upload_path=GOLDEN_DIR)

    # Provide the URL for the processed image
    processed_image_url = f"/static/{golden_image_name}"

    return templates.TemplateResponse(
        "processed_page.html", 
        {"request": request, "processed_image_url": processed_image_url}
    )
@app.get("/emotion-recognition/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="emotion_recognition.html"
    )
