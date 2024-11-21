import os
from PIL import Image

def transform_photo(image_path: str, destination_folder: str) -> str:
    try:
        with Image.open(image_path) as img:
            img = img.resize((256, 256))
            if img.mode in ('RGBA', 'LA'):
                img = img.convert('RGB')
            img = img.convert("L")

            file_name, _ = os.path.splitext(os.path.basename(image_path))[0]
            new_file_name = f"{file_name}_transformed.jpg"
            new_file_path = os.path.join(destination_folder, new_file_name)
            
            img.save(new_file_path, "JPEG")
            return new_file_name
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")