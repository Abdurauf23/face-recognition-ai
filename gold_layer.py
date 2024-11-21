import csv
import os

def update_csv(image_path: str, predicted_probability, upload_path: str) -> str:
    # Ensure the file has a .csv extension
    if not image_path.endswith(".csv"):
        image_path = os.path.splitext(image_path)[0] + ".csv"

    # Construct the full file path
    file_path = os.path.join(upload_path, image_path)

    with open(file_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Path_to_image", "Predicted_output", "Predicted_probability"])
        writer.writerow([image_path, "emotion", predicted_probability])
    
    return image_path
