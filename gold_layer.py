import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


class EmotionRecognition:
    def __init__(self, model_path, csv_path, gold_layer, device=None):
        self.model_path = model_path
        self.csv_path = csv_path
        self.gold_layer = gold_layer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self.default_transform()
        self.model = self.build_model()
        self.load_model()

    @staticmethod
    def default_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Single-channel normalization
        ])

    def build_model(self):
        model = models.resnet152(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 7)  # Assuming 7 emotion classes
        return model

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):
        img = Image.open(image_path).resize((256,256)).convert('L')  # Grayscale
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            max_probability, _ = torch.max(probabilities, dim=1)

        predicted_class_idx = predicted.item()
        predicted_probability = max_probability.item() * 100
        return predicted_class_idx, predicted_probability

    def save_prediction_image(self, image_path, predicted_class, predicted_probability):
        # Open the original image
        img = Image.open(image_path).convert('L')
        plt.imshow(img, cmap='gray')
        plt.title(f"{predicted_class} - {predicted_probability:.2f}%")
        plt.axis('off')

        # Save the image with the prediction in /gold_layer
        output_path = os.path.join(
            self.gold_layer,
            f"{os.path.basename(image_path).split('/')[-1].split('.')[0]}_predicted.jpg"
        )
        if not os.path.exists(self.gold_layer):
            os.makedirs(self.gold_layer)
        plt.savefig(output_path)
        plt.close()
        print(f"Prediction: {output_path}")
        return output_path

    def save_prediction_to_csv(self, image_name, predicted_class, predicted_probability):
        # Create or append to the CSV
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=['file', 'predicted_class', 'predicted_probability'])
        else:
            df = pd.read_csv(self.csv_path)

        new_row = {
              'file': image_name,
              'predicted_class': predicted_class,
              'predicted_probability': predicted_probability
          }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(self.csv_path, index=False)
        # print(f"Prediction saved to CSV: {self.csv_path}")

    def process_image(self, image_path):
        class_to_idx = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        predicted_idx, predicted_prob = self.predict(image_path)
        predicted_class = idx_to_class[predicted_idx]

        # Save prediction to CSV
        self.save_prediction_to_csv(os.path.basename(image_path), predicted_class, predicted_prob)

        # Save prediction image to /gold_layer
        return self.save_prediction_image(image_path, predicted_class, predicted_prob)

        # print(f"Processed {image_path}: {predicted_class} ({predicted_prob:.2f}%)")
