from ultralytics import YOLO

from IPython.display import display, Image

import requests
import zipfile
import io
import os
import yaml

# URL of the dataset
url = "https://app.roboflow.com/ds/XIoAKveJkT?key=8JJBNShNQK"

# Download the zip file
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Create a directory to extract the files
dataset_dir = "downloaded_dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Extract all files
zip_file.extractall(dataset_dir)

# Update the data.yaml file
yaml_path = os.path.join(dataset_dir, 'data.yaml')
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)

data['path'] = os.path.abspath(dataset_dir)

with open(yaml_path, 'w') as file:
    yaml.dump(data, file, sort_keys=False)

model = YOLO("yolov8n-obb.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data=yaml_path, epochs=100, iterations=100, optimizer="AdamW", plots=False, save=False, val=False)
