from ultralytics import YOLO

from IPython.display import display, Image

from roboflow import Roboflow

rf = Roboflow(api_key="YLXTeUbudCvbIf1n2CSf")
project = rf.workspace("vayun-goel-jlreu").project("pipeline1-boundingboxweld")
version = project.version(2)
dataset = version.download("yolov8-obb")

import yaml

with open(f'{dataset.location}/data.yaml', 'r') as file:
    data = yaml.safe_load(file)

data['path'] = dataset.location

with open(f'{dataset.location}/data.yaml', 'w') as file:
    yaml.dump(data, file, sort_keys=False)

model = YOLO("yolov8n-obb.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data=f"{dataset.location}/data.yaml", epochs=100, iterations=100, optimizer="AdamW", plots=False, save=False, val=False)
