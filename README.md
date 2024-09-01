# Weld-Defects-Segmentation
Clone the RIAWELC dataset from the official repository by executing the following command 
```bash
git clone https://github.com/stefyste/RIAWELC.git
```

For the Data preparation and model training execute the following command

```bash
python3 Dataprep.py
```

SqueezeNet Implementation for classification of Defects on RIAWELC dataset.

## Stage-1 (YOLOv8- Object Detection Part)

We have utilised the YOLOv8[https://github.com/orgs/ultralytics/discussions/7472] object detection model to detect the welding region inside the input image to reduce the processing in the further model as the X-Ray welding images are high resolution.

### Sample Output
([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true](https://github.com/Rishabh-S1899/Weld-Defects-Segmentation/blob/4f0aa6d3ca73f377a96d6356f378088d3b476734/Initial_Results/val_batch1_pred.jpg))

Run the Train Yolo Notebook by replacing the following code portion 

```python

%pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Your_API_key")
project = rf.workspace("user").project("pipeline1-boundingboxweld")
version = project.version(1)
dataset = version.download("yolov8-obb")
````
