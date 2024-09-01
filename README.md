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

We have utilised the [YOLOv8](https://github.com/orgs/ultralytics/discussions/7472) object detection model to detect the welding region inside the input image to reduce the processing in the further model as the X-Ray welding images are high resolution.

### Sample Output

![val_batch1_pred](https://github.com/user-attachments/assets/548ef311-a98e-4782-a3e6-f8d8b1211b50)


Run the Train Yolo Notebook by replacing the following code portion 

```python

%pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Your_API_key")
project = rf.workspace("user").project("pipeline1-boundingboxweld")
version = project.version(1)
dataset = version.download("yolov8-obb")
````
With your own version of the Download code generated from the roboflow. Export the data in the form of YOLOv8-obb format. 
