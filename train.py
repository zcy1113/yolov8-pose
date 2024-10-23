import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

if __name__ == '__main__':
    model_yaml = "yolov8-pose.yaml"
    data_yaml = "data.yaml"
    pre_model = "yolov8l-pose.pt"
    model = YOLO(model_yaml, task='pose').load(pre_model)
    results = model.train(data=data_yaml, batch=16, project=None)
