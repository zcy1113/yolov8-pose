from ultralytics import YOLO

if __name__ =='__main__':
    model_yaml= 'E:\WorkSpace\ylolov8s\ultralytics-main\yolov8.yaml'
    data_yaml = 'E:\WorkSpace\ylolov8s\ultralytics-main\data.yaml'
    pro_model='E:\WorkSpace\ylolov8s\ultralytics-main\yolov8s.pt'
# Load a model
    model = YOLO(model_yaml,task='detect').load(pro_model)

    results = model.train(data=data_yaml, epochs=500, batch=16, imgsz=640)  # train the model
 
