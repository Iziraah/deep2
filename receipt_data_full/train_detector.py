from ultralytics import YOLO

# Загрузка предобученной модели YOLO
model = YOLO("yolov8m.pt")

# Обучение на разметке
model.train(
    data="C:/Users/Mariia/Desktop/deep2/receipt_data_full/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="receipt-fields-detector"
)