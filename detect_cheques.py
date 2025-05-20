import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
from ultralytics import YOLO
import easyocr
import pandas as pd

# === Настройки ===
CHECK_MODEL_PATH = 'check_not_check_model.h5'
FIELD_MODEL_PATH = 'runs/detect/receipt-fields-detector2/weights/best.pt'
INPUT_DIR = 'test_cheques'
RESULT_CSV = 'final_extracted_cheques.csv'
IMG_SIZE = (224, 224)

# === Загрузка моделей ===
check_model = load_model(CHECK_MODEL_PATH)
field_model = YOLO(FIELD_MODEL_PATH)
ocr_reader = easyocr.Reader(['ru', 'en'])

# === Функция: чек или не чек ===
def is_check(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = check_model.predict(img_array)[0][0]
    # Исправленная логика: чек, если prediction <= 0.5
    is_cheque = prediction <= 0.5
    return is_cheque, prediction

# === OCR по обрезанным областям ===
def extract_text_from_crops(img_path, results):
    img = cv2.imread(img_path)
    extracted = {"file": os.path.basename(img_path), "amount": "", "receiver": "", "date": ""}

    for r in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = r.tolist()
        cls_name = field_model.model.names[int(cls)]
        crop = img[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue
        text = ocr_reader.readtext(crop, detail=0)
        full_text = " ".join(text).strip()
        extracted[cls_name] = full_text

    return extracted

# === Обработка всех изображений ===
results_list = []
for filename in os.listdir(INPUT_DIR):
    image_path = os.path.join(INPUT_DIR, filename)
    is_chk, confidence = is_check(image_path)

    print(f"{filename}: is_check={is_chk}, confidence={confidence:.2f}")

    if is_chk:
        results = field_model(image_path)
        extracted = extract_text_from_crops(image_path, results)
        results_list.append(extracted)
    else:
        print(f"❌ Пропущено (не чек): {filename}")

# === Сохранение результатов ===
df = pd.DataFrame(results_list)
df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
print(f"✅ Готово! Сохранено в {RESULT_CSV}")