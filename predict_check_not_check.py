import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Параметры
IMG_SIZE = (224, 224)
MODEL_PATH = 'check_not_check_model.h5'
TEST_FOLDER = 'test_cheques'

# Загружаем модель
model = load_model(MODEL_PATH)

# Получаем список файлов
files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in files:
    img_path = os.path.join(TEST_FOLDER, filename)

    # Загружаем и подготавливаем изображение
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Предсказание
    prediction = model.predict(img_array)[0][0]
    label = "ЧЕК ✅" if prediction <= 0.5 else "НЕ ЧЕК ❌"

    print(f"{filename}: {label} (уверенность: {prediction:.2%})")

    plt.imshow(img)
    plt.title(f"{label} ({prediction:.2%})")
    plt.axis('off')
    plt.show()
