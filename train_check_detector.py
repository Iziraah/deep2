import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Гиперпараметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = 'data'
CLASSES_USED = ['check', 'not_check']  

# Только нужные классы
def filter_classes(directory, allowed_classes):
    for item in os.listdir(directory):
        path = os.path.join(directory, item)
        if os.path.isdir(path) and item not in allowed_classes:
            print(f"Убираем из рассмотрения: {path}")
            os.rename(path, path + "_ignore")


filter_classes(DATA_DIR, CLASSES_USED)

# Генераторы с нормализацией и аугментацией
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=5,
    zoom_range=0.05,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    classes=CLASSES_USED,  
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    classes=CLASSES_USED,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')  # бинарная классификация
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# Сохраняем модель
model.save("check_not_check_model.h5")
print("Модель успешно сохранена: check_not_check_model.h5")

# Возвращаем названия папок на место
for item in os.listdir(DATA_DIR):
    if item.endswith('_ignore'):
        os.rename(os.path.join(DATA_DIR, item), os.path.join(DATA_DIR, item.replace('_ignore', '')))
