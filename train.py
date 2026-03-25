import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. SHRUNK IMAGE SIZE: From 224 to 128 (Saves TONS of time)
IMG_SHAPE = (128, 128) 
BATCH_SIZE = 64 # Processing more images at once
DATA_DIR = "plantvillage dataset/" 

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    subset='validation'
)

# Updated Model section in train.py
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(), # Keeps the math stable
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Forces the AI to use different 'neurons'
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. FAST TRAINING
print("Starting high-speed training...")
model.fit(train_data, validation_data=val_data, epochs=3)

model.save("plant_doctor_model.h5")
print("Done! Model saved.")