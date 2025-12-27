import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore



# Initialize the ImageDataGenerator with basic preprocessing
datagen = ImageDataGenerator(rescale=1./255)

# 1. Create a data generator with optional preprocessing/augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values between 0 and 1
    horizontal_flip=True,   # Augmentation: randomly flip images
    zoom_range=0.2,
    rotation_range=20,
    validation_split=0.2    # Split your data into training and validation
)

# 2. Create train and validation iterators using `.flow_from_directory()`
train_data = train_datagen.flow_from_directory(
    "Facial Recognition/Dataset/train",         # Replace with your dataset folder
    target_size=(48, 48),   # Resize all images to 48x48
    batch_size=32,
    class_mode='categorical',
    subset='training',      # Use part of the dataset for training
    shuffle=True
)

val_data = train_datagen.flow_from_directory(
    "Facial Recognition/Dataset/train",      # Same folder as above
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='validation',    # Use the other part for validation
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale=1./255)  # No augmentation!

test_data = test_datagen.flow_from_directory(
    "Facial Recognition/Dataset/test",         # Your test folder
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=False                # Important: keep the order for evaluation
)


# Check the loaded data
images, labels = next(train_data)
print("Image batch shape:", images.shape)   # e.g., (32, 48, 48, 3)
print("Label batch shape:", labels.shape)   # e.g., (32, 7)

# Define the model
model = Sequential()

# 1st Convolutional Block
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Block
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout
          (0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

# Output Layer (7 classes for facial expressions)
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Summary
model.summary()

optimizer = Adam(learning_rate=0.001)  # Experiment with different values
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5)

# Save the best model based on validation loss
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', save_best_only=True)

model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stop, checkpoint])

# Evaluate on validation/test set
loss, accuracy = model.evaluate(val_data, steps=len(val_data))
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

model.save('my_model')  # saves as a folder named 'my_model'