from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, np
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import pandas as pd

# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = '../input/fruits-360/Training'
validation_data_dir = '../input/fruits-360/Test'
nb_train_samples = 56781
nb_validation_samples = 19053
epochs = 1
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(150))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(111, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

np.save('class_indices', train_generator.class_indices)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    seed=42)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42)

test_generator.reset()

pred = model.predict_generator(
    test_generator,
    steps=test_generator.n//test_generator.batch_size,
    verbose=1)

print(pred)

predicted_class_indices = np.argmax(pred, axis=1)

print(predicted_class_indices)

labels = (train_generator.class_indices)
print(labels)
labels = dict((v, k) for k, v in labels.items())
print(labels)
predictions = [labels[k] for k in predicted_class_indices]

print(predictions)

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv("results.csv", index=False)

model.save('model.h5')