import os
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import backend as K, Sequential
from keras.layers import np, Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

img_width, img_height = 100, 100
train_data_dir = '../input/fruits-360/Training'
test_data_dir = '../input/fruits-360/Test'
epochs = 5
batch_size = 32

output_dir = '../output/' + datetime.now().isoformat()
os.mkdir(output_dir)

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode='categorical',
    shuffle=False)

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=16, kernel_size=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=256, kernel_size=1, activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.40))

# fully-connected layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.40))

model.add(Dense(111, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

with open(output_dir + '/model_architecture.yml', 'w') as f:
    f.write(model.to_yaml())

plot_model(model, to_file=output_dir + '/model.png', show_shapes=True)


with open(output_dir + '/modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

with tf.device("/device:GPU:0"):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size)

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_dir + '/accuracy.png')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_dir + '/loss.png')

print("evaluation time")
evaluation = model.evaluate_generator(test_generator, steps=test_generator.n // test_generator.batch_size, verbose=1)

print(evaluation)
with open(output_dir + '/evaluation.txt', 'w') as f:
    f.write(str(evaluation[0]) + "\n")
    f.write(str(evaluation[1]))

print("prediction time")
test_generator.reset()

pred = model.predict_generator(
    test_generator,
    steps=test_generator.n // test_generator.batch_size,
    verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv(output_dir + "/predictions.csv", index=False)

np.save(output_dir + '/class_indices', train_generator.class_indices)
model.save(output_dir + '/model.h5')
