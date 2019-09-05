import pandas as pd
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import np
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = '../input/fruits-360/Training'
validation_data_dir = '../input/fruits-360/Test'
nb_train_samples = 56781
nb_validation_samples = 19053
epochs = 5
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

base_model = Xception(include_top=False, input_shape = input_shape)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(111, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
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

test_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42)

with tf.device("/device:GPU:0"):
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

test_generator.reset()

pred = model.predict_generator(
    test_generator,
    steps=test_generator.n//test_generator.batch_size,
    verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)

labels = (train_generator.class_indices)

labels = dict((v, k) for k, v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv("results.csv", index=False)

model.save('model.h5')