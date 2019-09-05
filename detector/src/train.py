from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, np
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
import pandas as pd
from keras.applications.inception_v3 import InceptionV3

# dimensions of our images.
img_width, img_height = 100, 100

train_data_dir = '../input/fruits-360/Training'
validation_data_dir = '../input/fruits-360/Test'
nb_train_samples = 56781
nb_validation_samples = 19053
epochs = 16
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(111, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')



# model = Sequential()
#
# model.add(base_model)
# model.add(GlobalAveragePooling2D())

# model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=32, kernel_size=2, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=64, kernel_size=2, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Conv2D(filters=128, kernel_size=2, activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=2))
#
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(150))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
# model.add(Dense(111, activation='softmax'))
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

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


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

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

predicted_class_indices = np.argmax(pred, axis=1)

labels = (train_generator.class_indices)

labels = dict((v, k) for k, v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv("results.csv", index=False)

model.save('model.h5')