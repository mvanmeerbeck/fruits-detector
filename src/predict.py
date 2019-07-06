from os.path import isfile
from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1. / 255)
myrtmp_addr = "rtmp://localhost:1935/live/test"

cap = cv2.VideoCapture(myrtmp_addr)

model = load_model('model.h5')

if isfile('class_indices.npy'):
    labels = np.load('class_indices.npy', allow_pickle=True).item()

labels = dict((v, k) for k, v in labels.items())

while(True):
    ret, image = cap.read()

    image = cv2.resize(image,(100,100))

    image = image[...,::-1].astype(np.float64)
    image = np.reshape(image,[1,100,100,3])
    image = np.array(image, dtype=np.float64)
    image = test_datagen.standardize(image)

    prediction = model.predict_classes(image)
    predictions = [labels[k] for k in prediction]

    print(predictions)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




# image = cv2.imread('../input/fruits-360/Test/Guava/30_100.jpg')
# image = image[...,::-1].astype(np.float32)
#
# model = load_model('model.h5')
#
# image = cv2.resize(image,(100,100))
# image = np.reshape(image,[1,100,100,3])
# image = np.array(image, dtype=np.float64)
# image = test_datagen.standardize(image)
# print(image)
# prediction = model.predict_classes(image)
# print(prediction)
#
# if isfile('class_indices.npy'):
#     labels = np.load('class_indices.npy').item()
#
# print(labels)
# labels = dict((v, k) for k, v in labels.items())
# print(labels)
# predictions = [labels[k] for k in prediction]
#
# print(predictions)