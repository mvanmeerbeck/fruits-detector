from os.path import isfile
from keras.models import load_model
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import rootpath
import pika
import json
from pymongo import MongoClient
import time

path = rootpath.detect()

client = MongoClient('mongodb://mongo:27017/')
db = client.fruitsdetector

connection = pika.BlockingConnection(pika.URLParameters('amqp://guest:guest@rabbitmq:5672/%2F'))
channel = connection.channel()

channel.queue_declare(queue='streams', durable=True)

model = load_model(path + '/output/model.h5')

if isfile(path + '/output/class_indices.npy'):
    labels = np.load(path + '/src/class_indices.npy', allow_pickle=True).item()

labels = dict((v, k) for k, v in labels.items())

def predict(ch, method, properties, body):
    print(" [x] Received %r" % body, type(body))
    message = json.loads(body)

    print(message, type(message))

    ch.basic_ack(delivery_tag=method.delivery_tag)

    stream = db.streams.find_one({'name': message['name']})

    print(stream)

    if True == stream['play']:
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        myrtmp_addr = "rtmp://nginx:1935/stream/" + message['name']
        last_prediction = ''

        cap = cv2.VideoCapture(myrtmp_addr)

        while (True == stream['play']):
            ret, image = cap.read()

            if ret==True:
                image = cv2.resize(image, (100, 100))

                image = image[..., ::-1].astype(np.float64)
                image = np.reshape(image, [1, 100, 100, 3])
                image = np.array(image, dtype=np.float64)
                image = test_datagen.standardize(image)

                prediction_classes = model.predict_classes(image)
                predictions = [labels[k] for k in prediction_classes]

                if last_prediction != predictions[0]:
                    print('update prediction to ' + predictions[0])
                    db.streams.update_one(
                        {'name': message['name']},
                        {'$set': {'prediction': predictions[0]}},
                        upsert=True,
                    )
                    last_prediction = predictions[0]
            else:
                time.sleep(1)
                stream = db.streams.find_one({'name': message['name']})
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    print(" [x] Done")

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='streams', on_message_callback=predict)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()