import pymongo
from flask import Flask, jsonify
from flask import request
from flask import render_template
from pymongo import MongoClient
import pika
import json
from bson.json_util import dumps
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
client = MongoClient('mongodb://mongo:27017/')
db = client.fruitsdetector


@app.route('/play', methods=['POST'])
def play():
    app.logger.debug(request.get_data())
    db.streams.update_one(
        {'name': request.form.get('name')},
        {'$set': {'play': True}},
        upsert=True,
    )

    data = {
        'name': request.form.get('name'),
        'play': True,
    }
    message = json.dumps(data)

    connection = pika.BlockingConnection(pika.URLParameters('amqp://guest:guest@rabbitmq:5672/%2F'))
    channel = connection.channel()
    channel.queue_declare(queue='streams', durable=True)

    result = channel.basic_publish(
        exchange='',
        routing_key='streams',
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))

    app.logger.debug(result)

    connection.close()

    return 'play'


@app.route('/done', methods=['POST'])
def done():
    db.streams.update_one(
        {'name': request.form.get('name')},
        {'$set': {'play': False}},
        upsert=True,
    )

    return 'done'


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/streams', methods=['GET'])
def streams():
    streams = dumps(db.streams.find().sort("play", pymongo.DESCENDING))

    return streams

@app.route('/streams/<name>', methods=['GET'])
def stream(name):
    stream = dumps(db.streams.find_one({'name': name}))

    return stream

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
