from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/play', methods=['POST'])
def publish():
    app.logger.debug('Body: %s', request.get_data())
    return 'playing'

@app.route('/done', methods=['POST'])
def done():
    app.logger.debug('Body: %s', request.get_data())
    return 'done'

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')