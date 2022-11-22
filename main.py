import random
import string

import tensorflow as tf
from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin

from config import config


def create_app(arg_environment):
    local_app = Flask(__name__)
    local_app.config.from_object(arg_environment)
    return local_app


environment = config['development']
app = create_app(environment)
generator = tf.keras.models.load_model('dolphin_generator_epoch_5999.h5')
CORS(app, support_credentials=True)

@app.route('/ping', methods=['GET'])
def get_ping():
    return 'pong'


@cross_origin(supports_credentials=True)
@app.route('/predict', methods=['POST'])
def post_predict():
    data = request.json
    predictions = []

    for question in data['questions']:
        random_answer = ''.join(random.choices(string.ascii_lowercase, k=20))
        predictions.append({'question': question, 'answer': random_answer})

    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
