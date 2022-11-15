import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

from config import config


def create_app(arg_environment):
    local_app = Flask(__name__)
    local_app.config.from_object(arg_environment)
    return local_app


environment = config['development']
app = create_app(environment)
generator = tf.keras.models.load_model('dolphin_generator_epoch_5999.h5')


@app.route('/ping', methods=['GET'])
def get_ping():
    return 'pong'


@app.route('/predict', methods=['POST'])
def post_predict():
    # Predecir imagen
    noise = np.random.uniform(-1.0, 1.0, size=(1, 32))
    image = generator.predict(noise).squeeze()

    # Guardar imagen generada
    plt.imsave('image_generated.jpeg', image, cmap='Greys')

    # Re-escalar imagen
    image = cv2.imread('image_generated.jpeg')
    image = cv2.resize(image, dsize=(280, 280), interpolation=cv2.INTER_CUBIC)
    plt.imsave('image_generated.jpeg', image, cmap='Greys')

    return send_file(
        'image_generated.jpeg',
        mimetype='image/png',
        as_attachment=False)


@app.route('/handle', methods=['POST'])
def post_handle():
    data = request.json
    response = {'message': 'Bienvenido ' + data['name'] + ', cursas ' + str(data['subject'])}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
