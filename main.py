from flask import Flask
from flask import jsonify
from config import config


def create_app(enviroment):
    app = Flask(__name__)

    app.config.from_object(enviroment)

    return app


environment = config['development']
app = create_app(environment)


@app.route('/ping', methods=['GET'])
def get_ping():
    return 'pong'


@app.route('/hi/<name>', methods=['GET'])
def get_hi(name):
    response = {'message': 'Bienvenido ' + name}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
