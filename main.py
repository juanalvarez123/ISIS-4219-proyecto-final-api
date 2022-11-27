from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin
from transformers import pipeline

from config import config


def create_app(arg_environment):
    local_app = Flask(__name__)
    local_app.config.from_object(arg_environment)
    return local_app


qa_pipeline = pipeline(
    "question-answering",
    model="mrm8488/bert-multi-cased-finetuned-xquadv1",
    tokenizer="mrm8488/bert-multi-cased-finetuned-xquadv1"
)
environment = config['development']
app = create_app(environment)
CORS(app, support_credentials=True)


@app.route('/ping', methods=['GET'])
def get_ping():
    return 'pong'


@cross_origin(supports_credentials=True)
@app.route('/v1/predict', methods=['POST'])
def post_predict():
    data = request.json
    predictions = []

    for question in data['questions']:
        answer = qa_pipeline({
            'context': data['text'],
            'question': question})
        predictions.append({'question': question, 'answer': answer})

    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
