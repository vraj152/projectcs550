from flask import Flask,request
from flask_cors import CORS
import front_file as ff
import json
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/reccomend', methods=['GET'])
def home():
    raw_input = request.args.get('userId')
    rec_list = ff.take_userInput(int(raw_input))
    response = json.dumps(rec_list, cls=NpEncoder)
    return response

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == "__main__":
    app.run(host='0.0.0.0')