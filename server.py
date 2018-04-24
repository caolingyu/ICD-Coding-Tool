# coding=utf-8

from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import torch
from torch.autograd import Variable
import json
import numpy as np
import jieba
import datasets
import csv
from collections import defaultdict
from models import ConvAttnPool

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

# Define some constants
MODEL_DIR = "./saved_models/conv_attn_Apr_10_01:02/model_best_f1_micro.pth"
VOCAB_FILE = "./data/vocab.csv"
CODE_FILE = "./data/code_list.csv"
EMBED_FILE = None
FILTER_SIZE = 4
NUM_FILTER_MAPS = 500
LMBDA = 0
GPU = False
EMBED_SIZE = 100

def load_codelist(codelist):
    ind2c = defaultdict(str)
    with open(codelist, 'r', encoding='gb18030') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            ind2c[i] = line
    return ind2c

def input2array(input):
    data = []
    jieba.load_userdict('./preprocessing/dict.txt')
    cut = jieba.cut(input)
    cut = ' '.join(cut)
    text_raw = [w for w in cut.split()]
    text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in cut.split()]
    data.append(text)
    data = np.array(text)
    data = np.expand_dims(data, axis=0)
    return data, text_raw

ind2w, w2ind = datasets.load_vocab_dict(VOCAB_FILE)
ind2c = load_codelist(CODE_FILE)
c2ind = {c:i for i,c in ind2c.items()}
dicts = (ind2w, w2ind, ind2c, c2ind)
desc_dict = datasets.load_code_descriptions()


@app.route('/api/test', methods=['POST'])
def test():
    model = torch.load(MODEL_DIR)
    r = request.json
    # r_json = json.loads(r)
    data_raw = r['data']
    data, text_raw = input2array(data_raw)
    data = Variable(torch.LongTensor(data), volatile=True)
    model.train(False)
    
    output, _, alpha = model(data, target=None)
    output = output.data.cpu().numpy()
    alpha = alpha.data.cpu().numpy()
    # output = np.round(output)
    top5 = output.argsort()[0][-5:][::-1]
    results = [ind2c[i] for i in top5]
    alpha = [alpha.tolist()[0][i][0:len(text_raw)] for i in top5]
    prob = [output.tolist()[0][i] for i in top5]
    desc = []
    for item in results:
        if item in desc_dict:
            desc.append(desc_dict[item])
        else:
            desc.append('Unknown')
    # response
    response = {
        'results': results,
        'prob': prob,
        'desc': desc,
        'text': text_raw,
        'alpha': alpha
    }
    # encode response using jsonpickle
    # response_pickled = json.dumps(response)
    # return Response(response=response_pickled, status=200, mimetype="application/json")
    return jsonify(response)


# start flask app
app.run(host="0.0.0.0", port=5000)