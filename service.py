# coding=utf-8

import torch
import jieba
import datasets
import numpy as np
import csv
from collections import defaultdict
from models import ConvAttnPool
from torch.autograd import Variable

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
    text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in cut.split()]
    data.append(text)
    data = np.array(text)
    data = np.expand_dims(data, axis=0)
    return data

ind2w, w2ind = datasets.load_vocab_dict(VOCAB_FILE)
ind2c = load_codelist(CODE_FILE)
c2ind = {c:i for i,c in ind2c.items()}
dicts = (ind2w, w2ind, ind2c, c2ind)

# dicts = datasets.load_lookups('./data/train.csv', VOCAB_FILE)
# ind2w, w2ind, ind2c, c2ind = dicts[0], dicts[1], dicts[2], dicts[3]

# # Load structure
# model = ConvAttnPool(100, EMBED_FILE, FILTER_SIZE, NUM_FILTER_MAPS, LMBDA, GPU, dicts, embed_size=EMBED_SIZE)
# # Load para
# model.load_state_dict(torch.load(MODEL_DIR, map_location=lambda storage, loc: storage))
model = torch.load(MODEL_DIR)

text = "生儿高胆红素血症"
text = input2array(text)
data = Variable(torch.LongTensor(text), volatile=True)

model.train(False)

output, _, _ = model(data, target=None)
output = output.data.cpu().numpy()
# output = np.round(output)
top5 = output.argsort()[0][-5:][::-1]
print(top5)
results = [ind2c[i] for i in top5]
print(results)
print(ind2c)
print(output)