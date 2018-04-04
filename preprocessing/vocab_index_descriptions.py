# coding=utf-8
import datasets
from constants import *
from nltk.corpus import stopwords

import csv
import jieba

from tqdm import tqdm

def vocab_index_descriptions(vocab_file, vectors_file):
    #load lookups
    ind2w, w2ind = datasets.load_vocab_dict(vocab_file)
    desc_dict = datasets.load_code_descriptions()
        
    with open(vectors_file, 'w', encoding='gb18030') as of:
        w = csv.writer(of, delimiter=' ')
        w.writerow(["CODE", "VECTOR"])
        for code, desc in tqdm(desc_dict.items()):
            tokens = jieba.cut(desc)
            inds = [w2ind[t] if t in w2ind.keys() else len(w2ind)+1 for t in tokens]
            w.writerow([code] + [str(i) for i in inds])
