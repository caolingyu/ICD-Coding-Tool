# coding=utf-8
"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys

from constants import *

class Batch:
    def __init__(self, desc_embed):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        self.desc_embed = desc_embed
        self.descs = []

    def add_instance(self, row, ind2c, c2ind, w2ind, dv_dict, num_labels):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        labels = set()
        # hadm_id = int(row[1])
        text = row[2]
        length = int(row[3])
        cur_code_set = set()
        labels_idx = np.zeros(num_labels)
        labelled = False
        desc_vecs = []
        #get codes as a multi-hot vector
        for l in row[1].split():
            if l in c2ind.keys():
                code = int(c2ind[l])
                labels_idx[code] = 1
                cur_code_set.add(code)
                labelled = True
        if not labelled:
            return
        if self.desc_embed:
            for code in cur_code_set:
                l = ind2c[code]
                if l in dv_dict.keys():
                    #need to copy or description padding will get screwed up
                    desc_vecs.append(dv_dict[l][:])
                else:
                    desc_vecs.append([len(w2ind)+1])
        #OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        #truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]

        #build instance
        self.docs.append(text)
        self.labels.append(labels_idx)
        # self.hadm_ids.append(hadm_id)
        self.code_set = self.code_set.union(cur_code_set)
        if self.desc_embed:
            self.descs.append(pad_desc_vecs(desc_vecs))
        #reset length
        self.length = min(self.max_length, length)

    def pad_docs(self):
        #pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

    def to_ret(self):
        return np.array(self.docs), np.array(self.labels), np.array(self.hadm_ids), self.code_set,\
               np.array(self.descs)

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

def data_generator(filename, dicts, batch_size, num_labels, desc_embed=False):
    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations
            num_labels: size of label output space
            desc_embed: true if using DR-CAML (lambda > 0)
        Yields:
            np arrays with data for training loop.
    """
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts[0], dicts[1], dicts[2], dicts[3], dicts[5]
    with open(filename, 'r', encoding='gb18030') as infile:
        r = csv.reader(infile)
        #header
        next(r)
        cur_inst = Batch(desc_embed)
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                #clear
                cur_inst = Batch(desc_embed)
            cur_inst.add_instance(row, ind2c, c2ind, w2ind, dv_dict, num_labels)
        cur_inst.pad_docs()
        yield cur_inst.to_ret()

def load_code_freqs(train_path):
    """
        Inputs:
            train_path: path to training data
        Returns:
            a dict of (code: frequency) pairs for train, dev, and test datasets combined, and the total number of examples
    """
    num_lines = 0
    freqs = defaultdict(float)
    with open(train_path, encoding='gb18030') as train_file:
        r = csv.reader(train_file)
        #header
        next(r)
        for row in r:
            num_lines += 1
            labels = row[1].split(' ')
            if len(labels) > 0:
                for label in labels:
                    if label != '':
                        freqs[label] += 1

    with open(train_path.replace("train", 'test'), encoding='gb18030') as dev_file:
        r = csv.reader(dev_file)
        #header
        next(r)
        for row in r:
            num_lines += 1
            labels = row[1].split(' ')
            if len(labels) > 0:
                for label in labels:
                    if label != '':
                        freqs[label] += 1

    freqs = {c: freq / float(num_lines) for c, freq in freqs.items()}
    return freqs, num_lines

def load_vocab_dict(vocab_file):
    #reads vocab_file into two lookups
    ind2w = defaultdict(str)
    with open(vocab_file, 'r', encoding='gb18030') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                ind2w[i+1] = line.rstrip()
    w2ind = {w:i for i,w in ind2w.items()}
    return ind2w, w2ind

def load_lookups(train_path, vocab_file, desc_embed=False):
    """
        Inputs:
            train_path: path to train dataset
            vocab_file: path to vocab
            desc_embed: true if using DR-CAML
        Outputs:
            vocab lookups, ICD code lookups, description lookup, description one-hot vector lookup
    """
    #get vocab lookups
    ind2w, w2ind = load_vocab_dict(vocab_file)

    #get code and description lookups
    ind2c, desc_dict = load_full_codes(train_path, desc_embed)
    c2ind = {c:i for i,c in ind2c.items()}

    #get desription one-hot vector lookup
    if desc_embed:
        dv_dict = load_description_vectors()
    else:
        dv_dict = None

    dicts = (ind2w, w2ind, ind2c, c2ind, desc_dict, dv_dict)
    return dicts

# def load_full_codes(train_path, desc_embed):
#     """
#         Inputs:
#             train_path: path to train dataset
#         Outputs:
#             code lookup, description lookup
#     """
#     #get description lookup
#     if desc_embed:
#         desc_dict = load_code_descriptions()
#     else:
#         desc_dict = []
#     #build code lookups from appropriate datasets
#     codes = set()
#     for split in ['train', 'test']:
#         with open(train_path.replace('train', split), 'r', encoding='gb18030') as f:
#             lr = csv.reader(f)
#             next(lr)
#             for row in lr:
#                 for code in row[1].split():
#                     codes.add(code)
#     ind2c = defaultdict(str)
#     ind2c.update({i:c for i,c in enumerate(codes)})
#     return ind2c, desc_dict

def load_full_codes(train_path, desc_embed):
    """
        Inputs:
            train_path: path to train dataset
        Outputs:
            code lookup, description lookup
    """
    #get description lookup
    if desc_embed:
        desc_dict = load_code_descriptions()
    else:
        desc_dict = []
    #build code lookups from appropriate datasets
    ind2c = defaultdict(str)
    with open('./data/code_list.csv', 'r', encoding='gb18030') as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            ind2c[i] = line
    return ind2c, desc_dict

def load_code_descriptions():
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)

    with open('./data/ICD_Descriptions.csv', 'r', encoding='gb18030') as labelfile:
        f = csv.reader(labelfile)
        next(f)
        for row in f:
            code = row[0] if row[0] != '' else row[1]
            code = code.strip('_')
            if code not in desc_dict.keys():
                desc_dict[code] = ' '.join(row[2:])
    return desc_dict

def load_description_vectors():
    #load description one-hot vectors from file
    dv_dict = {}
    data_dir = DATA_DIR
    with open("%s/description_vectors.vocab" % (data_dir), 'r', encoding='gb18030') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        #header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict
