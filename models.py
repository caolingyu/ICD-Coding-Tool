# coding=utf-8
"""
    Holds PyTorch models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np

import math
import random
import sys
import time

sys.path.append('../')
from constants import *
from preprocessing import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, Y, embed_file, dicts, lmbda=0, dropout=0.5, gpu=False, embed_size=100):
        super(BaseModel, self).__init__()
        self.gpu = gpu
        self.Y = Y
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)
        self.lmbda = lmbda

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))

            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts[0])
            self.embed = nn.Embedding(vocab_size+2, embed_size)


    def get_loss(self, yhat, target, diffs=None):
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #add description regularization loss if relevant
        if self.lmbda > 0 and diffs is not None:
            diff = torch.stack(diffs).mean()
            if random.random() > 0.99:
                #keep track of this while training
                print("loss: %.5f" % loss.data[0])
                print("diff: %.5f" % diff.data[0])
                print("total loss: %.5f" % (loss + diff).data[0])
            loss = loss + diff
        return loss

    def embed_descriptions(self, desc_data):
        #label description embedding via convolutional layer
        #number of labels is inconsistent across instances, so have to iterate
        b_batch = []
        for inst in desc_data:
            if len(inst) > 0:
                lt = Variable(torch.LongTensor(inst))
                d = self.desc_embedding(lt)
                d = d.transpose(1,2)
                d = self.label_conv(d)
                d = F.max_pool1d(F.tanh(d), kernel_size=d.size()[2])
                d = d.squeeze(2)
                b_inst = self.label_fc1(d)
                b_batch.append(b_inst)
            else:
                b_batch.append([])
        return b_batch

    def compare_label_embeddings(self, target, b_batch, desc_data):
        #description regularization loss 
        #b is the embedding from description conv
        #iterate over batch because each instance has different # labels
        diffs = []
        for i,bi in enumerate(b_batch):
            ti = target[i]
            inds = torch.nonzero(ti.data).squeeze().cpu().numpy()

            zi = self.final[inds,:]
            diff = (zi - bi).mul(zi - bi).mean()

            #multiply by number of labels to make sure overall mean is balanced with regard to number of labels
            diffs.append(self.lmbda*diff*bi.size()[0])
        return diffs

    def params_to_optimize(self):
        return self.parameters()

# =========================================================================================

class ConvAttnPool(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, lmbda, gpu, dicts, embed_size=100, dropout=0.5):
        super(ConvAttnPool, self).__init__(Y, embed_file, dicts, lmbda, dropout=dropout, gpu=gpu, embed_size=embed_size)

        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(kernel_size/2))
        xavier_uniform(self.conv.weight)

        #context vectors for computing attention as in 2.2
        self.U = nn.Linear(num_filter_maps, Y)
        self.U.bias.data.fill_(0)
        self.U.bias.requires_grad = False

        #final layer: create a matrix to use for the L binary classifiers as in 2.3
        if gpu:
            self.final = Variable(torch.zeros((Y, num_filter_maps)).cuda())
            self.final_bias = Variable(torch.zeros(Y).cuda())
        else:
            self.final = Variable(torch.zeros((Y, num_filter_maps)))
            self.final_bias = Variable(torch.zeros(Y))
        xavier_uniform(self.final)
        self.final.requires_grad = True
        self.final_bias.requires_grad = True

        #conv for label descriptions as in 2.5
        #description module has its own embedding and convolution layers
        if lmbda > 0:
            W = self.embed.weight.data
            self.desc_embedding = nn.Embedding(W.size()[0], W.size()[1])
            self.desc_embedding.weight.data = W.clone()

            self.label_conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size, padding=int(kernel_size/2))
            xavier_uniform(self.label_conv.weight)

            self.label_fc1 = nn.Linear(num_filter_maps, num_filter_maps)
            xavier_uniform(self.label_fc1.weight)
        
    def forward(self, x, target, desc_data=None, get_attention=True):
        #get embeddings and apply dropout
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #apply convolution and nonlinearity (tanh)
        x = F.tanh(self.conv(x).transpose(1,2))
        y = []
        ss = []
        alphas = []
        for x_i in x:
            #apply attention
            alpha_i = F.softmax(self.U(x_i).t())
            #document representations are weighted sums using the attention. Can compute all at once as a matmul
            m_i = alpha_i.mm(x_i)

            #final layer classification
            y_i = self.final.mul(m_i).sum(dim=1).add(self.final_bias)

            #save attention
            alphas.append(alpha_i)
            y.append(y_i)

        r = torch.stack(y)
        alpha = torch.stack(alphas)
        
        if desc_data is not None:
            desc_data = desc_data
            #run descriptions through description module
            b_batch = self.embed_descriptions(desc_data)
            #get l2 similarity loss
            diffs = self.compare_label_embeddings(target, b_batch, desc_data)
        else:
            diffs = None
            
        #final sigmoid to get predictions
        yhat = F.sigmoid(r)
        if target is not None:
            loss = self.get_loss(yhat, target, diffs)
        else:
            loss = 0
        return yhat, loss, alpha
    
    def params_to_optimize(self):
        #use this to include final layer parameters and exclude attention (U)'s bias term
        ps = []
        for param in self.embed.parameters():
            ps.append(param)
        for param in self.conv.parameters():
            ps.append(param)
        for i,param in enumerate(self.U.parameters()):
            #don't add bias
            if i == 0:
                ps.append(param)
        if self.lmbda > 0:
            for param in self.desc_embedding.parameters():
                ps.append(param)
            for param in self.label_conv.parameters():
                ps.append(param)
            for param in self.label_fc1.parameters():
                ps.append(param)
        ps.append(self.final)
        ps.append(self.final_bias)
        return ps

# =========================================================================================

class VanillaConv(BaseModel):

    def __init__(self, Y, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5):
        super(VanillaConv, self).__init__(Y, embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        #initialize conv layer as in 2.1
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, Y)
        xavier_uniform(self.fc.weight)

    def forward(self, x, target, desc_data=None, get_attention=False):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)

        #conv/max-pooling
        c = self.conv(x)
        if get_attention:
            #get argmax vector too
            x, argmax = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2], return_indices=True)
            attn = self.construct_attention(argmax, c.size()[2])
        else:
            x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
            attn = None
        x = x.squeeze(dim=2)

        #linear output
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        loss = self.get_loss(yhat, target)
        return yhat, loss, attn

    def construct_attention(self, argmax, num_windows):
        attn_batches = []
        for argmax_i in argmax:
            attns = []
            for i in range(num_windows):
                #generate mask to select indices of conv features where max was i
                mask = (argmax_i == i).repeat(1,self.Y).t()
                #apply mask to every label's weight vector and take the sum to get the 'attention' score
                weights = self.fc.weight[mask].view(-1,self.Y)
                if len(weights.size()) > 0:
                    window_attns = weights.sum(dim=0)
                    attns.append(window_attns)
                else:
                    #this window was never a max
                    attns.append(Variable(torch.zeros(self.Y)).cuda())
            #combine
            attn = torch.stack(attns)
            attn_batches.append(attn)
        attn_full = torch.stack(attn_batches)
        #put it in the right form for passing to interpret
        attn_full = attn_full.transpose(1,2)
        return attn_full

# =========================================================================================

class VanillaRNN(BaseModel):
    """
        General on purpose - can be LSTM or GRU
    """

    def __init__(self, Y, embed_file, dicts, rnn_dim, cell_type, num_layers, gpu, embed_size=100, bidirectional=False):
        super(VanillaRNN, self).__init__(Y, embed_file, dicts, embed_size=embed_size)
        self.gpu = gpu
        self.rnn_dim = rnn_dim
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        #recurrent unit
        if self.cell_type == 'lstm':
            self.rnn = nn.LSTM(self.embed_size, self.rnn_dim/self.num_directions, self.num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(self.embed_size, self.rnn_dim/self.num_directions, self.num_layers, bidirectional=bidirectional)
        #linear output
        self.final = nn.Linear(self.rnn_dim, Y)

        #arbitrary initialization
        self.batch_size = 16
        self.hidden = self.init_hidden()

    def forward(self, x, target, desc_data=None, get_attention=False):
        #clear hidden state at the start of each batch
        self.refresh(x.size()[0])

        #embed
        embeds = self.embed(x).transpose(0,1)
        #apply RNN
        out, self.hidden = self.rnn(embeds, self.hidden)

        #get final hidden state in the appropriate way
        last_hidden = self.hidden[0] if self.cell_type == 'lstm' else self.hidden
        last_hidden = last_hidden[-1] if self.num_directions == 1 else last_hidden[-2:].transpose(0,1).contiguous().view(self.batch_size, -1)
        #apply linear layer and sigmoid to get predictions
        yhat = F.sigmoid(self.final(last_hidden))
        loss = self.get_loss(yhat, target)
        return yhat, loss, None

    def init_hidden(self):
        if self.gpu:
            h_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                  self.rnn_dim/self.num_directions).zero_())
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.cuda.FloatTensor(self.num_directions*self.num_layers, self.batch_size,
                                                      self.rnn_dim/self.num_directions).zero_())
                return (h_0, c_0)
            else:
                return h_0
        else:
            h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.rnn_dim/self.num_directions))
            if self.cell_type == 'lstm':
                c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, self.batch_size, self.rnn_dim/self.num_directions))
                return (h_0, c_0)
            else:
                return h_0

    def refresh(self, batch_size):
        self.batch_size = batch_size
        self.hidden = self.init_hidden()


# =========================================================================================

class Encoder(BaseModel):
    def __init__(self, Y, embed_file, dicts, embed_size=100, hidden_size=200, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__(Y, embed_file, dicts, embed_size=embed_size)
        # self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layers = n_layers
        self.dropout = dropout
        # self.embedding = nn.Embedding(input_size,embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_length=None, hidden=None):
        '''
        :param input_seqs: 
            Variable of shape (num_step(T),batch_size(B)), sorted decreasingly by lengths(for packing)
        :param input:
            list of sequence length
        :param hidden:
            initial state of GRU
        :returns:
            GRU outputs in shape (T,B,hidden_size(H))
            last hidden stat of RNN(i.e. last output for GRU)
        '''
        embedded = self.embed(input_seqs)
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(embedded, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method=None, hidden_size=200):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]


class Decoder(BaseModel):
    def __init__(self, Y, embed_file, dicts, hidden_size=200, embed_size=100, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__(Y, embed_file, dicts, embed_size=embed_size)
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = Y
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        # self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, Y)

    def forward(self, word_input, target, last_hidden, encoder_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embed(word_input)
        # word_embedded = self.embed(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        yhat = F.sigmoid(self.out(output))
        loss = self.get_loss(yhat, target)
        # Return final output, hidden state
        return yhat, loss, attn_weights


class Seq2Seq(BaseModel):
    def __init__(self, encoder, decoder, Y, embed_file, dicts):
        super(Seq2Seq, self).__init__(Y, embed_file, dicts)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, desc_data=None):
        # batch_size = src.size(1)
        # max_len = trg.size(0)
        # vocab_size = self.decoder.output_size
        # outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        # for t in range(1, max_len):
        yhat, loss, attn_weights = self.decoder(
                output, trg, hidden, encoder_output)
        # outputs[t] = output
        # is_teacher = random.random() < teacher_forcing_ratio
        # top1 = output.data.max(1)[1]
        # output = Variable(trg.data[t] if is_teacher else top1).cuda()

        return yhat, loss, attn_weights
    
    def forward(self, src, trg, desc_data=None):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size))

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, trg, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs

    def params_to_optimize(self):
        return self.parameters()

