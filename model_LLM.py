import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import os
from tqdm import tqdm
import re
import ast
import pandas as pd
import copy
import warnings
from math import log


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_text2seq = nn.Linear(768, self.hidden_size, bias=True)
        self.linear_four = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, 1, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def parse_embedding(self, embedding_str):
        embedding_values = ast.literal_eval(embedding_str)
        embedding_tensor = trans_to_cuda(torch.tensor(embedding_values).float())
        return embedding_tensor
    def LLM_enhance(self, text_long_list):  # (b,d)
        text_long_emb = [self.parse_embedding(text_long) for text_long in text_long_list]
        text_long_emb = torch.cat(text_long_emb, dim=0)
        text_long_emb = self.linear_text2seq(text_long_emb)

        return text_long_emb #[b,d]

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def uniformity(self, x):
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, emb_1, emb_2):
        align = self.alignment(emb_1, emb_2)
        uniform = (self.uniformity(emb_1) + self.uniformity(emb_2)) / 2
        loss = align + uniform
        return loss

    def compute_scores(self, hidden, mask, text_long, text_short):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, 1)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)

        long_text_emb = self.LLM_enhance(text_long)
        short_text_emb = self.LLM_enhance(text_short)

        au_loss_1 = self.calculate_loss(a, long_text_emb)
        au_loss_2 = self.calculate_loss(ht, short_text_emb)

        l_alpha = self.linear_four(torch.sigmoid(a + long_text_emb))
        s_alpha = self.linear_five(torch.sigmoid(ht + short_text_emb))
        a = a * l_alpha
        ht = ht * s_alpha

        a = self.linear_transform(torch.cat([a, ht], 1))


        b = self.embedding.weight[1:]
        scores = torch.matmul(a, b.transpose(1, 0))

        return scores, (au_loss_1 + au_loss_2) * 0.1

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def parse_embedding(embedding_str):
    embedding_values = ast.literal_eval(embedding_str)
    embedding_tensor = trans_to_cuda(torch.tensor(embedding_values).float())
    return embedding_tensor


def forward(model, i, data):
    alias_inputs, A, items, mask, targets, text_long, text_short = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    scores, au_loss = model.compute_scores(seq_hidden, mask, text_long, text_short)
    return targets, scores, au_loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, au_loss = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss = loss + au_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f, au_loss: %.4f' % (j, len(slices), loss.item(), au_loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit_5, hit_10, hit_20 = [], [], []
    mrr_5, mrr_10, mrr_20 = [], [], []
    ndcg_5, ndcg_10, ndcg_20 = [], [], []

    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores, au_loss= forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()

        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit_5.append(np.isin(target - 1, score[:5]))
            hit_10.append(np.isin(target - 1, score[:10]))
            hit_20.append(np.isin(target - 1, score[:20]))

            if len(np.where(score == target - 1)[0]) == 0:
                mrr_5.append(0)
                mrr_10.append(0)
                mrr_20.append(0)
                ndcg_5.append(0)
                ndcg_10.append(0)
                ndcg_20.append(0)
            else:
                rank = np.where(score == target - 1)[0][0] + 1
                if rank <= 5:
                    mrr_5.append(1 / rank)
                    ndcg_5.append(1 / log(rank + 1, 2))
                if rank <= 10:
                    mrr_10.append(1 / rank)
                    ndcg_10.append(1 / log(rank + 1, 2))
                if rank <= 20:
                    mrr_20.append(1 / rank)
                    ndcg_20.append(1 / log(rank + 1, 2))

    hit_5 = np.mean(hit_5) * 100
    hit_10 = np.mean(hit_10) * 100
    hit_20 = np.mean(hit_20) * 100
    mrr_5 = np.mean(mrr_5) * 100
    mrr_10 = np.mean(mrr_10) * 100
    mrr_20 = np.mean(mrr_20) * 100
    ndcg_5 = np.mean(ndcg_5) * 100
    ndcg_10 = np.mean(ndcg_10) * 100
    ndcg_20 = np.mean(ndcg_20) * 100

    return hit_5, hit_10, hit_20, mrr_5, mrr_10, mrr_20, ndcg_5, ndcg_10, ndcg_20
