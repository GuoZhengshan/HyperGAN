#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.triples = triples
        self.arity = len(triples[0]) - 1
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.true_list = self.get_true_element(self.triples)
        
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        positive_sample = tuple(self.triples[idx].numpy())
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
            mask = np.in1d(
                negative_sample,
                self.true_list[self.mode - 1][tuple(positive_sample[:self.mode] + positive_sample[self.mode + 1:])],
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]     # 只是头或尾实体e的下标
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.from_numpy(negative_sample)     # 数组转换成张量
        
        positive_sample = torch.LongTensor(positive_sample)
        # positive_sample是一个triplet，negative_sample是头或尾实体e的下标，要配合self.mode来使用：mode == head-batch，负样本就会是head变化
        return positive_sample, negative_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        # data是batch size大小的__getitem__
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode

    
    @staticmethod
    def get_true_element(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        mlist = [{} for _ in range(len(triples[0]) - 1)]
        for triple in triples:
            triple = tuple(triple.numpy())
            for index, item in enumerate(triple):
                if index == 0:
                    continue
                t = tuple(triple[:index] + triple[index + 1:])
                if t not in mlist[index - 1]:
                    mlist[index - 1][t] = []
                mlist[index - 1][t].append(item)

        for dict in mlist:
            for t in dict.keys():
                dict[t] = np.array(list(set(dict[t])))

        return mlist

    
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)    # 展开成tenor数组，就没有()了
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))
            
        return positive_sample, negative_sample, filter_bias, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
    
class OneShotIterator(object):
    def __init__(self, iterator):
        self.arity = len(iterator)
        self.iterator = []
        for value in iterator.values():
            self.iterator.append(self.one_shot_iterator(value))
        self.step = 0
        
    def __next__(self):
        self.step += 1
        idx = self.step % self.arity
        data = next(self.iterator[idx])
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data