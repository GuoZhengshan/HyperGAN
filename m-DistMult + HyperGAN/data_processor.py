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
        positive_sample = self.triples[idx]
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
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size
        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample = torch.from_numpy(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)
        return positive_sample, negative_sample, self.mode
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]
        return positive_sample, negative_sample, mode

    @staticmethod
    def get_true_element(triples):
        mlist = [{} for _ in range(len(triples[0]) - 1)]
        for triple in triples:
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