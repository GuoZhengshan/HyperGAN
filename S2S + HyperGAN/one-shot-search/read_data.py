import os
import torch
import numpy as np
from itertools import count
from collections import namedtuple, defaultdict


class nary_dataloader:

    def __init__(self, data_dir="../KG_Data_nary/WikiPeople-3/"):
        self.train_data = self.load_data(os.path.join(data_dir, "train.txt"))
        self.valid_data = self.load_data(os.path.join(data_dir, "valid.txt"))
        self.test_data = self.load_data(os.path.join(data_dir, "test.txt"))
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations if i not in self.train_relations] + [i for i in self.test_relations if i not in self.train_relations]
        self.relations = sorted(list(set(self.relations)))

    def load_data(self, data_dir):
        with open(data_dir, "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[0] for d in data])))
        return relations

    def get_entities(self, data):
        ent = []
        for k in range(1, len(data[0])):
            ent = ent + [d[k] for d in data]
        entities = sorted(list(set(ent)))
        return entities



class binary_dataLoader:
    def __init__(self, task_dir):
        self.inPath = task_dir

        print("The toolkit is importing datasets.\n")
        with open(os.path.join(self.inPath, "relation2id.txt")) as f:
            tmp = f.readline()
            self.n_rel = int(tmp.strip())
            print("The total of relations is {}".format(self.n_rel))

        with open(os.path.join(self.inPath, "entity2id.txt")) as f:
            tmp = f.readline()
            self.n_ent = int(tmp.strip())
            print("The total of entities is {}".format(self.n_ent))

        self.train_head, self.train_tail, self.train_rela = self.read_data("train2id.txt")
        self.valid_head, self.valid_tail, self.valid_rela = self.read_data("valid2id.txt")
        self.test_head,  self.test_tail,  self.test_rela  = self.read_data("test2id.txt")

    def read_data(self, filename):
        allList = []
        head = []
        tail = []
        rela = []
        with open(os.path.join(self.inPath, filename)) as f:
            tmp = f.readline()
            total = int(tmp.strip())
            for i in range(total):
                tmp = f.readline()
                h, t, r = tmp.strip().split()
                h, t, r = int(h), int(t), int(r)
                allList.append((h, t, r))

        allList.sort(key=lambda l:(l[0], l[1], l[2]))

        head.append(allList[0][0])
        tail.append(allList[0][1])
        rela.append(allList[0][2])

        for i in range(1, total):
            if allList[i] != allList[i-1]:
                h, t, r = allList[i]
                head.append(h)
                tail.append(t)
                rela.append(r)
        return head, tail, rela

    def read_triplets(self, filename):
        pos_head = []
        pos_tail = []
        pos_rela = []
        neg_head = []
        neg_tail = []
        neg_rela = []
        with open(os.path.join(self.inPath, filename)) as f:
            lines = f.readlines()
            for tmp in lines:
                h, t, r, l = tmp.strip().split()
                h, t, r, l = int(h), int(t), int(r), int(l)
                if l>0:
                    pos_head.append(h)
                    pos_tail.append(t)
                    pos_rela.append(r)
                else:
                    neg_head.append(h)
                    neg_tail.append(t)
                    neg_rela.append(r)
        return pos_head, pos_tail, pos_rela, neg_head, neg_tail, neg_rela

    def graph_size(self):
        return (self.n_ent, self.n_rel)

    def load_data(self, index):
        if index == 'train':
            return self.train_head, self.train_tail, self.train_rela
        elif index == 'valid':
            return self.valid_head, self.valid_tail, self.valid_rela
        else:
            return self.test_head,  self.test_tail,  self.test_rela

    def load_triplets(self, index):
        self.valid_ph, self.valid_pt, self.valid_pr, self.valid_nh, self.valid_nt, self.valid_nr = self.read_triplets('valid_neg.txt')
        self.test_ph, self.test_pt, self.test_pr, self.test_nh, self.test_nt, self.test_nr = self.read_triplets('test_neg.txt')

        if index == 'valid':
            return (self.valid_ph, self.valid_pt, self.valid_pr), (self.valid_nh, self.valid_nt, self.valid_nr)
        elif index == 'test':
            return (self.test_ph, self.test_pt, self.test_pr), (self.test_nh, self.test_nt, self.test_nr)
        else:
            raise NotImplementedError

    def heads_tails(self):
        all_heads = self.train_head + self.valid_head + self.test_head
        all_tails = self.train_tail + self.valid_tail + self.test_tail
        all_relas = self.train_rela + self.valid_rela + self.test_rela
        
        print(len(all_heads), len(all_tails), len(all_relas))

        heads = defaultdict(lambda: set())
        tails = defaultdict(lambda: set())
        for h, t, r in zip(all_heads, all_tails, all_relas):
            tails[(h, r)].add(t)
            heads[(t, r)].add(h)


        heads_sp = {}
        tails_sp = {}
        for k in heads.keys():
            heads_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(heads[k])]),
                                                   torch.ones(len(heads[k])), torch.Size([self.n_ent]))
        for k in tails.keys():
            tails_sp[k] = torch.sparse.FloatTensor(torch.LongTensor([list(tails[k])]),
                                                   torch.ones(len(tails[k])), torch.Size([self.n_ent]))

        print("heads/tails size:", len(tails), len(heads))#, len(heads_cache), len(tails_cache))

        return heads_sp, tails_sp


