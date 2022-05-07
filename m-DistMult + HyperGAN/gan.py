import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_processor import TrainDataset,OneShotIterator


class FCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, output_dim=1):
        super(FCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.F1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.F2 = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, score):
        score = self.F1(score)
        score = self.dropout(torch.tanh(score))
        score = self.F2(score)
        score = torch.sigmoid(score)
        return score


class BaseGAN(object):
    def __init__(self, train_data_idxs, args, model, device, nentity, nrelation):
        self.name = None
        self.model = model
        self.device = device
        self.nentity = nentity
        self.nrelation = nrelation
        self.train_data_idxs = train_data_idxs
        self.arity = len(self.train_data_idxs[0]) - 1
        self.threshold = 0.5
        self.args = args

    def get_vector(self, sample, mode="single"):
        if mode == "single":
            element = [self.model.R(sample[:, 0])]
            for i in range(1, self.arity + 1):
                element.append(self.model.E(sample[:, i]))
            return torch.cat(element, dim=1)
        else:
            pos, neg = sample
            batch_size, negative_sample_size = neg.size(0), neg.size(1)
            element = [self.model.R(pos[:, 0]).unsqueeze(1).expand(-1, negative_sample_size, -1).contiguous()]
            for j in range(1, self.arity + 1):
                if j == mode:
                    element.append(self.model.E(neg.view(-1)).view(batch_size, negative_sample_size, -1))
                else:
                    element.append(self.model.E(pos[:, j]).unsqueeze(1).expand(-1, negative_sample_size, -1).contiguous())
            element = torch.cat(element, dim=2)
        return element

    @staticmethod
    def test_ave_score(trainer):
        true_score = []
        i = 0
        true_triples = trainer.train_data_idxs
        while i < len(true_triples):
            j = min(i + 1024, len(true_triples))
            true_sample = torch.LongTensor(true_triples[i: j]).to(trainer.device)
            true_score.extend(trainer.classifier(trainer.get_vector(true_sample)).view(-1).tolist())
            i = j
        true_score = np.array(true_score)
        trainer.threshold = 0.5
        true_left = (true_score >= trainer.threshold).sum()
        print({
            "true mean": true_score.mean(),
            "true left": true_left,
            "accuracy": true_left / len(true_score),
        })

    def merge_confidence_weight(self, pos, label, miss_ent_domain, pred):
        self.classifier.eval()
        pos = torch.tensor(pos, dtype=torch.long).to(self.device)
        batch_size, negative_sample_size = pos.size(0), self.nentity
        neg = torch.arange(0, self.nentity).unsqueeze(0).expand(batch_size, -1).contiguous().to(self.device)
        element = [self.model.R(pos[:, 0]).unsqueeze(1).expand(-1, negative_sample_size, -1).contiguous()]
        i = 1
        for j in range(1, self.arity + 1):
            if j == miss_ent_domain:
                element.append(self.model.E(neg.view(-1)).view(batch_size, negative_sample_size, -1))
            else:
                element.append(self.model.E(pos[:, i]).unsqueeze(1).expand(-1, negative_sample_size, -1).contiguous())
                i += 1
        element = torch.cat(element, dim=2)
        confidence_weight = self.classifier(element).squeeze(2)    # 128 * 9549
        confidence_weight = (1 - label) * confidence_weight
        confidence_weight[confidence_weight == 0] = -math.inf
        confidence_weight = F.softmax(confidence_weight, dim=1) + label
        return confidence_weight.detach() * pred


class HyperGAN(BaseGAN):
    def __init__(self, train_data_idxs, args, model, device, nentity, nrelation, hidden_dim, hard=False):
        super(HyperGAN, self).__init__(train_data_idxs, args, model, device, nentity, nrelation)
        self.name = "HyperGAN"
        self.classifier = FCN(args.edim * (self.arity + 1), hidden_dim=hidden_dim).to(device)
        self.generator = FCN(args.edim * (self.arity + 1), hidden_dim=hidden_dim).to(device)
        self.clf_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01)
        self.gen_optimizer = torch.optim.SGD(self.generator.parameters(), lr=0.01)
        self.threshold = 0.5
        self.hard = hard

    def generate(self, pos, neg, mode, n_sample=1):
        batch_size, negative_sample_size = neg.size(0), neg.size(1)
        neg_vector = self.get_vector((pos, neg), mode=mode)
        neg_scores = self.generator(neg_vector)
        neg_probs = torch.softmax(neg_scores, dim=1).reshape(batch_size, negative_sample_size)
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor).unsqueeze(1).expand(batch_size, n_sample)
        sample_idx = torch.multinomial(neg_probs, n_sample, replacement=True)
        sample_neg = neg[row_idx, sample_idx.data.cpu()].view(batch_size, n_sample)
        return pos, sample_neg, neg_scores, sample_idx, row_idx

    def discriminate(self, pos, neg, mode):
        self.classifier.train()
        self.clf_optimizer.zero_grad()
        pos_vector = self.get_vector(pos)
        neg_vector = self.get_vector((pos, neg), mode=mode)
        pos_scores = self.classifier(pos_vector)
        neg_scores = self.classifier(neg_vector).reshape_as(pos_scores)

        target = torch.cat([torch.ones(pos_scores.size()), torch.zeros(neg_scores.size())]).to(self.device)
        loss = F.binary_cross_entropy(torch.cat([pos_scores, neg_scores]), target)
        loss.backward()
        self.clf_optimizer.step()
        return loss, torch.tanh((neg_scores - pos_scores).sum())

    @staticmethod
    def train_HyperGAN(trainer, epochs):
        trainer.model.eval()
        name = []
        for i in range(1, trainer.arity + 1):
            name.append('train_dataset_%d' %(i))
        iterator, dataloader = {}, {}
        for i, each in enumerate(name):
            iterator[each] = TrainDataset(trainer.train_data_idxs,
                                          trainer.nentity,
                                          trainer.nrelation,
                                          trainer.args.negative_sample_size,
                                          i + 1)
            dataloader[each] = DataLoader(iterator[each],
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=0,
                                          collate_fn=TrainDataset.collate_fn)
        trainer.train_iterator = OneShotIterator(dataloader)

        epoch_reward, epoch_loss, avg_reward = 0, 0, 0
        num = math.ceil(len(trainer.train_data_idxs) / 128)
        steps = num * epochs * trainer.arity
        for i in range(steps):
            trainer.generator.train()
            positive_sample, negative_sample, mode = next(trainer.train_iterator)
            positive_sample = positive_sample.to(trainer.device)
            negative_sample = negative_sample.to(trainer.device)
            pos, neg, scores, sample_idx, row_idx = trainer.generate(positive_sample, negative_sample, mode)
            loss, rewards = trainer.discriminate(pos, neg, mode)
            epoch_reward += torch.sum(rewards)
            epoch_loss += loss
            rewards = rewards - avg_reward
            trainer.generator.zero_grad()
            log_probs = F.log_softmax(scores, dim=1)
            reinforce_loss = torch.sum(Variable(rewards) * log_probs[row_idx.cuda(), sample_idx.data])
            reinforce_loss.backward()
            trainer.gen_optimizer.step()
            trainer.generator.eval()