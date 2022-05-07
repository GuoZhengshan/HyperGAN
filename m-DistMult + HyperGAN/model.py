import torch
import torch.nn.functional as F


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return
    def forward(self, pred1, tar1):
        loss = F.binary_cross_entropy(pred1, tar1)
        return loss


class MDistMult(torch.nn.Module):
    def __init__(self, d, d_e, d_r, device, **kwargs):
        super(MDistMult, self).__init__()
        self.E = torch.nn.Embedding(len(d.entities), embedding_dim=d_e, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), embedding_dim=d_r, padding_idx=0)
        self.E.weight.data = (1e-3 * torch.randn((len(d.entities), d_e), dtype=torch.float).to(device))
        self.R.weight.data = (1e-3*torch.randn((len(d.relations), d_r), dtype=torch.float).to(device))
        self.loss = MyLoss()
        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout = torch.nn.Dropout(kwargs["hidden_dropout"])
        self.bne = torch.nn.BatchNorm1d(d_e)
        self.bnr = torch.nn.BatchNorm1d(d_r)
        self.bnw = torch.nn.BatchNorm1d(d_e)
        self.ary = len(d.train_data[0])-1

    def forward(self, r_idx, e_idx, miss_ent_domain):
        r = self.bnr(self.R(r_idx))
        x = r

        if self.ary == 3:
            e2, e3 = self.E(e_idx[0]), self.E(e_idx[1])
            e2, e3 = self.bne(e2), self.bne(e3)
            e2, e3 = self.input_dropout(e2), self.input_dropout(e3)
            x = x * e2 * e3
            x = self.bnw(x)
            x = self.hidden_dropout(x)
            x = torch.mm(x, self.E.weight.transpose(1, 0))

        if self.ary == 4:
            e2, e3, e4 = self.E(e_idx[0]), self.E(e_idx[1]), self.E(e_idx[2])
            e2, e3, e4 = self.bne(e2), self.bne(e3), self.bne(e4)       # 128 * 25
            e2, e3, e4 = self.input_dropout(e2), self.input_dropout(e3), self.input_dropout(e4)
            x = x * e2 * e3 * e4
            x = self.bnw(x)
            x = self.hidden_dropout(x)
            x = torch.mm(x, self.E.weight.transpose(1, 0))

        pred = F.softmax(x, dim=1)
        return pred
