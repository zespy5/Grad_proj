import torch
import torch.nn as nn


class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super().__init__()
        self.reg_loss = RegLoss()
        self.gamma = gamma

    def forward(self, pos_score, neg_scores, parameters):
        diff = pos_score - neg_scores
        is_same = diff != 0
        sig_diff = torch.sigmoid(diff)

        num = torch.sum(is_same)

        loss = -torch.log(self.gamma + sig_diff)
        loss = is_same * loss
        loss = torch.sum(loss) / num

        reg_loss = self.reg_loss(parameters)
        return loss + reg_loss
