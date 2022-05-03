# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class Adversarial(object):
    def __init__(self, model, eps=0.2):
        self.model = model
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, emb_name='embedding'):
        pass

    def restore(self, emb_name='embedding'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.emb_backup
                para.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FGSM(Adversarial):
    def __init__(self, model, eps=0.2):
        super(FGSM, self).__init__(model, eps)

    def attack(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.emb_backup[name] = param.data.clone()
                r_hat = self.eps * param.grad.sign()
                param.data.add_(r_hat)


# FGM
class FGM(object):
    def __init__(self, model: nn.Module, eps=0.2):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, emb_name='embedding'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}


class PGD(Adversarial):
    def __init__(self, model, eps=0.1, alpha=0.3):
        super(PGD, self).__init__(model, eps)
        self.alpha = alpha
        self.grad_backup = {}

    def attack(self, emb_name='embedding', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def project(self, emb_name, param_data):
        r = param_data - self.emb_backup[emb_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[emb_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FreeLB(Adversarial):
    def __int__(self, model, eps=0.1, alpha=0.3):
        super(FreeLB, self).__int__(model, eps)
        self.alpha = alpha

    def lookup_emb(self, input_ids, emb_name='embedding'):
        return self.model.embedding(input_ids)

    def attack(self, delta, embeds_init, emb_name='embedding'):
        delta.requires_grad_()
        input_embeds = delta + embeds_init
        self.model()


class Smart(Adversarial):
    def __init__(self, model, eps=0.1, alpha=0.3):
        super(Smart, self).__init__(model, eps)
        self.alpha = alpha

    def lookup_emb(self, input_ids, emb_name='embedding'):
        return self.model.embedding(input_ids)

    def generate_noise(self, embed, eps=1e-4):
        noise = torch.zeros_like(embed).normal_(0, 1) * eps
        noise.detach()
        noise.requires_grad_()
        return noise

    def stable_kl(self, logit, target, epsilon=1e-6, reduce=True):
        logit = logit.view(-1, logit.size(-1)).float()
        # target = target.view(-1, target.size(-1)).float()
        ones = torch.sparse.torch.eye(10)
        target = ones.index_select(0, target)
        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        y = F.log_softmax(target, 1).exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()







