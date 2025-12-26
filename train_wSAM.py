import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from models import *
from datasets import *
from utils import *

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors


# ============================================================
# Standard SAM Optimizer (Sharpness-Aware Minimization)
# Reference: Foret et al., ICLR 2021
# ============================================================
class SAM:
    def __init__(self, optimizer, model, rho=0.05):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.state = defaultdict(dict)
        self.last_grad_norm = None  

    @torch.no_grad()
    def first_step(self):
        grads = []
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))

        grad_norm = torch.norm(torch.stack(grads), p=2) + 1e-16
        self.last_grad_norm = grad_norm.item()

        scale = self.rho / grad_norm
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if "eps" not in self.state[p]:
                self.state[p]["eps"] = torch.zeros_like(p.grad, device=p.grad.device)
            eps = scale * p.grad
            self.state[p]["eps"] = eps
            p.add_(eps)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for _, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])

        self.optimizer.step()
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "sam_state": self.state,
            "rho": self.rho,
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.state = state_dict["sam_state"]
        if "rho" in state_dict:
            self.rho = state_dict["rho"]


 
class Solver(object):
    def __init__(self, args):
        super(Solver, self).__init__()

        self.args = args
        self.log_path = os.path.join(self.args.output_path, "log.txt")
        self.emotions = ["hap", "sad", "neu", "ang", "sur", "dis", "fea"]
        self.best_wa = 0
        self.best_ua = 0

        if len(self.args.gpu_ids) > 0:
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device(
            f'cuda:{self.args.gpu_ids[0]}' if self.args.gpu_ids else 'cpu'
        )

        seed = self.args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        self.model = create_model(self.args)
        if len(self.args.gpu_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.gpu_ids)
        self.model.to(self.device)

        self.train_dataloader = create_dataloader(self.args, "train")
        self.test_dataloader = create_dataloader(self.args, "test")

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.args.label_smoothing
        ).to(self.device)

        base_optimizer = torch.optim.AdamW(
            list(self.model.parameters()),
            lr=self.args.lr,
            eps=self.args.eps,
            weight_decay=self.args.weight_decay
        )
        self.optimizer = SAM(base_optimizer, self.model, rho=0.05)

        self.scheduler = build_scheduler(
            self.args,
            self.optimizer.optimizer,
            len(self.train_dataloader)
        )

        if args.resume:
            checkpoint = torch.load(args.resume, map_location=self.device)
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch'])
            )
            self.args.start_epoch = checkpoint['epoch'] + 1
            self.best_wa = checkpoint['best_wa']
            self.best_ua = checkpoint['best_ua']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])


    def DSM(self, logits, features, target):
        loss_ce = self.criterion(logits, target)
        loss_sc = self.DSM(
            features, target, temperature=0.07, beta=0.5, eta=0.2
        )
        return loss_ce, loss_sc

    def run(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            inf = '********************' + str(epoch) + '********************'
            start_time = time.time()

            with open(self.log_path, 'a') as f:
                f.write(inf + '\n')
            print(inf)

            train_acc, train_loss = self.train(epoch)

            val_acc, val_loss, features, targets = self.validate(epoch)

            is_best = (val_acc[0] > self.best_wa) or (val_acc[1] > self.best_ua)
            self.best_wa = max(val_acc[0], self.best_wa)
            self.best_ua = max(val_acc[1], self.best_ua)

            self.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'best_wa': self.best_wa,
                'best_ua': self.best_ua,
                'optimizer': self.optimizer.state_dict(),
                'args': self.args
            }, is_best)

            epoch_time = time.time() - start_time
            msg = self.get_acc_msg(epoch, train_acc, train_loss, val_acc, val_loss,
                                   self.best_wa, self.best_ua, epoch_time)
            with open(self.log_path, 'a') as f:
                f.write(msg)
            print(msg)

            if is_best:
                cm_msg = self.get_confusion_msg(val_acc[2])
                with open(self.log_path, 'a') as f:
                    f.write(cm_msg)
                print(cm_msg)

                cm = []
                for row in val_acc[2]:
                    row = row / np.sum(row)
                    cm.append(row)

                fig_path = os.path.join(self.args.output_path, "fig_best.png")
                ax = seaborn.heatmap(
                    cm,
                    xticklabels=self.emotions,
                    yticklabels=self.emotions,
                    cmap='rocket_r'
                )
                figure = ax.get_figure()
                figure.savefig(fig_path)
                plt.close()

                self.tsne_visualization(features, targets, epoch)

        return self.best_ua, self.best_ua

    def train(self, epoch):
        self.model.train()
        all_pred, all_target = [], []
        all_loss = 0.0

        lambda_ce = 10.0
        lambda_sc = 0.1

        for i, (images, target) in enumerate(self.train_dataloader):
            print(f"Training epoch \t{epoch}: {i + 1}\\{len(self.train_dataloader)}", end='\r')

            images = images.to(self.device)
            target = target.to(self.device)

            logits, feature = self.model(images)

            loss_ce, loss_sc = self.DSM(logits, feature, target)
            total_loss = lambda_ce * loss_ce + lambda_sc * loss_sc
            total_loss.backward(retain_graph=True)

            self.optimizer.first_step()

            logits, feature = self.model(images)

            loss_ce, loss_sc = self.DSM(logits, feature, target)
            total_loss = lambda_ce * loss_ce + lambda_sc * loss_sc
            total_loss.backward()

            self.optimizer.second_step()

            pred = torch.argmax(logits, 1).cpu().detach().numpy()
            target_np = target.cpu().numpy()
            all_pred.extend(pred)
            all_target.extend(target_np)

            all_loss += total_loss.item()

            self.scheduler.step_update(epoch * len(self.train_dataloader) + i)

        acc1 = accuracy_score(all_target, all_pred)
        acc2 = balanced_accuracy_score(all_target, all_pred)
        loss = all_loss / len(self.train_dataloader)

        return [acc1, acc2], loss

    def validate(self, epoch):
        self.model.eval()

        all_pred, all_target = [], []
        all_loss = 0.0

        scatter_all_x = []
        scatter_all_y = []

        for i, (images, target) in enumerate(self.test_dataloader):
            print(f"Testing epoch \t{epoch}: {i + 1}\\{len(self.test_dataloader)}", end='\r')

            images = images.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                output, feature = self.model(images)

            loss = self.criterion(output, target)

            pred = torch.argmax(output, 1).cpu().detach().numpy()
            target_np = target.cpu().numpy()

            all_pred.extend(pred)
            all_target.extend(target_np)
            all_loss += loss.item()

            scatter_all_x.extend(feature.detach().cpu().numpy())
            scatter_all_y.extend(target_np)

        acc1 = accuracy_score(all_target, all_pred)             
        acc2 = balanced_accuracy_score(all_target, all_pred)    

        c_m = confusion_matrix(all_target, all_pred)
        loss = all_loss / len(self.test_dataloader)

        scatter_all_x = np.array(scatter_all_x)
        scatter_all_y = np.array(scatter_all_y)

        return [acc1, acc2, c_m], loss, scatter_all_x, scatter_all_y

    def save(self, state, is_best):
        if is_best:
            checkpoint_path = os.path.join(self.args.output_path, "model_best.pth")
            torch.save(state, checkpoint_path)

        checkpoint_path = os.path.join(self.args.output_path, "model_latest.pth")
        torch.save(state, checkpoint_path)

    def get_acc_msg(self, epoch, train_acc, train_loss, val_acc, val_loss,
                    best_wa, best_ua, epoch_time):
        msg = """\nEpoch {} Train\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                   Epoch {} Test\t: WA:{:.2%}, \tUA:{:.2%}, \tloss:{:.4f}
                   Epoch {} Best\t: WA:{:.2%}, \tUA:{:.2%}
                   Epoch {} Time\t: {:.1f}s\n\n""".format(
            epoch, train_acc[0], train_acc[1], train_loss,
            epoch, val_acc[0], val_acc[1], val_loss,
            epoch, best_wa, best_ua,
            epoch, epoch_time
        )
        return msg

    def get_confusion_msg(self, confusion_matrix_):
        msg = "Confusion Matrix:\n"
        for i in range(len(confusion_matrix_)):
            msg += self.emotions[i]
            for cell in confusion_matrix_[i]:
                msg += "\t" + str(cell)
            msg += "\n"
        for emotion in self.emotions:
            msg += "\t" + emotion
        msg += "\n\n"
        return msg
