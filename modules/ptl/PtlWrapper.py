import gc
import torch
import torch.nn as nn
import pytorch_lightning as ptl
import torchmetrics

from modules.metrics.SparseTopKCategoricalAccuracy import SparseTop5CategoricalAccuracy, SparseTop1CategoricalAccuracy
from modules.metrics.MaP import MaP5


class PtlWrapper(ptl.LightningModule):
    def __init__(self, model, lr=1e-3, loss_f=nn.CrossEntropyLoss(reduction='none'), sch_settings=None):
        super().__init__()
        if sch_settings is not None:
            self.scheduler = sch_settings['scheduler']
            self.sch_kwargs = {k: v for k, v in sch_settings.items() if k != 'scheduler'}
        else:
            self.scheduler = False

        self.lr = lr
        self.model = model
        self.softmax = nn.Softmax()

        #  metrics
        self.loss = loss_f
        self.map5 = MaP5
        self.top1 = torchmetrics.Accuracy(top_k=1)
        self.top5 = torchmetrics.Accuracy(top_k=5)

    def forward(self, inp):
        return self.model(inp)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.scheduler:
            self.sch = self.scheduler(self.optimizer, **self.sch_kwargs)
            return [self.optimizer], [self.sch]
        else:
            return [self.optimizer]


    def training_step(self, train_batch, batch_idx):
        input, label = train_batch['input'], train_batch['label']

        #print(input.shape)
        #print(label.shape)

        _, pred_cos_dist = self.model(input, label)
        pred_cos_dist = pred_cos_dist.type(torch.float32)
        pred_prob = self.softmax(pred_cos_dist)

        loss = self.loss(pred_cos_dist, label)
        loss = torch.mean(loss)

        map5 = self.map5(pred_prob, label)

        label = torch.argmax(label, dim=1)
        top1 = self.top1(pred_prob, label)
        top5 = self.top5(pred_prob, label)

        return {'loss': loss, 'map5': map5.detach(), 'top5': top5.detach(), 'top1': top1.detach()}


    def training_epoch_end(self, outputs):
        #print("TR END")

        loss, map5, top5, top1 = [], [], [], []
        for out in outputs:
            loss.append(out['loss'])
            map5.append(out['map5'])
            top5.append(out['top5'])
            top1.append(out['top1'])

        loss = torch.stack(tensors=loss)
        loss = torch.mean(loss).detach()

        map5 = torch.stack(tensors=map5)
        map5 = torch.mean(map5).detach()

        top5 = torch.stack(tensors=top5)
        top5 = torch.mean(top5).detach()

        top1 = torch.stack(tensors=top1)
        top1 = torch.mean(top1).detach()

        self.log("train_cross_entropy_loss", loss.detach(), on_epoch=True, prog_bar=True)
        self.log("train_map5", map5.detach(), on_epoch=True, prog_bar=True)
        self.log("train_top5", top5.detach(), on_epoch=True, prog_bar=True)
        self.log("train_top1", top1.detach(), on_epoch=True, prog_bar=True)

        gc.collect()


    def validation_step(self, val_batch, batch_idx):
        #print("VALIDATION")

        input, label = val_batch['input'], val_batch['label']

        _, pred_cos_dist = self.model(input, label)
        pred_cos_dist = pred_cos_dist.type(torch.float32)
        pred_prob = self.softmax(pred_cos_dist)



        loss = self.loss(pred_cos_dist, label)
        loss = torch.mean(loss)

        map5 = self.map5(pred_prob, label)

        label = torch.argmax(label, dim=1)
        top1 = self.top1(pred_prob, label)
        top5 = self.top5(pred_prob, label)

        return {'loss': loss.detach(), 'map5': map5.detach(), 'top5': top5.detach(), 'top1': top1.detach()}


    def validation_epoch_end(self, outputs):
        #print("VAL END")

        loss, map5, top5, top1 = [], [], [], []
        for out in outputs:
            map5.append(out['map5'])
            top5.append(out['top5'])
            top1.append(out['top1'])
            loss.append(out['loss'])

        map5 = torch.stack(tensors=map5)
        map5 = torch.mean(map5).detach()

        top5 = torch.stack(tensors=top5)
        top5 = torch.mean(top5).detach()

        top1 = torch.stack(tensors=top1)
        top1 = torch.mean(top1).detach()

        loss = torch.stack(tensors=loss)
        loss = torch.mean(loss).detach()

        self.log("val_map5", map5.detach(), on_epoch=True, prog_bar=True)
        self.log("val_top5", top5.detach(), on_epoch=True, prog_bar=True)
        self.log("val_top1", top1.detach(), on_epoch=True, prog_bar=True)
        self.log("val_loss", loss.detach(), on_epoch=True, prog_bar=True)

        gc.collect()
