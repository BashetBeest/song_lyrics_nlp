import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from dataset import MetroLyrics, MetroLyricsDataModule


class LitModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=[100, 50], lr=0.02):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lr = lr

        self.save_hyperparameters()

        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()

        self.l1 = torch.nn.Linear(input_size, hidden_size[0])
        self.relu = torch.nn.LeakyReLU() # torch.nn.Sigmoid() # torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = torch.nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x = torch.Tensor(x.toarray()).to("cuda")
        res = self.relu(self.l1(x))
        res = self.relu(self.l2(res))
        res = torch.softmax(self.l3(res), dim=1)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = torch.LongTensor(y).to("cuda")
        y_hat = self(x) # basically the same as self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.acc(y_hat, y)
        self.log_dict({"train_loss": loss, "train_acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.LongTensor(y).to("cuda")
        y_hat = self(x) # basically the same as self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.acc(y_hat, y)
        self.log_dict({"val_loss": loss, "val_acc": acc})
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.LongTensor(y).to("cuda")
        y_hat = self(x) # basically the same as self.forward(x)
        loss = self.loss(y_hat, y)
        acc = self.acc(y_hat, y)
        self.log_dict({"test_loss": loss, "test_acc": acc}, on_step=True)
        return {"loss": loss, "true": y, "pred": y_hat}

    def test_epoch_end(self, outputs):
        true = torch.stack([y for o in outputs for y in o["true"]]).cpu()
        pred = torch.stack([y_hat for o in outputs for y_hat in o["pred"]])
        pred = torch.argmax(pred, dim=1).cpu()
        conf_mat = confusion_matrix(true, pred, labels=range(self.output_size))
        labels = [self.num_to_genre[i] for i in range(self.output_size)]
        print(conf_mat, conf_mat.shape)
        print(labels)
        figure = plt.figure()
        axes = figure.add_subplot(111)
        caxes = axes.matshow(conf_mat)
        figure.colorbar(caxes)
        plt.xticks(range(self.output_size), labels)
        plt.yticks(range(self.output_size), labels)
        plt.tick_params(labelsize=7)
        plt.savefig("conf_mat.png")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
