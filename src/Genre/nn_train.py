import pytorch_lightning as pl
import torch
import torchmetrics

from dataset import MetroLyrics, MetroLyricsDataModule


class LitModel(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size=100, lr=0.02):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lr = lr

        self.save_hyperparameters()

        self.loss = torch.nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()

        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.Tensor(x.toarray()).to("cuda")
        res = self.relu(self.l1(x))
        res = torch.softmax(self.l2(res), dim=1)
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
        self.log_dict({"test_loss": loss, "test_acc": acc})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
