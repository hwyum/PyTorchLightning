from argparse import ArgumentParser
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import pytorch_lightning as pl
from model.ops import Flatten, Permute
from model.utils import Tokenizer, PadSequence
from model.data import Corpus
from model.split import split_to_jamo


class LitCharCNN(pl.LightningModule):
    """CharCNN Lightning Module"""
    def __init__(self, conf) -> None:
        """Instantiating CharCNN class

        Args:
            num_classes (int): the number of classes
            embedding_dim (int): the dimension of embedding vector for token
            vocab (model.utils.Vocab): the instance of model.utils.Vocab
        """
        super().__init__()
        self.hparams = conf
        self.tokenizer = None
        with open(self.hparams.vocab, mode="rb") as io:
            self.vocab = pickle.load(io)

        self._extractor = nn.Sequential(nn.Embedding(len(self.vocab), self.hparams.embedding_dim, self.vocab.to_indices(self.vocab.padding_token)),
                                        Permute(),
                                        nn.Conv1d(in_channels=self.hparams.embedding_dim, out_channels=256, kernel_size=7),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 3),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=7),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 3),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3),
                                        nn.ReLU(),
                                        nn.MaxPool1d(3, 3),
                                        Flatten())

        self._classifier = nn.Sequential(nn.Linear(in_features=1792, out_features=512),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=512, out_features=512),
                                         nn.ReLU(),
                                         nn.Dropout(),
                                         nn.Linear(in_features=512, out_features=self.hparams.num_classes))
        self.apply(self._initailze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self._extractor(x)
        score = self._classifier(feature)
        return score

    def _initailze(self, layer):
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            nn.init.kaiming_uniform_(layer.weight)

    def prepare_data(self):
        pad_sequence = PadSequence(length=self.hparams.length, pad_val=self.vocab.to_indices(self.vocab.padding_token))
        tokenizer = Tokenizer(vocab=self.vocab, split_fn=split_to_jamo, pad_fn=pad_sequence)
        self.tokenizer = tokenizer

    def setup(self, step):
        return

    # todo
    def configure_optimizers(self):
        return optim.Adam(params=self.parameters(), lr=self.hparams.learning_rate)

    # todo
    def train_dataloader(self):
        tr_ds = Corpus(self.hparams.train, self.tokenizer.split_and_transform)
        return DataLoader(tr_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)

        # add loggins
        logs = {'loss':loss}
        return {'loss':loss, 'log':logs}

    def validation_dataloader(self):
        val_ds = Corpus(self.hparams.validation, self.tokenizer.split_and_transform)
        return DataLoader(val_ds, batch_size=self.hparams.batch_size, num_workers=4)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        return {'val_loss':loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--model_config",
            default="conf/model/charcnn.json",
            help="directory containing charcnn.json",
        )
        parser.add_argument("--epochs", default=5, help="number of epochs of training")
        parser.add_argument("--batch_size", default=256, help="batch size of training")
        parser.add_argument(
            "--learning_rate", default=1e-3, help="learning rate of training"
        )
        parser.add_argument(
            "--summary_step", default=500, help="logging performance at each step"
        )
        parser.add_argument("--fix_seed", action="store_true", default=False)
        return parser
