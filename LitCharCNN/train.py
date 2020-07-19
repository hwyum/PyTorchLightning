from argparse import ArgumentParser
from model.lightning_module import LitCharCNN
from pytorch_lightning import Trainer
from LitCharCNN.utils import Config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        default="conf/dataset/nsmc.json",
        help="directory containing nsmc.json",
    )

    # add model specific args
    # add PROGRAM level args
    # parser.add_argument('--conda_env', type=str, default='some_name')
    # parser.add_argument('--notification_email', type=str, default='will@email.com')

    # add model specific args
    parser = LitCharCNN.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    # ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # print(args.__dict__)

    dataset_config = Config(args.dataset_config)
    model_config = Config(args.model_config)

    hparams = args.__dict__
    hparams.update(dataset_config.dict)
    hparams.update(model_config.dict)

    model = LitCharCNN(hparams)
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

