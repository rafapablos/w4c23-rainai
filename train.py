# Weather4cast 2023 Starter Kit
#
# This Starter Kit builds on and extends the Weather4cast 2022 Starter Kit,
# the original license for which is included below.
#
# In line with the provisions of this license, all changes and additional
# code are also released unde the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#

# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
#
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import os
import torch
import datetime
import argparse
import boto3
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from w4c23.models.unet import UNetModule
from w4c23.models.swin import SWINModule
from w4c23.callbacks.log_metrics import LogMetrics
from w4c23.callbacks.log_images import ImageLogger
from w4c23.utils.data_utils import get_cuda_memory_usage, tensor_to_submission_file
from w4c23.utils.config import load_config
from w4c23.utils.w4c_dataloader import RainData

pl.seed_everything(42, workers=True)


class DataModule(pl.LightningDataModule):
    """Class to handle training/validation/predict/heldout splits."""

    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params
        self.training_params = training_params
        if mode in ["train"]:
            print("Loading TRAINING/VALIDATION dataset")
            self.train_ds = RainData("training", **self.params)
            self.val_ds = RainData("validation", **self.params)

            print(f"Training dataset size: {len(self.train_ds)}")
        if mode in ["val"]:
            print("Loading VALIDATION dataset")
            self.val_ds = RainData("validation", **self.params)
        if mode in ["predict"]:
            print("Loading PREDICTION/TEST dataset")
            self.test_ds = RainData("test", **self.params)
        if mode in ["heldout"]:
            print("Loading HELD-OUT dataset")
            self.test_ds = RainData("heldout", **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        dl = DataLoader(
            dataset,
            batch_size=self.training_params["batch_size"],
            num_workers=self.training_params["n_workers"],
            shuffle=shuffle,
            pin_memory=pin,
        )
        return dl

    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)

    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)


def load_model(Model, params, checkpoint_path=""):
    """Load a model from a checkpoint or from scratch if checkpoint_path=''"""
    p = {**params["experiment"], **params["dataset"], **params["train"]}
    if checkpoint_path == "":
        print("-> Modelling from scratch!  (no checkpoint loaded)")
        model = Model(params["model"], p)
    else:
        print(f"-> Loading model checkpoint: {checkpoint_path}")
        model = Model.load_from_checkpoint(
            checkpoint_path, model_params=params["model"], params=p
        )
    return model


def get_trainer(gpus, params):
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params["experiment"]["name"]
    version = version + "_" + date_time
    max_epochs = params["train"]["max_epochs"]
    print("Training for", max_epochs, "epochs")

    # Set logger for wandb or tensorboard
    if params["experiment"]["logging"] == "wandb":
        # Get wandb key (this is only required for wandb logging in aws)
        if params["experiment"]["aws"]:
            client = boto3.client("ssm", region_name="eu-central-1")
            try:
                os.environ["WANDB_API_KEY"] = client.get_parameter(
                    Name="salami-training-w4c23-wandb-api-key", WithDecryption=True
                )["Parameter"]["Value"]
                print("WandB should be running in online mode")
            except Exception as e:  # pylint: disable=bare-except
                print(e)
                print("WandB could not get an API key and is running in offline mode")
                os.environ["WANDB_MODE"] = "offline"

        logger = pl_loggers.WandbLogger(
            project="w4c23",
            name=params["experiment"]["sub_folder"] + "_" + version,
            log_model="all",
            save_dir=params["experiment"]["experiment_folder"],
        )
    elif params["experiment"]["logging"] == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=params["experiment"]["experiment_folder"],
            name=params["experiment"]["sub_folder"],
            version=version,
            log_graph=True,
            default_hp_metric=False,
        )
    else:
        logger = False

    # Callbacks
    # Model saving and early stopping
    callback_funcs = [
        ModelCheckpoint(
            monitor="val/loss",
            save_top_k=2,
            save_last=True,
            filename="epoch={epoch}-step={step}-val_loss={val/loss:.6f}",
            auto_insert_metric_name=False,
        ),
    ]
    if params["train"]["early_stopping"]:
        callback_funcs.append(
            EarlyStopping(
                monitor="val/loss", patience=params["train"]["patience"], mode="min"
            )
        )
    # Add metrics and image logging
    if params["experiment"]["logging"] != "none":
        callback_funcs.append(
            LogMetrics(
                num_leadtimes=params["model"]["forecast_length"],
                probabilistic=params["train"]["probabilistic"],
                buckets=params["model"]["buckets"],
                logging=params["experiment"]["logging"],
            )
        )
        callback_funcs.append(ImageLogger(params["experiment"]["logging"]))

    # Training accelerators
    accelerator = None
    if gpus[0] == -1:
        gpus = None
    else:
        accelerator = "cuda"
    print(f"====== process started on the following GPUs: {gpus} ======")

    trainer = pl.Trainer(
        devices=gpus,
        max_epochs=max_epochs,
        deterministic=params["model"]["deterministic"],
        logger=logger,
        callbacks=callback_funcs,
        precision=params["experiment"]["precision"],
        gradient_clip_val=params["model"]["gradient_clip_val"],
        gradient_clip_algorithm=params["model"]["gradient_clip_algorithm"],
        accelerator=accelerator,
        strategy="ddp_find_unused_parameters_false",
        profiler="simple",
        log_every_n_steps=5,
        accumulate_grad_batches=params["train"]["accumulate_grad_batches"]
        if "accumulate_grad_batches" in params["train"].keys()
        else 1,
    )

    return trainer


def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)
    tensor_to_submission_file(scores, predict_params)


def do_val(trainer, model, val_data):
    scores = trainer.validate(model, dataloaders=val_data)
    print(scores[0])


def train(params, gpus, mode, checkpoint_path, model):
    """Main training/evaluation method."""

    # Remove extra regions/years in predict mode and disable logging
    if mode == "predict":
        params["dataset"]["regions"] = [params["predict"]["region_to_predict"]]
        params["dataset"]["years"] = [params["predict"]["year_to_predict"]]
        params["experiment"]["logging"] = "none"

    # ------------
    # Model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params["dataset"], params["train"], mode)
    model = load_model(model, params, checkpoint_path)

    # ------------
    # Trainer
    # ------------
    trainer = get_trainer(gpus, params)

    # ------------
    # Train & final validation
    # ------------
    if mode == "train":
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)

    # ------------
    # Validation
    # ------------
    if mode == "val":
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_val(trainer, model, data.val_dataloader())

    # ------------
    # Prediction
    # ------------
    if mode == "predict" or mode == "heldout":
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        print(
            "REGIONS: ",
            params["dataset"]["regions"],
            params["predict"]["region_to_predict"],
        )
        if params["predict"]["region_to_predict"] not in params["dataset"]["regions"]:
            print(
                'EXITING... "regions" and "regions to predict" must indicate the same region name in your config file.'
            )
        else:
            do_predict(trainer, model, params["predict"], data.test_dataloader())


def update_params_based_on_args(options):
    config_p = os.path.join("configurations", options.config_path)
    params = load_config(config_p)

    if options.name != "":
        print(params["experiment"]["name"])
        params["experiment"]["name"] = options.name
    if options.epochs is not None:
        params["train"]["max_epochs"] = options.epochs
    if options.batch_size is not None:
        params["train"]["batch_size"] = options.batch_size
    if options.num_workers is not None:
        params["train"]["n_workers"] = options.num_workers
    if options.input_path != "":
        params["dataset"]["data_root"] = options.input_path
    if options.output_path != "":
        params["experiment"]["experiment_folder"] = options.output_path
    if options.region_to_predict != "":
        params["predict"]["region_to_predict"] = options.region_to_predict
    if options.year_to_predict != "":
        params["predict"]["year_to_predict"] = options.year_to_predict
    if options.submission_out_dir != "":
        params["predict"]["submission_out_dir"] = options.submission_out_dir
    return params


def set_parser():
    """Set custom parser."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-f",
        "--config_path",
        type=str,
        required=True,
        help="path to config-yaml",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        nargs="+",
        required=False,
        default=1,
        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        default="train",
        help="choose mode: train (default) / val / predict",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=False,
        default="",
        help="init a model from a checkpoint path. '' as default (random weights)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=False,
        default="",
        help="Set the name of the experiment",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        required=False,
        default=None,
        help="Set the epochs of the experiment",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        required=False,
        default=None,
        help="Set the batch size of the experiment",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        required=False,
        default=None,
        help="Set the number of workers of the experiment",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=False,
        default="",
        help="Set the input path",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        default="",
        help="Set the output path",
    )
    parser.add_argument(
        "-p",
        "--region_to_predict",
        type=str,
        required=False,
        default="",
        help="Set the region to predict",
    )
    parser.add_argument(
        "-y",
        "--year_to_predict",
        type=str,
        required=False,
        default="",
        help="Set the year to predict",
    )
    parser.add_argument(
        "-s",
        "--submission_out_dir",
        type=str,
        required=False,
        default="",
        help="Set the submission output directory",
    )
    return parser


def main():
    parser = set_parser()
    options = parser.parse_args()
    params = update_params_based_on_args(options)
    selected_model = params["model"]["model_name"]
    if selected_model == "2D_UNET_base":
        model = UNetModule
    elif selected_model == "SWIN":
        model = SWINModule
    train(params, options.gpus, options.mode, options.checkpoint, model)


if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU
    python train.py --gpus 2 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    """
