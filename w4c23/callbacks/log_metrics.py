"""
Metrics callback.
"""

import os
import torch
import torchmetrics
import pytorch_lightning as pl

import w4c23.callbacks.metrics as metrics
from w4c23.utils.buckets import BUCKET_CONSTANTS

# from torchmetrics.utilities import check_forward_full_state_property


class LogMetrics(pl.Callback):
    def __init__(self, num_leadtimes, probabilistic, buckets, logging):
        super().__init__()
        self.num_leadtimes = num_leadtimes
        self.probabilistic = probabilistic

        if buckets != "none":
            self.buckets = BUCKET_CONSTANTS[buckets]
        else:
            self.buckets = None

        self.logging = logging
        self.thresholds = [0.2, 1, 5, 10, 15]

        # # Code for checking if a metric can be optimized
        # check_forward_full_state_property(
        #     metrics.MeanSquaredError,
        #     input_args={
        #         "prediction": torch.Tensor([0.5, 2.5]),
        #         "label": torch.Tensor([1.0, 2.0]),
        #         "mask": torch.zeros([2], dtype=bool),
        #     },
        # )

    def _threshold_str(self, threshold):
        """Remove .0 and change . by -"""
        return f"{threshold:g}".replace(".", "-")

    def setup(self, trainer, pl_module, stage):
        # Setup scalar metrics
        scalar_metrics = {}
        scalar_metrics["mse"] = metrics.MeanSquaredError()
        scalar_metrics["mae"] = metrics.MeanAverageError()

        for threshold in self.thresholds:
            csi = metrics.CriticalSuccessIndex(threshold=threshold)
            scalar_metrics[f"csi_{self._threshold_str(threshold)}"] = csi
        scalar_metrics["avg_csi"] = metrics.AverageCriticalSuccessIndex(
            thresholds=self.thresholds
        )

        if self.probabilistic:
            scalar_metrics["crps"] = metrics.ContinuousRankedProbabilityScore(
                self.buckets
            )

        # Create metric collections and put metrics on module to automatically place on correct device
        val_scalar_metrics = torchmetrics.MetricCollection(scalar_metrics)
        pl_module.val_metrics = val_scalar_metrics.clone(prefix="val/")

        # Lead time metrics
        lead_time_metrics = {}
        lead_time_metrics[f"mse"] = metrics.MeanSquaredError(
            num_leadtimes=self.num_leadtimes
        )
        for threshold in self.thresholds:
            csi = metrics.CriticalSuccessIndex(
                threshold=threshold, num_leadtimes=self.num_leadtimes
            )
            lead_time_metrics[f"csi_{self._threshold_str(threshold)}"] = csi
        lead_time_metrics["avg_csi"] = metrics.AverageCriticalSuccessIndex(
            thresholds=self.thresholds, num_leadtimes=self.num_leadtimes
        )
        pl_module.lead_time_metrics = torchmetrics.MetricCollection(lead_time_metrics)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called after each validation batch with scalar and lead time metrics"""
        _, label, metadata = batch
        pl_module.val_metrics(outputs, label, metadata["target"]["mask"])
        pl_module.lead_time_metrics(outputs, label, metadata["target"]["mask"])

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log validation scalar metrics
        pl_module.log_dict(
            pl_module.val_metrics, on_step=False, on_epoch=True, sync_dist=True
        )

        # Compute and log lead time metrics
        lead_time_metrics = pl_module.lead_time_metrics.compute()
        lead_time_metrics_dict = {}
        wandb_data = []
        for metric_name, arr in lead_time_metrics.items():
            # Add to logging dictionary
            for leadtime, value in enumerate(arr):
                lead_time_metrics_dict[f"val_time/{metric_name}_{leadtime+1}"] = value
            # Save to file (tensorboard)
            if self.logging == "tensorboard":
                file_path = os.path.join(
                    pl_module.logger.log_dir, f"val_lead_time_{metric_name}.pt"
                )
                torch.save(arr.cpu(), file_path)
            # Generate table for wandb
            elif self.logging == "wandb":
                columns = ["metric"] + [f"t_{i+1}" for i in range(len(arr))]
                wandb_data.append([metric_name] + arr.tolist())

        # Save table in wandb
        if self.logging == "wandb":
            pl_module.logger.log_table(
                key="leadtimes", columns=columns, data=wandb_data
            )

        # Save lead time metrics over time
        pl_module.log_dict(
            lead_time_metrics_dict, on_step=False, on_epoch=True, sync_dist=True
        )
        pl_module.lead_time_metrics.reset()
