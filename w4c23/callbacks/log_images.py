"""
For the first validation batch, it logs images with the predicitions and true labels
"""

import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from pysteps.visualization.precipfields import plot_precip_field


class ImageLogger(pl.Callback):
    def __init__(self, logging):
        super().__init__()
        self.logging = logging
        self.step = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if batch_idx != 0:
            return

        summary_writer = pl_module.logger.experiment

        _, label, metadata = batch
        batch_mask = metadata["target"]["mask"].cpu()
        batch_labels = label.cpu()
        batch_preds = outputs.intensity.cpu()
        batch_size = batch_preds.size(0)

        for i in range(batch_size):
            figure = self.plotSampleComparison(batch_labels, batch_preds, batch_mask, i)
            if self.logging == "tensorboard":
                summary_writer.add_figure(f"val_examples/{i}", figure, self.step)
            else:
                summary_writer.log({f"val_examples/{i}": figure})

        self.step += 1

    def plotSampleComparison(self, target, prediction, mask, sample_index):
        # Mask out values
        prediction[mask] = np.nan
        target[mask] = np.nan

        forecast_length = target.shape[2]
        fig, axes = plt.subplots(nrows=2, ncols=forecast_length, figsize=(20, 3))

        # Add color bar
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.17, 0.02, 0.65])

        # Plot labels
        for t in range(forecast_length):
            target_sample = target[sample_index, 0, t]
            plot_precip_field(
                target_sample, colorbar=t == 0, ax=axes[0][t], cax=cbar_ax
            )

        # Plot predictions
        for t in range(forecast_length):
            pred_sample = prediction[sample_index, 0, t]
            plot_precip_field(pred_sample, colorbar=False, ax=axes[1][t])
        return fig
