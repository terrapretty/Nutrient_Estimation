# train_with_bg_removal.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_squared_error, r2_score
from src.dataset import SierraDataset, sierra_collate_fn
from src.hs_utils import hs_train_transforms, hs_val_transforms
from src.model import HSResNet18

torch.set_float32_matmul_precision("medium")

# Define hyperparameters
num_epochs = 100
learning_rate = 1e-3
batch_size = 8
input_channels = 462  # Set this to the correct number of input channels for your dataset
num_outputs = 6  # The number of regression targets
img_size = (160, 160)

root_path = "/path/to/output"  # Path where the new preprocessed data is stored
train_path = os.path.join(root_path, "train")
val_path = os.path.join(root_path, "val")
gt_path = "/path/to/ground_truth.csv"

# Initialize wandb logger
wandb_logger = WandbLogger(project="nutrient-estimation")  # Replace with your project name

class HSResNet18Lightning(pl.LightningModule):
    def __init__(self, input_channels, num_outputs, learning_rate):
        super(HSResNet18Lightning, self).__init__()
        self.model = HSResNet18(input_channels=input_channels, num_outputs=num_outputs)
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        val_rmse = mean_squared_error(targets.cpu().numpy(), outputs.cpu().numpy(), squared=False)
        val_r2 = r2_score(targets.cpu().numpy(), outputs.cpu().numpy())
        
        self.log("val_loss", loss)
        self.log("val_rmse", val_rmse)
        self.log("val_r2", val_r2)

        return {"val_loss": loss, "targets": targets, "preds": outputs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    # Define the dataset and data loaders
    train_transforms = hs_train_transforms(crop_size=img_size)
    train_dataset = SierraDataset(csv_file=gt_path, root_dir=train_path, transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sierra_collate_fn, drop_last=True, num_workers=15)

    val_transforms = hs_val_transforms(crop_size=img_size)
    val_dataset = SierraDataset(csv_file=gt_path, root_dir=val_path, transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sierra_collate_fn, drop_last=False, num_workers=15)

    # Initialize the model
    model = HSResNet18Lightning(input_channels=input_channels, num_outputs=num_outputs, learning_rate=learning_rate)

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=num_epochs, logger=wandb_logger, log_every_n_steps=10)

    # Train the model
    trainer.fit(model, train_loader, val_loader)
