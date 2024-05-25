
from torch.utils.data import DataLoader
from transformers import DetrImageProcessor
from pytorch_lightning import Trainer
import os
from dataset import Manga109Dataset
from model import Detr

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }


TRAIN_DATASET = Manga109Dataset(
    image_directory_path="Dataset/train/images",
    image_processor=image_processor,
    train=True)
VAL_DATASET = Manga109Dataset(
    image_directory_path="Dataset/valid/images",
    image_processor=image_processor,
    train=False)

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
MAX_EPOCHS = 50
CHECKPOINT= "facebook/detr-resnet-50"


model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, num_labels=4, train_loder=TRAIN_DATALOADER, valid_loader=VAL_DATALOADER, CHECKPOINT=CHECKPOINT)
trainer = Trainer(
    devices=1,                   # Number of devices to use (1 for CPU)
    accelerator="gpu",           # Use CPU as the accelerator
    max_epochs=MAX_EPOCHS,       # Maximum number of epochs
    gradient_clip_val=0.1,       # Gradient clipping value
    accumulate_grad_batches=8,   # Number of batches to accumulate for gradient updates
    log_every_n_steps=5          # Logging frequency
)
trainer.fit(model)
